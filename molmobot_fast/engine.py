import concurrent.futures
import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from molmobot_fast.patches import (
    ActionExpertCachedContext,
    _finalize_fp8,
    hash_inputs,
    patch_action_expert,
    patch_compile_backbone,
    patch_flash_attention,
    patch_fp8_quantize,
    patch_molmobot,
    patch_tensorrt_backbone,
)

log = logging.getLogger(__name__)


class FastMolmoBot:
    """Optimized MolmoBot inference engine.

    Loads the upstream model, applies all optimizations via runtime patches,
    and exposes a simple ``predict()`` API. No MolmoBot source files are modified.

    Usage::

        from molmobot_fast import FastMolmoBot

        bot = FastMolmoBot()  # auto-downloads allenai/MolmoBot-DROID
        actions = bot.predict(
            images=[cam1, cam2],
            task="pick up the red block",
            state=np.array([...]),
        )
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        hf_repo: str = "allenai/MolmoBot-DROID",
        device: str = "cuda",
        num_flow_steps: int = 10,
        cuda_graph: bool = True,
        flash_attn: bool = True,
        compile_backbone: bool = True,
        async_preprocess: bool = True,
        cache_backbone: bool = False,
        fp8: bool = False,
        tensorrt: bool = False,
    ):
        if checkpoint is None:
            from huggingface_hub import snapshot_download
            log.info(f"Downloading {hf_repo} ...")
            checkpoint = snapshot_download(hf_repo)

        t0 = time.perf_counter()

        self._load_model(checkpoint, device, num_flow_steps)
        self._apply_patches(
            cuda_graph, flash_attn, compile_backbone, async_preprocess, fp8, tensorrt)
        self._init_async(async_preprocess, device)

        self._cache_backbone = cache_backbone
        self._backbone_cache_key: Optional[bytes] = None
        self._cached_layer_states = None
        self._cached_encoder_attn_mask = None
        self._backbone_cache_hits = 0
        self._backbone_cache_misses = 0
        self._gpu_buffer_pool: Optional[Dict[str, torch.Tensor]] = None

        self._warmup(device)
        dt = time.perf_counter() - t0
        log.info(f"FastMolmoBot ready in {dt:.1f}s  "
                 f"(steps={num_flow_steps}, graph={cuda_graph}, fa2={flash_attn}, "
                 f"compile={compile_backbone}, fp8={fp8}, trt={tensorrt}, "
                 f"async={async_preprocess})")

    # ── model loading (upstream code, no patches) ────────────────────────

    def _load_model(self, checkpoint: str, device: str, num_flow_steps: int):
        from olmo.train.checkpointer import load_model_state
        from olmo.models.model_config import BaseModelConfig
        from olmo.util import resource_path

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._num_flow_steps = num_flow_steps

        config_path = resource_path(checkpoint, "config.yaml")
        self._model_config = BaseModelConfig.load(config_path, key="model")

        self.action_horizon = getattr(self._model_config, "action_horizon", 16)
        self.action_dim = getattr(self._model_config, "action_dim", 7)

        with torch.device("meta"):
            self._model = self._model_config.build_model()
        self._model.to(torch.bfloat16)
        self._model.to_empty(device=self.device)
        load_model_state(checkpoint, self._model)
        self._model.to(self.device, dtype=torch.bfloat16)
        self._model.eval()

        self._preprocessor = self._model_config.build_preprocessor(
            for_inference=True, is_training=False,
            max_seq_len=self._model_config.llm.max_sequence_length)
        self._collator = self._model_config.build_collator(
            self._preprocessor.get_output_shapes(), pad_mode=None, include_metadata=True)

        self._state_pre = None
        rp = getattr(self._model_config, "robot_preprocessor", None)
        if rp:
            self._state_pre = rp.build_preprocessor()
        self._action_post = None
        rpo = getattr(self._model_config, "robot_postprocessor", None)
        if rpo:
            self._action_post = rpo.build_postprocessor()

    # ── apply optimizations ──────────────────────────────────────────────

    def _apply_patches(self, cuda_graph, flash_attn, compile_bb, async_preprocess, fp8, tensorrt):
        patch_action_expert(self._model)
        patch_molmobot(self._model)
        self._model._enable_compiled_ae_step = bool(
            compile_bb and async_preprocess and cuda_graph
        )
        # In this stack, FA2 tends to regress latency once compile_backbone is on.
        use_flash = flash_attn and not compile_bb
        if use_flash:
            patch_flash_attention(self._model)
        if fp8:
            patch_fp8_quantize(self._model)
        self._needs_fp8_finalize = fp8
        self._deferred_compile_bb = compile_bb and not tensorrt
        self._deferred_tensorrt = tensorrt
        self._deferred_cuda_graph = cuda_graph

    def _init_async(self, async_preprocess, device):
        self._async = async_preprocess
        self._executor = None
        self._pending = None
        self._h2d_stream = None
        if async_preprocess and device == "cuda":
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._h2d_stream = torch.cuda.Stream(device=self.device)

    # ── warmup ───────────────────────────────────────────────────────────

    def _warmup(self, device, n=3):
        def _run_n(count):
            for _ in range(count):
                imgs = [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(2)]
                self.predict(images=imgs, task="warmup", state=np.random.randn(8).astype(np.float32))
                if device == "cuda":
                    torch.cuda.synchronize()

        if self._needs_fp8_finalize:
            log.info("FP8 calibration: collecting activation stats...")
            _run_n(n)
            _finalize_fp8(self._model)
            self._needs_fp8_finalize = False

        if self._deferred_tensorrt:
            patch_tensorrt_backbone(self._model)
        elif self._deferred_compile_bb:
            patch_compile_backbone(self._model)
        if self._deferred_cuda_graph:
            self._model.enable_cuda_graph()

        _run_n(n)

    # ── preprocessing ────────────────────────────────────────────────────

    def _prepare_images(self, images):
        if isinstance(images, np.ndarray):
            images = [images] if images.ndim == 3 else [images[i] for i in range(images.shape[0])]
        result = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                result.append(img)
            else:
                result.append(img)
        return result

    def _prepare_state(self, state):
        if state is None:
            return None
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if self._state_pre:
            try:
                state = self._state_pre.normalize_state(state, "synthmanip")
            except Exception:
                pass
        return state

    def _cpu_preprocess(self, images, task, state):
        images = self._prepare_images(images)
        state = self._prepare_state(state)
        example = {"style": "demo", "question": task,
                   "image": images if len(images) > 1 else images[0]}
        if state is not None:
            example["state"] = state
        processed = self._preprocessor(example)
        batch = self._collator([processed])
        batch = self._pin(batch)
        if self._cache_backbone:
            input_ids = batch.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                # Keep cache-key hashing on CPU tensors to avoid GPU->CPU sync.
                batch["_backbone_cache_key"] = hash_inputs(input_ids, None)
        return batch

    @staticmethod
    def _pin(batch):
        if isinstance(batch, torch.Tensor):
            return batch.pin_memory() if not batch.is_pinned() and batch.device.type == "cpu" else batch
        elif isinstance(batch, dict):
            return {k: FastMolmoBot._pin(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [FastMolmoBot._pin(x) for x in batch]
        return batch

    def _to_gpu(self, batch):
        if self._gpu_buffer_pool:
            out = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    buf = self._gpu_buffer_pool.get(k)
                    if buf is not None and buf.shape == v.shape and buf.dtype == v.dtype:
                        buf.copy_(v, non_blocking=True)
                        out[k] = buf
                    else:
                        t = v.to(self.device, non_blocking=True)
                        self._gpu_buffer_pool[k] = t
                        out[k] = t
                else:
                    out[k] = v
            return out
        else:
            self._gpu_buffer_pool = {}
            out = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    t = v.to(self.device, non_blocking=True)
                    self._gpu_buffer_pool[k] = t
                    out[k] = t
                else:
                    out[k] = v
            return out

    # ── inference ────────────────────────────────────────────────────────

    def predict(
        self,
        images: Union[List[np.ndarray], np.ndarray],
        task: str = "complete the task",
        state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run inference. Returns (action_horizon, action_dim) numpy array."""

        if self._async and self._pending is not None:
            batch = self._pending.result()
            with torch.cuda.stream(self._h2d_stream):
                batch = self._to_gpu(batch)
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
        else:
            batch = self._cpu_preprocess(images, task, state)
            batch = self._to_gpu(batch)

        mi = {k: batch[k] for k in (
            "input_ids", "attention_mask", "position_ids", "response_mask",
            "images", "image_masks", "token_pooling", "low_res_token_pooling", "states",
        ) if k in batch and batch.get(k) is not None}
        cache_key = batch.get("_backbone_cache_key") if self._cache_backbone else None

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self._cache_backbone:
                actions = self._run_cached(mi, cache_key=cache_key)
            else:
                actions = self._model.generate_actions(
                    **mi, num_steps=self._num_flow_steps)

        if self._async and self._executor:
            self._pending = self._executor.submit(self._cpu_preprocess, images, task, state)

        out = actions.detach().cpu().numpy()
        if self._action_post:
            try:
                out = self._action_post.unnormalize_action(out, "synthmanip")
            except Exception:
                pass
        return out[0]

    def _run_cached(self, mi, cache_key=None):
        key = cache_key if cache_key is not None else hash_inputs(mi["input_ids"], None)
        if key == self._backbone_cache_key and self._cached_layer_states is not None:
            self._backbone_cache_hits += 1
            ls, em = self._cached_layer_states, self._cached_encoder_attn_mask
        else:
            self._backbone_cache_misses += 1
            bk = {k: v for k, v in mi.items() if k != "states"}
            ls, em = self._model.run_backbone_only(**bk)
            self._cached_layer_states, self._cached_encoder_attn_mask = ls, em
            self._backbone_cache_key = key
        return self._model.generate_actions_from_cache(
            ls, em, states=mi.get("states"), num_steps=self._num_flow_steps)

    @property
    def num_flow_steps(self):
        return self._num_flow_steps
