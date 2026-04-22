import logging
import subprocess
import threading
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("molmobot_fast").setLevel(logging.WARNING)

class _GPUMon:
    """Sample GPU util/memory in a lightweight background thread."""

    def __init__(self, gpu: int = 0, hz: int = 20):
        self._gpu = gpu
        self._hz = hz
        self._samples = []
        self._stop = threading.Event()
        self._t = None

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(2)

    def _run(self):
        while not self._stop.is_set():
            try:
                o = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(self._gpu),
                    ],
                    text=True,
                    timeout=1,
                ).strip().split(",")
                if len(o) == 2:
                    self._samples.append((float(o[0]), float(o[1])))
            except Exception:
                pass
            time.sleep(1 / self._hz)

    @property
    def util(self):
        return np.mean([sample[0] for sample in self._samples]) if self._samples else 0

    @property
    def mem(self):
        return np.mean([sample[1] for sample in self._samples]) if self._samples else 0


NUM_CAMERAS, IMG_H, IMG_W, STATE_DIM = 2, 360, 640, 8
TASK = "pick up the red block and place it in the bowl"
WARMUP, ITERS = 8, 30


def make_obs():
    imgs = [
        np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
        for _ in range(NUM_CAMERAS)
    ]
    state = np.random.randn(STATE_DIM).astype(np.float32)
    return imgs, state


def sample_obs(reuse_images: bool, fixed_imgs):
    imgs, state = make_obs()
    if reuse_images:
        imgs = fixed_imgs
    return imgs, state


def run_config(label, ckpt, reuse_images=False, **kwargs):
    from molmobot_fast.engine import FastMolmoBot

    flow_steps = kwargs.pop("num_flow_steps", 10)
    bot = FastMolmoBot(checkpoint=ckpt, num_flow_steps=flow_steps, **kwargs)
    fixed_imgs = make_obs()[0] if reuse_images else None

    for _ in range(WARMUP):
        imgs, state = sample_obs(reuse_images, fixed_imgs)
        bot.predict(images=imgs, task=TASK, state=state)
        torch.cuda.synchronize()

    mon = _GPUMon()
    mon.start()
    lats = []
    for _ in range(ITERS):
        imgs, state = sample_obs(reuse_images, fixed_imgs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        actions = bot.predict(images=imgs, task=TASK, state=state)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    mon.stop()

    lats = np.array(lats)
    assert actions.shape == (bot.action_horizon, bot.action_dim), f"Bad shape {actions.shape}"
    assert np.isfinite(actions).all(), "NaN/Inf in actions"

    del bot
    torch.cuda.empty_cache()
    return {
        "label": label,
        "mean": np.mean(lats),
        "med": np.median(lats),
        "p95": np.percentile(lats, 95),
        "std": np.std(lats),
        "gpu": mon.util,
        "mem": mon.mem,
        "hz": 1000 / np.mean(lats) * 16,
    }

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def print_table(rows):
    base = rows[0]["mean"]
    W = 92

    print()
    print(f"{BOLD}{'─' * W}{RESET}")
    print(f"{BOLD}  MolmoBot-Fast  Ablation Benchmark   (H100 80GB, bf16, 2×640×360 cameras){RESET}")
    print(f"{BOLD}{'─' * W}{RESET}")
    hdr = (
        f"  {'Configuration':<42} {'Mean':>7} {'Eff.Hz':>7} {'GPU%':>5} {'Speedup':>8}"
    )
    print(f"{BOLD}{hdr}{RESET}")
    print(f"  {'─' * 88}")

    for i, r in enumerate(rows):
        sp = base / r["mean"]
        sp_str = f"{sp:.1f}x"
        if sp > 1.2:
            sp_col = GREEN
        elif sp < 0.9:
            sp_col = RED
        else:
            sp_col = DIM

        mean_s = f"{r['mean']:.1f}ms"
        hz_s = f"{r['hz']:.0f}"
        gpu_s = f"{r['gpu']:.0f}%"

        if i == 0:
            line = (
                f"  {r['label']:<42} {mean_s:>7} {hz_s:>7} {gpu_s:>5} {DIM}{'base':>8}{RESET}"
            )
        else:
            line = (
                f"  {r['label']:<42} {mean_s:>7} {hz_s:>7} {gpu_s:>5} {sp_col}{sp_str:>8}{RESET}"
            )
        print(line)

        if i == 0 or (i < len(rows) - 1 and rows[i + 1].get("_sep")):
            print(f"  {'─' * 88}")

    print(f"{BOLD}{'─' * W}{RESET}")
    print()


def main():
    from huggingface_hub import snapshot_download

    print("Downloading model ...")
    ckpt = snapshot_download("allenai/MolmoBot-DROID")

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} ({p.total_memory / 1024**3:.0f}GB)")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}\n")

    rows = []

    print("▸ [1/7] Baseline (no optimizations) ...")
    rows.append(
        run_config(
            "Baseline (no optimizations)",
            ckpt,
            cuda_graph=False,
            flash_attn=False,
            compile_backbone=False,
            async_preprocess=False,
        )
    )

    print("▸ [2/7] + CUDA Graph ...")
    rows.append(
        run_config(
            "+ CUDA Graph",
            ckpt,
            cuda_graph=True,
            flash_attn=False,
            compile_backbone=False,
            async_preprocess=False,
        )
    )

    print("▸ [3/7] + CUDA Graph + FlashAttention-2 ...")
    rows.append(
        run_config(
            "+ CUDA Graph + FlashAttention-2",
            ckpt,
            cuda_graph=True,
            flash_attn=True,
            compile_backbone=False,
            async_preprocess=False,
        )
    )

    print("▸ [4/7] + CUDA Graph + FA2 + Compiled Backbone ...")
    rows.append(
        run_config(
            "+ CUDA Graph + FA2 + Compiled Backbone",
            ckpt,
            cuda_graph=True,
            flash_attn=True,
            compile_backbone=True,
            async_preprocess=False,
        )
    )

    print("▸ [5/7] + All opts (async pipeline) ...")
    rows.append(
        run_config(
            "+ All opts (10 steps)",
            ckpt,
            cuda_graph=True,
            flash_attn=True,
            compile_backbone=True,
            async_preprocess=True,
        )
    )
    rows[-1]["_sep"] = True

    print("▸ [6/7] All opts, 5 flow steps ...")
    rows.append(
        run_config(
            "All opts, 5 flow steps",
            ckpt,
            cuda_graph=True,
            flash_attn=True,
            compile_backbone=True,
            async_preprocess=True,
            num_flow_steps=5,
        )
    )

    rows[-1]["_sep"] = True

    print("▸ [7/7] All opts + backbone cache (5 steps, same obs) ...")
    rows.append(
        run_config(
            "All opts + backbone cache (5 steps, same obs)",
            ckpt,
            cuda_graph=True,
            flash_attn=True,
            compile_backbone=True,
            async_preprocess=True,
            cache_backbone=True,
            num_flow_steps=5,
            reuse_images=True,
        )
    )

    print_table(rows)


if __name__ == "__main__":
    main()
