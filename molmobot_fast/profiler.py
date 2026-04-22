"""
In-depth profiler for MolmoBot-Fast inference.

Produces:
  1. Per-stage timing (preprocess, H2D, backbone, action expert, D2H)
  2. Per-layer breakdown within backbone (each LLM block, each ViT block)
  3. CUDA kernel analysis (top kernels by time, categorized by type)
  4. Memory bandwidth & arithmetic intensity estimates
  5. Roofline analysis (compute-bound vs memory-bound per stage)
  6. Chrome trace file for visual inspection in chrome://tracing

Usage:
    python -m molmobot_fast.profiler                    # full report
    python -m molmobot_fast.profiler --trace out.json   # + Chrome trace
    python -m molmobot_fast.profiler --no-compile       # profile without compile
"""

import argparse
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.cuda

logging.basicConfig(level=logging.WARNING)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("molmobot_fast").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
#  GPU hardware specs (for bandwidth / roofline calculations)
# ═══════════════════════════════════════════════════════════════════════════

GPU_SPECS = {
    "NVIDIA H100 80GB HBM3":  {"hbm_bw_GBs": 3350, "bf16_tflops": 1979, "fp8_tflops": 3958},
    "NVIDIA H100":             {"hbm_bw_GBs": 3350, "bf16_tflops": 1979, "fp8_tflops": 3958},
    "NVIDIA A100-SXM4-80GB":  {"hbm_bw_GBs": 2039, "bf16_tflops":  312, "fp8_tflops":  312},
    "NVIDIA A100-SXM4-40GB":  {"hbm_bw_GBs": 1555, "bf16_tflops":  312, "fp8_tflops":  312},
}

def get_gpu_spec():
    name = torch.cuda.get_device_name(0)
    for key, spec in GPU_SPECS.items():
        if key in name:
            return {**spec, "name": name}
    return {"hbm_bw_GBs": 2000, "bf16_tflops": 300, "fp8_tflops": 300, "name": name}


# ═══════════════════════════════════════════════════════════════════════════
#  CUDA event-based stage timer
# ═══════════════════════════════════════════════════════════════════════════

class CUDATimer:
    """Collects CUDA-timed intervals for named stages."""

    def __init__(self):
        self._records: Dict[str, List[float]] = defaultdict(list)
        self._stack = []

    @contextmanager
    def stage(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._stack.append((name, start, end))

    def sync_and_collect(self):
        torch.cuda.synchronize()
        for name, start, end in self._stack:
            self._records[name].append(start.elapsed_time(end))
        self._stack.clear()

    @property
    def records(self):
        return dict(self._records)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-layer hooks for backbone profiling
# ═══════════════════════════════════════════════════════════════════════════

class LayerProfiler:
    """Attaches CUDA event hooks to individual model layers."""

    def __init__(self):
        self._hooks = []
        self._events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def attach(self, model):
        for i, blk in enumerate(model.transformer.blocks):
            self._attach_one(blk, f"llm.block.{i:02d}")

        vb = getattr(model, "vision_backbone", None)
        if vb:
            iv = getattr(vb, "image_vit", None)
            if iv:
                tr = getattr(iv, "transformer", None)
                if tr:
                    rb = getattr(tr, "resblocks", None)
                    if rb:
                        for i, blk in enumerate(rb):
                            self._attach_one(blk, f"vit.block.{i:02d}")

        ae = model.action_expert
        for i, blk in enumerate(ae.blocks):
            self._attach_one(blk, f"ae.block.{i:02d}")

    def _attach_one(self, module, name):
        def pre_hook(mod, args, _name=name):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            mod._prof_start = ev

        def post_hook(mod, args, output, _name=name):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._events[_name].append((mod._prof_start, end))

        self._hooks.append(module.register_forward_pre_hook(pre_hook))
        self._hooks.append(module.register_forward_hook(post_hook))

    def collect(self) -> Dict[str, List[float]]:
        torch.cuda.synchronize()
        result = {}
        for name, pairs in self._events.items():
            result[name] = [s.elapsed_time(e) for s, e in pairs]
        return result

    def clear(self):
        for v in self._events.values():
            v.clear()

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  Model parameter / memory analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_model_memory(model) -> dict:
    """Count parameters and estimate memory traffic per forward pass."""
    sections = {
        "llm": ("transformer",),
        "vit": ("vision_backbone",),
        "action_expert": ("action_expert",),
    }
    result = {}
    for section, prefixes in sections.items():
        total_params = 0
        total_bytes = 0
        layer_info = []
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in prefixes):
                n = param.numel()
                b = n * param.element_size()
                total_params += n
                total_bytes += b
                layer_info.append((name, n, b, param.shape, param.dtype))
        result[section] = {
            "params": total_params,
            "bytes": total_bytes,
            "params_M": total_params / 1e6,
            "bytes_MB": total_bytes / 1e6,
            "layers": layer_info,
        }
    result["total"] = {
        "params": sum(v["params"] for v in result.values() if isinstance(v, dict) and "params" in v),
        "bytes": sum(v["bytes"] for v in result.values() if isinstance(v, dict) and "bytes" in v),
    }
    result["total"]["params_M"] = result["total"]["params"] / 1e6
    result["total"]["bytes_MB"] = result["total"]["bytes"] / 1e6
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel categorization from torch profiler events
# ═══════════════════════════════════════════════════════════════════════════

KERNEL_CATEGORIES = {
    "gemm":     ["gemm", "cutlass", "cublas", "sm90_xmma", "scaled_mm", "s16816gemm", "Gemm",
                 "nvjet_tst_", "triton_mm", "triton_tem"],
    "attention": ["flash", "fmha", "sdpa", "softmax", "Attention", "dot_product"],
    "elementwise": ["vectorized", "elementwise", "Pointwise", "unrolled", "fused_adam",
                    "CatArrayBatchedCopy", "copy_", "fill_"],
    "reduce":   ["reduce", "welford", "norm", "sum", "mean", "layernorm", "rmsnorm"],
    "memory":   ["memcpy", "memset", "Memcpy", "Memset"],
}

def categorize_kernel(name: str) -> str:
    lower = name.lower()
    for cat, keywords in KERNEL_CATEGORIES.items():
        if any(kw.lower() in lower for kw in keywords):
            return cat
    return "other"


def analyze_profiler_events(prof) -> dict:
    """Extract kernel-level stats from torch.profiler output."""
    events = prof.key_averages()

    kernels = []
    categories = defaultdict(lambda: {"time_us": 0, "count": 0})
    total_cuda_us = 0

    for ev in events:
        t = ev.self_device_time_total
        if t > 0 and ev.device_type == torch.autograd.DeviceType.CUDA:
            cat = categorize_kernel(ev.key)
            kernels.append({
                "name": ev.key[:80],
                "self_cuda_us": t,
                "count": ev.count,
                "avg_us": t / max(ev.count, 1),
                "category": cat,
            })
            categories[cat]["time_us"] += t
            categories[cat]["count"] += ev.count
            total_cuda_us += t

    kernels.sort(key=lambda k: k["self_cuda_us"], reverse=True)

    cat_summary = {}
    for cat, data in sorted(categories.items(), key=lambda x: -x[1]["time_us"]):
        cat_summary[cat] = {
            "time_us": data["time_us"],
            "time_ms": data["time_us"] / 1000,
            "pct": 100.0 * data["time_us"] / max(total_cuda_us, 1),
            "count": data["count"],
        }

    return {
        "top_kernels": kernels[:30],
        "categories": cat_summary,
        "total_cuda_us": total_cuda_us,
        "total_cuda_ms": total_cuda_us / 1000,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Roofline / bandwidth analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_roofline(mem_info: dict, stage_timings: dict, layer_timings: dict,
                     gpu_spec: dict, num_flow_steps: int):
    """Estimate compute vs memory boundedness per stage.

    The backbone (ViT+LLM) runs as one fused stage, so we split its time
    proportionally using per-layer measurements when available.
    """
    results = {}
    backbone_ms = np.median(stage_timings["backbone_full"]) if "backbone_full" in stage_timings and stage_timings["backbone_full"] else 0

    vit_layer_ms = sum(np.median(v) for k, v in layer_timings.items() if k.startswith("vit.block."))
    llm_layer_ms = sum(np.median(v) for k, v in layer_timings.items() if k.startswith("llm.block."))
    total_layer_ms = vit_layer_ms + llm_layer_ms

    if total_layer_ms > 0 and backbone_ms > 0:
        vit_frac = vit_layer_ms / total_layer_ms
        llm_frac = llm_layer_ms / total_layer_ms
    else:
        vit_frac = mem_info["vit"]["bytes"] / (mem_info["vit"]["bytes"] + mem_info["llm"]["bytes"])
        llm_frac = 1.0 - vit_frac

    for section, time_ms, weight_bytes, repeats in [
        ("vit",            backbone_ms * vit_frac, mem_info["vit"]["bytes"],            1),
        ("llm",            backbone_ms * llm_frac, mem_info["llm"]["bytes"],            1),
        ("action_expert",  np.median(stage_timings.get("action_expert_flow", [0])),
                           mem_info["action_expert"]["bytes"], num_flow_steps),
    ]:
        if time_ms <= 0:
            continue
        time_s = time_ms / 1000.0
        total_weight_bytes = weight_bytes * repeats
        achieved_bw_GBs = (total_weight_bytes / 1e9) / time_s
        bw_utilization = achieved_bw_GBs / gpu_spec["hbm_bw_GBs"]

        results[section] = {
            "time_ms": time_ms,
            "weight_bytes_MB": weight_bytes / 1e6,
            "total_read_MB": total_weight_bytes / 1e6,
            "repeats": repeats,
            "achieved_bw_GBs": achieved_bw_GBs,
            "peak_bw_GBs": gpu_spec["hbm_bw_GBs"],
            "bw_utilization": bw_utilization,
            "bound": "memory-bw" if bw_utilization > 0.3 else "kernel-launch/overhead",
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Instrumented predict (splits stages)
# ═══════════════════════════════════════════════════════════════════════════

def profiled_predict(bot, images, task, state, timer: CUDATimer):
    """Run predict with fine-grained CUDA timing of each stage."""
    import torch

    with timer.stage("cpu_preprocess"):
        batch = bot._cpu_preprocess(images, task, state)

    with timer.stage("h2d_transfer"):
        batch = bot._to_gpu(batch)

    mi = {k: batch[k] for k in (
        "input_ids", "attention_mask", "position_ids", "response_mask",
        "images", "image_masks", "token_pooling", "low_res_token_pooling", "states",
    ) if k in batch and batch.get(k) is not None}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        states_arg = mi.pop("states", None)
        images_arg = mi.get("images")

        with timer.stage("backbone_vit"):
            pass

        with timer.stage("backbone_full"):
            backbone_kw = {k: v for k, v in mi.items() if k not in ("labels", "loss_masks")}
            layer_states, enc_mask = bot._model.run_backbone_only(**backbone_kw)

        if states_arg is not None:
            states_proc = bot._model.adapt_state_based_on_mode(states_arg)
        else:
            states_proc = None

        with timer.stage("action_expert_context"):
            steps = bot._num_flow_steps
            batch_size = layer_states[0].shape[0]
            device = layer_states[0].device
            trajectory = torch.randn(
                (batch_size, bot._model.config.action_horizon, bot._model.config.action_dim),
                device=device)
            cached_ctx = bot._model.action_expert.precompute_context(
                layer_states, encoder_attention_mask=enc_mask,
                state_embeddings=states_proc,
                states_mode=bot._model.config.states_mode)

        with timer.stage("action_expert_flow"):
            if bot._model._use_cuda_graph:
                from molmobot_fast.patches import _run_flow_loop_cudagraph
                trajectory = _run_flow_loop_cudagraph(
                    bot._model, trajectory, layer_states, cached_ctx, steps)
            else:
                dt = 1.0 / steps
                for i in range(steps):
                    t = torch.full((batch_size,), i / steps, device=device)
                    velocity = bot._model.action_expert(
                        trajectory, t, layer_states, cached_context=cached_ctx)
                    trajectory.add_(velocity, alpha=dt)

    with timer.stage("d2h_transfer"):
        out = trajectory.detach().cpu().numpy()

    return out[0]


# ═══════════════════════════════════════════════════════════════════════════
#  Report printer
# ═══════════════════════════════════════════════════════════════════════════

B = "\033[1m"
D = "\033[2m"
G = "\033[32m"
Y = "\033[33m"
C = "\033[36m"
R = "\033[31m"
Z = "\033[0m"

def fmt_ms(v): return f"{v:8.2f} ms"
def fmt_pct(v): return f"{v:5.1f}%"
def bar(pct, width=30):
    filled = int(pct / 100 * width)
    return f"{'█' * filled}{'░' * (width - filled)}"


def print_report(stage_timings, layer_timings, kernel_info, mem_info, roofline, gpu_spec, num_flow_steps):
    W = 100
    print(f"\n{B}{'═' * W}{Z}")
    print(f"{B}  MolmoBot-Fast  Inference Profiler{Z}")
    print(f"{B}{'═' * W}{Z}")

    # ── GPU info
    print(f"\n{B}  GPU{Z}: {gpu_spec['name']}")
    print(f"  HBM Bandwidth: {gpu_spec['hbm_bw_GBs']} GB/s  |  BF16: {gpu_spec['bf16_tflops']} TFLOPS  |  FP8: {gpu_spec['fp8_tflops']} TFLOPS")

    # ── Model memory
    print(f"\n{B}  Model Parameters & Weight Memory{Z}")
    print(f"  {'Section':<20} {'Params':>10} {'Memory':>10} {'% of total':>12}")
    print(f"  {'─' * 55}")
    for sec in ["llm", "vit", "action_expert"]:
        info = mem_info[sec]
        pct = 100.0 * info["bytes"] / mem_info["total"]["bytes"]
        print(f"  {sec:<20} {info['params_M']:>8.1f}M {info['bytes_MB']:>8.1f}MB {pct:>10.1f}%")
    print(f"  {'─' * 55}")
    print(f"  {'TOTAL':<20} {mem_info['total']['params_M']:>8.1f}M {mem_info['total']['bytes_MB']:>8.1f}MB")

    # ── Stage timing
    print(f"\n{B}  Stage Breakdown (median of profiled iterations){Z}")
    print(f"  {'Stage':<30} {'Median':>10} {'Mean':>10} {'Std':>8} {'% of total':>12}  Visual")
    print(f"  {'─' * 90}")

    total_ms = 0
    stage_order = ["cpu_preprocess", "h2d_transfer", "backbone_full",
                   "action_expert_context", "action_expert_flow", "d2h_transfer"]
    stage_labels = {
        "cpu_preprocess": "CPU Preprocess",
        "h2d_transfer": "Host → Device",
        "backbone_full": "Backbone (ViT+LLM)",
        "action_expert_context": "AE Context Build",
        "action_expert_flow": f"AE Flow Loop (×{num_flow_steps})",
        "d2h_transfer": "Device → Host",
    }
    for s in stage_order:
        if s in stage_timings and stage_timings[s]:
            total_ms += np.median(stage_timings[s])

    for s in stage_order:
        if s not in stage_timings or not stage_timings[s]:
            continue
        vals = np.array(stage_timings[s])
        med = np.median(vals)
        mean = np.mean(vals)
        std = np.std(vals)
        pct = 100.0 * med / total_ms if total_ms > 0 else 0
        label = stage_labels.get(s, s)
        col = G if pct < 10 else (Y if pct < 40 else R)
        print(f"  {label:<30} {fmt_ms(med)} {fmt_ms(mean)} {std:>6.2f}ms {col}{fmt_pct(pct):>12}{Z}  {bar(pct)}")

    print(f"  {'─' * 90}")
    print(f"  {'TOTAL':<30} {fmt_ms(total_ms)}")

    # ── Per-layer breakdown
    if layer_timings:
        print(f"\n{B}  Per-Layer Breakdown (median per call){Z}")

        for prefix, label in [("vit.block.", "ViT Blocks"), ("llm.block.", "LLM Blocks"), ("ae.block.", "Action Expert Blocks")]:
            layers = {k: v for k, v in layer_timings.items() if k.startswith(prefix)}
            if not layers:
                continue

            total_layer_ms = sum(np.median(v) for v in layers.values())
            print(f"\n  {B}{label}{Z}  (total: {total_layer_ms:.2f} ms)")
            print(f"  {'Layer':<20} {'Median':>10} {'Mean':>10} {'Std':>8} {'% of section':>14}")
            print(f"  {'─' * 65}")

            sorted_layers = sorted(layers.items())
            for name, vals in sorted_layers:
                v = np.array(vals)
                med = np.median(v)
                pct = 100.0 * med / total_layer_ms if total_layer_ms > 0 else 0
                short = name.replace(prefix, "")
                print(f"  {short:<20} {fmt_ms(med)} {fmt_ms(np.mean(v))} {np.std(v):>6.2f}ms {fmt_pct(pct):>14}")

            layer_meds = [np.median(v) for v in layers.values()]
            print(f"  {'─' * 65}")
            print(f"  {'min/max/std':<20} {min(layer_meds):>8.2f}ms / {max(layer_meds):>7.2f}ms / {np.std(layer_meds):>5.2f}ms")

    # ── Bandwidth / roofline
    if roofline:
        print(f"\n{B}  Memory Bandwidth Analysis (batch=1 roofline){Z}")
        print(f"  {'Section':<20} {'Time':>10} {'Weights':>10} {'×':>3} {'Total read':>11} "
              f"{'Achieved BW':>13} {'Peak BW':>10} {'Util':>8}  {'Bound'}")
        print(f"  {'─' * 105}")
        for sec, r in roofline.items():
            util_col = G if r["bw_utilization"] > 0.5 else (Y if r["bw_utilization"] > 0.2 else R)
            rep = r.get("repeats", 1)
            print(f"  {sec:<20} {r['time_ms']:>8.1f}ms {r['weight_bytes_MB']:>8.0f}MB "
                  f"{rep:>3} {r['total_read_MB']:>9.0f}MB "
                  f"{r['achieved_bw_GBs']:>10.0f} GB/s {r['peak_bw_GBs']:>7.0f} GB/s "
                  f"{util_col}{r['bw_utilization']*100:>6.1f}%{Z}  {r['bound']}")

    # ── CUDA kernel analysis
    if kernel_info:
        print(f"\n{B}  CUDA Kernel Time by Category{Z}")
        print(f"  {'Category':<15} {'Time':>10} {'% of GPU':>10} {'Calls':>8}  Visual")
        print(f"  {'─' * 70}")
        for cat, data in kernel_info["categories"].items():
            pct = data["pct"]
            col = R if pct > 40 else (Y if pct > 15 else G)
            print(f"  {cat:<15} {data['time_ms']:>8.1f}ms {col}{pct:>8.1f}%{Z} {data['count']:>8}  {bar(pct, 25)}")

        print(f"\n{B}  Top 20 CUDA Kernels by Self Time{Z}")
        print(f"  {'#':>3} {'Kernel':<65} {'Time':>10} {'Calls':>7} {'Avg':>10} {'Cat':<10}")
        print(f"  {'─' * 108}")
        for i, k in enumerate(kernel_info["top_kernels"][:20]):
            cat_col = R if k["category"] == "gemm" else (Y if k["category"] == "attention" else D)
            print(f"  {i+1:>3} {k['name']:<65} {k['self_cuda_us']/1000:>8.1f}ms {k['count']:>7} "
                  f"{k['avg_us']:>8.1f}µs {cat_col}{k['category']:<10}{Z}")

    # ── Summary
    print(f"\n{B}  Key Findings{Z}")
    if "backbone_full" in stage_timings and stage_timings["backbone_full"]:
        bb_med = np.median(stage_timings["backbone_full"])
        bb_pct = 100.0 * bb_med / total_ms
        print(f"  • Backbone (ViT+LLM): {Y}{bb_pct:.0f}%{Z} of total ({bb_med:.1f}ms / {total_ms:.1f}ms)")
    if "action_expert_flow" in stage_timings and stage_timings["action_expert_flow"]:
        ae_med = np.median(stage_timings["action_expert_flow"])
        ae_pct = 100.0 * ae_med / total_ms
        print(f"  • Action expert flow: {Y}{ae_pct:.0f}%{Z} of total ({ae_med:.1f}ms, {num_flow_steps} steps, "
              f"{ae_med/num_flow_steps:.1f}ms/step)")
    if roofline:
        print()
        for sec, r in roofline.items():
            if r["bw_utilization"] > 0.5:
                print(f"  • {C}{sec}{Z}: {G}memory-bandwidth bound{Z} at {r['bw_utilization']*100:.0f}% of peak "
                      f"({r['achieved_bw_GBs']:.0f}/{r['peak_bw_GBs']} GB/s)")
                print(f"    → INT4 weight quant would read {r['weight_bytes_MB']/4:.0f}MB instead of "
                      f"{r['weight_bytes_MB']:.0f}MB → ~{min(3.5, 1/(1-0.75*r['bw_utilization'])):.1f}x faster")
            elif r["bw_utilization"] > 0.2:
                print(f"  • {C}{sec}{Z}: {Y}partially bandwidth bound{Z} ({r['bw_utilization']*100:.0f}% BW util)")
            else:
                print(f"  • {C}{sec}{Z}: {R}kernel-launch/overhead bound{Z} ({r['bw_utilization']*100:.0f}% BW util, "
                      f"CUDA graph helps here)")

    ae_ctx_med = np.median(stage_timings.get("action_expert_context", [0]))
    if ae_ctx_med > 0:
        print(f"\n  • AE context precompute: {ae_ctx_med:.1f}ms (amortized once per call, not per flow step)")

    print(f"\n{B}{'═' * W}{Z}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="MolmoBot-Fast In-Depth Profiler")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--iterations", type=int, default=10, help="Profiled iterations (after warmup)")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--trace", type=str, default=None, help="Path for Chrome trace JSON output")
    args = p.parse_args()

    gpu_spec = get_gpu_spec()

    # Load model
    print(f"Loading FastMolmoBot (compile={'off' if args.no_compile else 'on'}, "
          f"graph={'off' if args.no_cuda_graph else 'on'}, "
          f"fa2={'off' if args.no_flash_attn else 'on'}, "
          f"flow_steps={args.flow_steps}) ...")

    from molmobot_fast import FastMolmoBot
    bot = FastMolmoBot(
        checkpoint=args.checkpoint,
        num_flow_steps=args.flow_steps,
        cuda_graph=not args.no_cuda_graph,
        flash_attn=not args.no_flash_attn,
        compile_backbone=not args.no_compile,
        async_preprocess=False,
    )

    mem_info = analyze_model_memory(bot._model)

    # Warmup
    print(f"Warmup ({args.warmup} calls) ...")
    for _ in range(args.warmup):
        imgs = [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(2)]
        bot.predict(images=imgs, task="pick up the red block", state=np.random.randn(8).astype(np.float32))
        torch.cuda.synchronize()

    # Phase 1: Stage + layer timing with CUDA events
    print(f"Profiling stages + layers ({args.iterations} iterations) ...")
    timer = CUDATimer()
    layer_prof = LayerProfiler()
    layer_prof.attach(bot._model)

    for _ in range(args.iterations):
        imgs = [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(2)]
        state = np.random.randn(8).astype(np.float32)
        profiled_predict(bot, imgs, "pick up the red block", state, timer)
        timer.sync_and_collect()

    stage_timings = timer.records
    layer_timings = layer_prof.collect()
    layer_prof.detach()

    # Phase 2: torch.profiler for kernel-level analysis
    print("Profiling CUDA kernels (3 iterations) ...")
    kernel_info = None
    trace_path = args.trace

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for _ in range(3):
            imgs = [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(2)]
            state = np.random.randn(8).astype(np.float32)
            profiled_predict(bot, imgs, "pick up the red block", state, CUDATimer())
            torch.cuda.synchronize()

    kernel_info = analyze_profiler_events(prof)

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"Chrome trace saved to: {trace_path}")

    # Roofline analysis
    roofline = compute_roofline(mem_info, stage_timings, layer_timings, gpu_spec, args.flow_steps)

    # Print report
    print_report(stage_timings, layer_timings, kernel_info, mem_info, roofline, gpu_spec, args.flow_steps)


if __name__ == "__main__":
    main()
