import argparse
import subprocess
import sys
import threading
import time
import logging

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


class _GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.05):
        self._gpu_id, self._interval = gpu_id, interval
        self._samples, self._stop, self._thread = [], threading.Event(), None

    def start(self):
        self._samples.clear(); self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True); self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=2)

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits", "-i", str(self._gpu_id)],
                    text=True, timeout=1).strip()
                parts = out.split(",")
                if len(parts) == 2: self._samples.append((float(parts[0]), float(parts[1])))
            except Exception: pass
            time.sleep(self._interval)

    @property
    def gpu_util(self): return np.mean([s[0] for s in self._samples]) if self._samples else 0
    @property
    def gpu_mem(self): return np.mean([s[1] for s in self._samples]) if self._samples else 0


def run(args):
    from molmobot_fast import FastMolmoBot

    bot = FastMolmoBot(
        checkpoint=args.checkpoint,
        num_flow_steps=args.flow_steps,
        cuda_graph=not args.no_cuda_graph,
        flash_attn=not args.no_flash_attn,
        compile_backbone=not args.no_compile,
        async_preprocess=not args.no_async,
        cache_backbone=args.cache_backbone,
        fp8=args.fp8,
        tensorrt=args.tensorrt,
    )

    def make_obs():
        imgs = [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(2)]
        return imgs, np.random.randn(8).astype(np.float32)

    fixed_imgs = make_obs()[0] if args.cache_backbone else None

    TASK = "pick up the red block"
    log.info("Warmup (5 calls with actual prompt to stabilize CUDA graph) ...")
    for _ in range(5):
        imgs = fixed_imgs if args.cache_backbone else make_obs()[0]
        bot.predict(images=imgs, task=TASK, state=make_obs()[1])
        torch.cuda.synchronize()

    monitor = _GPUMonitor()
    monitor.start()

    latencies = []
    for i in range(args.iterations):
        imgs = fixed_imgs if args.cache_backbone else make_obs()[0]
        _, state = make_obs()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bot.predict(images=imgs, task=TASK, state=state)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 10 == 0:
            log.info(f"  [{i+1}/{args.iterations}] last-10 avg: {np.mean(latencies[-10:]):.1f} ms")

    monitor.stop()
    lat = np.array(latencies)
    cps = 1000.0 / np.mean(lat)

    print()
    print("=" * 60)
    print(f"  Mean latency:    {np.mean(lat):7.1f} ms")
    print(f"  Median latency:  {np.median(lat):7.1f} ms")
    print(f"  P95 latency:     {np.percentile(lat, 95):7.1f} ms")
    print(f"  Chunks/sec:      {cps:7.1f}")
    print(f"  Effective Hz:    {cps * bot.action_horizon:7.0f}  (horizon={bot.action_horizon})")
    print(f"  Flow steps:      {bot.num_flow_steps}")
    print(f"  GPU utilization: {monitor.gpu_util:5.0f} %")
    print(f"  GPU memory:      {monitor.gpu_mem:5.0f} MB")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="MolmoBot-Fast Benchmark")
    p.add_argument("--checkpoint", type=str, default=None, help="Path or auto-download allenai/MolmoBot-DROID")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--cache-backbone", action="store_true", help="Reuse same images (backbone cache test)")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-async", action="store_true")
    p.add_argument("--fp8", action="store_true",
                   help="[LOSSY] FP8 dynamic quantization of all linear layers")
    p.add_argument("--tensorrt", action="store_true",
                   help="[LOSSLESS*] TensorRT backbone compilation (replaces torch.compile)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
