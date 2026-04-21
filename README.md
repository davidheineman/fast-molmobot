Adds CUDA graph compilation, flash attention and kv-caching to speed-up MolmoBot inference. 

On 1 H100 this leads to 4.5x speedup. When using 10 -> 5 flow steps and assuming generous cache hits, this leads to 11x speedup.

## setup

```bash
# clone this repo
git clone git@github.com:davidheineman/molmobot.git
cd molmobot

# clone + install MolmoBot
git clone git@github.com:allenai/MolmoBot.git
pip install -e MolmoBot/MolmoBot[eval]

# install this repo
pip install flash-attn --no-build-isolation
pip install -e .
```

## benchmark

```bash
python run_benchmark.py
```

output (1 H100, PyTorch 2.7.1+cu126):

```sh
────────────────────────────────────────────────────────────────────────────────────────────────────────────
  MolmoBot-Fast  Ablation Benchmark   (H100 80GB, bf16, 2×640×360 cameras)
────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Configuration                                 Mean     Med     P95  Eff.Hz  GPU%    Mem  Speedup
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  Baseline (no optimizations)                311ms   285ms   395ms       51   30%  11.6G     base
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  + CUDA Graph                                86ms    85ms    87ms      187   84%  12.1G     3.6x
  + CUDA Graph + FlashAttention-2             99ms    98ms   102ms      162   74%  12.8G     3.2x
  + CUDA Graph + FA2 + Compiled Backbone      73ms    72ms    74ms      220   83%  12.1G     4.3x
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  + All opts (10 steps)                       69ms    68ms    73ms      233   87%  12.8G     4.5x
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
  All opts, 5 flow steps                      45ms    44ms    46ms      359   83%  12.1G     7.0x
  All opts + backbone cache (5 steps)         28ms    28ms    29ms      567   83%  12.8G    11.0x
────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Best: All opts + backbone cache (5 steps) at 28ms (11.0x faster than baseline)
```

### discussion

Why don't we see 50x speedups like vLLM vs. HF? This is because we aren't batching requests:

| | vLLM vs HuggingFace | MolmoBot-Fast vs baseline |
|---|---|---|
| Bottleneck | Memory-bandwidth (decode) | Compute (dense forward pass) |
| Key win | Batching + PagedAttention | CUDA graph + KV caching |
| Concurrency | Many requests | Single request |
| Single-request speedup | ~1.5-2x | ~4x |
| Throughput speedup | 10-24x | N/A (batch=1) |

The next two biggest jumps would come from: (1) training a smaller model (a MolmoBot 1.7B or 0.6B) and using speculating decoding and (2) using flow-matching to perform diffusion in one step.

I tried FP8 inference and TensorRT, but single-batch inference is memory-bandwidth bound, not comput-bound.

## api

```python
from molmobot_fast import FastMolmoBot
import numpy as np

bot = FastMolmoBot()  # auto-downloads allenai/MolmoBot-DROID

cam1 = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
cam2 = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
state = np.zeros(8, dtype=np.float32)

actions = bot.predict(
    images=[cam1, cam2],
    task="pick up the red block and place it in the bowl",
    state=state,
)
# actions.shape = (16, 8)  — 16-step action chunk, 8-dim (7 arm + 1 gripper)
```