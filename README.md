Adds CUDA graph compilation, flash attention and kv-caching to speed-up [MolmoBot](https://github.com/allenai/MolmoBot) inference. 

On 1 H100 this leads to 4.5x speedup. When using 10 -> 5 flow steps and assuming generous cache hits, this leads to 11x speedup.

## setup

```bash
# clone this repo
git clone git@github.com:davidheineman/fast-molmobot.git
cd fast-molmobot

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
────────────────────────────────────────────────────────────────────────────────────────────
  MolmoBot-Fast  Ablation Benchmark   (H100 80GB, bf16, 2×640×360 cameras)
────────────────────────────────────────────────────────────────────────────────────────────
  Configuration                                 Mean  Eff.Hz  GPU%  Speedup
  ────────────────────────────────────────────────────────────────────────────────────────
  Baseline (no optimizations)                261.9ms      61   33%     base
  ────────────────────────────────────────────────────────────────────────────────────────
  + CUDA Graph                                77.3ms     207   83%     3.4x
  + CUDA Graph + FlashAttention-2             81.6ms     196   81%     3.2x
  + CUDA Graph + FA2 + Compiled Backbone      64.4ms     248   85%     4.1x
  ────────────────────────────────────────────────────────────────────────────────────────
  + All opts (10 steps)                       40.4ms     396   91%     6.5x
  ────────────────────────────────────────────────────────────────────────────────────────
  All opts, 5 flow steps                      30.2ms     529   86%     8.7x
  All opts + backbone cache (5 steps, same obs)  13.3ms    1207   64%    19.8x
────────────────────────────────────────────────────────────────────────────────────────────
```

### discussion

**vLLM**: Why don't we see 50x speedups like vLLM vs. HF? This is because we aren't batching requests. vLLM's biggest speedup is 10x throughput via better managing batches.

**Changes to model**: The next two biggest jumps would come from: (1) training a smaller model (a MolmoBot 1.7B or 0.6B) and using speculating decoding and (2) using flow-matching to perform diffusion in one step.

**Failures**: I tried FP8 inference and TensorRT, but single-batch inference is memory-bandwidth bound, not comput-bound.

**Profiler**: Finally, when we run a profiler, we can see the bottleneck is in the 500M DiT model which runs 10 diffusion steps for each chunk of actions:

| Stage | Time | % |
|---|---:|---:|
| AE Flow Loop (x10) | 35.6 ms | 58.0% |
| Backbone (ViT+LLM) | 19.9 ms | 32.5% |
| CPU Preprocess | 4.2 ms | 6.8% |
| AE Context Build | 1.6 ms | 2.6% |
| H2D + D2H transfers | 0.12 ms | ~0.2% |

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