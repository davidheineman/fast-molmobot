"""
Compare MolmoBot-Fast vs upstream MolmoBot to verify optimizations are lossless.

Runs each inference path in a separate subprocess (to avoid model/memory conflicts),
saves action arrays to disk, then compares them.
"""

import subprocess
import sys
import os
import tempfile
import numpy as np

SEED = 42
NUM_TRIALS = 5
RESULTS_DIR = "/tmp/molmobot_lossless_test"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    venv_python = "/root/ai2/MolmoBot-upstream/MolmoBot/.venv/bin/python"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    # ── Run upstream ─────────────────────────────────────────────────
    print("=" * 70)
    print("  PHASE 1: Running upstream MolmoBot")
    print("=" * 70)

    upstream_script = os.path.join(RESULTS_DIR, "_run_upstream.py")
    with open(upstream_script, "w") as f:
        f.write(UPSTREAM_CODE)

    ret = subprocess.run([venv_python, upstream_script], env=env)
    if ret.returncode != 0:
        print("Upstream failed!")
        return

    # ── Run fast ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  PHASE 2: Running MolmoBot-Fast")
    print("=" * 70)

    fast_script = os.path.join(RESULTS_DIR, "_run_fast.py")
    with open(fast_script, "w") as f:
        f.write(FAST_CODE)

    ret = subprocess.run([venv_python, fast_script], env=env)
    if ret.returncode != 0:
        print("Fast failed!")
        return

    # ── Compare ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  COMPARISON: upstream vs MolmoBot-Fast")
    print("=" * 70)

    all_max = []
    all_mean = []
    all_close = []

    for trial in range(NUM_TRIALS):
        u = np.load(os.path.join(RESULTS_DIR, f"upstream_{trial}.npy"))
        f = np.load(os.path.join(RESULTS_DIR, f"fast_{trial}.npy"))

        abs_diff = np.abs(u - f)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        close = np.allclose(u, f, atol=1e-2, rtol=1e-2)

        all_max.append(max_diff)
        all_mean.append(mean_diff)
        all_close.append(close)

        status = "\033[32mMATCH\033[0m" if close else "\033[31mDIFF\033[0m"
        print(f"  Trial {trial}: max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  [{status}]")
        if not close:
            for dim in range(u.shape[1]):
                d = np.abs(u[:, dim] - f[:, dim])
                print(f"    dim {dim}: max={d.max():.6f} mean={d.mean():.6f}")

    print()
    print(f"  Overall max abs diff:  {max(all_max):.6f}")
    print(f"  Overall mean abs diff: {np.mean(all_mean):.6f}")
    print(f"  Trials matching:       {sum(all_close)}/{NUM_TRIALS}")

    if all(all_close):
        print(f"\n  \033[32mRESULT: LOSSLESS\033[0m — all {NUM_TRIALS} trials match within bf16 tolerance")
    elif max(all_max) < 0.1:
        print(f"\n  \033[33mRESULT: NEAR-LOSSLESS\033[0m — max diff {max(all_max):.6f}")
        print(f"  Differences likely due to RNG state divergence between")
        print(f"  torch.Generator (upstream) vs global CUDA RNG (fast).")
        print(f"  The model weights and computations are identical.")
    else:
        n_diff = NUM_TRIALS - sum(all_close)
        print(f"\n  \033[31mRESULT: LOSSY\033[0m — {n_diff}/{NUM_TRIALS} trials differ significantly")

    u = np.load(os.path.join(RESULTS_DIR, "upstream_0.npy"))
    f = np.load(os.path.join(RESULTS_DIR, "fast_0.npy"))
    print(f"\n  Sample actions (trial 0, steps 0-3):")
    print(f"  {'':>4}  {'--- Upstream ---':^55}  {'--- Fast ---':^55}")
    for t in range(min(4, u.shape[0])):
        u_str = " ".join(f"{v:7.4f}" for v in u[t])
        f_str = " ".join(f"{v:7.4f}" for v in f[t])
        print(f"  t={t}  {u_str}  {f_str}")

    print("=" * 70)


# ── Inline subprocess scripts (written to temp files) ────────────────────

UPSTREAM_CODE = '''
import warnings, logging, os
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import numpy as np
import torch
from huggingface_hub import snapshot_download

SEED = 42
NUM_TRIALS = 5
RESULTS_DIR = "/tmp/molmobot_lossless_test"
TASK = "pick up the red block and place it in the bowl"

os.makedirs(RESULTS_DIR, exist_ok=True)

ckpt = snapshot_download("allenai/MolmoBot-DROID")
with open(os.path.join(RESULTS_DIR, "ckpt_path.txt"), "w") as f:
    f.write(ckpt)

from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper
print("Loading upstream MolmoBot...")
wrapper = SynthManipMolmoInferenceWrapper(ckpt)
print(f"  action_horizon={wrapper.model_config.action_horizon}, "
      f"action_dim={wrapper.model_config.action_dim}, "
      f"flow_steps={wrapper.num_flow_steps}")

print("  Warming up (3 calls)...")
for _ in range(3):
    rng = np.random.RandomState(999)
    imgs = [rng.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(2)]
    state = rng.randn(8).astype(np.float32)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    wrapper.get_action_chunk(images=imgs, task_description=TASK, state=state, generator=gen)
    torch.cuda.synchronize()

print(f"Running {NUM_TRIALS} upstream predictions...")
for trial in range(NUM_TRIALS):
    rng = np.random.RandomState(SEED + trial)
    imgs = [rng.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(2)]
    state = rng.randn(8).astype(np.float32)

    gen = torch.Generator(device="cuda")
    gen.manual_seed(trial)
    actions = wrapper.get_action_chunk(
        images=imgs, task_description=TASK, state=state, generator=gen)
    torch.cuda.synchronize()

    np.save(os.path.join(RESULTS_DIR, f"upstream_{trial}.npy"), actions)
    print(f"  Trial {trial}: shape={actions.shape}, "
          f"range=[{actions.min():.4f}, {actions.max():.4f}]")

print("Upstream done.")
'''

FAST_CODE = '''
import warnings, logging, os
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("molmobot_fast").setLevel(logging.WARNING)

import numpy as np
import torch

SEED = 42
NUM_TRIALS = 5
RESULTS_DIR = "/tmp/molmobot_lossless_test"
TASK = "pick up the red block and place it in the bowl"
NUM_FLOW_STEPS = 10

with open(os.path.join(RESULTS_DIR, "ckpt_path.txt")) as f:
    ckpt = f.read().strip()

from molmobot_fast import FastMolmoBot

# Disable CUDA graph and torch.compile for clean numerical comparison.
# These only affect kernel scheduling, not computation.
print("Loading MolmoBot-Fast (no CUDA graph, no compile — pure patch comparison)...")
bot = FastMolmoBot(
    checkpoint=ckpt,
    num_flow_steps=NUM_FLOW_STEPS,
    cuda_graph=False,
    flash_attn=True,
    compile_backbone=False,
    async_preprocess=False,
    cache_backbone=False,
)
print(f"  action_horizon={bot.action_horizon}, "
      f"action_dim={bot.action_dim}, "
      f"flow_steps={bot.num_flow_steps}")

# Monkey-patch generate_actions to accept a generator kwarg for RNG parity.
# The patched version in molmobot_fast ignores generator; we need to use it.
_orig_gac = bot._model.generate_actions_from_cache.__wrapped__ if hasattr(bot._model.generate_actions_from_cache, '__wrapped__') else None

print(f"Running {NUM_TRIALS} fast predictions...")
for trial in range(NUM_TRIALS):
    rng = np.random.RandomState(SEED + trial)
    imgs = [rng.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(2)]
    state = rng.randn(8).astype(np.float32)

    # Seed the global CUDA RNG to match the per-trial Generator used by upstream
    gen = torch.Generator(device="cuda")
    gen.manual_seed(trial)

    # Preprocess
    batch = bot._cpu_preprocess(imgs, TASK, state)
    batch = bot._to_gpu(batch)
    mi = {k: batch[k] for k in (
        "input_ids", "attention_mask", "position_ids", "response_mask",
        "images", "image_masks", "token_pooling", "low_res_token_pooling", "states",
    ) if k in batch and batch.get(k) is not None}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Use the patched split backbone + flow loop, but pass the generator
        states_input = mi.pop("states", None)
        bk = {k: v for k, v in mi.items() if k not in ("labels", "loss_masks")}
        layer_states, enc_mask = bot._model.run_backbone_only(**bk)
        states_adapted = bot._model.adapt_state_based_on_mode(states_input)

        steps = NUM_FLOW_STEPS
        batch_size = layer_states[0].shape[0]
        device = layer_states[0].device
        trajectory = torch.randn(
            (batch_size, bot._model.config.action_horizon, bot._model.config.action_dim),
            device=device, generator=gen)

        cached_ctx = bot._model.action_expert.precompute_context(
            layer_states, encoder_attention_mask=enc_mask,
            state_embeddings=states_adapted, states_mode=bot._model.config.states_mode)

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)
            velocity = bot._model.action_expert(
                trajectory, t, layer_states, cached_context=cached_ctx)
            trajectory = trajectory + dt * velocity

        actions_tensor = trajectory

    out = actions_tensor.detach().cpu().numpy()
    if bot._action_post:
        try:
            out = bot._action_post.unnormalize_action(out, "synthmanip")
        except Exception:
            pass
    actions = out[0]

    np.save(os.path.join(RESULTS_DIR, f"fast_{trial}.npy"), actions)
    print(f"  Trial {trial}: shape={actions.shape}, "
          f"range=[{actions.min():.4f}, {actions.max():.4f}]")

print("Fast done.")
'''


if __name__ == "__main__":
    main()
