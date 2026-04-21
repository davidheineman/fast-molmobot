#!/usr/bin/env bash
# Run MolmoBot-DROID simulation evaluation on MolmoSpaces benchmarks.
#
# Usage:
#   bash run_sim_eval.sh                           # default: pick-and-place (200 episodes)
#   bash run_sim_eval.sh --benchmark pick          # Franka pick (200 episodes)
#   bash run_sim_eval.sh --benchmark pick_next_to  # Franka pick-and-place-next-to
#   bash run_sim_eval.sh --max-episodes 10         # quick test with 10 episodes
#
# Prerequisites: run setup_sim_eval.sh first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOLMOBOT_DIR="${SCRIPT_DIR}/MolmoBot-upstream/MolmoBot"
CKPT_DIR="${SCRIPT_DIR}/molmobot_droid_ckpt"
OUTPUT_DIR="${SCRIPT_DIR}/eval_output"
BENCHMARK="pick_and_place"
MAX_EPISODES=""
TASK_HORIZON=600
NUM_WORKERS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        --task-horizon) TASK_HORIZON="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --checkpoint) CKPT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BENCH_BASE="${MOLMOBOT_DIR}/.venv/lib/python3.11/site-packages/assets/benchmarks/molmospaces-bench-v2/ithor"

case "$BENCHMARK" in
    pick_and_place|pnp)
        BENCH_PATH="${BENCH_BASE}/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark"
        EVAL_CONFIG="olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig"
        TASK_HORIZON=600
        ;;
    pick)
        BENCH_PATH="${BENCH_BASE}/FrankaPickHardBench/FrankaPickHardBench_20260212_200ep_json_benchmark"
        EVAL_CONFIG="olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig"
        TASK_HORIZON=400
        ;;
    pick_next_to)
        BENCH_PATH="${BENCH_BASE}/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260212_200ep_json_benchmark"
        EVAL_CONFIG="olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig"
        TASK_HORIZON=600
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        echo "Available: pick_and_place, pick, pick_next_to"
        exit 1
        ;;
esac

if [ ! -d "$BENCH_PATH" ]; then
    echo "Benchmark path not found: $BENCH_PATH"
    echo "Run setup_sim_eval.sh first to download assets."
    exit 1
fi

cd "$MOLMOBOT_DIR"
source .venv/bin/activate

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu

echo "=== MolmoBot Sim Eval ==="
echo "  Benchmark:  $BENCHMARK"
echo "  Checkpoint: $CKPT_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Horizon:    $TASK_HORIZON steps"
echo "  Workers:    $NUM_WORKERS"
if [ -n "$MAX_EPISODES" ]; then
    echo "  Max eps:    $MAX_EPISODES"
fi
echo ""

CMD=(
    python launch_scripts/run_eval.py
    --checkpoint_path "$CKPT_DIR"
    --benchmark_path "$BENCH_PATH"
    --eval_config_cls "$EVAL_CONFIG"
    --task_horizon "$TASK_HORIZON"
    --num_workers "$NUM_WORKERS"
    --output_dir "$OUTPUT_DIR"
)

exec "${CMD[@]}"
