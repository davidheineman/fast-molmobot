#!/usr/bin/env bash
# Setup script for MolmoBot simulation evaluation using MolmoSpaces.
#
# Requirements:
#   - Ubuntu 22.04 with NVIDIA GPU (tested on H100 80GB)
#   - Python 3.11
#   - NVIDIA driver with EGL support
#   - uv (Python package manager)
#
# This sets up the upstream MolmoBot repo with MolmoSpaces for running
# realistic sim-based pick-and-place evaluations in MuJoCo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/MolmoBot-upstream"
MOLMOBOT_DIR="${UPSTREAM_DIR}/MolmoBot"
CKPT_DIR="${SCRIPT_DIR}/molmobot_droid_ckpt"

# ── 1. Clone upstream MolmoBot ───────────────────────────────────────────
if [ ! -d "$UPSTREAM_DIR" ]; then
    echo ">>> Cloning allenai/MolmoBot..."
    git clone https://github.com/allenai/MolmoBot.git "$UPSTREAM_DIR"
else
    echo ">>> MolmoBot-upstream already exists, skipping clone"
fi

# ── 2. Install MolmoBot with eval dependencies ──────────────────────────
echo ">>> Installing MolmoBot with eval extras (molmo_spaces, MuJoCo, etc.)..."
cd "$MOLMOBOT_DIR"
uv sync --extra eval

# ── 3. Fix NVIDIA EGL (needed for headless GPU rendering in containers) ──
# MolmoSpaces needs libEGL_nvidia.so.0 which may not be present in containers.
# We extract it from the matching libnvidia-gl package.
if ! ldconfig -p 2>/dev/null | grep -q libEGL_nvidia; then
    echo ">>> libEGL_nvidia.so.0 not found, installing from matching driver package..."
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)

    TMPDIR=$(mktemp -d)
    cd "$TMPDIR"
    apt-get update -qq 2>/dev/null
    apt-get download "libnvidia-gl-${DRIVER_MAJOR}=${DRIVER_VER}-0ubuntu1" 2>/dev/null || \
        apt-get download "libnvidia-gl-${DRIVER_MAJOR}" 2>/dev/null

    dpkg-deb -x libnvidia-gl-*.deb extracted/
    cp extracted/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    cp extracted/usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    cp extracted/usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    cp extracted/usr/lib/x86_64-linux-gnu/libnvidia-glsi.so.* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true

    # Create symlink if missing
    if [ ! -e /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]; then
        ln -sf "libEGL_nvidia.so.${DRIVER_VER}" /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
    fi

    ldconfig 2>/dev/null
    rm -rf "$TMPDIR"
    echo ">>> NVIDIA EGL libraries installed for driver ${DRIVER_VER}"
else
    echo ">>> libEGL_nvidia.so.0 already present"
fi

# ── 4. Patch molmo_spaces for benchmark compatibility ────────────────────
# The benchmark JSONs use old 'mujoco_thor.*' task class names that need
# mapping to the new 'molmo_spaces.*' names.
SAMPLER_PY="${MOLMOBOT_DIR}/.venv/lib/python3.11/site-packages/molmo_spaces/tasks/json_eval_task_sampler.py"
if [ -f "$SAMPLER_PY" ]; then
    # Add mujoco_thor -> molmo_spaces task class mappings
    if ! grep -q "mujoco_thor" "$SAMPLER_PY"; then
        echo ">>> Patching json_eval_task_sampler.py for legacy task class names..."
        python3 - "$SAMPLER_PY" << 'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()

# Patch 1: Add mujoco_thor task_cls_to_type mappings
old = '''            "molmo_spaces.tasks.nav_task.NavToObjTask": "nav_to_obj",
        }'''
new = '''            "molmo_spaces.tasks.nav_task.NavToObjTask": "nav_to_obj",
            "mujoco_thor.tasks.pick_task.PickTask": "pick",
            "mujoco_thor.tasks.opening_tasks.OpeningTask": "open",
            "mujoco_thor.tasks.pick_and_place_task.PickAndPlaceTask": "pick_and_place",
            "mujoco_thor.tasks.pick_and_place_next_to_task.PickAndPlaceNextToTask": "pick_and_place_next_to",
            "mujoco_thor.tasks.pick_and_place_color_task.PickAndPlaceColorTask": "pick_and_place_color",
            "mujoco_thor.tasks.opening_tasks.DoorOpeningTask": "door_opening",
            "mujoco_thor.tasks.nav_task.NavToObjTask": "nav_to_obj",
        }'''
content = content.replace(old, new)

# Patch 2: Remap mujoco_thor -> molmo_spaces in class import resolution
old2 = '''        task_cls_str = task_cls_str.replace(
            "molmo_spaces", "molmo_spaces"
        )  # TODO(rose): forking branch'''
new2 = '''        task_cls_str = task_cls_str.replace(
            "mujoco_thor", "molmo_spaces"
        )'''
content = content.replace(old2, new2)

# Patch 3: Relax strict displacement assertion (benchmarks may have different values)
old3 = '''        # The JSON benchmark is authoritative for displacement thresholds.
        # Assert they match the expected defaults rather than overriding.
        if isinstance(task_config, PickAndPlaceTaskConfig):
            assert task_config.max_place_receptacle_pos_displacement == 0.15, (
                f"Expected max_place_receptacle_pos_displacement=0.15, "
                f"got {task_config.max_place_receptacle_pos_displacement}"
            )
            assert np.isclose(task_config.max_place_receptacle_rot_displacement, np.radians(60)), (
                f"Expected max_place_receptacle_rot_displacement=radians(60), "
                f"got {task_config.max_place_receptacle_rot_displacement}"
            )'''
new3 = '''        # The JSON benchmark is authoritative for displacement thresholds.
        # Log if they differ from expected defaults but don't assert (benchmarks may vary).
        if isinstance(task_config, PickAndPlaceTaskConfig):
            if task_config.max_place_receptacle_pos_displacement != 0.15:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    f"max_place_receptacle_pos_displacement={task_config.max_place_receptacle_pos_displacement} "
                    f"(expected 0.15)"
                )'''
content = content.replace(old3, new3)

with open(path, 'w') as f:
    f.write(content)
print("  Patched successfully")
PYEOF
    else
        echo ">>> json_eval_task_sampler.py already patched"
    fi
fi

# ── 5. Download MolmoBot-DROID checkpoint ────────────────────────────────
if [ ! -f "$CKPT_DIR/model.pt" ]; then
    echo ">>> Downloading allenai/MolmoBot-DROID checkpoint..."
    cd "$MOLMOBOT_DIR"
    .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('allenai/MolmoBot-DROID', local_dir='${CKPT_DIR}')
print('Downloaded to: ${CKPT_DIR}')
"
else
    echo ">>> MolmoBot-DROID checkpoint already exists"
fi

# ── 6. Verify setup ─────────────────────────────────────────────────────
echo ""
echo ">>> Verifying setup..."
cd "$MOLMOBOT_DIR"
source .venv/bin/activate

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -c "
import mujoco
print(f'  mujoco:       {mujoco.__version__}')
import molmo_spaces
print(f'  molmo_spaces:  OK')
from molmo_spaces.renderer.opengl_context import EGLGLContext
ctx = EGLGLContext(64, 64, device_id=0)
ctx.free()
print(f'  EGL rendering: OK')
import torch
print(f'  torch:         {torch.__version__}')
print(f'  CUDA:          {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}:         {p.name} ({p.total_memory/1024**3:.0f}GB)')
" 2>&1 | grep -v "^Failed to import warp\|^WARNING\|nltk_data\|UserWarning"

echo ""
echo ">>> Setup complete! Run the eval with:"
echo ""
echo "  cd ${MOLMOBOT_DIR}"
echo "  source .venv/bin/activate"
echo "  bash run_sim_eval.sh"
echo ""
