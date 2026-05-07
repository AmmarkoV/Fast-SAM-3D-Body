#!/usr/bin/env bash
# fast_sam_3dbody_cpp/setup.sh
#
# 1. Build the C++ shared library and CLI (cmake + make).
# 2. Create a Python venv with the packages required by both Python frontends:
#      fast_sam_3dbody_frontend.py      – needs only opencv-python + numpy
#      fast_sam_3dbody_frontend-3D.py   – also needs torch, pyrender, roma, etc.
#
# Usage (from repo root or from fast_sam_3dbody_cpp/):
#   bash fast_sam_3dbody_cpp/setup.sh
#   bash fast_sam_3dbody_cpp/setup.sh --cuda-arch 89        # RTX 4090
#   bash fast_sam_3dbody_cpp/setup.sh --ort-dir /opt/onnxruntime
#   bash fast_sam_3dbody_cpp/setup.sh --cpu-only            # no CUDA torch
#   bash fast_sam_3dbody_cpp/setup.sh --skip-build          # venv only
#   bash fast_sam_3dbody_cpp/setup.sh --skip-venv           # build only
#   bash fast_sam_3dbody_cpp/setup.sh -j 4                  # limit make jobs

set -euo pipefail


THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
VENV_DIR="${SCRIPT_DIR}/../venv"

# ── Defaults ───────────────────────────────────────────────────────────────────
CUDA_ARCH=""
ORT_DIR=""
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
JOBS=$(nproc 2>/dev/null || echo 4)
SKIP_BUILD=0
SKIP_VENV=0

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda-arch)    CUDA_ARCH="$2";            shift 2 ;;
        --cuda-arch=*)  CUDA_ARCH="${1#*=}";       shift ;;
        --ort-dir)      ORT_DIR="$2";              shift 2 ;;
        --ort-dir=*)    ORT_DIR="${1#*=}";         shift ;;
        --cpu-only)     TORCH_INDEX="https://download.pytorch.org/whl/cpu"; shift ;;
        --skip-build)   SKIP_BUILD=1;              shift ;;
        --skip-venv)    SKIP_VENV=1;               shift ;;
        -j)             JOBS="$2";                 shift 2 ;;
        -j*)            JOBS="${1#-j}";            shift ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | head -20 | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done



# ── External dependency: RGBDAcquisition ───────────────────────────────────────
_RGBDA_REPO="https://github.com/AmmarkoV/RGBDAcquisition"
_RGBDA_DIR="RGBDAcquisition"

echo "=== Setting up RGBDAcquisition ==="
if [[ -d "${_RGBDA_DIR}/.git" ]]; then
    echo "  Repository already exists — pulling latest …"
    git -C "${_RGBDA_DIR}" pull --ff-only \
        || echo "  Warning: git pull failed; continuing with existing checkout."
else
    echo "  Cloning ${_RGBDA_REPO} …"
    git clone "${_RGBDA_REPO}" "${_RGBDA_DIR}"
fi

_make_symlink() {
    local link="$1" target="$2"
    if [[ -L "${link}" ]]; then
        echo "  Symlink '${link}' already exists — skipping."
    elif [[ -e "${link}" ]]; then
        echo "  Warning: '${link}' exists but is not a symlink — skipping."
    else
        ln -s "${target}" "${link}"
        echo "  Created: ${link} → ${target}"
    fi
    if [[ ! -e "${link}" ]]; then
        echo "  Warning: '${link}' target does not exist yet (run after a full clone)."
    fi
}

_make_symlink "GraphicsEngine" \
    "${_RGBDA_DIR}/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library"
_make_symlink "AmMatrix" \
    "${_RGBDA_DIR}/tools/AmMatrix"




# ── C++ build ──────────────────────────────────────────────────────────────────
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "=== Building C++ library and CLI ==="
    mkdir -p "${BUILD_DIR}"

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
    [[ -n "${CUDA_ARCH}" ]] && CMAKE_ARGS+=" -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
    [[ -n "${ORT_DIR}" ]]   && CMAKE_ARGS+=" -DONNX_RUNTIME_DIR=${ORT_DIR}"

    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" ${CMAKE_ARGS}
    cmake --build "${BUILD_DIR}" -- -j"${JOBS}"

    echo ""
    echo "Build outputs:"
    for f in "${BUILD_DIR}/libfast_sam_3dbody.so" "${BUILD_DIR}/fast_sam_3dbody_run"; do
        [[ -f "$f" ]] && echo "  $(du -sh "$f" | cut -f1)  $f" || echo "  [missing] $f"
    done
fi

# ── Python venv ────────────────────────────────────────────────────────────────
if [[ "${SKIP_VENV}" -eq 0 ]]; then
    echo ""
    echo "=== Creating Python venv at ${VENV_DIR} ==="

    python3 -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    pip install --upgrade pip --quiet

    echo "--- Core (both frontends) ---"
    pip install numpy opencv-python

    echo "--- 3D frontend: torch ---"
    pip install torch torchvision --index-url "${TORCH_INDEX}"

    echo "--- 3D frontend: sam_3d_body deps ---"
    # pyrender + trimesh: 3D mesh rendering
    # roma: rotation representations
    # einops: tensor ops used in backbone/decoder modules
    # timm: ViT layer utilities (drop_path, trunc_normal_)
    # omegaconf + yacs: config loading in sam_3d_body.utils.config
    # huggingface_hub: checkpoint auto-download on first run
    pip install pyrender trimesh roma einops timm omegaconf yacs huggingface_hub braceexpand pytorch_lightning termcolor

    deactivate

    echo ""
    echo "Venv ready: ${VENV_DIR}"
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the venv, then run from the repo root:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  cd ${REPO_ROOT}"
echo ""
echo "  # Lightweight frontend (2D skeleton only):"
echo "  python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py --from assets/teaser.png"
echo ""
echo "  # 3D frontend (full mesh rendering):"
echo "  python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend-3D.py --from assets/teaser.png"
echo ""
echo "Models must be prepared first (once, from the repo root with the original venv active):"
echo "  python fast_sam_3dbody_cpp/prepare_models.py --checkpoint ./checkpoints/sam-3d-body-dinov3"
