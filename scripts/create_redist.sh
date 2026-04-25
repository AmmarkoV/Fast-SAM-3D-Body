#!/usr/bin/env bash
# scripts/create_redist.sh
#
# Package all runtime model files for the C++ pipeline into a redistributable zip.
#
# The zip can be copied to any machine with:
#   - A compatible GPU (CUDA ≥ 12.x, sm_86 or set CMAKE_CUDA_ARCHITECTURES)
#   - ONNX Runtime (auto-downloaded by CMake if not found)
#   - OpenCV and ggml (fetched by CMake)
#
# Usage:
#   cd /path/to/Fast-SAM-3D-Body
#   bash scripts/create_redist.sh                  # C++ models only
#   bash scripts/create_redist.sh --with-python     # also include Python checkpoint
#   bash scripts/create_redist.sh --with-body       # also include body_model.pt (~664 MB)
#   bash scripts/create_redist.sh --output /tmp     # write zip to /tmp/
#
# Output: fast_sam_3dbody_models_YYYYMMDD.zip
#
# Zip contents:
#   models/
#   ├── backbone.onnx          (stub – weights are in backbone.onnx.data)
#   ├── backbone.onnx.data     (~3.2 GB)
#   ├── decoder.onnx           (~174 MB)
#   ├── pipeline.gguf          (~5 MB)
#   ├── yolo.onnx              (~81 MB)
#   ├── body_model.pt          (~664 MB, optional --with-body)
#   └── python_checkpoint/     (optional --with-python)
#       ├── model.ckpt         (~2.0 GB)
#       ├── model_config.yaml
#       └── assets/
#           └── mhr_model.pt   (~664 MB)

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="${REPO_ROOT}/fast_sam_3dbody_cpp/onnx"
CKPT_DIR="${REPO_ROOT}/checkpoints/sam-3d-body-dinov3"
OUTPUT_DIR="${REPO_ROOT}"
WITH_PYTHON=0
WITH_BODY=0

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-python) WITH_PYTHON=1; shift ;;
        --with-body)   WITH_BODY=1;   shift ;;
        --output)      OUTPUT_DIR="$2"; shift 2 ;;
        --output=*)    OUTPUT_DIR="${1#--output=}"; shift ;;
        -h|--help)
            sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# //' | sed 's/^#//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ────────────────────────────────────────────────────────────────────
check() {
    local f="$1"
    if [[ ! -f "$f" ]]; then
        echo "  [MISSING] $f"
        echo ""
        echo "Run model preparation first:"
        echo "  python fast_sam_3dbody_cpp/prepare_models.py --checkpoint ${CKPT_DIR}"
        exit 1
    fi
    local size
    size=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "  [OK]  $size  $f"
}

# ── Verify required models exist ───────────────────────────────────────────────
echo "Checking required model files …"
check "${ONNX_DIR}/backbone.onnx"
check "${ONNX_DIR}/backbone.onnx.data"
check "${ONNX_DIR}/decoder.onnx"
check "${ONNX_DIR}/pipeline.gguf"
check "${ONNX_DIR}/yolo.onnx"

if [[ "${WITH_BODY}" -eq 1 ]]; then
    check "${ONNX_DIR}/body_model.pt"
fi

if [[ "${WITH_PYTHON}" -eq 1 ]]; then
    check "${CKPT_DIR}/model.ckpt"
    check "${CKPT_DIR}/model_config.yaml"
    check "${CKPT_DIR}/assets/mhr_model.pt"
fi
echo ""

# ── Build the zip ──────────────────────────────────────────────────────────────
DATE=$(date +%Y%m%d)
ZIP_NAME="fast_sam_3dbody_models_${DATE}.zip"
ZIP_PATH="${OUTPUT_DIR}/${ZIP_NAME}"
STAGING=$(mktemp -d)
trap 'rm -rf "${STAGING}"' EXIT

echo "Staging files …"

# C++ inference models
mkdir -p "${STAGING}/models"
cp "${ONNX_DIR}/backbone.onnx"      "${STAGING}/models/"
cp "${ONNX_DIR}/backbone.onnx.data" "${STAGING}/models/"
cp "${ONNX_DIR}/decoder.onnx"       "${STAGING}/models/"
cp "${ONNX_DIR}/pipeline.gguf"      "${STAGING}/models/"
cp "${ONNX_DIR}/yolo.onnx"          "${STAGING}/models/"

if [[ "${WITH_BODY}" -eq 1 ]]; then
    cp "${ONNX_DIR}/body_model.pt"  "${STAGING}/models/"
fi

# Python checkpoint (optional)
if [[ "${WITH_PYTHON}" -eq 1 ]]; then
    mkdir -p "${STAGING}/models/python_checkpoint/assets"
    cp "${CKPT_DIR}/model.ckpt"         "${STAGING}/models/python_checkpoint/"
    cp "${CKPT_DIR}/model_config.yaml"  "${STAGING}/models/python_checkpoint/"
    cp "${CKPT_DIR}/assets/mhr_model.pt" \
                                        "${STAGING}/models/python_checkpoint/assets/"
fi

# README explaining how to use the models
cat > "${STAGING}/models/USAGE.md" << 'USAGE_EOF'
# fast_sam_3dbody models

Place these files in `fast_sam_3dbody_cpp/onnx/` relative to the repo root, then build:

```bash
cd fast_sam_3dbody_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Run:
```bash
./fast_sam_3dbody_run \
    --onnx-dir ../onnx \
    --gguf     ../onnx/pipeline.gguf \
    --yolo     ../onnx/yolo.onnx \
    --from     /path/to/image.jpg
```

Python lightweight frontend (no extra Python deps beyond opencv+numpy):
```bash
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py --from /path/to/image.jpg
```

Python 3D frontend (requires python_checkpoint/ and sam_3d_body package):
```bash
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend-3D.py \
    --checkpoint python_checkpoint/model.ckpt \
    --mhr-model  python_checkpoint/assets/mhr_model.pt \
    --from       /path/to/image.jpg
```
USAGE_EOF

# ── Create zip ────────────────────────────────────────────────────────────────
echo "Creating ${ZIP_PATH} …"
(cd "${STAGING}" && zip -r "${ZIP_PATH}" models/)

echo ""
echo "Done."
echo "  Archive : ${ZIP_PATH}"
echo "  Size    : $(du -sh "${ZIP_PATH}" | cut -f1)"
echo ""
echo "Contents:"
unzip -l "${ZIP_PATH}" | tail -n +4 | head -n -2 | awk '{printf "  %s  %s\n", $1, $4}'
