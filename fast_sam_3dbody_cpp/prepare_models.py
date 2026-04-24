#!/usr/bin/env python3
"""
One-shot script to produce all runtime models for the C++ pipeline.

Outputs into fast_sam_3dbody_cpp/onnx/:
  backbone.onnx        - DINOv3 backbone
  decoder.onnx         - SAM-3D-Body decoder
  body_model.onnx      - MHR body model
  pipeline.gguf        - MHR + camera projection heads
  yolo.onnx            - YOLO11m-pose detector

Usage:
  # From the project root:
  python fast_sam_3dbody_cpp/prepare_models.py

  # Or with custom checkpoint:
  python fast_sam_3dbody_cpp/prepare_models.py \
      --checkpoint /path/to/sam-3d-body-dinov3

  # Skip a step (useful if ONNX is already exported):
  python fast_sam_3dbody_cpp/prepare_models.py --skip yolo
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
ONNX_DIR = os.path.join(ROOT, "onnx")

DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "sam-3d-body-dinov3")
DEFAULT_YOLO = os.path.join(PROJECT_ROOT, "yolo11m-pose.pt")

# Alternately look in checkpoints/yolo/
YOLO_ALT = os.path.join(PROJECT_ROOT, "checkpoints", "yolo", "yolo11m-pose.pt")


def _run(cmd, **kw):
    """Run a subprocess, propagating stdout/stderr."""
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd, **kw)


def export_onnx(checkpoint: str):
    """Export backbone.onnx, decoder.onnx, body_model.onnx."""
    script = os.path.join(ROOT, "export_onnx.py")
    if not os.path.exists(script):
        print(f"ERROR: {script} not found")
        sys.exit(1)

    env = {**os.environ,
           "SKIP_KEYPOINT_PROMPT": "1",
           "MHR_NO_CORRECTIVES": "1"}

    _run([sys.executable, script,
          "--checkpoint", checkpoint,
          "--output", ONNX_DIR,
          "--stage", "all"], env=env)


def export_gguf(checkpoint: str, dtype: str = "f16"):
    """Export pipeline.gguf (MHR + camera heads)."""
    script = os.path.join(ROOT, "convertModelToGGUF.py")
    if not os.path.exists(script):
        print(f"ERROR: {script} not found")
        sys.exit(1)

    _run([sys.executable, script,
          "--checkpoint", checkpoint,
          "--output", os.path.join(ONNX_DIR, "pipeline.gguf"),
          "--dtype", dtype])


def export_yolo():
    """Export yolo.onnx from the YOLO11m-pose checkpoint via ultralytics."""
    yolo_path = DEFAULT_YOLO if os.path.exists(DEFAULT_YOLO) else YOLO_ALT
    if not os.path.exists(yolo_path):
        print(f"WARNING: YOLO checkpoint not found at {yolo_path}")
        print("  Skipping YOLO export. Place yolo.onnx manually or run:")
        print(f"    ultralytics export {yolo_path} export format=onnx")
        return

    onnx_path = os.path.join(ONNX_DIR, "yolo.onnx")
    print(f"  Exporting YOLO from {yolo_path}")

    from ultralytics import YOLO
    model = YOLO(yolo_path)
    out = model.export(format="onnx", imgsz=640, half=False, simplify=True)

    # ultralytics creates the .onnx beside the source .pt by default
    src = os.path.splitext(yolo_path)[0] + ".onnx"
    if os.path.abspath(src) != os.path.abspath(onnx_path):
        shutil.copy2(src, onnx_path)
        print(f"  Copied {src} -> {onnx_path}")
    else:
        onnx_path = src

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  yolo.onnx  {size_mb:.1f} MB  OK")


def main():
    ap = argparse.ArgumentParser(description="Prepare all runtime models for fast_sam_3dbody_cpp")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="SAM-3D-Body checkpoint dir (contains model.ckpt + assets/mhr_model.pt)")
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"],
                    help="GGUF weight dtype (default: f16)")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["onnx", "gguf", "yolo"],
                    help="Skip one or more steps")
    args = ap.parse_args()

    os.makedirs(ONNX_DIR, exist_ok=True)

    # Validate checkpoint
    ckpt_file = os.path.join(args.checkpoint, "model.ckpt")
    mhr_file  = os.path.join(args.checkpoint, "assets", "mhr_model.pt")
    if not os.path.exists(ckpt_file):
        print(f"ERROR: model.ckpt not found at {ckpt_file}")
        sys.exit(1)
    if not os.path.exists(mhr_file):
        print(f"ERROR: mhr_model.pt not found at {mhr_file}")
        sys.exit(1)

    steps = [
        ("Export ONNX models (backbone, decoder, body_model)", "onnx" not in args.skip, export_onnx, [args.checkpoint]),
        ("Export GGUF heads (pipeline.gguf)", "gguf" not in args.skip, export_gguf, [args.checkpoint, args.dtype]),
        ("Export YOLO detector (yolo.onnx)", "yolo" not in args.skip, export_yolo, []),
    ]

    for idx, (name, should_run, fn, args_) in enumerate(steps, 1):
        print(f"\n{'=' * 60}")
        if should_run:
            print(f"[  {idx}/{len(steps)}  ] {name} ...")
            fn(*args_)
        else:
            print(f"[  skip  ] {name}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Model summary:")
    expected = [
        ("backbone.onnx",   "required"),
        ("decoder.onnx",    "required"),
        ("pipeline.gguf",   "required"),
        ("yolo.onnx",       "required"),
        ("body_model.pt",   "optional – ggml impl planned"),
    ]
    for fname, note in expected:
        fpath = os.path.join(ONNX_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1e6
            print(f"  {fname:25s}  {size:7.1f} MB  ✓")
        else:
            print(f"  {fname:25s}  MISSING  ({note})")

    print(f"\nModels ready in: {ONNX_DIR}")
    print("Build the C++ pipeline:")
    print(f"  cd {ROOT}/build && cmake .. && make -j$(nproc)")
    print("Run (MHR params only, fastest):")
    print(f"  ./fast_sam_3dbody_run --onnx-dir {ONNX_DIR} --gguf {ONNX_DIR}/pipeline.gguf --yolo {ONNX_DIR}/yolo.onnx --from 0 --skip-body")


if __name__ == "__main__":
    main()
