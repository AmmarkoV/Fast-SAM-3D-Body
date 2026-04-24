#!/usr/bin/env python3
"""
Convert SAM-3D-Body MHR head + camera head to GGUF.

The GGUF file stores:
  KV metadata  — image normalisation, sizes, output dimensions, default focal length
  Tensors      — weights of the two small FFN heads (MHR proj + camera proj)
                 These run on-device via ggml inside the C++ pipeline, allowing
                 quantisation (Q8_0 / Q4_K_M) for the projection step.

Weight layout (matches ggml ne=[Cout, Cin] for Linear):
  mhr_proj.{fc0,fc1}.{weight,bias}
  cam_proj.{fc0,fc1}.{weight,bias}

Usage:
  cd /path/to/Fast-SAM-3D-Body
  python fast_sam_3dbody_cpp/convertModelToGGUF.py \\
      --checkpoint ./checkpoints/sam-3d-body-dinov3 \\
      --output     ./fast_sam_3dbody_cpp/onnx/pipeline.gguf \\
      --dtype      f16
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _ffn_tensors(ffn_module, prefix: str, dtype_str: str):
    """
    Extract weight/bias pairs from an FFN (sam_3d_body/modules/transformer.py).

    The FFN stores layers as:
      ffn.layers[0] = Sequential(Linear, ReLU, Dropout)
      ffn.layers[1] = Linear
      ffn.layers[2] = Dropout

    Returns list of (name, numpy_array) ready for GGUF.
    """
    tensors = []
    fc_idx = 0
    for child in ffn_module.layers:
        # Sequential holds (Linear, ReLU, Dropout)
        import torch.nn as nn
        if isinstance(child, nn.Sequential):
            for sub in child:
                if isinstance(sub, nn.Linear):
                    w = sub.weight.detach().cpu().numpy()  # [Cout, Cin]
                    b = sub.bias.detach().cpu().numpy()    # [Cout]
                    w = w.astype(np.float16 if dtype_str == "f16" else np.float32)
                    b = b.astype(np.float32)               # biases stay f32
                    tensors.append((f"{prefix}.fc{fc_idx}.weight", w))
                    tensors.append((f"{prefix}.fc{fc_idx}.bias",   b))
                    fc_idx += 1
        elif isinstance(child, nn.Linear):
            w = child.weight.detach().cpu().numpy()
            b = child.bias.detach().cpu().numpy()
            w = w.astype(np.float16 if dtype_str == "f16" else np.float32)
            b = b.astype(np.float32)
            tensors.append((f"{prefix}.fc{fc_idx}.weight", w))
            tensors.append((f"{prefix}.fc{fc_idx}.bias",   b))
            fc_idx += 1
        # Dropout / DropPath — no parameters
    return tensors


# ──────────────────────────────────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────────────────────────────────
def convert(checkpoint_dir: str, output_path: str, dtype_str: str = "f16"):
    import os as _os
    _os.environ.setdefault("MHR_NO_CORRECTIVES", "1")
    _os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")

    print("Loading model …")
    from sam_3d_body.build_models import load_sam_3d_body

    ckpt = os.path.join(checkpoint_dir, "model.ckpt")
    mhr  = os.path.join(checkpoint_dir, "assets", "mhr_model.pt")
    model, cfg = load_sam_3d_body(checkpoint_path=ckpt, mhr_path=mhr)
    model.eval()
    print("Model loaded.")

    # ── collect tensors ───────────────────────────────────────────────────
    print(f"\nCollecting tensors (dtype={dtype_str}) …")
    tensors = []
    tensors += _ffn_tensors(model.head_pose.proj,    "mhr_proj", dtype_str)
    tensors += _ffn_tensors(model.head_camera.proj,  "cam_proj", dtype_str)

    if not tensors:
        raise RuntimeError("No tensors found — check model attribute names")

    for name, arr in tensors:
        print(f"  {name:40s}  {str(arr.shape):20s}  {arr.dtype}")

    # ── metadata ──────────────────────────────────────────────────────────
    img_mean = list(cfg.MODEL.IMAGE_MEAN)
    img_std  = list(cfg.MODEL.IMAGE_STD)
    img_size = int(cfg.MODEL.IMAGE_SIZE[0])

    # Determine FFN dimensions from the collected tensors
    mhr_in_dim  = int(tensors[0][1].shape[1])  # fc0.weight Cin
    mhr_hid_dim = int(tensors[0][1].shape[0])  # fc0.weight Cout
    mhr_out_dim = int(tensors[-3][1].shape[0]) # last weight Cout before cam tensors
    # Camera head follows MHR — last two (weight+bias) are cam_proj.fc1
    cam_out_dim = int(tensors[-1][1].shape[0] if "cam_proj" in tensors[-1][0]
                      else 3)

    # Decoder dim (context dim for backbone features)
    backbone_dim  = 1280  # dinov3_vith16plus
    backbone_stride = 16

    meta = {
        "arch": "sam3dbody_heads",
        "image_size":      img_size,
        "image_mean":      img_mean,
        "image_std":       img_std,
        "backbone_dim":    backbone_dim,
        "backbone_stride": backbone_stride,
        "feat_h":          img_size // backbone_stride,
        "feat_w":          img_size // backbone_stride,
        "decoder_dim":     mhr_in_dim,
        "mhr_hidden_dim":  mhr_hid_dim,
        "npose":           mhr_out_dim,
        # MHR parameter offsets inside npose-vector
        "global_rot_offset": 0,
        "global_rot_dim":    6,
        "body_cont_offset":  6,
        "body_cont_dim":     260,
        "shape_offset":      266,
        "shape_dim":         45,
        "scale_offset":      311,
        "scale_dim":         28,
        "hand_offset":       339,
        "hand_dim":          108,
        "face_offset":       447,
        "face_dim":          72,
        # Camera head
        "cam_out_dim":    cam_out_dim,   # 3 = tx, ty, tz
        # Body model sizes
        "num_vertices":   18439,
        "num_skel_joints": 127,
        "num_keypoints":   70,
        # Defaults
        "default_focal_length": 800.0,
        "person_thresh":  0.5,
        "person_nms":     0.45,
        "dtype":          dtype_str,
    }

    # ── write GGUF ────────────────────────────────────────────────────────
    from gguf import GGUFWriter, GGMLQuantizationType
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    writer = GGUFWriter(output_path, arch="sam3dbody_heads")
    writer.add_string("general.name",        "sam3dbody_heads")
    writer.add_string("general.architecture","sam3dbody_heads")
    writer.add_string("sam3dbody.meta_json",  json.dumps(meta))
    writer.add_uint32("sam3dbody.image_size", img_size)
    writer.add_uint32("sam3dbody.decoder_dim", mhr_in_dim)
    writer.add_uint32("sam3dbody.npose",       mhr_out_dim)

    qtype = GGMLQuantizationType.F16 if dtype_str == "f16" else GGMLQuantizationType.F32
    for name, arr in tensors:
        writer.add_tensor(name, arr, raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n✓ Written {len(tensors)} tensors → {output_path}  ({size_mb:.1f} MB)")
    print(f"  dtype: {dtype_str}  |  image_size: {img_size}")
    print(f"  decoder_dim: {mhr_in_dim}  |  npose: {mhr_out_dim}")
    print("\nQuantise further (optional):")
    print(f"  llama.cpp/build/bin/quantize {output_path} pipeline_q8.gguf Q8_0")


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Convert MHR head + camera head weights to GGUF"
    )
    ap.add_argument("--checkpoint", default="./checkpoints/sam-3d-body-dinov3")
    ap.add_argument("--output",     default="./fast_sam_3dbody_cpp/onnx/pipeline.gguf")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f16")
    args = ap.parse_args()
    convert(args.checkpoint, args.output, args.dtype)


if __name__ == "__main__":
    main()
