#!/usr/bin/env python3
"""Extract LBS buffers from body_model.pt → body_model.lbs (binary).

Version 2 adds scale_mean [68] and scale_comps [28×68] at the end of the file.
These come from the full model checkpoint (--ckpt), not from body_model.pt.
If --ckpt is not provided, scale data is written as all-zeros (no PCA scaling).
"""
import argparse, struct, os
import numpy as np
import torch

MAGIC   = 0x4C425300  # 'LBS\0'
VERSION = 2
N_SCALE_PC  = 28
N_SCALE_OUT = 68

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt",    nargs="?", default="fast_sam_3dbody_cpp/onnx/body_model.pt")
    ap.add_argument("out",   nargs="?", default="fast_sam_3dbody_cpp/onnx/body_model.lbs")
    ap.add_argument("--ckpt", default="", help="Full model checkpoint (.ckpt) with head_pose.scale_mean/scale_comps")
    args = ap.parse_args()

    print(f"Loading {args.pt} ...")
    m  = torch.jit.load(args.pt, map_location="cpu")
    sd = m.state_dict()

    def g(key, dtype=None):
        a = sd[key].numpy()
        return a.astype(dtype) if dtype else a

    PT          = g("character_torch.parameter_transform.parameter_transform")   # [889,249] f32
    offsets     = g("character_torch.skeleton.joint_translation_offsets")        # [127,3]   f32
    prerotations= g("character_torch.skeleton.joint_prerotations")               # [127,4]   f32
    parents     = g("character_torch.skeleton.joint_parents",     np.int32)      # [127]     i32
    inv_bind    = g("character_torch.linear_blend_skinning.inverse_bind_pose")   # [127,8]   f32
    skin_jidx   = g("character_torch.linear_blend_skinning.skin_indices_flattened", np.int32)  # [51337] i32
    skin_w      = g("character_torch.linear_blend_skinning.skin_weights_flattened")            # [51337] f32
    skin_vidx   = g("character_torch.linear_blend_skinning.vert_indices_flattened", np.int32) # [51337] i32
    base_shape  = g("character_torch.blend_shape.base_shape")                    # [18439,3]         f32
    shape_vecs  = g("character_torch.blend_shape.shape_vectors")                 # [45,18439,3]      f32
    face_vecs   = g("face_expressions_model.shape_vectors")                      # [72,18439,3]      f32

    assert PT.shape          == (889, 249),   PT.shape
    assert offsets.shape     == (127, 3),     offsets.shape
    assert prerotations.shape== (127, 4),     prerotations.shape
    assert parents.shape     == (127,),       parents.shape
    assert inv_bind.shape    == (127, 8),     inv_bind.shape
    assert skin_jidx.shape   == (51337,),     skin_jidx.shape
    assert skin_w.shape      == (51337,),     skin_w.shape
    assert skin_vidx.shape   == (51337,),     skin_vidx.shape
    assert base_shape.shape  == (18439, 3),   base_shape.shape
    assert shape_vecs.shape  == (45, 18439, 3), shape_vecs.shape
    assert face_vecs.shape   == (72, 18439, 3), face_vecs.shape
    print("All body model shapes OK.")

    # Load scale PCA parameters from the full model checkpoint
    if args.ckpt:
        print(f"Loading scale params from {args.ckpt} ...")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        sd2 = ckpt if isinstance(ckpt, dict) and "backbone.encoder.cls_token" in ckpt else ckpt.get("state_dict", ckpt)
        scale_mean  = sd2["head_pose.scale_mean"].numpy().astype(np.float32)   # [68]
        scale_comps = sd2["head_pose.scale_comps"].numpy().astype(np.float32)  # [28,68]
        assert scale_mean.shape  == (N_SCALE_OUT,),            scale_mean.shape
        assert scale_comps.shape == (N_SCALE_PC, N_SCALE_OUT), scale_comps.shape
        print(f"  scale_mean[:5] = {scale_mean[:5].tolist()}")
    else:
        print("WARNING: no --ckpt provided, writing zero scale_mean/scale_comps (shape will be wrong)")
        scale_mean  = np.zeros(N_SCALE_OUT, dtype=np.float32)
        scale_comps = np.zeros((N_SCALE_PC, N_SCALE_OUT), dtype=np.float32)

    # Header: 8 × u32 (little-endian)
    # magic, version, n_joints, n_skin, n_verts, n_shape_pc, n_face_pc, pt_cols
    # pt_rows = n_joints*7 = 889
    # version 2 appends scale_mean [68] + scale_comps [28×68] after face_vecs
    header = struct.pack("<8I", MAGIC, VERSION, 127, 51337, 18439, 45, 72, 249)

    print(f"Writing {args.out} ...")
    with open(args.out, "wb") as f:
        f.write(header)
        f.write(PT.astype(np.float32).tobytes())           # 889*249*4
        f.write(offsets.astype(np.float32).tobytes())      # 127*3*4
        f.write(prerotations.astype(np.float32).tobytes()) # 127*4*4
        f.write(parents.astype(np.int32).tobytes())        # 127*4
        f.write(inv_bind.astype(np.float32).tobytes())     # 127*8*4
        f.write(skin_jidx.astype(np.int32).tobytes())      # 51337*4
        f.write(skin_w.astype(np.float32).tobytes())       # 51337*4
        f.write(skin_vidx.astype(np.int32).tobytes())      # 51337*4
        f.write(base_shape.astype(np.float32).tobytes())   # 18439*3*4
        f.write(shape_vecs.astype(np.float32).tobytes())   # 45*18439*3*4
        f.write(face_vecs.astype(np.float32).tobytes())    # 72*18439*3*4
        # version 2: scale PCA data
        f.write(scale_mean.tobytes())                      # 68*4
        f.write(scale_comps.tobytes())                     # 28*68*4

    sz = os.path.getsize(args.out)
    print(f"Done — {sz:,} bytes ({sz/1e6:.1f} MB)")
    print(f"  joint_parents[0:5] = {parents[:5].tolist()}")
    print(f"  PT[0,0:3]          = {PT[0,:3].tolist()}")
    print(f"  base_shape[0]      = {base_shape[0].tolist()}")
    print(f"  inv_bind[0]        = {inv_bind[0].tolist()}")

if __name__ == "__main__":
    main()
