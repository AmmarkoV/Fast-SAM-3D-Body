#!/usr/bin/env python3
"""Extract LBS buffers from body_model.pt → body_model.lbs (binary)."""
import struct, sys, os
import numpy as np
import torch

MAGIC   = 0x4C425300  # 'LBS\0'
VERSION = 1

def main():
    pt_path  = sys.argv[1] if len(sys.argv) > 1 else "fast_sam_3dbody_cpp/onnx/body_model.pt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "fast_sam_3dbody_cpp/onnx/body_model.lbs"

    print(f"Loading {pt_path} ...")
    m  = torch.jit.load(pt_path, map_location="cpu")
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
    print("All shapes OK.")

    # Header: 8 × u32 (little-endian)
    # magic, version, n_joints, n_skin, n_verts, n_shape_pc, n_face_pc, pt_cols
    # pt_rows = n_joints*7 = 889
    header = struct.pack("<8I", MAGIC, VERSION, 127, 51337, 18439, 45, 72, 249)

    print(f"Writing {out_path} ...")
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(PT.astype(np.float32).tobytes())           # 889*249*4 = 885,228 B
        f.write(offsets.astype(np.float32).tobytes())      # 127*3*4   =   1,524 B
        f.write(prerotations.astype(np.float32).tobytes()) # 127*4*4   =   2,032 B
        f.write(parents.astype(np.int32).tobytes())        # 127*4     =     508 B
        f.write(inv_bind.astype(np.float32).tobytes())     # 127*8*4   =   4,064 B
        f.write(skin_jidx.astype(np.int32).tobytes())      # 51337*4   = 205,348 B
        f.write(skin_w.astype(np.float32).tobytes())       # 51337*4   = 205,348 B
        f.write(skin_vidx.astype(np.int32).tobytes())      # 51337*4   = 205,348 B
        f.write(base_shape.astype(np.float32).tobytes())   # 18439*3*4 = 221,268 B
        f.write(shape_vecs.astype(np.float32).tobytes())   # 45*18439*3*4 = 9,957,060 B
        f.write(face_vecs.astype(np.float32).tobytes())    # 72*18439*3*4 =15,931,296 B

    sz = os.path.getsize(out_path)
    print(f"Done — {sz:,} bytes ({sz/1e6:.1f} MB)")
    print(f"  joint_parents[0:5] = {parents[:5].tolist()}")
    print(f"  PT[0,0:3]          = {PT[0,:3].tolist()}")
    print(f"  base_shape[0]      = {base_shape[0].tolist()}")
    print(f"  inv_bind[0]        = {inv_bind[0].tolist()}")

if __name__ == "__main__":
    main()
