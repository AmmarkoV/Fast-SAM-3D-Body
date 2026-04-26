#!/usr/bin/env python3
# Export the MHR body mesh topology (T-pose vertices + face indices) to a .tri file
# that can be loaded by model_loader_tri.h for use in the C rendering pipeline.
#
# Run once offline:
#   python tools/export_mhr_topology.py \
#       --checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt \
#       --mhr_path   checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
#       --output     fast_sam_3dbody_cpp/body_mesh.tri

import argparse
import struct
import numpy as np
import torch

# TRI format version (must match TRI_LOADER_VERSION in model_loader_tri.h)
TRI_LOADER_VERSION = 10

# Header struct layout (little-endian, 136 bytes):
#   char TRIMagic[5]  +  3 bytes padding  +  11 x uint32  +  16 x float  +  5 x uint32
TRI_HEADER_FMT = "<5s3x" + "I" * 11 + "16f" + "I" * 5


def pack_header(name_bytes, n_verts, n_normals, n_tex, n_colors, n_indices, n_bones):
    identity = [1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1]
    return struct.pack(
        TRI_HEADER_FMT,
        b"TRI3D",         # TRIMagic[5]
        # --- 3x padding ---
        TRI_LOADER_VERSION,  # triType
        len(name_bytes),     # nameSize
        4,                   # floatSize = sizeof(float)
        0,                   # drawType = triangles
        n_verts,             # numberOfVertices  (total floats = vertexCount * 3)
        n_normals,           # numberOfNormals   (total floats = vertexCount * 3)
        n_tex,               # numberOfTextureCoords
        n_colors,            # numberOfColors
        n_indices,           # numberOfIndices   (total uint32s = triangleCount * 3)
        n_bones,             # numberOfBones
        0,                   # rootBone
        *identity,           # boneGlobalInverseTransform[16]
        0, 0, 0, 0, 0,       # textureData{Width,Height,Channels,BindGLBuffer,UploadedToGPU}
    )


def compute_vertex_normals(verts, faces):
    """Per-vertex normals by area-weighted averaging of face normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)        # (F, 3) unnormalized face normals
    lengths = np.linalg.norm(fn, axis=1, keepdims=True)
    fn /= np.maximum(lengths, 1e-8)
    vn = np.zeros_like(verts)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    lengths = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.maximum(lengths, 1e-8)).astype(np.float32)


def get_tpose_verts(model):
    """Run MHR with zero pose/shape to get the rest-pose mesh."""
    head = model.head_pose
    device = next(head.parameters()).device
    z = lambda *shape: torch.zeros(*shape, device=device)
    with torch.no_grad():
        verts = head.mhr_forward(
            global_trans=z(1, 3),
            global_rot=z(1, 3),
            body_pose_params=z(1, 133),
            hand_pose_params=z(1, 108),
            scale_params=z(1, 28),
            shape_params=z(1, 45),
            expr_params=z(1, 72),
            return_keypoints=False,
        )
    # verts is (1, V, 3); flip Y/Z back (mhr_forward does not flip here)
    v = verts[0].cpu().numpy().astype(np.float32)
    return v


def write_tri(output_path, verts, faces, normals, name="mhr_body"):
    name_bytes = name.encode("ascii")
    V = verts.shape[0]
    F = faces.shape[0]

    header = pack_header(
        name_bytes,
        n_verts=V * 3,       # float count
        n_normals=V * 3,     # float count
        n_tex=0,
        n_colors=0,
        n_indices=F * 3,     # uint32 count
        n_bones=0,
    )

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(name_bytes)
        f.write(verts.flatten().astype(np.float32).tobytes())
        f.write(normals.flatten().astype(np.float32).tobytes())
        f.write(faces.flatten().astype(np.uint32).tobytes())

    size_kb = (len(header) + len(name_bytes) +
               V * 3 * 4 + V * 3 * 4 + F * 3 * 4) / 1024
    print(f"Wrote {output_path}  ({V} vertices, {F} faces, {size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
        default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--mhr_path",
        default="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    parser.add_argument("--output",
        default="fast_sam_3dbody_cpp/body_mesh.tri")
    args = parser.parse_args()

    from sam_3d_body import load_sam_3d_body
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} ...")
    model, _ = load_sam_3d_body(args.checkpoint, device=device, mhr_path=args.mhr_path)

    print("Extracting face topology ...")
    faces = model.head_pose.faces.cpu().numpy().astype(np.int32)   # (36874, 3)

    print("Computing T-pose vertices ...")
    verts = get_tpose_verts(model)                                  # (18439, 3)

    print("Computing vertex normals ...")
    normals = compute_vertex_normals(verts, faces)                  # (18439, 3)

    write_tri(args.output, verts, faces, normals)


if __name__ == "__main__":
    main()
