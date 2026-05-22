#!/usr/bin/env python3
"""Render the same image with pyrender (Python reference) for side-by-side comparison."""
import os, sys
import numpy as np
import cv2

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

from sam_3d_body.visualization.renderer import Renderer

def main():
    data = np.load("/tmp/verify_transforms/ref_data.npz", allow_pickle=False)
    frame = data["frame"].astype(np.uint8)
    verts = data["pred_vertices"]
    cam_t = data["pred_cam_t"].astype(np.float32)
    fl    = float(data["focal_length"])
    faces = data["faces"]

    renderer = Renderer(focal_length=fl, faces=faces)
    out = renderer(verts.astype(np.float32),
                   cam_t,
                   frame.copy(),
                   mesh_base_color=(0.65, 0.75, 0.9),
                   scene_bg_color=(1, 1, 1)) * 255
    cv2.imwrite("/tmp/pyrender_ref.png", out.astype(np.uint8))
    print("Saved /tmp/pyrender_ref.png")

if __name__ == "__main__":
    main()
