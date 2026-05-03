#!/usr/bin/env python3
"""
verify_transforms.py — End-to-end 1:1 verification of the C++ render math
against the Python reference (pyrender), without rendering anything.

Strategy:
  1. Run the C engine + Python MHR on a reference frame to get
        pred_vertices [V,3]   in camera-adjacent space (already Y,Z flipped)
        pred_cam_t    [3]     CLIFF translation
        focal_length  scalar
  2. Project every vertex to image-pixel coordinates two ways:
        (A) PyRender path  (= the original code's reference)
            pyrender does: mesh.apply_transform(Rx(180))
                           camera at world position (-tx, ty, tz)
                           IntrinsicsCamera (fx, fy, W/2, H/2)
                           OpenGL projection, NDC y-up, save flips vertically.
            Closed form (after the vertical save flip):
               pixel_x = fx * (X+tx) / (Z+tz) + W/2
               pixel_y = fy * (Y+ty) / (Z+tz) + H/2
        (B) C++ OpenGL path  (mhr_pose_driver.h matrices reproduced exactly)
            view = diag(1, -1, -1, 1) with translation (tx, -ty, -tz)
            proj = standard GL perspective from focal length
            v_clip = proj * view * v_model
            ndc = v_clip[:3] / v_clip[3]
            screen_y_GL = (ndc.y + 1) * H/2     (origin bottom-left)
            pixel_y_image = H - 1 - screen_y_GL  (because glReadPixels then cv::flip)
  3. Diff per-vertex pixel positions — should be zero (within floating point).
     Also dump the model bounds in every space (model, view, NDC) so we can
     visually sanity-check the camera placement.

The script does NOT need a display — it only multiplies matrices.

Usage:
    python fast_sam_3dbody_cpp/scripts/verify_transforms.py \\
        --image notebook/images/dancing.jpg \\
        --lib-dir fast_sam_3dbody_cpp/build \\
        --onnx-dir fast_sam_3dbody_cpp/onnx
"""
import argparse
import ctypes
import os
import sys

import numpy as np
import cv2
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _repo_root)

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "fsb_frontend",
    os.path.join(_repo_root, "fast_sam_3dbody_cpp", "fast_sam_3dbody_frontend-3D.py"),
)
_fsb_frontend = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fsb_frontend)
FsbConfig    = _fsb_frontend.FsbConfig
FsbResult    = _fsb_frontend.FsbResult
load_library = _fsb_frontend.load_library

from sam_3d_body.build_models import load_sam_3d_body  # noqa: E402


# ── Matrix helpers ──────────────────────────────────────────────────────────

def mhr_camera_matrices(focal_length: float, pred_cam_t, img_w: int, img_h: int):
    """Reproduce mhr_pose_driver.h::mhr_camera_matrices in pure numpy.

    Returns (proj, view) as 4x4 column-major numpy arrays."""
    near, far = 0.01, 100.0
    p00 = 2.0 * focal_length / float(img_w)
    p11 = 2.0 * focal_length / float(img_h)
    p22 = -(far + near) / (far - near)
    p32 = -2.0 * far * near / (far - near)

    # Column-major: out_proj[col, row]
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = p00
    proj[1, 1] = p11
    proj[2, 2] = p22
    proj[3, 2] = p32
    proj[2, 3] = -1.0
    proj[3, 3] = 0.0
    # Above is column-major-as-(col,row).  Convert to standard row-major (row, col):
    proj = proj.T.copy()

    tx, ty, tz = pred_cam_t
    view = np.array([
        [1.0,  0.0,  0.0,  tx],
        [0.0, -1.0,  0.0, -ty],
        [0.0,  0.0, -1.0, -tz],
        [0.0,  0.0,  0.0,  1.0],
    ], dtype=np.float64)
    return proj, view


def project_opengl(verts, focal_length, pred_cam_t, W, H):
    """Project N vertices through the C++ OpenGL pipeline (no rasterization)."""
    proj, view = mhr_camera_matrices(focal_length, pred_cam_t, W, H)
    mvp = proj @ view  # row-major matmul
    Vh = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)  # [N, 4]
    clip = (mvp @ Vh.T).T  # [N, 4]
    ndc = clip[:, :3] / clip[:, 3:4]
    # OpenGL viewport: ndc.x in [-1,1] → screen.x in [0,W], ndc.y in [-1,1] → screen.y in [0,H]
    # Origin at bottom-left.  These are continuous pixel coordinates.
    screen_x = (ndc[:, 0] + 1.0) * W * 0.5
    screen_y_gl = (ndc[:, 1] + 1.0) * H * 0.5
    # Convert continuous GL pixel coords (origin bottom-left) to image coords (top-left).
    # cv::flip(img, 0) maps pixel center y_gl ↔ y_img with y_img = H - y_gl.
    pixel_y = H - screen_y_gl
    return np.stack([screen_x, pixel_y], axis=1), ndc, clip


def project_pyrender(verts, focal_length, pred_cam_t, W, H):
    """Project via the analytical pyrender formula (single equation).

    The original Python pipeline:
       camera_translation = (-tx, ty, tz)
       Rx(180) applied to verts
    Net (algebraic) result on pixel coords:
       pixel_x = fx*(X+tx)/(Z+tz) + W/2
       pixel_y = fy*(Y+ty)/(Z+tz) + H/2
    where (X,Y,Z) is the CAMERA-ADJACENT input vertex.
    """
    fx = fy = focal_length
    cx_img = 0.5 * W
    cy_img = 0.5 * H
    X = verts[:, 0]
    Y = verts[:, 1]
    Z = verts[:, 2]
    tx, ty, tz = pred_cam_t
    denom = Z + tz
    pixel_x = fx * (X + tx) / denom + cx_img
    pixel_y = fy * (Y + ty) / denom + cy_img
    return np.stack([pixel_x, pixel_y], axis=1)


# ── Run pipeline once to get reference data ────────────────────────────────

def run_pipeline(args):
    print(f"[*] Loading SAM-3D-Body Python model from {args.checkpoint}")
    model, _ = load_sam_3d_body(
        checkpoint_path=args.checkpoint,
        mhr_path=args.mhr_model,
        device=args.device,
    )
    model.eval()
    faces = model.head_pose.faces.cpu().numpy()

    print(f"[*] Loading C engine from {args.lib_dir}")
    lib = load_library(args.lib_dir)
    handle = lib.fsb_create()
    cfg = FsbConfig(
        onnx_dir        = args.onnx_dir.encode(),
        gguf_path       = os.path.join(args.onnx_dir, "pipeline.gguf").encode(),
        yolo_path       = os.path.join(args.onnx_dir, "yolo.onnx").encode(),
        cuda_device     = args.cuda,
        skip_body_model = 1,
        person_thresh   = 0.5,
        person_nms_iou  = 0.45,
        max_persons     = 1,
        focal_x         = 0.0, focal_y = 0.0,
        principal_x     = 0.0, principal_y = 0.0,
    )
    if not lib.fsb_load(handle, ctypes.byref(cfg)):
        raise RuntimeError("fsb_load failed")

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)
    H, W = frame.shape[:2]

    bgr_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    ResultArray = FsbResult * 8
    results_buf = ResultArray()
    n = lib.fsb_process_bgr(handle, bgr_ptr, W, H, results_buf, 8)
    print(f"[*] C engine returned {n} person(s)")
    if n == 0:
        raise RuntimeError("No detections")

    r = results_buf[0]
    def _t(arr, n):
        return torch.tensor(list(arr)[:n], dtype=torch.float32, device=args.device).unsqueeze(0)

    with torch.no_grad():
        out = model.head_pose.mhr_forward(
            global_trans=torch.zeros(1, 3, device=args.device),
            global_rot=_t(r.global_rot, 3),
            body_pose_params=_t(r.body_pose, 133),
            hand_pose_params=_t(r.hand_pose, 108),
            scale_params=_t(r.scale, 28),
            shape_params=_t(r.shape, 45),
            expr_params=_t(r.face_params, 72),
            return_keypoints=True,
        )
    if isinstance(out, tuple):
        verts, j3d = out[0], out[1]
    else:
        verts, j3d = out, None

    verts = verts.clone()
    verts[..., [1, 2]] *= -1
    if j3d is not None:
        j3d = j3d[:, :70].clone()
        j3d[..., [1, 2]] *= -1

    pred_vertices = verts[0].cpu().float().numpy()
    pred_cam_t = np.array(list(r.pred_cam_t[:3]), dtype=np.float64)
    focal_length = float(r.focal_length)
    j3d_np = j3d[0].cpu().float().numpy() if j3d is not None else None

    lib.fsb_destroy(handle)

    return dict(frame=frame, W=W, H=H, faces=faces,
                pred_vertices=pred_vertices.astype(np.float64),
                pred_cam_t=pred_cam_t, focal_length=focal_length,
                j3d=j3d_np)


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    cpp_dir = os.path.join(_repo_root, "fast_sam_3dbody_cpp")
    p = argparse.ArgumentParser()
    p.add_argument("--image",      default=os.path.join(_repo_root, "notebook/images/dancing.jpg"))
    p.add_argument("--lib-dir",    default=os.path.join(cpp_dir, "build"))
    p.add_argument("--onnx-dir",   default=os.path.join(cpp_dir, "onnx"))
    p.add_argument("--checkpoint", default=os.path.join(_repo_root, "checkpoints/sam-3d-body-dinov3/model.ckpt"))
    p.add_argument("--mhr-model",  default=os.path.join(_repo_root, "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"))
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cuda",       type=int, default=0)
    p.add_argument("--out-dir",    default="/tmp/verify_transforms")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = run_pipeline(args)
    V = data["pred_vertices"]
    cam_t = data["pred_cam_t"]
    fl = data["focal_length"]
    W, H = data["W"], data["H"]

    print(f"\nImage: {W}x{H}")
    print(f"focal_length = {fl:.4f}")
    print(f"pred_cam_t   = {cam_t}")
    print(f"verts bounds:")
    print(f"  X [{V[:,0].min():.4f}, {V[:,0].max():.4f}]")
    print(f"  Y [{V[:,1].min():.4f}, {V[:,1].max():.4f}]")
    print(f"  Z [{V[:,2].min():.4f}, {V[:,2].max():.4f}]")

    # Project both ways
    px_pyrender = project_pyrender(V, fl, cam_t, W, H)
    px_opengl, ndc, clip = project_opengl(V, fl, cam_t, W, H)

    print(f"\nNDC bounds (OpenGL path):")
    print(f"  ndc.x [{ndc[:,0].min():.4f}, {ndc[:,0].max():.4f}]")
    print(f"  ndc.y [{ndc[:,1].min():.4f}, {ndc[:,1].max():.4f}]")
    print(f"  ndc.z [{ndc[:,2].min():.4f}, {ndc[:,2].max():.4f}]")

    print(f"\nProjected pixel bounds:")
    print(f"  pyrender x [{px_pyrender[:,0].min():.2f}, {px_pyrender[:,0].max():.2f}]   "
          f"y [{px_pyrender[:,1].min():.2f}, {px_pyrender[:,1].max():.2f}]")
    print(f"  opengl   x [{px_opengl[:,0].min():.2f}, {px_opengl[:,0].max():.2f}]   "
          f"y [{px_opengl[:,1].min():.2f}, {px_opengl[:,1].max():.2f}]")

    diff = np.abs(px_pyrender - px_opengl)
    print(f"\nMax per-vertex pixel disagreement: dx={diff[:,0].max():.6f}  dy={diff[:,1].max():.6f}")
    print(f"Mean disagreement:                 dx={diff[:,0].mean():.6f}  dy={diff[:,1].mean():.6f}")

    if diff.max() < 1e-2:
        print("\n*** PROJECTION MATH MATCHES — pyrender ≡ C++ OpenGL view+proj ***")
    else:
        print("\n!!! PROJECTION MATH DIVERGES — there is a bug in mhr_pose_driver.h !!!")

    # Save data for any follow-up debugging
    np.savez(os.path.join(args.out_dir, "ref_data.npz"),
             frame=data["frame"], faces=data["faces"],
             pred_vertices=V, pred_cam_t=cam_t,
             focal_length=fl, W=W, H=H,
             j3d=data["j3d"])
    print(f"\nSaved reference data to {args.out_dir}/ref_data.npz")


if __name__ == "__main__":
    main()
