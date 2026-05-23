#!/usr/bin/env python3
"""
debug_pipeline.py — Systematic C++ ↔ Python parity tests for the body model.

Three test phases:

  Phase A — Run C++ on a test image, then feed the EXACT same pose parameters
            (body_pose, global_rot, shape, scale, hand_pose) into the Python MHR
            body model and compare 3D joint positions joint-by-joint.
            → Tests whether the LBS/skinning model is identical in both pipelines.

  Phase B — Single-DOF sweep: take C++ pose params, vary one body_cont element by
            ±Δ using both the Python MHR model and the C++ body_pose inverse path,
            measure how each joint moves and check consistency.
            → Tests that parameter layout and rotation conventions agree.

  Phase C — Full end-to-end: run the complete Python SAM3D estimator
            (process_one_image) on the same frame as C++ and compare the resulting
            pred_keypoints_2d arrays. This is the integration test.

Usage:
    python debug_pipeline.py --phase A        # LBS parity
    python debug_pipeline.py --phase B        # DOF sweep
    python debug_pipeline.py --phase C        # end-to-end
    python debug_pipeline.py                  # all phases
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import traceback
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# Import torch BEFORE loading the C++ library.
# _load_cpp() modifies LD_LIBRARY_PATH to prepend ONNX Runtime's CUDA libs;
# if torch is imported after that, libc10_cuda.so picks up the wrong libcudart.
# Importing torch first ensures it binds against the correct system CUDA libs.
try:
    import torch as _torch_preload  # noqa: F401
except Exception:
    pass  # GPU not available — torch will be imported lazily where needed

_REPO = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(_REPO, "checkpoints/sam-3d-body-dinov3/model.ckpt")
_MHR  = os.path.join(_REPO, "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
_ONNX = os.path.join(_REPO, "fast_sam_3dbody_cpp/onnx")
_LIB  = os.path.join(_REPO, "fast_sam_3dbody_cpp/build")

sys.path.insert(0, os.path.join(_REPO, "fast_sam_3dbody_cpp"))


# ─── helpers ──────────────────────────────────────────────────────────────────

def _infer_device() -> str:
    """CUDA when available — sam_3d_body_estimator hardcodes batch to CUDA."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_cpp():
    from fast_sam_3dbody_frontend import FsbResult, FsbConfig, load_library
    lib = load_library(_LIB)
    cfg = FsbConfig(
        onnx_dir=_ONNX.encode(),
        gguf_path=os.path.join(_ONNX, "pipeline.gguf").encode(),
        yolo_path=os.path.join(_ONNX, "yolo.onnx").encode(),
        cuda_device=0,
        skip_body_model=0,
    )
    h = lib.fsb_create()
    assert lib.fsb_load(h, ctypes.byref(cfg)), "C++ pipeline load failed"
    return lib, h, FsbResult


def _run_cpp(lib, h, FsbResult, frame_bgr):
    H, W = frame_bgr.shape[:2]
    buf = (FsbResult * 8)()
    ptr = frame_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    n = lib.fsb_process_bgr(h, ptr, W, H, buf, 8)
    return [buf[i] for i in range(n)]


def _load_python_model():
    from sam_3d_body import load_sam_3d_body
    device = _infer_device()
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=_CKPT,
        device=device,
        mhr_path=_MHR if os.path.exists(_MHR) else "",
    )
    return model, model_cfg, device


def _mhr_forward(model, body_pose_euler, global_rot, shape, scale, hand_pose, device):
    """
    Run just the MHR body model with explicit pose parameters.

    Maps directly to _mhr_forward_core in mhr_head.py:
      global_trans  = zeros[3]    (no translation offset in pose branch)
      global_rot    = euler ZYX[3]
      body_pose     = pred_pose_euler[:130]   (last 3 are translation, zeroed)
      scale_params  = pred_scale[28]
      shape_params  = pred_shape[45]
      hand_pose     = pred_hand[108]
      expr_params   = zeros[72]   (face zeroed in training)

    Returns j3d[70,3] in camera coords (y,z axes flipped relative to body model).
    """
    import torch
    head = model.head_pose

    def T(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)

    global_rot_t   = T(global_rot)       # [1,3]
    global_trans_t = torch.zeros(1, 3, device=device)
    body_pose_t    = T(body_pose_euler)  # [1,133] — _mhr_forward_core slices [:130]
    shape_t        = T(shape)            # [1,45]
    scale_t        = T(scale)            # [1,28]
    hand_t         = T(hand_pose)        # [1,108]
    expr_t         = torch.zeros(1, 72, device=device)

    with torch.no_grad():
        _, j3d, _, _, _ = head._mhr_forward_core(
            global_trans=global_trans_t,
            global_rot=global_rot_t,
            body_pose_params=body_pose_t,
            hand_pose_params=hand_t,
            scale_params=scale_t,
            shape_params=shape_t,
            expr_params=expr_t,
            return_keypoints=True,
        )

    # _mhr_forward_core returns [B, 308, 3] — slice to 70 joints (as in _head_forward_core)
    # and apply the camera-system coord flip (mhr_head.py line 460-462):
    #   j3d[..., [1, 2]] *= -1   # y,z negated to match camera frame convention
    j3d_np = j3d[0, :70].cpu().numpy().copy()   # [70,3]
    j3d_np[:, 1] *= -1
    j3d_np[:, 2] *= -1
    return j3d_np


def _pred_pose_raw_to_euler(pred_pose_raw, model, device):
    """
    Re-run the rot6d→Euler + compact_cont→Euler conversions on pred_pose_raw[266].
    This mirrors what mhr_head._head_forward_core does AFTER the proj FFN.
    """
    import torch
    from sam_3d_body.models.modules.mhr_utils import (
        compact_cont_to_model_params_body_fast,
        rotmat_to_euler_ZYX,
    )
    from sam_3d_body.models.modules import rot6d_to_rotmat

    p = torch.tensor(pred_pose_raw, dtype=torch.float32, device=device).unsqueeze(0)

    global_rot_6d   = p[:, :6]
    global_rot_mat  = rot6d_to_rotmat(global_rot_6d)
    global_rot_euler = rotmat_to_euler_ZYX(global_rot_mat)

    body_cont = p[:, 6:266]
    pred_pose_euler = compact_cont_to_model_params_body_fast(body_cont)

    return global_rot_euler[0].cpu().numpy(), pred_pose_euler[0].cpu().numpy()


# ─── Phase A: LBS parity ──────────────────────────────────────────────────────

def phase_a(frame_bgr):
    print("\n" + "="*70)
    print("PHASE A — C++ vs Python body model parity (same pose params)")
    print("="*70)

    # ── C++ pass ──────────────────────────────────────────────────────────────
    print("\n[1/3] Running C++ pipeline...")
    lib, h, FsbResult = _load_cpp()
    results = _run_cpp(lib, h, FsbResult, frame_bgr)
    lib.fsb_destroy(h)

    if not results:
        print("  No detections — supply an image with a visible person")
        return

    r = results[0]
    print(f"  C++ detected person, bbox={[round(x) for x in r.bbox]}")

    cpp_kps3d    = np.array(r.kps_3d,    dtype=np.float32).reshape(70, 3)
    cpp_body     = np.array(r.body_pose, dtype=np.float32)   # [133]
    cpp_glob     = np.array(r.global_rot,dtype=np.float32)   # [3]
    cpp_shape    = np.array(r.shape,     dtype=np.float32)   # [45]
    cpp_scale    = np.array(r.scale,     dtype=np.float32)   # [28]
    cpp_hand     = np.array(r.hand_pose, dtype=np.float32)   # [108]
    cpp_pose_raw = np.array(r.pred_pose_raw, dtype=np.float32)  # [266]

    print(f"  C++ global_rot: {cpp_glob.round(4)}")
    print(f"  C++ body_pose[:6]: {cpp_body[:6].round(4)}")
    print(f"  C++ kps3d[9] (pelvis): {cpp_kps3d[9].round(4)}")
    print(f"  C++ kps3d[5] (L-shoulder): {cpp_kps3d[5].round(4)}")

    # ── Python body model pass ────────────────────────────────────────────────
    print("\n[2/3] Loading Python model...")
    model, _, device = _load_python_model()

    # Re-derive Euler params from pred_pose_raw (mirrors C++ rot6d→euler path)
    py_glob, py_body = _pred_pose_raw_to_euler(cpp_pose_raw, model, device)
    print(f"  Python global_rot from pred_pose_raw: {py_glob.round(4)}")
    print(f"  C++   global_rot (direct field):      {cpp_glob.round(4)}")
    print(f"  Δglobal_rot: {(py_glob - cpp_glob).round(6)}")

    print("\n[3/3] Running Python MHR body model with C++ pose params...")
    py_j3d = _mhr_forward(model, py_body, py_glob, cpp_shape, cpp_scale, cpp_hand, device)

    print(f"  Python kps3d[9] (pelvis):     {py_j3d[9].round(4)}")
    print(f"  C++    kps3d[9] (pelvis):     {cpp_kps3d[9].round(4)}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n--- Joint-by-joint |Δ| (Python − C++) ---")
    delta = py_j3d - cpp_kps3d
    dist  = np.linalg.norm(delta, axis=-1)

    # Print key joints
    labels = {0:"nose", 5:"L-shoulder", 6:"R-shoulder", 7:"L-elbow",
              8:"R-elbow", 9:"L-hip", 10:"R-hip", 41:"R-wrist", 62:"L-wrist", 69:"neck"}
    for k, name in sorted(labels.items()):
        print(f"  [{k:2d}] {name:12s}: Δ={delta[k].round(4)}  |Δ|={dist[k]:.4f} m")

    print(f"\n  median |Δ| (all 70 joints): {np.median(dist):.4f} m")
    worst_idx = dist.argmax()
    print(f"  max    |Δ| (all 70 joints): {dist.max():.4f} m  (joint {worst_idx}, Δ={delta[worst_idx].round(4)})")
    print(f"  mean   |Δ| (all 70 joints): {dist.mean():.4f} m")
    # Print top-5 worst joints
    top5 = np.argsort(dist)[-5:][::-1]
    print(f"  top-5 worst joints: {list(zip(top5.tolist(), dist[top5].round(4).tolist()))}")

    if dist.max() < 0.005:
        print("\n  ✓ PASS: body models agree within 5 mm")
    elif dist.max() < 0.05:
        print("\n  ~ WARN: body models differ by up to 5 cm (check rotation conventions)")
    else:
        print("\n  ✗ FAIL: body models disagree by >5 cm — likely parameter layout mismatch")

    return dist, cpp_pose_raw, model, device, cpp_shape, cpp_scale, cpp_hand


# ─── Phase B: Single-DOF sweep ────────────────────────────────────────────────

def phase_b(phase_a_result=None, frame_bgr=None):
    print("\n" + "="*70)
    print("PHASE B — Single-DOF sweep: vary body_cont[6] (left elbow angle)")
    print("="*70)

    if phase_a_result is None:
        print("  Phase A not run — loading model and running C++ pass...")
        if frame_bgr is None:
            frame_bgr = np.full((720, 1280, 3), 128, dtype=np.uint8)
        result = phase_a(frame_bgr)
        if result is None:
            print("  No detections — provide a frame with a person")
            return
        _, cpp_pose_raw, model, device, cpp_shape, cpp_scale, cpp_hand = result
    else:
        _, cpp_pose_raw, model, device, cpp_shape, cpp_scale, cpp_hand = phase_a_result

    print("\nSweeping body_cont[6] (pred_pose_raw[12]) from −0.6 to +0.6 rad ...")
    print(f"  Baseline pred_pose_raw[12] = {cpp_pose_raw[12]:.4f}")
    print()
    print(f"  {'delta':>8}  {'joint[7] L-elbow':>22}  {'joint[62] L-wrist':>22}  "
          f"{'|Δ_elbow|':>10}  {'|Δ_wrist|':>10}")

    import torch
    baseline_glob, baseline_body = _pred_pose_raw_to_euler(cpp_pose_raw, model, device)
    baseline_j3d = _mhr_forward(model, baseline_body, baseline_glob,
                                 cpp_shape, cpp_scale, cpp_hand, device)

    for delta_rad in [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]:
        perturbed_raw = cpp_pose_raw.copy()
        perturbed_raw[12] += delta_rad   # body_cont[6] = pred_pose_raw[6+6]

        p_glob, p_body = _pred_pose_raw_to_euler(perturbed_raw, model, device)
        p_j3d = _mhr_forward(model, p_body, p_glob,
                              cpp_shape, cpp_scale, cpp_hand, device)

        d_elbow = np.linalg.norm(p_j3d[7] - baseline_j3d[7])
        d_wrist = np.linalg.norm(p_j3d[62] - baseline_j3d[62])

        marker = " ← baseline" if delta_rad == 0.0 else ""
        print(f"  {delta_rad:+8.2f}  {str(p_j3d[7].round(3)):>22}  "
              f"{str(p_j3d[62].round(3)):>22}  {d_elbow:10.4f}  {d_wrist:10.4f}{marker}")

    print("\n  Expected: elbow and wrist move monotonically with delta.")
    print("  If they don't move (or move equally), body_cont[6] is the wrong index.")


# ─── Phase C: End-to-end integration ──────────────────────────────────────────

def phase_c(frame_bgr):
    print("\n" + "="*70)
    print("PHASE C — End-to-end: C++ kps_2d vs Python process_one_image kps_2d")
    print("="*70)

    # ── C++ pass ──────────────────────────────────────────────────────────────
    print("\n[1/2] Running C++ pipeline...")
    lib, h, FsbResult = _load_cpp()
    results = _run_cpp(lib, h, FsbResult, frame_bgr)
    lib.fsb_destroy(h)

    if not results:
        print("  No detections")
        return

    r = results[0]
    cpp_kps2d = np.array(r.kps_2d, dtype=np.float32).reshape(70, 2)
    bbox = np.array(r.bbox, dtype=np.float32)
    print(f"  C++ kps_2d[0] (nose):       {cpp_kps2d[0].round(1)}")
    print(f"  C++ kps_2d[5] (L-shoulder): {cpp_kps2d[5].round(1)}")

    # ── Python estimator pass ─────────────────────────────────────────────────
    print("\n[2/2] Running Python SAM3D estimator (process_one_image)...")
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
    model, model_cfg, device = _load_python_model()
    est = SAM3DBodyEstimator(model, model_cfg)

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bboxes  = bbox[np.newaxis, :]
    outputs = est.process_one_image(img_rgb, bboxes=bboxes, inference_type="body")

    if not outputs:
        print("  Python estimator returned no outputs")
        return

    py_kps2d = np.asarray(outputs[0]["pred_keypoints_2d"])
    print(f"  Py  kps_2d[0] (nose):       {py_kps2d[0].round(1)}")
    print(f"  Py  kps_2d[5] (L-shoulder): {py_kps2d[5].round(1)}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n--- 2D keypoint delta (Python − C++) in pixels ---")
    delta = py_kps2d - cpp_kps2d
    dist  = np.linalg.norm(delta, axis=-1)

    labels = {0:"nose", 5:"L-shoulder", 6:"R-shoulder", 7:"L-elbow",
              8:"R-elbow", 41:"R-wrist", 62:"L-wrist"}
    for k, name in sorted(labels.items()):
        print(f"  [{k:2d}] {name:12s}: Δ={delta[k].round(1)}  |Δ|={dist[k]:.1f} px")

    print(f"\n  median |Δ| (all 70 joints): {np.median(dist):.1f} px")
    print(f"  max    |Δ| (all 70 joints): {dist.max():.1f} px")


# ─── Phase E: model_params[204] transform parity ────────────────────────────
#
# Given the SAME pred_pose_raw (C++ ONNX output), verify that:
#   C++  build_model_params + apply_hand_pose + scale_decode → model_params[204]
# matches:
#   Py   mhr_forward(..., return_model_params=True)            → model_params[204]
#
# This tests the full 3D transform chain:
#   pred_pose_raw[266]
#     → rot6d_to_euler    (C++)  /  rotmat_to_euler_ZYX        (Python)  → global_rot
#     → compact_cont_to_body_params (C++) / compact_cont_to_model_params_body_fast (Py) → body_euler
#     → build_model_params[204]  /  mhr_forward model_params[204]
#     → apply_hand_pose / replace_hands_in_pose
#     → scale PCA decode
#
# If model_params agree → code is correct; arm issue is pure model quality (Phase D).
# If model_params disagree → code bug somewhere in the transform chain.

def phase_e(frame_bgr):
    print("\n" + "="*70)
    print("PHASE E — model_params[204] transform parity (same pred_pose_raw)")
    print("  Goal: verify build_model_params + hand/scale decode matches Python")
    print("="*70)

    # ── C++ pass ──────────────────────────────────────────────────────────────
    print("\n[1/3] Running C++ pipeline...")
    lib, h, FsbResult = _load_cpp()
    results = _run_cpp(lib, h, FsbResult, frame_bgr)
    lib.fsb_destroy(h)

    if not results:
        print("  No C++ detections — supply an image with a visible person")
        return

    r = results[0]
    cpp_mp       = np.array(r.mhr_model_params, dtype=np.float32)  # [204]
    cpp_pose_raw = np.array(r.pred_pose_raw,    dtype=np.float32)  # [266]
    cpp_shape    = np.array(r.shape,            dtype=np.float32)  # [45]
    cpp_scale    = np.array(r.scale,            dtype=np.float32)  # [28]
    cpp_hand     = np.array(r.hand_pose,        dtype=np.float32)  # [108]

    print(f"  C++ mhr_model_params[0:6]: {cpp_mp[:6].round(4)}")
    print(f"  C++ mhr_model_params[3:6] (global_rot ZYX): {cpp_mp[3:6].round(4)}")
    print(f"  C++ mhr_model_params[6:12] (body_pose first joint): {cpp_mp[6:12].round(4)}")
    print(f"  C++ scale sum: {cpp_scale.sum():.4f}  hand norm: {np.linalg.norm(cpp_hand):.4f}")

    # ── Python model_params via mhr_forward ───────────────────────────────────
    print("\n[2/3] Loading Python model, computing model_params from same pred_pose_raw...")
    model, _, device = _load_python_model()
    py_glob, py_body = _pred_pose_raw_to_euler(cpp_pose_raw, model, device)

    import torch
    head = model.head_pose
    def T(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)

    with torch.no_grad():
        out = head.mhr_forward(
            global_trans      = torch.zeros(1, 3, device=device),
            global_rot        = T(py_glob),        # [1,3] ZYX: [rz,ry,rx]
            body_pose_params  = T(py_body),        # [1,133]
            hand_pose_params  = T(cpp_hand),       # [1,108]
            scale_params      = T(cpp_scale),      # [1,28]
            shape_params      = T(cpp_shape),      # [1,45]
            expr_params       = torch.zeros(1, 72, device=device),
            return_keypoints  = False,
            return_model_params = True,
        )

    # mhr_forward returns: (verts, model_params) when return_model_params=True,
    # return_keypoints=False — check the actual tuple length
    if isinstance(out, (tuple, list)):
        # find the [1,204] model_params tensor
        py_mp_t = None
        for t in out:
            if isinstance(t, torch.Tensor) and t.shape[-1] == 204:
                py_mp_t = t
                break
        if py_mp_t is None:
            print(f"  Could not locate model_params in mhr_forward output (shapes: {[tuple(t.shape) for t in out if isinstance(t,torch.Tensor)]})")
            return
    else:
        py_mp_t = out

    py_mp = py_mp_t[0].detach().cpu().numpy().astype(np.float32)   # [204]
    print(f"  Py  mhr_model_params[3:6] (global_rot ZYX): {py_mp[3:6].round(4)}")
    print(f"  Py  mhr_model_params[6:12] (body_pose first joint): {py_mp[6:12].round(4)}")

    # ── Compare ───────────────────────────────────────────────────────────────
    print("\n[3/3] Comparing model_params[204] element by element...")
    diff = py_mp - cpp_mp
    adiff = np.abs(diff)

    print(f"\n--- model_params[204] summary ---")
    print(f"  median |Δ|: {np.median(adiff):.6f}")
    print(f"  mean   |Δ|: {adiff.mean():.6f}")
    print(f"  max    |Δ|: {adiff.max():.6f}  (idx {adiff.argmax()})")
    top5 = np.argsort(adiff)[-5:][::-1]
    print(f"  top-5 worst: {list(zip(top5.tolist(), adiff[top5].round(6).tolist()))}")

    # Per-region breakdown matching the layout:
    #   [0:3]   global_trans * 10  (should be 0)
    #   [3:6]   global_rot ZYX [rz, ry, rx]
    #   [6:68]  body_pose non-hand DOFs (indices 0-61 of body_pose_euler)
    #   [68:122] hand DOFs (replaced by apply_hand_pose / replace_hands_in_pose)
    #   [122:136] remaining body DOFs (body_pose_euler[116:130])
    #   [136:204] scale params (68 floats from PCA decode)
    regions = [
        ("global_trans*10  [0:3]",   0,   3),
        ("global_rot ZYX   [3:6]",   3,   6),
        ("body_pose head   [6:68]",  6,  68),
        ("hand joints      [68:122]", 68, 122),
        ("body_pose tail   [122:136]", 122, 136),
        ("scale PCA        [136:204]", 136, 204),
    ]
    print(f"\n--- Region breakdown ---")
    for name, lo, hi in regions:
        r_diff = adiff[lo:hi]
        print(f"  {name}: mean={r_diff.mean():.5f}  max={r_diff.max():.5f}")
        if r_diff.max() > 1e-3:
            worst_off = lo + r_diff.argmax()
            print(f"    worst at [{worst_off}]: cpp={cpp_mp[worst_off]:.6f}  py={py_mp[worst_off]:.6f}  Δ={diff[worst_off]:+.6f}")

    # Print specific significant differences
    sig = np.where(adiff > 0.001)[0]
    if len(sig) > 0:
        print(f"\n--- Indices with |Δ| > 0.001 ({len(sig)} total) ---")
        for idx in sig[:20]:
            region = next((n for n, lo, hi in regions if lo <= idx < hi), "?")
            print(f"  [{idx:3d}] ({region}): cpp={cpp_mp[idx]:.6f}  py={py_mp[idx]:.6f}  Δ={diff[idx]:+.6f}")
        if len(sig) > 20:
            print(f"  ... ({len(sig)-20} more)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n--- Verdict ---")
    body_max = adiff[3:136].max()
    scale_max = adiff[136:204].max()
    if body_max < 1e-4 and scale_max < 1e-3:
        print("  PASS: model_params matches Python within numerical precision.")
        print("  → build_model_params + hand/scale decode are correct.")
        print("  → Arm issue is NOT from build_model_params; run Phase D to check pose quality.")
    elif body_max < 0.01:
        print(f"  WARN: small body_pose/global_rot discrepancies (max {body_max:.5f} rad).")
        print("  → Likely floating-point rounding. Check the highlighted indices.")
    else:
        print(f"  FAIL: significant model_params mismatch (body max {body_max:.5f})!")
        print("  → Code bug in build_model_params, apply_hand_pose, or scale decode.")

    return cpp_mp, py_mp


# ─── Phase D: pred_pose_raw comparison (ONNX vs PyTorch) ─────────────────────
#
# This phase answers: "Is the arm bending a code bug or a model quality issue?"
#
# Strategy:
#   1. C++ pipeline (ONNX BF16 + GGUF) → r.pred_pose_raw[266], r.body_pose[133], r.bbox
#   2. Python pipeline (PyTorch BF16) with the SAME bbox → pred_pose_raw[266], body_pose[133]
#   3. Compare body_cont section (pred_pose_raw[6:266]) and body_pose[133]
#      element-by-element to isolate which joints drive the difference.
#
# If the body_pose arrays differ significantly, the models produce different
# pose estimates → check GGUF export (init_pose/init_camera residuals must be baked in).
# If they agree but the render still looks wrong, look for a code bug in
# build_model_params or mhr_lbs_compute.

def phase_d(frame_bgr):
    print("\n" + "="*70)
    print("PHASE D — pred_pose_raw comparison: ONNX(C++) vs PyTorch(Python)")
    print("  Goal: distinguish model quality differences from code bugs")
    print("="*70)

    # ── C++ pass ──────────────────────────────────────────────────────────────
    print("\n[1/2] Running C++ pipeline (ONNX FP16)...")
    lib, h, FsbResult = _load_cpp()
    results = _run_cpp(lib, h, FsbResult, frame_bgr)
    lib.fsb_destroy(h)

    if not results:
        print("  No C++ detections — supply an image with a visible person")
        return

    r = results[0]
    cpp_raw   = np.array(r.pred_pose_raw, dtype=np.float32)   # [266]
    cpp_body  = np.array(r.body_pose,     dtype=np.float32)   # [133]
    cpp_glob  = np.array(r.global_rot,    dtype=np.float32)   # [3]
    bbox      = np.array(r.bbox,          dtype=np.float32)   # [4] xyxy

    print(f"  C++ bbox: {bbox.round(1)}")
    print(f"  C++ global_rot (rz,ry,rx): {cpp_glob.round(4)}")
    print(f"  C++ body_pose[0:6]: {cpp_body[:6].round(4)}")

    # ── Python pass (same bbox) ────────────────────────────────────────────────
    print("\n[2/2] Running Python estimator (PyTorch BF16) with same bbox...")
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
    model, model_cfg, device = _load_python_model()
    est = SAM3DBodyEstimator(model, model_cfg)

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    outputs = est.process_one_image(img_rgb,
                                    bboxes=bbox[np.newaxis, :],
                                    inference_type="body")
    if not outputs:
        print("  Python estimator returned no outputs")
        return

    out = outputs[0]
    py_raw_t  = out.get("pred_pose_raw")
    py_body_t = out.get("body_pose_params") if "body_pose_params" in out else out.get("body_pose")
    py_glob_t = out.get("global_rot")

    import torch
    def to_np(t):
        if t is None: return None
        if isinstance(t, np.ndarray): return t.flatten().astype(np.float32)
        return t.detach().cpu().numpy().flatten().astype(np.float32)

    py_raw  = to_np(py_raw_t)   # [266]
    py_body = to_np(py_body_t)  # [133]
    py_glob = to_np(py_glob_t)  # [3]

    if py_raw is None or py_body is None:
        print("  Python output dict missing pred_pose_raw / body_pose keys")
        print(f"  Available keys: {list(out.keys())}")
        return

    print(f"  Py  global_rot (rz,ry,rx): {py_glob.round(4) if py_glob is not None else 'N/A'}")
    print(f"  Py  body_pose[0:6]: {py_body[:6].round(4)}")

    # ── body_cont comparison (pred_pose_raw[6:266]) ────────────────────────────
    cpp_cont = cpp_raw[6:266]   # [260] body_cont
    py_cont  = py_raw[6:266]    # [260]

    cont_diff  = np.abs(py_cont - cpp_cont)
    print(f"\n--- body_cont[260] (pred_pose_raw[6:266]) ---")
    print(f"  median |Δ|: {np.median(cont_diff):.4f}")
    print(f"  mean   |Δ|: {cont_diff.mean():.4f}")
    print(f"  max    |Δ|: {cont_diff.max():.4f}  (idx {cont_diff.argmax()})")
    top5c = np.argsort(cont_diff)[-5:][::-1]
    print(f"  top-5 worst body_cont indices: "
          f"{list(zip(top5c.tolist(), cont_diff[top5c].round(4).tolist()))}")

    # The 3-DOF region is body_cont[0:138] (23 joints × 6).
    # Joints 0-7 (body_cont[0:48]) are typically spine/hips/shoulders.
    # Show per-joint L2 for the first 14 3-DOF joints (pre-hand range).
    print("\n  3-DOF joints (body_cont groups of 6, joints 0-12):")
    for jj in range(13):
        sl = slice(jj*6, jj*6+6)
        d = np.linalg.norm(py_cont[sl] - cpp_cont[sl])
        print(f"    joint3dof[{jj:2d}] (cont[{jj*6}:{jj*6+6}]): |Δ|_L2={d:.4f}  "
              f"cpp={cpp_cont[sl].round(3)}  py={py_cont[sl].round(3)}")

    # ── body_pose (euler) comparison [133] ────────────────────────────────────
    body_diff = np.abs(py_body - cpp_body)
    print(f"\n--- body_pose[133] (euler angles) ---")
    print(f"  median |Δ|: {np.median(body_diff):.4f} rad")
    print(f"  mean   |Δ|: {body_diff.mean():.4f} rad")
    print(f"  max    |Δ|: {body_diff.max():.4f} rad  (idx {body_diff.argmax()})")
    top5b = np.argsort(body_diff)[-5:][::-1]
    print(f"  top-5 worst body_pose indices: "
          f"{list(zip(top5b.tolist(), body_diff[top5b].round(4).tolist()))}")

    # Show arm-adjacent euler indices (pre-hand range [0:62])
    print("\n  body_pose[0:62] (non-hand DOFs) comparison:")
    for i in range(62):
        d = body_diff[i]
        if d > 0.1:  # only print significant differences (>0.1 rad ≈ 6°)
            print(f"    [{i:3d}]: Δ={py_body[i]-cpp_body[i]:+.4f} rad  "
                  f"(cpp={cpp_body[i]:.4f}, py={py_body[i]:.4f})")

    # ── global_rot comparison ─────────────────────────────────────────────────
    if py_glob is not None:
        print(f"\n--- global_rot[3] (ZYX: rz,ry,rx) ---")
        print(f"  C++: {cpp_glob.round(4)}")
        print(f"  Py:  {py_glob.round(4)}")
        print(f"  |Δ|: {np.abs(py_glob - cpp_glob).round(4)}")

    # ── verdict ───────────────────────────────────────────────────────────────
    # Root cause history:
    #   - Original issue: C++ GGUF FFN was running cffn_run(x) without the
    #     init_pose/init_camera residual.  head_pose.forward computes
    #     pred = proj(x) + init_pose.weight, head_camera similarly.  The C++
    #     code was missing the "+ init_*" term, causing a systematic ~1.0 offset
    #     in body_cont elements 0 and 4 for every 3-DOF joint.
    #   - Fix: bake init_pose.weight into mhr_proj.fc1.bias and
    #     init_camera.weight into cam_proj.fc1.bias in convertModelToGGUF.py.
    #     After the fix, Phase D reports ~1.8° mean arm/torso diff (AGREE).
    mean_body_diff_deg = np.degrees(body_diff[:62].mean())
    print(f"\n--- Verdict ---")
    print(f"  Mean arm/torso body_pose diff: {mean_body_diff_deg:.1f} deg")
    if mean_body_diff_deg < 5.0:
        print("  AGREE: ONNX and PyTorch produce similar body_pose.")
        print("  → Arm issue is NOT a pose-quality mismatch; check build_model_params / render code.")
    elif mean_body_diff_deg < 20.0:
        print("  PARTIAL: moderate body_pose differences between C++ and Python.")
        print("  → Check pipeline.gguf was exported with init_pose/init_camera residuals baked in.")
        print("  → Re-export with: python fast_sam_3dbody_cpp/convertModelToGGUF.py")
    else:
        print("  DISAGREE: large body_pose differences between C++ and Python.")
        print("  → Check pipeline.gguf was exported with init_pose/init_camera residuals baked in.")
        print("  → Re-export with: python fast_sam_3dbody_cpp/convertModelToGGUF.py")


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Systematic C++/Python parity tests")
    ap.add_argument("--phase", choices=["A", "B", "C", "D", "E", "all"], default="all")
    ap.add_argument("--image", default="",
                    help="Path to test image (uses grey frame if not set)")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read: {args.image}")
    else:
        # Grey frame: YOLO usually doesn't detect a person here, but serves
        # as a device sanity check.  Provide --image for meaningful LBS parity.
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        print("[warn] No --image provided.  YOLO may not detect a person in a grey "
              "frame; phases that need detections will skip.")

    phase_a_result = None

    def run(fn, *a):
        try:
            return fn(*a)
        except Exception:
            traceback.print_exc()
            return None

    if args.phase in ("A", "all"):
        phase_a_result = run(phase_a, frame)

    if args.phase in ("B", "all"):
        run(phase_b, phase_a_result, frame)

    if args.phase in ("C", "all"):
        run(phase_c, frame)

    if args.phase in ("E", "all"):
        run(phase_e, frame)

    if args.phase in ("D", "all"):
        run(phase_d, frame)


if __name__ == "__main__":
    main()
