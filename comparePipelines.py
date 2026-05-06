#!/usr/bin/env python3
"""
comparePipelines.py
Compares MHR -> 3D vertex transformation between the Python reference pipeline
and the C++ backend (via ctypes).

Skips all neural network inference (YOLO, backbone, decoder, FFN heads).
Starts from MHR pose parameters and verifies both pipelines produce identical vertices.

Usage:
  # Full run (extract + compare + C++ subprocess)
  python comparePipelines.py --image notebook/images/dancing.jpg \\
      --checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt \\
      --mhr-model checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \\
      --detector yolo --detector-path checkpoints/yolo \\
      --save-params mhr_params_dancing.npz

  # From saved params (fast -- skips Python pipeline)
  python comparePipelines.py --load-params mhr_params_dancing.npz \\
      --checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt \\
      --mhr-model checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \\
      --image notebook/images/dancing.jpg

  # Skip C++ entirely (just verify Python body model)
  python comparePipelines.py --load-params mhr_params_dancing.npz \\
      --checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt \\
      --mhr-model checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \\
      --skip-cpp-pipeline
"""

# Must be set before any OpenGL import
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import ctypes
import json
import sys
import time

import numpy as np

# Add repo root so sam_3d_body package is importable
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)


# ────────────────────────────────────────────────────────────────────────────────
# ctypes structs (must match fast_sam_3dbody_capi.h exactly)
# ────────────────────────────────────────────────────────────────────────────────

class FsbConfig(ctypes.Structure):
    _fields_ = [
        ("onnx_dir",        ctypes.c_char_p),
        ("gguf_path",       ctypes.c_char_p),
        ("yolo_path",       ctypes.c_char_p),
        ("cuda_device",     ctypes.c_int),
        ("skip_body_model", ctypes.c_int),
        ("person_thresh",   ctypes.c_float),
        ("person_nms_iou",  ctypes.c_float),
        ("max_persons",     ctypes.c_int),
        ("focal_x",         ctypes.c_float),
        ("focal_y",         ctypes.c_float),
        ("principal_x",     ctypes.c_float),
        ("principal_y",     ctypes.c_float),
    ]


class FsbResult(ctypes.Structure):
    _fields_ = [
        ("bbox",         ctypes.c_float * 4),
        ("focal_length", ctypes.c_float),
        ("pred_cam_t",   ctypes.c_float * 3),
        ("global_rot",   ctypes.c_float * 3),
        ("body_pose",    ctypes.c_float * 133),
        ("shape",        ctypes.c_float * 45),
        ("scale",        ctypes.c_float * 28),
        ("hand_pose",    ctypes.c_float * 108),
        ("face_params",  ctypes.c_float * 72),
        ("yolo_kps",     ctypes.c_float * 51),
        ("has_yolo_kps", ctypes.c_int),
        ("kps_3d",       ctypes.c_float * 210),
        ("kps_2d",       ctypes.c_float * 140),
        ("has_kps",      ctypes.c_int),
    ]


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def load_library(lib_dir):
    """Load libfast_sam_3dbody.so via ctypes."""
    lib_path = os.path.join(lib_dir, "libfast_sam_3dbody.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")

    prev = os.environ.get("LD_LIBRARY_PATH", "")
    ort_lib = os.path.join(lib_dir, "onnxruntime_dl", "lib")
    os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, [lib_dir, ort_lib, prev]))

    lib = ctypes.CDLL(lib_path)
    lib.fsb_create.restype  = ctypes.c_void_p
    lib.fsb_create.argtypes = []
    lib.fsb_destroy.restype  = None
    lib.fsb_destroy.argtypes = [ctypes.c_void_p]
    lib.fsb_load.restype  = ctypes.c_int
    lib.fsb_load.argtypes = [ctypes.c_void_p, ctypes.POINTER(FsbConfig)]
    lib.fsb_process_bgr.restype  = ctypes.c_int
    lib.fsb_process_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(FsbResult),
        ctypes.c_int,
    ]
    return lib


def print_separator(char="=", width=72):
    print(char * width)


def print_section(title):
    print()
    print_separator("=", 72)
    print(f"  {title}")
    print_separator("=", 72)


def compare_arrays(name, a, b, atol=1e-5):
    """Compare two numpy arrays, print max diff and pass/fail."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  FAIL  {name}: shape mismatch {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a - b)
    max_diff = float(diff.max())
    idx = np.unravel_index(diff.argmax(), a.shape)
    ok = max_diff <= atol
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name:30s}  shape={str(a.shape):15s}  max_diff={max_diff:.2e}"
          f"  (atol={atol:.0e})")
    if not ok:
        print(f"         max diff at index {idx}, a={a[*idx]:.6f}, b={b[*idx]:.6f}")
    return ok


# ────────────────────────────────────────────────────────────────────────────────
# Phase A: Extract MHR parameters from the Python pipeline
# ────────────────────────────────────────────────────────────────────────────────

def extract_python_params(image_path, checkpoint_path, mhr_model_path,
                          detector_name, detector_path, device="cuda",
                          person_idx=0):
    """
    Run the full Python pipeline on an image and extract MHR parameters.
    Returns a dict of numpy arrays.
    """
    import torch
    import cv2
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    print_section("Phase A: Extracting MHR params from Python pipeline")
    print(f"  Image:       {image_path}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  MHR model:   {mhr_model_path}")
    print(f"  Detector:    {detector_name} ({detector_path})")
    print(f"  Device:      {device}")

    t0 = time.perf_counter()
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_model_path,
        device=device,
    )
    model.eval()
    print(f"  Model loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    from tools.build_detector import HumanDetector
    detector = HumanDetector(name=detector_name, device=device, path=detector_path)
    print(f"  Detector loaded")

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=detector,
    )

    print(f"\n  Running inference on {image_path} ...")
    t0 = time.perf_counter()
    outputs = estimator.process_one_image(image_path)
    print(f"  Inference took {(time.perf_counter()-t0)*1000:.0f} ms")

    if len(outputs) == 0:
        raise RuntimeError("No persons detected in the image")
    print(f"  Detected {len(outputs)} person(s), using person #{person_idx}")

    out = outputs[person_idx]

    params = {}
    params["global_rot"]        = np.asarray(out["global_rot"], dtype=np.float32)
    params["body_pose"]         = np.asarray(out["body_pose_params"], dtype=np.float32)
    params["shape"]             = np.asarray(out["shape_params"], dtype=np.float32)
    params["scale"]             = np.asarray(out["scale_params"], dtype=np.float32)
    params["hand_pose"]         = np.asarray(out["hand_pose_params"], dtype=np.float32)
    params["face_params"]       = np.asarray(out["expr_params"], dtype=np.float32)
    params["pred_cam_t"]        = np.asarray(out["pred_cam_t"], dtype=np.float32)
    params["focal_length"]      = float(out["focal_length"])
    params["bbox"]              = np.asarray(out["bbox"], dtype=np.float32)
    params["pred_vertices"]     = np.asarray(out["pred_vertices"], dtype=np.float32)
    params["pred_keypoints_3d"] = np.asarray(out["pred_keypoints_3d"], dtype=np.float32)
    params["pred_keypoints_2d"] = np.asarray(out["pred_keypoints_2d"], dtype=np.float32)

    assert params["global_rot"].shape == (3,)
    assert params["body_pose"].shape == (133,)
    assert params["shape"].shape == (45,)
    assert params["scale"].shape == (28,)
    assert params["hand_pose"].shape == (108,)
    assert params["face_params"].shape == (72,)
    assert params["pred_cam_t"].shape == (3,)
    assert params["pred_vertices"].shape == (18439, 3)
    assert params["pred_keypoints_3d"].shape == (70, 3)
    assert params["pred_keypoints_2d"].shape == (70, 2)

    img = cv2.imread(image_path)
    params["image_h"] = img.shape[0]
    params["image_w"] = img.shape[1]

    # Assemble the full 204-vector that mhr_lbs_compute receives.
    # Mirrors Python mhr_forward: [zeros(3), global_rot, body_pose[:130], scales]
    # where scales = scale_mean + scale_params @ scale_comps  [68 values]
    # Hand joints must be PCA-decoded (mirrors replace_hands_in_pose + apply_hand_pose).
    import torch
    head = model.head_pose
    dev  = next(head.parameters()).device
    scale_t = torch.tensor(params["scale"][None], dtype=torch.float32, device=dev)
    scales  = (head.scale_mean[None, :] + scale_t @ head.scale_comps).detach().cpu().numpy()[0]
    mp = np.zeros(204, dtype=np.float32)
    mp[3:6]    = params["global_rot"]
    mp[6:136]  = params["body_pose"][:130]
    mp[136:204] = scales

    # Overwrite hand joint positions with PCA-decoded Euler angles.
    # Python mhr_forward replaces body_pose hand slots via replace_hands_in_pose;
    # C++ apply_hand_pose does the same. Without this, the NPZ hand joints are
    # the raw decoder output instead of the PCA-decoded values mhr_forward uses.
    hand_pose_mean  = head.hand_pose_mean.detach().cpu().numpy()       # [54]
    hand_pose_comps = head.hand_pose_comps.detach().cpu().numpy()      # [54×54]
    hand_idxs_left  = head.hand_joint_idxs_left.detach().cpu().numpy().astype(int)   # [27]
    hand_idxs_right = head.hand_joint_idxs_right.detach().cpu().numpy().astype(int)  # [27]
    hand_params     = params["hand_pose"]  # [108]: [:54] left, [54:] right

    from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_hand

    def _decode_hand(h54, idxs27):
        decoded = hand_pose_mean + h54 @ hand_pose_comps          # [54]
        decoded_t = torch.tensor(decoded, dtype=torch.float32)
        euler27 = compact_cont_to_model_params_hand(decoded_t).numpy()  # [27]
        for i, idx in enumerate(idxs27):
            if 0 <= idx < 136:
                mp[idx] = euler27[i]

    _decode_hand(hand_params[:54], hand_idxs_left)
    _decode_hand(hand_params[54:], hand_idxs_right)

    params["mhr_model_params"] = mp

    print(f"\n  Extracted parameter shapes:")
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            print(f"    {k:25s} {v.shape}  min={v.min():.4f}  max={v.max():.4f}")
        else:
            print(f"    {k:25s} {v}")

    return params, model, device


# ────────────────────────────────────────────────────────────────────────────────
# Phase B: Run Python MHR body model on the extracted params
# ────────────────────────────────────────────────────────────────────────────────

def run_python_body_model(model, params, device="cuda"):
    """
    Run model.head_pose.mhr_forward() on the given params.
    This is the PURE Python reference path: MHR params -> vertices.
    """
    import torch

    print_section("Phase B: Python MHR body model forward pass")

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

    global_rot  = _t(params["global_rot"])
    body_pose   = _t(params["body_pose"])
    shape       = _t(params["shape"])
    scale       = _t(params["scale"])
    hand_pose   = _t(params["hand_pose"])
    face_params = _t(params["face_params"])

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.head_pose.mhr_forward(
            global_trans=torch.zeros(1, 3, device=device),
            global_rot=global_rot,
            body_pose_params=body_pose,
            hand_pose_params=hand_pose,
            scale_params=scale,
            shape_params=shape,
            expr_params=face_params,
            return_keypoints=True,
        )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  mhr_forward took {elapsed:.1f} ms")

    if isinstance(out, tuple):
        verts, j3d = out[0], out[1]
    else:
        verts, j3d = out, None

    verts = verts.clone()
    verts[..., [1, 2]] *= -1

    if j3d is not None:
        j3d = j3d[:, :70].clone()
        j3d[..., [1, 2]] *= -1

    verts_np = verts[0].cpu().float().numpy()
    j3d_np   = j3d[0].cpu().float().numpy() if j3d is not None else None

    print(f"  vertices shape: {verts_np.shape}")
    if j3d_np is not None:
        print(f"  keypoints_3d shape: {j3d_np.shape}")

    return verts_np, j3d_np


# ────────────────────────────────────────────────────────────────────────────────
# Phase C: Populate FsbResult + run C++ frontend body model path
# ────────────────────────────────────────────────────────────────────────────────

def fsb_result_from_params(params):
    """
    Populate a ctypes FsbResult struct from a params dict.
    This simulates what the C++ pipeline would produce.
    """
    result = FsbResult()
    result.focal_length = float(params["focal_length"])

    for i in range(4):
        result.bbox[i] = float(params["bbox"][i])
    for i in range(3):
        result.pred_cam_t[i] = float(params["pred_cam_t"][i])
        result.global_rot[i] = float(params["global_rot"][i])

    for i in range(133):
        result.body_pose[i] = float(params["body_pose"][i])
    for i in range(45):
        result.shape[i] = float(params["shape"][i])
    for i in range(28):
        result.scale[i] = float(params["scale"][i])
    for i in range(108):
        result.hand_pose[i] = float(params["hand_pose"][i])
    for i in range(72):
        result.face_params[i] = float(params["face_params"][i])

    result.has_yolo_kps = 0
    result.has_kps = 0

    return result


def run_cpp_frontend_body_model(model, result, device="cuda",
                                 image_h=480, image_w=640,
                                 principal_x=0.0, principal_y=0.0):
    """
    Replicates fsb_result_to_output() from fast_sam_3dbody_frontend-3D.py.
    Takes a ctypes FsbResult, extracts params, runs Python MHR body model.
    """
    import torch

    print_section("Phase C: C++ frontend -> Python body model path")

    def _t(arr, n):
        return torch.tensor(list(arr)[:n], dtype=torch.float32, device=device).unsqueeze(0)

    global_rot  = _t(result.global_rot,  3)
    body_pose   = _t(result.body_pose,  133)
    shape       = _t(result.shape,       45)
    scale       = _t(result.scale,       28)
    hand_pose   = _t(result.hand_pose,  108)
    face_params = _t(result.face_params, 72)

    print(f"  global_rot from FsbResult: {global_rot[0].cpu().numpy()}")
    print(f"  body_pose[0:6] from FsbResult: {body_pose[0, :6].cpu().numpy()}")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.head_pose.mhr_forward(
            global_trans=torch.zeros(1, 3, device=device),
            global_rot=global_rot,
            body_pose_params=body_pose,
            hand_pose_params=hand_pose,
            scale_params=scale,
            shape_params=shape,
            expr_params=face_params,
            return_keypoints=True,
        )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  mhr_forward took {elapsed:.1f} ms")

    if isinstance(out, tuple):
        verts, j3d = out[0], out[1]
    else:
        verts, j3d = out, None

    verts = verts.clone()
    verts[..., [1, 2]] *= -1

    if j3d is not None:
        j3d = j3d[:, :70].clone()
        j3d[..., [1, 2]] *= -1

    verts_np = verts[0].cpu().float().numpy()

    if j3d is not None:
        j3d_np = j3d[0].cpu().float().numpy()
        pred_cam_t = np.array(list(result.pred_cam_t[:3]))
        fl = float(result.focal_length)
        px = principal_x if principal_x > 0.0 else image_w * 0.5
        py = principal_y if principal_y > 0.0 else image_h * 0.5

        j3d_cam = j3d_np + pred_cam_t
        dz = np.maximum(j3d_cam[:, 2:3], 1e-4)
        j2d = j3d_cam[:, :2] / dz * fl + np.array([px, py])
    else:
        j3d_np = None
        j2d = np.zeros((70, 2), dtype=np.float32)

    pred_cam_t = np.array(list(result.pred_cam_t[:3]))
    bbox = np.array(list(result.bbox[:4]))

    print(f"  vertices shape: {verts_np.shape}")
    print(f"  keypoints_3d shape: {j3d_np.shape if j3d_np is not None else 'None'}")
    print(f"  keypoints_2d shape: {j2d.shape}")

    return {
        "bbox": bbox,
        "focal_length": float(result.focal_length),
        "pred_cam_t": pred_cam_t,
        "pred_vertices": verts_np,
        "pred_keypoints_3d": j3d_np,
        "pred_keypoints_2d": j2d,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Phase D: Full C++ pipeline roundtrip (for comparison)
# ────────────────────────────────────────────────────────────────────────────────

def run_cpp_full_pipeline(lib_dir, image_path, onnx_dir, cuda_device=0,
                           person_thresh=0.5, person_nms_iou=0.45,
                           max_persons=0, fx=0.0, fy=0.0, cx=0.0, cy=0.0,
                           skip_body_model=False):
    """
    Run the C++ pipeline on an image in a subprocess.
    Returns a list of dicts, one per detected person.

    Runs in a subprocess to avoid CUDA context conflicts with PyTorch.

    If skip_body_model=True, only the FFN head output is available
    (no vertices/keypoints). This avoids the LBS code path.
    """
    label = "C++ pipeline (params only)" if skip_body_model else "C++ pipeline (full)"
    print_section(f"Phase D: {label} (subprocess)")

    output_npz = "/tmp/cpp_pipeline_results.npz"

    script = f'''
import ctypes
import os
import sys
import numpy as np

os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + \\
    "{lib_dir}" + ":" + os.path.join("{lib_dir}", "onnxruntime_dl", "lib")

lib_path = os.path.join("{lib_dir}", "libfast_sam_3dbody.so")
if not os.path.exists(lib_path):
    print(f"Library not found: {{lib_path}}")
    sys.exit(1)

class FsbConfig(ctypes.Structure):
    _fields_ = [
        ("onnx_dir",        ctypes.c_char_p),
        ("gguf_path",       ctypes.c_char_p),
        ("yolo_path",       ctypes.c_char_p),
        ("cuda_device",     ctypes.c_int),
        ("skip_body_model", ctypes.c_int),
        ("person_thresh",   ctypes.c_float),
        ("person_nms_iou",  ctypes.c_float),
        ("max_persons",     ctypes.c_int),
        ("focal_x",         ctypes.c_float),
        ("focal_y",         ctypes.c_float),
        ("principal_x",     ctypes.c_float),
        ("principal_y",     ctypes.c_float),
    ]

class FsbResult(ctypes.Structure):
    _fields_ = [
        ("bbox",         ctypes.c_float * 4),
        ("focal_length", ctypes.c_float),
        ("pred_cam_t",   ctypes.c_float * 3),
        ("global_rot",   ctypes.c_float * 3),
        ("body_pose",    ctypes.c_float * 133),
        ("shape",        ctypes.c_float * 45),
        ("scale",        ctypes.c_float * 28),
        ("hand_pose",    ctypes.c_float * 108),
        ("face_params",  ctypes.c_float * 72),
        ("yolo_kps",     ctypes.c_float * 51),
        ("has_yolo_kps", ctypes.c_int),
        ("kps_3d",       ctypes.c_float * 210),
        ("kps_2d",       ctypes.c_float * 140),
        ("has_kps",      ctypes.c_int),
    ]

lib = ctypes.CDLL(lib_path)
lib.fsb_create.restype  = ctypes.c_void_p
lib.fsb_create.argtypes = []
lib.fsb_destroy.restype  = None
lib.fsb_destroy.argtypes = [ctypes.c_void_p]
lib.fsb_load.restype  = ctypes.c_int
lib.fsb_load.argtypes = [ctypes.c_void_p, ctypes.POINTER(FsbConfig)]
lib.fsb_process_bgr.restype  = ctypes.c_int
lib.fsb_process_bgr.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                                ctypes.c_int, ctypes.c_int, ctypes.POINTER(FsbResult), ctypes.c_int]

import cv2
onnx_dir = "{onnx_dir}"
gguf_path = os.path.join(onnx_dir, "pipeline.gguf")
yolo_path = os.path.join(onnx_dir, "yolo.onnx")
if not os.path.exists(yolo_path):
    yolo_path = "{os.path.join(_repo_root, 'checkpoints', 'yolo', 'yolo11m-pose.onnx')}"

cfg = FsbConfig(
    onnx_dir        = onnx_dir.encode(),
    gguf_path       = gguf_path.encode(),
    yolo_path       = yolo_path.encode(),
    cuda_device     = {cuda_device},
    skip_body_model = {1 if skip_body_model else 0},
    person_thresh   = {person_thresh},
    person_nms_iou  = {person_nms_iou},
    max_persons     = {max_persons},
    focal_x         = {fx},
    focal_y         = {fy},
    principal_x     = {cx},
    principal_y     = {cy},
)

handle = lib.fsb_create()
if not handle:
    print("fsb_create returned NULL")
    sys.exit(1)
if not lib.fsb_load(handle, ctypes.byref(cfg)):
    print("fsb_load failed")
    lib.fsb_destroy(handle)
    sys.exit(1)

frame = cv2.imread("{image_path}")
if frame is None:
    print(f"Cannot read image: {image_path}")
    lib.fsb_destroy(handle)
    sys.exit(1)
H, W = frame.shape[:2]
print(f"  Image: {{W}}x{{H}}")

MAX_RESULTS = 32
ResultArray = FsbResult * MAX_RESULTS
results_buf = ResultArray()
bgr_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
import time
t0 = time.perf_counter()
n = lib.fsb_process_bgr(handle, bgr_ptr, W, H, results_buf, MAX_RESULTS)
print(f"  C++ inference took {{(time.perf_counter()-t0)*1000:.0f}} ms")
print(f"  Detected {{n}} person(s)")

save = {{}}
for i in range(n):
    r = results_buf[i]
    prefix = f"p{{i}}"
    save[prefix + "_bbox"]         = np.array(list(r.bbox), dtype=np.float32)
    save[prefix + "_focal_length"] = np.array([float(r.focal_length)], dtype=np.float32)
    save[prefix + "_pred_cam_t"]   = np.array(list(r.pred_cam_t), dtype=np.float32)
    save[prefix + "_global_rot"]   = np.array(list(r.global_rot), dtype=np.float32)
    save[prefix + "_body_pose"]    = np.array(list(r.body_pose), dtype=np.float32)
    save[prefix + "_shape"]        = np.array(list(r.shape), dtype=np.float32)
    save[prefix + "_scale"]        = np.array(list(r.scale), dtype=np.float32)
    save[prefix + "_hand_pose"]    = np.array(list(r.hand_pose), dtype=np.float32)
    save[prefix + "_face_params"]  = np.array(list(r.face_params), dtype=np.float32)
    if r.has_kps:
        save[prefix + "_kps_3d"] = np.array(list(r.kps_3d), dtype=np.float32).reshape(70, 3)
        save[prefix + "_kps_2d"] = np.array(list(r.kps_2d), dtype=np.float32).reshape(70, 2)
    save["num_persons"] = np.array([n], dtype=np.int32)

np.savez("{output_npz}", **save)
lib.fsb_destroy(handle)
print(f"  Results saved")
'''

    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=False,
        timeout=120,
        cwd=_repo_root,
    )

    if result.returncode != 0:
        raise RuntimeError(f"C++ subprocess exited with code {result.returncode}")

    data = np.load(output_npz)
    n = int(data["num_persons"][0])
    print(f"  Loaded {n} result(s) from subprocess")

    results = []
    for i in range(n):
        p = f"p{i}"
        r = {
            "bbox": data[p + "_bbox"],
            "focal_length": float(data[p + "_focal_length"][0]),
            "pred_cam_t": data[p + "_pred_cam_t"],
            "global_rot": data[p + "_global_rot"],
            "body_pose": data[p + "_body_pose"],
            "shape": data[p + "_shape"],
            "scale": data[p + "_scale"],
            "hand_pose": data[p + "_hand_pose"],
            "face_params": data[p + "_face_params"],
            "kps_3d": data[p + "_kps_3d"] if (p + "_kps_3d") in data else None,
            "kps_2d": data[p + "_kps_2d"] if (p + "_kps_2d") in data else None,
        }
        results.append(r)

    return results


# ────────────────────────────────────────────────────────────────────────────────
# Phase E: Parameter-level comparison (Python vs C++ extracted params)
# ────────────────────────────────────────────────────────────────────────────────

def compare_params_python_vs_cpp(py_params, cpp_results, person_idx=0):
    """
    Compare the MHR parameters extracted from the Python pipeline
    against those from the C++ pipeline.
    """
    print_section("Phase E: Comparing MHR parameters (Python vs C++)")

    cpp = cpp_results[person_idx]
    py = py_params

    all_ok = True
    fields = [
        ("global_rot",   py["global_rot"],   cpp["global_rot"],   1e-3),
        ("body_pose",    py["body_pose"],    cpp["body_pose"],    1e-3),
        ("shape",        py["shape"],        cpp["shape"],        1e-4),
        ("scale",        py["scale"],        cpp["scale"],        1e-4),
        ("hand_pose",    py["hand_pose"],    cpp["hand_pose"],    1e-3),
        ("face_params",  py["face_params"],  cpp["face_params"],  1e-4),
        ("pred_cam_t",   py["pred_cam_t"],   cpp["pred_cam_t"],   1e-3),
        ("focal_length", np.array([py["focal_length"]]),
                         np.array([cpp["focal_length"]]), 1e-3),
        ("bbox",         py["bbox"],         cpp["bbox"],         1.0),
    ]

    for name, a, b, atol in fields:
        ok = compare_arrays(name, a, b, atol=atol)
        if not ok:
            all_ok = False

    return all_ok


# ────────────────────────────────────────────────────────────────────────────────
# Phase F: Vertex-level comparison
# ────────────────────────────────────────────────────────────────────────────────

def compare_vertices(name, a, b, atol=1e-3):
    """
    Compare vertex arrays. a and b are [N, 3] float32.
    Returns True if all vertices match within tolerance.
    """
    print_section(f"Phase F: Vertex comparison ({name})")

    if a is None or b is None:
        print(f"  FAIL  vertices: one of the arrays is None")
        return False

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        print(f"  FAIL  shape mismatch: {a.shape} vs {b.shape}")
        return False

    diff = np.abs(a - b)
    diff_per_vertex = np.linalg.norm(diff, axis=1)

    max_diff = float(diff_per_vertex.max())
    mean_diff = float(diff_per_vertex.mean())
    median_diff = float(np.median(diff_per_vertex))
    max_component = float(diff.max())
    idx = np.unravel_index(diff.argmax(), diff.shape)
    ok = max_component <= atol

    print(f"  Vertices: {a.shape[0]}")
    print(f"  Per-vertex Euclidean distance:")
    print(f"    max:    {max_diff:.6f}")
    print(f"    mean:   {mean_diff:.6f}")
    print(f"    median: {median_diff:.6f}")
    print(f"  Max component diff: {max_component:.2e}  (atol={atol:.0e})")
    print(f"    at vertex {idx[0]}, component {idx[1]}")
    print(f"    a = ({a[idx[0], 0]:.4f}, {a[idx[0], 1]:.4f}, {a[idx[0], 2]:.4f})")
    print(f"    b = ({b[idx[0], 0]:.4f}, {b[idx[0], 1]:.4f}, {b[idx[0], 2]:.4f})")

    status = "PASS" if ok else "FAIL"
    print(f"  {status}")
    return ok


# ────────────────────────────────────────────────────────────────────────────────
# Phase G: Keypoint comparison
# ────────────────────────────────────────────────────────────────────────────────

def compare_keypoints(py_kps, cpp_kps, label="3D keypoints"):
    """Compare 3D keypoint arrays."""
    print_section(f"Keypoint comparison ({label})")
    if py_kps is None or cpp_kps is None:
        print(f"  SKIP  one of the arrays is None")
        return True
    return compare_arrays(label, py_kps, cpp_kps, atol=1e-2)


# ────────────────────────────────────────────────────────────────────────────────
# Side-by-side mesh rendering
# ────────────────────────────────────────────────────────────────────────────────

def render_side_by_side(py_verts, py_cam_t, cpp_verts, cpp_cam_t, faces,
                         focal_length, image_w=2250, image_h=1500,
                         output_path="comparePipelines.jpg"):
    """
    Render two meshes side by side on black background using pyrender.

    Each panel is image_w x image_h. The output is 2*image_w x image_h.
    Left panel  = Python pipeline result  (orange mesh)
    Right panel = C++ pipeline result     (cyan mesh)
    """
    import cv2
    import pyrender
    import trimesh

    print_section("Side-by-side mesh rendering")
    print(f"  Output: {output_path}")
    print(f"  Each panel: {image_w}x{image_h}")

    def _render_mesh(verts, cam_t, base_color, label):
        """Render a single mesh on a black background."""
        black = np.zeros((image_h, image_w, 3), dtype=np.float32)

        renderer = pyrender.OffscreenRenderer(
            viewport_height=image_h,
            viewport_width=image_w,
        )

        camera_translation = cam_t.copy()
        camera_translation[0] *= -1.0

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(base_color[2], base_color[1], base_color[0], 1.0),
        )

        mesh = trimesh.Trimesh(verts.copy(), faces.copy())

        # 180-degree X rotation (matches Python pipeline convention)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0], ambient_light=(0.4, 0.4, 0.4)
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation

        camera = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=image_w / 2.0,
            cy=image_h / 2.0,
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose, name="camera")

        # Add lights
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)
            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            light_pose = np.eye(4)
            light_pose[:3, :3] = np.c_[x, y, z]
            scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                pose=light_pose,
            )

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        color = color.astype(np.float32) / 255.0

        # Composite onto black background
        alpha = color[:, :, 3:]
        result = black.copy()
        result = result * (1.0 - alpha) + color[:, :, :3] * alpha

        print(f"  {label}: rendered {verts.shape[0]} vertices")
        return result

    # Python result: warm orange
    py_panel = _render_mesh(py_verts, py_cam_t,
                            base_color=(1.0, 0.6, 0.2), label="Python pipeline")

    # C++ result: cool cyan
    cpp_panel = _render_mesh(cpp_verts, cpp_cam_t,
                             base_color=(0.2, 0.8, 1.0), label="C++ pipeline")

    # Concatenate side by side
    combined = np.concatenate([py_panel, cpp_panel], axis=1)

    # Convert to BGR for OpenCV
    combined_bgr = (combined[:, :, ::-1] * 255).astype(np.uint8)

    cv2.imwrite(output_path, combined_bgr)
    print(f"  Saved {output_path} ({combined_bgr.shape[1]}x{combined_bgr.shape[0]})")

    return combined_bgr


# ────────────────────────────────────────────────────────────────────────────────
# Save / load params
# ────────────────────────────────────────────────────────────────────────────────

def save_params(params, path):
    """Save MHR params to a .npz file."""
    np.savez(path, **{k: v for k, v in params.items() if isinstance(v, (np.ndarray, int, float))})
    print(f"  Saved params to {path}")


def load_params(path):
    """Load MHR params from a .npz file."""
    print_section(f"Loading params from {path}")
    data = np.load(path)
    params = {}
    for k in data.files:
        v = data[k]
        if v.ndim == 0:
            v = float(v)
        else:
            v = v.astype(np.float32)
        params[k] = v
    print(f"  Loaded {len(params)} arrays:")
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            print(f"    {k:25s} {v.shape}")
        else:
            print(f"    {k:25s} {v}")
    return params


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare MHR -> 3D vertex pipeline: Python vs C++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", help="Input image path (for extraction mode)")
    parser.add_argument("--checkpoint", default=os.path.join(
        _repo_root, "checkpoints", "sam-3d-body-dinov3", "model.ckpt"),
                        help="Path to SAM-3D-Body model.ckpt")
    parser.add_argument("--mhr-model", default=os.path.join(
        _repo_root, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt"),
                        help="Path to mhr_model.pt")
    parser.add_argument("--detector", default="yolo", help="Detector name")
    parser.add_argument("--detector-path", default=os.path.join(
        _repo_root, "checkpoints", "yolo"), help="Detector path")
    parser.add_argument("--device", default="cuda", help="PyTorch device")
    parser.add_argument("--person-idx", type=int, default=0, help="Person index to use")

    parser.add_argument("--load-params", help="Load saved MHR params instead of running Python pipeline")
    parser.add_argument("--save-params", help="Save extracted MHR params to .npz file")

    parser.add_argument("--lib-dir", default=os.path.join(
        _repo_root, "fast_sam_3dbody_cpp", "build"),
                        help="Directory containing libfast_sam_3dbody.so")
    parser.add_argument("--onnx-dir", default=os.path.join(
        _repo_root, "fast_sam_3dbody_cpp", "onnx"),
                        help="Directory containing backbone.onnx, decoder.onnx, pipeline.gguf")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device for C++ engine")
    parser.add_argument("--skip-cpp-pipeline", action="store_true",
                        help="Skip full C++ pipeline roundtrip (Phase D)")

    args = parser.parse_args()

    has_python_extraction = args.image is not None
    has_saved_params = args.load_params is not None

    if not has_python_extraction and not has_saved_params:
        parser.error("Either --image or --load-params is required")

    if not has_saved_params and not has_python_extraction:
        parser.error("Need params from somewhere")

    # ── Step 1: Get MHR params ──────────────────────────────────────────────
    if has_saved_params:
        py_params = load_params(args.load_params)
    elif has_python_extraction:
        py_params, model, device = extract_python_params(
            args.image, args.checkpoint, args.mhr_model,
            args.detector, args.detector_path,
            device=args.device, person_idx=args.person_idx,
        )
        if args.save_params:
            save_params(py_params, args.save_params)
    else:
        raise RuntimeError("No params available")

    # ── Step 2: Load Python model for body model forward ────────────────────
    if has_saved_params:
        import torch
        print_section("Loading Python model for body model forward")
        device = args.device
        t0 = time.perf_counter()
        from sam_3d_body import load_sam_3d_body
        model, _ = load_sam_3d_body(
            checkpoint_path=args.checkpoint,
            mhr_path=args.mhr_model,
            device=device,
        )
        model.eval()
        print(f"  Model loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    # ── Step 3: Run Python body model on extracted params ───────────────────
    py_verts, py_j3d = run_python_body_model(model, py_params, device=device)

    # Verify: the re-run should match the original vertices
    print_section("Verification: re-run vs original Python vertices")
    compare_arrays("pred_vertices", py_verts, py_params["pred_vertices"], atol=1e-4)
    compare_arrays("pred_keypoints_3d", py_j3d, py_params["pred_keypoints_3d"], atol=1e-4)

    # ── Step 4: Populate FsbResult and run via C++ frontend path ────────────
    fsb_result = fsb_result_from_params(py_params)
    image_h = int(py_params.get("image_h", 480))
    image_w = int(py_params.get("image_w", 640))

    cpp_output = run_cpp_frontend_body_model(
        model, fsb_result, device=device,
        image_h=image_h, image_w=image_w,
    )

    # ── Step 5: Compare vertices ────────────────────────────────────────────
    print()
    print_separator("#", 72)
    print("  RESULTS: Python body model vs C++ frontend -> Python body model")
    print_separator("#", 72)

    ok1 = compare_vertices(
        "Python direct vs C++ frontend path",
        py_verts, cpp_output["pred_vertices"],
        atol=1e-4,
    )

    if py_j3d is not None and cpp_output["pred_keypoints_3d"] is not None:
        compare_keypoints(py_j3d, cpp_output["pred_keypoints_3d"], "3D keypoints")

    # ── Step 6: C++ pipeline roundtrip ──────────────────────────────────────
    cpp_results = None
    cpp2_output = None
    ok2 = False
    param_ok = False

    if not args.skip_cpp_pipeline:
        print()
        print_separator("~", 72)

        # Try params-only path (skip_body_model=1) - avoids LBS code path
        try:
            cpp_results = run_cpp_full_pipeline(
                args.lib_dir, args.image, args.onnx_dir,
                cuda_device=args.cuda_device,
                skip_body_model=True,
            )
        except (RuntimeError, FileNotFoundError) as e:
            print(f"  SKIP  C++ pipeline (params): {e}")
            cpp_results = None

        if cpp_results is not None:
            # Compare params
            param_ok = compare_params_python_vs_cpp(
                py_params, cpp_results, person_idx=args.person_idx
            )

            # Now run Python body model on C++ extracted params
            cpp_fsb = fsb_result_from_params({
                "global_rot":   cpp_results[args.person_idx]["global_rot"],
                "body_pose":    cpp_results[args.person_idx]["body_pose"],
                "shape":        cpp_results[args.person_idx]["shape"],
                "scale":        cpp_results[args.person_idx]["scale"],
                "hand_pose":    cpp_results[args.person_idx]["hand_pose"],
                "face_params":  cpp_results[args.person_idx]["face_params"],
                "pred_cam_t":   cpp_results[args.person_idx]["pred_cam_t"],
                "focal_length": cpp_results[args.person_idx]["focal_length"],
                "bbox":         cpp_results[args.person_idx]["bbox"],
            })

            cpp2_output = run_cpp_frontend_body_model(
                model, cpp_fsb, device=device,
                image_h=image_h, image_w=image_w,
            )

            # Compare C++-extracted params vs Python-extracted params -> vertices
            print()
            print_separator("#", 72)
            print("  RESULTS: Python pipeline vs Full C++ pipeline (via Python body model)")
            print_separator("#", 72)

            ok2 = compare_vertices(
                "Python pipeline vs C++ pipeline (via Python body model)",
                py_verts, cpp2_output["pred_vertices"],
                atol=1e-2,
            )

            # Compare C++ native keypoints vs Python keypoints
            if cpp_results[args.person_idx]["kps_3d"] is not None:
                compare_keypoints(
                    py_j3d,
                    cpp_results[args.person_idx]["kps_3d"],
                    "C++ native 3D keypoints vs Python",
                )

            # Compare 2D keypoints
            if cpp_results[args.person_idx]["kps_2d"] is not None:
                compare_keypoints(
                    py_params["pred_keypoints_2d"],
                    cpp_results[args.person_idx]["kps_2d"],
                    "C++ native 2D keypoints vs Python",
                )

    # ── Step 7: Side-by-side mesh rendering ─────────────────────────────────
    print()
    print_separator("~", 72)

    faces = model.head_pose.faces.cpu().numpy()
    output_path = os.path.join(_repo_root, "comparePipelines.jpg")

    try:
        render_side_by_side(
            py_verts=py_verts,
            py_cam_t=py_params["pred_cam_t"],
            cpp_verts=cpp2_output["pred_vertices"] if cpp2_output is not None else py_verts,
            cpp_cam_t=cpp2_output["pred_cam_t"] if cpp2_output is not None else py_params["pred_cam_t"],
            faces=faces,
            focal_length=float(py_params["focal_length"]),
            image_w=int(py_params.get("image_w", 2250)),
            image_h=int(py_params.get("image_h", 1500)),
            output_path=output_path,
        )
    except Exception as e:
        print(f"  SKIP  rendering failed: {e}")

    # ── Final summary ───────────────────────────────────────────────────────
    print()
    print_separator("=", 72)
    print("  SUMMARY")
    print_separator("=", 72)
    print()
    print("  Test 1: Python body model consistency (re-run vs original)")
    print("    -> Verifies mhr_forward is deterministic")
    print()
    print("  Test 2: C++ FsbResult -> Python body model vs direct Python")
    print("    -> Verifies ctypes roundtrip doesn't lose precision")
    print(f"    -> {'PASS' if ok1 else 'FAIL'}")
    print()

    if not args.skip_cpp_pipeline and cpp_results is not None:
        print("  Test 3: C++ FFN params -> Python body model vs Python pipeline")
        print(f"    -> {'PASS' if ok2 else 'FAIL'}")
        print("    -> Compares vertices when C++ FFN output feeds Python MHR model")
        print()
        print("  Test 4: C++ param extraction vs Python param extraction")
        print(f"    -> {'PASS' if param_ok else 'FAIL'} (see Phase E for details)")
    elif not args.skip_cpp_pipeline:
        print("  Test 3/4: SKIPPED (C++ pipeline subprocess failed)")

    print()
    print("  Diagnosis:")
    print("    Test 2 FAIL -> ctypes FsbResult roundtrip loses precision")
    print("    Test 3 FAIL -> C++ FFN head / param decoding differs from Python")
    print("    Test 4 FAIL -> C++ and Python extract different params from same image")
    print()


if __name__ == "__main__":
    main()
