#!/usr/bin/env python3
"""
fast_sam_3dbody_frontend.py
Python frontend for the C++ SAM-3D-Body pipeline.

Loads libfast_sam_3dbody.so via ctypes, runs inference,
and draws COCO skeleton + MHR pose info on the frame.

Usage:
  python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py --from 0
  python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py --from /path/to/video.mp4
  python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py --from /path/to/image.jpg
  python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py --from 0 --max-skeletons 3
"""

import argparse
import ctypes
import os
import sys
import time

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# ctypes structs matching fast_sam_3dbody_capi.h
# ──────────────────────────────────────────────────────────────────────────────

class FsbConfig(ctypes.Structure):
    _fields_ = [
        ("onnx_dir",       ctypes.c_char_p),
        ("gguf_path",      ctypes.c_char_p),
        ("yolo_path",      ctypes.c_char_p),
        ("cuda_device",    ctypes.c_int),
        ("skip_body_model",ctypes.c_int),
        ("person_thresh",  ctypes.c_float),
        ("person_nms_iou", ctypes.c_float),
        ("max_persons",    ctypes.c_int),
        ("focal_x",        ctypes.c_float),
        ("focal_y",        ctypes.c_float),
        ("principal_x",    ctypes.c_float),
        ("principal_y",    ctypes.c_float),
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


def load_library(lib_dir: str) -> ctypes.CDLL:
    lib_path = os.path.join(lib_dir, "libfast_sam_3dbody.so")
    if not os.path.exists(lib_path):
        sys.exit(f"Library not found: {lib_path}\nBuild the project first.")

    # Add the lib directory to LD_LIBRARY_PATH so transitive .so deps are found
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


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton drawing
# ──────────────────────────────────────────────────────────────────────────────

# COCO 17-keypoint skeleton connections
COCO_EDGES = [
    (0, 1), (0, 2),           # nose → eyes
    (1, 3), (2, 4),           # eyes → ears
    (0, 5), (0, 6),           # nose → shoulders
    (5, 6),                    # shoulder bar
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # torso
    (11, 12),                  # hip bar
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

# Per-joint colour (BGR)
_KP_COLORS = [
    (255, 0,   255),   # 0  nose         – magenta
    (255, 85,  0),     # 1  left_eye
    (0,   85,  255),   # 2  right_eye
    (255, 170, 0),     # 3  left_ear
    (0,   170, 255),   # 4  right_ear
    (255, 255, 0),     # 5  left_shoulder  – yellow
    (0,   255, 255),   # 6  right_shoulder – cyan
    (255, 128, 0),     # 7  left_elbow
    (0,   128, 255),   # 8  right_elbow
    (0,   255, 0),     # 9  left_wrist     – green
    (0,   0,   255),   # 10 right_wrist    – blue
    (128, 255, 0),     # 11 left_hip
    (0,   255, 128),   # 12 right_hip
    (128, 0,   255),   # 13 left_knee
    (255, 0,   128),   # 14 right_knee
    (255, 0,   0),     # 15 left_ankle     – red
    (0,   128, 128),   # 16 right_ankle
]

# Edge colour: blend of the two endpoint colours
def _edge_color(i: int, j: int) -> tuple:
    c1, c2 = _KP_COLORS[i], _KP_COLORS[j]
    return tuple((c1[k] + c2[k]) // 2 for k in range(3))

VIS_THRESHOLD = 0.3   # minimum keypoint visibility to draw


# ──────────────────────────────────────────────────────────────────────────────
# MHR70 keypoint drawing (body + hands + extra joints)
# ──────────────────────────────────────────────────────────────────────────────

# MHR70 keypoint indices:
#  0-20: body/foot (nose, eyes, ears, shoulders, elbows, hips, knees, ankles, toes, heels)
# 21-41: right hand (thumb→pinky + wrist)
# 42-62: left hand (thumb→pinky + wrist)
# 63-69: extra (olecranon, cubital-fossa, acromion, neck)
_MHR_BODY   = range(0, 21)
_MHR_RHAND  = range(21, 42)
_MHR_LHAND  = range(42, 63)
_MHR_EXTRA  = range(63, 70)

# MHR70 skeleton edges (from mhr70.py skeleton_info)
MHR70_EDGES = [
    # Body limbs
    (13, 11), (11, 9),   # left leg
    (14, 12), (12, 10),  # right leg
    (9, 10),              # hip bar
    (5, 9),               # left torso
    (6, 10),              # right torso
    (5, 6),               # shoulder bar
    (5, 7),               # left arm
    (6, 8),               # right arm
    # Head
    (1, 2),               # eyes
    (0, 1), (0, 2),       # nose → eyes
    (1, 3), (2, 4),       # eyes → ears
    (3, 5), (4, 6),       # ears → shoulders
    # Feet
    (13, 15), (13, 16), (13, 17),  # left foot
    (14, 18), (14, 19), (14, 20),  # right foot
    # Left hand: wrist(62) → fingers
    (62, 45), (45, 44), (44, 43), (43, 42),  # thumb
    (62, 48), (48, 47), (47, 46),            # index
    (62, 51), (51, 50), (50, 49),            # middle (note: index order from mhr70)
    (62, 54), (54, 53), (53, 52),            # ring
    (62, 57), (57, 56), (56, 55),            # pinky
    # Right hand: wrist(41) → fingers
    (41, 24), (24, 23), (23, 22), (22, 21),  # thumb
    (41, 27), (27, 26), (26, 25),            # index
    (41, 30), (30, 29), (29, 28),            # middle
    (41, 33), (33, 32), (32, 31),            # ring
    (41, 36), (36, 35), (35, 34),            # pinky
]

# Per-region colors (BGR)
_MHR_BODY_COLOR   = (0, 255, 255)    # cyan
_MHR_RHAND_COLOR  = (0, 128, 255)    # blue-orange
_MHR_LHAND_COLOR  = (0, 255, 0)      # green
_MHR_EXTRA_COLOR  = (255, 255, 255)  # white

def _mhr_color(idx: int) -> tuple:
    if idx in _MHR_LHAND:
        return _MHR_LHAND_COLOR
    elif idx in _MHR_RHAND:
        return _MHR_RHAND_COLOR
    elif idx in _MHR_EXTRA:
        return _MHR_EXTRA_COLOR
    return _MHR_BODY_COLOR


def draw_mhr70(frame: np.ndarray, result: FsbResult,
               kp_radius: int = 3, edge_thick: int = 1) -> None:
    """Draw 70 MHR keypoints projected to 2D (in-place)."""
    if not result.has_kps:
        return

    kps = np.array(result.kps_2d, dtype=np.float32).reshape(70, 2)
    H, W = frame.shape[:2]

    # Edges
    for i, j in MHR70_EDGES:
        pt1 = (int(kps[i, 0]), int(kps[i, 1]))
        pt2 = (int(kps[j, 0]), int(kps[j, 1]))
        # Cull out-of-frame
        if not (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                0 <= pt2[0] < W and 0 <= pt2[1] < H):
            continue
        col = _mhr_color(i)
        cv2.line(frame, pt1, pt2, col, edge_thick, cv2.LINE_AA)

    # Joints
    for k in range(70):
        pt = (int(kps[k, 0]), int(kps[k, 1]))
        if not (0 <= pt[0] < W and 0 <= pt[1] < H):
            continue
        cv2.circle(frame, pt, kp_radius, _mhr_color(k), -1, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, result: FsbResult,
                  kp_radius: int = 5, edge_thick: int = 2) -> None:
    """Draw YOLO COCO skeleton on frame (in-place)."""
    if not result.has_yolo_kps:
        return

    kps = np.array(result.yolo_kps, dtype=np.float32).reshape(17, 3)

    # Edges first (drawn under dots)
    for i, j in COCO_EDGES:
        if kps[i, 2] < VIS_THRESHOLD or kps[j, 2] < VIS_THRESHOLD:
            continue
        pt1 = (int(kps[i, 0]), int(kps[i, 1]))
        pt2 = (int(kps[j, 0]), int(kps[j, 1]))
        cv2.line(frame, pt1, pt2, _edge_color(i, j), edge_thick, cv2.LINE_AA)

    # Joints
    for k in range(17):
        if kps[k, 2] < VIS_THRESHOLD:
            continue
        pt = (int(kps[k, 0]), int(kps[k, 1]))
        cv2.circle(frame, pt, kp_radius, _KP_COLORS[k], -1, cv2.LINE_AA)
        cv2.circle(frame, pt, kp_radius, (0, 0, 0), 1,  cv2.LINE_AA)


def draw_bbox(frame: np.ndarray, result: FsbResult,
              color=(0, 255, 0), idx: int = 0) -> None:
    x1, y1, x2, y2 = (int(v) for v in result.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Small label showing global orientation
    gx, gy, gz = result.global_rot[0], result.global_rot[1], result.global_rot[2]
    label = f"#{idx}  rx={gx:.2f} ry={gy:.2f} rz={gz:.2f}"
    cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_pose_bars(frame: np.ndarray, results: list[FsbResult],
                   panel_w: int = 180) -> None:
    """Draw a small side panel showing first 12 body_pose angles as bars."""
    if not results:
        return
    H = frame.shape[0]
    bar_h = max(H // 14, 14)
    panel = np.zeros((H, panel_w, 3), dtype=np.uint8)
    r = results[0]
    bp = np.array(r.body_pose[:12])

    for i, val in enumerate(bp):
        norm = (np.clip(val, -1.5, 1.5) + 1.5) / 3.0  # 0..1
        bar_len = int(norm * (panel_w - 4))
        y0, y1 = i * bar_h + 2, (i + 1) * bar_h - 2
        # background
        cv2.rectangle(panel, (2, y0), (panel_w - 2, y1), (40, 40, 40), -1)
        # bar
        col = (0, int(255 * (1 - norm)), int(255 * norm))
        cv2.rectangle(panel, (2, y0), (2 + bar_len, y1), col, -1)
        cv2.putText(panel, f"bp{i}", (2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    frame[:H, frame.shape[1] - panel_w:] = panel


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

_PERSON_COLORS = [
    (0, 255, 0), (0, 200, 255), (255, 80, 0),
    (255, 0, 200), (200, 255, 0), (0, 100, 255),
]


def parse_args():
    p = argparse.ArgumentParser(description="SAM-3D-Body Python frontend")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx")
    build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")

    p.add_argument("--lib-dir",     default=build,
                   help="Directory containing libfast_sam_3dbody.so")
    p.add_argument("--onnx-dir",    default=onnx)
    p.add_argument("--gguf",        default=os.path.join(onnx, "pipeline.gguf"))
    p.add_argument("--yolo",        default=os.path.join(onnx, "yolo.onnx"))
    p.add_argument("--from",        dest="src", default="0",
                   help="Webcam index, image path, or video path")
    p.add_argument("--cuda",        type=int, default=0)
    p.add_argument("--max-skeletons", type=int, default=0,
                   help="Maximum skeletons to process and draw (0 = unlimited)")
    p.add_argument("--thresh",      type=float, default=0.5)
    p.add_argument("--nms",         type=float, default=0.45)
    p.add_argument("--fx",          type=float, default=0.0)
    p.add_argument("--fy",          type=float, default=0.0)
    p.add_argument("--cx",          type=float, default=0.0)
    p.add_argument("--cy",          type=float, default=0.0)
    p.add_argument("--headless",    action="store_true",
                   help="Do not open a display window")
    p.add_argument("--out",         default="",
                   help="Write visualised output to this file (image or video)")
    p.add_argument("--skip-body",   action="store_true",
                   help="Skip body model (faster, no hand/foot keypoints)")
    return p.parse_args()


def main():
    args = parse_args()

    lib = load_library(args.lib_dir)

    # ── Init pipeline ─────────────────────────────────────────────────────────
    handle = lib.fsb_create()
    if not handle:
        sys.exit("fsb_create() returned NULL")

    cfg = FsbConfig(
        onnx_dir       = args.onnx_dir.encode(),
        gguf_path      = args.gguf.encode(),
        yolo_path      = args.yolo.encode(),
        cuda_device    = args.cuda,
        skip_body_model= 1 if args.skip_body else 0,
        person_thresh  = args.thresh,
        person_nms_iou = args.nms,
        max_persons    = args.max_skeletons,
        focal_x        = args.fx,
        focal_y        = args.fy,
        principal_x    = args.cx,
        principal_y    = args.cy,
    )

    print("Loading pipeline …")
    t0 = time.perf_counter()
    if not lib.fsb_load(handle, ctypes.byref(cfg)):
        lib.fsb_destroy(handle)
        sys.exit("Pipeline load failed")
    print(f"Pipeline ready in {(time.perf_counter()-t0)*1000:.0f} ms")

    # Result buffer (enough for any realistic number of people per frame)
    MAX_RESULTS = max(args.max_skeletons if args.max_skeletons > 0 else 32, 32)
    ResultArray = FsbResult * MAX_RESULTS
    results_buf = ResultArray()

    # ── Open input ────────────────────────────────────────────────────────────
    src = args.src
    is_image = False
    cap = None

    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        if any(src.lower().endswith(e) for e in img_exts):
            is_image = True
        else:
            cap = cv2.VideoCapture(src)

    if not is_image and (cap is None or not cap.isOpened()):
        lib.fsb_destroy(handle)
        sys.exit(f"Cannot open input: {src}")

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if args.out:
        if is_image:
            pass  # written at the end with cv2.imwrite
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
            w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer  = cv2.VideoWriter(args.out, fourcc, fps_src, (w_src, h_src))

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_count = 0
    fps_ema = 0.0
    fps_alpha = 0.1
    prev_t = time.perf_counter()

    while True:
        if is_image:
            frame = cv2.imread(src)
            if frame is None:
                sys.exit(f"Cannot read image: {src}")
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

        H, W = frame.shape[:2]
        bgr_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        t_inf = time.perf_counter()
        n = lib.fsb_process_bgr(handle, bgr_ptr, W, H, results_buf, MAX_RESULTS)
        inf_ms = (time.perf_counter() - t_inf) * 1000
        frame_count += 1

        # ── Visualise ─────────────────────────────────────────────────────────
        vis = frame.copy()
        people: list[FsbResult] = [results_buf[i] for i in range(n)]

        for idx, r in enumerate(people):
            color = _PERSON_COLORS[idx % len(_PERSON_COLORS)]
            draw_bbox(vis, r, color=color, idx=idx)
            draw_skeleton(vis, r, kp_radius=5, edge_thick=2)
            draw_mhr70(vis, r, kp_radius=2, edge_thick=1)

        if people:
            draw_pose_bars(vis, people)

        # ── HUD ───────────────────────────────────────────────────────────────
        now = time.perf_counter()
        fps_ema = fps_alpha / (now - prev_t) + (1 - fps_alpha) * fps_ema
        prev_t = now

        hud = f"FPS {fps_ema:.1f}  |  {inf_ms:.0f} ms  |  {n} person(s)"
        if args.max_skeletons:
            hud += f"  [max {args.max_skeletons}]"
        cv2.putText(vis, hud, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # ── Output ────────────────────────────────────────────────────────────
        if writer:
            writer.write(vis)
        if args.out and is_image:
            cv2.imwrite(args.out, vis)
            print(f"Saved {args.out}")

        if not args.headless:
            cv2.imshow("SAM-3D-Body", vis)
            key = cv2.waitKey(1 if not is_image else 0) & 0xFF
            if key == ord("q") or key == 27:
                break

        if is_image:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if writer:
        writer.release()
    if cap:
        cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    lib.fsb_destroy(handle)

    if frame_count:
        print(f"Processed {frame_count} frame(s)  avg {fps_ema:.1f} fps")


if __name__ == "__main__":
    main()
