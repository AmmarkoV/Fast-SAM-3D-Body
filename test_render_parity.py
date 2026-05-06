#!/usr/bin/env python3
"""
test_render_parity.py — Given IDENTICAL MHR pose params, compare:

  A. Python mhr_forward (correctives=True)  → pyrender
  B. Python mhr_forward (correctives=False) → pyrender
  C. C-LBS verts (from /tmp/cpp_lbs_verts.bin)  → pyrender

All three use pyrender with the SAME camera, so projection math is NOT a variable.
This isolates whether the issue is correctives, LBS code, or camera setup.

Usage:
    python test_render_parity.py [--image notebook/images/dancing.jpg]

Outputs:
    /tmp/render_parity/
        A_py_correctives.png      — Python MHR with correctives=True
        B_py_no_correctives.png   — Python MHR with correctives=False
        C_cpp_lbs_pyrender.png    — C-LBS verts rendered with pyrender
        diff_AB.png               — |A-B| diff image (correctives effect)
        diff_BC.png               — |B-C| diff image (C-LBS vs Python, no correctives)
        summary.txt               — vertex diff stats
"""
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import torch
import cv2

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.visualization.renderer import Renderer

OUT_DIR = "/tmp/render_parity"
os.makedirs(OUT_DIR, exist_ok=True)


def render_verts(renderer, verts, cam_t, frame):
    """Render verts with pyrender, return uint8 BGR."""
    out = renderer(verts.astype(np.float32),
                   cam_t.astype(np.float32),
                   frame.copy(),
                   mesh_base_color=(0.65, 0.75, 0.9),
                   scene_bg_color=(1, 1, 1))
    img = (out * 255).clip(0, 255).astype(np.uint8)
    return img


def save_diff(path, a, b):
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    diff = diff.clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, diff)
    mean_diff = diff.mean()
    max_diff  = diff.max()
    return mean_diff, max_diff


def mhr_run(head_pose, global_rot, body_pose, hand_pose, scale, shape, face_params,
            apply_correctives, device):
    """Run mhr_forward with the given correctives flag."""
    head_pose.apply_correctives = apply_correctives
    def _t(arr, n):
        return torch.tensor(arr[:n], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = head_pose.mhr_forward(
            global_trans=torch.zeros(1, 3, device=device),
            global_rot=_t(global_rot, 3),
            body_pose_params=_t(body_pose, 133),
            hand_pose_params=_t(hand_pose, 108),
            scale_params=_t(scale, 28),
            shape_params=_t(shape, 45),
            expr_params=_t(face_params, 72),
            return_keypoints=False,
        )
    if isinstance(out, tuple):
        verts = out[0]
    else:
        verts = out
    verts = verts.clone()
    verts[..., [1, 2]] *= -1   # camera-adjacent flip (Y,Z)
    return verts[0].cpu().float().numpy()


def main():
    ckpt = os.path.join(_root, "checkpoints/sam-3d-body-dinov3/model.ckpt")
    mhr_pt = os.path.join(_root, "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")

    # ── Load params ─────────────────────────────────────────────────────────────
    # Prefer params saved by a previous pipeline run
    ref_npz = "/tmp/verify_transforms/ref_data.npz"
    cpp_npz = "/tmp/cpp_pipeline_results.npz"

    if os.path.exists(cpp_npz):
        d = np.load(cpp_npz, allow_pickle=False)
        global_rot  = d["p0_global_rot"]
        body_pose   = d["p0_body_pose"]
        hand_pose   = d["p0_hand_pose"]
        scale       = d["p0_scale"]
        shape       = d["p0_shape"]
        face_params = d["p0_face_params"]
        pred_cam_t  = d["p0_pred_cam_t"]
        focal_length = float(np.asarray(d["p0_focal_length"]).ravel()[0])
        print(f"[*] Loaded params from {cpp_npz}")
    else:
        raise FileNotFoundError(f"No params found. Run comparePipelines.py first to produce {cpp_npz}")

    # Load reference frame
    if os.path.exists(ref_npz):
        ref_data = np.load(ref_npz, allow_pickle=False)
        frame = ref_data["frame"].astype(np.uint8)
        faces = ref_data["faces"]
    else:
        # Fall back: load frame from image
        img_path = os.path.join(_root, "notebook/images/dancing.jpg")
        frame = cv2.imread(img_path)
        faces = None

    H, W = frame.shape[:2]
    print(f"[*] Frame: {W}×{H}  focal={focal_length:.2f}  cam_t={pred_cam_t}")

    # ── Load model ───────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading SAM-3D-Body model on {device}")
    model, _ = load_sam_3d_body(checkpoint_path=ckpt, mhr_path=mhr_pt, device=device)
    model.eval()
    head_pose = model.head_pose
    if faces is None:
        faces = head_pose.faces.cpu().numpy()

    renderer = Renderer(focal_length=focal_length, faces=faces)

    # ── A: Python mhr_forward with correctives=True ──────────────────────────────
    print("[*] Running A: Python MHR (correctives=True) ...")
    verts_A = mhr_run(head_pose, global_rot, body_pose, hand_pose, scale, shape, face_params,
                      apply_correctives=True, device=device)
    img_A = render_verts(renderer, verts_A, pred_cam_t, frame)
    cv2.imwrite(os.path.join(OUT_DIR, "A_py_correctives.png"), img_A)
    print(f"  verts_A bounds: X[{verts_A[:,0].min():.4f},{verts_A[:,0].max():.4f}]  Y[{verts_A[:,1].min():.4f},{verts_A[:,1].max():.4f}]  Z[{verts_A[:,2].min():.4f},{verts_A[:,2].max():.4f}]")

    # ── B: Python mhr_forward with correctives=False ─────────────────────────────
    print("[*] Running B: Python MHR (correctives=False) ...")
    verts_B = mhr_run(head_pose, global_rot, body_pose, hand_pose, scale, shape, face_params,
                      apply_correctives=False, device=device)
    img_B = render_verts(renderer, verts_B, pred_cam_t, frame)
    cv2.imwrite(os.path.join(OUT_DIR, "B_py_no_correctives.png"), img_B)
    print(f"  verts_B bounds: X[{verts_B[:,0].min():.4f},{verts_B[:,0].max():.4f}]  Y[{verts_B[:,1].min():.4f},{verts_B[:,1].max():.4f}]  Z[{verts_B[:,2].min():.4f},{verts_B[:,2].max():.4f}]")

    # ── C: C-LBS verts from dump ─────────────────────────────────────────────────
    cpp_bin = "/tmp/cpp_lbs_verts.bin"
    verts_C = None
    if os.path.exists(cpp_bin):
        with open(cpp_bin, "rb") as f:
            n_v, n_c = np.frombuffer(f.read(8), dtype=np.int32)
            verts_C = np.frombuffer(f.read(int(n_v)*int(n_c)*4),
                                     dtype=np.float32).reshape(n_v, n_c)
        print(f"[*] Loaded C-LBS verts from {cpp_bin}: {verts_C.shape}")
        print(f"  verts_C bounds: X[{verts_C[:,0].min():.4f},{verts_C[:,0].max():.4f}]  Y[{verts_C[:,1].min():.4f},{verts_C[:,1].max():.4f}]  Z[{verts_C[:,2].min():.4f},{verts_C[:,2].max():.4f}]")
        img_C = render_verts(renderer, verts_C, pred_cam_t, frame)
        cv2.imwrite(os.path.join(OUT_DIR, "C_cpp_lbs_pyrender.png"), img_C)
    else:
        print(f"[!] {cpp_bin} not found — run render binary first to generate it")
        print("    fast_sam_3dbody_cpp/build/fast_sam_3dbody_render --from notebook/images/dancing.jpg --save /tmp/cpp_render.png")

    # ── Diffs ────────────────────────────────────────────────────────────────────
    print("\n── Vertex diffs ─────────────────────────────────────────────────────")
    dAB = np.abs(verts_A - verts_B)
    print(f"  A vs B (correctives effect):  max={dAB.max()*100:.2f}cm  mean={dAB.mean()*100:.3f}cm")

    if verts_C is not None:
        dBC = np.abs(verts_B - verts_C)
        dAC = np.abs(verts_A - verts_C)
        print(f"  B vs C (Python-noCorr vs C-LBS): max={dBC.max()*100:.2f}cm  mean={dBC.mean()*100:.3f}cm")
        print(f"  A vs C (Python+Corr  vs C-LBS): max={dAC.max()*100:.2f}cm  mean={dAC.mean()*100:.3f}cm")

    print("\n── Pixel diffs ──────────────────────────────────────────────────────")
    mAB_mean, mAB_max = save_diff(os.path.join(OUT_DIR, "diff_AB.png"), img_A, img_B)
    print(f"  diff_AB (correctives ON vs OFF): mean={mAB_mean:.2f}  max={mAB_max}")

    if verts_C is not None:
        mBC_mean, mBC_max = save_diff(os.path.join(OUT_DIR, "diff_BC.png"), img_B, img_C)
        mAC_mean, mAC_max = save_diff(os.path.join(OUT_DIR, "diff_AC.png"), img_A, img_C)
        print(f"  diff_BC (py-noCorr  vs C-LBS):  mean={mBC_mean:.2f}  max={mBC_max}")
        print(f"  diff_AC (py+Corr    vs C-LBS):  mean={mAC_mean:.2f}  max={mAC_max}")

    # Write summary
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Params source: {cpp_npz}\n")
        f.write(f"Frame: {W}×{H}  focal={focal_length:.2f}  cam_t={pred_cam_t}\n\n")
        f.write(f"Vertex diffs (m):\n")
        f.write(f"  A(py+corr) vs B(py-corr): max={dAB.max():.6f}  mean={dAB.mean():.6f}\n")
        if verts_C is not None:
            f.write(f"  B(py-corr) vs C(cpp-lbs): max={dBC.max():.6f}  mean={dBC.mean():.6f}\n")
            f.write(f"  A(py+corr) vs C(cpp-lbs): max={dAC.max():.6f}  mean={dAC.mean():.6f}\n")
        f.write(f"\nPixel diffs:\n")
        f.write(f"  diff_AB: mean={mAB_mean:.2f} max={mAB_max}\n")
        if verts_C is not None:
            f.write(f"  diff_BC: mean={mBC_mean:.2f} max={mBC_max}\n")
            f.write(f"  diff_AC: mean={mAC_mean:.2f} max={mAC_max}\n")

    print(f"\nOutputs saved to {OUT_DIR}/")
    print(f"Summary: {summary_path}")
    print(f"\nConclusion:")
    if dAB.max() > 0.01:
        print(f"  Correctives shift vertices by up to {dAB.max()*100:.1f}cm (mean {dAB.mean()*100:.2f}cm).")
    if verts_C is not None:
        if dAC.max() < 0.001:
            print("  *** C-LBS+correctives ≡ Python+correctives (<1mm). Rendering is PIXEL-EQUIVALENT. ***")
        elif dAC.max() < 0.01:
            print(f"  C-LBS+correctives ≈ Python+correctives ({dAC.max()*1000:.1f}mm sub-cm). Close enough.")
        else:
            print(f"  C-LBS+correctives DIVERGES from Python+correctives ({dAC.max()*100:.1f}cm) — needs fix.")
        if dBC.max() < 0.001:
            print("  Note: without correctives, C-LBS matches Python (<1mm).")
        elif dBC.max() > 0.01:
            print(f"  Note: C-LBS without correctives differs by {dBC.max()*100:.1f}cm (expected — correctives are ON).")


if __name__ == "__main__":
    main()
