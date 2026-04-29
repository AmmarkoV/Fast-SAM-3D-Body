#!/usr/bin/env python3
"""
dump_joint_transforms.py — Joint transform debugger for the MHR LBS pipeline.

Reads the .lbs body model file and the first-frame dump written by the C renderer
(/tmp/mhr_lbs_dump.bin), then replicates every step of the LBS forward pass in
numpy and prints per-joint values so they can be compared with C stderr output.

Usage:
    # 1. Run the C renderer once; it writes /tmp/mhr_lbs_dump.bin on the first frame.
    ./build/fast_sam_3dbody_render --lbs onnx/body_model.lbs ... --from video.avi

    # 2. Run this script against the dump:
    python scripts/dump_joint_transforms.py \\
        --lbs onnx/body_model.lbs \\
        --dump /tmp/mhr_lbs_dump.bin \\
        [--joints 0 1 2 3]        # optional: restrict output to these joint indices

The script also runs an independent sanity check (rotation-only identity pose) so
you can verify the quaternion math without needing a real inference dump.
"""

import argparse
import struct
import sys
import numpy as np


# ── .lbs file reader (matches C mhr_lbs_load) ────────────────────────────────

def load_lbs(path):
    """Read the binary .lbs body model file and return a dict of numpy arrays."""
    with open(path, "rb") as f:
        raw = f.read()

    off = 0
    def read_u32(n):
        nonlocal off
        vals = struct.unpack_from(f"{n}I", raw, off)
        off += 4 * n
        return vals

    def read_f32(n):
        nonlocal off
        arr = np.frombuffer(raw, dtype=np.float32, count=n, offset=off).copy()
        off += 4 * n
        return arr

    def read_i32(n):
        nonlocal off
        arr = np.frombuffer(raw, dtype=np.int32, count=n, offset=off).copy()
        off += 4 * n
        return arr

    magic, version, n_joints, n_skin, n_verts, n_shape_pc, n_face_pc, pt_cols = read_u32(8)
    assert magic == 0x4C425300, f"bad magic 0x{magic:08x}"
    assert version == 1,        f"bad version {version}"

    pt_rows = n_joints * 7

    lbs = {
        "n_joints":   n_joints,
        "n_skin":     n_skin,
        "n_verts":    n_verts,
        "n_shape_pc": n_shape_pc,
        "n_face_pc":  n_face_pc,
        "pt_cols":    pt_cols,
        "pt_rows":    pt_rows,
    }

    lbs["PT"]                 = read_f32(pt_rows * pt_cols).reshape(pt_rows, pt_cols)
    lbs["joint_offsets"]      = read_f32(n_joints * 3).reshape(n_joints, 3)
    lbs["joint_prerotations"] = read_f32(n_joints * 4).reshape(n_joints, 4)  # XYZW
    lbs["joint_parents"]      = read_i32(n_joints)
    lbs["inv_bind_pose"]      = read_f32(n_joints * 8).reshape(n_joints, 8)
    lbs["skin_joint_idx"]     = read_i32(n_skin)
    lbs["skin_weights"]       = read_f32(n_skin)
    lbs["skin_vert_idx"]      = read_i32(n_skin)
    lbs["base_shape"]         = read_f32(n_verts * 3).reshape(n_verts, 3)
    lbs["shape_vectors"]      = read_f32(n_shape_pc * n_verts * 3).reshape(n_shape_pc, n_verts, 3)
    lbs["face_vectors"]       = read_f32(n_face_pc  * n_verts * 3).reshape(n_face_pc,  n_verts, 3)

    return lbs


def load_dump(path):
    """Read the first-frame binary dump written by the C renderer."""
    with open(path, "rb") as f:
        raw = f.read()
    off = 0

    n_joints, pt_cols = struct.unpack_from("2i", raw, off); off += 8
    model_params = np.frombuffer(raw, dtype=np.float32, count=204, offset=off).copy(); off += 4*204
    # shape and face coeffs follow (variable length — read rest)
    remaining = (len(raw) - off) // 4
    rest = np.frombuffer(raw, dtype=np.float32, count=remaining, offset=off).copy()
    return model_params, rest, n_joints, pt_cols


# ── numpy quaternion helpers matching the C code exactly ─────────────────────

def qmul(a, b):
    """Hamilton product a * b  (XYZW).  R_result = R_a @ R_b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        bx*aw + bw*ax + bz*ay - by*az,
        by*aw - bz*ax + bw*ay + bx*az,
        bz*aw + by*ax - bx*ay + bw*az,
        bw*aw - bx*ax - by*ay - bz*az,
    ], dtype=np.float64)


def qrot(q, v):
    """Rotate vector v by unit quaternion q (XYZW): q*v*q^{-1}."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0*(qy*vz - qz*vy)
    ty = 2.0*(qz*vx - qx*vz)
    tz = 2.0*(qx*vy - qy*vx)
    return np.array([
        vx + qw*tx + (qy*tz - qz*ty),
        vy + qw*ty + (qz*tx - qx*tz),
        vz + qw*tz + (qx*ty - qy*tx),
    ], dtype=np.float64)


def euler_xyz_to_quat(rx, ry, rz):
    """
    ZYX Euler (rx,ry,rz) → unit quaternion XYZW.
    PyMomentum XYZ intrinsic convention: R = Rz(rz) * Ry(ry) * Rx(rx).
    q = qz * qy * qx  so that rotating a vector gives Rz(Ry(Rx(v))).

    The angles arriving here are produced by rot6d_to_euler / batchXYZfrom6D
    and match what the parameter transform (PT) outputs in joint_params[3:6].
    """
    hx, hy, hz = rx*0.5, ry*0.5, rz*0.5
    qx = np.array([np.sin(hx), 0.0,        0.0,        np.cos(hx)])
    qy = np.array([0.0,        np.sin(hy), 0.0,        np.cos(hy)])
    qz = np.array([0.0,        0.0,        np.sin(hz), np.cos(hz)])
    tmp = qmul(qz, qy)   # R_tmp = Rz @ Ry
    return qmul(tmp, qx)  # R_q   = Rz @ Ry @ Rx


def quat_to_rotmat(q):
    """XYZW quaternion → 3×3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def rot6d_to_euler_correct(d6):
    """
    6D rotation → ZYX Euler (rx, ry, rz).

    Matches Python batchXYZfrom6D / rot6d_to_rotmat:
      col0 = normalize(d6[0:3])            ← first COLUMN of R
      col1 = Gram-Schmidt(d6[3:6], col0)   ← second COLUMN of R
      col2 = col0 × col1

    ZYX extraction from bottom row [R[2,0], R[2,1], R[2,2]] and left column top:
      rx = atan2(R[2,1], R[2,2])  = atan2(col1[2], col2[2])   [e12, e22]
      ry = asin(-R[2,0])          = asin(-col0[2])              [e02]
      rz = atan2(R[1,0], R[0,0]) = atan2(col0[1], col0[0])    [e01, e00]

    NOTE: a previous C bug used col2 elements (e20,e21) in place of col0/col1
    elements (e02,e12), effectively extracting the inverse rotation's angles.
    """
    a = np.array(d6[:3], dtype=np.float64)
    b = np.array(d6[3:], dtype=np.float64)
    col0 = a / (np.linalg.norm(a) + 1e-8)
    col1 = b - np.dot(col0, b) * col0
    col1 /= (np.linalg.norm(col1) + 1e-8)
    col2 = np.cross(col0, col1)

    # col0[row] = R[row, 0],  col1[row] = R[row, 1],  col2[row] = R[row, 2]
    e02 = col0[2]   # R[2,0]
    e12 = col1[2]   # R[2,1]
    e22 = col2[2]   # R[2,2]
    e01 = col0[1]   # R[1,0]
    e00 = col0[0]   # R[0,0]

    rx = np.arctan2(e12, e22)
    ry = np.arcsin(np.clip(-e02, -1.0, 1.0))
    rz = np.arctan2(e01, e00)
    return np.array([rx, ry, rz])


def rot6d_to_euler_buggy(d6):
    """
    The OLD (buggy) C implementation — kept here for comparison.
    Uses transposed indices: e20/e21/e10 instead of e02/e12/e01,
    which extracts angles for R^T (the inverse rotation).
    """
    a = np.array(d6[:3], dtype=np.float64)
    b = np.array(d6[3:], dtype=np.float64)
    col0 = a / (np.linalg.norm(a) + 1e-8)
    col1 = b - np.dot(col0, b) * col0
    col1 /= (np.linalg.norm(col1) + 1e-8)
    col2 = np.cross(col0, col1)

    e20 = col2[0]   # R[0,2]  ← WRONG for ry
    e21 = col2[1]   # R[1,2]  ← WRONG for rx
    e22 = col2[2]   # R[2,2]
    e10 = col1[0]   # R[0,1]  ← WRONG for rz
    e00 = col0[0]   # R[0,0]

    rx = np.arctan2(e21, e22)
    ry = np.arcsin(np.clip(-e20, -1.0, 1.0))
    rz = np.arctan2(e10, e00)
    return np.array([rx, ry, rz])


# ── LBS forward pass (mirrors mhr_lbs_compute in C) ─────────────────────────

def lbs_forward(lbs, model_params, shape_coeffs=None, face_coeffs=None, show_joints=None):
    """
    Run the full LBS forward pass in numpy, printing intermediate values.

    Parameters match mhr_lbs_compute:
      model_params [204]  — assembled by build_model_params / mhr_forward
      shape_coeffs [n_shape_pc] — identity blend shape weights (zeros if None)
      face_coeffs  [n_face_pc]  — facial blend shape weights  (zeros if None)
      show_joints  — list of joint indices to print; None = all
    """
    nj  = lbs["n_joints"]
    nv  = lbs["n_verts"]
    PT  = lbs["PT"].astype(np.float64)
    npc = lbs["pt_cols"]

    if shape_coeffs is None: shape_coeffs = np.zeros(lbs["n_shape_pc"])
    if face_coeffs  is None: face_coeffs  = np.zeros(lbs["n_face_pc"])
    if show_joints  is None: show_joints  = list(range(min(nj, 10)))  # first 10 by default

    # ── Step 1: unposed vertices ──────────────────────────────────────────────
    unposed = lbs["base_shape"].copy().astype(np.float64)
    for i, c in enumerate(shape_coeffs):
        if c != 0.0: unposed += c * lbs["shape_vectors"][i]
    for i, c in enumerate(face_coeffs):
        if c != 0.0: unposed += c * lbs["face_vectors"][i]

    print(f"[Step 1] unposed bounds: "
          f"x[{unposed[:,0].min():.3f},{unposed[:,0].max():.3f}] "
          f"y[{unposed[:,1].min():.3f},{unposed[:,1].max():.3f}] "
          f"z[{unposed[:,2].min():.3f},{unposed[:,2].max():.3f}]")

    # ── Step 2: joint_params = PT @ model_params ──────────────────────────────
    # model_params layout: [0:3]=global_trans*10, [3:6]=global_rot ZYX,
    #                      [6:136]=body_pose_euler[:130], [136:204]=scales
    input_vec = np.zeros(npc, dtype=np.float64)
    mp = np.array(model_params, dtype=np.float64)
    input_vec[:min(npc, 204)] = mp[:min(npc, 204)]
    joint_params = (PT @ input_vec).reshape(nj, 7)

    print(f"\n[Step 2] PT @ model_params → joint_params [{nj}×7]")
    print(f"  model_params[0:6] = {mp[:6]}  (global_trans*10, global_rot ZYX)")
    for j in show_joints:
        jp = joint_params[j]
        print(f"  joint[{j:3d}] t=({jp[0]:.4f},{jp[1]:.4f},{jp[2]:.4f}) "
              f"euler_xyz=({jp[3]:.4f},{jp[4]:.4f},{jp[5]:.4f}) log2s={jp[6]:.4f}")

    # ── Step 3: local TRS per joint ───────────────────────────────────────────
    # q_local[j] = joint_prerotations[j] * euler_xyz_to_quat(jp[3], jp[4], jp[5])
    # The euler angles are in PyMomentum XYZ intrinsic convention:
    #   R = Rz(jp[5]) * Ry(jp[4]) * Rx(jp[3])
    print(f"\n[Step 3] local TRS")
    offsets = lbs["joint_offsets"].astype(np.float64)
    pres    = lbs["joint_prerotations"].astype(np.float64)
    t_local = np.zeros((nj, 3))
    q_local = np.zeros((nj, 4))
    s_local = np.zeros(nj)
    LN2 = np.log(2.0)

    for j in range(nj):
        jp  = joint_params[j]
        t_local[j] = offsets[j] + jp[:3]
        q_euler    = euler_xyz_to_quat(jp[3], jp[4], jp[5])
        q_local[j] = qmul(pres[j], q_euler)   # R_local = R_pre @ R_euler
        s_local[j] = np.exp(jp[6] * LN2)
        if j in show_joints:
            R = quat_to_rotmat(q_local[j])
            print(f"  joint[{j:3d}] t_local={t_local[j]}  s={s_local[j]:.5f}")
            print(f"           q_euler   = {q_euler}  (from euler {jp[3:6]})")
            print(f"           q_local   = {q_local[j]}")
            print(f"           R_local   =\n{R}")

    # ── Step 4: FK chain ─────────────────────────────────────────────────────
    # g_q[j] = g_q[parent] * q_local[j]   → R_global[j] = R_parent @ R_local[j]
    # g_t[j] = g_t[parent] + g_s[parent] * rotate(g_q[parent], t_local[j])
    print(f"\n[Step 4] FK chain (global TRS)")
    parents = lbs["joint_parents"]
    g_t = np.zeros((nj, 3))
    g_q = np.zeros((nj, 4))
    g_s = np.zeros(nj)

    for j in range(nj):
        p = parents[j]
        if p < 0:
            g_t[j] = t_local[j]
            g_q[j] = q_local[j]
            g_s[j] = s_local[j]
        else:
            g_s[j] = g_s[p] * s_local[j]
            g_q[j] = qmul(g_q[p], q_local[j])
            rt     = qrot(g_q[p], t_local[j])
            g_t[j] = g_t[p] + g_s[p] * rt
        if j in show_joints:
            print(f"  joint[{j:3d}] parent={p:3d}  "
                  f"g_t=({g_t[j,0]:.4f},{g_t[j,1]:.4f},{g_t[j,2]:.4f})  "
                  f"g_q={g_q[j]}  g_s={g_s[j]:.5f}")

    # ── Step 5: skin TRS = global(j) ∘ inv_bind(j) ───────────────────────────
    # skin_q = g_q @ ib_q    → R_skin = R_global @ R_inv_bind
    # skin_t = g_t + g_s * rotate(g_q, ib_t)
    print(f"\n[Step 5] skin TRS")
    ib  = lbs["inv_bind_pose"].astype(np.float64)  # [nj, 8]: [tx,ty,tz, qx,qy,qz,qw, scale]
    skin_t = np.zeros((nj, 3))
    skin_q = np.zeros((nj, 4))
    skin_s = np.zeros(nj)

    for j in range(nj):
        ib_t = ib[j, :3]
        ib_q = ib[j, 3:7]
        ib_s = ib[j, 7]
        skin_s[j] = g_s[j] * ib_s
        skin_q[j] = qmul(g_q[j], ib_q)
        rt         = qrot(g_q[j], ib_t)
        skin_t[j]  = g_t[j] + g_s[j] * rt
        if j in show_joints:
            print(f"  joint[{j:3d}]  "
                  f"skin_t=({skin_t[j,0]:.4f},{skin_t[j,1]:.4f},{skin_t[j,2]:.4f})  "
                  f"skin_s={skin_s[j]:.5f}")

    # ── Step 6: LBS ──────────────────────────────────────────────────────────
    out_verts = np.zeros((nv, 3))
    sji = lbs["skin_joint_idx"]
    sw  = lbs["skin_weights"]
    svi = lbs["skin_vert_idx"]
    for k in range(lbs["n_skin"]):
        ji  = sji[k]; vi = svi[k]; w = sw[k]; sx = skin_s[ji]
        pv  = qrot(skin_q[ji], unposed[vi])
        out_verts[vi] += w * (skin_t[ji] + sx * pv)

    # Step 7: Y,Z flip (matches Python mhr_head.py verts[..., [1,2]] *= -1)
    out_verts[:, 1] *= -1.0
    out_verts[:, 2] *= -1.0

    print(f"\n[Step 6+7] output vertex bounds: "
          f"x[{out_verts[:,0].min():.3f},{out_verts[:,0].max():.3f}] "
          f"y[{out_verts[:,1].min():.3f},{out_verts[:,1].max():.3f}] "
          f"z[{out_verts[:,2].min():.3f},{out_verts[:,2].max():.3f}]")
    return out_verts


# ── sanity check: test rot6d_to_euler fix ────────────────────────────────────

def sanity_check_euler():
    """
    Verify rot6d_to_euler_correct vs rot6d_to_euler_buggy on a known rotation.
    Creates R = Rz(30°)*Ry(20°)*Rx(10°) and checks extraction.
    """
    rx_deg, ry_deg, rz_deg = 10.0, 20.0, 30.0
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    # Build R = Rz * Ry * Rx explicitly
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    R  = Rz @ Ry @ Rx

    # 6D representation: first two COLUMNS of R
    d6 = np.concatenate([R[:, 0], R[:, 1]])

    got_correct = rot6d_to_euler_correct(d6)
    got_buggy   = rot6d_to_euler_buggy(d6)

    expected = np.array([rx, ry, rz])
    err_correct = np.abs(got_correct - expected)
    err_buggy   = np.abs(got_buggy   - expected)

    print("=== Euler extraction sanity check ===")
    print(f"  Input:   R = Rz({rz_deg}°)*Ry({ry_deg}°)*Rx({rx_deg}°)")
    print(f"  Expected (rx,ry,rz) deg: {np.degrees(expected)}")
    print(f"  correct  (rx,ry,rz) deg: {np.degrees(got_correct)}  "
          f"max_err={np.degrees(err_correct).max():.4f}°")
    print(f"  buggy    (rx,ry,rz) deg: {np.degrees(got_buggy)}   "
          f"max_err={np.degrees(err_buggy).max():.4f}°  ← should be WRONG")

    ok = err_correct.max() < 1e-5
    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dump MHR LBS joint transforms")
    parser.add_argument("--lbs",    required=True, help="Path to body_model.lbs")
    parser.add_argument("--dump",   default="/tmp/mhr_lbs_dump.bin",
                        help="First-frame binary dump from C renderer")
    parser.add_argument("--joints", nargs="*", type=int, default=None,
                        help="Joint indices to print (default: 0..9)")
    parser.add_argument("--sanity", action="store_true", default=True,
                        help="Run Euler extraction sanity check (default: on)")
    args = parser.parse_args()

    if args.sanity:
        ok = sanity_check_euler()
        print()
        if not ok:
            print("FATAL: Euler sanity check failed — fix rot6d_to_euler before debugging LBS.",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Loading .lbs: {args.lbs}")
    lbs = load_lbs(args.lbs)
    print(f"  joints={lbs['n_joints']}  verts={lbs['n_verts']}  "
          f"shape_pc={lbs['n_shape_pc']}  face_pc={lbs['n_face_pc']}  "
          f"pt_cols={lbs['pt_cols']}")

    print(f"\nLoading first-frame dump: {args.dump}")
    try:
        model_params, rest, n_joints_dump, pt_cols_dump = load_dump(args.dump)
    except FileNotFoundError:
        print(f"  [WARN] dump file not found; using identity (zero) model_params")
        model_params = np.zeros(204, dtype=np.float32)
        rest = np.array([])

    print(f"  model_params[0:6] = {model_params[:6]}  (global_trans*10, global_rot ZYX)")
    print(f"  model_params[3:6] = {model_params[3:6]}  (global_rot rx={np.degrees(model_params[3]):.1f}° "
          f"ry={np.degrees(model_params[4]):.1f}° rz={np.degrees(model_params[5]):.1f}°)")

    # Split rest into shape and face coeffs (sizes from lbs)
    ns = lbs["n_shape_pc"]
    nf = lbs["n_face_pc"]
    shape_coeffs = rest[:ns]  if len(rest) >= ns      else np.zeros(ns)
    face_coeffs  = rest[ns:ns+nf] if len(rest) >= ns+nf else np.zeros(nf)

    show_joints = args.joints if args.joints is not None else list(range(min(lbs["n_joints"], 10)))

    print(f"\n{'='*60}")
    print(f"LBS FORWARD PASS  (showing joints {show_joints})")
    print(f"{'='*60}\n")

    lbs_forward(lbs, model_params, shape_coeffs, face_coeffs, show_joints=show_joints)


if __name__ == "__main__":
    main()
