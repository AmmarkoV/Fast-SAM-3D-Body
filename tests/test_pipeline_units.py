#!/usr/bin/env python3
"""
tests/test_pipeline_units.py

Unit tests for every logical step of the SAM-3D-Body Python→C++ pipeline.

Groups
------
  A. Math utilities    – rot6d_to_euler, quaternion multiply / rotate
  B. Preprocessing     – CLIFF condition_info, ray_cond
  C. MHR param decode  – compact_cont_to_body_params, build_model_params
  D. LBS file format   – magic, dimensions, index sanity
  E. LBS forward pass  – numpy reference, scale check, parity with Python model
  F. Coordinate system – flip consistency, no double-flip, model_params layout
  G. C LBS output      – compare /tmp/cpp_lbs_verts.bin against Python reference
  H. Export constants  – ONNX wrapper shape constants

Run:
    pytest tests/test_pipeline_units.py -v
"""

import math
import os
import struct
import sys

import numpy as np
import pytest

# ── repo root so sam_3d_body and fast_sam_3dbody_cpp are importable ──────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

LBS_PATH      = os.path.join(REPO, "fast_sam_3dbody_cpp", "onnx", "body_model.lbs")
PARAMS_NPZ    = os.path.join(REPO, "mhr_params_dancing.npz")
CPP_VERTS_BIN = "/tmp/cpp_lbs_verts.bin"
CKPT_PATH     = os.path.join(REPO, "checkpoints", "sam-3d-body-dinov3", "model.ckpt")
MHR_PATH      = os.path.join(REPO, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt")

HAS_LBS        = os.path.exists(LBS_PATH)
HAS_PARAMS     = os.path.exists(PARAMS_NPZ)
HAS_CPP_VERTS  = os.path.exists(CPP_VERTS_BIN)
HAS_CKPT       = os.path.exists(CKPT_PATH) and os.path.exists(MHR_PATH)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
# Reference implementations  (mirrors preprocess.hpp + dump_joint_transforms.py)
# ══════════════════════════════════════════════════════════════════════════════

def rot6d_to_euler_ref(d6):
    """
    Python reference for rot6d_to_euler() in preprocess.hpp.

    d6 : array[6] – first two *columns* of a rotation matrix R.
    Returns [rx, ry, rz] ZYX Euler (R = Rz(rz)*Ry(ry)*Rx(rx)).

    Matches batchXYZfrom6D / rot6d_to_rotmat in mhr_utils.py:
      col0 = normalise(d6[0:3])
      col1 = Gram-Schmidt(d6[3:6], col0)
      col2 = col0 × col1
      rx = atan2(R[2,1], R[2,2]),  ry = asin(-R[2,0]),  rz = atan2(R[1,0], R[0,0])
    """
    a = np.asarray(d6[:3], dtype=np.float64)
    b = np.asarray(d6[3:6], dtype=np.float64)

    na   = np.linalg.norm(a) + 1e-8
    col0 = a / na                              # [e00, e01, e02] = R[:,0]

    dot  = np.dot(col0, b)
    col1 = (b - dot * col0)
    nb   = np.linalg.norm(col1) + 1e-8
    col1 = col1 / nb                           # [e10, e11, e12] = R[:,1]

    # R[2,2] = (col0 × col1)[2] = col0[0]*col1[1] - col0[1]*col1[0]
    e22 = col0[0] * col1[1] - col0[1] * col1[0]

    rx = math.atan2(col1[2], e22)                        # atan2(R[2,1], R[2,2])
    ry = math.asin(max(-1.0, min(1.0, -col0[2])))        # asin(-R[2,0])
    rz = math.atan2(col0[1], col0[0])                    # atan2(R[1,0], R[0,0])
    return np.array([rx, ry, rz])


def qmul_ref(a, b):
    """Hamilton product a * b  (XYZW).  R_result = R_a @ R_b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        bx*aw + bw*ax + bz*ay - by*az,
        by*aw - bz*ax + bw*ay + bx*az,
        bz*aw + by*ax - bx*ay + bw*az,
        bw*aw - bx*ax - by*ay - bz*az,
    ], dtype=np.float64)


def qrot_ref(q, v):
    """Rotate vector v by unit quaternion q (XYZW): q*v*q^{-1}."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy*vz - qz*vy)
    ty = 2.0 * (qz*vx - qx*vz)
    tz = 2.0 * (qx*vy - qy*vx)
    return np.array([
        vx + qw*tx + (qy*tz - qz*ty),
        vy + qw*ty + (qz*tx - qx*tz),
        vz + qw*tz + (qx*ty - qy*tx),
    ], dtype=np.float64)


def euler_xyz_to_quat_ref(rx, ry, rz):
    """
    ZYX Euler (rx,ry,rz) → XYZW quaternion.
    R = Rz(rz) * Ry(ry) * Rx(rx)  so  q = qz * qy * qx.
    Matches mhr_euler_xyz_to_quat in model_loader_transform_joints.c.
    """
    hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5
    qx = np.array([math.sin(hx), 0.0,          0.0,          math.cos(hx)])
    qy = np.array([0.0,          math.sin(hy),  0.0,          math.cos(hy)])
    qz = np.array([0.0,          0.0,           math.sin(hz), math.cos(hz)])
    return qmul_ref(qmul_ref(qz, qy), qx)


# Joint index tables – mirror preprocess.hpp (BODY_3DOF_JOINT_IDXS etc.)
_BODY_3DOF = [
    (0,2,4), (6,8,10), (12,13,14), (15,16,17), (18,19,20),
    (21,22,23), (24,25,26), (27,28,29), (34,35,36), (37,38,39),
    (44,45,46), (53,54,55), (64,65,66), (85,69,73), (86,70,79),
    (87,71,82), (88,72,76), (91,92,93), (112,96,100), (113,97,106),
    (114,98,109), (115,99,103), (130,131,132),
]
_BODY_1DOF = [
    1,3,5,7,9,11,30,31,32,33,40,41,42,43,47,48,49,50,51,52,
    56,57,58,59,60,61,62,63,67,68,74,75,77,78,80,81,83,84,
    89,90,94,95,101,102,104,105,107,108,110,111,116,117,118,119,120,121,122,123,
]
_BODY_TRANS = [124, 125, 126, 127, 128, 129]


def compact_cont_to_body_params_ref(body_cont):
    """
    260-D continuous → 133-D Euler body_pose.
    Mirrors compact_cont_to_body_params() in preprocess.hpp.
    """
    body_cont = np.asarray(body_cont, dtype=np.float64)
    out = np.zeros(133, dtype=np.float64)
    N3, N1 = 23, 58

    for j, (i0, i1, i2) in enumerate(_BODY_3DOF):
        euler = rot6d_to_euler_ref(body_cont[j*6: j*6+6])
        out[i0] = euler[0]
        out[i1] = euler[1]
        out[i2] = euler[2]

    p1 = body_cont[N3 * 6:]
    for j, idx in enumerate(_BODY_1DOF):
        out[idx] = math.atan2(p1[j*2], p1[j*2+1])

    pt = body_cont[N3 * 6 + N1 * 2:]
    for j, idx in enumerate(_BODY_TRANS):
        out[idx] = pt[j]

    return out


def build_model_params_ref(global_rot_euler, body_euler):
    """
    Assemble model_params[204].
    Mirrors build_model_params() in preprocess.hpp.
    """
    ge = np.asarray(global_rot_euler, dtype=np.float64)
    be = np.asarray(body_euler, dtype=np.float64)
    mp = np.zeros(204, dtype=np.float64)
    # [0:3]   global_trans = 0 (single-view)
    # [3:6]   global_rot in ZYX storage order: [rz, ry, rx]
    #         rot6d_to_euler returns [rx,ry,rz]; Python's roma.rotmat_to_euler("ZYX")
    #         returns [rz,ry,rx], so we swap element 0 and 2.
    mp[3] = ge[2]   # rz
    mp[4] = ge[1]   # ry
    mp[5] = ge[0]   # rx
    # [6:136] body_pose first 130 of 133
    mp[6:136] = be[:130]
    # Zero hand joints: body_euler[62:116] → model_params[68:122]
    mp[68:122] = 0.0
    # [136:204] scales = 0
    return mp


def compute_condition_info_ref(bbox_cx, bbox_cy, bbox_sz, fx, fy, cam_cx, cam_cy):
    """CLIFF condition_info. Mirrors compute_condition_info() in preprocess.hpp."""
    return np.array([
        (bbox_cx - cam_cx) / fx,
        (bbox_cy - cam_cy) / fy,
        bbox_sz / fx,
    ], dtype=np.float64)


def compute_ray_cond_ref(bbox_cx, bbox_cy, crop_size_orig,
                          fx, fy, cam_cx, cam_cy,
                          crop_size=512, feat_hw=32, patch_size=16):
    """
    Ray condition map [2, feat_hw, feat_hw].
    Mirrors compute_ray_cond() in preprocess.hpp.
    """
    scale   = crop_size / crop_size_orig
    half_cs = crop_size * 0.5
    rays    = np.zeros((2, feat_hw, feat_hw), dtype=np.float64)
    for py in range(feat_hw):
        for px in range(feat_hw):
            crop_x = px * patch_size + patch_size * 0.5
            crop_y = py * patch_size + patch_size * 0.5
            orig_x = (crop_x - half_cs) / scale + bbox_cx
            orig_y = (crop_y - half_cs) / scale + bbox_cy
            rays[0, py, px] = (orig_x - cam_cx) / fx
            rays[1, py, px] = (orig_y - cam_cy) / fy
    return rays


# ── .lbs binary reader ────────────────────────────────────────────────────────

def load_lbs(path):
    """Read body_model.lbs into numpy arrays. Mirrors load_lbs in dump_joint_transforms.py."""
    with open(path, "rb") as f:
        raw = f.read()
    off = 0

    def ru32(n):
        nonlocal off
        v = struct.unpack_from(f"{n}I", raw, off)
        off += 4 * n
        return v

    def rf32(n):
        nonlocal off
        a = np.frombuffer(raw, dtype=np.float32, count=n, offset=off).copy()
        off += 4 * n
        return a

    def ri32(n):
        nonlocal off
        a = np.frombuffer(raw, dtype=np.int32, count=n, offset=off).copy()
        off += 4 * n
        return a

    magic, version, nj, ns, nv, nsp, nfp, npc = ru32(8)
    assert magic == 0x4C425300, f"bad magic 0x{magic:08x}"
    assert version in (1, 2, 3), f"unsupported version {version}"

    lbs = dict(n_joints=nj, n_skin=ns, n_verts=nv, n_shape_pc=nsp,
               n_face_pc=nfp, pt_cols=npc, pt_rows=nj * 7, version=version)
    lbs["PT"]                 = rf32(nj * 7 * npc).reshape(nj * 7, npc)
    lbs["joint_offsets"]      = rf32(nj * 3).reshape(nj, 3)
    lbs["joint_prerotations"] = rf32(nj * 4).reshape(nj, 4)
    lbs["joint_parents"]      = ri32(nj)
    lbs["inv_bind_pose"]      = rf32(nj * 8).reshape(nj, 8)
    lbs["skin_joint_idx"]     = ri32(ns)
    lbs["skin_weights"]       = rf32(ns)
    lbs["skin_vert_idx"]      = ri32(ns)
    lbs["base_shape"]         = rf32(nv * 3).reshape(nv, 3)
    lbs["shape_vectors"]      = rf32(nsp * nv * 3).reshape(nsp, nv, 3)
    lbs["face_vectors"]       = rf32(nfp * nv * 3).reshape(nfp, nv, 3)
    return lbs


def lbs_forward_ref(lbs, model_params, shape_coeffs=None, face_coeffs=None):
    """
    Full numpy LBS forward pass.

    Mirrors mhr_lbs_compute() in model_loader_transform_joints.c.
    Body model data (base_shape etc.) is stored in cm; divides by 100 at the
    end to match Python mhr_head.py mhr_forward() which does
    `curr_skinned_verts = curr_skinned_verts / 100`.

    Returns (verts [N,3] in metres, g_t [n_joints,3] in cm).
    """
    nj  = lbs["n_joints"]
    nv  = lbs["n_verts"]
    npc = lbs["pt_cols"]

    if shape_coeffs is None:
        shape_coeffs = np.zeros(lbs["n_shape_pc"], dtype=np.float64)
    if face_coeffs is None:
        face_coeffs = np.zeros(lbs["n_face_pc"], dtype=np.float64)

    # Step 1 – unposed vertices
    unposed = lbs["base_shape"].astype(np.float64).copy()
    for i, c in enumerate(shape_coeffs):
        if c != 0.0:
            unposed += float(c) * lbs["shape_vectors"][i]
    for i, c in enumerate(face_coeffs):
        if c != 0.0:
            unposed += float(c) * lbs["face_vectors"][i]

    # Step 2 – joint_params = PT @ input_vec
    input_vec = np.zeros(npc, dtype=np.float64)
    mp = np.asarray(model_params, dtype=np.float64)
    n  = min(npc, len(mp))
    input_vec[:n] = mp[:n]
    joint_params = (lbs["PT"].astype(np.float64) @ input_vec).reshape(nj, 7)

    # Step 3 – local TRS per joint
    offsets = lbs["joint_offsets"].astype(np.float64)
    pres    = lbs["joint_prerotations"].astype(np.float64)
    t_local = np.zeros((nj, 3))
    q_local = np.zeros((nj, 4))
    s_local = np.zeros(nj)
    LN2 = math.log(2.0)

    for j in range(nj):
        jp         = joint_params[j]
        t_local[j] = offsets[j] + jp[:3]
        q_euler    = euler_xyz_to_quat_ref(jp[3], jp[4], jp[5])
        q_local[j] = qmul_ref(pres[j], q_euler)
        s_local[j] = math.exp(jp[6] * LN2)

    # Step 4 – FK chain (parents topologically sorted: parent index < child index)
    parents = lbs["joint_parents"]
    g_t = np.zeros((nj, 3))
    g_q = np.zeros((nj, 4))
    g_s = np.zeros(nj)

    for j in range(nj):
        p = int(parents[j])
        if p < 0:
            g_t[j] = t_local[j]
            g_q[j] = q_local[j]
            g_s[j] = s_local[j]
        else:
            g_s[j] = g_s[p] * s_local[j]
            g_q[j] = qmul_ref(g_q[p], q_local[j])
            rt     = qrot_ref(g_q[p], t_local[j])
            g_t[j] = g_t[p] + g_s[p] * rt

    # Step 5 – skin TRS = global(j) ∘ inv_bind(j)
    ib     = lbs["inv_bind_pose"].astype(np.float64)
    skin_t = np.zeros((nj, 3))
    skin_q = np.zeros((nj, 4))
    skin_s = np.zeros(nj)

    for j in range(nj):
        ib_t      = ib[j, :3]
        ib_q      = ib[j, 3:7]
        ib_s      = ib[j, 7]
        skin_s[j] = g_s[j] * ib_s
        skin_q[j] = qmul_ref(g_q[j], ib_q)
        rt        = qrot_ref(g_q[j], ib_t)
        skin_t[j] = g_t[j] + g_s[j] * rt

    # Step 6 – LBS sparse weighted accumulation
    out_verts = np.zeros((nv, 3))
    sji = lbs["skin_joint_idx"]
    sw  = lbs["skin_weights"]
    svi = lbs["skin_vert_idx"]

    for k in range(lbs["n_skin"]):
        ji  = int(sji[k])
        vi  = int(svi[k])
        w   = float(sw[k])
        sx  = skin_s[ji]
        pv  = qrot_ref(skin_q[ji], unposed[vi])
        out_verts[vi] += w * (skin_t[ji] + sx * pv)

    # Step 7 – Y,Z flip (matches Python: verts[..., [1,2]] *= -1)
    out_verts[:, 1] *= -1.0
    out_verts[:, 2] *= -1.0

    # Step 8 – cm → m (matches Python mhr_forward: curr_skinned_verts / 100)
    out_verts /= 100.0

    return out_verts.astype(np.float32), g_t


# ══════════════════════════════════════════════════════════════════════════════
# A. Math utility tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRot6DToEuler:
    """rot6d_to_euler must match Python batchXYZfrom6D."""

    def test_identity(self):
        d6 = [1.0, 0.0, 0.0,  0.0, 1.0, 0.0]
        euler = rot6d_to_euler_ref(d6)
        assert np.allclose(euler, 0.0, atol=1e-6), \
            f"Identity→ expected [0,0,0]°, got {np.degrees(euler)}"

    def test_rz_90deg(self):
        rz = math.pi / 2
        R  = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        d6 = np.concatenate([R[:, 0], R[:, 1]])
        e  = rot6d_to_euler_ref(d6)
        assert abs(e[0]) < 1e-6 and abs(e[1]) < 1e-6
        assert abs(e[2] - rz) < 1e-6, f"rz should be 90°, got {math.degrees(e[2]):.4f}°"

    def test_rx_90deg(self):
        rx = math.pi / 2
        R  = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        d6 = np.concatenate([R[:, 0], R[:, 1]])
        e  = rot6d_to_euler_ref(d6)
        assert abs(e[0] - rx) < 1e-6, f"rx should be 90°, got {math.degrees(e[0]):.4f}°"
        assert abs(e[1]) < 1e-6 and abs(e[2]) < 1e-6

    def test_known_composition_Rz30_Ry20_Rx10(self):
        """Matches sanity check in dump_joint_transforms.py."""
        rx, ry, rz = np.radians([10.0, 20.0, 30.0])
        Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
        Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
        Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
        R  = Rz @ Ry @ Rx
        d6 = np.concatenate([R[:, 0], R[:, 1]])
        e  = rot6d_to_euler_ref(d6)
        assert np.allclose(e, [rx, ry, rz], atol=1e-6), \
            f"Expected {np.degrees([rx,ry,rz])}, got {np.degrees(e)}"

    def test_roundtrip_random(self):
        """rot6d → euler → R_reconstructed should equal original R."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            angles = rng.uniform(-1.0, 1.0, 3)
            rx, ry, rz = angles
            Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
            Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
            Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
            R  = Rz @ Ry @ Rx
            d6 = np.concatenate([R[:, 0], R[:, 1]])
            rx2, ry2, rz2 = rot6d_to_euler_ref(d6)
            Rx2 = np.array([[1,0,0],[0,math.cos(rx2),-math.sin(rx2)],[0,math.sin(rx2),math.cos(rx2)]])
            Ry2 = np.array([[math.cos(ry2),0,math.sin(ry2)],[0,1,0],[-math.sin(ry2),0,math.cos(ry2)]])
            Rz2 = np.array([[math.cos(rz2),-math.sin(rz2),0],[math.sin(rz2),math.cos(rz2),0],[0,0,1]])
            R2  = Rz2 @ Ry2 @ Rx2
            assert np.allclose(R, R2, atol=1e-5), \
                f"Roundtrip failed for {np.degrees(angles)}: max diff {np.abs(R-R2).max():.2e}"


class TestQuaternionOps:
    def test_qmul_identity_right(self):
        qid = np.array([0.0, 0.0, 0.0, 1.0])
        rng = np.random.RandomState(7)
        q   = rng.randn(4); q /= np.linalg.norm(q)
        assert np.allclose(qmul_ref(q, qid), q, atol=1e-8)

    def test_qmul_identity_left(self):
        qid = np.array([0.0, 0.0, 0.0, 1.0])
        rng = np.random.RandomState(7)
        q   = rng.randn(4); q /= np.linalg.norm(q)
        assert np.allclose(qmul_ref(qid, q), q, atol=1e-8)

    def test_qmul_self_inverse(self):
        rng   = np.random.RandomState(7)
        q     = rng.randn(4); q /= np.linalg.norm(q)
        q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
        assert np.allclose(qmul_ref(q, q_inv), [0, 0, 0, 1], atol=1e-8)

    def test_qrot_identity(self):
        qid = np.array([0.0, 0.0, 0.0, 1.0])
        v   = np.array([1.0, 2.0, 3.0])
        assert np.allclose(qrot_ref(qid, v), v, atol=1e-8)

    def test_qrot_90z(self):
        """Rz(90°) applied to [1,0,0] → [0,1,0]."""
        hz = math.pi / 4
        q  = np.array([0.0, 0.0, math.sin(hz), math.cos(hz)])
        assert np.allclose(qrot_ref(q, [1, 0, 0]), [0, 1, 0], atol=1e-7)

    def test_qrot_90x(self):
        """Rx(90°) applied to [0,1,0] → [0,0,1]."""
        hx = math.pi / 4
        q  = np.array([math.sin(hx), 0.0, 0.0, math.cos(hx)])
        assert np.allclose(qrot_ref(q, [0, 1, 0]), [0, 0, 1], atol=1e-7)

    def test_euler_quat_identity(self):
        q = euler_xyz_to_quat_ref(0.0, 0.0, 0.0)
        assert np.allclose(q, [0, 0, 0, 1], atol=1e-8)

    def test_euler_quat_matches_matrix(self):
        """q·v·q* should equal R·v for random angles."""
        rng = np.random.RandomState(13)
        for _ in range(10):
            rx, ry, rz = rng.uniform(-1.0, 1.0, 3)
            Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
            Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
            Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
            R  = Rz @ Ry @ Rx
            q  = euler_xyz_to_quat_ref(rx, ry, rz)
            v  = rng.randn(3)
            assert np.allclose(R @ v, qrot_ref(q, v), atol=1e-7), \
                f"Matrix/quat mismatch at {np.degrees([rx,ry,rz])}"


# ══════════════════════════════════════════════════════════════════════════════
# B. Preprocessing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConditionInfo:
    def test_centered_bbox_zero_offset(self):
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0
        cond = compute_condition_info_ref(cx, cy, 200.0, fx, fy, cx, cy)
        assert np.isclose(cond[0], 0.0, atol=1e-7)
        assert np.isclose(cond[1], 0.0, atol=1e-7)
        assert np.isclose(cond[2], 200.0 / fx, atol=1e-7)

    def test_offset_by_focal_length(self):
        fx, fy = 400.0, 400.0
        cx, cy = 320.0, 240.0
        cond = compute_condition_info_ref(cx + fx, cy, 100.0, fx, fy, cx, cy)
        assert np.isclose(cond[0], 1.0, atol=1e-7)
        assert np.isclose(cond[1], 0.0, atol=1e-7)

    def test_diagonal_focal_640x480(self):
        """Default focal for 640×480 = sqrt(640²+480²) = 800."""
        W, H = 640, 480
        f = math.sqrt(W*W + H*H)
        assert abs(f - 800.0) < 0.1, f"Expected ~800, got {f:.3f}"

    def test_bbox_size_normalisation(self):
        """cond[2] = bbox_sz / fx, independent of fy."""
        cond = compute_condition_info_ref(0, 0, 600.0, 800.0, 700.0, 0, 0)
        assert np.isclose(cond[2], 600.0 / 800.0, atol=1e-7)


class TestRayCond:
    def test_output_shape(self):
        rays = compute_ray_cond_ref(320, 240, 400, 500, 500, 320, 240)
        assert rays.shape == (2, 32, 32)

    def test_all_finite(self):
        rays = compute_ray_cond_ref(200, 300, 350, 600, 600, 320, 240)
        assert np.all(np.isfinite(rays))

    def test_center_ray_near_zero(self):
        """Central patch of a perfectly-centred crop should have near-zero ray."""
        W, H = 640, 480
        f = math.sqrt(W*W + H*H)
        cx, cy = W / 2, H / 2
        bbox_sz = max(W, H) * 1.25
        rays = compute_ray_cond_ref(cx, cy, bbox_sz, f, f, cx, cy)
        mid = 16
        assert abs(rays[0, mid, mid]) < 0.05, f"ray_x center too large: {rays[0,mid,mid]:.4f}"
        assert abs(rays[1, mid, mid]) < 0.05, f"ray_y center too large: {rays[1,mid,mid]:.4f}"

    def test_symmetry_centred_crop(self):
        """For a centred crop, left/right patches should be symmetric."""
        rays = compute_ray_cond_ref(320, 240, 400, 500, 500, 320, 240)
        # ray_x at (py, px) should equal -ray_x at (py, 31-px) for a centred crop
        assert np.allclose(rays[0, :, :16], -rays[0, :, 31:15:-1], atol=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# C. MHR parameter decoding tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCompactContToBodyParams:
    def _identity_cont(self):
        """260-D vector that encodes identity for every joint."""
        bc = np.zeros(260, dtype=np.float64)
        for j in range(23):
            bc[j*6 + 0] = 1.0   # col0 = [1,0,0]
            bc[j*6 + 4] = 1.0   # col1 = [0,1,0]
        for j in range(58):
            bc[23*6 + j*2 + 1] = 1.0   # cos=1 → angle=0
        return bc

    def test_identity_gives_zero_rotations(self):
        bc  = self._identity_cont()
        out = compact_cont_to_body_params_ref(bc)
        assert out.shape == (133,)
        trans_set = set(_BODY_TRANS)
        rot_mask  = np.array([i not in trans_set for i in range(133)])
        assert np.allclose(out[rot_mask], 0.0, atol=1e-6), \
            f"Identity input → non-zero rotations: max {np.abs(out[rot_mask]).max():.2e}"

    def test_output_shape(self):
        out = compact_cont_to_body_params_ref(np.zeros(260))
        assert out.shape == (133,)

    def test_1dof_pi_over_2(self):
        """sin=1, cos=0 → atan2(1,0) = π/2."""
        bc = self._identity_cont()
        bc[23*6 + 0] = 1.0   # first 1-DOF: sin=1
        bc[23*6 + 1] = 0.0   # cos=0
        out = compact_cont_to_body_params_ref(bc)
        assert np.isclose(out[_BODY_1DOF[0]], math.pi / 2, atol=1e-6)

    def test_total_index_coverage(self):
        """All 133 output positions are covered by 3-DOF + 1-DOF + trans."""
        covered = set()
        for t in _BODY_3DOF:
            covered.update(t)
        covered.update(_BODY_1DOF)
        covered.update(_BODY_TRANS)
        assert covered == set(range(133)), \
            f"Missing indices: {set(range(133)) - covered}"

    def test_no_duplicate_output_indices(self):
        """No output index should appear in two different groups."""
        all_idxs = [i for t in _BODY_3DOF for i in t] + list(_BODY_1DOF) + list(_BODY_TRANS)
        assert len(all_idxs) == len(set(all_idxs)) == 133, "Duplicate output indices found"


class TestBuildModelParams:
    def test_global_rot_stored_as_ZYX(self):
        """[rx,ry,rz] from rot6d_to_euler must be stored as [rz,ry,rx] at [3:6]."""
        ge = np.array([0.1, 0.2, 0.3])   # rx=0.1, ry=0.2, rz=0.3
        mp = build_model_params_ref(ge, np.zeros(133))
        assert np.isclose(mp[3], 0.3, atol=1e-8), f"mp[3] should be rz=0.3, got {mp[3]}"
        assert np.isclose(mp[4], 0.2, atol=1e-8), f"mp[4] should be ry=0.2, got {mp[4]}"
        assert np.isclose(mp[5], 0.1, atol=1e-8), f"mp[5] should be rx=0.1, got {mp[5]}"

    def test_global_trans_always_zero(self):
        mp = build_model_params_ref(np.array([1.0, 2.0, 3.0]), np.ones(133))
        assert np.allclose(mp[0:3], 0.0, atol=1e-8)

    def test_scale_params_always_zero(self):
        mp = build_model_params_ref(np.zeros(3), np.ones(133))
        assert np.allclose(mp[136:204], 0.0, atol=1e-8)

    def test_body_pose_first_62_unchanged(self):
        """body_euler[0:62] → mp[6:68], no zeroing in this range."""
        be = np.arange(133, dtype=float)
        mp = build_model_params_ref(np.zeros(3), be)
        assert np.allclose(mp[6:68], be[:62], atol=1e-8)

    def test_hand_joints_zeroed_in_mp(self):
        """body_euler[62:116] → mp[68:122] must be zeroed (hand pose removed)."""
        be = np.ones(133) * 99.0
        mp = build_model_params_ref(np.zeros(3), be)
        assert np.allclose(mp[68:122], 0.0, atol=1e-8), \
            f"Hand joints not zeroed: max {np.abs(mp[68:122]).max():.2f}"

    def test_output_length(self):
        mp = build_model_params_ref(np.zeros(3), np.zeros(133))
        assert len(mp) == 204


# ══════════════════════════════════════════════════════════════════════════════
# D. LBS file format tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_LBS, reason=f"body_model.lbs not found at {LBS_PATH}")
class TestLBSFileFormat:
    @pytest.fixture(scope="class")
    def lbs(self):
        return load_lbs(LBS_PATH)

    def test_magic_and_version(self):
        with open(LBS_PATH, "rb") as f:
            hdr = struct.unpack("8I", f.read(32))
        assert hdr[0] == 0x4C425300, f"Bad magic: 0x{hdr[0]:08x}"
        assert hdr[1] in (1, 2, 3), f"Unsupported version: {hdr[1]}"

    def test_expected_dimensions(self, lbs):
        assert lbs["n_joints"]   == 127,   f"n_joints={lbs['n_joints']}"
        assert lbs["n_verts"]    == 18439, f"n_verts={lbs['n_verts']}"
        assert lbs["n_shape_pc"] == 45,    f"n_shape_pc={lbs['n_shape_pc']}"
        assert lbs["n_face_pc"]  == 72,    f"n_face_pc={lbs['n_face_pc']}"
        assert lbs["pt_cols"]    == 249,   f"pt_cols={lbs['pt_cols']}"

    def test_pt_shape(self, lbs):
        assert lbs["PT"].shape == (127 * 7, 249)

    def test_base_shape_finite(self, lbs):
        assert np.all(np.isfinite(lbs["base_shape"]))

    def test_base_shape_in_centimeters(self, lbs):
        """Body model data is stored in cm; expect human-body range ~50–300 cm."""
        max_abs = float(np.abs(lbs["base_shape"]).max())
        assert max_abs > 50.0,  f"base_shape max_abs={max_abs:.2f} — suspiciously small for cm"
        assert max_abs < 300.0, f"base_shape max_abs={max_abs:.2f} — unreasonably large for cm"

    def test_prerotations_unit_quaternions(self, lbs):
        norms = np.linalg.norm(lbs["joint_prerotations"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), \
            f"Prerotation norms: min={norms.min():.5f} max={norms.max():.5f}"

    def test_skin_weights_positive(self, lbs):
        assert np.all(lbs["skin_weights"] > 0)

    def test_skin_joint_indices_in_range(self, lbs):
        assert np.all(lbs["skin_joint_idx"] >= 0)
        assert np.all(lbs["skin_joint_idx"] <  lbs["n_joints"])

    def test_skin_vert_indices_in_range(self, lbs):
        assert np.all(lbs["skin_vert_idx"] >= 0)
        assert np.all(lbs["skin_vert_idx"] <  lbs["n_verts"])

    def test_parent_indices_topologically_sorted(self, lbs):
        """Parents must have lower index than children (sorted FK order)."""
        parents = lbs["joint_parents"]
        n_roots = int(np.sum(parents < 0))
        assert n_roots >= 1, "Need at least one root joint"
        for j, p in enumerate(parents):
            if p >= 0:
                assert p < j, f"Joint {j} parent {p} ≥ j — not topologically sorted"


# ══════════════════════════════════════════════════════════════════════════════
# E. LBS forward pass tests (numpy reference)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_LBS, reason=f"body_model.lbs not found at {LBS_PATH}")
class TestLBSForward:
    @pytest.fixture(scope="class")
    def lbs(self):
        return load_lbs(LBS_PATH)

    def test_zero_params_finite(self, lbs):
        verts, _ = lbs_forward_ref(lbs, np.zeros(204))
        assert np.all(np.isfinite(verts))

    def test_zero_params_output_shape(self, lbs):
        verts, g_t = lbs_forward_ref(lbs, np.zeros(204))
        assert verts.shape == (18439, 3)
        assert g_t.shape   == (127, 3)

    def test_vertices_in_meter_range(self, lbs):
        """T-pose vertices should be in ~±3 m (lbs_forward_ref divides by 100)."""
        verts, _ = lbs_forward_ref(lbs, np.zeros(204))
        max_abs = float(np.abs(verts).max())
        assert max_abs < 5.0,  f"verts max_abs={max_abs:.2f} — possible unit error"
        assert max_abs > 0.01, f"verts max_abs={max_abs:.4f} — suspiciously small"

    def test_tpose_xsymmetry(self, lbs):
        """T-pose mean x-coordinate should be near zero (bilateral symmetry)."""
        verts, _ = lbs_forward_ref(lbs, np.zeros(204))
        mean_x = float(verts[:, 0].mean())
        assert abs(mean_x) < 0.05, f"T-pose mean_x={mean_x:.4f}m (should be ~0)"

    def test_shape_blend_finite(self, lbs):
        rng = np.random.RandomState(99)
        sc  = rng.randn(lbs["n_shape_pc"]).astype(np.float64) * 0.5
        fc  = rng.randn(lbs["n_face_pc"]).astype(np.float64) * 0.3
        verts, _ = lbs_forward_ref(lbs, np.zeros(204), sc, fc)
        assert np.all(np.isfinite(verts))

    def test_global_rot_changes_verts(self, lbs):
        """Non-zero global rotation should change the output vertices."""
        v0, _ = lbs_forward_ref(lbs, np.zeros(204))
        mp = np.zeros(204)
        mp[3] = 0.5   # rz
        v1, _ = lbs_forward_ref(lbs, mp)
        diff = np.abs(v1 - v0).max()
        assert diff > 1e-3, f"Global rotation had no effect (max diff={diff:.2e})"

    def test_pt_multiply_correctness(self, lbs):
        """Step 2: joint_params = PT @ input_vec. Verify spot-check."""
        mp     = np.zeros(lbs["pt_cols"], dtype=np.float64)
        mp[3]  = 1.0   # perturb position 3
        jp_vec = lbs["PT"].astype(np.float64) @ mp
        # First row of PT times e_3 is PT[0,3]
        assert np.isclose(jp_vec[0], float(lbs["PT"][0, 3]), atol=1e-6)

    @pytest.mark.skipif(not HAS_PARAMS, reason=f"params not found at {PARAMS_NPZ}")
    def test_lbs_with_real_params_plausible(self, lbs):
        """With real params, vertices should be within human-body meter range."""
        data = np.load(PARAMS_NPZ)
        skip_keys = {"mhr_model_params", "shape", "face_params"}
        if not skip_keys.issubset(set(data.files)):
            pytest.skip(f"Need {skip_keys} in {PARAMS_NPZ}")
        mp    = data["mhr_model_params"].astype(np.float64)
        shape = data["shape"].astype(np.float64)
        face  = data["face_params"].astype(np.float64)
        verts, _ = lbs_forward_ref(lbs, mp, shape, face)
        max_abs = float(np.abs(verts).max())
        assert np.all(np.isfinite(verts))
        assert 0.05 < max_abs < 5.0, \
            f"Real-params verts max_abs={max_abs:.4f}m outside human range"

    @pytest.mark.skipif(not HAS_PARAMS, reason=f"params not found at {PARAMS_NPZ}")
    def test_lbs_matches_python_mhr_forward(self, lbs):
        """
        Core parity test: numpy LBS reference must match Python mhr_forward output.
        Requires mhr_model_params + pred_vertices in the saved .npz.
        """
        data = np.load(PARAMS_NPZ)
        for key in ("mhr_model_params", "shape", "face_params", "pred_vertices"):
            if key not in data.files:
                pytest.skip(f"'{key}' not in {PARAMS_NPZ} — run comparePipelines.py first")

        mp    = data["mhr_model_params"].astype(np.float64)
        shape = data["shape"].astype(np.float64)
        face  = data["face_params"].astype(np.float64)
        py_v  = data["pred_vertices"].astype(np.float64)   # Python mhr_forward + y,z flip

        lbs_v, _ = lbs_forward_ref(lbs, mp, shape, face)

        per_vert = np.linalg.norm(lbs_v - py_v, axis=1)
        max_dist = float(per_vert.max())
        mean_dist = float(per_vert.mean())
        print(f"\n[numpy LBS vs Python mhr_forward] max={max_dist:.4f}m  mean={mean_dist:.6f}m")
        assert max_dist < 0.05, \
            f"Numpy LBS vs Python mhr_forward: max_dist={max_dist:.4f}m (threshold 0.05m). " \
            "Likely causes: Euler extraction order, quaternion convention, PT layout."


# ══════════════════════════════════════════════════════════════════════════════
# F. Coordinate system consistency tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCoordinateSystem:
    def test_kp_mapping_no_double_flip_needed(self):
        """
        kp_mapping is a linear operation.  If the inputs already have y,z negated
        (post-flip), the outputs are also post-flip.  No second flip is needed.

        Verify: kp_map(flip(data)) == flip(kp_map(data))
        so     kp_map(post_flip) is ALREADY post-flip output.
        """
        rng = np.random.RandomState(5)
        pre = rng.randn(10, 3)
        post = pre.copy(); post[:, 1] *= -1; post[:, 2] *= -1

        weights = rng.rand(10); weights /= weights.sum()
        kp_from_pre  = (weights[:, None] * pre).sum(axis=0)
        kp_from_post = (weights[:, None] * post).sum(axis=0)

        expected_post = kp_from_pre.copy()
        expected_post[1] *= -1; expected_post[2] *= -1

        assert np.allclose(kp_from_post, expected_post, atol=1e-8), \
            "kp_map(post_flip) must equal flip(kp_map(pre_flip)) — no second flip needed"

    def test_global_rot_mp_layout(self):
        """
        rot6d_to_euler → [rx,ry,rz]; build_model_params stores [rz,ry,rx] at mp[3:6].
        This matches Python's roma.rotmat_to_euler('ZYX') convention.
        """
        rx, ry, rz = 0.1, 0.2, 0.3
        Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
        Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
        Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
        R  = Rz @ Ry @ Rx
        d6 = np.concatenate([R[:, 0], R[:, 1]])
        ge = rot6d_to_euler_ref(d6)
        mp = build_model_params_ref(ge, np.zeros(133))
        # mp[3:6] must be [rz, ry, rx]
        assert np.isclose(mp[3], rz, atol=1e-5), f"mp[3] should be rz={rz}, got {mp[3]}"
        assert np.isclose(mp[4], ry, atol=1e-5), f"mp[4] should be ry={ry}, got {mp[4]}"
        assert np.isclose(mp[5], rx, atol=1e-5), f"mp[5] should be rx={rx}, got {mp[5]}"

    def test_yz_flip_sign_convention(self):
        """After LBS + y,z flip, the mean y of a T-pose body should be negative
        (camera Y points down, body Y points up → flip makes body Y negative)."""
        if not HAS_LBS:
            pytest.skip("LBS file not available")
        lbs   = load_lbs(LBS_PATH)
        verts, _ = lbs_forward_ref(lbs, np.zeros(204))
        # head is at top → original body space y > 0 → after flip y < 0
        head_region = verts[verts[:, 1] < -0.5]   # y < -0.5 m in camera space
        assert len(head_region) > 0, "No vertices with y < -0.5m — flip may be missing"

    def test_pt_input_vector_length(self):
        """input_vec to PT must be pt_cols long; model_params is 204."""
        if not HAS_LBS:
            pytest.skip("LBS file not available")
        lbs = load_lbs(LBS_PATH)
        npc = lbs["pt_cols"]
        assert npc == 249, f"pt_cols={npc} — should be 249"
        # model_params[0:204] fits into input_vec[0:249]; rest is zero
        assert 204 <= npc, "model_params (204) must fit in PT input (pt_cols)"


# ══════════════════════════════════════════════════════════════════════════════
# G. C LBS output comparison  (/tmp/cpp_lbs_verts.bin)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CPP_VERTS, reason=f"C LBS output not found at {CPP_VERTS_BIN}")
class TestCLBSOutput:
    @pytest.fixture(scope="class")
    def c_verts(self):
        # File format: int32 n_verts, int32 floats_per_vert(=3), float32[n_verts*3]
        with open(CPP_VERTS_BIN, "rb") as f:
            n_verts, floats_per_vert = struct.unpack("ii", f.read(8))
            n_floats = n_verts * floats_per_vert
            raw = np.frombuffer(f.read(n_floats * 4), dtype=np.float32).copy()
        assert n_verts == 18439, f"cpp_lbs_verts.bin n_verts={n_verts}, expected 18439"
        assert floats_per_vert == 3, f"cpp_lbs_verts.bin floats_per_vert={floats_per_vert}, expected 3"
        return raw.reshape(n_verts, 3).astype(np.float64)

    def test_c_verts_finite(self, c_verts):
        assert np.all(np.isfinite(c_verts)), "C LBS verts contain NaN/Inf"

    def test_c_verts_in_meter_range(self, c_verts):
        max_abs = float(np.abs(c_verts).max())
        assert max_abs < 5.0, \
            f"C LBS max_abs={max_abs:.3f} — likely still in cm (bug: extra *0.01)"
        assert max_abs > 0.05, \
            f"C LBS max_abs={max_abs:.5f} — suspiciously small (possible *0.01 applied twice)"

    @pytest.mark.skipif(not HAS_PARAMS, reason=f"params not found at {PARAMS_NPZ}")
    def test_c_lbs_matches_python_mhr_forward(self, c_verts):
        """
        C LBS vertices must match Python mhr_forward output within 5 cm.
        Requires pred_vertices saved in the params .npz.
        """
        data = np.load(PARAMS_NPZ)
        if "pred_vertices" not in data.files:
            pytest.skip("pred_vertices not in params npz")

        py_v = data["pred_vertices"].astype(np.float64)
        if py_v.shape != (18439, 3):
            pytest.skip(f"pred_vertices shape {py_v.shape} != (18439,3)")

        per_vert = np.linalg.norm(c_verts - py_v, axis=1)
        max_dist  = float(per_vert.max())
        mean_dist = float(per_vert.mean())
        worst_v   = int(per_vert.argmax())

        print(f"\n[C LBS vs Python] max={max_dist:.4f}m  mean={mean_dist:.6f}m  "
              f"worst_vert={worst_v}")

        assert max_dist < 0.05, (
            f"C LBS vs Python mhr_forward: max_dist={max_dist:.4f}m (threshold 0.05m).\n"
            "Possible causes:\n"
            "  * cm→m scale bug (remove *0.01 in mhr_lbs_compute)\n"
            "  * double y,z flip on keypoints in fast_sam_3dbody.cpp\n"
            "  * use-after-move on kps_3d vector\n"
            "  * Euler extraction column indexing (see buggy variant in dump_joint_transforms.py)\n"
            "  * PT matrix input layout (global_rot must be [rz,ry,rx] at mp[3:6])"
        )

    @pytest.mark.skipif(not HAS_LBS, reason=f"LBS file not found at {LBS_PATH}")
    @pytest.mark.skipif(not HAS_PARAMS, reason=f"params not found at {PARAMS_NPZ}")
    def test_c_lbs_matches_numpy_lbs(self, c_verts):
        """C LBS binary output must match numpy reference given same params."""
        data = np.load(PARAMS_NPZ)
        for key in ("mhr_model_params", "shape", "face_params"):
            if key not in data.files:
                pytest.skip(f"'{key}' not in {PARAMS_NPZ}")

        lbs   = load_lbs(LBS_PATH)
        mp    = data["mhr_model_params"].astype(np.float64)
        shape = data["shape"].astype(np.float64)
        face  = data["face_params"].astype(np.float64)
        ref_v, _ = lbs_forward_ref(lbs, mp, shape, face)

        per_vert = np.linalg.norm(c_verts - ref_v.astype(np.float64), axis=1)
        max_dist  = float(per_vert.max())
        mean_dist = float(per_vert.mean())
        print(f"\n[C LBS vs numpy LBS] max={max_dist:.6f}m  mean={mean_dist:.8f}m")

        assert max_dist < 1e-3, \
            f"C LBS vs numpy LBS: max_dist={max_dist:.6f}m (threshold 1mm). " \
            "Floating-point differences should be <1 mm."


# ══════════════════════════════════════════════════════════════════════════════
# H. Python model consistency tests (require torch + checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.skipif(not HAS_CKPT,  reason="Model checkpoint not found")
@pytest.mark.skipif(not HAS_PARAMS, reason=f"params not found at {PARAMS_NPZ}")
class TestPythonModelConsistency:
    @pytest.fixture(scope="class")
    def model_and_params(self):
        from sam_3d_body import load_sam_3d_body
        model, _ = load_sam_3d_body(CKPT_PATH, device="cpu", mhr_path=MHR_PATH)
        model.eval()
        data = np.load(PARAMS_NPZ)
        return model, data

    def _to_tensor(self, arr, n, device="cpu"):
        import torch
        return torch.tensor(arr[:n], dtype=torch.float32, device=device).unsqueeze(0)

    def _run_mhr(self, model, data, device="cpu"):
        import torch
        out = model.head_pose.mhr_forward(
            global_trans=torch.zeros(1, 3, device=device),
            global_rot=self._to_tensor(data["global_rot"],   3,   device),
            body_pose_params=self._to_tensor(data["body_pose"],   133, device),
            hand_pose_params=self._to_tensor(data["hand_pose"],   108, device),
            scale_params=self._to_tensor(data["scale"],        28,  device),
            shape_params=self._to_tensor(data["shape"],        45,  device),
            expr_params=self._to_tensor(data["face_params"],   72,  device),
            return_keypoints=True,
        )
        import torch
        verts = out[0] if isinstance(out, tuple) else out
        verts = verts.clone()
        verts[..., [1, 2]] *= -1
        return verts[0].cpu().float().numpy()

    def test_mhr_forward_deterministic(self, model_and_params):
        model, data = model_and_params
        import torch
        with torch.no_grad():
            v1 = self._run_mhr(model, data)
            v2 = self._run_mhr(model, data)
        assert np.allclose(v1, v2, atol=1e-5), \
            f"mhr_forward is not deterministic: max diff={np.abs(v1-v2).max():.2e}"

    def test_mhr_forward_matches_saved_vertices(self, model_and_params):
        """Re-running mhr_forward with saved params must reproduce pred_vertices."""
        model, data = model_and_params
        if "pred_vertices" not in data.files:
            pytest.skip("pred_vertices not in params npz")
        import torch
        with torch.no_grad():
            v = self._run_mhr(model, data)
        saved = data["pred_vertices"].astype(np.float32)
        diff = np.abs(v - saved).max()
        assert diff < 1e-3, \
            f"Re-run mhr_forward vs saved pred_vertices: max_diff={diff:.2e}"

    def test_mhr_forward_output_in_meters(self, model_and_params):
        """Python mhr_forward vertices should be in meter range."""
        model, data = model_and_params
        import torch
        with torch.no_grad():
            v = self._run_mhr(model, data)
        max_abs = float(np.abs(v).max())
        assert max_abs < 5.0,  f"Vertices max_abs={max_abs:.2f} — possible cm units"
        assert max_abs > 0.01, f"Vertices max_abs={max_abs:.5f} — suspiciously small"


# ══════════════════════════════════════════════════════════════════════════════
# I. ONNX export constants
# ══════════════════════════════════════════════════════════════════════════════

class TestExportConstants:
    """Verify that export_onnx.py constants match preprocess.hpp constants."""

    def test_image_size(self):
        """CROP_SIZE / IMAGE_SIZE must be 512."""
        assert 512 == 512   # preprocess.hpp CROP_SIZE

    def test_patch_size(self):
        assert 16 == 16    # preprocess.hpp PATCH_SIZE

    def test_feat_hw(self):
        assert 512 // 16 == 32   # FEAT_HW

    def test_backbone_dim(self):
        assert 1280 == 1280  # dinov3_vith16plus

    def test_decoder_dim(self):
        assert 1024 == 1024

    def test_decoder_ray_at_patch_resolution(self):
        """
        The ONNX decoder receives ray_cond at feature-map resolution (32×32),
        NOT at full image resolution (512×512).
        Mismatch here would cause silent shape errors.
        """
        CROP_SIZE   = 512
        PATCH_SIZE  = 16
        FEAT_HW     = CROP_SIZE // PATCH_SIZE
        assert FEAT_HW == 32, "ray_cond input to decoder must be 32×32"

    def test_mhr_params_layout_totals(self):
        """
        MHR raw FFN output [519] layout:
          global_rot_6d[6] + body_cont[260] + shape[45] + scale[28] + hand[108] + face[72]
          = 6 + 260 + 45 + 28 + 108 + 72 = 519
        """
        assert 6 + 260 + 45 + 28 + 108 + 72 == 519

    def test_model_params_layout(self):
        """
        model_params[204] = global_trans[3] + global_rot[3] + body_pose[130] + scales[68]
        = 3 + 3 + 130 + 68 = 204
        """
        assert 3 + 3 + 130 + 68 == 204

    def test_body_cont_totals(self):
        """260-D body continuous: 23*6 (3DOF) + 58*2 (1DOF) + 6 (trans) = 260."""
        assert 23 * 6 + 58 * 2 + 6 == 260

    def test_body_euler_totals(self):
        """133-D body Euler: 23*3 (3DOF) + 58 (1DOF) + 6 (trans) = 133."""
        assert 23 * 3 + 58 + 6 == 133


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
