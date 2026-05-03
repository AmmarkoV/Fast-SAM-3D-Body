# C++ Pipeline — 3D Transformations, Math, and Discrepancies vs MATH.md

This document traces every transformation in the C++ implementation (`fast_sam_3dbody_cpp/`)
and highlights where it **matches** or **diverges** from the Python reference (MATH.md).

---

## Table of Contents

1. [C++ Pipeline Architecture](#1-c-pipeline-architecture)
2. [Image Preprocessing](#2-image-preprocessing)
3. [CLIFF Condition and Ray Condition](#3-cliff-condition-and-ray-condition)
4. [MHR Head FFN (CPU)](#4-mhr-head-ffn-cpu)
5. [Camera Parameter Conversion (C++)](#5-camera-parameter-conversion-c)
6. [Body Model: LBS Path](#6-body-model-lbs-path)
7. [Body Model: ONNX Path](#7-body-model-onnx-path)
8. [Model Parameters Assembly](#8-model-parameters-assembly)
9. [Keypoint Extraction](#9-keypoint-extraction)
10. [2D Keypoint Projection](#10-2d-keypoint-projection)
11. [YOLO Skeleton Drawing](#11-yolo-skeleton-drawing)
12. [OpenGL Rendering Pipeline](#12-opengl-rendering-pipeline)
13. [Discrepancy Summary](#13-discrepancy-summary)

---

## 1. C++ Pipeline Architecture

```
BGR uint8 image (OpenCV Mat)
    |
    v
[YOLO Pose Detection]              -> PersonDet[], keypoints_yolo[17*3]
    |
    v
[crop_and_normalise per person]    -> CHW float32 [3, 512, 512], ImageNet norm
    |
    v
[compute_condition_info]           -> CLIFF cond[3]
[compute_ray_cond]                 -> ray directions [2, 32, 32]
    |
    v
[backbone.onnx]                    -> features [B, 1280, 32, 32]
    |
    v
[decoder.onnx]                     -> pose_tokens [B, 1024]
    |
    v
[cffn_run(mhr_ffn)]                -> mhr_raw [B, 519]
[cffn_run(cam_ffn)]                -> cam_raw [B, 3]
    |
    +-- rot6d_to_euler()           -> global_rot_euler [3]
    +-- compact_cont_to_body_params() -> body_euler [133]
    +-- build_model_params()       -> model_params [204]
    |
    +-- [LBS path] mhr_lbs_compute()  -> verts [18439*3], joints [127*3] (meters, Y,Z flipped)
    +-- [ONNX path] body_model.onnx   -> verts [18439*3] (cm), skel [127*8] (cm)
    |
    v
[Post-processing]                  -> Y,Z flip, cm->m, keypoint mapping
    |
    v
[2D projection per keypoint]       -> keypoints_2d [70*2]
    |
    v
MHRResult[] returned to caller
```

**Source files:**
- Pipeline: `src/fast_sam_3dbody.cpp` (lines 520-993)
- Preprocessing: `src/preprocess.hpp`
- LBS: `GraphicsEngine/ModelLoader/model_loader_transform_joints.c` (lines 1221-1491)
- Rendering: `render/fast_sam_3dbody_render.cpp`
- Camera matrices: `render/mhr_pose_driver.h`

---

## 2. Image Preprocessing

### 2A. Bounding Box to Square Crop

```cpp
float cx   = (bbox_x1 + bbox_x2) * 0.5f;
float cy   = (bbox_y1 + bbox_y2) * 0.5f;
float bw   = bbox_x2 - bbox_x1;
float bh   = bbox_y2 - bbox_y1;
float side = std::max(bw, bh);
```

A square region centered on the bbox is extracted. If the crop extends past the
image boundary, the overflow region is padded with grey (114, 114, 114).

### 2B. Resize and Normalize

The cropped region is resized to 512x512, then converted:

```cpp
// BGR -> RGB, uint8 -> float32, HWC -> CHW
for each pixel:
    r = row[3*x + 2] / 255.f
    g = row[3*x + 1] / 255.f
    b = row[3*x + 0] / 255.f

    out_chw[0 * plane + y * 512 + x] = (r - 0.485) / 0.229
    out_chw[1 * plane + y * 512 + x] = (g - 0.456) / 0.224
    out_chw[2 * plane + y * 512 + x] = (b - 0.406) / 0.225
```

**MATCHES MATH.md** — Same ImageNet normalization as Python.

**Source:** `preprocess.hpp` lines 31-95.

---

## 3. CLIFF Condition and Ray Condition

### 3A. CLIFF Condition Info

```cpp
cond[0] = (bbox_cx - cam_cx) / focal_x;
cond[1] = (bbox_cy - cam_cy) / focal_y;
cond[2] = bbox_size           / focal_x;
```

Where `cam_cx = cx` (principal point X), `cam_cy = cy` (principal point Y).

**MATCHES MATH.md Section 4B** — Uses `USE_INTRIN_CENTER=true` path from Python
config. Normalizes the bbox offset and size by focal length.

**Source:** `preprocess.hpp` lines 104-114.

### 3B. Ray Condition Map

For each patch center in the 32x32 feature grid:

```cpp
crop_x = px * 16 + 8    // patch center in 512x512 crop
crop_y = py * 16 + 8
scale  = 512.0f / crop_size_orig
orig_x = (crop_x - 256) / scale + bbox_cx  // map back to original image
ray_x  = (orig_x - cam_cx) / focal_x
ray_y  = (orig_y - cam_cy) / focal_y
```

Output layout: `[2, 32, 32]` — channel 0 = ray_x, channel 1 = ray_y.

**Source:** `preprocess.hpp` lines 131-153.

---

## 4. MHR Head FFN (CPU)

### 4A. FFN Architecture

```
y = relu(x @ W0.T + b0) @ W1.T + b1
```

Two-layer FFN: `Linear(in, hid) + ReLU + Linear(hid, out)`.
Weights are loaded from GGUF in f16 and converted to f32 at load time.

Storage is row-major: `w0[i * in_dim + j]` = weight from input j to hidden i.

### 4B. MHR Raw Output Parsing

The 519-dim output is parsed identically to MATH.md Section 3A:

```
Offset  Size  Meaning
0       6     Global rotation (6D)
6       260   Body continuous
266     45    Shape
311     28    Scale
339     108   Hand pose
447     72    Face (zeroed)
```

### 4C. 6D Rotation to Euler ZYX

```cpp
// Gram-Schmidt orthonormalization of two column vectors
a = normalize(d6[0:3])    // column 0 of R
b = d6[3:6] - dot(a, d6[3:6]) * a
b = normalize(b)          // column 1 of R

// ZYX Euler extraction from R columns:
// R = [col0 | col1 | col2] where col_k = [R[0,k], R[1,k], R[2,k]]
rx = atan2(R[2,1], R[2,2])     // = atan2(e12, e22)
ry = asin(-R[2,0])              // = asin(-e02)
rz = atan2(R[1,0], R[0,0])     // = atan2(e01, e00)
```

Output: `[rx, ry, rz]`.

**MATCHES MATH.md Section 3B** — Same Gram-Schmidt + ZYX extraction.
Note: only R[2,0], R[2,1], R[2,2], R[1,0], R[0,0] are needed. The third
column of R is computed only partially (R[2,2] only).

### 4D. Body Continuous to Euler

```cpp
// 3-DOF joints: 23 joints, each with 6D rotation -> Euler ZYX
for j in 0..22:
    euler[3] = rot6d_to_euler(body_cont[j*6 : j*6+6])
    body_euler[BODY_3DOF_JOINT_IDXS[j]] = euler

// 1-DOF joints: 58 joints, each with (sin, cos) -> atan2
for j in 0..57:
    body_euler[BODY_1DOF_IDXS[j]] = atan2(sin, cos)

// Translation: 6 values copied directly
for j in 0..5:
    body_euler[BODY_TRANS_IDXS[j]] = body_cont[138 + 116 + j]
```

**MATCHES MATH.md Section 3C** — Same decomposition: 138 + 116 + 6 = 260.

**Source:** `preprocess.hpp` lines 264-341.

---

## 5. Camera Parameter Conversion (C++)

### 5A. Weak Perspective to Strong Perspective

```cpp
// fast_sam_3dbody.cpp lines 847-858
float s_val   = -cam[0];     // sign flip
float tx      =  cam[1];
float ty      = -cam[2];     // sign flip
float bw      = d.x2 - d.x1;
float bh      = d.y2 - d.y1;
float bbox_cx = (d.x1 + d.x2) * 0.5f;
float bbox_cy = (d.y1 + d.y2) * 0.5f;
float bs      = std::max(bw, bh) * 1.25f * s_val + 1e-8f;
float tz      = 2.0f * fx / bs;
float cx_off  = 2.0f * (bbox_cx - cx) / bs;
float cy_off  = 2.0f * (bbox_cy - cy) / bs;
r.pred_cam_t  = { tx + cx_off, ty + cy_off, tz };
```

**MATCHES MATH.md Section 4B** — Same CLIFF formulation:
- `bbox_size` = `max(bw, bh)` (the square crop side length)
- `default_scale_factor` = 1.25 (hardcoded, same as Python)
- `cx, cy` = principal point from camera intrinsics

### 5B. Camera Intrinsics

```cpp
// fast_sam_3dbody.cpp lines 532-535
float fx = (cfg.focal_x > 0) ? cfg.focal_x : float(W);
float fy = (cfg.focal_y > 0) ? cfg.focal_y : float(W);
float cx = (cfg.principal_x > 0) ? cfg.principal_x : float(W) * 0.5f;
float cy = (cfg.principal_y > 0) ? cfg.principal_y : float(H) * 0.5f;
```

Default: `fx = fy = image_width`, `cx = W/2`, `cy = H/2`.

**Source:** `fast_sam_3dbody.cpp` lines 532-535, 847-858.

---

## 6. Body Model: LBS Path

The native C LBS implementation replicates the MHR JIT model entirely in C.
It is used when `body_model.onnx` is unavailable but `body_model.lbs` exists.

### 6A. Step 1 — Unposed Vertices

```
unposed[v] = base_shape[v]
           + sum(shape_coeffs[i] * shape_vectors[i][v])
           + sum(face_coeffs[i]  * face_vectors[i][v])
```

Output: `[18439, 3]` in **centimeters**, MHR internal coordinate system (Y-up, Z-back).

### 6B. Step 2 — Joint Parameters via PT Matrix

```
joint_params = PT @ model_params[:pt_cols]
```

Where:
- `PT` = `[pt_rows x pt_cols]` sparse linear mapping (889 rows x 249 cols)
- `pt_rows = n_joints * 7 = 127 * 7 = 889`
- Each joint row: `[tx, ty, tz, rx, ry, rz, log2_scale]`

### 6C. Step 3 — Local TRS per Joint

```cpp
t_local[j] = joint_offsets[j] + joint_params[j][0:3]
q_local[j] = joint_prerotations[j] * euler_to_quat(rx, ry, rz)
s_local[j] = exp2(joint_params[j][6])
```

The Euler-to-quaternion conversion uses `R = Rz(rz) * Ry(ry) * Rx(rx)` (ZYX intrinsic).
Pre-rotation establishes the rest-frame orientation; the pose quaternion composes on top.

### 6D. Step 4 — Forward Kinematics

```cpp
// Parent-first traversal (joint_parents guarantees parent index < child index)
if (joint is root):
    g_t[j] = t_local[j]
    g_q[j] = q_local[j]
    g_s[j] = s_local[j]
else:
    p = joint_parents[j]
    g_s[j] = g_s[p] * s_local[j]
    g_q[j] = g_q[p] * q_local[j]        // R_global = R_parent @ R_local
    g_t[j] = g_t[p] + g_s[p] * rotate(g_q[p], t_local[j])
```

### 6E. Step 5 — Skin TRS (Deformation)

```cpp
// inv_bind_pose[j] = [tx, ty, tz, qx, qy, qz, qw, scale]
skin_q[j] = g_q[j] * inv_bind_q[j]
skin_t[j] = g_t[j] + g_s[j] * rotate(g_q[j], inv_bind_t[j])
skin_s[j] = g_s[j] * inv_bind_scale[j]
```

### 6F. Step 6 — Linear Blend Skinning

```cpp
for each vertex vi:
    out_verts[vi] = sum over skinning entries k for vi:
        w_k * (skin_t[ji_k] + skin_s[ji_k] * rotate(skin_q[ji_k], unposed[vi]))
```

Where `w_k` are skinning weights, `ji_k` is the joint index, and each vertex
is influenced by up to 4 joints.

### 6G. Step 7 — Y,Z Flip

```cpp
for v in 0..18438:
    out_verts[v*3+1] *= -1.f   // Y flip
    out_verts[v*3+2] *= -1.f   // Z flip
```

**This is the same Y,Z sign flip from MATH.md Section 3I.**
After this, vertices are in **camera-adjacent space**.

### 6H. Step 8 — Centimeters to Meters

```cpp
for v in 0..18438*3:
    out_verts[v] *= 0.01f
```

**MATCHES MATH.md Section 3G** — Same 0.01 scale factor.

### 6I. Step 9 — Joint Coordinates Output

```cpp
if (out_joints):
    for j in 0..126:
        out_joints[j*3+0] =  g_t[j*3+0] * 0.01f  // X in meters
        out_joints[j*3+1] = -g_t[j*3+1] * 0.01f  // Y flipped, in meters
        out_joints[j*3+2] = -g_t[j*3+2] * 0.01f  // Z flipped, in meters
```

Joint coordinates are transformed the same way as vertices: cm->meters + Y,Z flip.

**Source:** `model_loader_transform_joints.c` lines 1288-1491.

---

## 7. Body Model: ONNX Path

When `body_model.onnx` is available, the C++ pipeline uses ONNX Runtime for skinning.

### 7A. ONNX Model Input

```
shape_params[B, 45]
body_params[B, 204]    // from build_model_params()
face_params[B, 72]
apply_correctives = False
```

### 7B. ONNX Model Output

```
verts[B, 18439, 3]     // in centimeters, MHR internal space
skel_state[B, 127, 8]  // [joint_x, joint_y, joint_z, quat_x..w, padding]
```

### 7C. Post-Processing in C++

```cpp
// fast_sam_3dbody.cpp lines 902-912
// Flip Y,Z on vertices
for k in 0..18438:
    pred_vertices[k*3 + 1] *= -1.f
    pred_vertices[k*3 + 2] *= -1.f

// Extract joint coords from skel_state and convert cm->meters
// fast_sam_3dbody.cpp lines 926-941
for j in 0..126:
    joint_coords[j*3+0] = skel[j*8+0] * 0.01f
    joint_coords[j*3+1] = skel[j*8+1] * 0.01f
    joint_coords[j*3+2] = skel[j*8+2] * 0.01f
// Flip Y,Z on joints
    joint_coords[j*3+1] *= -1.f
    joint_coords[j*3+2] *= -1.f
```

**MATCHES MATH.md** — Same cm->meters conversion and Y,Z flip as Python reference.

**Source:** `fast_sam_3dbody.cpp` lines 902-941.

---

## 8. Model Parameters Assembly

### 8A. Layout

```
Offset  Size  Meaning
------  ----  -------
0       3     Global translation (zeroed, * 10 for cm scale)
3       3     Global rotation (Euler ZYX, but [rz,ry,rx] order)
6       130   Body pose (first 130 of 133 Euler joints, hand joints zeroed)
136     68    Scales (zeroed)
```

### 8B. Global Rotation Order Swap

```cpp
// preprocess.hpp lines 413-415
out.data[3] = global_rot_euler[2];  // rz (Z angle)
out.data[4] = global_rot_euler[1];  // ry (Y angle)
out.data[5] = global_rot_euler[0];  // rx (X angle)
```

`rot6d_to_euler()` returns `[rx, ry, rz]`. The PT matrix was trained with the
Python `roma.rotmat_to_euler("ZYX")` convention which returns `[rz, ry, rx]`.
Hence the swap: `[rx, ry, rz] -> [rz, ry, rx]`.

**MATCHES MATH.md** — The Python code does `rotmat_to_euler_ZYX(R)` which
returns `[rz, ry, rx]`. The C++ code computes `[rx, ry, rz]` and swaps to match.

### 8C. Hand Joint Zeroing

```cpp
// Zero hand joint params (indices 68-121 in model_params)
for i in 68..121:
    out.data[i] = 0.0f
```

These correspond to body pose indices 62-115 (offset by 6 for global_trans + global_rot).

**MATCHES MATH.md Section 3C** — `pred_pose_euler[:, mhr_param_hand_idxs] = 0`.

### 8D. Scale Zeroing

```cpp
// [136:204] = scales (zeroed)
// Already zeroed by default initialization
```

**DISCREPANCY:** The Python reference decompresses scales via PCA:
```python
scales = scale_mean + scale_params @ scale_comps
```
The C++ code zeros all 68 scale values. This means the body proportions
(slim vs broad, tall vs short) are not driven by the predicted scale parameters
in the C++ ONNX path. The LBS path also zeros scales in `build_model_params()`
but the render loop can optionally decode them from `lbs->scale_mean` and
`lbs->scale_comps` (see `fast_sam_3dbody_render.cpp` lines 522-531).

**Source:** `preprocess.hpp` lines 383-433.

---

## 9. Keypoint Extraction

### 9A. Sparse Keypoint Mapping

```cpp
// fast_sam_3dbody.cpp lines 944-965
const float* verts_ptr = pred_vertices.data();  // already Y,Z flipped
const float* joints_ptr = joint_coords.data();   // already Y,Z flipped

for each entry in kp_mapping:
    for c in 0..2:  // x, y, z
        row = entry.row * 3 + c
        col = entry.col
        if col < 18439:
            src_val = verts_ptr[col * 3 + c]
        else:
            src_val = joints_ptr[(col - 18439) * 3 + c]
        kps_3d[row] += src_val * entry.val
```

### 9B. LBS vs ONNX Path — Pre-flip State

**LBS path:** `pred_vertices` and `joint_coords` are already Y,Z-flipped by
`mhr_lbs_compute()` (step 7). No additional flip in the pipeline.

**ONNX path:** `pred_vertices` is flipped at line 908-912. `joint_coords` is
flipped at line 937-941. Both are in camera-adjacent space.

### 9C. Post-Extraction Flip

```cpp
// fast_sam_3dbody.cpp lines 967-972
for k in 0..69:
    kps_3d[k*3 + 1] *= -1.f  // Y flip
    kps_3d[k*3 + 2] *= -1.f  // Z flip
```

**DISCREPANCY — DOUBLE FLIP BUG:** The keypoints are extracted from vertices
and joints that are **already** Y,Z-flipped (camera-adjacent space). Flipping
them again undoes the flip, putting keypoints back into **MHR meter space**
(pre-flip convention: Y-up, Z-back).

In the Python reference (MATH.md Section 3H-3I):
1. Keypoints are extracted from **unflipped** vertices/joints (MHR meter space)
2. Then flipped **once** to camera-adjacent space

The C++ code:
1. Vertices/joints are **already flipped** (by LBS step 7 or by post-processing)
2. Keypoints extracted from already-flipped data = camera-adjacent space
3. Then flipped **again** = back to MHR meter space

**Result:** `keypoints_3d` in `MHRResult` are in MHR meter space (Y-up, Z-back)
instead of camera-adjacent space (Y-down, Z-forward). This causes the 2D
keypoint projection to produce wrong pixel coordinates, and the skeleton will
not overlay correctly on the image.

**Fix:** Remove the Y,Z flip at lines 967-972, or do not flip vertices/joints
before keypoint extraction.

**Source:** `fast_sam_3dbody.cpp` lines 944-974.

---

## 10. 2D Keypoint Projection

### 10A. Projection Formula

```cpp
// fast_sam_3dbody.cpp lines 977-987
for k in 0..69:
    dz = kps_3d[k*3 + 2] + pred_cam_t[2]
    dx = kps_3d[k*3 + 0] + pred_cam_t[0]
    dy = kps_3d[k*3 + 1] + pred_cam_t[1]
    if dz < 1e-4: dz = 1e-4
    kps_2d[k*2 + 0] = dx / dz * fx + cx
    kps_2d[k*2 + 1] = dy / dz * fx + cy
```

### 10B. Expected Formula (from MATH.md)

```
pixel_x = fx * (X + tx) / (Z + tz) + cx_img
pixel_y = fy * (Y + ty) / (Z + tz) + cy_img
```

### 10C. Discrepancies

**fx vs fy:** The C++ code uses `fx` for both the X and Y projection. If
`fx != fy` (different horizontal/vertical focal lengths), the Y projection
will be wrong. The Python reference uses `cam_int[:, :2, :2].diagonal()` which
extracts `fx` and `fy` separately.

In practice, the default is `fx = fy = image_width`, so this only matters if
the user configures asymmetric focal lengths.

**Source:** `fast_sam_3dbody.cpp` lines 977-987.

---

## 11. YOLO Skeleton Drawing

### 11A. COCO Joint Pairs

```cpp
static const int COCO_PAIRS[][2] = {
    {0,1},{0,2},{1,3},{2,4},                          // head
    {5,6},{5,7},{7,9},{6,8},{8,10},                   // arms
    {5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16} // torso+legs
};
```

17 links connecting 17 COCO joints. Drawn with orange lines and yellow dots.

### 11B. Drawing

```cpp
cv::line(img, {(int)kps[a*3], (int)kps[a*3+1]},
            {(int)kps[b*3], (int)kps[b*3+1]},
            cv::Scalar(255, 128, 0), 2, cv::LINE_AA);
cv::circle(img, {(int)kps[k*3], (int)kps[k*3+1]},
           5, cv::Scalar(0, 200, 255), -1, cv::LINE_AA);
```

This is drawn **only as fallback** when LBS mesh is unavailable. The YOLO
keypoints are in **image pixel coordinates** (scaled from YOLO 640x640 space
to original image dimensions).

**Source:** `fast_sam_3dbody_render.cpp` lines 300-327.

---

## 12. OpenGL Rendering Pipeline

### 12A. Vertex Data Flow

```
LBS output (camera-adjacent, meters, Y,Z flipped by LBS step 7)
    |
    v
memcpy into TRI_Model vertices
    |
    v
glBufferSubData -> GPU VBO
```

**MATCHES MATH.md Section 8A** — Vertices in camera-adjacent space are uploaded
directly to GPU.

### 12B. Projection Matrix

```cpp
// mhr_pose_driver.h lines 57-71
float p00 = 2.0f * focal_length / img_w;
float p11 = 2.0f * focal_length / img_h;
float p22 = -(far + near) / (far - near);  // far=100, near=0.01
float p32 = -2.0f * far * near / (far - near);

proj = | p00  0    0    0   |   // column-major
       |  0   p11  0    0   |
       |  0   0    p22  p32 |
       |  0   0   -1.0  0   |
```

**MATCHES MATH.md Section 8B** — Standard OpenGL perspective matrix.

### 12C. View Matrix

```cpp
// mhr_pose_driver.h lines 82-89
view = |  1   0   0    tx   |
       |  0  -1   0   -ty   |
       |  0   0  -1   -tz   |
       |  0   0   0    1    |
```

**MATCHES MATH.md Section 8C** — Same diagonal(1,-1,-1) + translation.

### 12D. Vertex Transformation Chain

For a vertex `v = (X, Y, Z)` in camera-adjacent space:

```
// View matrix application:
v_view = view * v
       = (X + tx, -Y - ty, -Z - tz, 1)

// Projection matrix application:
v_clip = proj * v_view
       = (p00*(X+tx), p11*(-Y-ty), p22*(-Z-tz) + p32, -(-Z-tz))
       = (p00*(X+tx), -p11*(Y+ty), p22*(Z+tz) + p32, Z+tz)

// GPU perspective divide:
ndc_x = p00 * (X+tx) / (Z+tz)
ndc_y = -p11 * (Y+ty) / (Z+tz)
ndc_z = (p22*(Z+tz) + p32) / (Z+tz)

// Viewport transform:
screen_x = (ndc_x + 1) * W / 2
screen_y = (ndc_y + 1) * H / 2
```

Substituting `p00 = 2*fx/W` and `p11 = 2*fy/H`:

```
screen_x = (2*fx/W * (X+tx)/(Z+tz) + 1) * W/2
         = fx * (X+tx)/(Z+tz) + W/2
         = fx * (X+tx)/(Z+tz) + cx_img

screen_y = (-2*fy/H * (Y+ty)/(Z+tz) + 1) * H/2
         = -fy * (Y+ty)/(Z+tz) + H/2
         = -fy * (Y+ty)/(Z+tz) + cy_img
```

**Note the negative sign on screen_y:** The OpenGL NDC has +Y pointing **up**
(bottom-left origin), while the image has +Y pointing **down** (top-left origin).
The view matrix flips Y (via `view[5] = -1`), and the viewport transform maps
NDC +Y to screen -Y. This is handled by the background quad UV mapping
(which flips the texture vertically).

**MATCHES MATH.md Section 8D-8G** — Same MVP composition, perspective divide,
and viewport transform.

### 12E. mat4_mul Implementation

```cpp
// fast_sam_3dbody_render.cpp lines 238-247
void mat4_mul(float dst[16], const float a[16], const float b[16]) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            dst[c*4+r] = 0;
            for (int k = 0; k < 4; ++k)
                dst[c*4+r] += a[k*4+r] * b[c*4+k];
        }
}
```

This computes `dst = a * b` in column-major layout. The call is:

```cpp
mat4_mul(mvp, proj, view);  // MVP = proj * view
```

**MATCHES MATH.md Section 8D** — Same composition order.

### 12F. Background Quad

```glsl
// UV coordinates flipped vertically to map image top-left to GL bottom-left
const vec2 UV[4] = vec2[](
    vec2(0.0,1.0), vec2(1.0,1.0),  // bottom row of quad -> top of texture
    vec2(0.0,0.0), vec2(1.0,0.0)   // top row of quad -> bottom of texture
);
```

The BGR frame from OpenCV is converted to RGB before uploading to the GPU
texture (`cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB)`).

**MATCHES MATH.md Section 8I** — Same vertical UV flip for image/texture origin mismatch.

### 12G. Fragment Shader

```glsl
vec3 L = normalize(vec3(0.3, 0.8, 0.5));
float d = clamp(dot(normalize(vNorm), L), 0.0, 1.0) * 0.7 + 0.3;
fragColor = vec4(vec3(0.65, 0.75, 0.9) * d, 0.7);
```

**MATCHES MATH.md Section 8H** — Same diffuse lighting with fixed direction
and alpha 0.7. Normals are T-pose (static, not skinned).

---

## 13. Discrepancy Summary

### DISCREPANCY 1: Double Y,Z Flip on Keypoints (HIGH SEVERITY)

**What:** `keypoints_3d` in `MHRResult` are in **MHR meter space** instead of
camera-adjacent space.

**Where:** `fast_sam_3dbody.cpp` lines 967-972.

**Root cause:** Vertices and joints are already Y,Z-flipped (by LBS step 7 or
by the ONNX post-processing at lines 908-941). The keypoint extraction reads
from these already-flipped coordinates. The additional flip at lines 967-972
undoes the flip.

**Effect:** 2D keypoint projection produces wrong pixel coordinates. The MHR
skeleton overlay will be misaligned with the mesh.

**Fix:** Remove the Y,Z flip at lines 967-972. The keypoints extracted from
already-flipped vertices/joints are already in camera-adjacent space.

### DISCREPANCY 2: fy Not Used in 2D Projection (LOW SEVERITY)

**What:** Both X and Y projection use `fx`. The Y projection should use `fy`.

**Where:** `fast_sam_3dbody.cpp` line 985:
```cpp
kps_2d[k*2 + 1] = dy / dz * fx + cy;  // should use fy
```

**Effect:** Only matters if `fx != fy`. Default is `fx = fy = image_width`.

**Fix:** Change to `dy / dz * fy + cy`.

### DISCREPANCY 3: Scale PCA Zeroed in C++ ONNX Path (MEDIUM SEVERITY)

**What:** The C++ `build_model_params()` zeros the scale parameters
(model_params[136:204]) instead of decompressing them via PCA:
```python
scales = scale_mean + scale_params @ scale_comps
```

**Where:** `preprocess.hpp` line 431.

**Effect:** The ONNX body model receives zero scales, producing a default-proportion
body. The LBS path has the same issue in `build_model_params()`, but the render
loop can optionally decode scales (render.cpp lines 522-531).

**Fix:** Load `scale_mean` and `scale_comps` from GGUF or a binary file, and
decode them in `build_model_params()`.

### DISCREPANCY 4: Global Rotation Euler Order Swap (INTENTIONAL, NOT A BUG)

**What:** C++ `rot6d_to_euler()` returns `[rx, ry, rz]` but `build_model_params()`
swaps to `[rz, ry, rx]` for the PT matrix.

**Why:** The PT matrix was trained with Python's `roma.rotmat_to_euler("ZYX")`
which returns `[rz, ry, rx]`. The C++ extraction computes `[rx, ry, rz]` and
reverses the order to match.

**This is correct** — it matches the Python reference.

### MATCHING ITEMS (Verified Correct)

| Item | Status |
|------|--------|
| Image normalization (ImageNet stats) | MATCH |
| CLIFF condition info | MATCH |
| 6D rotation -> Euler ZYX | MATCH |
| Body continuous -> Euler decomposition | MATCH |
| Camera parameter conversion (CLIFF) | MATCH |
| LBS skinning pipeline (6 steps) | MATCH |
| LBS Y,Z flip (step 7) | MATCH |
| LBS cm->meters (step 8) | MATCH |
| ONNX body model post-processing | MATCH |
| OpenGL projection matrix | MATCH |
| OpenGL view matrix | MATCH |
| MVP composition order | MATCH |
| mat4_mul column-major implementation | MATCH |
| Background quad UV flip | MATCH |
| Fragment shader lighting | MATCH |
| YOLO skeleton COCO pairs | MATCH |
| Hand joint zeroing | MATCH |
| Global translation zeroed | MATCH |
| Body pose truncated to 130 | MATCH |

---

## Complete C++ Transformation Chain

### LBS Path (body_model.lbs)

```
MHR Internal (cm, Y-up, Z-back)
    |
    |  LBS Step 1-6: skinning (cm, Y-up, Z-back)
    v
Skinned Vertices (cm, Y-up, Z-back)
    |
    |  LBS Step 7: Y,Z *= -1
    v
Camera-Adjacent Space (cm, Y-down, Z-forward)
    |
    |  LBS Step 8: * 0.01
    v
Camera-Adjacent Space (meters, Y-down, Z-forward)
    |
    |  memcpy -> TRI_Model -> GPU VBO
    v
[OpenGL Pipeline]
    |
    |  View matrix: diag(1,-1,-1) + translate(tx,-ty,-tz)
    v
View Space (meters, Y-up, Z-back, body at pred_cam_t)
    |
    |  Projection matrix: perspective(2*fx/W, 2*fy/H, near, far)
    v
Clip Space -> NDC -> Screen Space
    |
    |  Alpha blend with background quad
    v
Final Image
```

### Keypoint Path (BROKEN — double flip)

```
Camera-Adjacent Space (meters, from LBS or ONNX)
    |
    |  Keypoint extraction from already-flipped verts+joints
    v
Camera-Adjacent Space (meters, extracted keypoints)
    |
    |  BUG: Y,Z *= -1 (lines 967-972) — UNDOES the flip
    v
MHR Meter Space (meters, Y-up, Z-back)  <-- WRONG
    |
    |  + pred_cam_t, perspective divide
    v
Wrong 2D pixel coordinates
```

### Correct Keypoint Path (should be)

```
Camera-Adjacent Space (meters, from LBS or ONNX)
    |
    |  Keypoint extraction from already-flipped verts+joints
    v
Camera-Adjacent Space (meters, extracted keypoints)  <-- ALREADY CORRECT
    |
    |  + pred_cam_t, perspective divide
    v
Correct 2D pixel coordinates
```
