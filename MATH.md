# 3D Transformation Pipeline — MHR Skeleton Rendering on RGB Images

This document enumerates every coordinate system, matrix, and mathematical transformation
that occurs from the raw RGB webcam frame to the final overlaid skeleton + mesh image.
The pipeline has two rendering backends: **Python (pyrender)** and **C++ (OpenGL)**.
Both produce the same visual result but use different intermediate spaces.

---

## Table of Contents

1. [Overall Pipeline Flow](#1-overall-pipeline-flow)
2. [Coordinate Systems](#2-coordinate-systems)
3. [MHR Parametric Body Model Forward Pass](#3-mhr-parametric-body-model-forward-pass)
4. [Camera Parameter Prediction](#4-camera-parameter-prediction)
5. [3D-to-2D Keypoint Projection](#5-3d-to-2d-keypoint-projection)
6. [2D Skeleton Drawing on Image](#6-2d-skeleton-drawing-on-image)
7. [Python 3D Mesh Rendering (pyrender)](#7-python-3d-mesh-rendering-pyrender)
8. [C++ OpenGL Mesh Rendering](#8-c-opengl-mesh-rendering)
9. [Multi-Person Compositing](#9-multi-person-compositing)
10. [Complete Transformation Chain Summary](#10-complete-transformation-chain-summary)

---

## 1. Overall Pipeline Flow

```
RGB Image (H x W x 3, BGR from OpenCV)
    |
    v
[YOLO Person Detection]              -> bounding boxes [x1,y1,x2,y2]
    |
    v
[Image Preprocessing per person]     -> 512x512 crop, normalized to [-1,1]
    |
    v
[SAM Image Encoder]                  -> image embeddings
    |
    v
[Decoder + MHR Head]                 -> pose tokens -> MHR parameters
    |
    +-- MHR JIT Model                -> verts[B,18439,3] (cm), skel_state[B,127,8] (cm)
    |
    +-- Camera Head                  -> weak perspective [s, tx, ty]
    |
    v
[Coordinate System Conversions]      -> cm->meters, Y,Z flip
    |
    v
[Camera Translation + Projection]    -> 3D cam space, 2D pixel coords
    |
    +----------------------------------+----------------------------------+
    v                                  v                                  v
[2D Skeleton on Image]           [Python pyrender 3D mesh]        [C++ OpenGL 3D mesh]
```

**Source files:**
- Entry: `demo_webcam.py` -> `estimator.process_one_image()`
- Model: `sam_3d_body/models/meta_arch/sam3d_body.py`
- MHR body: `sam_3d_body/models/heads/mhr_head.py`
- Camera: `sam_3d_body/models/heads/camera_head.py`
- Projection: `sam_3d_body/models/modules/geometry_utils.py`
- 2D skeleton: `sam_3d_body/visualization/skeleton_visualizer.py`
- Python 3D: `sam_3d_body/visualization/renderer.py`
- Compositing: `tools/vis_utils.py`
- C++ OpenGL: `fast_sam_3dbody_cpp/render/fast_sam_3dbody_render.cpp`
- C++ camera: `fast_sam_3dbody_cpp/render/mhr_pose_driver.h`

---

## 2. Coordinate Systems

There are **five distinct coordinate systems** in this pipeline:

### 2A. Image Pixel Coordinate System

```
Origin: top-left corner of the image
  +X: rightward (column index)
  +Y: downward  (row index)
Range: x in [0, W), y in [0, H)
```

This is the standard OpenCV/image coordinate system. The skeleton is drawn in these coordinates.

### 2B. MHR Internal Model Space

```
Unit: centimeters
Origin: approximately at the body root (hip area in T-pose)
  +X: rightward  (subject's right)
  +Y: upward     (subject's up)
  +Z: backward   (subject's back / behind the body)
```

The raw MHR JIT model outputs vertices and joint coordinates in this space.
The body is centered near the origin in T-pose when all pose parameters are zero.

**Source:** `mhr_head.py` lines 591-600 — MHR JIT model returns centimeters.

### 2C. MHR Meter Space (Post Scale, Pre-Flip)

```
Unit: meters (obtained by multiplying MHR Internal by 0.01)
Origin: same as MHR Internal
  +X: rightward
  +Y: upward
  +Z: backward
```

**Transform from 2B:**
```
v_meter = v_cm * 0.01
```

**Source:** `mhr_head.py` lines 599-600:
```python
curr_skinned_verts = curr_skinned_verts * 0.01
curr_joint_coords = curr_joint_coords * 0.01
```

### 2D. Camera-Adjacent Space (Post Y,Z Flip)

```
Unit: meters
Origin: body root at approximately (0,0,0)
  +X: rightward  (from camera POV, matches image +X)
  +Y: downward   (from camera POV, matches image +Y)
  +Z: forward    (away from camera, into the scene)
```

**Transform from 2C:**
```
v_cam_adj[X] = v_meter[X]
v_cam_adj[Y] = -v_meter[Y]      # sign flip
v_cam_adj[Z] = -v_meter[Z]      # sign flip
```

In matrix form:
```
| 1  0  0  0 |   | X |   | X  |
| 0 -1  0  0 |   | Y |   |-Y  |
| 0  0 -1  0 | x | Z | = |-Z  |
| 0  0  0  1 |   | 1 |   | 1  |
```

This converts from the MHR convention (Y-up, Z-back) to a camera-like convention
(Y-down, Z-forward). The body is still centered at the origin — the camera translation
has not yet been applied.

**Source:** `mhr_head.py` lines 458-465 (in `_head_forward_core`):
```python
if verts is not None:
    verts = verts.clone()
    verts[..., [1, 2]] *= -1  # Camera system difference
j3d = j3d.clone()
j3d[..., [1, 2]] *= -1
```

### 2E. Camera Space (Post Translation)

```
Unit: meters
Origin: camera pinhole center (0,0,0)
  +X: rightward
  +Y: downward
  +Z: forward (away from camera, into the scene)
```

**Transform from 2D:**
```
v_camera = v_cam_adj + pred_cam_t
```

Where `pred_cam_t = [tx, ty, tz]` positions the body in front of the camera.
The camera sits at the origin looking along +Z.

**Source:** `camera_head.py` line 101:
```python
j3d_cam = points_3d + pred_cam_t.unsqueeze(1)
```

---

## 3. MHR Parametric Body Model Forward Pass

### 3A. Neural Network Output Parsing

The MHR Head FFN outputs a vector of dimension `npose = 519`:

```
Offset  Size  Meaning
------  ----  -------
0       6     Global rotation (6D orthogonal representation)
6       260   Body continuous parameters (PCA-compressed pose)
266     45    Shape parameters (PCA coefficients)
311     28    Scale parameters (PCA coefficients)
339     108   Hand pose parameters (54 per hand)
447     72    Face/expression parameters (zeroed out in inference)
```

**Source:** `mhr_head.py` lines 122-129, 867-902.

### 3B. 6D Rotation to Euler ZYX

The global rotation is predicted as a 6D vector (Zhou et al. CVPR 2019):

```
Input: global_rot_6d [6] = [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]

Reshape to two vectors:
  a1 = [r1_x, r1_y, r1_z]
  a2 = [2_x, r2_y, r2_z]

Gram-Schmidt orthonormalization:
  b1 = a1 / ||a1||
  b2 = a2 - (b1 . a2) * b1
  b2 = b2 / ||b2||
  b3 = cross(b1, b2)

Rotation matrix R = [b1 | b2 | b3]  (columns are basis vectors)

Convert R to Euler angles (ZYX convention):
  global_rot_euler = rotmat_to_euler_ZYX(R)  -> [roll, pitch, yaw]
```

**Source:** `geometry_utils.py` lines 85-105 (`rot6d_to_rotmat`).

### 3C. Body Continuous Parameters to Euler Angles

The 260-dim continuous body parameters are decompressed to 133 Euler angles
(one per joint, 3 angles per joint for 133/3 ~ 44 joints plus extras):

```
pred_pose_euler = compact_cont_to_model_params_body_fast(pred_pose_cont)
```

The hand joint indices (within the 133-dim output) are zeroed:
```
pred_pose_euler[:, mhr_param_hand_idxs] = 0
pred_pose_euler[:, -3:] = 0          # jaw angles zeroed
```

Then truncated to 130 parameters for the MHR model call:
```
body_pose_params = pred_pose_euler[..., :130]
```

### 3D. Scale PCA Decompression

```
scales[68] = scale_mean[68] + scale_params[28] @ scale_comps[28, 68]
```

**Source:** `mhr_head.py` line 571.

### 3E. Assembling MHR Model Parameters

The full MHR model parameter vector has 204 dimensions:

```
Offset  Size  Meaning
------  ----  -------
0       3     Global translation (* 10 for cm scale, but zeroed in single-view)
3       3     Global rotation (Euler XYZ)
6       130   Body pose (Euler angles)
136     68    Scales
```

```
full_pose_params = concat(global_trans * 10, global_rot, body_pose_params[:130])
# If hand model enabled, replace hand joints
model_params = concat(full_pose_params, scales)  -> [204]
```

**Source:** `mhr_head.py` lines 574-584.

**Important:** In single-view mode, `global_trans = [0, 0, 0]`. The body position
is controlled entirely by `pred_cam_t` (the camera translation), not by the model
translation.

### 3F. MHR JIT Model — Skinning

```
skinned_verts[B, 18439, 3], skel_state[B, 127, 8] = mhr(
    shape_params[B, 45],
    model_params[B, 204],
    expr_params[B, 72],
    apply_correctives
)
```

The MHR model internally performs:
1. Vertex displacement from shape PCA
2. Linear Blend Skinning (LBS) using 127 joint transformations
3. Corrective shape blending (optional, disabled via `MHR_NO_CORRECTIVES=1`)

Output `skel_state` layout per joint: `[joint_x, joint_y, joint_z, quat_x, quat_y, quat_z, quat_w, 1]`

**Source:** `mhr_head.py` lines 591-593.

### 3G. Post-Skinning Processing

```
# Extract joint coordinates and quaternions from skel_state
joint_coords[B, 127, 3] = skel_state[:, :, :3]
joint_quats[B, 127, 4]  = skel_state[:, :, 3:7]

# Convert from centimeters to meters
skinned_verts = skinned_verts * 0.01    # [B, 18439, 3] in meters
joint_coords  = joint_coords  * 0.01    # [B, 127, 3] in meters
```

### 3H. Keypoint Extraction (308 -> 70)

A sparse mapping matrix extracts 308 keypoints from the union of mesh vertices
and skeleton joints:

```
model_vert_joints = concat(skinned_verts, joint_coords, dim=1)  # [B, 18439+127, 3]

keypoint_mapping[308, 18566]  # sparse weight matrix

keypoints_308 = einsum('kv,bvc->bkc', keypoint_mapping, model_vert_joints)
               # [308, 18566] @ [B, 18566, 3] -> [B, 308, 3]
```

Then truncated to 70 keypoints (face keypoints removed):
```
j3d = keypoints_308[:, :70]  # [B, 70, 3]
```

**Source:** `mhr_head.py` lines 611-616, 457.

### 3I. Coordinate System Flip

The Y and Z axes are negated to convert from MHR model space to camera-adjacent space:

```
verts[..., [1, 2]] *= -1    # [B, 18439, 3], Y and Z negated
j3d[..., [1, 2]] *= -1      # [B, 70, 3], Y and Z negated
jcoords[..., [1, 2]] *= -1  # [B, 127, 3], Y and Z negated
```

This is a 180-degree rotation around the X-axis.

**Source:** `mhr_head.py` lines 458-465, 931-935.

---

## 4. Camera Parameter Prediction

### 4A. Weak Perspective Camera Prediction

The camera head predicts three parameters via a small FFN:

```
pred_cam = [s, tx, ty] = FFN(pose_token)
```

Where:
- `s` = weak perspective scale
- `tx` = 2D offset in x (pixel space, relative to center)
- `ty` = 2D offset in y (pixel space, relative to center)

**Source:** `camera_head.py` lines 36-43, 55-59.

### 4B. Weak Perspective to Strong Perspective Conversion (CLIFF)

The CLIFF formulation (from CameraHMR) converts weak perspective params to
full perspective camera translation:

```
// Sign flips (coordinate system difference between prediction and camera space):
s  = -pred_cam[0]     // flip sign of scale
tx =  pred_cam[1]
ty = -pred_cam[2]     // flip sign of ty

// Effective bounding box size in the full image:
bs = bbox_size * s * default_scale_factor + 1e-8
   // default_scale_factor = 1.25 (accounts for 1.25x bbox padding)

// Depth from focal length and bounding box size:
tz = 2 * focal_length / bs

// Centering offset (shifts body from image center to bbox center):
cx = 2 * (bbox_center_x - img_width / 2) / bs
cy = 2 * (bbox_center_y - img_height / 2) / bs

// Final camera translation vector:
pred_cam_t = [tx + cx, ty + cy, tz]
```

**Matrix interpretation:** The `tz = 2*f/bs` formula comes from equating the weak
perspective projection (`x_pixel = s * X * W/2 + cx`) with the strong perspective
projection (`x_pixel = f * X / Z + cx`):

```
s * W/2 = f * 1/Z  =>  Z = f / (s * W/2) = 2*f / (s*W)
```

Where `W` is replaced by `bs` (the effective bounding box size with scale factor).

**Source:** `camera_head.py` lines 77-98.

### 4C. Camera Intrinsic Matrix

The camera intrinsic matrix K is:

```
K = | fx  0  cx_img |
    |  0  fy  cy_img |
    |  0   0    1    |
```

Where:
- `fx = fy = focal_length` (predicted or from EXIF, in pixels)
- `cx_img = img_width / 2` (principal point at image center)
- `cy_img = img_height / 2`

**Source:** `geometry_utils.py` lines 175-198 (`get_intrinsic_matrix`).

---

## 5. 3D-to-2D Keypoint Projection

This section details how the 70 3D keypoints are projected to 2D pixel coordinates
for the skeleton overlay.

### 5A. Apply Camera Translation

Move from camera-adjacent space (body at origin) to camera space (camera at origin):

```
j3d_cam[N, 3] = j3d[N, 3] + pred_cam_t[3]
```

Broadcast addition: each of the 70 keypoints gets the same translation.

**Source:** `camera_head.py` line 101.

### 5B. Perspective Divide

```
j2d_raw[N, 2] = j3d_cam[N, :2] / j3d_cam[N, 2:3]
```

Explicitly:
```
j2d_raw_x = j3d_cam_x / j3d_cam_z
j2d_raw_y = j3d_cam_y / j3d_cam_z
```

This is the "divide by depth" operation that implements perspective projection.
It maps 3D camera coordinates to normalized image plane coordinates where
the focal length is 1.

### 5C. Apply Camera Intrinsics

```
j2d = j2d_raw @ K^T
```

Or explicitly:
```
pixel_x = j2d_raw_x * fx + cx_img
pixel_y = j2d_raw_y * fy + cy_img
```

**Source:** `geometry_utils.py` lines 201-215:
```python
def perspective_projection(x, K):
    y = x / x[:, :, -1].unsqueeze(-1)   # perspective divide
    y = y.bmm(K.transpose(-1, -2))       # apply intrinsics: y @ K^T
    return y[:, :, :2]                   # discard depth
```

### 5D. Combined Projection (Single Expression)

The full chain from camera-adjacent space to pixels:

```
pixel_x = fx * (j3d_x + tx) / (j3d_z + tz) + cx_img
pixel_y = fy * (j3d_y + ty) / (j3d_z + tz) + cy_img
```

Where `j3d` is in camera-adjacent space (post Y,Z flip, body at origin)
and `pred_cam_t = [tx, ty, tz]`.

---

## 6. 2D Skeleton Drawing on Image

### 6A. Keypoint Format for Drawing

The 2D keypoints `[70, 2]` are augmented with a confidence column of ones:

```
keypoints_draw[70, 3] = [keypoint_x, keypoint_y, 1.0]
```

**Source:** `vis_utils.py` lines 106-108:
```python
keypoints_2d = np.concatenate(
    [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
)
```

### 6B. Skeleton Links

65 links define which keypoints connect. Each link is a pair of indices
into the 70-keypoint array. Example links:

```
Link 0:  left_ankle(13) -- left_knee(11)
Link 1:  left_knee(11) -- left_hip(9)
Link 7:  left_shoulder(5) -- right_shoulder(6)
Link 12: left_eye(1) -- right_eye(2)
...
```

**Source:** `mhr70.py` lines 616-805 (`skeleton_info`).

### 6C. Drawing Operations

For each link `(i, j)`:
```
pos1 = (int(kpts[i, 0]), int(kpts[i, 1]))   // pixel coordinates
pos2 = (int(kpts[j, 0]), int(kpts[j, 1]))

cv2.line(image, pos1, pos2, color, thickness=2)
```

For each keypoint:
```
cv2.circle(image, (int(kpt[0]), int(kpt[1])), radius=5, color, -1)
```

The drawing happens directly in image pixel coordinates (coordinate system 2A).
No further transformation is needed.

**Source:** `skeleton_visualizer.py` lines 100-163.

---

## 7. Python 3D Mesh Rendering (pyrender)

This section covers the `Renderer` class that overlays the 3D mesh on the image
using pyrender (which uses OpenGL under the hood).

### 7A. Input to Renderer

```
vertices: [18439, 3]  — in camera-adjacent space (post Y,Z flip, body at origin)
cam_t: [3]            — pred_cam_t = [tx, ty, tz]
image: [H, W, 3]     — RGB image, values in [0, 255]
focal_length: float   — in pixels
```

### 7B. Camera Translation Sign Flip

```
camera_translation = cam_t.copy()
camera_translation[0] *= -1.0   // flip X component
```

This gives `[-tx, ty, tz]`. The X flip compensates for the coordinate system
difference between the MHR convention and pyrender's scene convention.

**Source:** `renderer.py` lines 183-184.

### 7C. Mesh Creation and 180-Degree X Rotation

```python
mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
mesh.apply_transform(rot)
```

This applies a 180-degree rotation around the X-axis to every vertex:

```
R_x(180) = |  1   0    0  |
           |  0  -1    0  |
           |  0   0   -1  |
```

Which is equivalent to `vertex[Y] *= -1, vertex[Z] *= -1`.

**Why this exists:** The vertices coming from MHR have already been Y,Z-flipped
once (in `mhr_head.py`). This rotation flips them again, resulting in **net
identity** on the vertex coordinates. The vertices end up in MHR meter space
(pre-flip orientation), and the camera is placed in a position that makes this
work correctly.

**Source:** `renderer.py` lines 196-210.

### 7D. Camera Placement in pyrender

The camera pose is a 4x4 translation matrix:

```
camera_pose = | 1  0  0  -tx |
              | 0  1  0   ty |
              | 0  0  0   tz |
              | 0  0  0   1  |
```

This places the camera at `(-tx, ty, tz)` in the scene. Since the mesh vertices
are at the body origin (approximately `(0,0,0)` with no global translation), the
camera is positioned **behind and to the left** of the body.

**Source:** `renderer.py` lines 219-230.

### 7E. pyrender Intrinsics Camera

```
camera = pyrender.IntrinsicsCamera(
    fx = focal_length,
    fy = focal_length,
    cx = width / 2,
    cy = height / 2,
    zfar = 1e12,
)
```

This creates a pinhole camera with the intrinsic matrix:

```
K = | fx  0  W/2 |
    |  0  fy H/2 |
    |  0   0   1  |
```

### 7F. Net Effect of Python Pipeline

The Python pipeline performs:

1. Vertices start in camera-adjacent space (post MHR Y,Z flip)
2. 180-degree X rotation flips Y,Z again -> net identity (vertices in MHR meter space)
3. Camera at `(-tx, ty, tz)` looking at origin
4. Intrinsics camera projects to pixels

The relative transformation between camera and body is:

```
Body vertex in MHR meter space: v = (X, Y, Z)
Camera at: (-tx, ty, tz)

Vector from camera to vertex (in camera frame):
  v_cam = v - camera_position = (X + tx, Y - ty, Z - tz)

Pixel projection:
  u = fx * (X + tx) / (Z - tz) + cx
  v = fy * (Y - ty) / (Z - tz) + cy
```

**Note:** This is NOT the same as the 2D keypoint projection formula. The 2D
keypoints use `j3d + pred_cam_t` with a camera at origin. The pyrender pipeline
uses the vertices at origin with camera at `(-tx, ty, tz)`. Both are mathematically
equivalent because only the **relative** position matters:

For 2D keypoints: `v_camera = v + pred_cam_t`, camera at origin
For pyrender: `v` at origin, camera at `(-tx, ty, tz)`

The vector from camera to vertex is the same in both formulations.

### 7G. Side View

For the side view, an additional 90-degree rotation around Y is applied before
the 180-degree X rotation:

```
rot_side = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
mesh.apply_transform(rot_side)
```

```
R_y(90) = |  0   0   1 |
          |  0   1   0 |
          | -1   0   0 |
```

And the camera translation is NOT applied (camera at origin):

```
camera_pose = identity
```

**Source:** `renderer.py` lines 198-207.

### 7H. Alpha Compositing

The renderer outputs an RGBA buffer. The mesh is composited onto the background
image using alpha blending:

```
output = mesh_RGB * alpha_mask + background_RGB * (1 - alpha_mask)
```

**Source:** `renderer.py` lines 254-255.

---

## 8. C++ OpenGL Mesh Rendering

The C++ pipeline renders meshes using raw OpenGL 3.3 core profile.

### 8A. Vertex Data Flow

```
pred_vertices from pipeline (already Y,Z-flipped, in camera-adjacent space)
    |
    v
memcpy into TRI_Model vertex buffer
    |
    v
glBufferSubData -> GPU VBO (GL_DYNAMIC_DRAW)
```

**Source:** `mhr_pose_driver.h` lines 32-37.

### 8B. OpenGL Projection Matrix

A standard OpenGL perspective matrix is built from the pinhole focal length:

```
p00 = 2.0 * focal_length / img_w
p11 = 2.0 * focal_length / img_h
p22 = -(far + near) / (far - near)
p32 = -2.0 * far * near / (far - near)

// Column-major layout:
proj = | p00   0    0     0    |
       |  0   p11   0     0    |
       |  0    0   p22   p32   |
       |  0    0   -1.0   0    |
```

Where `near = 0.01` and `far = 100.0`.

**Relation to camera intrinsics:** The OpenGL perspective matrix with
`p00 = 2*fx/W` is equivalent to the pinhole projection followed by a
mapping from pixel coordinates to NDC (Normalized Device Coordinates):

```
ndc_x = 2 * (pixel_x - W/2) / W = (fx * X/Z) / (W/2) = (2*fx/W) * (X/Z)
ndc_y = 2 * (pixel_y - H/2) / H = (fy * Y/Z) / (H/2) = (2*fy/H) * (Y/Z)
```

This assumes the principal point is at `(W/2, H/2)`.

**Source:** `mhr_pose_driver.h` lines 57-71.

### 8C. OpenGL View Matrix

```
view = |  1   0   0    tx   |
       |  0  -1   0   -ty   |
       |  0   0  -1   -tz   |
       |  0   0   0    1    |
```

**Purpose:** This matrix performs two operations:

1. **Y,Z sign flip** (diagonal elements `-1`): The vertices in the C++ pipeline
   have the MHR Y,Z flip applied (converting to camera-adjacent space). This
   flips them back to MHR meter space. This is the same as the 180-degree X
   rotation in the Python pipeline.

2. **Translation** (tx, -ty, -tz): Translates the vertices by `pred_cam_t`
   in the X direction and by `-pred_cam_t` in Y,Z (because the diagonal
   already negates Y,Z).

**Net effect on a vertex:**

```
v_model = (X, Y, Z, 1) in camera-adjacent space

v_view = view * v_model
       = (X + tx, -Y - ty, -Z - tz, 1)
       = (X + tx, -(Y + ty), -(Z + tz), 1)
```

After the Y,Z flip from the diagonal, the vertex is in MHR meter space shifted
by `pred_cam_t`. The projection matrix then projects this to clip space.

**Verification against Python pipeline:**

For a vertex `v = (X, Y, Z)` in MHR meter space (after view matrix undoes Y,Z flip):

```
Python: camera at (-tx, ty, tz), vertex at (X,Y,Z)
  -> vector from camera: (X+tx, Y-ty, Z-tz)

C++: view matrix gives (X+tx, -Y-ty, -Z-tz) in view space
  -> OpenGL projects (X+tx, -Y-ty, -Z-tz) with camera at origin looking -Z

  -> But OpenGL camera looks along -Z, so the "depth" is -( -Z-tz) = Z+tz
     which matches the camera-adjacent convention.
```

**Source:** `mhr_pose_driver.h` lines 73-89.

### 8D. MVP Composition

```
MVP = proj * view
```

No model matrix is needed since the mesh is already in the correct space
(camera-adjacent). The vertices are transformed as:

```
v_clip = MVP * v_model
```

**Source:** `fast_sam_3dbody_render.cpp` line 569.

### 8E. Vertex Shader

```glsl
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
```

This transforms each vertex from camera-adjacent space to clip space in one
matrix multiplication.

### 8F. GPU Perspective Divide

After the vertex shader, the GPU automatically performs the perspective divide:

```
ndc_x = gl_Position.x / gl_Position.w
ndc_y = gl_Position.y / gl_Position.w
ndc_z = gl_Position.z / gl_Position.w
```

The result is in NDC space: x,y,z in [-1, +1].

### 8G. Viewport Transform

The GPU automatically maps NDC to screen pixels:

```
screen_x = (ndc_x + 1) * img_w / 2
screen_y = (ndc_y + 1) * img_h / 2
```

This maps [-1, +1] to [0, img_w] and [0, img_h].

### 8H. Fragment Shader

```glsl
in vec3 vNorm;
void main() {
    vec3 L = normalize(vec3(0.3, 0.8, 0.5));
    float d = clamp(dot(normalize(vNorm), L), 0.0, 1.0) * 0.7 + 0.3;
    fragColor = vec4(vec3(0.65, 0.75, 0.9) * d, 0.7);
}
```

Diffuse lighting with a fixed light direction `(0.3, 0.8, 0.5)`. The mesh color
is light blue `(0.65, 0.75, 0.9)` modulated by `diffuse * 0.7 + ambient * 0.3`.
The alpha is `0.7` (semi-transparent).

**Note:** Normals are T-pose normals (static, not skinned). This is an
approximation — for correct lighting under deformation, normals should be
transformed by the inverse-transpose of the skinning Jacobian.

### 8I. Background Quad

A fullscreen quad is rendered first with the camera frame as a texture:

```glsl
// Vertex shader — generates 4 corners in NDC
const vec2 P[4] = vec2[](
    vec2(-1.0,-1.0), vec2(1.0,-1.0),
    vec2(-1.0, 1.0), vec2(1.0, 1.0)
);
gl_Position = vec4(P[gl_VertexID], 0.0, 1.0);
```

```glsl
// Fragment shader — samples texture with UV flipped vertically
// (image origin at top-left, GL texture origin at bottom-left)
const vec2 UV[4] = vec2[](
    vec2(0.0,1.0), vec2(1.0,1.0),
    vec2(0.0,0.0), vec2(1.0,0.0)
);
fragColor = vec4(texture(uTex, vUV).rgb, 1.0);
```

The BGR frame from OpenCV is converted to RGB before uploading to the GPU texture.

**Source:** `fast_sam_3dbody_render.cpp` lines 37-57, 204-234.

### 8J. Render Order and Blending

```
glEnable(GL_DEPTH_TEST);
glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
```

1. Clear color + depth buffers
2. Draw background quad (depth test disabled)
3. Draw mesh for each person (depth test + alpha blending enabled)

The mesh alpha of 0.7 means it blends as:
```
final_color = 0.7 * mesh_color + 0.3 * background_color
```

---

## 9. Multi-Person Compositing

### 9A. Depth Sorting

When multiple people are detected, they are sorted by depth (furthest first):

```
all_depths = [person.pred_cam_t[2] for person in outputs]  // tz values
outputs_sorted = sort(outputs, by=all_depths, descending=True)
```

This ensures that closer people occlude further ones correctly.

**Source:** `vis_utils.py` lines 100-101.

### 9B. Skeleton Overlay Order

Skeletons are drawn in depth-sorted order (furthest first):

```
for person in outputs_sorted:
    img_keypoints = draw_skeleton(img_keypoints, person.keypoints_2d)
```

**Source:** `vis_utils.py` lines 104-109.

### 9C. Combined Mesh Construction

For multi-person mesh rendering, all meshes are combined into one:

```
for person in outputs_sorted:
    all_pred_vertices.append(vertices + cam_t)  // translate to camera space
    all_faces.append(faces + vertex_offset)

all_pred_vertices = concat(all_pred_vertices)  // [N_people * 18439, 3]
all_faces = concat(all_faces)
```

A "fake" translation is extracted from the last person's mesh to re-center:

```
fake_pred_cam_t = (max(verts_last_person) + min(verts_last_person)) / 2
all_pred_vertices = all_pred_vertices - fake_pred_cam_t
```

This re-centered mesh + fake translation is passed to the renderer so that
all people appear in one pyrender scene.

**Source:** `vis_utils.py` lines 112-148.

---

## 10. Complete Transformation Chain Summary

### 10A. 3D Keypoint -> 2D Pixel (Skeleton)

```
MHR Internal (cm, Y-up, Z-back)
    |
    |  * 0.01  (cm -> meters)
    v
MHR Meter Space (meters, Y-up, Z-back)
    |
    |  Y *= -1, Z *= -1  (coordinate flip)
    v
Camera-Adjacent Space (meters, Y-down, Z-forward, body at origin)
    |
    |  + pred_cam_t[tx, ty, tz]  (camera translation)
    v
Camera Space (camera at origin, body at pred_cam_t)
    |
    |  x/z, y/z  (perspective divide)
    |  * fx + cx, * fy + cy  (camera intrinsics)
    v
Pixel Space (top-left origin, X-right, Y-down)
    |
    |  cv2.line(), cv2.circle()  (draw on image)
    v
Final Image
```

### 10B. 3D Mesh -> Overlay (Python pyrender)

```
MHR Internal (cm)
    |
    |  * 0.01, Y,Z flip
    v
Camera-Adjacent Space (meters, Y-down, Z-forward)
    |
    |  R_x(180) -> Y,Z flip again (net: back to MHR meter space)
    v
MHR Meter Space (Y-up, Z-back, body at origin)
    |
    |  Camera placed at (-tx, ty, tz)
    |  Intrinsics camera: K = diag(fx,fy) + principal point (W/2, H/2)
    v
pyrender projects to RGBA buffer
    |
    |  Alpha composite with background image
    v
Final Image
```

### 10C. 3D Mesh -> Overlay (C++ OpenGL)

```
Camera-Adjacent Space (meters, Y-down, Z-forward, from pipeline)
    |
    |  View matrix: diag(1, -1, -1) + translate(tx, -ty, -tz)
    |  -> undoes Y,Z flip, translates body
    v
View Space (meters, Y-up, Z-back, body at pred_cam_t)
    |
    |  Projection matrix: perspective(2*fx/W, 2*fy/H, near, far)
    v
Clip Space (homogeneous)
    |
    |  Perspective divide (GPU automatic)
    v
NDC Space [-1, +1]^3
    |
    |  Viewport transform (GPU automatic)
    v
Screen Space (pixel coordinates)
    |
    |  Alpha blend with background quad
    v
Final Image
```

### 10D. MHR70 Joint Index Map

```
Index  Joint Name              Index  Joint Name
-----  ----------------------  -----  ----------------------
0      nose                    35     right_ring_third_joint
1      left_eye                36     right_pinky_tip
2      right_eye               37     right_pinky_first_joint
3      left_ear                38     right_pinky_second_joint
4      right_ear               39     right_pinky_third_joint
5      left_shoulder           40     right_wrist
6      right_shoulder          41     left_thumb_tip
7      left_elbow              42     left_thumb_first_joint
8      right_elbow             43     left_thumb_second_joint
9      left_hip                44     left_thumb_third_joint
10     right_hip               45     left_index_tip
11     left_knee               46     left_index_first_joint
12     right_knee              47     left_index_second_joint
13     left_ankle              48     left_index_third_joint
14     right_ankle             49     left_middle_tip
15     left_big_toe_tip        50     left_middle_first_joint
16     left_small_toe_tip      51     left_middle_second_joint
17     left_heel               52     left_middle_third_joint
18     right_big_toe_tip       53     left_ring_tip
19     right_small_toe_tip     54     left_ring_first_joint
20     right_heel              55     left_ring_second_joint
21     right_thumb_tip         56     left_ring_third_joint
22     right_thumb_first_joint 57     left_pinky_tip
23     right_thumb_second_joint58     left_pinky_first_joint
24     right_thumb_third_joint 59     left_pinky_second_joint
25     right_index_tip         60     left_pinky_third_joint
26     right_index_first_joint 61     left_wrist
27     right_index_second_joint62     left_olecranon
28     right_index_third_joint 63     right_olecranon
29     right_middle_tip        64     left_cubital_fossa
30     right_middle_first_joint65     right_cubital_fossa
31     right_middle_second_joint66    left_acromion
32     right_middle_third_joint67     right_acromion
33     right_ring_tip          68     (not in 70, but exists in 308)
34     right_ring_first_joint  69     neck
35     right_ring_second_joint
```

### 10E. Skeleton Link Pairs (65 links)

```
Link  Joint A -- Joint B          Limb
----  -------------------------  ----------
 0    left_ankle -- left_knee     Left lower leg
 1    left_knee -- left_hip       Left upper leg
 2    right_ankle -- right_knee   Right lower leg
 3    right_knee -- right_hip     Right upper leg
 4    left_hip -- right_hip       Pelvis
 5    left_shoulder -- left_hip   Left torso
 6    right_shoulder -- right_hip Right torso
 7    left_shoulder -- right_shoulder  Shoulders
 8    left_shoulder -- left_elbow Left upper arm
 9    right_shoulder -- right_elbow Right upper arm
10    left_elbow -- left_wrist    Left forearm
11    right_elbow -- right_wrist  Right forearm
12    left_eye -- right_eye       Eyes
13    nose -- left_eye            Face
14    nose -- right_eye           Face
15    left_eye -- left_ear        Face
16    right_eye -- right_ear      Face
17    left_ear -- left_shoulder   Neck-left
18    right_ear -- right_shoulder Neck-right
19-24  Foot keypoints (toes, heel)
25-64  Hand keypoints (fingers)
```

---

## Key Equations Reference

### Camera Translation from Weak Perspective
```
s  = -pred_cam[0]
tx =  pred_cam[1]
ty = -pred_cam[2]
bs = bbox_size * s * 1.25 + 1e-8
tz = 2 * focal_length / bs
cx = 2 * (bbox_center_x - img_w/2) / bs
cy = 2 * (bbox_center_y - img_h/2) / bs
pred_cam_t = [tx + cx, ty + cy, tz]
```

### Perspective Projection
```
pixel_x = fx * (X + tx) / (Z + tz) + cx_img
pixel_y = fy * (Y + ty) / (Z + tz) + cy_img
```

### OpenGL Projection Matrix
```
proj[0]  = 2*fx/W    proj[5]  = 2*fy/H
proj[10] = -(f+n)/(f-n)  proj[11] = -2*f*n/(f-n)
proj[14] = -1.0
```

### OpenGL View Matrix
```
view[0]  = 1.0   view[5]  = -1.0   view[10] = -1.0
view[12] = tx    view[13] = -ty    view[14] = -tz
view[15] = 1.0
```

### MVP Transformation
```
v_clip = proj * view * v_model
v_ndc  = v_clip[:3] / v_clip[3]
screen = (v_ndc + 1) * (img_dim / 2)
```
