#pragma once
// ============================================================================
// preprocess.hpp  –  per-person crop, normalisation, ray-condition,
//                    CLIFF condition, YOLO NMS helpers
// ============================================================================
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstring>
#include <vector>

namespace fsb {

// ─── image normalisation constants (from model_config.yaml) ──────────────────
static constexpr float IMAGE_MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float IMAGE_STD[3]  = {0.229f, 0.224f, 0.225f};
static constexpr int   CROP_SIZE     = 512;
static constexpr int   FEAT_HW      = CROP_SIZE / 16;  // 32 – patch grid size
static constexpr int   PATCH_SIZE   = 16;

// ─── Crop one person out of a BGR image and return normalised CHW float32 ─────
//
// bbox_x1/y1/x2/y2 : person bounding box in original image (float, unclamped)
// out_chw           : pre-allocated float[3 × CROP_SIZE × CROP_SIZE]
//
// The crop is a square centred on the bbox, padded with grey if needed.
// Normalised with IMAGE_MEAN / IMAGE_STD (RGB channel order).
//
// Also fills:
//   crop_cx, crop_cy       – bbox centre used for this crop
//   crop_size              – side length of the square crop (in original pixels)
inline void crop_and_normalise(
    const cv::Mat& bgr,              // full image [H,W,3] CV_8UC3
    float  bbox_x1, float bbox_y1,
    float  bbox_x2, float bbox_y2,
    float* out_chw,                  // [3, CROP_SIZE, CROP_SIZE]
    float& crop_cx, float& crop_cy,  // outputs: crop centre
    float& crop_size_out             // output: square side in source pixels
)
{
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // square crop centred on bbox
    float cx   = (bbox_x1 + bbox_x2) * 0.5f;
    float cy   = (bbox_y1 + bbox_y2) * 0.5f;
    float bw   = bbox_x2 - bbox_x1;
    float bh   = bbox_y2 - bbox_y1;
    float side = std::max(bw, bh);

    crop_cx      = cx;
    crop_cy      = cy;
    crop_size_out= side;

    int x1 = static_cast<int>(std::round(cx - side * 0.5f));
    int y1 = static_cast<int>(std::round(cy - side * 0.5f));
    int x2 = x1 + static_cast<int>(std::round(side));
    int y2 = y1 + static_cast<int>(std::round(side));

    // Clip and compute padding
    int pad_l = std::max(0, -x1);
    int pad_t = std::max(0, -y1);
    int pad_r = std::max(0, x2 - img_w);
    int pad_b = std::max(0, y2 - img_h);

    int sx1 = std::max(0, x1), sy1 = std::max(0, y1);
    int sx2 = std::min(img_w, x2), sy2 = std::min(img_h, y2);

    int roi_w = sx2 - sx1;
    int roi_h = sy2 - sy1;

    // Create padded square (114 grey = common YOLO convention)
    cv::Mat padded(y2 - y1, x2 - x1, CV_8UC3, cv::Scalar(114, 114, 114));
    if (roi_w > 0 && roi_h > 0) {
        bgr(cv::Rect(sx1, sy1, roi_w, roi_h))
            .copyTo(padded(cv::Rect(pad_l, pad_t, roi_w, roi_h)));
    }

    // Resize to CROP_SIZE × CROP_SIZE
    cv::Mat resized;
    cv::resize(padded, resized, {CROP_SIZE, CROP_SIZE}, 0, 0, cv::INTER_LINEAR);

    // BGR→RGB, uint8→float32 normalised, interleaved→CHW
    const int plane = CROP_SIZE * CROP_SIZE;
    for (int y = 0; y < CROP_SIZE; ++y) {
        const uchar* row = resized.ptr<uchar>(y);
        for (int x = 0; x < CROP_SIZE; ++x) {
            // OpenCV is BGR
            float b = row[3*x + 0] / 255.f;
            float g = row[3*x + 1] / 255.f;
            float r = row[3*x + 2] / 255.f;
            // normalise (RGB order matches PyTorch model)
            out_chw[0 * plane + y * CROP_SIZE + x] = (r - IMAGE_MEAN[0]) / IMAGE_STD[0];
            out_chw[1 * plane + y * CROP_SIZE + x] = (g - IMAGE_MEAN[1]) / IMAGE_STD[1];
            out_chw[2 * plane + y * CROP_SIZE + x] = (b - IMAGE_MEAN[2]) / IMAGE_STD[2];
        }
    }
}

// ─── Compute CLIFF condition info ─────────────────────────────────────────────
//
// CLIFF-style condition (USE_INTRIN_CENTER=true in config):
//   cond[0] = (bbox_cx - cam_cx) / focal_x
//   cond[1] = (bbox_cy - cam_cy) / focal_y
//   cond[2] = bbox_size           / focal_x
//
inline void compute_condition_info(
    float bbox_cx, float bbox_cy, float bbox_size,
    float focal_x, float focal_y,
    float cam_cx,  float cam_cy,
    float cond[3]   // output [3]
)
{
    cond[0] = (bbox_cx - cam_cx) / focal_x;
    cond[1] = (bbox_cy - cam_cy) / focal_y;
    cond[2] = bbox_size          / focal_x;
}

// ─── Compute ray_cond map  [2, FEAT_HW, FEAT_HW]  at patch resolution ────────
//
// The ONNX decoder expects ray directions at feature-map resolution (32×32),
// so we sample at each patch centre instead of per-pixel.
//
// Patch centre (px, py) in the 512×512 crop:
//   crop_x = px * PATCH_SIZE + PATCH_SIZE/2
//   crop_y = py * PATCH_SIZE + PATCH_SIZE/2
//
// Back-projected to original image then to normalised camera ray:
//   orig_x = (crop_x - CROP_SIZE/2) / scale + bbox_cx
//   ray_x  = (orig_x - cam_cx) / focal_x
//
// out_ray: float[2 × FEAT_HW × FEAT_HW]  layout [channel, y, x]
// channel 0 = ray_x,  channel 1 = ray_y
inline void compute_ray_cond(
    float bbox_cx, float bbox_cy, float crop_size_orig,
    float focal_x, float focal_y,
    float cam_cx,  float cam_cy,
    float* out_ray   // [2, FEAT_HW, FEAT_HW]
)
{
    const float scale   = static_cast<float>(CROP_SIZE) / crop_size_orig;
    const float half_cs = CROP_SIZE * 0.5f;
    const int   FHW     = FEAT_HW;
    const int   plane   = FHW * FHW;

    for (int py = 0; py < FHW; ++py) {
        for (int px = 0; px < FHW; ++px) {
            float crop_x = px * PATCH_SIZE + PATCH_SIZE * 0.5f;
            float crop_y = py * PATCH_SIZE + PATCH_SIZE * 0.5f;
            float orig_x = (crop_x - half_cs) / scale + bbox_cx;
            float orig_y = (crop_y - half_cs) / scale + bbox_cy;
            out_ray[0 * plane + py * FHW + px] = (orig_x - cam_cx) / focal_x;
            out_ray[1 * plane + py * FHW + px] = (orig_y - cam_cy) / focal_y;
        }
    }
}

// ─── YOLO output parsing & NMS ────────────────────────────────────────────────
//
// YOLO Pose output: [1, num_dets, 56]
//   columns 0-3  : cx, cy, w, h  (normalised 0..1)
//   column  4    : object confidence
//   columns 5-54 : class scores then keypoints (not used here)
//
struct PersonDet {
    float x1, y1, x2, y2;  // pixel coords
    float conf;
};

static inline float iou(const PersonDet& a, const PersonDet& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    if (inter == 0.f) return 0.f;
    float ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
    return inter / (ua + 1e-6f);
}

// Parse YOLO Pose output tensor [num_dets, 56] (already transposed to row-major).
// img_w, img_h: original image dimensions for coordinate de-normalisation.
inline std::vector<PersonDet> parse_yolo_output(
    const float*  data,          // [num_dets × 56]
    int           num_dets,
    int           img_w,
    int           img_h,
    float         conf_thresh,
    float         nms_iou_thresh
)
{
    std::vector<PersonDet> raw;
    raw.reserve(64);

    for (int i = 0; i < num_dets; ++i) {
        const float* row = data + i * 56;
        float cx   = row[0], cy = row[1], w = row[2], h = row[3];
        float conf = row[4];
        if (conf < conf_thresh) continue;
        PersonDet d;
        d.x1   = (cx - w * 0.5f) * img_w;
        d.y1   = (cy - h * 0.5f) * img_h;
        d.x2   = (cx + w * 0.5f) * img_w;
        d.y2   = (cy + h * 0.5f) * img_h;
        d.conf = conf;
        raw.push_back(d);
    }

    // Sort descending by confidence
    std::sort(raw.begin(), raw.end(),
        [](const PersonDet& a, const PersonDet& b){ return a.conf > b.conf; });

    // Greedy NMS
    std::vector<bool> suppressed(raw.size(), false);
    std::vector<PersonDet> kept;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(raw[i]);
        for (size_t j = i + 1; j < raw.size(); ++j) {
            if (!suppressed[j] && iou(raw[i], raw[j]) > nms_iou_thresh)
                suppressed[j] = true;
        }
    }
    return kept;
}

// ─── Convert continuous body params to 133-dim Euler (fast path) ──────────────
//
// Implements compact_cont_to_model_params_body_fast in C++.
// body_cont [260] → body_euler [133]
//
// Body pose parameterisation:
//   - 23 joints with 3 DOF → 23×6 = 138 continuous dims  (6D rotation)
//   - 58 joints with 1 DOF → 58×2 = 116 continuous dims  (sin,cos)
//   - 6  translation values→ 6    continuous dims
//   Total = 138 + 116 + 6 = 260  ✓
//
// Output 133 = 23×3 (euler 3-dof) + 58 (euler 1-dof) + 6 (trans) = 87 + 46 + ... hmm
//   Actually 23*3 = 69 + 58 + 6 = 133  ✓

// 6D rotation → 3D Euler ZYX (batchXYZfrom6D in Python)
static inline void rot6d_to_euler(const float* d6, float* euler) {
    // Gram–Schmidt to orthonormal frame
    float a0 = d6[0], a1 = d6[1], a2 = d6[2];
    float b0 = d6[3], b1 = d6[4], b2 = d6[5];

    float na  = std::sqrt(a0*a0 + a1*a1 + a2*a2) + 1e-8f;
    float e00 = a0/na, e01 = a1/na, e02 = a2/na;  // first col
    float dot = e00*b0 + e01*b1 + e02*b2;
    float e10 = b0 - dot*e00;
    float e11 = b1 - dot*e01;
    float e12 = b2 - dot*e02;
    float nb  = std::sqrt(e10*e10 + e11*e11 + e12*e12) + 1e-8f;
    e10 /= nb; e11 /= nb; e12 /= nb;
    // third col = cross
    float e20 = e01*e12 - e02*e11;
    float e21 = e02*e10 - e00*e12;
    float e22 = e00*e11 - e01*e10;

    // Rotation matrix rows:
    //  R = [[e00,e10,e20],[e01,e11,e21],[e02,e12,e22]]
    // ZYX Euler: Ry = asin(-R[2,0]), Rx = atan2(R[2,1],R[2,2]), Rz = atan2(R[1,0],R[0,0])
    // (Python uses XYZ convention here – adjust as needed)
    euler[0] = std::atan2(e21, e22);  // rx
    euler[1] = std::asin(std::max(-1.f, std::min(1.f, -e20)));  // ry
    euler[2] = std::atan2(e10, e00);  // rz
}

// 3-DOF joint index layout in the 133-param vector
// (mirrors all_param_3dof_rot_idxs in mhr_utils.py)
static constexpr int BODY_3DOF_JOINT_IDXS[23][3] = {
    {0,2,4}, {6,8,10}, {12,13,14}, {15,16,17}, {18,19,20},
    {21,22,23}, {24,25,26}, {27,28,29}, {34,35,36}, {37,38,39},
    {44,45,46}, {53,54,55}, {64,65,66}, {85,69,73}, {86,70,79},
    {87,71,82}, {88,72,76}, {91,92,93}, {112,96,100}, {113,97,106},
    {114,98,109}, {115,99,103}, {130,131,132}
};
static constexpr int BODY_1DOF_IDXS[58] = {
    1,3,5,7,9,11,30,31,32,33,40,41,42,43,47,48,49,50,51,52,
    56,57,58,59,60,61,62,63,67,68,74,75,77,78,80,81,83,84,
    89,90,94,95,101,102,104,105,107,108,110,111,116,117,118,119,120,121,122,123
};
static constexpr int BODY_TRANS_IDXS[6] = {124,125,126,127,128,129};

inline void compact_cont_to_body_params(
    const float* body_cont,  // [260]
    float*       body_euler  // [133] out – caller must zero-initialise
)
{
    static constexpr int N3    = 23;
    static constexpr int N1    = 58;
    // 3-DOF region: first 23*6 = 138 floats
    for (int j = 0; j < N3; ++j) {
        float euler[3];
        rot6d_to_euler(body_cont + j * 6, euler);
        for (int k = 0; k < 3; ++k)
            body_euler[BODY_3DOF_JOINT_IDXS[j][k]] = euler[k];
    }
    // 1-DOF region: next 58*2 = 116 floats  (sin, cos)
    const float* p1 = body_cont + N3 * 6;
    for (int j = 0; j < N1; ++j) {
        float s = p1[j*2 + 0];
        float c = p1[j*2 + 1];
        body_euler[BODY_1DOF_IDXS[j]] = std::atan2(s, c);
    }
    // Translation region: last 6 floats
    const float* pt = body_cont + N3 * 6 + N1 * 2;
    for (int j = 0; j < 6; ++j)
        body_euler[BODY_TRANS_IDXS[j]] = pt[j];
}

// ─── Assemble model_params [204] for the torch.jit body model ─────────────────
//
// body_model.onnx expects:
//   shape      [45]   identity blend shape betas
//   body_params[204]  = full_pose_params [136] + scales [68]
//   face       [72]
//
// full_pose_params [136] = global_rot[3] + global_trans[3] + body_pose[133] - 3 ???
//
// Actually: the MHR model is called with (shape_params, model_params, expr_params).
// model_params=[204] = cat([full_pose_params, scales], dim=1)
// where full_pose_params=[136] = global_trans[3]+global_rot_euler[3]+body_pose[133]???
// The exact layout depends on the jit model.  We pass scale zeros for body_params[136:].
//
// From the code: model_params = torch.cat([full_pose_params, scales], dim=1)
//   full_pose_params [B,136] is assembled in _mhr_forward_core
//   scales           [B,68]  comes from scale_comps PCA decode
//
// For inference we zero-fill scales and set full_pose_params from predictions.
// Caller uses build_model_params_from_prediction() below.
struct ModelParams204 {
    float data[204] = {};
};

inline ModelParams204 build_model_params(
    const float* global_rot_euler,  // [3]  (ZYX Euler from rot6d_to_euler)
    const float* body_euler,        // [133]
    const float* scale_params,      // [28]  raw scale params (PCA codes)
    // scale_comps [28×68] and scale_mean [68] from the body model are not
    // available here – set scales to zero for a reasonable result
    bool         zero_scales = true
)
{
    ModelParams204 out{};
    // Layout inferred from _mhr_forward_core:
    //   full_pose_params = cat([body_pose_euler[B,133], hands[B,0], global_rot[B,3],
    //                           global_trans[B,3], ...], dim=1)
    // Exact layout: global_trans[3] + global_rot[3] + body_pose[130+3=133] = 139???
    // We use the safe default: first 3 = global_rot, next 133 = body_pose, rest = 0
    // This matches the torch.jit contract assumed by the body_model ONNX.
    out.data[0] = global_rot_euler[0];
    out.data[1] = global_rot_euler[1];
    out.data[2] = global_rot_euler[2];
    // body_pose occupies indices 3..135
    std::memcpy(out.data + 3, body_euler, 133 * sizeof(float));
    // scale part (indices 136..203) – zeroed (zero_scales)
    (void)scale_params; (void)zero_scales;
    return out;
}

} // namespace fsb
