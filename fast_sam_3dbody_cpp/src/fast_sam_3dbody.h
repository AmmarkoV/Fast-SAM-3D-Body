#pragma once
// ============================================================================
// fast_sam_3dbody.h  –  C++ interface for SAM-3D-Body pipeline
//
// Pipeline stages
// ───────────────
//  1. YOLO Pose (ONNX / TRT engine)      → person bounding boxes
//  2. Backbone (backbone.onnx)            → image feature map  [B,1280,32,32]
//  3. Decoder  (decoder.onnx)             → pose token         [B,1024]
//  4. MHR head  (pipeline.gguf, ggml)    → raw pose params     [B,519]
//  5. Camera head (pipeline.gguf, ggml)  → camera params       [B,3]
//  6. Body model  (body_model.onnx)       → vertices + joints
//
// Inputs expected in BGR uint8 (OpenCV default).
// ============================================================================

#include <array>
#include <string>
#include <vector>

namespace fsb {

// ─── Output per detected person ──────────────────────────────────────────────
struct MHRResult {
    // Bounding box in original image  [x1, y1, x2, y2]
    std::array<float, 4> bbox{};

    float focal_length = 0.f;          // Estimated / default focal length (pixels)

    // Camera translation  [tx, ty, tz]
    std::array<float, 3> pred_cam_t{};

    // ── Pose params ──────────────────────────────────────────────────────────
    // Global orientation – Euler ZYX  [rx, ry, rz]
    std::array<float, 3> global_rot{};

    // Body pose – MHR 133-dim Euler angles
    std::vector<float> body_pose;      // [133]

    // Shape betas  (SMPL-like identity blend shapes)
    std::vector<float> shape;          // [45]

    // Scale parameters
    std::vector<float> scale;          // [28]

    // Hand pose (left 54 + right 54 = 108)
    std::vector<float> hand_pose;      // [108]

    // Face expression
    std::vector<float> face_params;    // [72]

    // ── Geometry (populated when Pipeline::Config::skip_body_model = false) ──
    std::vector<float> pred_vertices;  // [18439 × 3]  SMPL-like mesh
    std::vector<float> keypoints_3d;   // [70 × 3]     3-D joints
    std::vector<float> keypoints_2d;   // [70 × 2]     projected 2-D
};

// ─── Pipeline configuration ───────────────────────────────────────────────────
struct PipelineConfig {
    // Paths
    std::string onnx_dir;           // Directory with backbone.onnx, decoder.onnx, body_model.onnx
    std::string gguf_path;          // Path to pipeline.gguf
    std::string yolo_path;          // YOLO model: .onnx or .engine (TRT)

    // Device
    int  cuda_device    = 0;        // CUDA device (-1 = CPU only)
    bool use_trt_ep     = false;    // Enable ONNX Runtime TensorRT EP (requires TRT install)
    bool use_fp16       = true;     // FP16 for ONNX EP

    // Inference options
    bool skip_body_model = false;   // Skip body model – no vertices/keypoints (faster)
    float person_thresh  = 0.50f;  // YOLO confidence threshold
    float person_nms_iou = 0.45f;  // YOLO NMS IoU threshold

    // Camera intrinsics – set to 0 to use default (fx = image_width)
    float focal_x = 0.f;
    float focal_y = 0.f;
    float principal_x = 0.f;       // 0 = image_width  / 2
    float principal_y = 0.f;       // 0 = image_height / 2
};

// ─── Pipeline class ───────────────────────────────────────────────────────────
class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    // Non-copyable, moveable
    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&)                 = default;
    Pipeline& operator=(Pipeline&&)      = default;

    // Load all models.  Returns false on failure.
    bool load(const PipelineConfig& cfg);

    // Release all resources.
    void free();

    // ── Core inference ────────────────────────────────────────────────────────
    // Process a single BGR image (width × height × 3, uint8).
    // Returns one MHRResult per detected person.
    std::vector<MHRResult> process_bgr(const uint8_t* bgr,
                                       int width, int height);

    // Convenience overload for OpenCV Mat (must be CV_8UC3 BGR).
    // Declared only if OpenCV is available; implemented in fast_sam_3dbody.cpp.
    struct cv_mat_tag {};
#if defined(FSB_HAS_OPENCV_MAT)
    std::vector<MHRResult> process_mat(const void* cv_mat_ptr);
#endif

    // True after a successful load().
    bool is_loaded() const;

    // Print loaded model info.
    void print_info() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace fsb
