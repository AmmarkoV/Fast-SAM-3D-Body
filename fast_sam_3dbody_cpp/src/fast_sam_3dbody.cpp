// ============================================================================
// fast_sam_3dbody.cpp  –  SAM-3D-Body inference pipeline
//
// Stage map:
//  YOLO (ONNX/TRT)  → person bboxes
//  backbone.onnx    → [B,1280,32,32]  image features
//  decoder.onnx     → [B,1024]        pose token
//  pipeline.gguf    → [B,519]+[B,3]   MHR params + camera params  (ggml)
//  body_model.onnx  → [B,18439,3]     SMPL-like vertices  (optional)
// ============================================================================

#define FSB_HAS_OPENCV_MAT  1

#include "fast_sam_3dbody.h"
#include "preprocess.hpp"

// ── ggml headers ─────────────────────────────────────────────────────────────
#if __has_include(<ggml/ggml.h>)
#  include <ggml/ggml.h>
#  include <ggml/ggml-alloc.h>
#  include <ggml/ggml-backend.h>
#  include <ggml/ggml-cpu.h>
#  include <ggml/gguf.h>
#else
#  include <ggml.h>
#  include <ggml-alloc.h>
#  include <ggml-backend.h>
#  include <ggml-cpu.h>
#  include <gguf.h>
#endif
#if defined(GGML_USE_CUDA)
#  if __has_include(<ggml/ggml-cuda.h>)
#    include <ggml/ggml-cuda.h>
#  elif __has_include(<ggml-cuda.h>)
#    include <ggml-cuda.h>
#  endif
#endif

// ── ONNX Runtime ─────────────────────────────────────────────────────────────
#include <onnxruntime_cxx_api.h>

// ── OpenCV ───────────────────────────────────────────────────────────────────
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// ── STL ──────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fsb {

// ─────────────────────────────────────────────────────────────────────────────
// Timing helper
// ─────────────────────────────────────────────────────────────────────────────
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// GGUF metadata
// ─────────────────────────────────────────────────────────────────────────────
struct GGUFMeta {
    uint32_t decoder_dim  = 1024;
    uint32_t npose        = 519;
    uint32_t cam_out_dim  = 3;
    uint32_t num_vertices = 18439;
    uint32_t num_kps      = 70;
    float    default_focal= 800.f;
    float    person_thresh= 0.5f;
    float    nms_iou      = 0.45f;
};

static uint32_t gguf_u32(gguf_context* c, const char* k, uint32_t def=0) {
    int id = gguf_find_key(c, k); return id>=0 ? gguf_get_val_u32(c, id) : def;
}

// ─────────────────────────────────────────────────────────────────────────────
// Small FFN  (MHR head / camera head)  – plain C++ CPU matmul
//
// Architecture: Linear(in, hid) + ReLU + Linear(hid, out)
// Weights loaded from GGUF (f16 weights converted to f32 on load).
//   {prefix}.fc0.{weight,bias}   –  shape [hid, in] / [hid]
//   {prefix}.fc1.{weight,bias}   –  shape [out, hid] / [out]
//
// Row-major storage: w0[i * in_dim + j] = weight from input j to hidden i.
// Inference: y = relu(x @ w0.T + b0) @ w1.T + b1
// ─────────────────────────────────────────────────────────────────────────────
struct CFFN {
    std::vector<float> w0, b0, w1, b1;
    int in_dim=0, hid_dim=0, out_dim=0;
};

static bool cffn_load(CFFN& ffn,
                      gguf_context*  gctx,
                      ggml_context*  wctx,   // created by gguf_init_from_file
                      FILE*          fp,
                      size_t         data_base,
                      const std::string& prefix)
{
    // Read one weight tensor by name; convert f16→f32 if needed.
    // Shape comes from the ggml context created alongside the gguf context.
    auto read_f32 = [&](const char* suffix, std::vector<float>& out) -> bool
    {
        std::string name = prefix + suffix;

        // Get shape from the ggml context
        ggml_tensor* t = ggml_get_tensor(wctx, name.c_str());
        if (!t) {
            fprintf(stderr, "[FFN] tensor not found: %s\n", name.c_str());
            return false;
        }
        size_t n    = ggml_nelements(t);
        int64_t idx = gguf_find_tensor(gctx, name.c_str());
        size_t  off = gguf_get_tensor_offset(gctx, idx);
        int     type = (int)gguf_get_tensor_type(gctx, idx);

        std::fseek(fp, (long)(data_base + off), SEEK_SET);
        out.resize(n);
        if (type == GGML_TYPE_F32) {
            if (std::fread(out.data(), sizeof(float), n, fp) != n) return false;
        } else if (type == GGML_TYPE_F16) {
            std::vector<uint16_t> tmp(n);
            if (std::fread(tmp.data(), sizeof(uint16_t), n, fp) != n) return false;
            ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int)n);
        } else {
            fprintf(stderr, "[FFN] unsupported weight type %d for %s\n", type, name.c_str());
            return false;
        }
        return true;
    };

    // Retrieve dimension info from ggml context tensors
    auto get_tensor = [&](const char* suffix) -> ggml_tensor* {
        return ggml_get_tensor(wctx, (prefix + suffix).c_str());
    };

    if (!read_f32(".fc0.weight", ffn.w0)) return false;
    if (!read_f32(".fc0.bias",   ffn.b0)) return false;
    if (!read_f32(".fc1.weight", ffn.w1)) return false;
    if (!read_f32(".fc1.bias",   ffn.b1)) return false;

    // ne[0]=Cin, ne[1]=Cout for weight matrices (GGML column-major vs numpy row-major)
    auto* w0t = get_tensor(".fc0.weight");
    auto* w1t = get_tensor(".fc1.weight");
    ffn.in_dim  = (int)w0t->ne[0];
    ffn.hid_dim = (int)w0t->ne[1];
    ffn.out_dim = (int)w1t->ne[1];
    return true;
}

// y = relu(x @ w.T + b)   x:[B,K]  w:[N,K]  b:[N]  → out:[B,N]
static void linear_relu(const float* x, const float* w, const float* b,
                        float* y, int B, int K, int N, bool relu)
{
    for (int bi = 0; bi < B; ++bi) {
        for (int n = 0; n < N; ++n) {
            float s = b[n];
            const float* xr = x + bi * K;
            const float* wr = w + n * K;
            for (int k = 0; k < K; ++k) s += xr[k] * wr[k];
            y[bi * N + n] = relu ? std::max(0.f, s) : s;
        }
    }
}

static std::vector<float> cffn_run(const CFFN& ffn, const float* x, int B)
{
    std::vector<float> h(B * ffn.hid_dim);
    linear_relu(x,       ffn.w0.data(), ffn.b0.data(),
                h.data(), B, ffn.in_dim,  ffn.hid_dim, true);

    std::vector<float> y(B * ffn.out_dim);
    linear_relu(h.data(), ffn.w1.data(), ffn.b1.data(),
                y.data(),  B, ffn.hid_dim, ffn.out_dim, false);
    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX Runtime session wrapper
// ─────────────────────────────────────────────────────────────────────────────
struct OrtSession {
    Ort::Env*             env     = nullptr;
    Ort::Session*         session = nullptr;
    Ort::MemoryInfo       mem_info{ nullptr };
    std::vector<std::string>       input_names_s,  output_names_s;
    std::vector<const char*>       input_names,    output_names;

    bool load(Ort::Env& e, const std::string& path, bool cuda, int device,
              bool fp16_io = false, bool trt_ep = false)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cuda && !trt_ep) {
            OrtCUDAProviderOptions cp{};
            cp.device_id = device;
            opts.AppendExecutionProvider_CUDA(cp);
        }
#if defined(USE_TENSORRT_EP)
        if (trt_ep) {
            OrtTensorRTProviderOptions tp{};
            tp.device_id = device;
            tp.trt_fp16_enable = fp16_io ? 1 : 0;
            opts.AppendExecutionProvider_TensorRT(tp);
        }
#endif
        try {
            session = new Ort::Session(e, path.c_str(), opts);
        } catch (const Ort::Exception& ex) {
            fprintf(stderr, "[ORT] load '%s' failed: %s\n", path.c_str(), ex.what());
            return false;
        }
        env = &e;

        Ort::AllocatorWithDefaultOptions alloc;
        size_t n_in  = session->GetInputCount();
        size_t n_out = session->GetOutputCount();
        input_names_s.resize(n_in);
        output_names_s.resize(n_out);
        input_names.resize(n_in);
        output_names.resize(n_out);
        for (size_t i = 0; i < n_in;  ++i)
            input_names_s[i]  = session->GetInputNameAllocated(i,  alloc).get(),
            input_names[i]    = input_names_s[i].c_str();
        for (size_t i = 0; i < n_out; ++i)
            output_names_s[i] = session->GetOutputNameAllocated(i, alloc).get(),
            output_names[i]   = output_names_s[i].c_str();

        mem_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        return true;
    }

    // Run with a single float32 input tensor (for backbone)
    std::vector<float> run1(const float* in_data,
                             const std::vector<int64_t>& in_shape,
                             size_t out_elems)
    {
        Ort::Value in_t = Ort::Value::CreateTensor<float>(
            mem_info, const_cast<float*>(in_data), in_shape[0]*in_shape[1]*in_shape[2]*in_shape[3],
            in_shape.data(), in_shape.size());
        auto out = session->Run(Ort::RunOptions{nullptr},
                                input_names.data(),  &in_t,    1,
                                output_names.data(), output_names.size());
        std::vector<float> result(out_elems);
        auto* src = out[0].GetTensorMutableData<float>();
        std::memcpy(result.data(), src, out_elems * sizeof(float));
        return result;
    }

    void free() { delete session; session = nullptr; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline::Impl
// ─────────────────────────────────────────────────────────────────────────────
struct Pipeline::Impl {
    PipelineConfig  cfg;
    GGUFMeta        meta;
    bool            loaded = false;

    // ONNX Runtime
    Ort::Env        ort_env{ORT_LOGGING_LEVEL_WARNING, "fast_sam_3dbody"};
    OrtSession      sess_yolo, sess_backbone, sess_decoder, sess_body;

    // CPU FFNs for MHR + camera heads (weights loaded from GGUF)
    CFFN mhr_ffn, cam_ffn;

    // ── load ──────────────────────────────────────────────────────────────────
    bool load(const PipelineConfig& c) {
        cfg = c;

        bool cuda = cfg.cuda_device >= 0;
        int  dev  = cfg.cuda_device;

        // ── ONNX sessions ─────────────────────────────────────────────────────
        auto opath = [&](const char* f) {
            return cfg.onnx_dir + "/" + f;
        };

        printf("[FSB] Loading backbone … "); fflush(stdout);
        if (!sess_backbone.load(ort_env, opath("backbone.onnx"), cuda, dev,
                                cfg.use_fp16, false))
            return false;
        printf("OK\n");

        printf("[FSB] Loading decoder  … "); fflush(stdout);
        if (!sess_decoder.load(ort_env, opath("decoder.onnx"), cuda, dev,
                               cfg.use_fp16, false))
            return false;
        printf("OK\n");

        if (!cfg.skip_body_model) {
            // Prefer body_model.onnx; fall back gracefully to body_model.pt
            // (body_model.pt requires LibTorch – planned via ggml, see TODO below)
            std::string bm_onnx = opath("body_model.onnx");
            std::ifstream bm_check(bm_onnx);
            if (bm_check.good()) {
                bm_check.close();
                printf("[FSB] Loading body_model.onnx … "); fflush(stdout);
                if (!sess_body.load(ort_env, bm_onnx, cuda, dev, false, false))
                    return false;
                printf("OK\n");
            } else {
                printf("[FSB] body_model.onnx not found; vertex output disabled.\n");
                printf("[FSB] (body_model.pt exists – ggml implementation planned)\n");
                // Not a fatal error: vertices / keypoints will be empty in MHRResult
            }
        }

        // YOLO – optional (might not exist for image-only usage)
        if (!cfg.yolo_path.empty()) {
            printf("[FSB] Loading YOLO … "); fflush(stdout);
            if (!sess_yolo.load(ort_env, cfg.yolo_path, cuda, dev, false, false)) {
                fprintf(stderr, "[FSB] YOLO load failed – detection disabled\n");
            } else {
                printf("OK\n");
            }
        }

        // ── ggml / GGUF ───────────────────────────────────────────────────────
        printf("[FSB] Loading pipeline.gguf … "); fflush(stdout);
        if (!load_gguf(cfg.gguf_path)) return false;
        printf("OK\n");

        loaded = true;
        return true;
    }

    bool load_gguf(const std::string& path) {
        // Only use gguf for metadata + weight bytes; inference runs in plain C++.
        gguf_context* gctx = nullptr;
        ggml_context* tmp_ctx = nullptr;
        { struct gguf_init_params p{true, &tmp_ctx}; gctx = gguf_init_from_file(path.c_str(), p); }
        if (!gctx) { fprintf(stderr, "[FSB] Cannot open GGUF: %s\n", path.c_str()); return false; }

        meta.decoder_dim   = gguf_u32(gctx, "sam3dbody.decoder_dim", 1024);
        meta.npose         = gguf_u32(gctx, "sam3dbody.npose",        519);
        meta.default_focal = 800.f;
        meta.person_thresh = cfg.person_thresh;
        meta.nms_iou       = cfg.person_nms_iou;

        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) { gguf_free(gctx); if (tmp_ctx) ggml_free(tmp_ctx); return false; }
        size_t data_base = gguf_get_data_offset(gctx);

        bool ok = cffn_load(mhr_ffn, gctx, tmp_ctx, fp, data_base, "mhr_proj")
               && cffn_load(cam_ffn, gctx, tmp_ctx, fp, data_base, "cam_proj");

        std::fclose(fp);
        gguf_free(gctx);
        if (tmp_ctx) ggml_free(tmp_ctx);
        if (!ok) return false;

        printf("[FSB] FFNs: MHR(%dx%d->%d) Cam(%dx%d->%d)\n",
               mhr_ffn.in_dim, mhr_ffn.hid_dim, mhr_ffn.out_dim,
               cam_ffn.in_dim, cam_ffn.hid_dim, cam_ffn.out_dim);
        return true;
    }

    // ── process_bgr ───────────────────────────────────────────────────────────
    std::vector<MHRResult> process_bgr(const uint8_t* bgr, int W, int H) {
        cv::Mat img(H, W, CV_8UC3, const_cast<uint8_t*>(bgr));
        return process_mat(img, W, H);
    }

    std::vector<MHRResult> process_mat(const cv::Mat& bgr, int W, int H) {
        auto t_total = Clock::now();

        // ── camera intrinsics ─────────────────────────────────────────────────
        float fx = (cfg.focal_x    > 0.f) ? cfg.focal_x    : float(W);
        float fy = (cfg.focal_y    > 0.f) ? cfg.focal_y    : float(W);
        float cx = (cfg.principal_x> 0.f) ? cfg.principal_x: float(W) * 0.5f;
        float cy = (cfg.principal_y> 0.f) ? cfg.principal_y: float(H) * 0.5f;

        // ── person detection ──────────────────────────────────────────────────
        auto t0 = Clock::now();
        std::vector<PersonDet> dets;

        if (sess_yolo.session) {
            // Resize to YOLO input (640×640 is common)
            const int YW = 640, YH = 640;
            cv::Mat yolo_in;
            cv::resize(bgr, yolo_in, {YW, YH});
            // HWC uint8 → CHW float32 [0,1]
            std::vector<float> yolo_buf(3 * YH * YW);
            for (int y = 0; y < YH; ++y) {
                const uchar* row = yolo_in.ptr<uchar>(y);
                for (int x = 0; x < YW; ++x) {
                    yolo_buf[0*YH*YW + y*YW + x] = row[3*x+2] / 255.f; // R
                    yolo_buf[1*YH*YW + y*YW + x] = row[3*x+1] / 255.f; // G
                    yolo_buf[2*YH*YW + y*YW + x] = row[3*x+0] / 255.f; // B
                }
            }
            // Run YOLO – output shape: [1, num_dets, 56] (or [1, 56, num_dets] depending on export)
            Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> in_shape{1, 3, YH, YW};
            Ort::Value in_t = Ort::Value::CreateTensor<float>(
                mi, yolo_buf.data(), yolo_buf.size(), in_shape.data(), 4);

            try {
                auto outs = sess_yolo.session->Run(
                    Ort::RunOptions{nullptr},
                    sess_yolo.input_names.data(),  &in_t,  1,
                    sess_yolo.output_names.data(), 1);

                auto info   = outs[0].GetTensorTypeAndShapeInfo();
                auto shape  = info.GetShape();
                // shape is typically [1, 56, num_dets] for YOLOv8/11 pose
                // transpose to [num_dets, 56] if needed
                int nd = 0;
                const float* raw = outs[0].GetTensorData<float>();
                std::vector<float> row_major;

                if (shape.size() == 3) {
                    if (shape[1] == 56) {
                        // [1, 56, num_dets] → need transpose
                        nd = (int)shape[2];
                        row_major.resize(nd * 56);
                        for (int j = 0; j < nd; ++j)
                            for (int k = 0; k < 56; ++k)
                                row_major[j*56+k] = raw[k*nd + j];
                    } else {
                        // [1, num_dets, 56]
                        nd = (int)shape[1];
                        row_major.assign(raw, raw + nd * 56);
                    }
                }
                // scale from YOLO 640×640 space to original image space
                float sx = float(W) / YW, sy = float(H) / YH;
                dets = parse_yolo_output(row_major.data(), nd,
                                         cfg.person_thresh, cfg.person_nms_iou);
                for (auto& d : dets) {
                    d.x1 *= sx; d.x2 *= sx;
                    d.y1 *= sy; d.y2 *= sy;
                }
            } catch (const Ort::Exception& e) {
                fprintf(stderr, "[FSB] YOLO inference error: %s\n", e.what());
            }
        }

        // Fallback: full image as single detection
        if (dets.empty()) {
            dets.push_back({ 0.f, 0.f, float(W), float(H), 1.f });
        }
        printf("[FSB] detection: %.1f ms  persons: %zu\n", ms(t0), dets.size());

        // ── per-person crops ──────────────────────────────────────────────────
        const int B = (int)dets.size();
        const int plane = CROP_SIZE * CROP_SIZE;

        // Pre-allocate batch buffers
        const int ray_plane = FEAT_HW * FEAT_HW;
        std::vector<float> batch_crops   (B * 3 * plane);
        std::vector<float> batch_cond    (B * 3);
        std::vector<float> batch_ray     (B * 2 * ray_plane);
        std::vector<float> crop_cx_v(B), crop_cy_v(B), crop_sz_v(B);

        t0 = Clock::now();
        for (int i = 0; i < B; ++i) {
            const auto& d = dets[i];
            float* img_ptr = batch_crops.data() + i * 3 * plane;
            float& ccx     = crop_cx_v[i];
            float& ccy     = crop_cy_v[i];
            float& csz     = crop_sz_v[i];

            crop_and_normalise(bgr, d.x1, d.y1, d.x2, d.y2,
                               img_ptr, ccx, ccy, csz);

            float* cond_ptr = batch_cond.data() + i * 3;
            compute_condition_info(ccx, ccy, csz, fx, fy, cx, cy, cond_ptr);

            float* ray_ptr = batch_ray.data() + i * 2 * ray_plane;
            compute_ray_cond(ccx, ccy, csz, fx, fy, cx, cy, ray_ptr);
        }
        printf("[FSB] preprocess: %.1f ms\n", ms(t0));

        // ── backbone ─────────────────────────────────────────────────────────
        t0 = Clock::now();
        const int FEAT_HW = CROP_SIZE / 16;   // 32
        const int BACKBONE_DIM = 1280;
        const size_t feat_elems = (size_t)B * BACKBONE_DIM * FEAT_HW * FEAT_HW;

        std::vector<int64_t> img_shape{B, 3, CROP_SIZE, CROP_SIZE};
        Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value img_t = Ort::Value::CreateTensor<float>(
            mi, batch_crops.data(), batch_crops.size(), img_shape.data(), 4);
        auto backbone_out = sess_backbone.session->Run(
            Ort::RunOptions{nullptr},
            sess_backbone.input_names.data(),  &img_t,  1,
            sess_backbone.output_names.data(), 1);
        const float* feat_ptr = backbone_out[0].GetTensorData<float>();
        std::vector<float> features(feat_ptr, feat_ptr + feat_elems);
        printf("[FSB] backbone:   %.1f ms\n", ms(t0));

        // ── decoder ──────────────────────────────────────────────────────────
        t0 = Clock::now();
        const int DECODER_DIM = (int)meta.decoder_dim;
        const size_t token_elems = (size_t)B * DECODER_DIM;

        std::vector<int64_t> feat_shape{B, BACKBONE_DIM, FEAT_HW, FEAT_HW};
        std::vector<int64_t> cond_shape{B, 3};
        std::vector<int64_t> ray_shape {B, 2, FEAT_HW, FEAT_HW};

        Ort::Value feat_t = Ort::Value::CreateTensor<float>(
            mi, features.data(), features.size(), feat_shape.data(), 4);
        Ort::Value cond_t = Ort::Value::CreateTensor<float>(
            mi, batch_cond.data(), batch_cond.size(), cond_shape.data(), 2);
        Ort::Value ray_t  = Ort::Value::CreateTensor<float>(
            mi, batch_ray.data(), batch_ray.size(), ray_shape.data(), 4);

        std::vector<Ort::Value> dec_inputs;
        dec_inputs.push_back(std::move(feat_t));
        dec_inputs.push_back(std::move(cond_t));
        dec_inputs.push_back(std::move(ray_t));

        std::vector<const char*>& dec_in_names  = sess_decoder.input_names;
        std::vector<const char*>& dec_out_names = sess_decoder.output_names;

        auto decoder_out = sess_decoder.session->Run(
            Ort::RunOptions{nullptr},
            dec_in_names.data(),  dec_inputs.data(),  dec_inputs.size(),
            dec_out_names.data(), 1);
        const float* token_ptr = decoder_out[0].GetTensorData<float>();
        std::vector<float> pose_tokens(token_ptr, token_ptr + token_elems);
        printf("[FSB] decoder:    %.1f ms\n", ms(t0));

        // ── MHR head (CPU FFN) ────────────────────────────────────────────────
        t0 = Clock::now();
        std::vector<float> mhr_raw  = cffn_run(mhr_ffn, pose_tokens.data(), B);
        std::vector<float> cam_raw  = cffn_run(cam_ffn, pose_tokens.data(), B);
        printf("[FSB] MHR FFN:    %.1f ms\n", ms(t0));

        // ── body model (optional) ─────────────────────────────────────────────
        std::vector<float> all_verts, all_skel;
        if (!cfg.skip_body_model && sess_body.session) {
            t0 = Clock::now();
            // Build per-person body model inputs
            const int NPOSE = (int)meta.npose;
            std::vector<float> batch_shape  (B * 45, 0.f);
            std::vector<float> batch_bparams(B * 204, 0.f);
            std::vector<float> batch_face   (B * 72,  0.f);

            for (int i = 0; i < B; ++i) {
                const float* raw_i = mhr_raw.data() + i * NPOSE;
                // Parse: global_rot_6d[6] + body_cont[260] + shape[45] + scale[28] + hand[108] + face[72]
                const float* global_rot_6d  = raw_i;
                const float* body_cont      = raw_i + 6;
                const float* shape          = raw_i + 266;
                const float* face           = raw_i + 447;

                // Convert global rot 6D → Euler
                float global_rot_euler[3];
                rot6d_to_euler(global_rot_6d, global_rot_euler);

                // Convert body continuous params → 133-dim Euler
                float body_euler[133] = {};
                compact_cont_to_body_params(body_cont, body_euler);

                // Build model_params [204]
                ModelParams204 mp = build_model_params(global_rot_euler, body_euler, nullptr, true);

                // Copy into batch buffers
                std::memcpy(batch_shape.data()   + i * 45,  shape, 45  * sizeof(float));
                std::memcpy(batch_bparams.data() + i * 204, mp.data, 204 * sizeof(float));
                std::memcpy(batch_face.data()    + i * 72,  face,  72  * sizeof(float));
            }

            std::vector<int64_t> shape_sh  {B, 45};
            std::vector<int64_t> bparam_sh {B, 204};
            std::vector<int64_t> face_sh   {B, 72};

            Ort::Value shape_t  = Ort::Value::CreateTensor<float>(mi, batch_shape.data(),   B*45,  shape_sh.data(),  2);
            Ort::Value bparam_t = Ort::Value::CreateTensor<float>(mi, batch_bparams.data(), B*204, bparam_sh.data(), 2);
            Ort::Value face_t   = Ort::Value::CreateTensor<float>(mi, batch_face.data(),    B*72,  face_sh.data(),   2);

            // apply_correctives = False (constant bool tensor)
            bool corr_val = false;
            std::vector<int64_t> scalar_sh{};
            Ort::Value corr_t = Ort::Value::CreateTensor<bool>(mi, &corr_val, 1,
                                                                scalar_sh.data(), 0);

            std::vector<Ort::Value> body_ins;
            body_ins.push_back(std::move(shape_t));
            body_ins.push_back(std::move(bparam_t));
            body_ins.push_back(std::move(face_t));
            body_ins.push_back(std::move(corr_t));

            auto body_out = sess_body.session->Run(
                Ort::RunOptions{nullptr},
                sess_body.input_names.data(),  body_ins.data(),  4,
                sess_body.output_names.data(), 2);

            const float* vp = body_out[0].GetTensorData<float>();
            const float* sp = body_out[1].GetTensorData<float>();
            size_t vn = (size_t)B * 18439 * 3;
            size_t sn = (size_t)B * 127   * 8;
            all_verts.assign(vp, vp + vn);
            all_skel.assign(sp,  sp + sn);
            printf("[FSB] body_model: %.1f ms\n", ms(t0));
        }

        // ── assemble MHRResult per person ────────────────────────────────────
        std::vector<MHRResult> results(B);
        const int NPOSE = (int)meta.npose;

        for (int i = 0; i < B; ++i) {
            MHRResult& r   = results[i];
            const auto& d  = dets[i];
            const float* p = mhr_raw.data() + i * NPOSE;

            r.bbox = { d.x1, d.y1, d.x2, d.y2 };

            // Camera
            const float* cam = cam_raw.data() + i * 3;
            r.pred_cam_t = { cam[0], cam[1], cam[2] };
            r.focal_length = fx;

            // Global rotation 6D → Euler
            const float* g6d = p;
            float ge[3];
            rot6d_to_euler(g6d, ge);
            r.global_rot = { ge[0], ge[1], ge[2] };

            // Body pose
            const float* bc = p + 6;
            float be[133] = {};
            compact_cont_to_body_params(bc, be);
            r.body_pose.assign(be, be + 133);

            // Shape [45]
            r.shape.assign(p + 266, p + 266 + 45);

            // Scale [28]
            r.scale.assign(p + 311, p + 311 + 28);

            // Hand pose [108]
            r.hand_pose.assign(p + 339, p + 339 + 108);

            // Face [72]
            r.face_params.assign(p + 447, p + 447 + 72);

            // Vertices (optional)
            if (!all_verts.empty()) {
                size_t off = (size_t)i * 18439 * 3;
                r.pred_vertices.assign(all_verts.begin() + off,
                                       all_verts.begin() + off + 18439*3);
                // Flip y,z to match camera system (matches Python code: [1,2] *= -1)
                for (size_t k = 0; k < 18439; ++k) {
                    r.pred_vertices[k*3 + 1] *= -1.f;
                    r.pred_vertices[k*3 + 2] *= -1.f;
                }
            }
        }

        printf("[FSB] total: %.1f ms  (%d persons)\n", ms(t_total), B);
        return results;
    }

    void free_all() {
        // CFFN weights are plain vectors – cleaned up automatically
        mhr_ffn = CFFN{};
        cam_ffn = CFFN{};
        sess_backbone.free();
        sess_decoder.free();
        sess_body.free();
        sess_yolo.free();
        loaded = false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline  (public interface)
// ─────────────────────────────────────────────────────────────────────────────
Pipeline::Pipeline()  : impl_(new Impl) {}
Pipeline::~Pipeline() { free(); delete impl_; }

bool Pipeline::load(const PipelineConfig& cfg) {
    return impl_->load(cfg);
}
void Pipeline::free() {
    if (impl_) impl_->free_all();
}
bool Pipeline::is_loaded() const {
    return impl_ && impl_->loaded;
}
void Pipeline::print_info() const {
    if (!impl_ || !impl_->loaded) { printf("[FSB] not loaded\n"); return; }
    const auto& m = impl_->meta;
    printf("\n=== fast_sam_3dbody ===\n");
    printf("  decoder_dim : %u\n", m.decoder_dim);
    printf("  npose       : %u\n", m.npose);
    printf("  num_vertices: %u\n", m.num_vertices);
    printf("  num_kps     : %u\n", m.num_kps);
    printf("  default_f   : %.0f\n", m.default_focal);
    printf("=======================\n\n");
}
std::vector<MHRResult> Pipeline::process_bgr(const uint8_t* bgr, int w, int h) {
    return impl_->process_bgr(bgr, w, h);
}

} // namespace fsb
