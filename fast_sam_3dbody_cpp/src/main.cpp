#include "fast_sam_3dbody.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Config
{
    std::string onnx_dir    = "./onnx";
    std::string gguf_path   = "./onnx/pipeline.gguf";
    std::string yolo_path   = "./onnx/yolo.onnx";
    std::string input_src   = "0";     // webcam index or path to image/video
    int         cuda_device = 0;
    bool        use_trt     = false;
    bool        fp16        = true;
    bool        skip_body      = false;
    bool        zero_face      = true;
    float       person_thresh  = 0.50f;
    float       person_nms_iou = 0.45f;
    float       focal_x        = 0.f;
    float       focal_y        = 0.f;
    float       cx             = 0.f;
    float       cy             = 0.f;
    bool        headless    = false;
    bool        info_only   = false;
    int         render_w    = 0;     // GL window width  (0 = match input)
    int         render_h    = 0;     // GL window height (0 = match input)
    int         cap_w       = 0;     // capture width  (0 = driver default)
    int         cap_h       = 0;     // capture height (0 = driver default)
    double      cap_fps     = 0.0;   // capture fps    (0 = driver default)
};

static void print_usage(const char* prog)
{
    printf("Usage: %s [options]\n\n", prog);
    printf("  --onnx-dir PATH   Directory with backbone/decoder/body_model ONNX files\n");
    printf("  --gguf PATH       pipeline.gguf (MHR + camera heads)\n");
    printf("  --yolo PATH       YOLO pose model (.onnx or .engine)\n");
    printf("  --from SRC        Webcam index (0,1,..) or path to image/video\n");
    printf("  --size W H        Webcam capture resolution (default: driver default)\n");
    printf("  --fps Z           Webcam capture framerate  (default: driver default)\n");
    printf("  --cuda DEVICE     CUDA device index (default 0; -1 = CPU)\n");
    printf("  --trt             Enable ONNX Runtime TensorRT EP\n");
    printf("  --no-fp16         Disable FP16 for ONNX EP\n");
    printf("  --skip-body       Skip body model (no vertices / keypoints)\n");
    printf("  --dev-face        Enable face expression params (disabled by default)\n");
    printf("  --thresh T        YOLO person confidence threshold (default 0.50)\n");
    printf("  --nms T           YOLO NMS IoU threshold (default 0.45)\n");
    printf("  --fx F            Camera focal length x (0 = image width)\n");
    printf("  --fy F            Camera focal length y (0 = image width)\n");
    printf("  --cx F            Principal point x (0 = width/2)\n");
    printf("  --cy F            Principal point y (0 = height/2)\n");
    printf("  --render-size W H GL window width and height in pixels (default: match input)\n");
    printf("  --headless        Do not open display windows\n");
    printf("  --info            Print pipeline info and exit\n");
    printf("  --help / -h       This message\n");
}

static Config parse_args(int argc, char** argv)
{
    Config c;
    for (int i = 1; i < argc; ++i)
    {
#define ARG1(flag, field, conv) \
        if (!strcmp(argv[i], flag) && i+1 < argc) { c.field = conv(argv[++i]); continue; }
        ARG1("--onnx-dir", onnx_dir,    std::string)
        ARG1("--gguf",     gguf_path,   std::string)
        ARG1("--yolo",     yolo_path,   std::string)
        ARG1("--from",     input_src,   std::string)
        ARG1("--cuda",     cuda_device, std::stoi)
        ARG1("--thresh",   person_thresh,  std::stof)
        ARG1("--nms",      person_nms_iou, std::stof)
        ARG1("--fx",       focal_x,     std::stof)
        ARG1("--fy",       focal_y,     std::stof)
        ARG1("--cx",       cx,          std::stof)
        ARG1("--cy",       cy,          std::stof)
#undef ARG1
        if (!strcmp(argv[i], "--render-size") && i+2 < argc)
        {
            c.render_w = std::stoi(argv[++i]);
            c.render_h = std::stoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--size") && i+2 < argc)
        {
            c.cap_w = std::stoi(argv[++i]);
            c.cap_h = std::stoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--fps") && i+1 < argc)
        {
            c.cap_fps = std::stod(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--trt"))
        {
            c.use_trt   = true;
            continue;
        }
        if (!strcmp(argv[i], "--no-fp16"))
        {
            c.fp16      = false;
            continue;
        }
        if (!strcmp(argv[i], "--skip-body"))
        {
            c.skip_body = true;
            continue;
        }
        if (!strcmp(argv[i], "--dev-face"))
        {
            c.zero_face = false;
            continue;
        }
        if (!strcmp(argv[i], "--headless"))
        {
            c.headless  = true;
            continue;
        }
        if (!strcmp(argv[i], "--info"))
        {
            c.info_only = true;
            continue;
        }
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(1);
    }
    return c;
}

// ---------------------------------------------------------------------------
// Print one MHRResult to stdout
// ---------------------------------------------------------------------------
static void print_result(int person_idx, const fsb::MHRResult& r)
{
    printf("  person[%d]  bbox=[%.1f,%.1f,%.1f,%.1f]  focal=%.1f  cam_t=[%.3f,%.3f,%.3f]\n",
           person_idx,
           r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3],
           r.focal_length,
           r.pred_cam_t[0], r.pred_cam_t[1], r.pred_cam_t[2]);
    printf("             global_rot=[%.4f,%.4f,%.4f]\n",
           r.global_rot[0], r.global_rot[1], r.global_rot[2]);

    // body pose summary (first 9 values)
    printf("             body_pose[0..8]=[");
    for (int j = 0; j < 9 && j < (int)r.body_pose.size(); ++j)
        printf("%.4f%s", r.body_pose[j], j+1<9 && j+1<(int)r.body_pose.size() ? "," : "");
    printf("...]\n");

    // shape summary
    printf("             shape[0..4]=[");
    for (int j = 0; j < 5 && j < (int)r.shape.size(); ++j)
        printf("%.4f%s", r.shape[j], j+1<5 && j+1<(int)r.shape.size() ? "," : "");
    printf("...]\n");

    if (!r.keypoints_3d.empty())
    {
        printf("             kp3d[0]=[%.3f,%.3f,%.3f]\n",
               r.keypoints_3d[0], r.keypoints_3d[1], r.keypoints_3d[2]);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Config c = parse_args(argc, argv);

    fsb::PipelineConfig pcfg;
    pcfg.onnx_dir       = c.onnx_dir;
    pcfg.gguf_path      = c.gguf_path;
    pcfg.yolo_path      = c.yolo_path;
    pcfg.cuda_device    = c.cuda_device;
    pcfg.use_trt_ep     = c.use_trt;
    pcfg.use_fp16       = c.fp16;
    pcfg.skip_body_model  = c.skip_body;
    pcfg.zero_face_params = c.zero_face;
    pcfg.person_thresh  = c.person_thresh;
    pcfg.person_nms_iou = c.person_nms_iou;
    pcfg.focal_x        = c.focal_x;
    pcfg.focal_y        = c.focal_y;
    pcfg.principal_x    = c.cx;
    pcfg.principal_y    = c.cy;

    fsb::Pipeline pipeline;
    {
        auto t0 = Clock::now();
        if (!pipeline.load(pcfg))
        {
            fprintf(stderr, "[main] Pipeline load failed.\n");
            return 1;
        }
        printf("[main] Pipeline loaded in %.1f ms\n", ms_since(t0));
    }

    pipeline.print_info();
    if (c.info_only)
    {
        pipeline.free();
        return 0;
    }

    // -----------------------------------------------------------------------
    // Open input source
    // -----------------------------------------------------------------------
    cv::VideoCapture cap;
    bool is_image = false;

    bool src_is_int = !c.input_src.empty() &&
                      c.input_src.find_first_not_of("0123456789") == std::string::npos;

    if (src_is_int)
    {
        cap.open(std::stoi(c.input_src));
    }
    else
    {
        // Treat as image if it has a known image extension
        const char* img_exts[] = {".jpg",".jpeg",".png",".bmp",".tiff",".webp", nullptr};
        for (int k = 0; img_exts[k]; ++k)
        {
            auto ext = img_exts[k];
            auto elen = strlen(ext);
            if (c.input_src.size() >= elen &&
                    c.input_src.compare(c.input_src.size()-elen, elen, ext) == 0)
            {
                is_image = true;
                break;
            }
        }
        if (!is_image) cap.open(c.input_src);
    }

    if (!is_image && cap.isOpened())
    {
        if (c.cap_w > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH,  c.cap_w);
        if (c.cap_h > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, c.cap_h);
        if (c.cap_fps > 0.0) cap.set(cv::CAP_PROP_FPS,      c.cap_fps);
    }

    if (!is_image && !cap.isOpened())
    {
        fprintf(stderr, "[main] Cannot open input: %s\n", c.input_src.c_str());
        pipeline.free();
        return 1;
    }

    // -----------------------------------------------------------------------
    // Inference loop
    // -----------------------------------------------------------------------
    cv::Mat frame;
    int     frame_count  = 0;
    double  total_inf_ms = 0.0;
    auto    loop_start   = Clock::now();

    while (true)
    {
        if (is_image)
        {
            frame = cv::imread(c.input_src);
            if (frame.empty())
            {
                fprintf(stderr, "[main] Cannot read image.\n");
                break;
            }
        }
        else
        {
            if (!cap.read(frame) || frame.empty()) break;
        }

        auto t0 = Clock::now();
        std::vector<fsb::MHRResult> results =
            pipeline.process_bgr(frame.data, frame.cols, frame.rows);
        double inf_ms = ms_since(t0);
        total_inf_ms += inf_ms;
        ++frame_count;

        printf("frame %d  |  %.1f ms  |  %d person(s)\n",
               frame_count, inf_ms, (int)results.size());
        for (int i = 0; i < (int)results.size(); ++i)
            print_result(i, results[i]);

        if (is_image) break;

        // FPS every 30 frames
        if (frame_count % 30 == 0)
        {
            double wall_s = ms_since(loop_start) / 1000.0;
            printf("[fps] inf=%.1f  wall=%.1f\n",
                   frame_count * 1000.0 / (total_inf_ms > 0 ? total_inf_ms : 1),
                   frame_count / (wall_s > 0 ? wall_s : 1));
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    if (frame_count > 0)
    {
        double wall_s = ms_since(loop_start) / 1000.0;
        printf("\n--- Summary (%d frames) ---\n", frame_count);
        printf("  Inf fps  : %.1f\n", frame_count * 1000.0 / (total_inf_ms > 0 ? total_inf_ms : 1));
        printf("  Wall fps : %.1f\n", frame_count / (wall_s > 0 ? wall_s : 1));
        printf("  Inf ms   : %.2f / frame\n", total_inf_ms / frame_count);
    }

    pipeline.free();
    return 0;
}
