// fast_sam_3dbody_render.cpp
// OpenGL overlay renderer: reads a camera/image source, runs the MHR body-pose
// pipeline, and draws the deformed 3D mesh on top of the input frame.
//
// Usage:
//   fast_sam_3dbody_render --onnx-dir DIR --gguf pipeline.gguf
//       --yolo yolo.onnx [--mesh body_mesh.tri] [--from 0|path]
//
// Controls: close the window to exit.

// GLEW must come before any other GL header.
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>

extern "C" {
#include "../GraphicsEngine/System/glx3.h"
#include "../GraphicsEngine/ModelLoader/model_loader_tri.h"
#include "../GraphicsEngine/ModelLoader/model_loader_transform_joints.h"
}

#include "../src/fast_sam_3dbody.h"
#include "mhr_pose_driver.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Inline GLSL shaders ──────────────────────────────────────────────────────

// Background fullscreen quad — uses gl_VertexID, no VBO needed.
static const char* QUAD_VERT = R"glsl(
    #version 330 core
    const vec2 P[4] = vec2[](
        vec2(-1.0,-1.0), vec2(1.0,-1.0),
        vec2(-1.0, 1.0), vec2(1.0, 1.0)
    );
    const vec2 UV[4] = vec2[](
        vec2(0.0,1.0), vec2(1.0,1.0),
        vec2(0.0,0.0), vec2(1.0,0.0)
    );
    out vec2 vUV;
    void main() { gl_Position = vec4(P[gl_VertexID],0.0,1.0); vUV = UV[gl_VertexID]; }
)glsl";

static const char* QUAD_FRAG = R"glsl(
    #version 330 core
    in  vec2      vUV;
    uniform sampler2D uTex;
    out vec4 fragColor;
    void main() { fragColor = vec4(texture(uTex, vUV).rgb, 1.0); }
)glsl";

// Body mesh — simple directional light from fixed direction.
static const char* MESH_VERT = R"glsl(
    #version 330 core
    layout(location=0) in vec3 aPos;
    layout(location=1) in vec3 aNorm;
    uniform mat4 uMVP;
    out vec3 vNorm;
    void main() {
        gl_Position = uMVP * vec4(aPos, 1.0);
        vNorm = aNorm;
    }
)glsl";

static const char* MESH_FRAG = R"glsl(
    #version 330 core
    in  vec3 vNorm;
    out vec4 fragColor;
    void main() {
        vec3 L = normalize(vec3(0.3, 0.8, 0.5));
        float d = clamp(dot(normalize(vNorm), L), 0.0, 1.0) * 0.7 + 0.3;
        fragColor = vec4(vec3(0.65, 0.75, 0.9) * d, 0.7);
    }
)glsl";

// ── GL helpers ───────────────────────────────────────────────────────────────

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512] = {}; glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
        fprintf(stderr, "[shader] %s\n", buf);
    }
    return s;
}

static GLuint link_program(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    GLuint v = compile_shader(GL_VERTEX_SHADER,   vs);
    GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[512] = {}; glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
        fprintf(stderr, "[program] %s\n", buf);
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ── GPU mesh state ───────────────────────────────────────────────────────────

struct MeshGPU {
    GLuint vao, vbo_pos, vbo_norm, ebo;
    GLsizei n_indices;
};

static MeshGPU upload_mesh_once(const struct TRI_Model* m) {
    MeshGPU g{};
    g.n_indices = (GLsizei)m->header.numberOfIndices;

    glGenVertexArrays(1, &g.vao);
    glBindVertexArray(g.vao);

    // Positions — DYNAMIC (updated every frame via glBufferSubData)
    glGenBuffers(1, &g.vbo_pos);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo_pos);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(m->header.numberOfVertices * sizeof(float)),
                 m->vertices, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Normals — STATIC (T-pose normals are good enough for an overlay)
    glGenBuffers(1, &g.vbo_norm);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo_norm);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(m->header.numberOfNormals * sizeof(float)),
                 m->normal, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Indices — STATIC
    glGenBuffers(1, &g.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(m->header.numberOfIndices * sizeof(unsigned int)),
                 m->indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    return g;
}

// ── Background texture ───────────────────────────────────────────────────────

struct BgTex { GLuint id; int w, h; bool ready; };

static BgTex create_bg_tex() {
    BgTex t{0, 0, 0, false};
    glGenTextures(1, &t.id);
    glBindTexture(GL_TEXTURE_2D, t.id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    return t;
}

// Upload a BGR frame. Converts to RGB so the sampler returns correct colours.
static void upload_bg_frame(BgTex& t, const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    glBindTexture(GL_TEXTURE_2D, t.id);
    if (!t.ready || bgr.cols != t.w || bgr.rows != t.h) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     bgr.cols, bgr.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        t.w = bgr.cols; t.h = bgr.rows; t.ready = true;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        bgr.cols, bgr.rows,
                        GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ── 4x4 matrix multiply (column-major) ──────────────────────────────────────

static void mat4_mul(float dst[16], const float a[16], const float b[16]) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) 
        {
            dst[c*4+r] = 0.f;
            for (int k = 0; k < 4; ++k)
                dst[c*4+r] += a[k*4+r] * b[c*4+k];
        }
}

static void mat4_print(const char * label,float m[16])
{
 fprintf(stderr,"%s\n",label);
 fprintf(stderr,"_________________________\n");
 fprintf(stderr,"%0.2f %0.2f %0.2f %0.2f\n",m[0],m[1],m[2],m[3]);
 fprintf(stderr,"%0.2f %0.2f %0.2f %0.2f\n",m[4],m[5],m[6],m[7]);
 fprintf(stderr,"%0.2f %0.2f %0.2f %0.2f\n",m[8],m[9],m[10],m[11]);
 fprintf(stderr,"%0.2f %0.2f %0.2f %0.2f\n",m[12],m[13],m[14],m[15]);
 fprintf(stderr,"_________________________\n");
}



int mat4_transpose(float * mat)
{
  if (mat!=0)
  {
  /*       -------  TRANSPOSE ------->
      0   1   2   3           0  4  8   12
      4   5   6   7           1  5  9   13
      8   9   10  11          2  6  10  14
      12  13  14  15          3  7  11  15   */

  float tmp;
  tmp = mat[1]; mat[1]=mat[4];  mat[4]=tmp;
  tmp = mat[2]; mat[2]=mat[8];  mat[8]=tmp;
  tmp = mat[3]; mat[3]=mat[12]; mat[12]=tmp;


  tmp = mat[6]; mat[6]=mat[9]; mat[9]=tmp;
  tmp = mat[13]; mat[13]=mat[7]; mat[7]=tmp;
  tmp = mat[14]; mat[14]=mat[11]; mat[11]=tmp;
  } else
  { //Believe it or not this is the fastest branch prediction :P
    return 0;
  }

 return 1;
}
// ── Callbacks required by glx3.c ─────────────────────────────────────────────

extern "C" {
    // Called by glx3_checkEvents() on key/mouse events.
    int handleUserInput(int key, int x, int y) { (void)key; (void)x; (void)y; return 1; }
    // Called by glx3_checkEvents() when the window is resized.
    int windowSizeUpdated(unsigned int w, unsigned int h) { (void)w; (void)h; return 1; }
}

// ── YOLO skeleton joint pairs (COCO 17-joint order) ─────────────────────────

static const int COCO_PAIRS[][2] = 
{
    {0,1},{0,2},{1,3},{2,4},                          // head
    {5,6},{5,7},{7,9},{6,8},{8,10},                   // arms
    {5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16} // torso+legs
};
static const int N_COCO_PAIRS = 17;

static void draw_yolo_skeleton(cv::Mat& img,
                                const std::vector<float>& kps,
                                float conf_thresh = 0.3f) 
{
    if ((int)kps.size() < 51) return;
    // Draw limb lines first, then joint dots on top
    for (int p = 0; p < N_COCO_PAIRS; ++p) {
        int a = COCO_PAIRS[p][0], b = COCO_PAIRS[p][1];
        if (kps[a*3+2] < conf_thresh || kps[b*3+2] < conf_thresh) continue;
        cv::line(img,
                 {(int)kps[a*3], (int)kps[a*3+1]},
                 {(int)kps[b*3], (int)kps[b*3+1]},
                 cv::Scalar(255, 128, 0), 2, cv::LINE_AA);
    }
    for (int k = 0; k < 17; ++k) {
        if (kps[k*3+2] < conf_thresh) continue;
        cv::circle(img, {(int)kps[k*3], (int)kps[k*3+1]},
                   5, cv::Scalar(0, 200, 255), -1, cv::LINE_AA);
    }
}

// ── Save GL framebuffer to file ──────────────────────────────────────────────

static void save_framebuffer(const std::string& path, int w, int h) {
    std::vector<uint8_t> px(w * h * 3);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    // glReadPixels gives bottom-up rows; flip vertically
    cv::Mat img(h, w, CV_8UC3, px.data());
    cv::flip(img, img, 0);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imwrite(path, img);
    printf("Saved: %s\n", path.c_str());
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, const char** argv) {
    std::string onnx_dir  = "./onnx";
    std::string gguf_path = "./onnx/pipeline.gguf";
    std::string yolo_path = "./onnx/yolo.onnx";
    std::string mesh_path = "./body_mesh.tri";
    std::string lbs_path  = "";
    std::string src       = "0";
    std::string save_path = "";
    int  cuda_device = 0;
    bool use_trt  = false;
    bool fp16     = true;

    for (int i = 1; i < argc; ++i) {
#define A1(flag, field, conv) \
        if (!strcmp(argv[i], flag) && i+1<argc) { field = conv(argv[++i]); continue; }
        A1("--onnx-dir", onnx_dir,  std::string)
        A1("--gguf",     gguf_path, std::string)
        A1("--yolo",     yolo_path, std::string)
        A1("--mesh",     mesh_path, std::string)
        A1("--lbs",      lbs_path,  std::string)
        A1("--from",     src,       std::string)
        A1("--save",     save_path, std::string)
        A1("--cuda",     cuda_device, std::stoi)
#undef A1
        if (!strcmp(argv[i], "--trt"))     { use_trt = true;  continue; }
        if (!strcmp(argv[i], "--no-fp16")) { fp16    = false; continue; }
    }

    // ── Pipeline ─────────────────────────────────────────────────────────────
    fsb::Pipeline pipeline;
    {
        fsb::PipelineConfig cfg;
        cfg.onnx_dir        = onnx_dir;
        cfg.gguf_path       = gguf_path;
        cfg.yolo_path       = yolo_path;
        cfg.cuda_device     = cuda_device;
        cfg.use_trt_ep      = use_trt;
        cfg.use_fp16        = fp16;
        cfg.skip_body_model = true;    // LBS runs natively in C; skip body_model.onnx
        if (!pipeline.load(cfg)) {
            fprintf(stderr, "Failed to load pipeline\n"); return 1;
        }
    }

    // ── Video/image source ────────────────────────────────────────────────────
    bool is_image = false;
    cv::Mat static_img;
    cv::VideoCapture cap;
    {
        bool numeric = !src.empty() &&
                       (src[0]=='-' || isdigit((unsigned char)src[0]));
        if (numeric) {
            cap.open(std::stoi(src));
        } else {
            static_img = cv::imread(src);
            if (!static_img.empty()) {
                is_image = true;
            } else {
                cap.open(src);
                if (!cap.isOpened()) { fprintf(stderr,"Cannot open: %s\n", src.c_str()); return 1; }
            }
        }
    }

    // Determine initial window size from first frame
    cv::Mat probe;
    if (is_image) probe = static_img;
    else          cap >> probe;
    if (probe.empty()) { fprintf(stderr, "Empty frame\n"); return 1; }
    int W = probe.cols, H = probe.rows;

    // ── GLX window ────────────────────────────────────────────────────────────
    if (!start_glx3_stuff(W, H, 1, argc, argv)) {
        fprintf(stderr, "Failed to start GLX window\n"); return 1;
    }
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "GLEW init failed\n"); return 1;
    }

    // ── Shaders ───────────────────────────────────────────────────────────────
    GLuint prog_quad = link_program(QUAD_VERT, QUAD_FRAG);
    GLuint prog_mesh = link_program(MESH_VERT, MESH_FRAG);
    GLint  mvp_loc   = glGetUniformLocation(prog_mesh, "uMVP");
    GLint  tex_loc   = glGetUniformLocation(prog_quad, "uTex");

    // ── Load body mesh from .tri ──────────────────────────────────────────────
    struct TRI_Model* tri_model = tri_allocateModel();
    if (!tri_loadModel(mesh_path.c_str(), tri_model)) {
        fprintf(stderr, "Cannot load mesh: %s\n", mesh_path.c_str()); return 1;
    }
    printf("Mesh loaded: %u vertices, %u indices\n",
           tri_model->header.numberOfVertices / 3,
           tri_model->header.numberOfIndices / 3);

    MeshGPU mesh_gpu = upload_mesh_once(tri_model);

    // ── Load LBS data ─────────────────────────────────────────────────────────
    if (lbs_path.empty()) lbs_path = onnx_dir + "/body_model.lbs";
    struct MHR_LBS_Data* lbs = mhr_lbs_load(lbs_path.c_str());
    if (!lbs) fprintf(stderr, "Warning: LBS data not loaded — mesh will not deform\n");
    std::vector<float> lbs_out(MHR_VERTEX_FLOATS, 0.f);

    // Empty VAO for the quad (we use gl_VertexID in the vertex shader)
    GLuint quad_vao;
    glGenVertexArrays(1, &quad_vao);

    BgTex bg = create_bg_tex();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ── Render loop ───────────────────────────────────────────────────────────
    cv::Mat frame;
    while (glx3_checkEvents()) 
    {
        if (is_image) 
        {
            frame = static_img;
        } else 
        {
            cap >> frame;
            if (frame.empty()) break;
        }

        // Inference
        auto results = pipeline.process_bgr(frame.data, frame.cols, frame.rows);

        // Annotate frame: draw YOLO skeleton when LBS mesh is unavailable.
        cv::Mat vis = frame.clone();
        bool any_mesh = lbs && !results.empty();
        any_mesh = true;
        if (!any_mesh) {
            for (const auto& r : results)
                draw_yolo_skeleton(vis, r.keypoints_yolo);
        }

        // Upload background (with optional skeleton annotation)
        upload_bg_frame(bg, vis);

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, W, H);

        // ── Background quad ───────────────────────────────────────────────────
        glDisable(GL_DEPTH_TEST);
        glUseProgram(prog_quad);
        glUniform1i(tex_loc, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, bg.id);
        glBindVertexArray(quad_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glEnable(GL_DEPTH_TEST);

        // ── Mesh overlay for each detected person ─────────────────────────────
        glUseProgram(prog_mesh);
        for (const auto& r : results) {
            if (!lbs) continue;

            // Run native C LBS forward pass, stream result to GPU
            mhr_lbs_compute(lbs,
                            r.mhr_model_params.data(),
                            r.shape.data(),
                            r.face_params.data(),
                            lbs_out.data());
            mhr_update_mesh_vertices(tri_model, lbs_out.data());
            glBindBuffer(GL_ARRAY_BUFFER, mesh_gpu.vbo_pos);
            glBufferSubData(GL_ARRAY_BUFFER, 0,
                            MHR_VERTEX_FLOATS * sizeof(float),
                            tri_model->vertices);

            // Build MVP = projection * view
            float proj[16], view[16], mvp[16];
            mhr_camera_matrices(proj, view,
                                r.focal_length, r.pred_cam_t.data(),
                                W, H);


            //view[0]=1.0; view[1]=0.0; view[2]=0.0; view[3]=0.0;
            //view[4]=0.0; view[5]=1.0; view[6]=0.0; view[7]=0.0;
            //view[8]=0.0; view[9]=0.0; view[10]=1.0; view[11]=100.0;
            //view[12]=0.0; view[13]=0.0; view[14]=0.0; view[15]=1.0;
            mat4_mul(mvp, proj, view);
            //mat4_transpose(mvp);
            mat4_print("Projection",proj);
            mat4_print("View",view);
            mat4_print("MVP",mvp);

            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp);

            glBindVertexArray(mesh_gpu.vao);
            glDrawElements(GL_TRIANGLES, mesh_gpu.n_indices,
                           GL_UNSIGNED_INT, nullptr);
        }
        glBindVertexArray(0);

        glx3_endRedraw();

        // Save and exit if --save was given, or after one frame for static images
        if (!save_path.empty()) {
            save_framebuffer(save_path, W, H);
            break;
        }
        if (is_image) break;   // keep window open only for live sources
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    mhr_lbs_free(lbs);
    tri_freeModel(tri_model);
    stop_glx3_stuff();
    return 0;
}
