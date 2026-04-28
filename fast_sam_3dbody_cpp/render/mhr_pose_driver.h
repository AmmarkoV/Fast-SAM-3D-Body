#pragma once
// mhr_pose_driver.h
// Per-frame bridge between MHR inference output and the TRI rendering model.
//
// Two responsibilities:
//   1. Copy pred_vertices into the TRI model's vertex buffer (no skinning needed;
//      the body_model.onnx already outputs the fully-deformed mesh).
//   2. Build OpenGL projection + view matrices from the MHR camera parameters
//      so the rendered mesh aligns with the source image.

#include <string.h>  // memcpy
#include "../GraphicsEngine/ModelLoader/model_loader_tri.h"

// Total number of vertices and floats in the MHR body mesh.
#define MHR_VERTEX_COUNT  18439
#define MHR_VERTEX_FLOATS (MHR_VERTEX_COUNT * 3)

// ---------------------------------------------------------------------------
// mhr_update_mesh_vertices
//
// Overwrites the TRI model's vertex buffer with the current frame's
// pred_vertices.  Call this every frame before uploading to the GPU.
//
// pred_vertices : float array of MHR_VERTEX_FLOATS elements, layout [x0,y0,z0, x1,...]
//                 This is MHRResult::pred_vertices.data() from the C++ pipeline.
//
// Note: MHR applies a Y/Z sign flip after skinning (verts[Y,Z] *= -1) to convert
// from the model's internal coordinate system to a camera-facing one.  The C++
// pipeline should replicate that flip.  If the mesh appears mirrored along Y or Z
// during rendering, toggle the flip in the pipeline and retest.
// ---------------------------------------------------------------------------
static inline void mhr_update_mesh_vertices(struct TRI_Model   *model,
                                             const float        *pred_vertices)
{
    if (!model || !model->vertices || !pred_vertices) { return; }
    memcpy(model->vertices, pred_vertices, MHR_VERTEX_FLOATS * sizeof(float));
}

// ---------------------------------------------------------------------------
// mhr_camera_matrices
//
// Builds column-major OpenGL matrices from the MHR pinhole camera parameters.
//
// out_proj  [16] : GL projection matrix (column-major, right-handed NDC)
// out_view  [16] : GL model-view matrix (pure translation by pred_cam_t)
// focal_length   : MHRResult::focal_length (pixels)
// pred_cam_t [3] : MHRResult::pred_cam_t  (tx, ty, tz in metres)
// img_w, img_h   : source image dimensions (pixels)
// ---------------------------------------------------------------------------
static inline void mhr_camera_matrices(float       out_proj[16],
                                        float       out_view[16],
                                        float       focal_length,
                                        const float pred_cam_t[3],
                                        int         img_w,
                                        int         img_h)
{
    const float near_plane = 0.01f;
    const float far_plane  = 100.0f;

    // Standard OpenGL perspective from pinhole focal length (pixels).
    // Principal point assumed at image centre (MHR default).
    float p00 = 2.0f * focal_length / (float)img_w;
    float p11 = 2.0f * focal_length / (float)img_h;
    float p22 = -(far_plane + near_plane) / (far_plane - near_plane);
    float p32 = -2.0f * far_plane * near_plane / (far_plane - near_plane);

    // Column-major layout: out[col*4 + row]
    out_proj[ 0] = p00;   out_proj[ 4] = 0.0f; out_proj[ 8] = 0.0f;  out_proj[12] = 0.0f;
    out_proj[ 1] = 0.0f;  out_proj[ 5] = p11;  out_proj[ 9] = 0.0f;  out_proj[13] = 0.0f;
    out_proj[ 2] = 0.0f;  out_proj[ 6] = 0.0f; out_proj[10] = p22;   out_proj[14] = p32;
    out_proj[ 3] = 0.0f;  out_proj[ 7] = 0.0f; out_proj[11] = -1.0f; out_proj[15] = 0.0f;

    // Model-view matrix that replicates the Python pyrender pipeline:
    //   Python:   verts[Y,Z] *= -1  (LBS flip)  then
    //             renderer applies 180° rotation around X  (flips Y,Z again → net identity on verts)
    //             camera placed at (-tx, ty, tz) looking in -Z
    //   OpenGL:   C++ LBS already has the [Y,Z]*=-1 flip applied to verts, so we
    //             need to undo it with another Y,Z sign flip on the diagonal (-1,-1),
    //             then translate by pred_cam_t matching "verts + pred_cam_t" in Python.
    //   Result:   v_view = (x+tx, -(−y)−ty, -(−z)−tz) = (x+tx, y−ty, z−tz)
    //             which matches pyrender camera space exactly.
    out_view[ 0] = 1.0f;            out_view[ 4] = 0.0f;
    out_view[ 8] = 0.0f;            out_view[12] =  pred_cam_t[0];
    out_view[ 1] = 0.0f;            out_view[ 5] = -1.0f;
    out_view[ 9] = 0.0f;            out_view[13] = -pred_cam_t[1];
    out_view[ 2] = 0.0f;            out_view[ 6] = 0.0f;
    out_view[10] = -1.0f;           out_view[14] = -pred_cam_t[2];
    out_view[ 3] = 0.0f;            out_view[ 7] = 0.0f;
    out_view[11] = 0.0f;            out_view[15] = 1.0f;
}
