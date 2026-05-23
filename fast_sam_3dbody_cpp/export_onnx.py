#!/usr/bin/env python3
"""
Export SAM-3D-Body pipeline stages to ONNX for fast C++ inference.

Produces three ONNX files in the output directory:
  backbone.onnx        — DINOv3 backbone: [B,3,512,512] → [B,1280,32,32]
  decoder.onnx         — ray_cond_emb + 6-layer transformer decoder:
                           features [B,1280,32,32] + condition [B,3]
                           + ray_cond [B,2,512,512] → pose_token [B,1024]
  body_model.onnx      — MHR torch.jit body model:
                           shape [B,45] + body_params [B,204] + face [B,72]
                           → verts [B,18439,3] + skel [B,127,8]

Usage:
  cd /path/to/Fast-SAM-3D-Body
  SKIP_KEYPOINT_PROMPT=1 MHR_NO_CORRECTIVES=1 \\
      python fast_sam_3dbody_cpp/export_onnx.py \\
          --checkpoint ./checkpoints/sam-3d-body-dinov3 \\
          --output     ./fast_sam_3dbody_cpp/onnx

Prerequisites:
  pip install onnx onnxruntime onnxsim   (onnxsim optional but recommended)
"""

import argparse
import os
import sys
import types

# ── environment flags must be set before importing the model ──────────────
os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")
os.environ.setdefault("BODY_INTERM_PRED_LAYERS", "0,1,2")
os.environ.setdefault("HAND_INTERM_PRED_LAYERS", "0,1")
os.environ.setdefault("KEYPOINT_PROMPT_INTERM_INTERVAL", "999")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────
# Constants matching model_config.yaml
# ──────────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 512
PATCH_SIZE   = 16
FEAT_H = FEAT_W = IMAGE_SIZE // PATCH_SIZE   # 32
BACKBONE_DIM = 1280                           # dinov3_vith16plus
DECODER_DIM  = 1024


# ══════════════════════════════════════════════════════════════════════════
# 1. Backbone wrapper  (reuses logic from convert_backbone_tensorrt.py)
# ══════════════════════════════════════════════════════════════════════════
class BackboneWrapper(nn.Module):
    """DINOv3 backbone: [B,3,512,512] → [B,1280,32,32]."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_intermediate_layers(
            x, n=1, reshape=True, norm=True
        )[-1]


class _Float32Conv(nn.Module):
    """Wraps a Conv module to run in float32 even inside torch.autocast(bfloat16).

    ORT CUDA EP does not support bfloat16 Conv (ONNX opset 18/20 type constraint).
    Parameters are upcasted to float32 so the ONNX graph contains a float32 Conv
    node surrounded by explicit Cast nodes, while downstream attention ops remain
    in bfloat16 via the outer autocast context.
    """
    def __init__(self, conv: nn.Module):
        super().__init__()
        self.conv = conv.float()  # upcast params: Conv node will be float32 in ONNX

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            return self.conv(x.float()).to(x.dtype)


def _patch_swiglu_silu_float32(encoder: nn.Module) -> None:
    """Force SwiGLU FFN blocks to compute silu in float32 during BF16 export.

    ORT CUDA EP's com.microsoft.QuickGelu kernel only supports float32.
    With BF16 autocast, silu(x) = x * sigmoid(x) becomes BF16 Mul+Sigmoid which
    ORT fuses as QuickGelu(BF16) — missing kernel.
    By running silu in float32 we get QuickGelu(float32) which ORT does support.
    Attention linears remain BF16 via the outer autocast context.
    """
    import torch.nn.functional as F_

    def _make_f32_silu_forward(m):
        def forward(x):
            x1 = m.w1(x)
            x2 = m.w2(x)
            with torch.autocast("cuda", enabled=False):
                hidden = F_.silu(x1.float()) * x2.float()
            return m.w3(hidden.to(x.dtype))
        return forward

    for module in encoder.modules():
        if hasattr(module, "w1") and hasattr(module, "w2") and hasattr(module, "w3"):
            module.forward = _make_f32_silu_forward(module)


def _patch_encoder(encoder):
    """Patch prepare_tokens_with_masks to avoid mask_token graph edge.
    Also wraps patch_embed in _Float32Conv (ORT CUDA EP rejects BF16 Conv in opset 18/20).
    Patches SwiGLU silu to float32 (ORT has no BF16 QuickGelu kernel, float32 is supported).
    """
    encoder.patch_embed = _Float32Conv(encoder.patch_embed)
    _patch_swiglu_silu_float32(encoder)

    def _prepare_tokens_patched(self, x, masks=None):
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)
        cls_token = self.cls_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1, 0, cls_token.shape[-1],
                dtype=cls_token.dtype, device=cls_token.device,
            )
        x = torch.cat([
            cls_token.expand(B, -1, -1),
            storage_tokens.expand(B, -1, -1),
            x,
        ], dim=1)
        return x, (H, W)

    encoder.prepare_tokens_with_masks = types.MethodType(
        _prepare_tokens_patched, encoder
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. Decoder wrapper  (decoder + ray_cond_emb → pose token)
# ══════════════════════════════════════════════════════════════════════════
class BodyDecoderWrapper(nn.Module):
    """
    Wraps CameraEncoder (ray_cond) + 6-layer PromptableDecoder.

    Inputs
    ------
    features    : [B, 1280, 32, 32]  backbone features
    cond_info   : [B, 3]             CLIFF condition (dx/f, dy/f, b/f)
    ray_cond    : [B, 2, 512, 512]   per-pixel normalised ray directions

    Output
    ------
    pose_token  : [B, 1024]          first decoder output token
    """

    def __init__(self, model):
        super().__init__()
        self.ray_cond_emb    = model.ray_cond_emb
        self.decoder         = model.decoder
        self.init_pose       = model.init_pose
        self.init_camera     = model.init_camera
        self.init_to_token   = model.init_to_token_mhr
        self.prev_to_token   = model.prev_to_token_mhr
        self.prompt_encoder  = model.prompt_encoder
        self.prompt_to_token = model.prompt_to_token

        # Optional extra token embeddings
        for attr in ("keypoint_embedding", "keypoint3d_embedding", "hand_box_embedding"):
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

        # Disable intermediate predictions so the decoder returns plain tensors
        self.decoder.do_interm_preds = False
        # Disable keypoint token update — not needed for ONNX export
        self.decoder.keypoint_token_update = None

        # Replace GELU with ReLU to work around torch.export bug with gelu → convert_to_relu
        for m in self.modules():
            if isinstance(m, nn.GELU):
                m.approximate = "tanh"  # use tanh approximation which exports cleanly

        # With --bf16 export, torch.autocast handles BF16 for linear ops.
        # Wrap ray_cond_emb.conv in _Float32Conv so it stays float32 in the ONNX
        # graph (ORT CUDA EP rejects bfloat16 Conv in opset 18/20).
        self.ray_cond_emb.conv = _Float32Conv(self.ray_cond_emb.conv)
        # prompt_encoder has internal float32 casts; autocast leaves those in float32.

    def forward(
        self,
        features:  torch.Tensor,   # [B, 1280, 32, 32]
        cond_info: torch.Tensor,   # [B, 3]
        ray_cond:  torch.Tensor,   # [B, 2, 32, 32]  — already at patch resolution
    ) -> torch.Tensor:             # [B, 1024]

        B     = features.shape[0]
        dev   = features.device
        dtype = features.dtype

        # ── CameraEncoder (inlined, without F.interpolate) ────────────────
        # ray_cond is already at feature-map resolution [B, 2, H, W]
        _h, _w = features.shape[2], features.shape[3]
        rays = ray_cond.permute(0, 2, 3, 1)                         # [B, H, W, 2]
        rays = torch.cat([rays, torch.ones_like(rays[..., :1])], -1) # [B, H, W, 3]
        rays_emb = self.ray_cond_emb.camera(pos=rays.reshape(B, -1, 3))  # [B, H*W, 99]
        rays_emb = rays_emb.reshape(B, _h, _w, -1).permute(0, 3, 1, 2).contiguous()
        z = torch.cat([features, rays_emb], dim=1)
        features = self.ray_cond_emb.norm(self.ray_cond_emb.conv(z))  # [B, 1280, H, W]

        # ── build initial estimate ────────────────────────────────────────
        init_pose   = self.init_pose.weight.expand(B, -1).unsqueeze(1)    # [B,1,P]
        init_camera = self.init_camera.weight.expand(B, -1).unsqueeze(1)  # [B,1,3]
        init_est    = torch.cat([init_pose, init_camera], dim=-1)          # [B,1,P+3]

        # ── pose token ───────────────────────────────────────────────────
        init_input = torch.cat(
            [cond_info.unsqueeze(1).to(dtype), init_est], dim=-1
        )                                                   # [B,1,3+P+3]
        token_seq = self.init_to_token(init_input)          # [B,1,1024]

        # ── previous-estimate token ───────────────────────────────────────
        prev_emb  = self.prev_to_token(init_est)            # [B,1,1024]

        # ── dummy keypoint prompt (label=-2 encodes "no keypoints") ──────
        # prompt_encoder uses sin/cos positional encoding; ONNX Cos/Sin don't support
        # bfloat16, so force float32 here and cast the result back to the active dtype.
        kps = torch.full((B, 1, 3), 0.0, device=dev, dtype=torch.float32)
        kps[:, :, -1] = -2.0
        with torch.autocast("cuda", enabled=False):
            prompt_emb, _ = self.prompt_encoder(keypoints=kps)  # [B,1,backbone_dim] float32
        prompt_emb    = self.prompt_to_token(prompt_emb.to(dtype))    # [B,1,1024]

        # ── token sequence + augment (positional info per token) ─────────
        token_seq = torch.cat([token_seq, prev_emb, prompt_emb], dim=1)   # [B,3,1024]
        tok_aug   = torch.zeros_like(token_seq)
        tok_aug[:, 1] = prev_emb[:, 0]
        tok_aug[:, 2] = prompt_emb[:, 0]

        for attr in ("keypoint_embedding", "keypoint3d_embedding", "hand_box_embedding"):
            if hasattr(self, attr):
                emb      = getattr(self, attr).weight.unsqueeze(0).expand(B, -1, -1)
                token_seq = torch.cat([token_seq, emb],                  dim=1)
                tok_aug   = torch.cat([tok_aug,   torch.zeros_like(emb)], dim=1)

        # ── image positional encoding ─────────────────────────────────────
        # get_dense_pe also uses sin/cos; same float32 guard.
        with torch.autocast("cuda", enabled=False):
            img_pe = self.prompt_encoder.get_dense_pe(features.shape[-2:])    # float32 [1,C,h,w]
        img_pe = img_pe.to(dtype).expand(B, -1, -1, -1)                       # [B,C,h,w]

        # ── run decoder ───────────────────────────────────────────────────
        out = self.decoder(
            token_embedding        = token_seq,
            image_embedding        = features,
            token_augment          = tok_aug,
            image_augment          = img_pe,
            token_mask             = None,
            channel_first          = True,
            token_to_pose_output_fn= None,   # no intermediate predictions
        )
        # With do_interm_preds=False the decoder returns (token_emb, image_emb)
        if isinstance(out, (tuple, list)):
            token_out = out[0]
        else:
            token_out = out

        return token_out[:, 0]   # pose token: [B, 1024]


# ══════════════════════════════════════════════════════════════════════════
# 3. MHR body model wrapper  (torch.jit → ONNX)
# ══════════════════════════════════════════════════════════════════════════
class BodyModelWrapper(nn.Module):
    """
    Wraps the MHR torch.jit body model with correctives disabled.

    Inputs:  shape [B,45]  body_params [B,204]  face [B,72]
    Outputs: verts [B,18439,3]  skel [B,127,8]
    """

    def __init__(self, mhr_jit):
        super().__init__()
        self.mhr = mhr_jit

    def forward(
        self,
        shape:       torch.Tensor,   # [B, 45]
        body_params: torch.Tensor,   # [B, 204]
        face:        torch.Tensor,   # [B, 72]
    ):
        verts, skel = self.mhr(shape, body_params, face, False)  # apply_correctives=False
        return verts, skel


# ══════════════════════════════════════════════════════════════════════════
# Export helpers
# ══════════════════════════════════════════════════════════════════════════
def _simplify(path: str):
    """Run onnx-simplifier if available."""
    try:
        import onnxsim
        import onnx
        model = onnx.load(path)
        try:
            model_sim, ok = onnxsim.simplify(model)
        except (RuntimeError, Exception):
            # onnxsim can't handle BF16 graphs or some ir_version quirks — skip
            print(f"  [onnxsim] simplify skipped for {os.path.basename(path)}")
            return
        if ok:
            onnx.save(model_sim, path)
            print(f"  [onnxsim] simplified {os.path.basename(path)}")
        else:
            print("  [onnxsim] simplification failed – keeping original")
    except ImportError:
        pass


def export_backbone(model, out_dir: str, opset: int = 18, bf16: bool = False):
    path = os.path.join(out_dir, "backbone.onnx")
    print(f"\n── backbone → {path}  ({'autocast bfloat16' if bf16 else 'float32'})")

    encoder = model.backbone.encoder
    _patch_encoder(encoder)
    wrapper = BackboneWrapper(encoder)
    wrapper.eval().cuda()  # parameters stay float32; autocast handles BF16 compute

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda", dtype=torch.float32)

    import contextlib
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

    with ctx, torch.no_grad():
        out = wrapper(dummy)
    print(f"   in {tuple(dummy.shape)}  out {tuple(out.shape)}  output dtype={out.dtype}")

    with ctx:
        torch.onnx.export(
            wrapper, dummy, path,
            input_names=["image"],
            output_names=["features"],
            dynamic_axes={"image": {0: "B"}, "features": {0: "B"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"   {os.path.getsize(path)/1e6:.1f} MB  ✓")
    _simplify(path)


def export_decoder(model, out_dir: str, opset: int = 18, bf16: bool = False):
    path = os.path.join(out_dir, "decoder.onnx")
    print(f"\n── decoder → {path}  ({'autocast bfloat16' if bf16 else 'float32'})")

    wrapper = BodyDecoderWrapper(model)
    wrapper.eval().cuda()  # parameters stay float32; autocast handles BF16 compute

    B = 1
    feat  = torch.randn(B, BACKBONE_DIM, FEAT_H, FEAT_W, device="cuda", dtype=torch.float32)
    cond  = torch.randn(B, 3,            device="cuda", dtype=torch.float32)
    ray   = torch.randn(B, 2, FEAT_H,   FEAT_W,  device="cuda", dtype=torch.float32)

    import contextlib
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

    with ctx, torch.no_grad():
        token = wrapper(feat, cond, ray)
    print(f"   pose_token shape: {tuple(token.shape)}  output dtype={token.dtype}")

    with ctx:
        torch.onnx.export(
            wrapper,
            (feat, cond, ray),
            path,
            input_names =["features", "condition_info", "ray_cond"],
            output_names=["pose_token"],
            dynamic_axes={
                "features":       {0: "B"},
                "condition_info": {0: "B"},
                "ray_cond":       {0: "B"},
                "pose_token":     {0: "B"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"   {os.path.getsize(path)/1e6:.1f} MB  ✓")
    _simplify(path)


def export_body_model(model, out_dir: str, opset: int = 18):
    """
    Save the MHR body model (mhr_model.pt) as a TorchScript file.

    The pymomentum LinearBlendSkinning ops inside MHRDemo are incompatible with
    the ONNX TorchScript exporter (causes heap corruption).  We therefore save
    the model as a TorchScript .pt file; the C++ side loads it with LibTorch
    (torch::jit::load) rather than ONNX Runtime.

    If you do not need per-vertex output (vertices / keypoints), pass
    --skip-body to the C++ executable and skip this step entirely.
    """
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt")
    dst = os.path.join(out_dir, "body_model.pt")

    print(f"\n── body_model (TorchScript) → {dst}")

    if not os.path.exists(src):
        print(f"   WARNING: {src} not found – skipping body model")
        return

    import shutil
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / 1e6
    print(f"   {size_mb:.1f} MB  ✓  (copied from checkpoint)")
    print("   NOTE: loaded in C++ via torch::jit::load(), not ONNX Runtime.")
    print("         Use --skip-body if you only need MHR pose parameters.")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Export SAM-3D-Body pipeline to ONNX")
    ap.add_argument("--checkpoint", default="./checkpoints/sam-3d-body-dinov3",
                    help="Directory with model.ckpt + assets/mhr_model.pt")
    ap.add_argument("--output", default="./fast_sam_3dbody_cpp/onnx",
                    help="Output directory for .onnx files")
    ap.add_argument("--stage", choices=["backbone", "decoder", "body_model", "all"],
                    default="all")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--bf16", action="store_true",
                    help="Export backbone and decoder in BFloat16 to match the model's "
                         "training compute dtype. Requires an Ampere or newer GPU (sm_80+) "
                         "and ORT 1.16+. Fixes arm joint estimation errors caused by the "
                         "FP32 vs BF16 precision mismatch in the default ONNX export.")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading model …")
    from sam_3d_body.build_models import load_sam_3d_body
    ckpt  = os.path.join(args.checkpoint, "model.ckpt")
    mhr   = os.path.join(args.checkpoint, "assets", "mhr_model.pt")
    model, _ = load_sam_3d_body(checkpoint_path=ckpt, mhr_path=mhr)
    model.eval()
    print("Model loaded.")

    if args.bf16:
        print("\n[--bf16] Exporting backbone and decoder in BFloat16 (matching training precision).")

    if args.stage in ("backbone", "all"):
        export_backbone(model, args.output, args.opset, bf16=args.bf16)
    if args.stage in ("decoder", "all"):
        export_decoder(model, args.output, args.opset, bf16=args.bf16)
    if args.stage in ("body_model", "all"):
        export_body_model(model, args.output, args.opset)

    print("\n✓ Export complete.")
    print(f"  Files in: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
