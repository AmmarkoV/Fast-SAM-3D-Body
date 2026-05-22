"""Minimal script to find the exact source of the CUDA/CPU device mismatch."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'fast_sam_3dbody_cpp')

import cv2, numpy as np, torch
from sam_3d_body import load_sam_3d_body
from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

model, cfg = load_sam_3d_body(
    checkpoint_path='checkpoints/sam-3d-body-dinov3/model.ckpt',
    device='cpu',
    mhr_path='checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt',
)
est = SAM3DBodyEstimator(model, cfg)

print(f"est.device: {est.device}")
print(f"image_mean: {model.image_mean.device}")

frame = np.full((480, 640, 3), 128, dtype=np.uint8)
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
bbox = np.array([[0, 0, 640, 480]], dtype=np.float32)

# Call process_one_image without any exception handling
try:
    outputs = est.process_one_image(img_rgb, bboxes=bbox, inference_type="body")
    print("Success! outputs:", len(outputs))
    if outputs:
        kps = outputs[0].get("pred_keypoints_2d")
        print("pred_keypoints_2d:", np.asarray(kps).shape if kps is not None else None)
except RuntimeError as e:
    print(f"\n[ERROR] RuntimeError: {e}")
    import traceback
    traceback.print_exc()
