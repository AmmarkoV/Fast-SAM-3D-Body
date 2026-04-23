#!/usr/bin/env python3
"""
SAM 3D Body Webcam Demo - Real-time 3D body estimation from webcam
Usage: python demo_webcam.py [--camera-index 0] [--model facebook/sam-3d-body-dinov3]

 FOV_MODEL=s FOV_LEVEL=0 MHR_NO_CORRECTIVES=1 python demo_webcam.py --detector yolo --detector_model ./checkpoints/yolo/yolo11m.engine

Controls:
  q / ESC - quit
  s       - save current frame to output directory
"""

import argparse
import os
import sys
import time

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
import torch

from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together


def main(args):
    print("=" * 60)
    print("SAM 3D Body - Webcam Demo")
    print("=" * 60)
    print(f"Camera: {args.camera_index}")
    print(f"Model: {args.model}")
    print(f"Detector: {args.detector}" + (f" ({args.detector_model})" if args.detector in ["yolo", "yolo_pose"] else ""))
    print(f"Local Checkpoint: {'yes (' + args.local_checkpoint + ')' if args.local_checkpoint else 'no (using HuggingFace)'}")
    print(f"Frame Skip: {args.frame_skip} (process 1 in {args.frame_skip + 1} frames)")
    print()
    print("Controls: q/ESC=quit  s=save frame")
    print()

    # ============ Load model ============
    print("[*] Loading model...")
    t0 = time.time()

    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name=args.detector,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"[*] Model loaded ({elapsed:.1f}s) on GPU")
    else:
        elapsed = time.time() - t0
        print(f"[*] Model loaded ({elapsed:.1f}s) on CPU")

    # ============ Open webcam ============
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[!] Failed to open camera {args.camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ============ Main loop ============
    os.makedirs(args.output_dir, exist_ok=True)
    frame_count = 0
    last_frame = None

    while True:
        ret, img_bgr = cap.read()
        if not ret:
            print("[!] Failed to read frame")
            break

        # Throttle processing
        frame_count += 1
        if frame_count % (args.frame_skip + 1) != 0:
            # Display cached result
            if last_frame is not None:
                cv2.imshow("SAM 3D Body - Webcam", last_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        # Run inference
        # Save frame temporarily for the estimator
        tmp_path = os.path.join(args.output_dir, "_tmp_frame.jpg")
        cv2.imwrite(tmp_path, img_bgr)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = estimator.process_one_image(
            tmp_path,
            hand_box_source=args.hand_box_source,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.time() - t0

        # Build visualization
        if outputs:
            vis = visualize_sample_together(img_bgr, outputs, estimator.faces).astype(np.uint8)
        else:
            vis = img_bgr.copy()

        # Draw FPS / latency overlay
        fps = 1.0 / inference_time if inference_time > 0 else 0
        h, w = vis.shape[:2]
        info_h = int(h * 0.08)
        vis[0:info_h, 0:300] = [0, 0, 0]
        cv2.putText(vis, f"FPS: {fps:.1f}  ({inference_time*1000:.0f}ms)", (10, int(info_h * 0.55)),
                     cv2.FONT_HERSHEY_SIMPLEX, info_h * 0.018, (0, 255, 255), 2)
        cv2.putText(vis, f"Persons: {len(outputs)}", (10, int(info_h * 0.85)),
                     cv2.FONT_HERSHEY_SIMPLEX, info_h * 0.018, (0, 255, 255), 2)

        last_frame = vis
        cv2.imshow("SAM 3D Body - Webcam", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_path = os.path.join(args.output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(save_path, vis)
            print(f"  Saved: {save_path}")
        elif key in (ord("q"), 27):
            break

    # ============ Cleanup ============
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("[*] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3D Body Webcam Demo")

    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", type=str, default="facebook/sam-3d-body-dinov3",
                        choices=["facebook/sam-3d-body-dinov3", "facebook/sam-3d-body-vith"],
                        help="Model to use (default: facebook/sam-3d-body-dinov3)")
    parser.add_argument("--detector", type=str, default="yolo",
                        choices=["vitdet", "yolo", "yolo_pose"],
                        help="Person detector (default: yolo)")
    parser.add_argument("--hand-box-source", type=str, default="body_decoder",
                        choices=["body_decoder", "yolo_pose"],
                        help="Hand box source (default: body_decoder)")
    parser.add_argument("--detector-model", type=str, default="./checkpoints/yolo/yolo11n.pt",
                        help="YOLO model path (default: ./checkpoints/yolo/yolo11n.pt)")
    parser.add_argument("--local-checkpoint", type=str, default="./checkpoints/sam-3d-body-dinov3",
                        help="Local checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="./webcam_output",
                        help="Directory to save frames when pressing 's' (default: ./webcam_output)")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Process 1 in N frames to reduce GPU load (default: 4, i.e. ~every 5th frame)")

    main(parser.parse_args())
