#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

./build/fast_sam_3dbody_render --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --mesh ./body_mesh.tri --lbs onnx/body_model.lbs --from scripts/CardioWorkoutCut3.avi $@ 


#./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from /dev/video0 > /tmp/render_raw.txt $@ 

exit 0
