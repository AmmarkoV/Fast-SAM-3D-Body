#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"

valgrind --tool=callgrind  ../build/fast_sam_3dbody_render --onnx-dir ../onnx --gguf ../onnx/pipeline.gguf --yolo ../onnx/yolo.onnx --mesh ../body_mesh.tri --from ../../notebook/images/dancing.jpg --save /tmp/render_test.png > /tmp/render_raw.txt $@ 2>error.txt  
exit 0
