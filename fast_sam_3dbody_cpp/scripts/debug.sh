#!/bin/bash
#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./ModelDumpD Ammar.obj Ammar $@ 2>error.tx
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"

valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ../build/fast_sam_3dbody_render --onnx-dir ../onnx --gguf ../onnx/pipeline.gguf --yolo ../onnx/yolo.onnx --mesh ../body_mesh.tri --from ../../notebook/images/dancing.jpg --save /tmp/render_test.png > /tmp/render_raw.txt $@ 2> ../error.txt 


exit $?
