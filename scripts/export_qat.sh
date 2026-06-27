#!/bin/bash
# After the qat / qat-float-control jobs: copy the best checkpoints into $RUNS/models and export
# the QAT ones to Q/DQ ONNX and TensorRT INT8. The quantization ranges come from training, so no
# calibration data is needed. Runs inside the export-qat Slurm job (GPU for TensorRT).
set -euo pipefail
for size in n s; do
    qat="$RUNS/qat/qat_yolo26${size}/weights/best.pt"
    ft="$RUNS/qat/ft_yolo26${size}/weights/best.pt"
    [[ -f "$ft" ]] && cp "$ft" "$RUNS/models/yolo26${size}_ft.pt"
    [[ -f "$qat" ]] || continue
    cp "$qat" "$RUNS/models/yolo26${size}_qat.pt"
    python - "$RUNS/models/yolo26${size}_qat.pt" <<'PY'
import sys
from ultralytics import YOLO

for fmt in ("onnx", "engine"):
    print(YOLO(sys.argv[1]).export(format=fmt, quantize=8, nms=False, imgsz=640, batch=1, device=0))
PY
done
