#!/bin/bash
# COCO (val2017, the calibration subset and, for QAT, all of train2017) and the official YOLO26
# checkpoints. Run on a login node: compute nodes may lack internet.
#
#   bash scripts/download_data.sh            # into $DATA/coco and $RUNS/models (cluster) or ./data, ./runs
#   bash scripts/download_data.sh --small    # without train2017: enough for everything except QAT
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:-data}"
RUNS="${RUNS:-runs}"
FULL="--full"
[[ "${1:-}" == "--small" ]] && FULL=""
python -m edge_opt.data --out "$DATA/coco" $FULL
mkdir -p "$RUNS/models"
python - "$RUNS/models" <<'PY'
import shutil, sys
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset

for name in ("yolo26n.pt", "yolo26s.pt"):
    target = Path(sys.argv[1]) / name
    if not target.exists():
        shutil.move(attempt_download_asset(name), target)
    print(target)
PY
