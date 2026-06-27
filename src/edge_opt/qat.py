"""Quantization-aware training (QAT) of YOLO26 with Ultralytics and NVIDIA ModelOpt.

    edge-qat --weights yolo26n.pt --data data/coco_full/coco.yaml --epochs 10 --name qat_yolo26n
    edge-qat --weights yolo26n.pt --data data/coco_full/coco.yaml --epochs 10 --name ft_yolo26n --float

Ultralytics (>= 8.4.143) inserts ModelOpt fake-quantizers in front of every convolution
(``train(quantize=8)``), calibrates their ranges on a few batches and then fine-tunes the
weights with the ranges fixed. The head's bare output convolutions stay in float. The
checkpoint exports straight to ONNX with Q/DQ nodes (no calibration at export), which ONNX
Runtime and TensorRT run in INT8.

QAT here is a short fine-tune of the released COCO weights, not training from scratch: SGD
with a learning rate 10x below the usual YOLO fine-tuning value, cosine decay, a short
warm-up, mosaic switched off for the last epochs. ``--float`` runs the identical schedule
without quantizers: the control that separates what QAT recovers from what the extra
training alone changes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def train(
    weights: str,
    data: Path,
    name: str,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 64,
    lr0: float = 1e-3,
    quantize: bool = True,
    device: str | None = None,
    project: Path | None = None,
    workers: int = 8,
    fraction: float = 1.0,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    results_dir = Path(project or Path(os.environ.get("RUNS", "runs")) / "qat")
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        quantize=8 if quantize else None,
        optimizer="SGD",
        lr0=lr0,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=0.5,
        close_mosaic=max(1, epochs // 3),
        device=device,
        workers=workers,
        fraction=fraction,
        project=str(results_dir),
        name=name,
        exist_ok=True,
        plots=False,
    )
    return results_dir / name / "weights" / "best.pt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--data", type=Path, required=True, help="dataset YAML with the full train2017 split")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--float", dest="quantize", action="store_false", help="control run without QAT")
    parser.add_argument("--device")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="share (or count) of train images")
    args = parser.parse_args(argv)
    best = train(
        args.weights,
        args.data,
        args.name,
        args.epochs,
        args.imgsz,
        args.batch,
        args.lr0,
        args.quantize,
        args.device,
        args.project,
        args.workers,
        args.fraction,
    )
    print(best)


if __name__ == "__main__":
    main()
