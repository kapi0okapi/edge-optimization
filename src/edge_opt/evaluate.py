r"""COCO accuracy of any exported model (PyTorch, ONNX, OpenVINO, Core ML, TensorRT).

    edge-eval --models models/yolo26n.pt models/yolo26n.onnx models/yolo26n_int8.onnx \
        --data data/coco/coco.yaml --out runs/accuracy

Ultralytics' validator loads every format through the same backend wrapper, so all models
see identical preprocessing (letterbox to 640) and post-processing. On the full val2017 list
the final numbers come from the official COCO evaluation (faster-coco-eval), as in the
Ultralytics documentation. ``--images N`` evaluates on the first N images instead, with
Ultralytics' own mAP, for quick comparisons (sensitivity analysis, calibration settings).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from edge_opt.utils.threads import limit_onnxruntime_threads

log = logging.getLogger("edge_opt.evaluate")


def subset_yaml(data: Path, images: int, out_dir: Path) -> Path:
    """A copy of the dataset YAML whose val list holds only the first ``images`` images."""
    cfg = yaml.safe_load(Path(data).read_text())
    root = Path(cfg["path"])
    lines = (root / cfg["val"]).read_text().splitlines()[:images]
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / f"val_first{images}.txt").write_text("\n".join(lines) + "\n")
    path = out_dir / f"coco_first{images}.yaml"
    path.write_text(yaml.safe_dump({**cfg, "val": f"val_first{images}.txt"}, sort_keys=False))
    return path


def coco_metrics(predictions: Path, annotations: Path) -> dict[str, float]:
    """Official COCO box AP (pycocotools-equivalent faster-coco-eval) of a predictions.json."""
    from faster_coco_eval import COCO, COCOeval_faster

    truth = COCO(str(annotations))
    evaluation = COCOeval_faster(truth, truth.loadRes(str(predictions)), "bbox", print_function=log.debug)
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    return {"map50_95": float(evaluation.stats[0]), "map50": float(evaluation.stats[1])}


def count_images(data: Path) -> int:
    cfg = yaml.safe_load(Path(data).read_text())
    return len((Path(cfg["path"]) / cfg["val"]).read_text().split())


def model_name(path: str | Path) -> str:
    """``yolo26n.onnx`` -> ``yolo26n_onnx``; a training run's ``weights/best.pt`` -> ``<run>_pt``."""
    path = Path(path).resolve() if Path(path).exists() else Path(path)
    if path.name in {"best.pt", "last.pt"}:
        return f"{path.parent.parent.name}_pt"
    return path.name.replace(".", "_")


def evaluate(
    model_path: str | Path,
    data: Path,
    imgsz: int = 640,
    device: str = "cpu",
    out: Path | None = None,
    nms: bool = False,
) -> dict:
    """Checkpoints (.pt) are scored with the end-to-end head like the exports (``nms=False``),
    or with ``nms=True`` through the one-to-many head and NMS, as the documentation's main column."""
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    name = model_name(model_path) + ("-nms" if nms else "")
    extra = {"nms": None if nms else False} if str(model_path).endswith(".pt") else {}
    metrics = model.val(
        **extra,
        data=str(data),
        imgsz=imgsz,
        batch=1,
        device=device,
        plots=False,
        verbose=False,
        project=str(out or Path("runs/val")),
        name=name,
        exist_ok=True,
    )
    report = {
        "model": str(model_path),
        "name": name,
        "data": str(data),
        "images": count_images(data),
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "metric": "ultralytics",
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
        "device": device,
    }
    # on the full val2017 list Ultralytics also writes COCO-format predictions: score them officially
    predictions = Path(metrics.save_dir) / "predictions.json"
    annotations = (
        Path(yaml.safe_load(Path(data).read_text())["path"]) / "annotations" / "instances_val2017.json"
    )
    if predictions.exists() and annotations.exists() and report["images"] >= 5000:
        report["ultralytics"] = {"map50_95": report["map50_95"], "map50": report["map50"]}
        report |= coco_metrics(predictions, annotations) | {"metric": "coco"}
    return report


def main(argv: list[str] | None = None) -> None:
    limit_onnxruntime_threads()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, help="only the first N val images (quick runs)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nms", action="store_true", help=".pt only: one-to-many head + NMS")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    data = subset_yaml(args.data, args.images, args.out) if args.images else args.data
    suffix = f"_first{args.images}" if args.images else ""
    for path in args.models:
        nms = args.nms and str(path).endswith(".pt")
        target = args.out / f"{model_name(path)}{'-nms' if nms else ''}{suffix}.json"
        if target.exists():
            log.info("%s already evaluated", target.name)
            continue
        report = evaluate(path, data, args.imgsz, args.device, args.out / "val", nms)
        target.write_text(json.dumps(report, indent=2))
        print(f"{report['name']}: mAP50-95 {report['map50_95']:.4f}, mAP50 {report['map50']:.4f}")


if __name__ == "__main__":
    main()
