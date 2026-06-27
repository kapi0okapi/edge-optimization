r"""Export a YOLO26 checkpoint to every deployment format compared here.

    edge-export --weights yolo26n.pt --data data/coco/coco.yaml --out models \
        --variants onnx ul-int8 openvino openvino-int8

All exports use the NMS-free one-to-one head (``nms=False``): the network itself returns at
most 300 boxes and there is no NMS step to port or to time. INT8 variants are calibrated on
the train2017 subset (``split="train"``), never on val. Each export runs from its own copy of
the checkpoint, because Ultralytics names its outputs after the checkpoint and removes the
float ONNX it quantized. Results are named by precision: ``yolo26n_fp32.onnx``,
``yolo26n_int8_openvino_model``, ``yolo26n_fp16.engine``, ...

Variants:
    onnx            float32 ONNX (input for edge-quantize and the ONNX Runtime rows)
    ul-int8         Ultralytics' INT8 ONNX: ONNX Runtime static quantization of Conv/MatMul only
    openvino        OpenVINO IR, float
    openvino-int8   OpenVINO IR, INT8 by NNCF post-training quantization (head and Sigmoids in float)
    coreml          Core ML, float16 (macOS; can run on the Neural Engine)
    coreml-int8     Core ML with 8-bit palettized weights, float16 activations
    engine          TensorRT, float16 (NVIDIA GPU, built on the GPU it will run on)
    engine-int8     TensorRT, INT8 through explicit Q/DQ nodes from NVIDIA ModelOpt
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

# variant: (export arguments, precision tag used in the file name)
VARIANTS = {
    "onnx": ({"format": "onnx"}, "fp32"),
    "ul-int8": ({"format": "onnx", "quantize": 8}, "ul-int8"),
    "openvino": ({"format": "openvino"}, "fp32"),
    "openvino-int8": ({"format": "openvino", "quantize": 8}, "int8"),
    "coreml": ({"format": "coreml", "quantize": 16}, "fp16"),
    "coreml-int8": ({"format": "coreml", "quantize": 8}, "w8"),
    "engine": ({"format": "engine", "quantize": 16}, "fp16"),
    "engine-int8": ({"format": "engine", "quantize": 8}, "int8"),
}
# formats whose INT8 export needs calibration images (Core ML palettizes weights without data)
CALIBRATED = {"onnx", "openvino", "engine"}
# Ultralytics recognises a format by these endings when loading a model
SUFFIX = {"onnx": ".onnx", "openvino": "_openvino_model", "coreml": ".mlpackage", "engine": ".engine"}


def export(
    weights: Path,
    variant: str,
    out: Path,
    data: Path | None = None,
    imgsz: int = 640,
    calibration_images: int = 512,
    device: str | None = None,
) -> Path:
    from ultralytics import YOLO

    export_args, tag = VARIANTS[variant]
    args = dict(export_args, nms=False, imgsz=imgsz, batch=1)
    if args.get("quantize") == 8 and args["format"] in CALIBRATED:
        if data is None:
            raise ValueError(f"{variant} needs --data for calibration")
        args |= {"data": str(data), "split": "train", "fraction": calibration_images}
    if device:
        args["device"] = device
    stem = Path(weights).stem
    target = out / f"{stem}_{tag}{SUFFIX[args['format']]}"  # e.g. yolo26n_int8_openvino_model
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        copy = Path(tmp) / Path(weights).name
        shutil.copy(weights, copy)
        started = time.perf_counter()
        produced = Path(YOLO(str(copy)).export(**args))
        seconds = time.perf_counter() - started
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(produced, target)
    info = {"weights": str(weights), "variant": variant, "args": args, "export_seconds": seconds}
    target.with_name(target.name + ".json").write_text(json.dumps(info, indent=2))
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=Path, required=True, help="e.g. yolo26n.pt (downloaded if missing)")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["onnx"])
    parser.add_argument("--data", type=Path, help="dataset YAML with the calibration images as 'train'")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calibration-images", type=int, default=512)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    weights = args.weights
    if not weights.exists():  # an official checkpoint name: download it once into --out
        from ultralytics.utils.downloads import attempt_download_asset

        weights = args.out / weights.name
        if not weights.exists():
            shutil.move(attempt_download_asset(weights.name), weights)
    for variant in args.variants:
        path = export(weights, variant, args.out, args.data, args.imgsz, args.calibration_images, args.device)
        print(f"{variant}: {path}")


if __name__ == "__main__":
    main()
