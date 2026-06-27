"""INT8 post-training quantization of an exported YOLO26 ONNX model with ONNX Runtime.

    edge-quantize --model models/yolo26n.onnx --calib data/coco/calib2017.txt --out models/yolo26n_int8.onnx
    edge-quantize ... --exclude head-postprocess          # keep the box/score assembly in float
    edge-quantize ... --exclude head-postprocess --float-groups 23/one2one_cv3 --method percentile

Static quantization in QDQ format: weights int8, symmetric, one scale per output channel;
activations int8 with ranges from calibration images. Ranges come from MinMax, Entropy or
Percentile calibration. Nodes can be left in float: a rule for YOLO26's end-to-end head
(``head-postprocess``), whole model blocks (``--exclude-blocks``), or an explicit list.

The rule exists because the NMS-free head ends by concatenating box coordinates in pixels
(0-640) with class probabilities (0-1) into one tensor before top-k selection. One int8
scale for both leaves the probabilities with a step of ~2.5: every score rounds to zero.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import onnx

from edge_opt.utils.threads import limit_onnxruntime_threads

METHODS = {"minmax": "MinMax", "entropy": "Entropy", "percentile": "Percentile"}
_BLOCK = re.compile(r"^/model\.(\d+)/")


def preprocess(path: str | Path, size: int) -> np.ndarray:
    """Letterbox like Ultralytics (gray padding, centred), RGB, [0, 1], NCHW float32."""
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top : top + nh, left : left + nw] = image
    return np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def read_image_list(path: Path) -> list[Path]:
    """An Ultralytics ``*.txt`` image list (paths relative to its folder) or a folder of images."""
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return [(path.parent / line.strip()).resolve() for line in path.read_text().splitlines() if line.strip()]


class ImageReader:
    """ONNX Runtime calibration data reader over a list of images. ``set_range`` lets the
    calibrator consume it in chunks: the histogram methods (Entropy, Percentile) otherwise keep
    every activation of every image in memory until the end, which for 512 images at 640 x 640
    is far more than a laptop has."""

    def __init__(self, images: Iterable[Path], input_name: str, size: int):
        self.images, self.input_name, self.size = list(images), input_name, size
        self.index, self.end = 0, len(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.index, self.end = start_index, min(end_index, len(self.images))

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= self.end:
            return None
        self.index += 1
        return {self.input_name: preprocess(self.images[self.index - 1], self.size)}

    def rewind(self) -> None:
        self.index, self.end = 0, len(self.images)


def block_of(node_name: str) -> int | None:
    match = _BLOCK.match(node_name)
    return int(match.group(1)) if match else None


def head_postprocess_nodes(model: onnx.ModelProto) -> list[str]:
    """Nodes of the last block that are not part of its convolution branches: the decoding of
    boxes, the concatenation with the scores and the top-k selection."""
    last = max(b for n in model.graph.node if (b := block_of(n.name)) is not None)
    prefix = f"/model.{last}/"
    return [
        n.name
        for n in model.graph.node
        # "/model.23/Concat_3" is post-processing, "/model.23/cv2.0/..." a convolution branch
        if n.name.startswith(prefix) and n.name.count("/") <= 2
    ]


def nodes_in_blocks(model: onnx.ModelProto, blocks: Iterable[int]) -> list[str]:
    blocks = set(blocks)
    return [n.name for n in model.graph.node if block_of(n.name) in blocks]


def group_of(node_name: str, head_block: int) -> str | None:
    """``/model.7/cv1/conv/Conv`` -> ``7``; ``/model.23/one2one_cv3.1/...`` -> ``23/one2one_cv3``."""
    block = block_of(node_name)
    if block is None:
        return None
    if block != head_block:
        return str(block)
    parts = node_name.split("/")
    if len(parts) <= 3:
        return None
    branch = re.sub(r"\.\d+$", "", parts[2])
    return f"{block}/{branch}"


def groups(model: onnx.ModelProto) -> dict[str, list[str]]:
    head = max(b for n in model.graph.node if (b := block_of(n.name)) is not None)
    out: dict[str, list[str]] = {}
    for node in model.graph.node:
        if (g := group_of(node.name, head)) is not None:
            out.setdefault(g, []).append(node.name)
    return dict(sorted(out.items(), key=lambda kv: (int(kv[0].split("/")[0]), kv[0])))


def copy_metadata(source: onnx.ModelProto, target_path: Path) -> None:
    """Ultralytics reads class names, stride and the end-to-end flag from the ONNX metadata."""
    target = onnx.load(str(target_path))
    have = {p.key for p in target.metadata_props}
    for prop in source.metadata_props:
        if prop.key not in have:
            target.metadata_props.add(key=prop.key, value=prop.value)
    onnx.save(target, str(target_path))


def quantize(
    model_path: Path,
    images: list[Path],
    out: Path,
    method: str = "minmax",
    exclude: list[str] | None = None,
    per_channel: bool = True,
    percentile: float = 99.999,
    cache: Path | None = None,
    chunk: int = 16,
) -> dict:
    """``cache``: a JSON file with calibration ranges; written on the first call, reused after
    (the ranges of all tensors are collected, so excluding other nodes later is fine)."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # The calibration cache needs onnxruntime >= 1.25; ModelOpt pins the CUDA build to 1.24 on
    # Linux, where every variant then calibrates on its own (slower, same ranges).
    if "calibration_cache_path" not in inspect.signature(quantize_static).parameters:
        cache = None
    cache_kwargs = {"calibration_cache_path": cache} if cache else {}
    model = onnx.load(str(model_path))
    size = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    images = list(images)[: len(images) // chunk * chunk]  # calibration runs in chunks of ``chunk``
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(str(model_path), str(prepared), skip_symbolic_shape=True)
        quantize_static(
            str(prepared),
            str(out),
            None if cache and Path(cache).is_file() else ImageReader(images, model.graph.input[0].name, size),
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=exclude or [],
            calibrate_method=getattr(CalibrationMethod, METHODS[method]),
            extra_options={
                "CalibStridedMinMax": chunk,
                **({"CalibPercentile": percentile} if method == "percentile" else {}),
            },
            **cache_kwargs,
        )
    copy_metadata(model, out)
    report = {
        "model": str(model_path),
        "output": str(out),
        "method": method,
        "per_channel": per_channel,
        "calibration_images": len(images),
        "excluded_nodes": len(exclude or []),
        "excluded": exclude or [],
        "size_mb": out.stat().st_size / 1e6,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    limit_onnxruntime_threads()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True, help="float ONNX model (nms=False export)")
    parser.add_argument("--calib", type=Path, required=True, help="image list (.txt) or folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, default=512, help="calibration images to use")
    parser.add_argument("--method", choices=sorted(METHODS), default="minmax")
    parser.add_argument("--exclude", choices=["none", "head-postprocess"], default="none")
    parser.add_argument("--exclude-blocks", type=int, nargs="*", default=[])
    parser.add_argument("--exclude-nodes", nargs="*", default=[])
    parser.add_argument(
        "--float-groups", nargs="*", default=[], help="groups from edge-sensitivity, e.g. 23/one2one_cv3"
    )
    parser.add_argument("--per-tensor", action="store_true", help="one weight scale per tensor")
    args = parser.parse_args(argv)

    model = onnx.load(str(args.model))
    exclude = list(args.exclude_nodes) + nodes_in_blocks(model, args.exclude_blocks)
    parts = groups(model)
    exclude += [n for g in args.float_groups for n in parts[g]]
    if args.exclude == "head-postprocess":
        exclude += head_postprocess_nodes(model)
    images = read_image_list(args.calib)[: args.images]
    report = quantize(
        args.model, images, args.out, args.method, sorted(set(exclude)), per_channel=not args.per_tensor
    )
    print(json.dumps({k: v for k, v in report.items() if k != "excluded"}, indent=2))


if __name__ == "__main__":
    main()
