"""Inference latency of exported models: batch 1, model call only, after warm-up.

    edge-latency --models models/yolo26n.onnx models/yolo26n_int8.onnx --runtime ort-cpu --threads 1 4
    edge-latency --models models/yolo26n_openvino_model --runtime openvino --threads 4
    edge-latency --models models/yolo26n.mlpackage --runtime coreml       # Apple Neural Engine / GPU / CPU
    edge-latency --models models/yolo26n.engine --runtime ultralytics     # TensorRT on the GPU cluster

Preprocessing and decoding are excluded: they are the same for every variant and depend on
the camera pipeline. The input is a real letterboxed image, not zeros. Each measurement is
the median of ``--runs`` calls; the 90th percentile shows the jitter that matters for a
camera running at a fixed frame rate.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from edge_opt.evaluate import model_name
from edge_opt.quantize import preprocess

RUNTIMES = ("ort-cpu", "ort-coreml", "ort-cuda", "ort-tensorrt", "openvino", "coreml", "ultralytics")


def sample_image() -> Path:
    from ultralytics.utils import ASSETS

    return ASSETS / "bus.jpg"


class OrtRunner:
    def __init__(self, path: Path, runtime: str, threads: int | None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        providers = {
            "ort-cpu": ["CPUExecutionProvider"],
            "ort-coreml": [("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}), "CPUExecutionProvider"],
            "ort-cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "ort-tensorrt": [
                ("TensorrtExecutionProvider", {"trt_int8_enable": True, "trt_fp16_enable": True}),
                "CUDAExecutionProvider",
            ],
        }[runtime]
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.input = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray):
        return self.session.run(None, {self.input: x})


class OpenVinoRunner:
    def __init__(self, path: Path, threads: int | None, device: str = "CPU"):
        import openvino as ov

        xml = next(Path(path).glob("*.xml")) if Path(path).is_dir() else Path(path)
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            config["INFERENCE_NUM_THREADS"] = threads
        self.model = ov.Core().compile_model(str(xml), device, config)
        self.request = self.model.create_infer_request()

    def __call__(self, x: np.ndarray):
        return self.request.infer({0: x})


class CoreMLRunner:
    def __init__(self, path: Path, units: str = "ALL"):
        import coremltools as ct

        self.model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
        spec = self.model.get_spec()
        self.input = spec.description.input[0].name
        self.image_input = spec.description.input[0].type.WhichOneof("Type") == "imageType"

    def __call__(self, x: np.ndarray):
        if self.image_input:
            from PIL import Image

            x = Image.fromarray((x[0].transpose(1, 2, 0) * 255).round().astype(np.uint8))
        return self.model.predict({self.input: x})


class UltralyticsRunner:
    """Any format through Ultralytics' AutoBackend (TensorRT engines, TorchScript, PyTorch)."""

    def __init__(self, path: Path, device: str):
        import torch
        from ultralytics.nn.autobackend import AutoBackend

        self.torch = torch
        self.device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)  # "0" -> cuda:0
        self.backend = AutoBackend(str(path), device=self.device, fp16=False, fuse=True)

    def __call__(self, x: np.ndarray):
        im = self.torch.from_numpy(x).to(self.device)
        if getattr(self.backend, "fp16", False):
            im = im.half()
        out = self.backend(im)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        return out


def make_runner(path: Path, runtime: str, threads: int | None, device: str, coreml_units: str):
    if runtime.startswith("ort-"):
        return OrtRunner(path, runtime, threads)
    if runtime == "openvino":
        return OpenVinoRunner(path, threads)
    if runtime == "coreml":
        return CoreMLRunner(path, coreml_units)
    return UltralyticsRunner(path, device)


def measure(runner, x: np.ndarray, warmup: int = 20, runs: int = 200) -> dict:
    for _ in range(warmup):
        runner(x)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        runner(x)
        times.append((time.perf_counter() - started) * 1000)
    q = statistics.quantiles(times, n=10)
    return {
        "median_ms": statistics.median(times),
        "p90_ms": q[-1],
        "mean_ms": statistics.fmean(times),
        "runs": runs,
    }


def size_mb(path: Path) -> float:
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / 1e6


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--runtime", choices=RUNTIMES, default="ort-cpu")
    parser.add_argument("--threads", type=int, nargs="+", default=[0], help="0 = runtime default")
    parser.add_argument("--device", default="cpu", help="for --runtime ultralytics")
    parser.add_argument(
        "--coreml-units", default="ALL", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"]
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("runs/latency"))
    parser.add_argument("--tag", default=platform.node().split(".")[0], help="hardware label in the output")
    args = parser.parse_args(argv)

    x = preprocess(sample_image(), args.imgsz)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.models:
        for threads in args.threads:
            runner = make_runner(path, args.runtime, threads or None, args.device, args.coreml_units)
            result = measure(runner, x, runs=args.runs)
            runtime = args.runtime + (f"-{args.coreml_units.lower()}" if args.runtime == "coreml" else "")
            report = {
                "model": str(path),
                "name": model_name(path),
                "runtime": runtime,
                "threads": threads,
                "hardware": args.tag,
                "processor": platform.processor() or platform.machine(),
                "size_mb": size_mb(path),
                **result,
            }
            name = f"{report['name']}_{runtime}_t{threads}_{args.tag}.json"
            (args.out / name).write_text(json.dumps(report, indent=2))
            print(
                f"{report['name']:>28} {runtime:>14} threads={threads}: "
                f"{result['median_ms']:.1f} ms (p90 {result['p90_ms']:.1f})"
            )


if __name__ == "__main__":
    main()
