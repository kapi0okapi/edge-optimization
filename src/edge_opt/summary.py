"""Accuracy and latency tables, the sensitivity ranking and an accuracy-latency plot.

edge-summary --accuracy runs/accuracy --latency runs/latency --sensitivity runs/sensitivity --out runs/summary
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Ultralytics documentation, COCO val2017 mAP50-95 at 640: (standard head with NMS, end-to-end head)
DOCUMENTED = {"yolo26n": (40.9, 40.1), "yolo26s": (48.6, 47.8), "yolo26m": (53.1, 52.5)}

LABELS = [
    (r"_pt$", "PyTorch FP32"),
    (r"_pt-nms$", "PyTorch FP32, one-to-many head + NMS"),
    (r"_ft_pt$", "PyTorch FP32, fine-tuned control"),
    (r"_qat_pt$", "PyTorch, QAT fake-quant"),
    (r"_fp32_onnx$", "ONNX FP32"),
    (r"_ul-int8_onnx$", "ONNX INT8, Ultralytics export"),
    (r"_int8-minmax_onnx$", "ONNX INT8, head decode in float"),
    (r"_int8-cls-float-(\w+)_onnx$", "ONNX INT8, class branch in float ({0})"),
    (r"_qat_onnx$", "ONNX INT8, QAT"),
    (r"_fp32_openvino_model$", "OpenVINO FP32"),
    (r"_int8_openvino_model$", "OpenVINO INT8 (NNCF)"),
    (r"_fp16_mlpackage$", "Core ML FP16"),
    (r"_w8_mlpackage$", "Core ML 8-bit weights"),
    (r"_fp16_engine$", "TensorRT FP16"),
    (r"_int8_engine$", "TensorRT INT8 (ModelOpt PTQ)"),
    (r"_qat_engine$", "TensorRT INT8, QAT"),
]


def split_name(name: str) -> tuple[str, str]:
    """``yolo26n_int8-cls-float-entropy_onnx`` -> model ``yolo26n`` and a readable variant label."""
    model = name.split("_")[0]
    label = name
    for pattern, text in LABELS:  # later, more specific patterns win
        if match := re.search(pattern, name):
            label = text.format(*match.groups())
    return model, label


def load(directory: Path | None, pattern: str = "*.json") -> list[dict]:
    if directory is None or not Path(directory).exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(Path(directory).rglob(pattern))]


def accuracy_table(reports: list[dict]) -> str:
    full = [r for r in reports if r.get("images", 5000) >= 5000]
    floats = {split_name(r["name"])[0]: r["map50_95"] for r in full if r["name"].endswith("_fp32_onnx")}
    lines = [
        "| Model | Variant | mAP50-95 | vs ONNX FP32 | Ultralytics docs |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(full, key=lambda r: (split_name(r["name"])[0], -r["map50_95"])):
        model, label = split_name(r["name"])
        ref = floats.get(model)
        delta = f"{100 * (r['map50_95'] - ref):+.1f}" if ref is not None else "–"
        doc = DOCUMENTED.get(model)
        doc_cell = ""
        if doc and label == "PyTorch FP32, one-to-many head + NMS":
            doc_cell = f"{doc[0]:.1f}"
        elif doc and label in ("PyTorch FP32", "ONNX FP32"):
            doc_cell = f"{doc[1]:.1f} (end-to-end)"
        delta = delta.replace("-0.0", "0.0").replace("+0.0", "0.0")
        lines.append(f"| {model} | {label} | {100 * r['map50_95']:.1f} | {delta} | {doc_cell} |")
    return "\n".join(lines)


def latency_table(reports: list[dict]) -> str:
    columns = sorted({(r["hardware"], r["runtime"], r["threads"]) for r in reports})
    rows = sorted({r["name"] for r in reports}, key=lambda n: split_name(n))
    cell = {(r["name"], r["hardware"], r["runtime"], r["threads"]): r["median_ms"] for r in reports}
    head = " | ".join(f"{hw} {rt}" + (f" {t}T" if t else "") for hw, rt, t in columns)
    lines = [f"| Model | Variant | MB | {head} |", "|---" * (len(columns) + 3) + "|"]
    sizes = {r["name"]: r["size_mb"] for r in reports}
    for name in rows:
        model, label = split_name(name)
        cells = [f"{cell[(name, *c)]:.1f}" if (name, *c) in cell else "–" for c in columns]
        lines.append(f"| {model} | {label} | {sizes[name]:.1f} | " + " | ".join(cells) + " |")
    return "Median latency in ms, batch 1, model call only.\n\n" + "\n".join(lines)


def sensitivity_table(directory: Path | None, top: int = 6) -> str:
    out = []
    for path in sorted(Path(directory).glob("*/sensitivity.json")) if directory else []:
        data = json.loads(path.read_text())
        float_map, int8_map = data["float"]["map50_95"], data["int8"]["map50_95"]
        lines = [
            f"**{path.parent.name}**: float {100 * float_map:.1f}, INT8 {100 * int8_map:.1f} "
            f"(first {data.get('images', 500)} val images)",
            "",
            "| Kept in float | mAP50-95 recovered |",
            "|---|---|",
        ]
        lines += [f"| {r['group']} | {100 * r['gain']:+.1f} |" for r in data["ranking"][:top]]
        for key in sorted(k for k in data if k.startswith("top")):
            groups = ", ".join(data[key].get("groups", []))
            lines.append(
                f"| top {key[3:]} together ({groups}) | {100 * (data[key]['map50_95'] - int8_map):+.1f} |"
            )
        out.append("\n".join(lines))
    return "\n\n".join(out)


RUNTIME_STYLE = {  # runtime -> (legend, colour)
    "ort-cpu": ("ONNX Runtime, CPU", "tab:blue"),
    "openvino": ("OpenVINO, CPU", "tab:purple"),
    "ort-coreml": ("ONNX Runtime, Core ML provider", "tab:green"),
    "coreml-all": ("Core ML (Neural Engine)", "tab:orange"),
    "ultralytics": ("TensorRT", "tab:red"),
    "ort-tensorrt": ("ONNX Runtime, TensorRT provider", "tab:brown"),
}
SHORT = [  # variant label -> short text on the plot
    ("Ultralytics export", "INT8 Ultralytics"),
    ("class branch in float (", "INT8 cls-float {}"),
    ("head decode in float", "INT8 head-float"),
    ("QAT", "INT8 QAT"),
    ("NNCF", "INT8 NNCF"),
    ("8-bit weights", "W8"),
    ("ModelOpt", "INT8"),
    ("FP16", "FP16"),
    ("FP32", "FP32"),
]


def short_label(label: str) -> str:
    for key, text in SHORT:
        if key in label:
            inner = label.split("(")[-1].rstrip(")") if "{}" in text else ""
            return text.format(inner)
    return label


def plot(accuracy: list[dict], latency: list[dict], hardware: str, path: Path) -> None:
    """mAP against latency on one machine: each runtime at its fastest thread count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = {r["name"]: r["map50_95"] for r in accuracy if r.get("images", 5000) >= 5000}
    best: dict[tuple[str, str], dict] = {}
    for r in latency:
        if r["hardware"] != hardware or r["name"] not in acc or r["runtime"] not in RUNTIME_STYLE:
            continue
        key = (r["name"], r["runtime"])
        if key not in best or r["median_ms"] < best[key]["median_ms"]:
            best[key] = r
    fig, ax = plt.subplots(figsize=(8, 5))
    seen = set()
    for i, ((name, runtime), r) in enumerate(sorted(best.items())):
        legend, colour = RUNTIME_STYLE[runtime]
        y = 100 * acc[name]
        ax.scatter(r["median_ms"], y, color=colour, s=45, label=None if legend in seen else legend, zorder=3)
        seen.add(legend)
        ax.annotate(
            short_label(split_name(name)[1]),
            (r["median_ms"], y),
            fontsize=7.5,
            xytext=(5, 4 if i % 2 else -9),
            textcoords="offset points",
        )
    ax.set_xlabel(f"median latency, batch 1 [ms] ({hardware})")
    ax.set_ylabel("COCO val2017 mAP50-95")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path)
    parser.add_argument("--latency", type=Path)
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    accuracy = [r for r in load(args.accuracy) if "map50_95" in r]
    latency = [r for r in load(args.latency) if "median_ms" in r]
    parts = [
        accuracy_table(accuracy),
        latency_table(latency) if latency else "",
        sensitivity_table(args.sensitivity),
    ]
    text = "\n\n".join(p for p in parts if p)
    (args.out / "summary.md").write_text(text + "\n")
    print(text)
    for hardware in sorted({r["hardware"] for r in latency}):
        plot(accuracy, latency, hardware, args.out / f"pareto_{hardware}.png")


if __name__ == "__main__":
    main()
