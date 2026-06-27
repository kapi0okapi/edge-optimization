r"""Which parts of the network lose accuracy in INT8? Leave-one-group-out analysis.

    edge-sensitivity --model models/yolo26n.onnx --data data/coco/coco.yaml \
        --calib data/coco/calib2017.txt --images 500 --keep-top 1 2 3 --out runs/sensitivity/yolo26n

The starting point is the INT8 model with the head's post-processing in float (without that,
mAP is zero). Each group of nodes is then kept in float in turn and the model is scored on the
first N val images: the mAP it recovers is that group's quantization cost. Groups are the
network's blocks (``model.0`` ... ``model.22``) and, in the detection head, the box and the
class branches separately. Finally the top k groups are kept in float together, which gives
the mixed-precision models on the accuracy-latency curve.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import onnx

from edge_opt.evaluate import evaluate, subset_yaml
from edge_opt.quantize import groups, head_postprocess_nodes, quantize, read_image_list
from edge_opt.utils.threads import limit_onnxruntime_threads

log = logging.getLogger("edge_opt.sensitivity")


def main(argv: list[str] | None = None) -> None:
    limit_onnxruntime_threads()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, default=500, help="val images per evaluation")
    parser.add_argument("--calib-images", type=int, default=512)
    parser.add_argument("--method", default="minmax")
    parser.add_argument("--keep-top", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.model))
    base_exclude = head_postprocess_nodes(model)
    images = read_image_list(args.calib)[: args.calib_images]
    data = subset_yaml(args.data, args.images, args.out)
    cache = args.out / f"calibration_{args.method}.json"
    results_path = args.out / "sensitivity.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    def score(name: str, exclude: list[str]) -> float:
        if name not in results:
            path = args.out / "models" / f"{name}.onnx"
            quantize(args.model, images, path, args.method, sorted(set(exclude)), cache=cache)
            report = evaluate(path, data, device=args.device, out=args.out / "val")
            results[name] = {
                "map50_95": report["map50_95"],
                "map50": report["map50"],
                "excluded": len(exclude),
            }
            results_path.write_text(json.dumps(results, indent=2))
            path.unlink()  # keep the numbers, not 24 models
        log.info("%s: mAP50-95 %.4f", name, results[name]["map50_95"])
        return results[name]["map50_95"]

    if "float" not in results:
        report = evaluate(args.model, data, device=args.device, out=args.out / "val")
        results["float"] = {"map50_95": report["map50_95"], "map50": report["map50"], "excluded": None}
    base = score("int8", base_exclude)
    parts = groups(model)
    gains = {
        g: score(f"float_{g.replace('/', '-')}", base_exclude + nodes) - base for g, nodes in parts.items()
    }
    ranking = sorted(gains, key=gains.get, reverse=True)
    for k in args.keep_top:
        kept = ranking[:k]
        exclude = base_exclude + [n for g in kept for n in parts[g]]
        score(f"top{k}", exclude)
        results[f"top{k}"]["groups"] = kept
    results["ranking"] = [{"group": g, "gain": gains[g], "nodes": len(parts[g])} for g in ranking]
    results["images"], results["calibration_images"] = args.images, len(images)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"float {results['float']['map50_95']:.4f}, int8 {base:.4f}")
    for row in results["ranking"][:8]:
        print(f"  keep {row['group']:>16} in float: {row['gain']:+.4f}")


if __name__ == "__main__":
    main()
