"""Quantization, grouping, data helpers and the summary, on tiny synthetic models (no downloads)."""

import json

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from edge_opt.data import calibration_subset
from edge_opt.latency import measure
from edge_opt.quantize import groups, head_postprocess_nodes, preprocess, quantize, read_image_list
from edge_opt.summary import accuracy_table, split_name

SIZE = 32


def tiny_detector(path):
    """A stand-in for YOLO26's end-to-end head: a backbone conv, a box branch scaled to pixels
    (0-640), a class branch through a sigmoid (0-1), both concatenated into one output."""
    rng = np.random.default_rng(0)

    def conv(name, cin, cout, x, y):
        w = numpy_helper.from_array(rng.normal(0, 0.3, (cout, cin, 3, 3)).astype(np.float32), f"{name}.w")
        b = numpy_helper.from_array(np.zeros(cout, np.float32), f"{name}.b")
        node = helper.make_node("Conv", [x, w.name, b.name], [y], name=name, pads=[1, 1, 1, 1])
        return node, [w, b]

    nodes, inits = [], []
    for name, cin, cout, x, y in [
        ("/model.0/conv/Conv", 3, 8, "images", "f0"),
        ("/model.1/one2one_cv2.0/one2one_cv2.0.0/conv/Conv", 8, 4, "f0", "box_raw"),
        ("/model.1/one2one_cv3.0/one2one_cv3.0.0/conv/Conv", 8, 2, "f0", "cls_raw"),
    ]:
        n, i = conv(name, cin, cout, x, y)
        nodes.append(n)
        inits += i
    inits.append(numpy_helper.from_array(np.array(640.0, np.float32), "scale"))
    nodes += [
        helper.make_node("Sigmoid", ["box_raw"], ["box01"], name="/model.1/Sigmoid_box"),
        helper.make_node("Mul", ["box01", "scale"], ["boxes"], name="/model.1/Mul_2"),
        helper.make_node("Sigmoid", ["cls_raw"], ["scores"], name="/model.1/Sigmoid"),
        helper.make_node("Concat", ["boxes", "scores"], ["output0"], name="/model.1/Concat_3", axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, SIZE, SIZE])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    model.metadata_props.add(key="end2end", value="True")
    onnx.save(model, path)
    return path


@pytest.fixture
def images(tmp_path):
    folder = tmp_path / "calib"
    folder.mkdir()
    rng = np.random.default_rng(1)
    for i in range(16):
        cv2.imwrite(str(folder / f"{i:03d}.jpg"), rng.integers(0, 255, (24, 40, 3), dtype=np.uint8))
    return read_image_list(folder)


def run(path, x):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"images": x})[0]


def test_preprocess_letterboxes_like_ultralytics(images):
    x = preprocess(images[0], SIZE)
    assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
    # 24 x 40 scaled to 19 x 32: gray rows above and below the image
    assert np.allclose(x[0, :, 0, :], 114 / 255) and np.allclose(x[0, :, -1, :], 114 / 255)


def test_head_rule_and_groups(tmp_path):
    model = onnx.load(tiny_detector(tmp_path / "tiny.onnx"))
    assert set(head_postprocess_nodes(model)) == {
        "/model.1/Sigmoid_box",
        "/model.1/Mul_2",
        "/model.1/Sigmoid",
        "/model.1/Concat_3",
    }
    assert set(groups(model)) == {"0", "1/one2one_cv2", "1/one2one_cv3"}


def test_naive_int8_erases_the_scores_and_the_head_rule_keeps_them(tmp_path, images):
    """The failure that makes naive INT8 YOLO26 score 0 mAP, in miniature."""
    path = tiny_detector(tmp_path / "tiny.onnx")
    model = onnx.load(path)
    x = preprocess(images[3], SIZE)
    reference = run(path, x)[:, 4:]
    naive = quantize(path, images, tmp_path / "naive.onnx")
    kept = quantize(path, images, tmp_path / "kept.onnx", exclude=head_postprocess_nodes(model))
    naive_scores, kept_scores = run(naive["output"], x)[:, 4:], run(kept["output"], x)[:, 4:]
    # one scale for 0-640 pixels and 0-1 probabilities: a step of ~2.5, every score rounds to 0
    assert np.abs(naive_scores).max() == 0.0
    assert np.abs(kept_scores - reference).max() < 0.05
    assert {p.key for p in onnx.load(kept["output"]).metadata_props} >= {"end2end"}


def test_calibration_subset_is_deterministic_runs_of_members():
    order = [f"{i:012d}.jpg" for i in range(1000)]
    chosen = calibration_subset(order, 100, block=25, seed=3)
    assert chosen == calibration_subset(order, 100, block=25, seed=3) and len(chosen) == 100
    starts = sorted({int(n[:12]) // 25 for n in chosen})
    assert len(starts) == 4  # four whole runs of 25 consecutive members


def test_measure_reports_percentiles():
    calls = []
    result = measure(lambda x: calls.append(1), np.zeros(1), warmup=3, runs=20)
    assert len(calls) == 23 and result["runs"] == 20
    assert 0 <= result["median_ms"] <= result["p90_ms"]


def test_names_and_accuracy_table():
    assert split_name("yolo26n_int8-cls-float-entropy_onnx") == (
        "yolo26n",
        "ONNX INT8, class branch in float (entropy)",
    )
    assert split_name("yolo26s_ft_pt")[1] == "PyTorch FP32, fine-tuned control"
    reports = [
        {"name": "yolo26n_fp32_onnx", "map50_95": 0.401, "images": 5000},
        {"name": "yolo26n_ul-int8_onnx", "map50_95": 0.350, "images": 5000},
        {"name": "yolo26n_int8-minmax_onnx", "map50_95": 0.380, "images": 500},  # subset: not in the table
    ]
    table = accuracy_table(json.loads(json.dumps(reports)))
    assert "| yolo26n | ONNX INT8, Ultralytics export | 35.0 | -5.1 |" in table
    assert "head decode" not in table
