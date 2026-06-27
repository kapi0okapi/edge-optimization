"""COCO val2017 for accuracy, a few hundred train2017 images for INT8 calibration.

    edge-data --out data/coco                 # val2017 (0.8 GB) + 1024 calibration images
    edge-data --out data/coco --full          # plus all of train2017 (19 GB), for QAT on the GPU cluster

Accuracy is measured on all 5000 val2017 images with the official COCO evaluation, like the
numbers in the Ultralytics documentation. Calibration images come from train2017, never from
val: the 19 GB archive is read with HTTP range requests. Its members are stored in random
order, so a few runs of consecutive members are a random sample that arrives in a handful of
requests (~170 MB). Labels are the Ultralytics conversion of the official annotations.

Layout after download (Ultralytics format):

    <out>/images/val2017/*.jpg, <out>/labels/val2017/*.txt, <out>/val2017.txt
    <out>/images/train2017/*.jpg, <out>/labels/train2017/*.txt, <out>/calib2017.txt
    <out>/annotations/instances_val2017.json
    <out>/coco.yaml          train: the calibration subset (export split="train"), val: val2017
    <out>/coco_full.yaml     train: all of train2017 (with --full)
"""

from __future__ import annotations

import argparse
import io
import random
import zipfile
from pathlib import Path

import requests
import yaml

from edge_opt.remote_zip import HttpRangeFile, extract_members

LABELS_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
VAL_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
TRAIN_ZIP = "http://images.cocodataset.org/zips/train2017.zip"


def coco_names() -> dict[int, str]:
    from ultralytics.utils import ROOT

    return yaml.safe_load((ROOT / "cfg" / "datasets" / "coco.yaml").read_text())["names"]


def calibration_subset(archive_order: list[str], n: int, block: int = 64, seed: int = 0) -> list[str]:
    """``n`` names from ``archive_order`` as non-overlapping runs of ``block`` consecutive members,
    the same for every run."""
    blocks = len(archive_order) // block
    starts = sorted(random.Random(seed).sample(range(blocks), -(-n // block)))
    chosen = [name for b in starts for name in archive_order[b * block : (b + 1) * block]]
    return sorted(chosen[:n])


def train_archive_order() -> list[str]:
    archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(TRAIN_ZIP), 1 << 22))
    members = sorted((i for i in archive.infolist() if not i.is_dir()), key=lambda i: i.header_offset)
    return [Path(i.filename).name for i in members]


def write_yaml(out: Path, name: str = "coco.yaml", train: str = "calib2017.txt") -> Path:
    path = out / name
    data = {"path": str(out.resolve()), "train": train, "val": "val2017.txt", "names": coco_names()}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def fetch(url: str, target: Path, chunk: int = 1 << 22) -> None:
    """Downloads ``url`` unless a complete copy is already at ``target``."""
    expected = int(requests.head(url, timeout=60).headers["Content-Length"])
    if target.exists() and target.stat().st_size == expected:
        return
    with requests.get(url, stream=True, timeout=60) as r, target.open("wb") as f:
        r.raise_for_status()
        for block in r.iter_content(chunk):
            f.write(block)


def download(out: Path, calibration_images: int = 1024, full: bool = False) -> Path:
    """``full`` also fetches all of train2017 (19 GB) for quantization-aware training and
    writes ``coco_full.yaml``; otherwise only the calibration subset is read."""
    out.mkdir(parents=True, exist_ok=True)
    labels = zipfile.ZipFile(io.BytesIO(requests.get(LABELS_ZIP, timeout=300).content))
    chosen = calibration_subset(train_archive_order(), calibration_images)
    wanted = {"coco/val2017.txt", "coco/annotations/instances_val2017.json"}
    wanted |= {f"coco/labels/train2017/{Path(n).stem}.txt" for n in chosen}
    prefixes = ("coco/labels/val2017/", "coco/labels/train2017/") if full else ("coco/labels/val2017/",)
    for info in labels.infolist():
        if not info.is_dir() and (info.filename in wanted or info.filename.startswith(prefixes)):
            target = out / info.filename.removeprefix("coco/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(labels.read(info))
    (out / "calib2017.txt").write_text("".join(f"./images/train2017/{n}\n" for n in chosen))

    for name, url, count in (("val2017", VAL_ZIP, 5000), ("train2017", TRAIN_ZIP, 118287)):
        if name == "train2017" and not full:
            extract_members(url, out / "images", names={f"train2017/{n}" for n in chosen}, buffer_mb=4)
            continue
        if len(list((out / "images" / name).glob("*.jpg"))) < count:
            archive = out / f"{name}.zip"  # a complete copy placed here is used as is
            fetch(url, archive)
            zipfile.ZipFile(archive).extractall(out / "images")
            archive.unlink()
    if full:
        listing = labels.read("coco/train2017.txt").decode()
        (out / "train2017.txt").write_text(listing)
        write_yaml(out, "coco_full.yaml", "train2017.txt")
    return write_yaml(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/coco"))
    parser.add_argument("--calibration-images", type=int, default=1024)
    parser.add_argument("--full", action="store_true", help="all of train2017 too, for QAT (19 GB)")
    args = parser.parse_args(argv)
    print(download(args.out, args.calibration_images, args.full))


if __name__ == "__main__":
    main()
