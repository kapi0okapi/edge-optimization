.PHONY: install dev lint format test data jobs

install:  ## laptop (macOS): ONNX Runtime CPU, OpenVINO, Core ML
	pip install -e ".[mac,dev]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

data:     ## COCO val2017 + 1024 calibration images (1 GB) and the YOLO26n/s checkpoints
	bash scripts/download_data.sh --small

jobs:     ## render all Slurm scripts into slurm/generated/
	python3 slurm/generate.py --all
