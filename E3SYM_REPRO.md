# E3Sym Simple Reproduction

This repo keeps E3Sym as an external dependency instead of copying its source
code into the PRS-Net reproduction. The wrapper scripts clone
`https://github.com/renwuli/e3sym.git` into `external/e3sym/`, which is ignored
by git, then export E3Sym predictions into the same `plane0`, `plane1`, ...
`.mat` format used by the PRS-Net evaluator.

## What This Adds

- `run_e3sym_test.sh`: clone E3Sym if needed and run pretrained inference.
- `scripts/e3sym_export_predictions.py`: import E3Sym, run its model, and write
  PRS-Net-style prediction `.mat` files.
- `run_e3sym_train.sh`: optional pass-through wrapper for E3Sym training.
- `evaluation/evaluate_official_benchmark.m`: now accepts an optional method
  prefix, so E3Sym rows can be written as `e3sym_raw` and `e3sym_filtered`.

The E3Sym repository includes `dataset/test/*.obj`, `dataset/1000.txt`, and
`pretrained/model.pth`, so it can evaluate the full official 1000-shape OBJ set
without relying on the incomplete local ShapeNet subset.

## Environment

Use a separate CUDA environment for E3Sym. The official README specifies:

```bash
conda create -n e3sym python=3.7.13
conda activate e3sym
```

Then from this repo:

```bash
bash run_e3sym_test.sh
python3 -m pip install -r external/e3sym/requirements.txt "scipy>=1.6"
bash run_e3sym_test.sh
```

If you want the script to attempt dependency installation automatically:

```bash
INSTALL_DEPS=1 bash run_e3sym_test.sh
```

E3Sym uses a CUDA extension for clustering. Running inference on a CPU-only
machine is not supported by the upstream implementation.

## Pretrained Inference

Default command:

```bash
bash run_e3sym_test.sh
```

Useful overrides:

```bash
GPU_IDS=0 \
BATCH_SIZE=1 \
NUM_WORKERS=2 \
OUTPUT_DIR=results_e3sym/official/test_pretrained \
bash run_e3sym_test.sh
```

Output:

```text
results_e3sym/official/test_pretrained/
  <shape_id>.mat
  e3sym_predictions.csv
  example_prediction.json
```

Each prediction `.mat` contains:

```text
plane0, plane1, ...
```

where each plane is `[a b c d]` for `a*x + b*y + c*z + d = 0`.

## Official Benchmark Evaluation

Run this in Windows MATLAB from the repo root after E3Sym inference:

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
addpath(fullfile(pwd, 'evaluation_old'))
addpath(fullfile(pwd, 'evaluation'))

evaluate_official_benchmark( ...
    fullfile(pwd, '1000.txt'), ...
    fullfile(pwd, 'evaluation_old', 'gt_planes.mat'), ...
    fullfile(pwd, 'results_e3sym', 'official', 'test_pretrained'), ...
    fullfile(pwd, 'external', 'e3sym', 'dataset', 'test'), ...
    fullfile(pwd, 'results_e3sym', 'official', 'test_pretrained', 'official_metrics.csv'), ...
    fullfile(pwd, 'results_e3sym', 'official', 'test_pretrained', 'official_summary.csv'), ...
    0.0004, ...
    pi / 6, ...
    true, ...
    'e3sym')
```

`e3sym_raw` is the direct clustered output from E3Sym. `e3sym_filtered` applies
the same duplicate-angle and SDE threshold filtering used for PRS-Net in this
reproduction, so treat it as an audit variant rather than an upstream E3Sym
postprocess.

## Optional Training

For training, E3Sym expects ShapeNet at `external/e3sym/dataset/ShapeNetCore.v2`.
You can link it through the wrapper:

```bash
SHAPENET_DIR=/path/to/ShapeNetCore.v2 \
GPU_IDS=0 \
bash run_e3sym_train.sh
```

This uses E3Sym's original `configs/train.yaml`.
