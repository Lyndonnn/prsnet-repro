# PRS-Net Reproduction Notes

This repo keeps the official PRS-Net training and testing code as the primary implementation. The added files are wrappers, dataset checks, path configurability, and smoke-test documentation.

For a lightweight E3Sym comparison workflow, see `E3SYM_REPRO.md`. It keeps
E3Sym in an ignored `external/e3sym/` clone, exports pretrained E3Sym
predictions as PRS-Net-style `plane*.mat` files, and reuses the official
benchmark evaluator.

## Concise Repo Diagnosis

- Training entry point: `train.py`
- Inference entry point: `test.py`
- Dataset loader: `data/sym_dataset.py`
- MATLAB preprocessing entry point: `preprocess/precomputeShapeData.m`
- Expected processed dataset:

```text
datasets/shapenet/
  train/
    <shape_id>_a1.mat
    <shape_id>_a2.mat
  test/
    <shape_id>.mat
```

Each `.mat` file must contain:

```text
Volume          32 x 32 x 32
surfaceSamples 3 x N
vertices       V x 3
faces          F x 3
axisangle      1 x 4
closestPoints  32 x 32 x 32 x 3
```

The Python loader uses `Volume`, `surfaceSamples`, and `closestPoints` for training. `test.py` also copies `vertices` and `faces` into output `.mat` files.

## Dependencies

Minimal train/test Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Historical conda environment closest to the official README:

```bash
conda env create -f environment.yml
conda activate prsnet-repro
```

MATLAB preprocessing requires:

- MATLAB
- gptoolbox on the MATLAB path, for `readOBJ` and geometry helpers
- `meshlpsampling` MEX
- `point_mesh_squared_distance` MEX

The official repo ships Windows `.mexw64` binaries and DLLs. Linux preprocessing needs rebuilt `.mexa64` versions, especially for `meshlpsampling`; the Python training step does not need MATLAB.

## Workflow A: Preprocess on Windows MATLAB, Train on Linux/Colab

This is the fastest faithful path because the official repository already includes Windows MEX binaries.

If you only need to verify the Python train/test wiring while waiting for ShapeNet or MATLAB, generate fake `.mat` files:

```bash
python3 scripts/make_synthetic_dataset.py --dataroot datasets/shapenet --overwrite
bash run_smoke.sh
```

This skips MATLAB preprocessing and is not meaningful training data.

If ShapeNet access is still pending, first run the bundled one-model smoke preprocessing path:

```matlab
cd preprocess
addpath(genpath('C:\path\to\gptoolbox'))
run_smoke_preprocess
```

That uses:

```text
preprocess/shapenet/02691156/1021a0914a7207aff927ed529ad90a11/models/model_normalized.obj
preprocess/data_split_smoke/
```

It is only for checking the pipeline, not for meaningful training.

1. Clone this repo on the Windows preprocessing machine.

```bash
git clone https://github.com/Lyndonnn/prsnet-repro.git
cd prsnet-repro
```

2. Download ShapeNetCore.v2. The category folder layout should look like:

```text
ShapeNetCore.v2/
  02691156/
    <shape_id>/
      models/model_normalized.obj
```

3. Start MATLAB from the repo root or `preprocess/`, add gptoolbox, and preprocess a tiny subset.

```matlab
cd preprocess
addpath(genpath('C:\path\to\gptoolbox'))
precomputeShapeData('C:\path\to\ShapeNetCore.v2', '..\datasets\shapenet', '.\data_split', {'02691156'}, 2, 1, 1)
```

Arguments are:

```text
precomputeShapeData(shapenet_dir, output_dir, split_dir, categories, train_limit, test_limit, num_aug_per_model)
```

For the command above, success means MATLAB writes at least:

```text
datasets/shapenet/train/<id>_a1.mat
datasets/shapenet/test/<id>.mat
```

4. Copy `datasets/shapenet` to the Linux or Colab machine. Do not commit large `.mat` files to GitHub.

5. On Linux or Colab:

```bash
git clone https://github.com/Lyndonnn/prsnet-repro.git
cd prsnet-repro
pip install -r requirements.txt
python3 scripts/check_dataset.py --dataroot datasets/shapenet --phases train test --require-nonempty
bash run_smoke.sh
```

## Workflow B: Full Linux Workflow With Rebuilt MEX Files

This is possible but more fragile than Workflow A.

1. Install MATLAB on Linux.
2. Install gptoolbox and add it with `addpath(genpath('/path/to/gptoolbox'))`.
3. Build or obtain Linux `.mexa64` binaries for:
   - `point_mesh_squared_distance`
   - `meshlpsampling`
4. Put the Linux MEX files on MATLAB's path before running preprocessing.
5. Run the same MATLAB command:

```matlab
cd preprocess
addpath(genpath('/path/to/gptoolbox'))
precomputeShapeData('/path/to/ShapeNetCore.v2', '../datasets/shapenet', './data_split', {'02691156'}, 2, 1, 1)
```

The likely blocker is `meshlpsampling`, because the official repo includes the Windows binary but not a Linux binary.

## Exact Commands

Check dataset:

```bash
python3 scripts/check_dataset.py --dataroot datasets/shapenet --phases train test --require-nonempty
```

Run the minimal smoke pipeline:

```bash
bash run_smoke.sh
```

Run training with defaults close to the official command:

```bash
bash run_train.sh
```

Run inference:

```bash
bash run_test.sh
```

Visualize predicted planes:

```bash
DATAROOT=datasets/shapenet \
EXP_NAME=airplane_101train_22test \
RESULTS_DIR=results \
MAX_FILES=22 \
PLANE_SCALE=0.35 \
PLANE_ALPHA=0.18 \
MESH_ALPHA=0.9 \
bash run_visualize.sh
```

For easier visual inspection, generate one image per plane with original
surface samples in black and reflected samples in the plane color:

```bash
DATAROOT=datasets/shapenet \
EXP_NAME=airplane_101train_22test \
RESULTS_DIR=results \
MAX_FILES=22 \
RENDER_MODE=points \
SPLIT_PLANES=1 \
SHOW_REFLECTION=1 \
PLANE_SCALE=0.25 \
PLANE_ALPHA=0.12 \
ZOOM=1.05 \
bash run_visualize.sh
```

For paper-style figures, hide axes and draw one selected plane:

```bash
DATAROOT=datasets/shapenet \
EXP_NAME=airplane_101train_22test \
RESULTS_DIR=results \
MAX_FILES=5 \
OUTPUT_DIR=results/airplane_101train_22test/test_latest/figures_paper \
RENDER_MODE=points \
PAPER_STYLE=1 \
PLANE_IDS=plane0 \
PLANE_SCALE=0.28 \
PLANE_ALPHA=0.22 \
POINT_SIZE=9 \
POINT_ALPHA=0.95 \
ZOOM=1.0 \
VIEW_ELEV=18 \
VIEW_AZIM=125 \
bash run_visualize.sh
```

Evaluate approximate SDE and a PCA baseline:

```bash
DATAROOT=datasets/shapenet \
EXP_NAME=airplane_101train_22test \
RESULTS_DIR=results \
MAX_FILES=22 \
INCLUDE_PCA=1 \
bash run_evaluate_sde.sh
```

Run a controlled tiny subset:

```bash
DATAROOT=datasets/shapenet \
EXP_NAME=prsnet_smoke \
GPU_IDS=0 \
BATCH_SIZE=1 \
MAX_DATASET_SIZE=1 \
NITER=1 \
NITER_DECAY=0 \
bash run_smoke.sh
```

Use CPU if needed:

```bash
GPU_IDS=-1 bash run_smoke.sh
```

## Configurable Paths

All wrappers can be configured by environment variables:

```text
DATAROOT          default datasets/shapenet
PYTHON            default python3
EXP_NAME          default exp, smoke default prsnet_smoke
GPU_IDS           default 0, use -1 for CPU
CHECKPOINTS_DIR   default checkpoints
RESULTS_DIR       default results
MAX_DATASET_SIZE  unset for all data, smoke default 1
BATCH_SIZE        train batch size
NITER             train epochs at initial LR
NITER_DECAY       decay epochs
NUM_PLANE         default 3
NUM_QUAT          default 3
```

The Python options also accept `--dataroot`, `--checkpoints_dir`, `--results_dir`, and `--max_dataset_size` directly.

## Outputs

Smoke training writes:

```text
checkpoints/prsnet_smoke/latest_net_PRSNet.pth
checkpoints/prsnet_smoke/loss_log.txt
```

Smoke inference writes:

```text
results/prsnet_smoke/test_latest/<shape_id>.mat
results/prsnet_smoke/test_latest/example_prediction.json
```

Visualization writes:

```text
results/<exp_name>/test_latest/visualizations/<shape_id>_planes.png
```

## Probable Failure Points, Ranked

1. Missing processed `.mat` files under `datasets/shapenet/train` and `datasets/shapenet/test`.
2. MATLAB preprocessing fails because `gptoolbox`, `meshlpsampling`, or `point_mesh_squared_distance` is not on the path.
3. Linux preprocessing fails because the official repo only ships Windows `.mexw64` binaries.
4. ShapeNet folder layout does not match `category/id/models/model_normalized.obj`.
5. The `.mat` schema is wrong, especially `surfaceSamples` not being `3 x N`.
6. CUDA is unavailable. The patched options fall back to CPU, but training will be slow.
7. `test.py` cannot find `checkpoints/<EXP_NAME>/latest_net_PRSNet.pth`; run training first or copy the checkpoint.

## GitHub / Colab Sync

After changing code locally:

```bash
git status --short
git add .
git commit -m "Add PRS-Net smoke reproduction wrappers"
git push origin main
```

In Colab or another machine:

```bash
git clone https://github.com/Lyndonnn/prsnet-repro.git
cd prsnet-repro
git pull origin main
```

Generated datasets, checkpoints, and results are ignored by git. Copy preprocessed data separately.
