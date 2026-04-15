# PRS-Net / E3Sym Reproduction

This repository is a practical reproduction workspace for planar reflective
symmetry detection on ShapeNet. It started from the PRS-Net codebase and adds
the preprocessing, inference, evaluation, and comparison utilities used in this
reproduction.

It is not the original official PRS-Net release. The original PRS-Net code and
paper are still the baseline implementation, but this repo now also includes:

- PRS-Net training and inference wrappers.
- Multi-category ShapeNet preprocessing helpers.
- Approximate and exact SDE evaluation.
- An official-benchmark workflow using `1000.txt` and `evaluation_old`.
- An E3Sym pretrained inference wrapper for comparison.
- MATLAB evaluation scripts that write CSV summaries.

## Repository Layout

```text
train.py                         PRS-Net training entry point
test.py                          PRS-Net inference entry point
run_train.sh                     PRS-Net training wrapper
run_test.sh                      PRS-Net inference wrapper
run_evaluate_sde.sh              Fast approximate SDE wrapper
run_visualize.sh                 Prediction visualization wrapper

preprocess/
  precomputeShapeData.m          Original-style PRS-Net preprocessing
  run_multi21_aug_preprocess.m   Multi-category augmented preprocessing
  precompute_official_eval_set.m Official 1000-ID eval-set builder

evaluation/
  evaluate_predictions.m         MATLAB exact SDE for PRS-Net predictions
  evaluate_official_benchmark.m  Official benchmark evaluator
  summarize_exact_sde.m          CSV summary helper

scripts/
  evaluate_sde.py                Fast nearest-neighbor SDE and PCA baseline
  e3sym_export_predictions.py    E3Sym prediction exporter

run_e3sym_test.sh                E3Sym pretrained inference wrapper
run_e3sym_train.sh               E3Sym training pass-through wrapper
E3SYM_REPRO.md                   E3Sym-specific notes
README_repro.md                  Longer PRS-Net workflow notes
```

Large generated artifacts are intentionally not tracked:

```text
datasets/
checkpoints/
results/
results_e3sym/
external/
evaluation_old/
```

## Main Environments

The reproduction used three machines/environments:

- Windows: MATLAB, gptoolbox, ShapeNet OBJ, official evaluation.
- Colab / Linux CUDA: PRS-Net training and inference, E3Sym inference.
- macOS / VS Code: code editing and GitHub version control.

## Python Setup for PRS-Net

For PRS-Net training and inference:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Colab usually already has PyTorch. The minimal required packages are in
`requirements.txt`.

## MATLAB Setup

MATLAB is needed for ShapeNet preprocessing and exact SDE evaluation.

Add gptoolbox and this repo's MATLAB folders:

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
addpath(fullfile(pwd, 'preprocess'))
addpath(fullfile(pwd, 'evaluation'))
addpath(fullfile(pwd, 'evaluation_old'))
```

The repo includes Windows `.mexw64` binaries from the PRS-Net/evaluation
packages. Linux MATLAB preprocessing requires rebuilt `.mexa64` binaries,
especially for `meshlpsampling` and `point_mesh_squared_distance`.

## Quick Smoke Test

To check Python train/test wiring without real ShapeNet:

```bash
python3 scripts/make_synthetic_dataset.py --dataroot datasets/shapenet --overwrite
bash run_smoke.sh
```

This only checks the pipeline. It is not meaningful training data.

## PRS-Net Preprocessing

For a real ShapeNet subset, run MATLAB on Windows. Example:

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
addpath(fullfile(pwd, 'preprocess'))

precomputeShapeData( ...
    'E:\ShapeNetCore.v2\ShapeNetCore.v2', ...
    fullfile(pwd, 'datasets', 'shapenet'), ...
    fullfile(pwd, 'preprocess', 'data_split'), ...
    {'02691156'}, ...
    1000, ...
    201, ...
    1)
```

Expected `.mat` schema:

```text
Volume          32 x 32 x 32
surfaceSamples 3 x N
vertices       V x 3
faces          F x 3
axisangle      1 x 4
closestPoints  32 x 32 x 32 x 3
```

For the multi-category augmented experiment used during reproduction:

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
addpath(fullfile(pwd, 'preprocess'))
run_multi21_aug_preprocess
```

The generated dataset was named:

```text
datasets/shapenet_multi21_50train_10test_aug4
```

## PRS-Net Training

On Colab or another CUDA machine:

```bash
cd /content/prsnet-repro
pip install -r requirements.txt

DATAROOT=datasets/shapenet_multi21_50train_10test_aug4 \
EXP_NAME=multi21_50train_10test_aug4 \
GPU_IDS=0 \
NUM_PLANE=3 \
NUM_QUAT=3 \
bash run_train.sh
```

Checkpoints are written under:

```text
checkpoints/<EXP_NAME>/
```

## PRS-Net Inference

```bash
DATAROOT=datasets/shapenet_multi21_50train_10test_aug4 \
EXP_NAME=multi21_50train_10test_aug4 \
GPU_IDS=0 \
WHICH_EPOCH=200 \
MAX_DATASET_SIZE=999999 \
bash run_test.sh
```

Predictions are written under:

```text
results/<EXP_NAME>/test_<WHICH_EPOCH>/
```

Each prediction `.mat` contains:

```text
plane0, plane1, plane2
```

where each plane is `[a b c d]` for:

```text
a*x + b*y + c*z + d = 0
```

## Fast Approximate SDE

This is useful for epoch sweeps, but it is not the official exact SDE.

```bash
DATAROOT=datasets/shapenet_multi21_50train_10test_aug4 \
EXP_NAME=multi21_50train_10test_aug4 \
RESULTS_DIR=results \
WHICH_EPOCH=200 \
INCLUDE_PCA=1 \
bash run_evaluate_sde.sh
```

For the 201-shape `multi21_50train_10test_aug4` test set, observed approximate
results included:

```text
epoch 100: prsnet mean_best_sde_nn = 0.00131004, pca = 0.00053911
epoch 200: prsnet mean_best_sde_nn = 0.00122496, pca = 0.00053911
epoch 400: prsnet mean_best_sde_nn = 0.00135811, pca = 0.00053911
epoch 500: prsnet mean_best_sde_nn = 0.00143440, pca = 0.00053911
latest:    prsnet mean_best_sde_nn = 0.00143440, pca = 0.00053911
```

The small multi-category checkpoint did not match paper quality.

## Official 1000-ID Benchmark Preprocessing

Unpack the official evaluation package into:

```text
evaluation_old/
  gt_planes.mat
  evaluation.m
  uniform_sampling.m
  point_mesh_squared_distance.mexw64
  readobjfromfile.mexw64
```

Build PRS-Net inputs and copy matching OBJ files:

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
run_official_eval_preprocess('E:\ShapeNetCore.v2\ShapeNetCore.v2')
```

This writes:

```text
datasets/shapenet_official_eval1000/test/*.mat
evaluation_old/objs/*.obj
```

The preprocessing intentionally does not augment or randomly rotate the official
test inputs. `gt_planes.mat` is in the OBJ coordinate system, so predictions
must be evaluated in that same coordinate system.

In this reproduction, only 545 of 1000 IDs were available with GT-compatible
OBJ files.

## PRS-Net on the Official 545-Shape Subset

The useful PRS-Net checkpoint available during reproduction was:

```text
/content/drive/MyDrive/prsnet-repro/outputs/table_full/checkpoints/300_net_PRSNet.pth
```

Link or copy it into the path expected by `test.py`:

```bash
cd /content/prsnet-repro
mkdir -p checkpoints/table_full
ln -sf /content/drive/MyDrive/prsnet-repro/outputs/table_full/checkpoints/300_net_PRSNet.pth \
       checkpoints/table_full/300_net_PRSNet.pth
```

Run inference:

```bash
DATAROOT=datasets/shapenet_official_eval1000 \
EXP_NAME=table_full \
CHECKPOINTS_DIR=checkpoints \
RESULTS_DIR=results_official_eval \
GPU_IDS=0 \
WHICH_EPOCH=300 \
MAX_DATASET_SIZE=999999 \
bash run_test.sh
```

Predictions:

```text
results_official_eval/table_full/test_300/
```

## Official MATLAB Evaluation

Run from Windows MATLAB:

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
addpath(fullfile(pwd, 'evaluation_old'))
addpath(fullfile(pwd, 'evaluation'))

pred_dir = fullfile(pwd, 'results_official_eval', 'table_full', 'test_300');
obj_dir = fullfile(pwd, 'evaluation_old', 'objs');

evaluate_official_benchmark( ...
    fullfile(pwd, '1000.txt'), ...
    fullfile(pwd, 'evaluation_old', 'gt_planes.mat'), ...
    pred_dir, ...
    obj_dir, ...
    fullfile(pred_dir, 'official_metrics.csv'), ...
    fullfile(pred_dir, 'official_summary.csv'), ...
    0.0004, ...
    pi / 6, ...
    true, ...
    'prsnet')
```

If your Windows checkout still has the old 9-argument evaluator, remove the last
`'prsnet'` argument. The numbers are unchanged; only method names differ.

## E3Sym Pretrained Inference

The E3Sym wrapper keeps the upstream repo in `external/e3sym/`, which is ignored
by git.

On Colab:

```bash
cd /content/prsnet-repro
git pull https://github.com/Lyndonnn/prsnet-repro.git main
INSTALL_DEPS=1 MAX_SHAPES=5 NUM_WORKERS=0 bash run_e3sym_test.sh
```

If the 5-shape test succeeds, use the same 545 OBJ subset as the PRS-Net
evaluation for the final comparison.

## E3Sym on the Same 545 GT-Sanity-Valid OBJ Subset

Copy the 545 valid OBJ files to Colab, for example:

```bash
cd /content/prsnet-repro
mkdir -p evaluation_old/objs
cp -r /content/drive/MyDrive/prsnet-repro/evaluation_old/objs/. evaluation_old/objs/
find evaluation_old/objs -maxdepth 1 -name "*.obj" | wc -l
```

Run a small test:

```bash
E3SYM_EVAL_ROOT=/content/prsnet-repro/evaluation_old/objs \
E3SYM_BENCHMARK_TXT=/content/prsnet-repro/1000.txt \
OUTPUT_DIR=/content/prsnet-repro/results_e3sym/official545/test_pretrained \
MAX_SHAPES=5 \
NUM_WORKERS=0 \
bash run_e3sym_test.sh
```

Run the full available subset:

```bash
E3SYM_EVAL_ROOT=/content/prsnet-repro/evaluation_old/objs \
E3SYM_BENCHMARK_TXT=/content/prsnet-repro/1000.txt \
OUTPUT_DIR=/content/prsnet-repro/results_e3sym/official545/test_pretrained \
NUM_WORKERS=0 \
bash run_e3sym_test.sh
```

Expected count:

```bash
find results_e3sym/official545/test_pretrained -maxdepth 1 -name "*.mat" | wc -l
# 545
```

Save to Drive:

```bash
mkdir -p /content/drive/MyDrive/prsnet-repro/e3sym_official545/test_pretrained
cp -r results_e3sym/official545/test_pretrained/. \
      /content/drive/MyDrive/prsnet-repro/e3sym_official545/test_pretrained/
```

Evaluate on Windows MATLAB using the same `evaluation_old/objs`:

```matlab
pred_dir = fullfile(pwd, 'results_e3sym', 'official545', 'test_pretrained');
obj_dir = fullfile(pwd, 'evaluation_old', 'objs');

evaluate_official_benchmark( ...
    fullfile(pwd, '1000.txt'), ...
    fullfile(pwd, 'evaluation_old', 'gt_planes.mat'), ...
    pred_dir, ...
    obj_dir, ...
    fullfile(pred_dir, 'official_metrics.csv'), ...
    fullfile(pred_dir, 'official_summary.csv'), ...
    0.0004, ...
    pi / 6, ...
    true, ...
    'e3sym')
```

## Summarizing All-Plane SDE

The summary CSV reports `mean_best_sde`, which is useful but can favor PCA
because PCA always produces three planes. For paper-style all-plane comparison,
also compute:

```matlab
T = readtable(fullfile(pred_dir, 'official_metrics.csv'));

methods = unique(T.method);
for i = 1:numel(methods)
    m = methods{i};
    rows = T(strcmp(T.method, m), :);
    fprintf('%s: mean_all_sde=%.6g, median_all_sde=%.6g, n=%d\n', ...
        m, mean(rows.sde_exact), median(rows.sde_exact), height(rows));
end
```

## Reproduction Results So Far

PRS-Net `table_full` epoch 300 on the 545 GT-compatible subset:

```text
gt:
  shapes = 545, planes = 615
  mean_best_sde = 3.34202178496e-05

prsnet_raw:
  shapes = 545, planes = 1635
  mean_best_sde = 0.000571420968165

prsnet_filtered:
  shapes = 309, planes = 419
  mean_best_sde = 0.000180550268736

pca:
  shapes = 545, planes = 1635
  mean_best_sde = 0.000103440033165
```

All-plane SDE on the same subset:

```text
gt:              mean_all_sde = 3.19201e-05, median = 1.43989e-06, n = 615
pca:             mean_all_sde = 8.13001e-04, median = 2.84250e-04, n = 1635
prsnet_filtered: mean_all_sde = 1.96554e-04, median = 1.96304e-04, n = 419
prsnet_raw:      mean_all_sde = 2.22990e-03, median = 1.10921e-03, n = 1635
```

This shows the reproduced PRS-Net filtered predictions outperform PCA in
all-plane SDE on the available official subset. E3Sym should be evaluated on the
same 545 `evaluation_old/objs` subset for a direct comparison with these
numbers.

## Method Notes

`prsnet_filtered` and `e3sym_filtered` in this repo use the same audit
post-processing:

- Remove near-duplicate planes by normal angle threshold `pi / 6`.
- Drop planes whose exact SDE is above `0.0004`.

These thresholds are reproduction choices. The released PRS-Net code does not
provide a complete official post-processing script with exact thresholds.

## Original PRS-Net Citation

```bibtex
@ARTICLE{9127500,
  author={L. Gao and L.-X. Zhang and H.-Y. Meng and Y.-H. Ren and Y.-K. Lai and L. Kobbelt},
  title={PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2020},
  doi={10.1109/TVCG.2020.3003823}
}
```

## Related Repositories

- PRS-Net project page: https://geometrylearning.com/prs-net/
- E3Sym repository: https://github.com/renwuli/e3sym
