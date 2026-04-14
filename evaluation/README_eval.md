# PRS-Net Exact SDE Evaluation

This folder adds a MATLAB evaluator for prediction `.mat` files written by
`test.py`. It is meant to complement `scripts/evaluate_sde.py`.

`scripts/evaluate_sde.py` computes a fast nearest-neighbor approximation.
`evaluate_predictions.m` computes point-to-mesh SDE with the same
`point_mesh_squared_distance` MEX dependency used by the official repo.

## Windows MATLAB Usage

Run from the repo root or from `evaluation/`:

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
addpath(fullfile(pwd, 'preprocess'))
addpath(fullfile(pwd, 'evaluation'))

evaluate_predictions( ...
    'D:\code\prsnet-repro\datasets\shapenet\test', ...
    'D:\code\prsnet-repro\results\airplane_1000train_201test_bs32\test_latest', ...
    'D:\code\prsnet-repro\results\airplane_1000train_201test_bs32\test_latest\exact_sde_metrics.csv', ...
    Inf, ...
    true)

summarize_exact_sde( ...
    'D:\code\prsnet-repro\results\airplane_1000train_201test_bs32\test_latest\exact_sde_metrics.csv', ...
    'D:\code\prsnet-repro\results\airplane_1000train_201test_bs32\test_latest\exact_sde_summary.csv')
```

Output columns:

```text
shape_id,method,plane_id,sde_exact,a,b,c,d,source_mat,prediction_mat
```

Lower `sde_exact` is better. Each shape gets rows for:

```text
prsnet plane0/plane1/plane2
pca pca0/pca1/pca2
```

The evaluator expects `point_mesh_squared_distance.mexw64` to be available on
the MATLAB path. The repo has one under `preprocess/`, and `evaluation.zip`
also contains one.

This evaluates the planes predicted by `test.py`. It is not the paper's
official 1000-shape GT-plane benchmark unless you also reproduce the official
`evaluation.zip` inputs (`1000.txt`, `objs/`, and `gt_planes.mat`).

## Official 1000-Shape Benchmark Path

If `evaluation.zip` has been unpacked into `evaluation_old/`, first build the
missing OBJ folder and a PRS-Net test dataset in the same coordinate system as
`gt_planes.mat`. This is intentionally identity-oriented, not randomly rotated.

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
run_official_eval_preprocess('E:\ShapeNetCore.v2\ShapeNetCore.v2')
```

Then copy or zip `datasets/shapenet_official_eval1000` to the training machine
and run inference with the checkpoint you want to audit:

```bash
DATAROOT=datasets/shapenet_official_eval1000 \
EXP_NAME=multi21_50train_10test_aug4 \
GPU_IDS=0 \
WHICH_EPOCH=200 \
MAX_DATASET_SIZE=999999 \
bash run_test.sh
```

Bring the result folder back next to the repo on Windows if needed, then run:

```matlab
cd('D:\code\prsnet-repro')
addpath(genpath('D:\tools\gptoolbox'))
addpath(fullfile(pwd, 'evaluation_old'))
addpath(fullfile(pwd, 'evaluation'))

evaluate_official_benchmark( ...
    fullfile(pwd, '1000.txt'), ...
    fullfile(pwd, 'evaluation_old', 'gt_planes.mat'), ...
    fullfile(pwd, 'results', 'multi21_50train_10test_aug4', 'test_200'), ...
    fullfile(pwd, 'evaluation_old', 'objs'), ...
    fullfile(pwd, 'results', 'multi21_50train_10test_aug4', 'test_200', 'official_metrics.csv'), ...
    fullfile(pwd, 'results', 'multi21_50train_10test_aug4', 'test_200', 'official_summary.csv'), ...
    0.0004, ...
    pi / 6, ...
    true)
```

The evaluator writes rows for:

```text
gt
prsnet_raw
prsnet_filtered
pca
```

`prsnet_filtered` applies the inference-stage checks described in later PRS-Net
comparisons: remove duplicate planes with dihedral angle below `pi / 6`, keeping
the lower-SDE plane, then drop planes whose SDE is above `0.0004`.
