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
