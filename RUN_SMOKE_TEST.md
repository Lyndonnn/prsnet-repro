# Smoke Test Runbook

Goal: verify the official PRS-Net preprocessing output can flow through training and inference on one or a few ShapeNet examples.

## 1. Preprocess a Tiny Subset

If you only need to test the Python train/test wiring before MATLAB or ShapeNet access is ready, generate fake `.mat` files:

```bash
python3 scripts/make_synthetic_dataset.py --dataroot datasets/shapenet --overwrite
bash run_smoke.sh
```

This does not test MATLAB preprocessing and is not meaningful training data.

Fastest path while waiting for ShapeNet approval: use the single sample mesh bundled with the official PRS-Net repo.

```matlab
cd preprocess
addpath(genpath('C:\path\to\gptoolbox'))
run_smoke_preprocess
```

Recommended fastest path for a real ShapeNet subset is Windows MATLAB because the official repo ships Windows MEX binaries.

```matlab
cd preprocess
addpath(genpath('C:\path\to\gptoolbox'))
precomputeShapeData('C:\path\to\ShapeNetCore.v2', '..\datasets\shapenet', '.\data_split', {'02691156'}, 2, 1, 1)
```

Expected output:

```text
datasets/shapenet/train/<id>_a1.mat
datasets/shapenet/test/<id>.mat
```

## 2. Install Python Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Colab, `torch` is usually preinstalled:

```bash
pip install -r requirements.txt
```

## 3. Check Dataset Layout

```bash
python3 scripts/check_dataset.py --dataroot datasets/shapenet --phases train test --require-nonempty
```

## 4. Run Minimal Pipeline

```bash
bash run_smoke.sh
```

CPU fallback:

```bash
GPU_IDS=-1 bash run_smoke.sh
```

Custom dataset path:

```bash
DATAROOT=/content/datasets/shapenet bash run_smoke.sh
```

## Success Outputs

```text
checkpoints/prsnet_smoke/latest_net_PRSNet.pth
results/prsnet_smoke/test_latest/<id>.mat
results/prsnet_smoke/test_latest/example_prediction.json
```

If this fails before training starts, the issue is almost always preprocessing output location or `.mat` schema.
