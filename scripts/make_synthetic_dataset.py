#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


def make_example(seed):
    rng = np.random.default_rng(seed)
    grid = 32
    volume = np.zeros((grid, grid, grid), dtype=np.float32)
    volume[10:22, 12:20, 13:19] = 1.0
    surface = rng.uniform(-0.45, 0.45, size=(3, 1000)).astype(np.float32)
    closest = np.zeros((grid, grid, grid, 3), dtype=np.float32)
    vertices = np.array(
        [
            [-0.2, -0.2, -0.2],
            [0.2, -0.2, -0.2],
            [0.0, 0.2, -0.2],
            [0.0, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [1, 2, 3],
            [1, 2, 4],
            [2, 3, 4],
            [1, 3, 4],
        ],
        dtype=np.int32,
    )
    return {
        "Volume": volume,
        "surfaceSamples": surface,
        "vertices": vertices,
        "faces": faces,
        "axisangle": np.array([[1, 0, 0, 0]], dtype=np.float32),
        "closestPoints": closest,
    }


def main():
    parser = argparse.ArgumentParser(description="Create a fake PRS-Net .mat dataset for Python smoke tests only.")
    parser.add_argument("--dataroot", default="datasets/shapenet", help="Output directory containing train/ and test/.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing synthetic files.")
    args = parser.parse_args()

    root = Path(args.dataroot)
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        (train_dir / "synthetic_a1.mat", 1),
        (test_dir / "synthetic.mat", 2),
    ]
    for path, seed in outputs:
        if path.exists() and not args.overwrite:
            raise SystemExit("%s already exists; pass --overwrite to replace it" % path)
        sio.savemat(path, make_example(seed))
        print("[make_synthetic_dataset] wrote %s" % path)

    print("[make_synthetic_dataset] This data is fake. Use it only to test train.py/test.py wiring.")


if __name__ == "__main__":
    main()
