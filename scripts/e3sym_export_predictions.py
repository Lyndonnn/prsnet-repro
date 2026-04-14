#!/usr/bin/env python3
"""Export E3Sym planes as PRS-Net-style prediction .mat files.

This script intentionally does not vendor E3Sym. It imports an external clone
of https://github.com/renwuli/e3sym and writes plane0/plane1/... fields so the
existing MATLAB official benchmark evaluator can read the predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")
    return data


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def _normalise_plane(plane):
    import numpy as np

    plane = np.asarray(plane, dtype=np.float64).reshape(4)
    n = np.linalg.norm(plane[:3])
    if n < 1e-12 or not np.all(np.isfinite(plane)):
        return None
    return plane / n


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_e3sym_root = root / "external" / "e3sym"
    parser = argparse.ArgumentParser(
        description="Run E3Sym inference and export plane predictions as .mat files."
    )
    parser.add_argument("--e3sym-root", type=Path, default=default_e3sym_root)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results_e3sym" / "official" / "test_pretrained",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--npoints", type=int, default=None)
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    e3sym_root = args.e3sym_root.resolve()
    if not e3sym_root.exists():
        raise FileNotFoundError(
            f"E3Sym root not found: {e3sym_root}. Run bash run_e3sym_test.sh first."
        )

    config_path = args.config or (e3sym_root / "configs" / "test.yaml")
    config = _load_yaml(config_path)

    eval_root = (args.eval_root or Path(config["eval_root"])).expanduser()
    if not eval_root.is_absolute():
        eval_root = e3sym_root / eval_root
    weights = (args.weights or Path(config["weights"])).expanduser()
    if not weights.is_absolute():
        weights = e3sym_root / weights

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or int(config.get("batch_size", 1))
    num_workers = args.num_workers if args.num_workers is not None else int(config.get("num_workers", 0))
    npoints = args.npoints or int(config.get("npoints", 512))
    seed = int(config.get("seed", 3407))

    old_cwd = Path.cwd()
    os.makedirs(e3sym_root / ".tmp", exist_ok=True)
    os.chdir(e3sym_root)
    sys.path.insert(0, str(e3sym_root))

    try:
        import numpy as np
        import torch
        from scipy.io import savemat

        from lib.dataset import ShapeNetEval
        from lib.model import SymmetryNet
        from lib.util import DataLoaderX

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "E3Sym requires CUDA for its custom clustering op. Run this on a CUDA machine "
                "or set up a CUDA-enabled PyTorch environment."
            )

        _set_seed(seed)
        device = torch.device(args.device)

        model = SymmetryNet(
            config["mlps"],
            config["ks"],
            config["radius"],
            int(config["rotations"]),
            float(config["thre"]),
            int(config["nsample"]),
            int(config["min_cluster_size"]),
        ).to(device)

        print(f"[e3sym_export] loading weights: {weights}")
        state = torch.load(weights, map_location=device)
        model.load_state_dict(state)
        model.eval()

        print(f"[e3sym_export] eval_root: {eval_root}")
        dataset = ShapeNetEval(str(eval_root), npoints)
        loader = DataLoaderX(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        rows = []
        preview = None
        processed = 0
        with torch.no_grad():
            for batch_index, points in enumerate(loader):
                points = points.to(device)
                cluster_plane, cluster_batch = model(points)
                cluster_plane = cluster_plane.detach().cpu().numpy()
                cluster_batch = cluster_batch.detach().cpu().numpy()

                batch_count = points.shape[0]
                for local_index in range(batch_count):
                    global_index = batch_index * batch_size + local_index
                    if args.max_shapes > 0 and global_index >= args.max_shapes:
                        break

                    shape_id = dataset.datas[global_index][0]
                    planes = cluster_plane[cluster_batch == local_index]
                    mat = {}
                    plane_values = []
                    for plane_index, plane in enumerate(planes):
                        normalised = _normalise_plane(plane)
                        if normalised is None:
                            continue
                        key = f"plane{len(plane_values)}"
                        mat[key] = normalised.reshape(1, 4)
                        plane_values.append(normalised.tolist())

                    prediction_path = output_dir / f"{shape_id}.mat"
                    savemat(prediction_path, mat)
                    rows.append(
                        {
                            "shape_index": global_index + 1,
                            "shape_id": shape_id,
                            "num_planes": len(plane_values),
                            "prediction_mat": str(prediction_path),
                        }
                    )
                    if preview is None:
                        preview = {
                            "shape_id": shape_id,
                            "num_planes": len(plane_values),
                            "planes": plane_values,
                        }
                    processed += 1

                if args.max_shapes > 0 and processed >= args.max_shapes:
                    break

        summary_csv = output_dir / "e3sym_predictions.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["shape_index", "shape_id", "num_planes", "prediction_mat"]
            )
            writer.writeheader()
            writer.writerows(rows)

        preview_path = output_dir / "example_prediction.json"
        with preview_path.open("w", encoding="utf-8") as f:
            json.dump(preview or {}, f, indent=2)

        print(f"[e3sym_export] wrote {processed} prediction .mat files to {output_dir}")
        print(f"[e3sym_export] wrote {summary_csv}")
        print(f"[e3sym_export] wrote {preview_path}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
