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


class ObjEvalDataset:
    """Minimal E3Sym eval dataset.

    The upstream ShapeNetEval class depends on Open3D and writes PLY caches. For
    this reproduction wrapper we only need deterministic point tensors from the
    official OBJ files, so this keeps inference dependencies smaller.
    """

    def __init__(self, root: Path, npoints: int):
        self.root = root
        self.npoints = npoints
        benchmark_txt = root.parent / "1000.txt"
        self.datas: list[tuple[str, str]] = []
        self.paths: list[Path] = []
        with benchmark_txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                shape_id, plane_num = parts[:2]
                obj_path = root / f"{shape_id}.obj"
                self.datas.append((shape_id, plane_num))
                self.paths.append(obj_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch
        from pytorch3d.io import load_obj
        from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
        from pytorch3d.structures import Meshes

        obj_path = self.paths[index]
        verts, faces, _ = load_obj(str(obj_path), load_textures=False)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        dense_count = max(10000, self.npoints * 4)
        dense_points = sample_points_from_meshes(mesh, dense_count)[0].float()
        if dense_points.shape[0] >= self.npoints:
            points, _ = sample_farthest_points(
                dense_points.unsqueeze(0), K=self.npoints, random_start_point=False
            )
            return points[0].float()

        pad_count = self.npoints - dense_points.shape[0]
        pad_indices = torch.randint(dense_points.shape[0], (pad_count,))
        return torch.cat([dense_points, dense_points[pad_indices]], dim=0).float()


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

        from lib.model import SymmetryNet
        from torch.utils.data import DataLoader

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
        dataset = ObjEvalDataset(eval_root, npoints)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

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
