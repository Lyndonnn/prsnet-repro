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


def _install_pytorch3d_fallback() -> None:
    """Provide the small pytorch3d.ops.ball_query surface E3Sym needs.

    Current Colab Python versions often cannot install PyTorch3D wheels. E3Sym
    only imports ball_query during inference, so a nearest-neighbor fallback is
    enough for this wrapper.
    """

    try:
        import pytorch3d.ops  # noqa: F401
        return
    except Exception:
        pass

    import types
    import torch

    def ball_query(points, dense_points, K=64, radius=0.15, return_nn=True):
        # points: [B, N, 3], dense_points: [B, M, 3]
        b, n, _ = points.shape
        m = dense_points.shape[1]
        k = min(int(K), int(m))
        dists = torch.cdist(points, dense_points)
        top_dists, top_idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=False)

        if k < K:
            pad = K - k
            top_idx = torch.cat([top_idx, top_idx[..., :1].expand(b, n, pad)], dim=-1)
            top_dists = torch.cat([top_dists, top_dists[..., :1].expand(b, n, pad)], dim=-1)

        invalid = top_dists > radius
        if invalid.any():
            top_idx = top_idx.clone()
            top_dists = top_dists.clone()
            top_idx[invalid] = top_idx[..., :1].expand_as(top_idx)[invalid]
            top_dists[invalid] = -1

        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        nn_points = torch.gather(
            dense_points.unsqueeze(1).expand(-1, n, -1, -1), 2, gather_idx
        )
        if return_nn:
            return top_dists, top_idx, nn_points
        return top_dists, top_idx

    pytorch3d_module = types.ModuleType("pytorch3d")
    ops_module = types.ModuleType("pytorch3d.ops")
    ops_module.ball_query = ball_query
    pytorch3d_module.ops = ops_module
    sys.modules["pytorch3d"] = pytorch3d_module
    sys.modules["pytorch3d.ops"] = ops_module


def _load_obj_mesh(path: Path):
    import numpy as np

    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                raw = line.strip().split()[1:]
                ids = []
                for token in raw:
                    vertex_id = token.split("/")[0]
                    if vertex_id:
                        ids.append(int(vertex_id) - 1)
                if len(ids) >= 3:
                    for i in range(1, len(ids) - 1):
                        faces.append([ids[0], ids[i], ids[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"Could not read mesh from {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _sample_mesh_points(path: Path, npoints: int):
    import numpy as np
    import torch

    vertices, faces = _load_obj_mesh(path)
    verts = torch.from_numpy(vertices)
    tris = torch.from_numpy(faces)
    tri_vertices = verts[tris]
    v0 = tri_vertices[:, 0, :]
    v1 = tri_vertices[:, 1, :]
    v2 = tri_vertices[:, 2, :]
    areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    if float(areas.sum()) <= 0:
        raise ValueError(f"Mesh has zero surface area: {path}")

    face_ids = torch.multinomial(areas, npoints, replacement=True)
    chosen = tri_vertices[face_ids]
    uv = torch.rand(npoints, 2)
    flip = uv.sum(dim=1) > 1
    uv[flip] = 1 - uv[flip]
    points = chosen[:, 0, :] + uv[:, :1] * (chosen[:, 1, :] - chosen[:, 0, :]) + uv[:, 1:] * (chosen[:, 2, :] - chosen[:, 0, :])
    return points.float()


class ObjEvalDataset:
    """Minimal E3Sym eval dataset.

    The upstream ShapeNetEval class depends on Open3D and writes PLY caches. For
    this reproduction wrapper we only need deterministic point tensors from the
    official OBJ files, so this keeps inference dependencies smaller.
    """

    def __init__(self, root: Path, npoints: int, benchmark_txt: Path | None = None, skip_missing: bool = True):
        self.root = root
        self.npoints = npoints
        benchmark_txt = benchmark_txt or (root.parent / "1000.txt")
        self.datas: list[tuple[str, str]] = []
        self.paths: list[Path] = []
        with benchmark_txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                shape_id, plane_num = parts[:2]
                obj_path = root / f"{shape_id}.obj"
                if skip_missing and not obj_path.exists():
                    continue
                self.datas.append((shape_id, plane_num))
                self.paths.append(obj_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return _sample_mesh_points(self.paths[index], self.npoints)


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_e3sym_root = root / "external" / "e3sym"
    parser = argparse.ArgumentParser(
        description="Run E3Sym inference and export plane predictions as .mat files."
    )
    parser.add_argument("--e3sym-root", type=Path, default=default_e3sym_root)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--benchmark-txt", type=Path, default=None)
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
    parser.add_argument("--no-skip-missing", action="store_true")
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
    benchmark_txt = args.benchmark_txt
    if benchmark_txt is not None:
        benchmark_txt = benchmark_txt.expanduser()
        if not benchmark_txt.is_absolute():
            benchmark_txt = _repo_root() / benchmark_txt

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

        _install_pytorch3d_fallback()
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
        if benchmark_txt is not None:
            print(f"[e3sym_export] benchmark_txt: {benchmark_txt}")
        dataset = ObjEvalDataset(
            eval_root,
            npoints,
            benchmark_txt=benchmark_txt,
            skip_missing=not args.no_skip_missing,
        )
        print(f"[e3sym_export] shapes with OBJ: {len(dataset)}")
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
