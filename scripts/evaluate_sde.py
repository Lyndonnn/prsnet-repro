#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree


def load_mat(path):
    return sio.loadmat(str(path), verify_compressed_data_integrity=False)


def load_points(data):
    if "surfaceSamples" in data:
        points = np.asarray(data["surfaceSamples"], dtype=np.float64)
        if points.shape[0] == 3:
            points = points.T
    elif "sample" in data:
        points = np.asarray(data["sample"], dtype=np.float64)
        if points.shape[0] == 3:
            points = points.T
    else:
        raise KeyError("missing surfaceSamples/sample")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be N x 3 or 3 x N, got %s" % (points.shape,))
    return points


def load_planes(data):
    keys = sorted(
        [key for key in data.keys() if key.startswith("plane")],
        key=lambda key: int(key.replace("plane", "")),
    )
    planes = []
    for key in keys:
        plane = np.asarray(data[key], dtype=np.float64).reshape(-1, 4)[0]
        norm = np.linalg.norm(plane[:3])
        if norm > 1e-12 and np.all(np.isfinite(plane)):
            planes.append((key, plane / norm))
    return planes


def reflect_points(points, plane):
    normal = plane[:3]
    d = plane[3]
    signed_dist = np.sum(points * normal[None, :], axis=1) + d
    return points - 2.0 * signed_dist[:, None] * normal[None, :]


def nn_sde(points, plane, tree):
    reflected = reflect_points(points, plane)
    distances, _ = tree.query(reflected, k=1)
    return float(np.mean(distances ** 2))


def pca_baseline_planes(points):
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    center = points.mean(axis=0)
    planes = []
    for i, normal in enumerate(vh):
        normal = normal / np.linalg.norm(normal)
        d = -float(np.dot(normal, center))
        planes.append(("pca%d" % i, np.concatenate([normal, [d]])))
    return planes


def evaluate_file(prediction_mat, source_mat, include_pca):
    prediction = load_mat(prediction_mat)
    source = load_mat(source_mat) if source_mat.is_file() else prediction
    points = load_points(source)
    tree = cKDTree(points)

    rows = []
    for method, planes in [("prsnet", load_planes(prediction))]:
        for plane_id, plane in planes:
            rows.append({
                "shape_id": prediction_mat.stem,
                "method": method,
                "plane_id": plane_id,
                "sde_nn": nn_sde(points, plane, tree),
                "a": plane[0],
                "b": plane[1],
                "c": plane[2],
                "d": plane[3],
                "source_mat": str(source_mat),
                "prediction_mat": str(prediction_mat),
            })

    if include_pca:
        for plane_id, plane in pca_baseline_planes(points):
            rows.append({
                "shape_id": prediction_mat.stem,
                "method": "pca",
                "plane_id": plane_id,
                "sde_nn": nn_sde(points, plane, tree),
                "a": plane[0],
                "b": plane[1],
                "c": plane[2],
                "d": plane[3],
                "source_mat": str(source_mat),
                "prediction_mat": str(prediction_mat),
            })
    return rows


def summarize(rows):
    by_method = {}
    for row in rows:
        key = (row["shape_id"], row["method"])
        if key not in by_method or row["sde_nn"] < by_method[key]["sde_nn"]:
            by_method[key] = row

    summary = {}
    for row in by_method.values():
        summary.setdefault(row["method"], []).append(row["sde_nn"])

    lines = []
    for method, values in sorted(summary.items()):
        arr = np.asarray(values, dtype=np.float64)
        lines.append({
            "method": method,
            "num_shapes": len(values),
            "mean_best_sde_nn": float(arr.mean()),
            "median_best_sde_nn": float(np.median(arr)),
            "min_best_sde_nn": float(arr.min()),
            "max_best_sde_nn": float(arr.max()),
        })
    return lines


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("[evaluate_sde] wrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Approximate PRS-Net SDE with nearest-neighbor surface samples.")
    parser.add_argument("--dataroot", default="datasets/shapenet", help="Dataset root containing test/.")
    parser.add_argument("--results-dir", default="results", help="Results root.")
    parser.add_argument("--exp-name", required=True, help="Experiment name.")
    parser.add_argument("--phase", default="test", help="Dataset/result phase.")
    parser.add_argument("--which-epoch", default="latest", help="Result epoch label.")
    parser.add_argument("--output", help="Metrics CSV path.")
    parser.add_argument("--summary-output", help="Summary CSV path.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of prediction files. 0 means all.")
    parser.add_argument("--include-pca", action="store_true", help="Also evaluate PCA symmetry-plane baseline.")
    args = parser.parse_args()

    prediction_dir = Path(args.results_dir) / args.exp_name / ("%s_%s" % (args.phase, args.which_epoch))
    if not prediction_dir.is_dir():
        raise FileNotFoundError("prediction directory does not exist: %s" % prediction_dir)

    prediction_files = sorted(path for path in prediction_dir.glob("*.mat") if path.is_file())
    if args.max_files > 0:
        prediction_files = prediction_files[:args.max_files]
    if not prediction_files:
        raise SystemExit("no prediction .mat files found in %s" % prediction_dir)

    rows = []
    for prediction_mat in prediction_files:
        source_mat = Path(args.dataroot) / args.phase / prediction_mat.name
        rows.extend(evaluate_file(prediction_mat, source_mat, args.include_pca))

    output = Path(args.output) if args.output else prediction_dir / "sde_metrics.csv"
    summary_output = Path(args.summary_output) if args.summary_output else prediction_dir / "sde_summary.csv"

    write_csv(output, rows, ["shape_id", "method", "plane_id", "sde_nn", "a", "b", "c", "d", "source_mat", "prediction_mat"])
    summary_rows = summarize(rows)
    write_csv(summary_output, summary_rows,
              ["method", "num_shapes", "mean_best_sde_nn", "median_best_sde_nn", "min_best_sde_nn", "max_best_sde_nn"])

    for row in summary_rows:
        print("[evaluate_sde] %(method)s num_shapes=%(num_shapes)d mean_best_sde_nn=%(mean_best_sde_nn).8f median=%(median_best_sde_nn).8f" % row)


if __name__ == "__main__":
    main()
