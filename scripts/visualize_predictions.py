#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PLANE_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]


def load_mat(path):
    return sio.loadmat(str(path), verify_compressed_data_integrity=False)


def normalize_faces(faces, num_vertices):
    faces = np.asarray(faces).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape F x 3, got %s" % (faces.shape,))
    if faces.size and faces.min() == 1:
        faces = faces - 1
    valid = np.all((faces >= 0) & (faces < num_vertices), axis=1)
    return faces[valid]


def load_planes(prediction):
    keys = sorted(
        [key for key in prediction.keys() if key.startswith("plane")],
        key=lambda key: int(key.replace("plane", "")),
    )
    planes = []
    for key in keys:
        plane = np.asarray(prediction[key], dtype=np.float64).reshape(-1, 4)[0]
        normal_norm = np.linalg.norm(plane[:3])
        if normal_norm > 1e-12:
            planes.append((key, plane))
    return planes


def plane_grid(plane, vertices):
    normal = plane[:3].astype(np.float64)
    d = float(plane[3])
    normal_norm = np.linalg.norm(normal)
    normal = normal / normal_norm
    d = d / normal_norm

    center = -d * normal
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    extent = np.ptp(vertices, axis=0).max()
    extent = max(float(extent), 1.0)
    radius = extent * 0.75
    coords = np.linspace(-radius, radius, 2)
    uu, vv = np.meshgrid(coords, coords)
    grid = center[None, None, :] + uu[:, :, None] * u + vv[:, :, None] * v
    return grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]


def set_axes_equal(ax, vertices):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def visualize_one(input_mat, prediction_mat, output_png, max_faces=5000, dpi=180):
    source = load_mat(input_mat)
    prediction = load_mat(prediction_mat)

    vertices = np.asarray(source.get("vertices", prediction.get("vertices")), dtype=np.float64)
    faces = np.asarray(source.get("faces", prediction.get("faces")))
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("%s vertices must have shape V x 3, got %s" % (input_mat, vertices.shape))
    faces = normalize_faces(faces, vertices.shape[0])

    if max_faces and len(faces) > max_faces:
        rng = np.random.default_rng(1)
        faces = faces[rng.choice(len(faces), size=max_faces, replace=False)]

    planes = load_planes(prediction)
    if not planes:
        raise ValueError("%s has no plane0/plane1/... prediction keys" % prediction_mat)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(vertices[faces], alpha=0.28, linewidths=0.1)
    mesh.set_facecolor("#b8b8b8")
    mesh.set_edgecolor("#777777")
    ax.add_collection3d(mesh)

    for idx, (key, plane) in enumerate(planes):
        xs, ys, zs = plane_grid(plane, vertices)
        color = PLANE_COLORS[idx % len(PLANE_COLORS)]
        ax.plot_surface(xs, ys, zs, color=color, alpha=0.34, linewidth=0, shade=False)
        normal = plane[:3] / np.linalg.norm(plane[:3])
        d = plane[3] / np.linalg.norm(plane[:3])
        center = -d * normal
        ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2],
                  length=0.18, color=color, normalize=True)
        ax.text(center[0], center[1], center[2], key, color=color)

    ax.set_title("%s\n%s" % (Path(input_mat).name, Path(prediction_mat).name))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_axes_equal(ax, vertices)
    ax.view_init(elev=22, azim=38)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    print("[visualize_predictions] wrote %s" % output_png)


def iter_prediction_files(results_dir, exp_name, phase, which_epoch):
    prediction_dir = Path(results_dir) / exp_name / ("%s_%s" % (phase, which_epoch))
    if not prediction_dir.is_dir():
        raise FileNotFoundError("prediction directory does not exist: %s" % prediction_dir)
    return prediction_dir, sorted(path for path in prediction_dir.glob("*.mat") if path.is_file())


def main():
    parser = argparse.ArgumentParser(description="Visualize PRS-Net predicted reflection planes on test meshes.")
    parser.add_argument("--input-mat", help="Single source .mat file from datasets/shapenet/test.")
    parser.add_argument("--prediction-mat", help="Single prediction .mat file from results/<exp>/test_latest.")
    parser.add_argument("--output", help="Single output PNG path.")
    parser.add_argument("--dataroot", default="datasets/shapenet", help="Dataset root containing test/.")
    parser.add_argument("--results-dir", default="results", help="Results root.")
    parser.add_argument("--exp-name", default="exp", help="Experiment name.")
    parser.add_argument("--phase", default="test", help="Dataset/result phase.")
    parser.add_argument("--which-epoch", default="latest", help="Result epoch label.")
    parser.add_argument("--output-dir", help="Output visualization directory. Defaults under result dir.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit batch visualization count. 0 means all.")
    parser.add_argument("--max-faces", type=int, default=5000, help="Mesh faces to draw per object. 0 means all.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    args = parser.parse_args()

    if args.input_mat or args.prediction_mat or args.output:
        if not (args.input_mat and args.prediction_mat and args.output):
            raise SystemExit("--input-mat, --prediction-mat, and --output must be provided together")
        visualize_one(Path(args.input_mat), Path(args.prediction_mat), Path(args.output), args.max_faces, args.dpi)
        return

    prediction_dir, prediction_files = iter_prediction_files(args.results_dir, args.exp_name, args.phase, args.which_epoch)
    if args.max_files > 0:
        prediction_files = prediction_files[:args.max_files]
    if not prediction_files:
        raise SystemExit("no prediction .mat files found in %s" % prediction_dir)

    output_dir = Path(args.output_dir) if args.output_dir else prediction_dir / "visualizations"
    for prediction_mat in prediction_files:
        input_mat = Path(args.dataroot) / args.phase / prediction_mat.name
        if not input_mat.is_file():
            input_mat = prediction_mat
            print("[visualize_predictions] source .mat missing; using prediction mesh: %s" % prediction_mat)
        output_png = output_dir / ("%s_planes.png" % prediction_mat.stem)
        visualize_one(input_mat, prediction_mat, output_png, args.max_faces, args.dpi)

    print("[visualize_predictions] visualized %d prediction files" % len(prediction_files))


if __name__ == "__main__":
    main()
