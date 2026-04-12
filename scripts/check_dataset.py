#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

try:
    import scipy.io as sio
except ImportError as exc:
    raise SystemExit("scipy is required to inspect PRS-Net .mat files. Run: pip install -r requirements.txt") from exc


REQUIRED_KEYS = ("Volume", "surfaceSamples", "vertices", "faces", "closestPoints")


def find_mats(phase_dir):
    return sorted(path for path in phase_dir.rglob("*.mat") if path.is_file())


def validate_mat(path, grid_size):
    errors = []
    try:
        data = sio.loadmat(str(path), verify_compressed_data_integrity=False)
    except Exception as exc:
        return ["could not load %s: %s" % (path, exc)]

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        errors.append("missing required keys: %s" % ", ".join(missing))
        return errors

    volume_shape = data["Volume"].shape
    sample_shape = data["surfaceSamples"].shape
    closest_shape = data["closestPoints"].shape

    if volume_shape != (grid_size, grid_size, grid_size):
        errors.append("Volume shape %s, expected (%d, %d, %d)" % (volume_shape, grid_size, grid_size, grid_size))
    if len(sample_shape) != 2 or sample_shape[0] != 3:
        errors.append("surfaceSamples shape %s, expected 3 x N" % (sample_shape,))
    if len(closest_shape) != 4 or closest_shape[-1] != 3:
        errors.append("closestPoints shape %s, expected %d x %d x %d x 3" %
                      (closest_shape, grid_size, grid_size, grid_size))

    print("[check_dataset]   sample=%s" % path)
    print("[check_dataset]   Volume=%s surfaceSamples=%s closestPoints=%s vertices=%s faces=%s" %
          (volume_shape, sample_shape, closest_shape, data["vertices"].shape, data["faces"].shape))
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate PRS-Net preprocessed ShapeNet .mat layout.")
    parser.add_argument("--dataroot", default="datasets/shapenet", help="Directory containing train/ and test/.")
    parser.add_argument("--phases", nargs="+", default=["train", "test"], help="Dataset phases to inspect.")
    parser.add_argument("--grid-size", type=int, default=32, help="Expected voxel grid size.")
    parser.add_argument("--sample-count", type=int, default=2, help="How many .mat files per phase to schema-check.")
    parser.add_argument("--require-nonempty", action="store_true", help="Fail if a phase has zero .mat files.")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    print("[check_dataset] dataroot=%s" % dataroot.resolve())

    failures = []
    for phase in args.phases:
        phase_dir = dataroot / phase
        print("[check_dataset] phase=%s dir=%s" % (phase, phase_dir))
        if not phase_dir.is_dir():
            failures.append("%s does not exist" % phase_dir)
            print("[check_dataset]   missing directory")
            continue

        mats = find_mats(phase_dir)
        print("[check_dataset]   .mat files=%d" % len(mats))
        for path in mats[:5]:
            print("[check_dataset]   discovered=%s" % path)
        if args.require_nonempty and not mats:
            failures.append("%s contains no .mat files" % phase_dir)
            continue

        for path in mats[:args.sample_count]:
            errors = validate_mat(path, args.grid_size)
            failures.extend("%s: %s" % (path, error) for error in errors)

    if failures:
        print("[check_dataset] FAILED")
        for failure in failures:
            print("[check_dataset]   %s" % failure)
        return 1

    print("[check_dataset] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
