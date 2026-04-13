#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def read_split_map(split_dir):
    split_dir = Path(split_dir)
    shape_to_category = {}
    for path in sorted(split_dir.glob("*_*.txt")):
        name = path.name
        if not (name.endswith("_train.txt") or name.endswith("_test.txt")):
            continue
        category = name.split("_", 1)[0]
        with path.open() as f:
            for line in f:
                shape_id = line.strip()
                if shape_id:
                    shape_to_category[shape_id] = category
    return shape_to_category


def base_shape_id(shape_id):
    return re.sub(r"_a\d+$", "", shape_id)


def read_rows(metrics_path):
    with Path(metrics_path).open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "sde_exact" in fieldnames:
        metric_name = "sde_exact"
    elif "sde_nn" in fieldnames:
        metric_name = "sde_nn"
    else:
        raise SystemExit("metrics CSV must contain sde_exact or sde_nn")
    return rows, metric_name


def best_per_shape(rows, metric_name, shape_to_category):
    best = {}
    for row in rows:
        shape_id = row["shape_id"]
        method = row["method"]
        category = shape_to_category.get(base_shape_id(shape_id), "unknown")
        value = float(row[metric_name])
        key = (category, shape_id, method)
        if key not in best or value < best[key]:
            best[key] = value
    return best


def summarize_values(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    mid = n // 2
    median = values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])
    return {
        "num_shapes": n,
        "mean": sum(values) / n,
        "median": median,
        "min": values[0],
        "max": values[-1],
    }


def summarize(best, metric_name):
    values_by_category_method = {}
    values_by_shape_method = {}
    categories = set()
    for (category, shape_id, method), value in best.items():
        categories.add(category)
        values_by_category_method.setdefault((category, method), []).append(value)
        values_by_shape_method[(category, shape_id, method)] = value

    rows = []
    for category in sorted(categories):
        methods = sorted(method for cat, method in values_by_category_method if cat == category)
        for method in methods:
            stats = summarize_values(values_by_category_method[(category, method)])
            if stats is None:
                continue
            rows.append({
                "category": category,
                "method": method,
                "num_shapes": stats["num_shapes"],
                "mean_best_" + metric_name: stats["mean"],
                "median_best_" + metric_name: stats["median"],
                "min_best_" + metric_name: stats["min"],
                "max_best_" + metric_name: stats["max"],
                "prsnet_wins": "",
                "pca_wins": "",
                "comparable_shapes": "",
            })

        shape_ids = sorted(
            shape_id
            for cat, shape_id, method in values_by_shape_method
            if cat == category and method == "prsnet"
        )
        prsnet_wins = 0
        pca_wins = 0
        comparable = 0
        for shape_id in shape_ids:
            prs_key = (category, shape_id, "prsnet")
            pca_key = (category, shape_id, "pca")
            if prs_key not in values_by_shape_method or pca_key not in values_by_shape_method:
                continue
            comparable += 1
            if values_by_shape_method[prs_key] < values_by_shape_method[pca_key]:
                prsnet_wins += 1
            else:
                pca_wins += 1
        if comparable:
            rows.append({
                "category": category,
                "method": "wins",
                "num_shapes": "",
                "mean_best_" + metric_name: "",
                "median_best_" + metric_name: "",
                "min_best_" + metric_name: "",
                "max_best_" + metric_name: "",
                "prsnet_wins": prsnet_wins,
                "pca_wins": pca_wins,
                "comparable_shapes": comparable,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize PRS-Net/PCA SDE metrics by ShapeNet category.")
    parser.add_argument("--metrics", required=True, help="sde_metrics.csv or exact_sde_metrics.csv")
    parser.add_argument("--split-dir", default="preprocess/data_split", help="Official split directory.")
    parser.add_argument("--output", required=True, help="Output category summary CSV.")
    args = parser.parse_args()

    shape_to_category = read_split_map(args.split_dir)
    rows, metric_name = read_rows(args.metrics)
    best = best_per_shape(rows, metric_name, shape_to_category)
    summary_rows = summarize(best, metric_name)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "method",
        "num_shapes",
        "mean_best_" + metric_name,
        "median_best_" + metric_name,
        "min_best_" + metric_name,
        "max_best_" + metric_name,
        "prsnet_wins",
        "pca_wins",
        "comparable_shapes",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("[summarize_sde_by_category] wrote %s" % output)
    for row in summary_rows:
        if row["method"] == "wins":
            print("%s PRS-Net wins %s / %s" % (
                row["category"], row["prsnet_wins"], row["comparable_shapes"]))


if __name__ == "__main__":
    main()
