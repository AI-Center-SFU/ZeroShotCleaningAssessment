from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=Path("scripts/outputs/tmatch/tmatch_keypoint_counts.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.counts_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    positive, negative = read_match_counts(args.counts_csv)
    threshold_rows = evaluate_thresholds(positive, negative)
    best_rows = select_best_thresholds(threshold_rows)
    best = choose_representative_threshold(best_rows)

    summary = build_summary(args.counts_csv, positive, negative, best_rows, best)

    write_thresholds(output_dir / "tmatch_threshold_errors.csv", threshold_rows)
    (output_dir / "tmatch_threshold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_distributions(
        output_dir / "tmatch_distributions.png",
        positive=positive,
        negative=negative,
        threshold=best["threshold"],
    )

    print_summary(summary)
    print(f"Saved thresholds: {output_dir / 'tmatch_threshold_errors.csv'}")
    print(f"Saved summary: {output_dir / 'tmatch_threshold_summary.json'}")
    print(f"Saved plot: {output_dir / 'tmatch_distributions.png'}")


def read_match_counts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positive = []
    negative = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            positive.append(int(row["positive_matches"]))
            negative.append(int(row["negative_matches"]))
    if not positive or not negative:
        raise ValueError(f"No match counts found in {path}")
    return np.array(positive, dtype=int), np.array(negative, dtype=int)


def evaluate_thresholds(positive: np.ndarray, negative: np.ndarray) -> list[dict[str, int | float]]:
    max_matches = int(max(positive.max(), negative.max()))
    rows = []
    for threshold in range(-1, max_matches + 1):
        fn = int(np.sum(positive <= threshold))
        fp = int(np.sum(negative > threshold))
        tn = int(np.sum(negative <= threshold))
        tp = int(np.sum(positive > threshold))
        errors = fn + fp
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        rows.append(
            {
                "threshold": threshold,
                "fn": fn,
                "fp": fp,
                "tp": tp,
                "tn": tn,
                "errors": errors,
                "accuracy": (tp + tn) / (len(positive) + len(negative)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fn_rate": fn / len(positive),
                "fp_rate": fp / len(negative),
            },
        )
    return rows


def select_best_thresholds(rows: list[dict[str, int | float]]) -> list[dict[str, int | float]]:
    min_errors = min(int(row["errors"]) for row in rows)
    return [row for row in rows if row["errors"] == min_errors]


def select_best_f1_thresholds(rows: list[dict[str, int | float]]) -> list[dict[str, int | float]]:
    max_f1 = max(float(row["f1"]) for row in rows)
    return [row for row in rows if row["f1"] == max_f1]


def choose_representative_threshold(
    best_rows: list[dict[str, int | float]],
) -> dict[str, int | float]:
    middle_index = len(best_rows) // 2
    return best_rows[middle_index]


def build_summary(
    counts_csv: Path,
    positive: np.ndarray,
    negative: np.ndarray,
    best_rows: list[dict[str, int | float]],
    best: dict[str, int | float],
) -> dict:
    best_f1_rows = select_best_f1_thresholds(evaluate_thresholds(positive, negative))
    best_f1 = choose_representative_threshold(best_f1_rows)
    return {
        "counts_csv": str(counts_csv),
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
        "rule": "same_scene if matches > Tmatch",
        "best_threshold": int(best["threshold"]),
        "best_threshold_min": int(best_rows[0]["threshold"]),
        "best_threshold_max": int(best_rows[-1]["threshold"]),
        "min_errors": int(best["errors"]),
        "fn": int(best["fn"]),
        "fp": int(best["fp"]),
        "tp": int(best["tp"]),
        "tn": int(best["tn"]),
        "accuracy": float(best["accuracy"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "fn_rate": float(best["fn_rate"]),
        "fp_rate": float(best["fp_rate"]),
        "best_f1_threshold": int(best_f1["threshold"]),
        "best_f1_threshold_min": int(best_f1_rows[0]["threshold"]),
        "best_f1_threshold_max": int(best_f1_rows[-1]["threshold"]),
        "best_f1": float(best_f1["f1"]),
        "best_f1_precision": float(best_f1["precision"]),
        "best_f1_recall": float(best_f1["recall"]),
        "best_f1_fn": int(best_f1["fn"]),
        "best_f1_fp": int(best_f1["fp"]),
        "best_f1_errors": int(best_f1["errors"]),
        "positive": describe(positive),
        "negative": describe(negative),
    }


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "min": int(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": int(values.max()),
    }


def write_thresholds(path: Path, rows: list[dict[str, int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def plot_distributions(
    path: Path,
    positive: np.ndarray,
    negative: np.ndarray,
    threshold: int,
) -> None:
    max_value = int(max(positive.max(), negative.max()))
    bins = np.linspace(0, max_value, 80)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        },
    )
    plt.figure(figsize=(11, 6))
    plt.hist(
        negative,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#d95f02",
        label="Разные сцены",
    )
    plt.hist(
        positive,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#1b9e77",
        label="Одна сцена",
    )
    plot_kde(negative, color="#d95f02")
    plot_kde(positive, color="#1b9e77")
    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Tmatch = {threshold}",
    )
    plt.xlabel("Число совпадающих ключевых точек M(Ia, Ib)")
    plt.ylabel("Плотность")
    plt.title("Распределения числа совпадающих ключевых точек")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_kde(values: np.ndarray, color: str) -> None:
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return

    if len(np.unique(values)) < 2:
        return
    xs = np.linspace(0, values.max(), 400)
    ys = gaussian_kde(values)(xs)
    plt.plot(xs, ys, color=color, linewidth=2)


def print_summary(summary: dict) -> None:
    print(f"Samples: positive={summary['positive_count']} negative={summary['negative_count']}")
    print(
        "Best Tmatch: "
        f"{summary['best_threshold']} "
        f"(optimal range {summary['best_threshold_min']}..{summary['best_threshold_max']})",
    )
    print(
        f"Errors: {summary['min_errors']} "
        f"FN={summary['fn']} FP={summary['fp']} "
        f"accuracy={summary['accuracy']:.4f} "
        f"F1={summary['f1']:.4f}",
    )
    print(
        "Best F1 threshold: "
        f"{summary['best_f1_threshold']} "
        f"(range {summary['best_f1_threshold_min']}..{summary['best_f1_threshold_max']}) "
        f"F1={summary['best_f1']:.4f} "
        f"precision={summary['best_f1_precision']:.4f} "
        f"recall={summary['best_f1_recall']:.4f}",
    )


if __name__ == "__main__":
    main()
