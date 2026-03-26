#!/usr/bin/env python3
"""
Plot rho and contrast_tau sensitivity curves using result.csv files.
Default metric: mean of domain_test_acc_best across domains.
"""

import argparse
import csv
import glob
import os
import re
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _extract_rho(path: str) -> Optional[float]:
    match = re.search(r"rho([0-9.]+)", path)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_contrast_tau(path: str) -> Optional[float]:
    match = re.search(r"ctau([0-9.]+)", path)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _read_result_csv(path: str) -> Tuple[List[float], Optional[float]]:
    domain_best = []
    best_avg = None
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = (row.get("metric_type") or "").strip()
            value_str = (row.get("value") or "").strip()
            if not value_str:
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            if metric == "domain_test_acc_best":
                domain_best.append(value)
            elif metric == "domain_test_acc_best_avg":
                best_avg = value
    return domain_best, best_avg


def _collect_points_for_rho(
    base_dir: str,
    contrast_tau: float,
    use_avg_line: bool,
    extra_glob: str,
) -> List[Tuple[float, float, str]]:
    pattern = os.path.join(base_dir, extra_glob, "result.csv")
    paths = glob.glob(pattern)
    points = []
    ctau_tag = f"ctau{contrast_tau}"
    for path in paths:
        if ctau_tag not in path:
            continue
        rho = _extract_rho(path)
        if rho is None:
            continue
        domain_best, best_avg = _read_result_csv(path)
        if use_avg_line and best_avg is not None:
            acc = best_avg
        else:
            if not domain_best:
                continue
            acc = sum(domain_best) / len(domain_best)
        points.append((rho, acc, path))
    points.sort(key=lambda x: x[0])
    return points


def _collect_points_for_ctau(
    base_dir: str,
    fixed_rho: float,
    use_avg_line: bool,
    extra_glob: str,
) -> List[Tuple[float, float, str]]:
    pattern = os.path.join(base_dir, extra_glob, "result.csv")
    paths = glob.glob(pattern)
    points = []
    rho_tag = f"rho{fixed_rho}"
    for path in paths:
        if rho_tag not in path:
            continue
        ctau = _extract_contrast_tau(path)
        if ctau is None:
            continue
        domain_best, best_avg = _read_result_csv(path)
        if use_avg_line and best_avg is not None:
            acc = best_avg
        else:
            if not domain_best:
                continue
            acc = sum(domain_best) / len(domain_best)
        points.append((ctau, acc, path))
    points.sort(key=lambda x: x[0])
    return points


def _plot_series(ax, xs, ys, xlabel: str, title: str) -> None:
    ax.plot(
        xs,
        ys,
        "-o",
        linewidth=2.4,
        markersize=6.2,
        color="#1f77b4",
        markerfacecolor="#ffffff",
        markeredgewidth=1.6,
        label="DASM",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)
    span = max(ys) - min(ys)
    margin = max(0.002, span * 0.15)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#dddddd")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.1f}"))


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def _plot_single(
    points: List[Tuple[float, float, str]],
    output_path: str,
    xlabel: str,
    title: str,
) -> None:
    if not points:
        raise ValueError(f"No points found for {xlabel}. Check base_dir and filters.")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    _apply_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), dpi=180)
    _plot_series(ax, xs, ys, xlabel, title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rho sensitivity using result.csv files."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.environ.get("DASM_RHO_BASE_DIR", os.path.join(PROJECT_ROOT, "models_collection", "dasm_domain_gap", "Transformer", "result")),
        help="Base directory containing result.csv files.",
    )
    parser.add_argument(
        "--contrast_tau",
        type=float,
        default=0.5,
        help="Filter runs by contrast_tau (ctau).",
    )
    parser.add_argument(
        "--fixed_rho",
        type=float,
        default=0.03,
        help="Fixed rho for contrast_tau sensitivity.",
    )
    parser.add_argument(
        "--use_avg_line",
        action="store_true",
        help="Use domain_test_acc_best_avg if present.",
    )
    parser.add_argument(
        "--glob",
        dest="extra_glob",
        type=str,
        default="*",
        help="Extra glob under base_dir (e.g., 'dasm_er0.5_bs1024_*').",
    )
    parser.add_argument(
        "--output_rho",
        type=str,
        default=os.environ.get("DASM_RHO_OUTPUT", os.path.join(PROJECT_ROOT, "optimizers_collection", "DASM", "rho_sensitivity.png")),
        help="Output image path for rho sensitivity.",
    )
    parser.add_argument(
        "--output_ctau",
        type=str,
        default=os.environ.get("DASM_CTAU_OUTPUT", os.path.join(PROJECT_ROOT, "optimizers_collection", "DASM", "ctau_sensitivity.png")),
        help="Output image path for contrast_tau sensitivity.",
    )
    parser.add_argument(
        "--title_rho",
        type=str,
        default="Sensitivity to rho (contrast_tau=0.5)",
        help="Left plot title.",
    )
    parser.add_argument(
        "--title_ctau",
        type=str,
        default="Sensitivity to contrast_tau (rho=0.03)",
        help="Right plot title.",
    )
    args = parser.parse_args()

    rho_points = _collect_points_for_rho(
        base_dir=args.base_dir,
        contrast_tau=args.contrast_tau,
        use_avg_line=args.use_avg_line,
        extra_glob=args.extra_glob,
    )
    ctau_points = _collect_points_for_ctau(
        base_dir=args.base_dir,
        fixed_rho=args.fixed_rho,
        use_avg_line=args.use_avg_line,
        extra_glob=args.extra_glob,
    )
    _plot_single(rho_points, args.output_rho, "rho", args.title_rho)
    _plot_single(ctau_points, args.output_ctau, "contrast_tau", args.title_ctau)

    print(f"Saved: {args.output_rho}")
    print(f"Saved: {args.output_ctau}")
    print("Rho points:")
    for rho, acc, path in rho_points:
        print(f"  rho={rho:.4f}  acc={acc:.6f}  file={path}")
    print("Contrast_tau points:")
    for ctau, acc, path in ctau_points:
        print(f"  ctau={ctau:.4f}  acc={acc:.6f}  file={path}")


if __name__ == "__main__":
    main()
