"""
Extract best metrics from a training log JSON file.

CLI:
  python -m gpr.utils.extract_best_metrics --json /path/to/train_log.json

Default JSON path matches the provided Transformer training log. The script
writes result.csv into the same directory as the JSON file, overwriting if it
already exists.
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_epoch_acc(log_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Return (epoch_1_based, acc).
    Tie-breaking: first occurrence of the max value (earliest epoch).
    """
    accs: List[float] = log_data.get("epoch_acc", [])
    if not accs:
        raise ValueError("epoch_acc not found or empty in the log JSON.")

    best_idx, best_val = max(enumerate(accs), key=lambda item: item[1])
    return best_idx + 1, best_val  # 1-based epoch


def best_val_acc(log_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Return (epoch_1_based, acc) for the best validation accuracy.
    Tie-breaking: first occurrence of the max value (earliest epoch).
    """
    val_accs: List[float] = log_data.get("val_acc", [])
    if not val_accs:
        raise ValueError("val_acc not found or empty in the log JSON.")

    best_idx, best_val = max(enumerate(val_accs), key=lambda item: item[1])
    return best_idx + 1, best_val  # 1-based epoch


def best_domain_test_acc(log_data: Dict[str, Any]) -> Dict[str, Tuple[int, float]]:
    """
    For each domain, return the best (epoch_1_based, acc).
    domain_test_acc is expected to be a list of dicts per epoch.
    """
    domain_list: List[Dict[str, float]] = log_data.get("domain_test_acc", [])
    best: Dict[str, Tuple[int, float]] = {}

    for epoch_idx, entry in enumerate(domain_list):
        if not isinstance(entry, dict) or not entry:
            continue
        for domain, acc in entry.items():
            if domain not in best or acc > best[domain][1]:
                best[domain] = (epoch_idx + 1, acc)
            # On ties, keep the earliest (do nothing)
    return best


def write_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["metric_type", "domain", "epoch", "value"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract best metrics from a training log JSON file."
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        required=True,
        help="Path to the training log JSON file.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    log_path = args.json_path

    try:
        data = load_json(log_path)
        best_epoch, best_acc = best_epoch_acc(data)
        best_val_epoch, best_val_acc_value = best_val_acc(data)
        domain_best = best_domain_test_acc(data)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = [
        {
            "metric_type": "epoch_acc_best",
            "domain": "NA",
            "epoch": best_epoch,
            "value": best_acc,
        },
        {
            "metric_type": "val_acc_best",
            "domain": "NA",
            "epoch": best_val_epoch,
            "value": best_val_acc_value,
        }
    ]

    for domain in sorted(domain_best.keys()):
        epoch_idx, acc = domain_best[domain]
        rows.append(
            {
                "metric_type": "domain_test_acc_best",
                "domain": domain,
                "epoch": epoch_idx,
                "value": acc,
            }
        )

    # 计算所有domain最高测试精度的平均值
    if domain_best:
        best_accs = [acc for _, acc in domain_best.values()]
        avg_best_acc = sum(best_accs) / len(best_accs)
        rows.append(
            {
                "metric_type": "domain_test_acc_best_avg",
                "domain": "ALL",
                "epoch": "NA",
                "value": avg_best_acc,
            }
        )

    csv_path = os.path.join(os.path.dirname(log_path), "result.csv")
    try:
        write_csv(csv_path, rows)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to write CSV: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote results to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

