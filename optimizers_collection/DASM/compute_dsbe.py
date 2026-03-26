#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def softmax_entropy(values, tau):
    if not values:
        return None
    m = max(values)
    exps = [math.exp((v - m) / tau) for v in values]
    s = sum(exps)
    if s == 0:
        return None
    probs = [e / s for e in exps]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def main():
    parser = argparse.ArgumentParser(
        description="Compute DSBE from DASM train_logs JSON (domain_sharpness)."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to train_logs_*.json",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Temperature for Gibbs distribution (default: 0.1)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="If set, compute DSBE only for this 1-based epoch.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path. If not set, write next to JSON.",
    )
    args = parser.parse_args()

    log_path = Path(args.json)
    with log_path.open() as f:
        logs = json.load(f)

    domain_sharpness = logs.get("domain_sharpness", [])
    if not domain_sharpness:
        raise ValueError("domain_sharpness is missing or empty in the JSON.")

    rows = []
    if args.epoch is not None:
        idx = args.epoch - 1
        if idx < 0 or idx >= len(domain_sharpness):
            raise IndexError("epoch is out of range for domain_sharpness.")
        d = domain_sharpness[idx]
        dsbe = softmax_entropy(list(d.values()), args.tau) if d else None
        rows.append({"epoch": args.epoch, "dsbe": dsbe})
    else:
        for i, d in enumerate(domain_sharpness, start=1):
            dsbe = softmax_entropy(list(d.values()), args.tau) if d else None
            rows.append({"epoch": i, "dsbe": dsbe})

    out_path = Path(args.out) if args.out else log_path.with_suffix(".dsbe.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "dsbe"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote DSBE results to: {out_path}")


if __name__ == "__main__":
    main()
