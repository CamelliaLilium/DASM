#!/usr/bin/env python
"""
Compact dataset_small validation runner for SAM vs repaired DASM.

Supports two modes:
  - iid: All 4 domains in training and testing
  - holdout: Train on 3 domains, evaluate on 1 unseen stego domain

Usage:
  python run_dataset_small_validation.py --mode iid --ers 0.1 0.5 --methods sam dasm --dry_run
  python run_dataset_small_validation.py --mode holdout --holdout_domain PMS --ers 0.5 --methods dasm
"""

import argparse
import os
import sys
import subprocess
import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 30
DEFAULT_SEED = 42
DEFAULT_DATA_ROOT = "dataset_small"
DEFAULT_RESULT_ROOT = "experiments/w5_adgm_repair/results"

# DASM hyperparameters from run.sh:20-24
DASM_PARAMS = {
    "use_dasm": True,
    "use_contrast": True,
    "contrast_tau": 0.1,
    "rho": 0.03,
}

# SAM hyperparameters
SAM_PARAMS = {
    "use_sam": True,
    "rho": 0.03,
}

VALID_METHODS = ["sam", "dasm"]
VALID_MODES = ["iid", "holdout"]
ALL_DOMAINS = ["QIM", "PMS", "LSB", "AHCM"]


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compact dataset_small validation runner for SAM vs repaired DASM"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_MODES,
        default="iid",
        help="Validation mode: iid (all domains) or holdout (train on 3, test on 1)"
    )
    
    parser.add_argument(
        "--ers",
        type=float,
        nargs="+",
        default=[0.1, 0.5],
        help="Embedding rates to test (default: 0.1 0.5)"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["sam", "dasm"],
        help="Methods to test: sam, dasm (default: sam dasm)"
    )
    
    parser.add_argument(
        "--holdout_domain",
        type=str,
        default="PMS",
        choices=ALL_DOMAINS,
        help="Domain to hold out in holdout mode (default: PMS)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=f"Data root directory (default: {DEFAULT_DATA_ROOT})"
    )
    
    parser.add_argument(
        "--result_root",
        type=str,
        default=DEFAULT_RESULT_ROOT,
        help=f"Result root directory (default: {DEFAULT_RESULT_ROOT})"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned runs without executing them"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS})"
    )
    
    return parser.parse_args()


# ============================================================================
# Validation
# ============================================================================

def validate_args(args):
    """Validate command-line arguments."""
    
    # Validate methods
    for method in args.methods:
        if method not in VALID_METHODS:
            print(f"ERROR: Invalid method '{method}'. Valid methods: {', '.join(VALID_METHODS)}", file=sys.stderr)
            sys.exit(1)
    
    # Validate mode
    if args.mode not in VALID_MODES:
        print(f"ERROR: Invalid mode '{args.mode}'. Valid modes: {', '.join(VALID_MODES)}", file=sys.stderr)
        sys.exit(1)
    
    # Validate holdout_domain
    if args.holdout_domain not in ALL_DOMAINS:
        print(f"ERROR: Invalid holdout_domain '{args.holdout_domain}'. Valid domains: {', '.join(ALL_DOMAINS)}", file=sys.stderr)
        sys.exit(1)
    
    # Validate ERs
    for er in args.ers:
        if er <= 0 or er > 1.0:
            print(f"ERROR: Invalid embedding rate {er}. Must be in (0, 1.0]", file=sys.stderr)
            sys.exit(1)


# ============================================================================
# Run Planning
# ============================================================================

def plan_runs(args) -> List[Dict[str, Any]]:
    """
    Plan the runs based on mode, methods, and ERs.
    
    Returns:
        List of run configurations, each with:
        - method: "sam" or "dasm"
        - er: embedding rate
        - mode: "iid" or "holdout"
        - train_domains: comma-separated domain list
        - test_domains: comma-separated domain list
        - output_dir: where results go
        - dataset_id: dataset identifier
    """
    runs = []
    
    for method in args.methods:
        for er in args.ers:
            # Determine train/test domains based on mode
            if args.mode == "iid":
                train_domains = ALL_DOMAINS
                test_domains = ALL_DOMAINS
            else:  # holdout
                train_domains = [d for d in ALL_DOMAINS if d != args.holdout_domain]
                test_domains = [args.holdout_domain]
            
            # Build dataset_id
            dataset_id = f"QIM+PMS+LSB+AHCM_{er}_1s"
            
            # Build output directory
            output_dir = os.path.join(
                args.result_root,
                args.mode,
                f"{method}_er{er}_seed{args.seed}"
            )
            
            run_config = {
                "method": method,
                "er": er,
                "mode": args.mode,
                "train_domains": ",".join(train_domains),
                "test_domains": ",".join(test_domains),
                "output_dir": output_dir,
                "dataset_id": dataset_id,
            }
            
            runs.append(run_config)
    
    return runs


# ============================================================================
# Command Building
# ============================================================================

def build_command(run_config: Dict[str, Any], args) -> List[str]:
    """
    Build the command to execute for a given run configuration.
    
    Returns:
        List of command arguments suitable for subprocess.run()
    """
    method = run_config["method"]
    er = run_config["er"]
    dataset_id = run_config["dataset_id"]
    train_domains = run_config["train_domains"]
    test_domains = run_config["test_domains"]
    output_dir = run_config["output_dir"]
    
    # Determine which script to use
    if method == "sam":
        script = "model_domain_generalization_sam.py"
    elif method == "dasm":
        script = "model_dasm_DomainGap.py"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Build base command
    cmd = [
        "python",
        script,
        "--dataset_id", dataset_id,
        "--embedding_rate", str(er),
        "--steg_algorithm", "Transformer",
        "--train_domains", train_domains,
        "--test_domains", test_domains,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--data_root", args.data_root,
        "--result_path", output_dir,
    ]
    
    cmd.extend(["--device", "cpu"])
    
    # Add method-specific parameters
    if method == "sam":
        cmd.append("--use_sam")
        cmd.extend(["--rho", str(SAM_PARAMS["rho"])])
    elif method == "dasm":
        cmd.append("--use_dasm")
        cmd.append("--use_contrast")
        cmd.extend(["--contrast_tau", str(DASM_PARAMS["contrast_tau"])])
        cmd.extend(["--rho", str(DASM_PARAMS["rho"])])
    
    return cmd


# ============================================================================
# Dry Run
# ============================================================================

def print_dry_run(runs: List[Dict[str, Any]], args):
    """Print the planned runs without executing them."""
    print(f"\n[DRY RUN] Plan: {len(runs)} runs")
    print(f"Mode: {args.mode}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"ERs: {', '.join(str(er) for er in args.ers)}")
    print(f"Seed: {args.seed}")
    print(f"Data root: {args.data_root}")
    print()
    
    for i, run_config in enumerate(runs, 1):
        method = run_config["method"]
        er = run_config["er"]
        mode = run_config["mode"]
        output_dir = run_config["output_dir"]
        
        print(f"  Run {i}: method={method}, er={er}, mode={mode}")
        print(f"    Output: {output_dir}")
        
        cmd = build_command(run_config, args)
        print(f"    Command: {' '.join(cmd)}")
        print()


# ============================================================================
# Execution
# ============================================================================

def execute_runs(runs: List[Dict[str, Any]], args):
    """Execute all planned runs and collect results."""
    
    results = []
    
    for i, run_config in enumerate(runs, 1):
        method = run_config["method"]
        er = run_config["er"]
        mode = run_config["mode"]
        output_dir = run_config["output_dir"]
        
        print(f"\n{'='*80}")
        print(f"Run {i}/{len(runs)}: {method} @ ER={er} ({mode} mode)")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        cmd = build_command(run_config, args)
        
        # Print command
        print(f"Command: {' '.join(cmd)}\n")
        
        # Execute
        log_file = os.path.join(output_dir, "train.log")
        try:
            with open(log_file, "w") as log_f:
                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=3600,  # 1 hour timeout
                )
            
            status = "success" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                print(f"WARNING: Run failed with exit code {result.returncode}")
                print(f"See log: {log_file}")
        
        except subprocess.TimeoutExpired:
            status = "timeout"
            print(f"ERROR: Run timed out after 1 hour")
        except Exception as e:
            status = "error"
            print(f"ERROR: {e}")
        
        # Record result
        results.append({
            "mode": mode,
            "method": method,
            "er": er,
            "seed": args.seed,
            "output_dir": output_dir,
            "status": status,
        })
    
    return results


# ============================================================================
# Summary
# ============================================================================

def write_summary(results: List[Dict[str, Any]], args):
    """Write a summary CSV of all runs."""
    
    summary_file = os.path.join(args.result_root, "summary.csv")
    os.makedirs(args.result_root, exist_ok=True)
    
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "method", "er", "seed", "output_dir", "status"]
        )
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSummary written to: {summary_file}")
    
    # Print summary to stdout
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in results:
        status_str = "OK" if result["status"] == "success" else "FAIL"
        print(f"{status_str} {result['method']:6s} @ ER={result['er']:.1f} ({result['mode']:7s}): {result['status']}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    # Validate arguments
    validate_args(args)
    
    # Plan runs
    runs = plan_runs(args)
    
    # Dry run or execute
    if args.dry_run:
        print_dry_run(runs, args)
    else:
        results = execute_runs(runs, args)
        write_summary(results, args)


if __name__ == "__main__":
    main()
