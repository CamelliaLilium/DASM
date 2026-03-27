#!/usr/bin/env python3
"""
Data probe script for dataset_small PKL files.

Reads PKL files and reports:
- Tuple format and length
- Train/test sample counts
- Per-domain AND per-class sample counts
- Estimated batch coverage at batch sizes 256, 512, 1024

Usage:
    python probe_dataset_small.py --data_root dataset_small --dataset_ids QIM+PMS+LSB+AHCM_0.1_1s QIM+PMS+LSB+AHCM_0.5_1s
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np


def load_dataset(data_root, dataset_id):
    """
    Load PKL file and return tuple data.
    
    Args:
        data_root: Root directory containing PKL files
        dataset_id: Dataset identifier (e.g., 'QIM+PMS+LSB+AHCM_0.1_1s')
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test)
               or (x_train, y_train, x_test, y_test, None, None) if no algo labels
    
    Raises:
        FileNotFoundError: If PKL file not found
        ValueError: If unsupported tuple format
    """
    pkl_file = os.path.join(data_root, f"{dataset_id}.pkl")
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"PKL file not found: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Support 6-tuple unified format or legacy formats
    if isinstance(data, tuple) and len(data) == 6:
        x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test = data
        return x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test
    elif isinstance(data, tuple) and len(data) == 4:
        x_train, y_train, x_test, y_test = data
        return x_train, y_train, x_test, y_test, None, None
    elif isinstance(data, tuple) and len(data) == 3:
        # Legacy format: (features, labels, algorithm_labels)
        x, y, algo = data
        return x, y, x, y, algo, algo
    else:
        raise ValueError(f"Unsupported PKL format (tuple length: {len(data) if isinstance(data, tuple) else 'not a tuple'})")


def analyze_dataset(x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test):
    """
    Analyze dataset structure and content.
    
    Returns:
        dict: Analysis results
    """
    # Convert to numpy arrays for analysis
    x_train_np = np.asarray(x_train)
    y_train_np = np.asarray(y_train)
    x_test_np = np.asarray(x_test)
    y_test_np = np.asarray(y_test)
    
    # Tuple format info
    tuple_length = 6 if algo_labels_train is not None else 4
    
    # Sample counts
    train_count = len(x_train_np)
    test_count = len(x_test_np)
    total_train = train_count
    
    # Per-domain and per-class counts
    per_domain_class_counts = {}
    
    if algo_labels_train is not None:
        algo_labels_train_np = np.asarray(algo_labels_train)
        
        # Domain names mapping
        domain_names = {0: "QIM", 1: "PMS", 2: "LSB", 3: "AHCM"}
        
        for domain_id in range(4):
            domain_mask = algo_labels_train_np == domain_id
            domain_name = domain_names.get(domain_id, f"domain_{domain_id}")
            
            # Get y values for this domain
            y_domain = y_train_np[domain_mask]
            
            # Count per class (assuming binary: 0=cover, 1=stego)
            # y might be 2D (cover_label, stego_label) or 1D
            if len(y_domain.shape) > 1 and y_domain.shape[1] >= 2:
                # 2D case: columns are [cover_label, stego_label]
                cover_counts = np.bincount(y_domain[:, 0].astype(int), minlength=2)
                stego_counts = np.bincount(y_domain[:, 1].astype(int), minlength=2)
                per_domain_class_counts[domain_name] = {
                    "total": int(np.sum(domain_mask)),
                    "cover_class_0": int(cover_counts[0]) if len(cover_counts) > 0 else 0,
                    "cover_class_1": int(cover_counts[1]) if len(cover_counts) > 1 else 0,
                    "stego_class_0": int(stego_counts[0]) if len(stego_counts) > 0 else 0,
                    "stego_class_1": int(stego_counts[1]) if len(stego_counts) > 1 else 0,
                }
            else:
                # 1D case: single label column
                class_counts = np.bincount(y_domain.astype(int), minlength=2)
                per_domain_class_counts[domain_name] = {
                    "total": int(np.sum(domain_mask)),
                    "class_0": int(class_counts[0]) if len(class_counts) > 0 else 0,
                    "class_1": int(class_counts[1]) if len(class_counts) > 1 else 0,
                }
    else:
        # No algo labels, just count by class
        if len(y_train_np.shape) > 1 and y_train_np.shape[1] >= 2:
            cover_counts = np.bincount(y_train_np[:, 0].astype(int), minlength=2)
            stego_counts = np.bincount(y_train_np[:, 1].astype(int), minlength=2)
            per_domain_class_counts["all"] = {
                "total": train_count,
                "cover_class_0": int(cover_counts[0]) if len(cover_counts) > 0 else 0,
                "cover_class_1": int(cover_counts[1]) if len(cover_counts) > 1 else 0,
                "stego_class_0": int(stego_counts[0]) if len(stego_counts) > 0 else 0,
                "stego_class_1": int(stego_counts[1]) if len(stego_counts) > 1 else 0,
            }
        else:
            class_counts = np.bincount(y_train_np.astype(int), minlength=2)
            per_domain_class_counts["all"] = {
                "total": train_count,
                "class_0": int(class_counts[0]) if len(class_counts) > 0 else 0,
                "class_1": int(class_counts[1]) if len(class_counts) > 1 else 0,
            }
    
    # Batch coverage estimates
    batch_coverage_estimates = {}
    for batch_size in [256, 512, 1024]:
        # Find smallest domain
        if algo_labels_train is not None:
            algo_labels_train_np = np.asarray(algo_labels_train)
            domain_counts = {}
            for domain_id in range(4):
                domain_mask = algo_labels_train_np == domain_id
                domain_counts[domain_id] = np.sum(domain_mask)
            
            smallest_domain_count = min(domain_counts.values())
            smallest_domain_id = min(domain_counts, key=domain_counts.get)
            smallest_domain_name = {0: "QIM", 1: "PMS", 2: "LSB", 3: "AHCM"}.get(smallest_domain_id, f"domain_{smallest_domain_id}")
            
            # Expected count per batch = (domain_count / total_train) * batch_size
            expected_per_batch = (smallest_domain_count / total_train) * batch_size
            
            batch_coverage_estimates[f"batch_{batch_size}"] = {
                "smallest_domain": smallest_domain_name,
                "smallest_domain_count": int(smallest_domain_count),
                "expected_samples_per_batch": round(expected_per_batch, 2),
                "coverage_ratio": round(expected_per_batch / smallest_domain_count, 4) if smallest_domain_count > 0 else 0.0,
            }
        else:
            # No domain info, estimate based on total
            batch_coverage_estimates[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "coverage_ratio": round(batch_size / total_train, 4),
            }
    
    return {
        "tuple_length": tuple_length,
        "train_count": train_count,
        "test_count": test_count,
        "per_domain_class_counts": per_domain_class_counts,
        "batch_coverage_estimates": batch_coverage_estimates,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe dataset_small PKL files and report structure/statistics"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset_small",
        help="Root directory containing PKL files (default: dataset_small)"
    )
    parser.add_argument(
        "--dataset_ids",
        type=str,
        nargs="+",
        required=True,
        help="Dataset IDs to probe (e.g., QIM+PMS+LSB+AHCM_0.1_1s QIM+PMS+LSB+AHCM_0.5_1s)"
    )
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create evidence directory if it doesn't exist
    evidence_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".sisyphus", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    
    # Process each dataset
    all_results = {}
    transcript_lines = []
    
    for dataset_id in args.dataset_ids:
        print(f"\n{'='*70}")
        print(f"Probing dataset: {dataset_id}")
        print(f"{'='*70}")
        transcript_lines.append(f"\n{'='*70}")
        transcript_lines.append(f"Probing dataset: {dataset_id}")
        transcript_lines.append(f"{'='*70}")
        
        try:
            # Load dataset
            x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test = load_dataset(
                args.data_root, dataset_id
            )
            
            # Analyze
            analysis = analyze_dataset(x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test)
            all_results[dataset_id] = analysis
            
            # Print results
            print(f"Tuple format: {analysis['tuple_length']}-tuple")
            print(f"Train samples: {analysis['train_count']}")
            print(f"Test samples: {analysis['test_count']}")
            print(f"\nPer-domain/class counts:")
            for domain, counts in analysis['per_domain_class_counts'].items():
                print(f"  {domain}: {counts}")
            print(f"\nBatch coverage estimates:")
            for batch_key, coverage in analysis['batch_coverage_estimates'].items():
                print(f"  {batch_key}: {coverage}")
            
            # Add to transcript
            transcript_lines.append(f"Tuple format: {analysis['tuple_length']}-tuple")
            transcript_lines.append(f"Train samples: {analysis['train_count']}")
            transcript_lines.append(f"Test samples: {analysis['test_count']}")
            transcript_lines.append(f"\nPer-domain/class counts:")
            for domain, counts in analysis['per_domain_class_counts'].items():
                transcript_lines.append(f"  {domain}: {counts}")
            transcript_lines.append(f"\nBatch coverage estimates:")
            for batch_key, coverage in analysis['batch_coverage_estimates'].items():
                transcript_lines.append(f"  {batch_key}: {coverage}")
            
            # Save JSON result
            # Extract ER from dataset_id (e.g., "QIM+PMS+LSB+AHCM_0.1_1s" -> "0.1")
            parts = dataset_id.split("_")
            if len(parts) >= 2:
                er = parts[-2]  # Second to last part before "1s"
            else:
                er = "unknown"
            
            json_file = os.path.join(results_dir, f"dataset_probe_{er}.json")
            with open(json_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nSaved JSON to: {json_file}")
            transcript_lines.append(f"\nSaved JSON to: {json_file}")
            
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            transcript_lines.append(f"ERROR: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            transcript_lines.append(f"ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            transcript_lines.append(f"ERROR: Unexpected error: {e}")
            sys.exit(1)
    
    # Save transcript
    transcript_file = os.path.join(evidence_dir, "task-1-probe-valid.txt")
    with open(transcript_file, 'w') as f:
        f.write("\n".join(transcript_lines))
    print(f"\n\nTranscript saved to: {transcript_file}")
    
    print(f"\n{'='*70}")
    print("Probe completed successfully!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
