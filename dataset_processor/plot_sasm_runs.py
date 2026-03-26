#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
汇总 sasm_train_multi 下各次实验日志，绘制对比曲线。
默认 base_dir: ./models_collection/Transformer/sasm_train_multi
"""

import os
import json
import argparse
from glob import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_run_tag(run_name: str):
    """
    run_name 形如: 20250101_123000_rho0.05_mu0.2_muMode-const_diff-pairwise
    """
    parts = run_name.split('_')
    info = {
        'rho': None,
        'mu_repr': None,
        'mu_mode': None,
        'diff_mode': None
    }
    for p in parts:
        if p.startswith('rho'):
            info['rho'] = p.replace('rho', '')
        elif p.startswith('muMode-'):
            info['mu_mode'] = p.replace('muMode-', '')
        elif p.startswith('diff-'):
            info['diff_mode'] = p.replace('diff-', '')
        elif p.startswith('mu'):
            info['mu_repr'] = p.replace('mu', '')
    return info


def load_runs(base_dir):
    runs = []
    for d in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, d)
        if not os.path.isdir(run_path):
            continue
        json_files = glob(os.path.join(run_path, "train_logs_*.json"))
        if not json_files:
            continue
        with open(json_files[0], 'r') as f:
            logs = json.load(f)
        meta = parse_run_tag(os.path.basename(run_path))
        runs.append({
            'name': os.path.basename(run_path),
            'path': run_path,
            'meta': meta,
            'logs': logs
        })
    return runs


def plot_group(runs, key, ylabel, title, save_path):
    plt.figure(figsize=(7, 4.5))
    for r in runs:
        vals = r['logs'].get(key, [])
        if not vals:
            continue
        xs = np.arange(1, len(vals) + 1)
        label = f"{r['meta'].get('diff_mode','?')}-{r['meta'].get('mu_mode','?')}-rho{r['meta'].get('rho','?')}-mu{r['meta'].get('mu_repr','?')}"
        plt.plot(xs, vals, label=label, linewidth=1.8)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default=os.environ.get('DASM_SASM_RUNS_ROOT', os.path.join(PROJECT_ROOT, 'models_collection', 'Transformer', 'sasm_train_multi')),
                        help='根目录，包含多个实验子目录')
    args = parser.parse_args()

    runs = load_runs(args.base_dir)
    if not runs:
        print(f"No runs found under {args.base_dir}")
        return

    out_dir = os.path.join(args.base_dir, 'summary_plots')
    os.makedirs(out_dir, exist_ok=True)

    # 全部对比
    plot_group(runs, 'val_acc', 'Validation Accuracy',
               'Val Acc (all runs)', os.path.join(out_dir, 'val_acc_all.png'))
    plot_group(runs, 'epoch_acc', 'Train Accuracy',
               'Train Acc (all runs)', os.path.join(out_dir, 'train_acc_all.png'))
    plot_group(runs, 'mu', 'Mu (effective)',
               'Mu Schedule (all runs)', os.path.join(out_dir, 'mu_all.png'))
    plot_group(runs, 'divergence_norm', 'Divergence Norm',
               'Divergence Norm (all runs)', os.path.join(out_dir, 'divergence_all.png'))

    # 按 diff_mode 分组输出
    grouped = defaultdict(list)
    for r in runs:
        grouped[r['meta'].get('diff_mode', 'unknown')].append(r)

    for diff, subruns in grouped.items():
        tag = diff if diff else 'unknown'
        plot_group(subruns, 'val_acc', 'Validation Accuracy',
                   f'Val Acc ({tag})', os.path.join(out_dir, f'val_acc_{tag}.png'))
        plot_group(subruns, 'mu', 'Mu (effective)',
                   f'Mu Schedule ({tag})', os.path.join(out_dir, f'mu_{tag}.png'))

    print(f"Summary plots saved to: {out_dir}")


if __name__ == '__main__':
    main()



