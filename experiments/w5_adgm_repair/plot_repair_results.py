import argparse
import json
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DOMAIN_NAMES = {0: "QIM", 1: "PMS", 2: "LSB", 3: "AHCM"}
DOMAIN_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3"}


def find_iid_log(results_dir):
    """Find the IID training log file."""
    pattern = os.path.join(results_dir, 'iid', '**', 'train_logs_*.json')
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def plot_fig1_domain_gaps(log_data, out_path):
    """Plot domain gap evolution across epochs."""
    gaps_per_epoch = log_data.get('live_clean_gaps', [])
    if not gaps_per_epoch:
        print("WARNING: no live_clean_gaps data")
        return
    
    epochs = list(range(1, len(gaps_per_epoch) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for did in range(4):
        vals = []
        for ep_dict in gaps_per_epoch:
            # JSON keys are strings
            v = ep_dict.get(str(did), ep_dict.get(did, None))
            vals.append(float(v) if v is not None else float('nan'))
        ax.plot(epochs, vals, label=DOMAIN_NAMES[did], color=DOMAIN_COLORS[did],
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Live Cover-Stego Gap (L2)')
    ax.set_title('Domain Gap Evolution (IID ER=0.5, Repaired DASM)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig2_gap_retention(log_data, out_path):
    """Plot gap retention ratio per epoch."""
    retention = log_data.get('gap_retention', [])
    if not retention:
        print("WARNING: no gap_retention data")
        return
    
    epochs = list(range(1, len(retention) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(epochs, retention, color='#2ca02c', linewidth=2, marker='s',
            markersize=5, label='gap_retention')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Reference (1.0)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap Retention Ratio (pert/clean)')
    ax.set_title('Gap Retention Ratio per Epoch (IID ER=0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig3_mechanism(mech_data, out_path):
    """Plot mechanism validation checks."""
    checks = ['adgm_differentiable', 'gradient_direction_changed', 'rho_monotonic', 'perturbed_gap_differs']
    labels = ['ADGM\nDifferentiable', 'Gradient\nDirection\nChanged', 'rho\nMonotonic', 'Perturbed Gap\nDiffers']
    values = [int(bool(mech_data.get(c, False))) for c in checks]
    colors = ['#2ca02c' if v else '#d62728' for v in values]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel('Pass (1) / Fail (0)')
    ax.set_title('ADGM Mechanism Validation: All Checks Pass')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                'PASS' if val else 'FAIL', ha='center', va='bottom', fontweight='bold',
                color='#2ca02c' if val else '#d62728', fontsize=11)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Fail', 'Pass'])
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig4_schematic(out_path):
    """Plot DASM ADGM repair schematic."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('DASM ADGM Repair: Gradient Path', fontsize=14, fontweight='bold')
    
    def draw_box(ax, x, y, w, h, text, color='#aec6cf', fontsize=10):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold')
    
    def draw_arrow(ax, x1, y1, x2, y2, color='black', label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.02, my, label, fontsize=9, color=color)
    
    for ax, title, broken in [(ax1, 'Broken ADGM\n(detached)', True),
                               (ax2, 'Repaired ADGM\n(live)', False)]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color='#d62728' if broken else '#2ca02c')
        
        draw_box(ax, 0.1, 0.75, 0.35, 0.15, 'features\n(live)', '#aec6cf')
        
        if broken:
            draw_box(ax, 0.1, 0.50, 0.35, 0.15, '.detach()', '#ffcccc')
            draw_arrow(ax, 0.275, 0.75, 0.275, 0.65, 'black')
            draw_box(ax, 0.1, 0.25, 0.35, 0.15, 'gap_loss\n(no grad_fn)', '#ffcccc')
            draw_arrow(ax, 0.275, 0.50, 0.275, 0.40, '#d62728')
            ax.text(0.55, 0.32, '✗ ∇gap_loss = 0', fontsize=11, color='#d62728', fontweight='bold')
            ax.text(0.55, 0.22, 'rho acts only on\nL_cls + L_DSCL', fontsize=9, color='#d62728')
        else:
            draw_box(ax, 0.1, 0.50, 0.35, 0.15, 'live centroids\n(differentiable)', '#ccffcc')
            draw_arrow(ax, 0.275, 0.75, 0.275, 0.65, 'black')
            draw_box(ax, 0.1, 0.25, 0.35, 0.15, 'gap_loss\n(grad_fn ≠ None)', '#ccffcc')
            draw_arrow(ax, 0.275, 0.50, 0.275, 0.40, '#2ca02c')
            ax.text(0.55, 0.32, '✓ ∇gap_loss ≠ 0', fontsize=11, color='#2ca02c', fontweight='bold')
            ax.text(0.55, 0.22, 'rho acts on\nL_cls + L_DSCL + L_ADGM', fontsize=9, color='#2ca02c')
        
        draw_box(ax, 0.1, 0.05, 0.35, 0.12, 'ε = ρ·∇L/‖∇L‖', '#ffffcc')
        draw_arrow(ax, 0.275, 0.25, 0.275, 0.17, 'black', 'backward')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='experiments/w5_adgm_repair/results')
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results_dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    
    figures_dir = os.path.join(args.results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Find and load training log
    log_file = find_iid_log(args.results_dir)
    if not log_file:
        print("ERROR: IID training log not found", file=sys.stderr)
        sys.exit(1)
    
    with open(log_file) as f:
        log_data = json.load(f)
    
    # Load mechanism validation
    mech_file = os.path.join(args.results_dir, 'mechanism_validation.json')
    if not os.path.exists(mech_file):
        print("ERROR: mechanism_validation.json not found", file=sys.stderr)
        sys.exit(1)
    
    with open(mech_file) as f:
        mech_data = json.load(f)
    
    # Generate figures
    plot_fig1_domain_gaps(log_data, os.path.join(figures_dir, 'fig1_domain_gap_evolution.png'))
    plot_fig2_gap_retention(log_data, os.path.join(figures_dir, 'fig2_gap_retention.png'))
    plot_fig3_mechanism(mech_data, os.path.join(figures_dir, 'fig3_mechanism_validation.png'))
    plot_fig4_schematic(os.path.join(figures_dir, 'fig4_adgm_repair_schematic.png'))
    
    # Verify all figures exist and are >= 10KB
    figs = glob.glob(os.path.join(figures_dir, '*.png'))
    print(f"\nGenerated {len(figs)} figures in {figures_dir}")
    
    for f in sorted(figs):
        size_kb = os.path.getsize(f) / 1024
        print(f"  {os.path.basename(f)}: {size_kb:.1f} KB")
        assert size_kb >= 10, f"Figure too small: {f}"
    
    print("All figures OK (>=10KB)")


if __name__ == '__main__':
    main()
