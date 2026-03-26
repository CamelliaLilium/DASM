"""
从保存的t-SNE数据重新绘制可视化图

用法:
  python replot_tsne.py --data_file tsne_data_epoch_10.npz --output_file tsne_replot_10.png
  python replot_tsne.py --data_dir ./tsne_results --epoch 10
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# ICML风格设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'text.usetex': False,  # 如果系统有LaTeX可以设为True
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def plot_tsne_visualization(features, class_labels, domain_labels, epoch, output_dir,
                            perplexity=30, n_iter=1000, random_state=42, fontsize=16):
    """
    ICML风格：绘制t-SNE可视化图（单独输出5张图）
    
    Args:
        features: (N, d_model) 特征向量
        class_labels: (N,) 类别标签 (Cover=0, Stego=1)
        domain_labels: (N,) 域标签 (QIM=0, PMS=1, LSB=2, AHCM=3, Cover=-1)
        epoch: 当前epoch
        output_dir: 输出目录
        perplexity: t-SNE perplexity参数
        n_iter: t-SNE迭代次数
        random_state: 随机种子
        fontsize: 字体大小
    """
    # ICML专业配色方案（优化对比度）
    colors = {
        'cover': '#2c3e50',      # 深蓝色（Cover）
        'cover_edge': '#1a252f', # Cover边缘色（更深）
        'stego_all': '#ff7f0e',  # 亮橙色（Stego All）
        'stego_all_edge': '#cc6600', # Stego All边缘色
        'background': '#FFFFFF',
        'grid': '#E0E0E0',
        'text': '#2C3E50'
    }
    
    # 各域Stego颜色
    domain_colors = {
        'AHCM': '#9467bd',  # 深紫色
        'LSB': '#17becf',   # 青绿色/Teal
        'PMS': '#bcbd22',   # 金黄色
        'QIM': '#e377c2'    # 品红色
    }
    
    # 各域Stego边缘色（稍深）
    domain_edge_colors = {
        'AHCM': '#7a4fa0',
        'LSB': '#1299a8',
        'PMS': '#9a9d1c',
        'QIM': '#b85fa0'
    }
    
    # 执行t-SNE降维到3D
    print(f"Computing t-SNE for epoch {epoch}...")
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter,
                random_state=random_state, verbose=1, n_jobs=-1)
    features_3d = tsne.fit_transform(features)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: Cover vs All Stego
    cover_mask = (class_labels == 0)
    stego_mask = (class_labels == 1)
    
    fig1 = plt.figure(figsize=(8, 6), facecolor='white')
    ax1 = fig1.add_subplot(111, projection='3d')
    
    if cover_mask.sum() > 0:
        ax1.scatter(features_3d[cover_mask, 0], features_3d[cover_mask, 1], features_3d[cover_mask, 2],
                   c=colors['cover'], label='Cover', s=15, alpha=0.35, 
                   edgecolors=colors['cover_edge'], linewidths=0.2, depthshade=True)
    if stego_mask.sum() > 0:
        ax1.scatter(features_3d[stego_mask, 0], features_3d[stego_mask, 1], features_3d[stego_mask, 2],
                   c=colors['stego_all'], label='Stego', s=15, alpha=0.7, 
                   edgecolors=colors['stego_all_edge'], linewidths=0.2, depthshade=True)
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=fontsize, labelpad=12, color=colors['text'])
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=fontsize, labelpad=12, color=colors['text'])
    ax1.set_zlabel('t-SNE Dimension 3', fontsize=fontsize, labelpad=12, color=colors['text'])
    ax1.legend(fontsize=fontsize, loc='upper left', frameon=True, 
              fancybox=False, shadow=False, framealpha=0.95, edgecolor='gray',
              markerscale=2.0)  # 增大图例中的点
    ax1.tick_params(labelsize=fontsize-1, colors=colors['text'])
    ax1.grid(True, alpha=0.25, color=colors['grid'], linestyle='--', linewidth=0.8)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor(colors['grid'])
    ax1.yaxis.pane.set_edgecolor(colors['grid'])
    ax1.zaxis.pane.set_edgecolor(colors['grid'])
    ax1.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    output_file1 = os.path.join(output_dir, f'Cover_All_{epoch}.png')
    plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1)
    plt.close()
    print(f"Saved: {output_file1}")
    
    # 图2-5: Cover vs 各个隐写域
    domain_names = ['AHCM', 'LSB', 'PMS', 'QIM']
    domain_ids_map = {'AHCM': 3, 'LSB': 2, 'PMS': 1, 'QIM': 0}
    
    for domain_name in domain_names:
        domain_id = domain_ids_map[domain_name]
        
        fig = plt.figure(figsize=(8, 6), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        # Cover样本
        cover_mask = (class_labels == 0)
        if cover_mask.sum() > 0:
            ax.scatter(features_3d[cover_mask, 0], features_3d[cover_mask, 1], features_3d[cover_mask, 2],
                      c=colors['cover'], label='Cover', s=15, alpha=0.35, 
                      edgecolors=colors['cover_edge'], linewidths=0.2, depthshade=True)
        
        # 该域的Stego样本
        domain_stego_mask = (class_labels == 1) & (domain_labels == domain_id)
        if domain_stego_mask.sum() > 0:
            domain_color = domain_colors[domain_name]
            domain_edge_color = domain_edge_colors[domain_name]
            ax.scatter(features_3d[domain_stego_mask, 0], features_3d[domain_stego_mask, 1],
                      features_3d[domain_stego_mask, 2],
                      c=domain_color, label=domain_name, s=15, alpha=0.7, 
                      edgecolors=domain_edge_color, linewidths=0.2, depthshade=True)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=fontsize, labelpad=12, color=colors['text'])
        ax.set_ylabel('t-SNE Dimension 2', fontsize=fontsize, labelpad=12, color=colors['text'])
        ax.set_zlabel('t-SNE Dimension 3', fontsize=fontsize, labelpad=12, color=colors['text'])
        ax.legend(fontsize=fontsize, loc='upper left', frameon=True, 
                 fancybox=False, shadow=False, framealpha=0.95, edgecolor='gray',
                 markerscale=2.0)  # 增大图例中的点
        ax.tick_params(labelsize=fontsize-1, colors=colors['text'])
        ax.grid(True, alpha=0.25, color=colors['grid'], linestyle='--', linewidth=0.8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(colors['grid'])
        ax.yaxis.pane.set_edgecolor(colors['grid'])
        ax.zaxis.pane.set_edgecolor(colors['grid'])
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'Cover_{domain_name}_{epoch}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.1)
        plt.close()
        print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Replot t-SNE visualization from saved data')
    
    # 两种模式：指定文件或指定目录+epoch
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_file', type=str,
                       help='Path to saved .npz data file')
    group.add_argument('--data_dir', type=str,
                       help='Directory containing tsne_data_epoch_*.npz files')
    
    parser.add_argument('--epoch', type=int, default=None,
                       help='Epoch number (only used with --data_dir)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output image file path (deprecated, use --output_dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for images (default: same as data directory)')
    parser.add_argument('--perplexity', type=float, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='t-SNE number of iterations')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for t-SNE')
    parser.add_argument('--fontsize', type=int, default=16,
                       help='Font size for plots (default: 16)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs for t-SNE computation (-1 for all cores)')
    
    args = parser.parse_args()
    
    # 确定数据文件
    if args.data_file:
        data_file = args.data_file
        epoch = None
    else:
        if args.epoch is None:
            print("Error: --epoch is required when using --data_dir")
            return 1
        data_file = os.path.join(args.data_dir, f'tsne_data_epoch_{args.epoch}.npz')
        epoch = args.epoch
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return 1
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    elif args.output_file:
        # 如果指定了output_file，使用其目录
        output_dir = os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.'
    else:
        if epoch is None:
            # 从文件名提取epoch
            basename = os.path.basename(data_file)
            epoch = int(basename.split('_')[-1].replace('.npz', ''))
        output_dir = os.path.dirname(data_file) if args.data_file else args.data_dir
    
    # 加载数据
    print(f"Loading data from: {data_file}")
    data = np.load(data_file)
    features = data['features']
    class_labels = data['class_labels']
    domain_labels = data['domain_labels']
    saved_epoch = int(data.get('epoch', epoch if epoch else 0))
    
    print(f"Loaded: {len(features)} samples, epoch={saved_epoch}")
    print(f"  Cover samples: {np.sum(class_labels == 0)}")
    print(f"  Stego samples: {np.sum(class_labels == 1)}")
    for domain_id, domain_name in [(0, 'QIM'), (1, 'PMS'), (2, 'LSB'), (3, 'AHCM')]:
        count = np.sum((class_labels == 1) & (domain_labels == domain_id))
        print(f"  {domain_name} samples: {count}")
    
    # 绘制（ICML风格，单独输出5张图）
    plot_tsne_visualization(
        features, class_labels, domain_labels, saved_epoch, output_dir,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.random_state,
        fontsize=args.fontsize
    )
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
