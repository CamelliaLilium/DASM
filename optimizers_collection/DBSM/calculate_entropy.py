import json
import numpy as np
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 强制使用 Agg 后端，确保在无显示器环境下运行
plt.switch_backend('Agg')

# 设置论文级别的绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'figure.titlesize': 18
})

def softmax_np(x, tau=0.1):
    """Numpy 实现的 Softmax，带有温度系数 tau"""
    e_x = np.exp((x - np.max(x)) / tau)
    return e_x / e_x.sum()

def calculate_dsbe(sharpness_dict, tau=0.1):
    """
    计算域锐度均衡熵 (Domain Sharpness Balance Entropy, DSBE)
    公式: H = -sum(alpha_k * log(alpha_k))
    其中 alpha_k = exp(S_k / tau) / sum(exp(S_j / tau))
    """
    if not sharpness_dict or len(sharpness_dict) == 0:
        return 0.0
    
    # 提取锐度值 S_k
    s_vals = np.array(list(sharpness_dict.values()), dtype=np.float32)
    
    # 使用 Numpy 实现的吉布斯分布映射
    probs = softmax_np(s_vals, tau=tau)
    
    # 计算熵值
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)

def analyze_and_plot(log_path, output_dir):
    # 1. 数据读取与处理
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found")
        return

    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    sharpness_history = logs.get('domain_sharpness', [])
    acc_history = logs.get('val_acc', [])
    loss_history = logs.get('epoch_loss', [])
    
    if not sharpness_history:
        print("Error: No 'domain_sharpness' data found in logs")
        return

    # 计算每一代的熵值
    entropy_values = [calculate_dsbe(d, tau=0.1) for d in sharpness_history]
    
    # 提取各域原始锐度曲线
    domains = list(sharpness_history[0].keys())
    domain_curves = {d: [h.get(d, 0) for h in sharpness_history] for d in domains}
    
    epochs = range(len(entropy_values))
    os.makedirs(output_dir, exist_ok=True)

    # 保存数据文件
    plot_data = {
        "epochs": list(epochs),
        "entropy": entropy_values,
        "sharpness": domain_curves,
        "accuracy": acc_history,
        "loss": loss_history
    }
    with open(os.path.join(output_dir, 'dbsm_plot_data.json'), 'w') as f:
        json.dump(plot_data, f, indent=4)

    # 2. 论文级联合绘图 (1x3 布局)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- 子图 1: Per-Domain Sharpness (逃离 K-域鞍点) ---
    for d_name, curve in domain_curves.items():
        ax1.plot(epochs, curve, label=f'Sharpness_{d_name}', alpha=0.8, linewidth=1.5)
    ax1.set_title('(a) Evolution of Domain Sharpness')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sharpness Value $S_k$')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 子图 2: Gibbs Entropy (景观平滑度均衡) ---
    ax2.plot(epochs, entropy_values, color='royalblue', linewidth=2.5, label='DSBE (Smoothness)')
    max_e = np.log(len(domains))
    ax2.axhline(y=max_e, color='grey', linestyle='--', label='Ideal Flatness ($\ln K$)')
    ax2.set_title('(b) Landscape Smoothness Entropy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gibbs Entropy $H_{DSBE}$')
    ax2.set_ylim(0, max_e * 1.1)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 子图 3: Performance Correlation (标注清晰版) ---
    line1, = ax3.plot(epochs, acc_history, color='tab:red', linewidth=2, label='Val Accuracy')
    ax3.set_ylabel('Accuracy', color='tab:red')
    ax3.tick_params(axis='y', labelcolor='tab:red')
    
    ax3_twin = ax3.twinx()
    line2, = ax3_twin.plot(epochs, loss_history, color='tab:green', alpha=0.6, linestyle='--', label='Train Loss')
    ax3_twin.set_ylabel('Loss', color='tab:green')
    ax3_twin.tick_params(axis='y', labelcolor='tab:green')
    
    ax3.set_title('(c) Generalization Performance')
    ax3.set_xlabel('Epoch')
    
    # 合并图例到 ax3
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='center right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('DBSM Mathematical Verification: Escape from K-Domain Saddle Points', y=1.05)
    plt.tight_layout()
    
    combined_path = os.path.join(output_dir, 'dbsm_paper_combined_analysis.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"DBSM 分析结果已保存至: {output_dir}")

if __name__ == "__main__":
    log_file = os.environ.get(
        "DASM_DBSM_ENTROPY_LOG",
        os.path.join(
            PROJECT_ROOT,
            "models_collection",
            "Transformer",
            "dbsm_adaptive_contrastive_train_AHCM_LSB_PMS_QIM_to_AHCM_LSB_PMS_QIM",
            "train_logs_QIM+PMS+LSB+AHCM_0.5_1s.json",
        ),
    )
    output = os.environ.get(
        "DASM_DBSM_ENTROPY_OUTPUT",
        os.path.join(PROJECT_ROOT, "optimizers_collection", "DBSM", "analysis_results"),
    )
    analyze_and_plot(log_file, output)
