import json
import numpy as np
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 强制使用 Agg 后端
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

def calculate_balance_entropy(vals_dict, tau=0.1, reverse=False):
    """
    为了与 DBSM 保持一致，计算基于吉布斯分布的熵。
    reverse=True 用于锐度（数值越低越好），reverse=False 用于准确率（数值越高越好）。
    """
    if not vals_dict or len(vals_dict) == 0:
        return 0.0
    vals = np.array(list(vals_dict.values()), dtype=np.float32)
    # 映射为概率分布。为了体现平衡，我们希望各域表现越接近熵越高。
    # 统一使用 softmax 映射。对于准确率，越高代表性能越好。
    inputs = -vals if reverse else vals
    probs = softmax_np(inputs, tau=tau)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)

def analyze_and_plot(log_path, output_dir):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found")
        return

    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    test_acc_history = logs.get('domain_test_acc', [])
    val_acc_history = logs.get('val_acc', [])
    loss_history = logs.get('epoch_loss', [])
    
    # 提取有效的域测试数据
    valid_test_acc = [d for d in test_acc_history if d]
    if not valid_test_acc:
        print("Error: No 'domain_test_acc' data found in logs")
        return

    # 计算熵值 (与 DBSM 逻辑对齐，反映域间均衡性)
    entropy_values = [calculate_balance_entropy(d, tau=0.1) for d in valid_test_acc]
    
    # 提取域性能曲线
    domains = list(valid_test_acc[0].keys())
    domain_curves = {d: [h.get(d, 0) for h in valid_test_acc] for d in domains}
    
    # 映射到 epoch 坐标
    test_interval = 5 
    test_epochs = [i * test_interval + test_interval for i in range(len(entropy_values))]
    all_epochs = range(len(val_acc_history))

    os.makedirs(output_dir, exist_ok=True)

    # 保存数据文件
    plot_data = {
        "test_epochs": test_epochs,
        "all_epochs": list(all_epochs),
        "entropy": entropy_values,
        "domain_acc": domain_curves,
        "val_accuracy": val_acc_history,
        "loss": loss_history
    }
    with open(os.path.join(output_dir, 'sam_plot_data.json'), 'w') as f:
        json.dump(plot_data, f, indent=4)

    # 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    # (a) Domain Accuracy Evolution
    for d_name, curve in domain_curves.items():
        ax1.plot(test_epochs, curve, label=f'Acc_{d_name}', marker='o', markersize=4, alpha=0.8)
    ax1.set_title('(a) SAM: Domain Performance Imbalance')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Domain Test Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # (b) Performance Balance Entropy (DPBE)
    ax2.plot(test_epochs, entropy_values, color='darkorange', linewidth=2.5, marker='s', label='DPBE (Balance)')
    max_e = np.log(len(domains))
    ax2.axhline(y=max_e, color='red', linestyle=':', alpha=0.5, label='Ideal Balance ($\ln K$)')
    ax2.set_title('(b) SAM: Performance Stability Entropy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gibbs Entropy $H_{DPBE}$')
    
    # 动态调整 Y 轴，使变化趋势明显
    y_min, y_max = min(entropy_values), max(entropy_values)
    padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    ax2.set_ylim(y_min - padding, min(max_e + padding, 1.45))
    
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # (c) Overall Performance
    line1, = ax3.plot(all_epochs, val_acc_history, color='tab:red', linewidth=2, label='Val Accuracy')
    ax3.set_ylabel('Accuracy', color='tab:red')
    ax3.tick_params(axis='y', labelcolor='tab:red')
    
    ax3_twin = ax3.twinx()
    line2, = ax3_twin.plot(all_epochs, loss_history, color='tab:green', alpha=0.6, linestyle='--', label='Train Loss')
    ax3_twin.set_ylabel('Loss', color='tab:green')
    ax3_twin.tick_params(axis='y', labelcolor='tab:green')
    
    ax3.set_title('(c) SAM: Generalization Performance')
    ax3.set_xlabel('Epoch')
    ax3.legend([line1, line2], ['Accuracy (Left)', 'Loss (Right)'], loc='center right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('SAM Baseline Verification: The "Averaging" Trap and Domain Bias', y=1.05)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'sam_paper_combined_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SAM 分析结果已保存至: {output_dir}")

if __name__ == "__main__":
    log_file = os.environ.get(
        "DASM_SAM_ENTROPY_LOG",
        os.path.join(
            PROJECT_ROOT,
            "models_collection",
            "Transformer",
            "sam_train_AHCM_LSB_PMS_QIM_to_AHCM_LSB_PMS_QIM",
            "train_logs_QIM+PMS+LSB+AHCM_0.5_1s.json",
        ),
    )
    output = os.environ.get(
        "DASM_SAM_ENTROPY_OUTPUT",
        os.path.join(PROJECT_ROOT, "optimizers_collection", "SAM", "analysis_results"),
    )
    analyze_and_plot(log_file, output)
