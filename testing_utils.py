import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def compute_domain_test_acc(model, dataset_name, args):
    """
    Compute test accuracy for a specific domain.
    Returns float accuracy (NaN if data not found).
    """
    # Ensure steg_algorithm attribute exists
    if not hasattr(args, 'steg_algorithm'):
        args.steg_algorithm = 'Transformer'  # Default to Transformer
    
    device = torch.device(args.device)
    model.eval()
    
    steg_path = os.path.join(args.test_data_root, dataset_name, 'Steg')
    cover_path = os.path.join(args.test_data_root, dataset_name, 'Cover')
    
    if not (os.path.exists(steg_path) and os.path.exists(cover_path)):
        return float('nan')
    
    try:
        x_val, y_val = get_alter_loaders_test(steg_path, cover_path)
        
        # 数据预处理 (根据不同模型)
        if args.steg_algorithm == 'FS-MDP':
            x_val = transfer_to_onehot(x_val)
        elif args.steg_algorithm == 'DVSF':
            # DVSF 使用前3维特征
            x_val = x_val[:, :, 0:3]
            x_val = np.where(x_val == -1, 200, x_val)
        elif args.steg_algorithm == 'DAEF-VS':
            # DAEF-VS 使用所有特征 (int64 for embedding)
            x_val = np.where(x_val == -1, 200, x_val)
        else:
            # 其他模型使用前7维特征
            x_val = x_val[:, :, 0:7]
            x_val = np.where(x_val == -1, 200, x_val)
        
        test_loader = convert_to_loader_test(x_val, y_val, args.batch_size)
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # DAEF-VS 需要 int64 类型 (Embedding 层要求)
                if args.steg_algorithm == 'DAEF-VS':
                    inputs = inputs.long()
                
                # 🔧 兼容对比学习模型 (DVSF, DAEF-VS)
                if args.steg_algorithm in ['DVSF', 'DAEF-VS']:
                    # 对比学习模型需要 triplet 输入 (重复3次)
                    batch_size = inputs.size(0)
                    input_triplet = torch.zeros(batch_size * 3, inputs.size(1), inputs.size(2)).to(device)
                    for i in range(batch_size):
                        input_triplet[3 * i] = inputs[i]
                        input_triplet[3 * i + 1] = inputs[i]
                        input_triplet[3 * i + 2] = inputs[i]
                    
                    # 对比学习模型返回 4 个值: (features_unsup, logits_cover, logits_steg, features_sup)
                    _, logits_cover, logits_steg, _ = model(input_triplet)
                    
                    # 使用 cover logits 进行预测 (也可以平均两者)
                    outputs = logits_cover
                else:
                    # 标准模型直接推理
                    outputs = model(inputs)
                
                _, label_indices = torch.max(labels, 1)
                if args.steg_algorithm == 'FS-MDP':
                    predicted = torch.round(outputs).squeeze()
                    correct_preds += (predicted == label_indices.float()).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    correct_preds += (predicted == label_indices).sum().item()
                total_preds += labels.size(0)
        
        accuracy = correct_preds / total_preds if total_preds > 0 else float('nan')
        return float(accuracy)
    except Exception:
        return float('nan')


def get_file_list_test(folder):
    """Get test file list"""
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


def parse_sample_test(file_path):
    """Parse file content and return sample data"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sample = []
    for line in lines:
        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample


def convert_to_loader_test(x_val, y_val, batch_size=64):
    """Convert test data to DataLoader"""
    x_test_tensor = torch.Tensor(x_val)
    y_test_tensor = torch.Tensor(y_val)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def eval_tensor_loader_classification_accuracy(model, loader, args, device=None):
    """
    Classification accuracy on a TensorDataset-backed loader (same convention as training scripts).
    Supports batches of (x, y) or (x, y, algorithm_labels). Empty or None loader -> nan.
    """
    if loader is None or len(loader.dataset) == 0:
        return float("nan")
    if device is None:
        device = torch.device(args.device)
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                inputs, labels, _ = batch_data
            else:
                inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            label_indices = labels.squeeze().long()
            outputs = model(inputs)
            if args.steg_algorithm == "FS-MDP":
                predicted = torch.round(outputs).squeeze()
                correct_preds += (predicted == label_indices.float()).sum().item()
            else:
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == label_indices).sum().item()
            total_preds += labels.size(0)
    return correct_preds / total_preds if total_preds > 0 else float("nan")


def get_alter_loaders_test(file_embed, file_no_embed, max_samples_per_class=5000):
    """
    Get test data loaders with optional sampling.
    
    Args:
        file_embed: Path to steg (positive) samples folder
        file_no_embed: Path to cover (negative) samples folder
        max_samples_per_class: Maximum number of samples to keep per class (default: 5000)
    """
    folders = [
        {"class": 1, "folder": file_embed},
        {"class": 0, "folder": file_no_embed}
    ]
    
    # Collect files for each class separately
    class_files = {0: [], 1: []}
    for folder in folders:
        files = get_file_list_test(folder["folder"])
        class_files[folder["class"]] = files
    
    # Randomly sample up to max_samples_per_class for each class
    sampled_files = []
    for class_label, files in class_files.items():
        if len(files) > max_samples_per_class:
            sampled = np.random.choice(files, size=max_samples_per_class, replace=False).tolist()
            print(f'  Class {class_label}: sampled {max_samples_per_class} from {len(files)} files', flush=True)
        else:
            sampled = files
            print(f'  Class {class_label}: using all {len(files)} files', flush=True)
        sampled_files.extend([(f, class_label) for f in sampled])
    
    # Shuffle all sampled files
    np.random.shuffle(sampled_files)
    
    # Use multiprocessing to speed up file parsing for large test sets
    from multiprocessing import Pool, cpu_count
    
    # Limit number of processes to avoid memory issues
    num_workers = min(cpu_count(), 8)
    
    # Parse files in parallel
    print(f'Loading {len(sampled_files)} test files using {num_workers} workers...', flush=True)
    with Pool(processes=num_workers) as pool:
        all_samples_x = pool.map(parse_sample_test, [item[0] for item in sampled_files])
    
    all_samples_y = [item[1] for item in sampled_files]
    np_all_samples_x = np.asarray(all_samples_x)
    np_all_samples_y = np.asarray(all_samples_y)

    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    y_train = encoder.fit_transform(np_all_samples_y.reshape(-1, 1))

    x_test = np_all_samples_x
    y_test_ori = np_all_samples_y
    y_test = encoder.transform(y_test_ori.reshape(-1, 1))

    return x_test, y_test


def transfer_to_onehot(data):
    """Convert integer data to one-hot encoding for FS-MDP."""
    num_samples, seq_len, _ = data.shape
    onehot_data = np.zeros((num_samples, seq_len, 192), dtype=np.float32)

    for i in range(num_samples):
        sample = data[i, :, :]
        temp = np.zeros((seq_len, 192))
        
        images = sample[:, 1:4].copy().astype(np.float32)
        images[:, 1] += 128.
        images[:, 2] += 160.
        
        for f in range(seq_len):
            for c in range(images.shape[1]):
                index = int(images[f, c])
                if 0 <= index < 192:
                    temp[f, index] = 1
        onehot_data[i, :, :] = temp
        
    return onehot_data


def test_external_datasets(args):
    """Test external datasets using saved checkpoint: QIM, PMS, LSB, AHCM"""
    # Ensure steg_algorithm attribute exists
    if not hasattr(args, 'steg_algorithm'):
        args.steg_algorithm = 'Transformer'  # Default to Transformer
    
    embedding_str = str(args.embedding_rate)
    test_datasets = [
        f'QIM_{embedding_str}',
        f'PMS_{embedding_str}',
        f'LSB_{embedding_str}',
        f'AHCM_{embedding_str}'
    ]
    
    # Load model
    from models_collection.Transformer.transformer import Classifier1
    model = Classifier1(args)
    device = torch.device(args.device)
    model = model.to(device)

    # Load best weights
    import sys
    sys.path.append(PROJECT_ROOT)
    from model_multi import get_model_filename
    from utils.naming import get_result_filename
    model_filename = get_model_filename(args)
    checkpoint_path = os.path.join(args.result_path, model_filename)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f'Loaded best checkpoint from: {checkpoint_path}')
    model.load_state_dict(best_checkpoint['model'])
    
    for dataset_name in test_datasets:
        accuracy = compute_domain_test_acc(model, dataset_name, args)
        if not np.isnan(accuracy):
            print(f"Test Accuracy on {dataset_name}: {accuracy:.4f}")
            
            # Write results to file
            os.makedirs(args.result_path, exist_ok=True)
            result_filename = get_result_filename(args)
            with open(os.path.join(args.result_path, result_filename), 'a') as f:
                f.write(f"{dataset_name} test acc %.4f\n" % accuracy)


def test_current_model(model, args):
    """Test current model on external datasets without loading checkpoint"""
    # Ensure steg_algorithm attribute exists
    if not hasattr(args, 'steg_algorithm'):
        args.steg_algorithm = 'Transformer'  # Default to Transformer
    
    embedding_str = str(args.embedding_rate)
    test_datasets = [
        f'QIM_{embedding_str}',
        f'PMS_{embedding_str}',
        f'LSB_{embedding_str}',
        f'AHCM_{embedding_str}'
    ]
    
    for dataset_name in test_datasets:
        if args.steg_algorithm == 'CCN':
            return

        accuracy = compute_domain_test_acc(model, dataset_name, args)
        if not np.isnan(accuracy):
            print(f"Test Accuracy on {dataset_name}: {accuracy:.4f}")
            
            # Write results to file
            os.makedirs(args.result_path, exist_ok=True)
            import sys
            sys.path.append(PROJECT_ROOT)
            from utils.naming import get_result_filename
            result_filename = get_result_filename(args)
            with open(os.path.join(args.result_path, result_filename), 'a') as f:
                f.write(f"{dataset_name} test acc %.4f\n" % accuracy)


def test_ccn_model(ccn_model_path, args):
    from models_collection.CCN.trainer import CNN_pitch
    import pickle
    from sklearn.decomposition import PCA
    from sklearn import svm

    embedding_str = str(args.embedding_rate)
    test_datasets = [
        f'QIM_{embedding_str}',
        f'PMS_{embedding_str}',
        f'LSB_{embedding_str}',
        f'AHCM_{embedding_str}'
    ]

    # Load trained PCA and SVM models
    with open(os.path.join(ccn_model_path, 'pca.pkl'), 'rb') as f:
        pca = pickle.load(f)
    with open(os.path.join(ccn_model_path, 'svm.pkl'), 'rb') as f:
        clf = pickle.load(f)

    for dataset_name in test_datasets:
        steg_path = os.path.join(args.test_data_root, dataset_name, 'Steg')
        cover_path = os.path.join(args.test_data_root, dataset_name, 'Cover')

        if os.path.exists(steg_path) and os.path.exists(cover_path):
            print(f'Testing {dataset_name}:')
            
            positive_files = [os.path.join(steg_path, p) for p in os.listdir(steg_path)]
            negative_files = [os.path.join(cover_path, p) for p in os.listdir(cover_path)]
            
            num_test_files = len(negative_files)

            X_test, Y_test = [], []
            for f in tqdm(negative_files, desc=f"Processing negative samples for {dataset_name}"):
                feature = CNN_pitch(f)
                feature_pca = pca.transform(feature.reshape(1, -1))
                X_test.append(feature_pca)
                Y_test.append(0)

            for f in tqdm(positive_files, desc=f"Processing positive samples for {dataset_name}"):
                feature = CNN_pitch(f)
                feature_pca = pca.transform(feature.reshape(1, -1))
                X_test.append(feature_pca)
                Y_test.append(1)

            X_test = np.row_stack(X_test)
            Y_predict = clf.predict(X_test)

            true_negative = np.sum((Y_predict[:num_test_files] == 0))
            false_positive = np.sum((Y_predict[:num_test_files] == 1))
            true_positive = np.sum((Y_predict[num_test_files:] == 1))
            false_negative = np.sum((Y_predict[num_test_files:] == 0))
            
            total_samples = len(Y_predict)
            accuracy = (true_positive + true_negative) / total_samples if total_samples > 0 else 0
            
            print(f"Test Accuracy on {dataset_name}: {accuracy:.4f}")

            # Write results to file
            os.makedirs(args.result_path, exist_ok=True)
            import sys
            sys.path.append(PROJECT_ROOT)
            from utils.naming import get_result_filename
            result_filename = get_result_filename(args)
            with open(os.path.join(args.result_path, result_filename), 'a') as f:
                f.write(f"{dataset_name} test acc %.4f\n" % accuracy)
        else:
            print(f'Test data not found for {dataset_name}')


def test_ss_qccn_model(model_path, args):
    from models_collection.SS_QCCN.trainer import G729_SS_QCCCN
    import pickle
    from sklearn.decomposition import PCA
    from sklearn import svm
    
    embedding_str = str(args.embedding_rate)
    test_datasets = [f'QIM_{embedding_str}', f'PMS_{embedding_str}', f'LSB_{embedding_str}', f'AHCM_{embedding_str}']

    with open(os.path.join(model_path, 'pca.pkl'), 'rb') as f:
        pca = pickle.load(f)
    with open(os.path.join(model_path, 'svm.pkl'), 'rb') as f:
        clf = pickle.load(f)

    for dataset_name in test_datasets:
        steg_path = os.path.join(args.test_data_root, dataset_name, 'Steg')
        cover_path = os.path.join(args.test_data_root, dataset_name, 'Cover')

        if os.path.exists(steg_path) and os.path.exists(cover_path):
            print(f'Testing {dataset_name}:')
            
            positive_files = [os.path.join(steg_path, p) for p in os.listdir(steg_path)]
            negative_files = [os.path.join(cover_path, p) for p in os.listdir(cover_path)]
            
            num_test_files = len(negative_files)
            X_test, Y_test = [], []

            for f in tqdm(negative_files, desc=f"Processing negative samples for {dataset_name}"):
                feature = G729_SS_QCCCN(f)
                feature_pca = pca.transform(feature.reshape(1, -1))
                X_test.append(feature_pca)
            
            for f in tqdm(positive_files, desc=f"Processing positive samples for {dataset_name}"):
                feature = G729_SS_QCCCN(f)
                feature_pca = pca.transform(feature.reshape(1, -1))
                X_test.append(feature_pca)

            X_test = np.row_stack(X_test)
            Y_predict = clf.predict(X_test)

            true_negative = np.sum(Y_predict[:num_test_files] == 0)
            true_positive = np.sum(Y_predict[num_test_files:] == 1)
            accuracy = (true_positive + true_negative) / len(Y_predict) if len(Y_predict) > 0 else 0
            
            print(f"Test Accuracy on {dataset_name}: {accuracy:.4f}")

            os.makedirs(args.result_path, exist_ok=True)
            import sys
            sys.path.append(PROJECT_ROOT)
            from utils.naming import get_result_filename
            result_filename = get_result_filename(args)
            with open(os.path.join(args.result_path, result_filename), 'a') as f:
                f.write(f"{dataset_name} test acc %.4f\n" % accuracy)
        else:
            print(f'Test data not found for {dataset_name}')


def plot_domain_test_acc_curves(gen_logs, args):
    """Plot domain test accuracy curves (float values)."""
    if 'domain_test_acc' not in gen_logs or not gen_logs['domain_test_acc']:
        return
    
    ds_id = args.dataset_id if args.dataset_id is not None else str(args.embedding_rate)
    # Clean ds_id for filename usage by taking only the basename
    ds_id_clean = os.path.basename(ds_id).replace('.pkl', '')
    plot_dir = os.path.join(args.result_path, f'training_plots_{ds_id_clean}')
    os.makedirs(plot_dir, exist_ok=True)
    
    epochs = list(range(1, len(gen_logs['domain_test_acc']) + 1))
    domain_names = ['QIM', 'PMS', 'LSB', 'AHCM']
    colors = ['blue', 'red', 'green', 'orange']
    
    plt.figure(figsize=(10, 6))
    
    # Collect data for each domain - all four domains should have data
    has_valid_data = False
    for domain_name, color in zip(domain_names, colors):
        acc_values = []
        valid_epochs = []
        for idx, epoch_data in enumerate(gen_logs['domain_test_acc']):
            if isinstance(epoch_data, dict) and domain_name in epoch_data:
                val = epoch_data[domain_name]
                # All domains should have data (0.0 if data not found, but still a valid value)
                if val is not None:
                    try:
                        val_float = float(val)
                        if not np.isnan(val_float):
                            acc_values.append(val_float)
                            valid_epochs.append(idx + 1)  # epoch is 1-indexed
                    except (ValueError, TypeError):
                        pass
        
        # Plot all domains as scatter points (original style)
        if len(acc_values) > 0:
            plt.plot(valid_epochs, acc_values, color=color, label=f'{domain_name} Test Acc', 
                    marker='o', linewidth=0, markersize=8, linestyle='', alpha=0.8)
            has_valid_data = True
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Domain Test Accuracy Evolution')
    # Only call legend if there's valid data to avoid matplotlib warning
    if has_valid_data:
        plt.legend(loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    # Set Y-axis range to [0, 1] for accuracy values
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'domain_test_acc_curve_{ds_id_clean}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
