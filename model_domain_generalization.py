import os
import json
import random
import pickle
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import datetime
from sklearn.model_selection import train_test_split
import argparse
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from sasm import AlgorithmAwareSAM, _hybrid_sasm_step
    SASM_IMPORT_ERROR = None
except ImportError as exc:
    AlgorithmAwareSAM = None
    _hybrid_sasm_step = None
    SASM_IMPORT_ERROR = exc


def _optimizer_is_sasm(optimizer):
    """Safe isinstance: optional sasm import sets AlgorithmAwareSAM to None."""
    return AlgorithmAwareSAM is not None and isinstance(optimizer, AlgorithmAwareSAM)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# dataset/ 与 DASM/ 同级：.../autodl-tmp/dataset/model_train
_DATASET_SIBLING = os.path.join(os.path.dirname(PROJECT_ROOT), 'dataset')
_DEFAULT_MODEL_TRAIN = os.path.join(_DATASET_SIBLING, 'model_train')
_DEFAULT_MODEL_TEST = os.path.join(_DATASET_SIBLING, 'model_test')

DOMAIN_MAP = {
    'QIM': 0,
    'PMS': 1,
    'LSB': 2,
    'AHCM': 3
}

DEFAULT_TRAIN_DATA_ROOT = os.environ.get('DASM_DATA_ROOT', _DEFAULT_MODEL_TRAIN)

def set_gpu(gpu_id):
    """Set the GPU to use."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch QIM Steganalysis Model')
    
    # Data related arguments
    parser.add_argument('--train_dataset', type=str, default='QIM', 
                        choices=['QIM', 'PMS', 'LSB', 'AHCM', 'combined'], help='Training dataset type')
    parser.add_argument('--embedding_rate', type=float, default=0.1, 
                        choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Embedding rate (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)')
    parser.add_argument('--sample_length', type=int, default=1000, 
                        help='Sample length (ms)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating (effective_batch_size = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--num_class', type=int, default=2, 
                        help='Number of classes')
    
    # Model related arguments
    parser.add_argument('--hidden_num', type=int, default=64, 
                        help='Hidden layer units')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.5, 
                        help='Dropout probability')
    parser.add_argument('--d_model', type=int, default=64, 
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, 
                        help='Feed-forward network dimension')
    parser.add_argument('--max_len', type=int, default=100, 
                        help='Maximum length for positional encoding')
    
    # Training related arguments
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay')
        
    # Path related arguments
    parser.add_argument('--result_path', type=str, 
                        default=os.environ.get('DASM_RESULT_ROOT', os.path.join(PROJECT_ROOT, 'results_domain_gen', 'models_base')),
                        help='Results save path')
    parser.add_argument('--data_root', type=str, 
                        default=os.environ.get('DASM_COMBINED_DATA_ROOT', _DEFAULT_MODEL_TRAIN),
                        help='Data root directory')
    parser.add_argument('--test_data_root', type=str, 
                        default=os.environ.get('DASM_TEST_DATA_ROOT', _DEFAULT_MODEL_TEST),
                        help='Test data root directory')
    parser.add_argument('--train_label_csv', type=str, 
                        default=None, 
                        help='Training label CSV file path')
    parser.add_argument('--dataset_id', type=str, default=None,
                        help='ID for combined dataset, used when train_dataset is "combined"')
    
    # Device related arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], help='Training device')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    # Other arguments
    parser.add_argument('--save_model', action='store_true', 
                        help='Whether to save model')
    parser.add_argument('--test_only', action='store_true', 
                        help='Test only mode')
    parser.add_argument('--alpha', type=float, default=-1, 
                        help='Focal Loss alpha parameter')
    parser.add_argument('--gamma', type=float, default=2, 
                        help='Focal Loss gamma parameter')
    

    # Algorithm selection argument
    parser.add_argument('--steg_algorithm', type=str, default='Transformer',
                        choices=['Transformer', 'LStegT', 'FS-MDP', 'CCN', 'SS-QCCN', 'SFFN', 'KFEF', 'DVSF', 'DAEF-VS'],
                        help='Steganalysis algorithm to use for the model architecture.')
    
    # SASM specific arguments
    parser.add_argument('--use_sasm', action='store_true',
                        help='Use SASM optimizer instead of standard optimizer')
    parser.add_argument('--rho', type=float, default=0.05,
                        help='Rho parameter for SASM')
    parser.add_argument('--adaptive_rho', action='store_true',
                        help='Use adaptive rho adjustment during training')
    parser.add_argument('--rho_min', type=float, default=0.01,
                        help='Minimum rho value for adaptive adjustment')
    parser.add_argument('--rho_max', type=float, default=1.0,
                        help='Maximum rho value for adaptive adjustment')
    
    parser.add_argument('--eval_step', type=int, default=20,
                        help='Run external eval every N epochs (0=disabled)')
    
    # ===================== Domain test evaluation =====================
    parser.add_argument('--domain_test_interval', type=int, default=5,
                        help='Interval (epochs) for domain test evaluation (0 to disable, default 5).')
    # =================================================================

    parser.add_argument('--train_domains', type=str, default='QIM,PMS,LSB,AHCM',
                        help='Comma-separated training domains (e.g., "QIM,PMS")')
    parser.add_argument('--test_domains', type=str, default='QIM,PMS,LSB,AHCM',
                        help='Comma-separated testing domains (e.g., "LSB,AHCM")')
    
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_data_paths(args):
    """Get data paths based on dataset type, with auto-detection for SASM."""
    sample_len_str = f"_{int(args.sample_length / 1000)}s"
    
    if args.train_dataset == 'combined':
        if not args.dataset_id:
            raise ValueError("dataset_id must be provided for combined dataset")
        file_embed = None
        file_no_embed = None
        
        if '/' in args.dataset_id:
            pkl_file = args.dataset_id if args.dataset_id.endswith('.pkl') else f"{args.dataset_id}.pkl"
        else:
            pkl_dir = os.path.join(args.data_root, 'combined_multi')
            if not os.path.exists(pkl_dir):
                pkl_dir = args.data_root
            if args.dataset_id.endswith('.pkl'):
                pkl_file = os.path.join(pkl_dir, args.dataset_id)
            else:
                base_pkl_name = f"{args.dataset_id}{sample_len_str}"
                if args.use_sasm:
                    sasm_pkl_file = os.path.join(pkl_dir, f"{base_pkl_name}_alg.pkl")
                    if os.path.exists(sasm_pkl_file):
                        print(f"INFO: --use_sasm is enabled. Automatically using algorithm-labeled dataset: {os.path.basename(sasm_pkl_file)}")
                        pkl_file = sasm_pkl_file
                    else:
                        print(f"WARNING: --use_sasm is enabled, but algorithm-labeled dataset not found at '{sasm_pkl_file}'.")
                        print("         Falling back to standard combined dataset. SASM may not work as expected.")
                        pkl_file = os.path.join(pkl_dir, f"{base_pkl_name}.pkl")
                else:
                    pkl_file = os.path.join(pkl_dir, f"{base_pkl_name}.pkl")
    else:
        embedding_str = str(args.embedding_rate)
        dataset_name = f'{args.train_dataset}_{embedding_str}'
        dataset_path = os.path.join(args.data_root, dataset_name)
        
        file_embed = os.path.join(dataset_path, 'Steg')
        file_no_embed = os.path.join(dataset_path, 'Cover')
        
        pkl_name = f'{args.train_dataset.lower()}_{embedding_str}{sample_len_str}_all.pkl'
        pkl_file = os.path.join(dataset_path, 'pklfile', pkl_name)

    return file_embed, file_no_embed, pkl_file


def parse_sample(base_path, file_name):
    """Parse file content and return sample data"""
    file_path = os.path.join(base_path, file_name)
    # Check if file path exists
    if not os.path.exists(file_path):
        print(f"File path does not exist: {file_path}")
        return None
    # Parse file content and return sample data
    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:
        # Split each line by spaces into integer list
        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample


def _generate_dataset_from_sources(file_embed, file_no_embed, pkl_file, args):
    """Generate dataset from source files, save to pkl, and return data."""
    print("Generating dataset from source files...")
    print(f"🔍 DEBUG: Data source paths:")
    print(f"   - file_embed: {file_embed}")
    print(f"   - file_no_embed: {file_no_embed}")
    print(f"   - pkl_file: {pkl_file}")

    # If pkl file doesn't exist or fails to load, regenerate it.
    files = os.listdir(file_embed)
    print(f"🔍 DEBUG: Embed files count: {len(files)}")
    files1 = os.listdir(file_no_embed)
    print(f"🔍 DEBUG: No-embed files count: {len(files1)}")
    
    # Check if directories exist and have files
    if len(files) == 0:
        print(f"⚠️  WARNING: No files found in embed directory: {file_embed}")
    if len(files1) == 0:
        print(f"⚠️  WARNING: No files found in no-embed directory: {file_no_embed}")
        print(f"⚠️  This will result in unbalanced dataset with only positive samples!")

    df = pd.read_csv(args.train_label_csv, header=None)
    print('################ embed数据进行处理 ##################')

    # 对embed数据进行处理
    file_list_embed = os.listdir(file_embed)
    file_classes_embed = {}
    print('开始遍历嵌入文件列表')
    for file in file_list_embed:
        file_name = os.path.splitext(file)[0]  # Use filename without extension as the key
        matching_row = df.loc[df[0] == file_name]  # Match class info by file name
        if not matching_row.empty:
            class_name = matching_row.iloc[0, 2]  # Get class info for file
            if file not in file_classes_embed:  # Check if file exists in dict
                file_classes_embed[file] = class_name  # Add file name and class info to dict
            else:
                print("File already exists in dictionary:", file_name)
        else:
            print("No matching class information found:", file_name)
    print("Number of files in file_classes_embed:", len(file_classes_embed))

    # Group files by class
    class_files_embed = {}
    for file, class_name in file_classes_embed.items():
        if class_name not in class_files_embed:
            class_files_embed[class_name] = []
        class_files_embed[class_name].append(file)
    # Print class names in class_files
    # print("Number of files in class_files_embed:", len(class_files_embed))
    # Output file count for each class
    # print("Number of files in class_files:", len(class_files_embed))
    print("Class names in class_files:", list(class_files_embed.keys()))

    # Initialize total file count
    total_files = 0
    # Output number of files for each class_name
    for class_name, files in class_files_embed.items():
        num_files = len(files)
        print(f"Number of files for class {class_name}: {num_files}")
        total_files += num_files
    print(f"Total number of files: {total_files}")

    print('################ noembed数据进行处理 ##################')
    
    # Process non-embedded data
    file_list_noembed = os.listdir(file_no_embed)
    file_classes_noembed = {}  # Store file names and class info for non-embedded files
    # Iterate through non-embedded file list to get class info for each file
    print('Starting to iterate through non-embedded file list')
    for file in file_list_noembed:
        file_name = file.split('_')[0]  # Extract file name
        # Assign special class ID for non-embedded files, using -1
        class_name = -1
        file_classes_noembed[file] = class_name  # Add file name and class info to dict
    print("Number of files in file_classes_noembed:", len(file_classes_noembed))
    
    # Group files by class
    class_files_noembed = {}
    for file, class_name in file_classes_noembed.items():
        if class_name not in class_files_noembed:
            class_files_noembed[class_name] = []
        class_files_noembed[class_name].append(file)
    # Print class names in class_files
    # print("Number of files in class_files_embed:", len(class_files_embed))
    # Output file count for each class
    # print("Number of files in class_files:", len(class_files_noembed))
    print("Class names in class_files:", list(class_files_noembed.keys()))

    # Initialize total file count
    total_files = 0
    # Output number of files for each class_name
    for class_name, files in class_files_noembed.items():
        num_files = len(files)
        print(f"Number of files for class {class_name}: {num_files}")
        total_files += num_files
    print(f"Total number of files: {total_files}")

    print('################ Data statistics completed, starting to build dataset ##################')
    # Initialize total file count
    total_files = 0
    # Output number of files for each class_name
    for class_name, files in class_files_embed.items():
        # Calculate number of files for each class
        num_files = len(files)
        print(f"Number of files for class {class_name}: {num_files}")
        # Add to total file count
        total_files += num_files
    
    for class_name, files in class_files_noembed.items():
        # Calculate number of files for each class
        num_files = len(files)
        print(f"Number of files for class {class_name}: {num_files}")
        # Add to total file count
        total_files += num_files
    # Output total file count
    print(f"Total number of files: {total_files}")
    print('Traversal completed, starting to split dataset')
    
    # Split into training and test sets
    train, test = [], []
    print('################ Building embedded dataset ##################')
    # Process each class in class_files dictionary
    for class_name, files in class_files_embed.items():
        random.shuffle(files)  # Randomly shuffle file order
        print('class_name', class_name)
        print(f"Total files for class {class_name}: {len(files)}")
        test_size = int(0.2 * len(files))  # Calculate test set size
        test_files = files[:test_size]  # Get test file list
        train_files = files[test_size:]  # Get train file list
        # Print test and train file counts
        print(f"Test files for class {class_name}: {len(test_files)}")
        print(f"Train files for class {class_name}: {len(train_files)}")

        # Add test files to test list
        for file in test_files:
            test.append([parse_sample(file_embed, file), class_name, 1])
        # Add train files to train list
        for file in train_files:
            train.append([parse_sample(file_embed, file), class_name, 1])

    print('################ Building non-embedded dataset ##################')
    
    # Build non-embedded dataset
    for class_name, files in class_files_noembed.items():
        print('Total non-embedded files:', len(files))
        test_size = int(0.2 * len(files))  # Calculate test set size
        print('Non-embedded test files:', test_size)

        # Split non-embedded files into train and test sets with 8:2 ratio
        test_files = files[:test_size]  # Get test file list
        train_files = files[test_size:]  # Get train file list

        # Add test files to test list
        for file in test_files:
            test.append([parse_sample(file_no_embed, file), 0, 0])

        # Add train files to train list
        for file in train_files:
            train.append([parse_sample(file_no_embed, file), 0, 0])

    print('################ Dataset building completed ##################')

    print("Number of training files:", len(train))
    print("Number of test files:", len(test))
    
    # Debug: Check label distribution before shuffling
    print(f"\n🔍 DEBUG: Label distribution before shuffling:")
    train_labels = [sample[2] for sample in train]  # Get the positive/negative labels
    test_labels = [sample[2] for sample in test]
    
    train_pos = sum(1 for label in train_labels if label == 1)
    train_neg = sum(1 for label in train_labels if label == 0)
    test_pos = sum(1 for label in test_labels if label == 1)
    test_neg = sum(1 for label in test_labels if label == 0)
    
    print(f"   - Training set: {train_pos} positive, {train_neg} negative")
    print(f"   - Test set: {test_pos} positive, {test_neg} negative")
    
    if train_neg == 0 or test_neg == 0:
        print(f"⚠️  CRITICAL: Missing negative samples! This will cause unbalanced dataset.")
        print(f"⚠️  Check if Cover/Raw directory exists and contains files.")

    print('################ Shuffling and splitting data and labels ##################')
    random.shuffle(train)
    random.shuffle(test)

    x_train, y_train, x_test, y_test = [], [], [], []

    # Split train dataset into x_train and y_train
    for sample in train:
        x_train.append(sample[0])  # Add sample features to x_train
        y_train.append(sample[1:])  # Add labels to y_train

    # Split test dataset into x_test and y_test
    for sample in test:
        x_test.append(sample[0])  # Add sample features to x_test
        y_test.append(sample[1:])  # Add labels to y_test

    # Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print("x_train.shape:", x_train.shape)  # Print training sample data shape
    print("y_train.shape", y_train.shape)  # Print training label data shape
    print("x_test.shape", x_test.shape)  # Print test sample data shape
    print("y_test:", y_test.shape)  # Print test label data shape

    print('Saving processed data to pkl file for next use.')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    
    # Save data to pkl file
    with open(pkl_file, 'wb') as f:
        pickle.dump((x_train, y_train, x_test, y_test), f)

    return x_train, y_train, x_test, y_test

def get_alter_loaders(args):
    """Get data loaders"""
    file_embed, file_no_embed, pkl_file = get_data_paths(args)
    
    if args.train_dataset == 'combined':
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"Combined dataset PKL file not found at: {pkl_file}")
        print(f"Loading combined dataset from saved pkl file: {pkl_file}")
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
            # Legacy SASM alg-labeled format: (features, labels, algorithm_labels)
            x, y, algo = data
            return x, y, x, y, algo, algo
        else:
            raise ValueError(f"Unsupported combined dataset PKL format at: {pkl_file}")
    
    # Try to load from pkl file first
    if os.path.exists(pkl_file):
        try:
            print(f"Loading dataset from saved pkl file: {pkl_file}")
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 6:
                x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test = data
            else:
                x_train, y_train, x_test, y_test = data
                algo_labels_train, algo_labels_test = None, None
            print("x_train.shape:", x_train.shape)
            print("y_train.shape", y_train.shape)
            print("x_test.shape", x_test.shape)
            print("y_test:", y_test.shape)
            return x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test
        except (ModuleNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Could not load pkl file '{os.path.basename(pkl_file)}'. Reason: {e}.")
            print("Will regenerate the dataset from source files.")

    # If pkl file doesn't exist or fails to load, regenerate it.
    result = _generate_dataset_from_sources(file_embed, file_no_embed, pkl_file, args)
    return result

# data transform
def convert_to_loader(x_train, y_train, x_test, y_test, algorithm_labels_train=None, algorithm_labels_test=None, batch_size=64):
    """Convert data to DataLoader with optional algorithm labels"""
    # Ensure numeric dtypes (object arrays -> float32/int64)
    try:
        x_train_np = np.asarray(x_train, dtype=np.float32)
        x_test_np = np.asarray(x_test, dtype=np.float32)
    except Exception:
        x_train_np = np.array(x_train, dtype=np.float32)
        x_test_np = np.array(x_test, dtype=np.float32)

    # y can be float (one-hot) or ints; coerce to float32 for generality
    try:
        y_train_np = np.asarray(y_train, dtype=np.float32)
        y_test_np = np.asarray(y_test, dtype=np.float32)
    except Exception:
        y_train_np = np.array(y_train, dtype=np.float32)
        y_test_np = np.array(y_test, dtype=np.float32)

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.from_numpy(x_train_np)
    y_train_tensor = torch.from_numpy(y_train_np)
    x_test_tensor = torch.from_numpy(x_test_np)
    y_test_tensor = torch.from_numpy(y_test_np)

    # Create training and test datasets
    if algorithm_labels_train is not None:
        algorithm_labels_train_tensor = torch.LongTensor(algorithm_labels_train)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor, algorithm_labels_train_tensor)
    else:
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        
    if algorithm_labels_test is not None:
        algorithm_labels_test_tensor = torch.LongTensor(algorithm_labels_test)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor, algorithm_labels_test_tensor)
    else:
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create training and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Custom multi-head attention module compatible with Hessian computation
class HessianCompatibleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)

        # Merge multiple heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        return self.w_o(attn_output)

# Custom transformer layer
class HessianCompatibleTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=256, dropout=0.1):
        super().__init__()

        self.self_attn = HessianCompatibleMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + residual connection
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network + residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# Modified PositionalEncoding compatible with Hessian computation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.5)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Get input sequence length
        seq_len = x.size(1)

        # Truncate positional encoding to match input sequence length
        pe = self.pe[:, :seq_len, :]

        # Add positional encoding to input tensor
        x = x + pe

        return x

# Define the PyTorch model
class Model1(nn.Module):
    def __init__(self, args):
        super(Model1, self).__init__()
        self.args = args
        
        self.embedding = nn.Embedding(256, args.d_model)
            
        self.position_embedding = PositionalEncoding(args.d_model, args.max_len)

        # Use custom transformer layers instead of nn.TransformerEncoder
        self.transformer_layers = nn.ModuleList([
            HessianCompatibleTransformerLayer(args.d_model, args.num_heads, args.d_ff, args.dropout)
            for _ in range(args.num_layers)
        ])

        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.long()
        emb_x = self.embedding(x)
        if emb_x.dim() == 4:
            emb_x = emb_x.mean(dim=2)
        elif emb_x.dim() != 3:
            raise ValueError(f"Unexpected embedding shape: {emb_x.shape}")

        # Add positional encoding
        emb_x = self.position_embedding(emb_x)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            emb_x = layer(emb_x)

        # Pooling
        outputs = self.pooling(emb_x.permute(0, 2, 1)).squeeze(2)

        return outputs

class Classifier1(nn.Module):
    def __init__(self, args):
        super(Classifier1, self).__init__()
        self.args = args
        self.model1 = Model1(args)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.d_model, args.num_class)

    def forward(self, x):
        x = self.model1(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)

def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args, target_loader=None):
    """Train the model. test_loader is validation (typically test split ∩ train_domains). target_loader optional."""
    from testing_utils import eval_tensor_loader_classification_accuracy

    best_acc = 0.0
    device = torch.device(args.device)

    # General logs
    gen_logs = {
        'epoch_loss': [],
        'epoch_acc': [],
        'val_acc': [],
        'target_acc': [],
        'lr': [],
        'domain_test_acc': [],  # List of per-epoch dicts: [{'QIM': 0.8, 'PMS': 0.7, ...}, ...]
    }

    print(f"Training on domains: {args.train_domains}, Testing on: {args.test_domains}", flush=True)
    
    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    if args.gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation: {args.gradient_accumulation_steps} steps", flush=True)
        print(f"Effective batch size: {effective_batch_size} (actual: {args.batch_size})", flush=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize gradients at the start of each epoch
        optimizer.zero_grad()
        
        batch_iter = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{args.epochs}',
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, batch_data in enumerate(batch_iter):
            # Handle different batch formats (with/without algorithm labels)
            if len(batch_data) == 3:
                inputs, labels, algorithm_labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                algorithm_labels = algorithm_labels.to(device)
            else:
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)
                algorithm_labels = None
            
            label_indices = labels.squeeze().long()
            
            
            if args.steg_algorithm == 'FS-MDP':
                # BCELoss expects single dimension targets (0 or 1)
                if labels.dim() == 2 and labels.size(1) == 2:
                    # One-hot encoded: extract positive class (index 1)
                    label_target = labels[:, 1].float().unsqueeze(1)  # Shape: (batch_size, 1)
                elif labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1):
                    # Already binary labels: use directly
                    label_target = labels.float().view(-1, 1)  # Shape: (batch_size, 1)
                else:
                    raise ValueError(f"Unexpected labels shape for FS_MDP: {labels.shape}")
            elif args.steg_algorithm == 'SFFN':
                # SFFN uses standard CrossEntropyLoss, expects class indices
                label_target = label_indices
            elif args.steg_algorithm == 'KFEF':
                # KFEF uses standard CrossEntropyLoss, expects class indices
                label_target = label_indices
            elif args.steg_algorithm == 'DVSF':
                # DVSF uses standard CrossEntropyLoss, expects class indices
                label_target = label_indices
            elif args.steg_algorithm == 'DAEF-VS':
                # DAEF-VS uses standard CrossEntropyLoss, expects class indices
                label_target = label_indices
            else:
                label_target = torch.eye(args.num_class).to(device)[label_indices].squeeze()

            # Build closures
            def hard_ce_closure(domain_mask=None, domain_id=None):
                optimizer.zero_grad()
                if domain_mask is not None:
                    x = inputs[domain_mask]
                    y = label_target[domain_mask]
                else:
                    x = inputs
                    y = label_target
                outputs = model(x) if domain_id is None else model(x, domain_id)
                loss = criterion(outputs, y)
                loss.backward()
                return loss

            # soft_ce_closure removed

            # Steganalysis-Aware Sharpness Minimization or standard optimization
            if _optimizer_is_sasm(optimizer):
                # Always use hard CE
                use_closure = hard_ce_closure
                
                # For Hybrid SFFN and KFEF, modify SASM to handle domain-specific parameters
                if args.steg_algorithm in ['HybridSFFN', 'KFEF']:
                    loss = _hybrid_sasm_step(optimizer, use_closure, algorithm_labels, model, inputs, label_target, criterion)
                else:
                    loss = optimizer.step(use_closure, algorithm_labels)
                
                # Get outputs for accuracy calculation
                outputs = model(inputs)
                
                # Print DGSAM statistics only once
                if (hasattr(optimizer, 'get_computation_stats') and 
                    not hasattr(optimizer, '_stats_printed')):
                    stats = optimizer.get_computation_stats()
                    if stats:
                        print(f"DGSAM: {stats['num_domains']} domains, "
                              f"{stats['total_gradient_computations']} grad computations, "
                              f"scaling factor: {stats['scaling_factor']:.4f}")
                        optimizer._stats_printed = True
            else:
                # Standard optimization with gradient accumulation
                outputs = model(inputs)
                loss = criterion(outputs, label_target)
                
                # Scale loss by accumulation steps
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                # Only update weights every N steps
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item() * inputs.size(0) * args.gradient_accumulation_steps

            # Calculate accuracy
            if args.steg_algorithm == 'FS-MDP':
                predicted = torch.round(outputs).squeeze()
                correct += (predicted == label_indices.float()).sum().item()
            else:
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == label_indices).sum().item()
            
            total += labels.size(0)
            
            # Collect domain performance for adaptive rho (only for DGSAM)
            if _optimizer_is_sasm(optimizer) and optimizer.adaptive_rho and algorithm_labels is not None:
                optimizer.collect_domain_performance(algorithm_labels, predicted, label_indices)
            
            # Periodic memory cleanup to prevent fragmentation
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update any remaining accumulated gradients
        if not _optimizer_is_sasm(optimizer):
            if (batch_idx + 1) % args.gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}", flush=True)
        gen_logs['epoch_loss'].append(float(epoch_loss))
        gen_logs['epoch_acc'].append(float(epoch_acc))
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation phase
        accuracy = eval_tensor_loader_classification_accuracy(model, test_loader, args, device)
        print(f"Validation Accuracy (test∩train_domains): {accuracy:.4f}")
        gen_logs['val_acc'].append(float(accuracy))

        if target_loader is not None and len(target_loader.dataset) > 0:
            t_acc = eval_tensor_loader_classification_accuracy(model, target_loader, args, device)
            if np.isnan(t_acc):
                print("Target domain Accuracy (test∩test_domains): n/a (empty or failed)")
                gen_logs['target_acc'].append(None)
            else:
                print(f"Target domain Accuracy (test∩test_domains): {t_acc:.4f}")
                gen_logs['target_acc'].append(float(t_acc))
        else:
            gen_logs['target_acc'].append(None)
        
        # Domain test evaluation (if enabled and at the right interval)
        if args.domain_test_interval > 0 and (epoch + 1) % args.domain_test_interval == 0:
            from testing_utils import compute_domain_test_acc
            embedding_str = str(args.embedding_rate)
            test_datasets = [
                f'QIM_{embedding_str}',
                f'PMS_{embedding_str}',
                f'LSB_{embedding_str}',
                f'AHCM_{embedding_str}'
            ]
            
            domain_test_acc = {}
            for dataset_name in test_datasets:
                domain_name = dataset_name.split('_')[0]  # Extract QIM, PMS, LSB, AHCM
                acc = compute_domain_test_acc(model, dataset_name, args)
                domain_test_acc[domain_name] = float(acc) if not np.isnan(acc) else 0.0
            
            gen_logs['domain_test_acc'].append(domain_test_acc)
        else:
            # Add empty dict for epochs without domain testing
            gen_logs['domain_test_acc'].append({})
        
        # Update learning rate scheduler
        # Note: DGSAM manages its own scheduler step internally. Standard optimizer steps below.
        
        # Adaptive rho adjustment, history recording, and LR printing
        if _optimizer_is_sasm(optimizer): # This handles SASM
            avg_grad_norm = optimizer.get_grad_norm()
            optimizer.record_epoch_stats(epoch, epoch_loss, epoch_acc, avg_grad_norm)

            if optimizer.adaptive_rho:
                old_rho = optimizer.current_rho
                optimizer.update_adaptive_rho(epoch, epoch_loss, avg_grad_norm)
                if abs(optimizer.current_rho - old_rho) > 1e-6:
                    print(f"Adaptive rho: {old_rho:.4f} -> {optimizer.current_rho:.4f}")
            
            # For SASM, the base optimizer's scheduler is managed internally or not used
            # in the same way. We print the base optimizer's LR if possible.
            if hasattr(optimizer.base_optimizer, 'param_groups'):
                 print(f"Learning Rate: {optimizer.base_optimizer.param_groups[0]['lr']:.6f}")

        else: # Standard optimizer
            scheduler.step()
            cur_lr = float(scheduler.get_last_lr()[0])
            print(f"Learning Rate: {cur_lr:.6f}")
            gen_logs['lr'].append(cur_lr)

        # Always save general logs & plots (with/without PCE)
        ds_id = args.dataset_id if args.dataset_id is not None else str(args.embedding_rate)
        # Clean ds_id for filename usage by taking only the basename
        ds_id_save = os.path.basename(ds_id).replace('.pkl', '')
        plot_dir = os.path.join(args.result_path, f'training_plots_{ds_id_save}')
        os.makedirs(plot_dir, exist_ok=True)
        # Save JSON
        with open(os.path.join(args.result_path, f'train_logs_{ds_id_save}.json'), 'w') as f:
            json.dump(gen_logs, f, indent=2)
        
        # Plot domain test accuracy curves
        from testing_utils import plot_domain_test_acc_curves
        plot_domain_test_acc_curves(gen_logs, args)
        # Plots
        if len(gen_logs['epoch_loss']) > 0:
            plt.figure(figsize=(6,4))
            xs = np.arange(1, len(gen_logs['epoch_loss'])+1)
            plt.plot(xs, gen_logs['epoch_loss'], 'b-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Train Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'train_loss_{ds_id_save}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        if len(gen_logs['epoch_acc']) > 0:
            plt.figure(figsize=(6,4))
            xs = np.arange(1, len(gen_logs['epoch_acc'])+1)
            plt.plot(xs, gen_logs['epoch_acc'], 'g-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Train Acc')
            plt.title('Training Accuracy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'train_acc_{ds_id_save}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        if len(gen_logs['val_acc']) > 0:
            plt.figure(figsize=(6,4))
            xs = np.arange(1, len(gen_logs['val_acc'])+1)
            plt.plot(xs, gen_logs['val_acc'], 'r-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Val Acc')
            plt.title('Validation Accuracy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'val_acc_{ds_id_save}.png'), dpi=150, bbox_inches='tight')
            plt.close()


        # Save best weights and write checkpoint info first
        is_best = accuracy > best_acc
        best_acc = max(accuracy, best_acc)
        if is_best and args.save_model:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'args': args,
            }, is_best, args.result_path, args)

            # Write checkpoint info to file first
            os.makedirs(args.result_path, exist_ok=True)
            result_filename = get_result_filename(args)
            with open(os.path.join(args.result_path, result_filename), 'a') as f:
                f.write("loaded best_checkpoint (epoch %d, best_acc %.4f)\n" % (epoch, best_acc))

        # Test on external datasets each epoch only if enabled
        # External eval every eval_step epochs (when enabled)
        if getattr(args, 'eval_step', 0) > 0 and ((epoch + 1) % args.eval_step == 0):
            from testing_utils import test_current_model
            test_current_model(model, args)
    
    # Training completed - generate analysis plots and save history
    if _optimizer_is_sasm(optimizer) and optimizer.adaptive_rho:
        # Generate training analysis plots
        plot_dir = os.path.join(args.result_path, 'training_plots')
        optimizer.generate_training_analysis_plots(plot_dir, args)
        
        # Save training history to JSON
        history_path = os.path.join(args.result_path, f'training_history_{ds_id_save}.json')
        optimizer.save_training_history(history_path, args)
        
        # Print final summary
        total_adjustments = len(optimizer.training_history['rho_adjustments'])
        final_rho = optimizer.current_rho
        print(f"\n=== DGSAM Adaptive Training Summary ===")
        print(f"Final rho: {final_rho:.4f} (range: {optimizer.rho_min}-{optimizer.rho_max})")
        print(f"Total rho adjustments: {total_adjustments}")
        print(f"Training plots saved to: {plot_dir}")
        print(f"Training history saved to: {history_path}")
        print("="*40)

def _get_base_name(args):
    """Helper function to create a base name for result files and models."""
    base_name = args.steg_algorithm
    # if args.train_dataset == 'combined':
    #     base_name += f"_{args.dataset_id}"
    # else:
    #     base_name += f"_{args.train_dataset.lower()}"
    
    # # Add sample length for uniqueness, e.g., '1s' from '1000ms'
    # base_name += f"_{int(args.sample_length / 1000)}s"

    # Add domain generalization info
    train_domain_names = '_'.join(sorted(set(args.train_domains.split(','))))
    test_domain_names = '_'.join(sorted(set(args.test_domains.split(','))))
    base_name += f"_{train_domain_names}_to_{test_domain_names}"
    
    return base_name

def get_model_filename(args):
    """Generate model filename with simplified SASM naming."""
    base_name = _get_base_name(args)
    
    if args.use_sasm:
        return f'model_best_sasm_{base_name}.pth.tar'
    else:
        return f'model_best_{base_name}.pth.tar'

def get_result_filename(args):
    """Generate result filename using unified naming across algorithms."""
    from utils.naming import get_result_filename as _unified_get_result_filename
    return _unified_get_result_filename(args)

def save_checkpoint(state, is_best, result_path, args):
    """Save model checkpoint."""
    if is_best:
        os.makedirs(result_path, exist_ok=True)
        model_filename = get_model_filename(args)
        model_path = os.path.join(result_path, model_filename)
        torch.save(state, model_path)
        print(f'Saved best checkpoint: {model_path}')

def ccn_main(args, ccn_model_path): # Pass the determined path
    # Thin shim: delegate to CCN runner to keep this file lean
    from models_collection.CCN.runner import run_ccn_domain_generalization
    run_ccn_domain_generalization(args)

def ss_qccn_main(args, model_path): # Pass the determined path
    # Thin shim: delegate to SS-QCCN runner
    from models_collection.SS_QCCN.runner import run_ss_qccn_domain_generalization
    run_ss_qccn_domain_generalization(args)

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    if not getattr(args, 'data_root', None):
        args.data_root = DEFAULT_TRAIN_DATA_ROOT

    # Auto-detect combined dataset when dataset_id is provided
    if args.dataset_id is not None:
        args.train_dataset = 'combined'
        print("Info: dataset_id specified, automatically setting train_dataset='combined'")

    # --- Dynamic Path Modification ---
    base_result_path = args.result_path

    # Append algorithm-specific directory to the path
    args.result_path = os.path.join(base_result_path, args.steg_algorithm)
    print(f"INFO: Final result path set to: {args.result_path}")
    # --- End of Dynamic Path Modification ---

    # If train_label_csv is not provided, generate a default path
    if args.train_label_csv is None:
        # For datasets generated from source, assume label file is inside the dataset-specific folder
        if args.train_dataset != 'combined':
            dataset_dir = os.path.join(args.data_root, f'{args.train_dataset}_{str(args.embedding_rate)}')
            args.train_label_csv = os.path.join(dataset_dir, 'train_label.csv')
            print(f"Info: --train_label_csv not specified, defaulting to '{args.train_label_csv}'")

    # Set GPU
    set_gpu(args.gpu)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Print arguments
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Handle CCN algorithm separately
    if args.steg_algorithm == 'CCN':
        ccn_main(args, args.result_path) # Pass the final determined path
        return
        
    if args.steg_algorithm == 'SS-QCCN':
        ss_qccn_main(args, args.result_path) # Pass the final determined path
        return

    # Delegate FS_MDP domain generalization to its runner when using combined dataset
    if args.steg_algorithm == 'FS-MDP' and args.train_dataset == 'combined':
        from models_collection.FS_MDP.runner import run_fs_mdp_domain_generalization
        run_fs_mdp_domain_generalization(args)
        return
    
    # Delegate LStegT domain generalization to its runner when using combined dataset
    if args.steg_algorithm == 'LStegT' and args.train_dataset == 'combined':
        from models_collection.LStegT.runner import run_lsegt_domain_generalization
        run_lsegt_domain_generalization(args)
        return

    # Delegate KFEF domain generalization to its runner when using combined dataset
    if args.steg_algorithm == 'KFEF' and args.train_dataset == 'combined':
        from models_collection.KFEF.runner import run_kfef_domain_generalization
        run_kfef_domain_generalization(args)
        return

    # Delegate Transformer domain generalization to its runner when using combined dataset
    if args.steg_algorithm == 'Transformer' and args.train_dataset == 'combined':
        from models_collection.Transformer.runner import run_transformer_domain_generalization
        run_transformer_domain_generalization(args)
        return

    # Delegate SFFN domain generalization to its runner when using combined dataset
    if args.steg_algorithm == 'SFFN' and args.train_dataset == 'combined':
        from models_collection.SFFN.runner import run_sffn_domain_generalization
        run_sffn_domain_generalization(args)
        return

    # Delegate DVSF domain generalization to its runner when using combined dataset
    if args.steg_algorithm == 'DVSF' and args.train_dataset == 'combined':
        from models_collection.DVSF.runner import run_dvsf_domain_generalization
        run_dvsf_domain_generalization(args)
        return

    # Delegate DAEF-VS domain generalization to its runner when using combined dataset
    if args.steg_algorithm == 'DAEF-VS' and args.train_dataset == 'combined':
        from models_collection.DAEF_VS.runner import run_daef_vs_domain_generalization
        run_daef_vs_domain_generalization(args)
        return

    # Test only mode
    if args.test_only:
        from testing_utils import test_external_datasets
        test_external_datasets(args)
        return
    
    # Load data
    print(f'Loading {args.train_dataset} dataset...')
    x_train, y_train, x_test, y_test, algorithm_labels_train, algorithm_labels_test = get_alter_loaders(args)
    
    # Parse domains and remove duplicates
    train_domain_names = sorted(set(args.train_domains.split(',')))
    test_domain_names = sorted(set(args.test_domains.split(',')))
    train_domain_ids = [DOMAIN_MAP.get(name, -1) for name in train_domain_names]
    test_domain_ids = [DOMAIN_MAP.get(name, -1) for name in test_domain_names]
    # Remove invalid IDs and warn
    train_domain_ids = [id for id in train_domain_ids if id != -1]
    test_domain_ids = [id for id in test_domain_ids if id != -1]
    if len(train_domain_ids) == 0 or len(test_domain_ids) == 0:
        raise ValueError("No valid domains specified after mapping.")

    # Filter training data based on algo_labels (numpy array)
    if algorithm_labels_train is not None:
        train_mask = np.isin(algorithm_labels_train, train_domain_ids)
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]
        algorithm_labels_train = algorithm_labels_train[train_mask]
    else:
        # If no algorithm labels, assume all data belongs to the requested domains
        # or that filtering is not possible this way.
        print("Warning: No algorithm labels found for training data. Skipping domain filtering.")

    # Filter testing data
    if algorithm_labels_test is not None:
        test_mask = np.isin(algorithm_labels_test, test_domain_ids)
        x_test = x_test[test_mask]
        y_test = y_test[test_mask]
        algorithm_labels_test = algorithm_labels_test[test_mask]
    else:
        print("Warning: No algorithm labels found for testing data. Skipping domain filtering.")

    # Check empty and balance
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Filtered dataset is empty. Check domain specifications.")
    # Warn if imbalanced (视情况添加)
    train_steg_ratio = np.mean(y_train[:,1]) if len(y_train) > 0 else 0
    test_steg_ratio = np.mean(y_test[:,1]) if len(y_test) > 0 else 0
    print(f"Filtered train samples: {len(x_train)} (steg ratio: {train_steg_ratio:.2f})")
    print(f"Filtered test samples: {len(x_test)} (steg ratio: {test_steg_ratio:.2f})")
    if abs(train_steg_ratio - 0.5) > 0.1:
        print("Warning: Training set steg/cover imbalance after filtering.")
    
    # Data preprocessing
    if args.steg_algorithm == 'FS-MDP':
        print("Converting data to one-hot encoding for FS-MDP.")
        from testing_utils import transfer_to_onehot
        x1_train = transfer_to_onehot(x_train)
        x1_test = transfer_to_onehot(x_test)
    else:
        x1_train = x_train[:, :, 0:7]
        x1_test = x_test[:, :, 0:7]
        # Replace all -1 values with 200 in training data
        x1_train = np.where(x1_train == -1, 200, x1_train)
        x1_test = np.where(x1_test == -1, 200, x1_test)

    print(f"Training data shape: {x1_train.shape}")
    
    # Label processing
    y1_train = y_train[:, 1:]
    y1_test = y_test[:, 1:]
    
    if args.steg_algorithm != 'FS-MDP':
        print(f"Data range: min={x1_train.min()}, max={x1_train.max()}")
        print(f"Has negative values: {(x1_train < 0).any()}")
        print(f"Exceeds 255: {(x1_train > 255).any()}")

    # Create data loaders
    train_loader, test_loader = convert_to_loader(x1_train, y1_train, x1_test, y1_test, 
                                                  algorithm_labels_train, algorithm_labels_test, args.batch_size)

    # Initialize model, optimizer, criterion
    # model = Classifier1(args).to(device) # Old model instantiation
    
    # Dynamic model loading based on algorithm selection
    if args.steg_algorithm == 'Transformer':
        from models_collection.Transformer.transformer import Classifier1
        model = Classifier1(args).to(device)
        print("Using Transformer model architecture.")
    elif args.steg_algorithm == 'LStegT':
        from models_collection.LStegT.lsegt import Classifier1 as LStegT_Classifier
        model = LStegT_Classifier(args).to(device)
        print("Using LStegT model architecture.")
    elif args.steg_algorithm == 'FS-MDP':
        from models_collection.FS_MDP.fs_mdp import FS_MDP_Wrapper
        model = FS_MDP_Wrapper(args).to(device)
        print("Using FS-MDP model architecture.")
    elif args.steg_algorithm == 'KFEF':
        from models_collection.KFEF.kfef import KFEFClassifier
        model = KFEFClassifier(args).to(device)
        print("Using KFEF model architecture (baseline-compatible).")
        model.count_parameters()
    elif args.steg_algorithm == 'DVSF':
        from models_collection.DVSF.dvsf import DVSFClassifier
        model = DVSFClassifier(args).to(device)
        print("Using DVSF model architecture.")
    elif args.steg_algorithm == 'DAEF-VS':
        from models_collection.DAEF_VS.daef_vs import DAEF_VS_Classifier
        model = DAEF_VS_Classifier(args).to(device)
        print("Using DAEF-VS model architecture.")
    else:
        raise ValueError(f"Unsupported steg_algorithm: {args.steg_algorithm}")
    
    # Initialize optimizer
    if args.use_sasm:
        if SASM_IMPORT_ERROR is not None:
            raise ImportError(
                "SASM support requires the optional 'sasm' module, which is not present in this repository."
            ) from SASM_IMPORT_ERROR
        print(f"Using SASM optimizer")
        optimizer = AlgorithmAwareSAM(
            model.parameters(),
            base_optimizer=AdamW,
            lr=args.lr,
            rho=args.rho,
            adaptive=args.adaptive_rho,
            weight_decay=args.weight_decay,
            adaptive_rho=args.adaptive_rho,
            rho_min=args.rho_min,
            rho_max=args.rho_max
        )
    else:
        print("Using standard optimizer")
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    print(f"Using CosineAnnealingLR scheduler with T_max={args.epochs}, eta_min=1e-6")
    
    if args.steg_algorithm == 'FS-MDP':
        criterion = nn.BCELoss()
        print("Using BCELoss for FS-MDP.")
    elif args.steg_algorithm == 'SFFN':
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for SFFN (single-stream architecture).")
    elif args.steg_algorithm == 'DVSF':
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for DVSF.")
    elif args.steg_algorithm == 'DAEF-VS':
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss for DAEF-VS.")
    else:
        criterion = nn.CrossEntropyLoss()

    # Train model
    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, args)

if __name__ == '__main__':
    main()
