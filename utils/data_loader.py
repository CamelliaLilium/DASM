"""
Unified data loader utilities for combined dataset loading.
Only supports combined datasets loaded from PKL files.
"""
import os
import pickle


def get_combined_dataset_path(data_root, dataset_id, sample_length=1000):
    """
    Get the path to combined dataset PKL file.
    
    Args:
        data_root: Root directory containing dataset PKL files
        dataset_id: Dataset identifier (e.g., 'QIM+PMS+LSB+AHCM_0.5_1s')
        sample_length: Sample length in milliseconds (default: 1000)
    
    Returns:
        str: Full path to the PKL file
    """
    if not dataset_id:
        raise ValueError("dataset_id must be provided for combined dataset")
    
    sample_len_str = f"_{int(sample_length / 1000)}s"
    pkl_dir = os.path.join(data_root)
    
    # Accept full filename (with .pkl) to avoid inference
    if dataset_id.endswith('.pkl'):
        pkl_file = os.path.join(pkl_dir, dataset_id)
    else:
        base_pkl_name = f"{dataset_id}{sample_len_str}"
        pkl_file = os.path.join(pkl_dir, f"{base_pkl_name}.pkl")
    
    return pkl_file


def load_combined_dataset(data_root, dataset_id, sample_length=1000):
    """
    Load combined dataset from PKL file.
    
    Args:
        data_root: Root directory containing dataset PKL files
        dataset_id: Dataset identifier (e.g., 'QIM+PMS+LSB+AHCM_0.5_1s')
        sample_length: Sample length in milliseconds (default: 1000)
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test, algo_labels_train, algo_labels_test)
               algo_labels may be None if not present in PKL file
    """
    pkl_file = get_combined_dataset_path(data_root, dataset_id, sample_length)
    
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
        # Legacy format: (features, labels, algorithm_labels)
        x, y, algo = data
        return x, y, x, y, algo, algo
    else:
        raise ValueError(f"Unsupported combined dataset PKL format at: {pkl_file}")














