import os
import shutil
import pickle
from typing import Tuple

import numpy as np
from tqdm import tqdm

from models_collection.common.domains import DOMAIN_MAP, parse_domain_names_to_ids
from models_collection.common.run_naming import build_run_tag
from models_collection.CCN.trainer import train_and_test_ccn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _write_temp_split(base_dir: str, x: np.ndarray, y: np.ndarray) -> Tuple[str, str]:
    steg_dir = os.path.join(base_dir, 'Steg')
    cover_dir = os.path.join(base_dir, 'Cover')
    _ensure_clean_dir(steg_dir)
    _ensure_clean_dir(cover_dir)

    for i, (sample, label_info) in enumerate(tqdm(zip(x, y), total=len(x), desc='Writing temp files')):
        label = int(label_info[1])
        target_dir = steg_dir if label == 1 else cover_dir
        file_path = os.path.join(target_dir, f'sample_{i}.txt')
        with open(file_path, 'w') as f:
            for row in sample:
                if len(row) >= 2:
                    f.write(f"{int(row[0])} {int(row[1])}\n")

    return steg_dir, cover_dir


def run_ccn_domain_generalization(args) -> None:
    """Run CCN with source-domain training and target-domain testing.

    - Requires combined_multi six-tuple PKL: (x_tr, y_tr, x_te, y_te, algo_tr, algo_te)
    - Uses args.train_domains / args.test_domains for domain filtering
    - Writes filtered splits to CCN-local temp directories, then trains/tests CCN
    - Saves model artifacts under CCN-local models directory, isolated by domain tuple
    - Optional external testing can be triggered via args.eval_step (>0)
    """
    # Resolve PKL path and load
    pkl_path = os.path.join(args.data_root, f'{args.dataset_id}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'Combined PKL not found: {pkl_path}')

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not (isinstance(data, tuple) and len(data) == 6):
        raise ValueError('Combined PKL must be a six-tuple: (x_tr,y_tr,x_te,y_te,algo_tr,algo_te).')

    x_tr, y_tr, x_te, y_te, algo_tr, algo_te = data

    # Parse and map domains
    train_ids = parse_domain_names_to_ids(args.train_domains)
    test_ids = parse_domain_names_to_ids(args.test_domains)
    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError('No valid domains resolved from train_domains/test_domains.')

    run_tag = build_run_tag(args)

    # Filter by domains on existing splits
    tr_mask = np.isin(algo_tr, np.array(train_ids))
    te_mask = np.isin(algo_te, np.array(test_ids))

    x_tr_f, y_tr_f = x_tr[tr_mask], y_tr[tr_mask]
    x_te_f, y_te_f = x_te[te_mask], y_te[te_mask]

    if len(x_tr_f) == 0 or len(x_te_f) == 0:
        raise ValueError('Filtered dataset is empty after domain selection.')

    # CCN-local directories
    ccn_root = os.environ.get(
        'DASM_CCN_ROOT',
        os.path.join(PROJECT_ROOT, 'models_collection', 'CCN'),
    )
    models_dir = os.path.join(ccn_root, 'models', run_tag)
    temp_train_dir = os.path.join(ccn_root, 'temp', run_tag)
    temp_test_dir = os.path.join(ccn_root, 'temp_test', run_tag)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(temp_train_dir), exist_ok=True)
    os.makedirs(os.path.dirname(temp_test_dir), exist_ok=True)

    # Materialize filtered splits as CCN txt format
    train_steg, train_cover = _write_temp_split(temp_train_dir, x_tr_f, y_tr_f)
    test_steg, test_cover = _write_temp_split(temp_test_dir, x_te_f, y_te_f)

    # Train and test CCN
    print('Running CCN with domain generalization (source -> target)...')
    train_and_test_ccn(train_steg, train_cover, test_steg, test_cover, models_dir)

    # Always test on individual domains and save results
    print('Testing CCN on individual domains...')
    # Set result_path to models_dir so results are saved alongside the model
    original_result_path = getattr(args, 'result_path', None)
    args.result_path = models_dir
    from testing_utils import test_ccn_model
    test_ccn_model(models_dir, args)
    # Restore original result_path if it was set
    if original_result_path is not None:
        args.result_path = original_result_path

