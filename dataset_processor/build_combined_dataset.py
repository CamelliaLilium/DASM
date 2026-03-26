import os
import argparse
import pickle
import random
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined multi-domain dataset (standard and SASM variants)")
    parser.add_argument("--data_root", type=str, default=os.environ.get("DASM_DATA_ROOT", os.path.join(PROJECT_ROOT, "dataset", "model_train")),
                        help="Root directory containing single-domain datasets")
    parser.add_argument("--embedding_rate", type=float, required=True,
                        help="Embedding rate of datasets to combine, e.g., 0.1")
    parser.add_argument("--domains", type=str, default="QIM,PMS,LSB,AHCM",
                        help="Comma-separated domain names to include (order defines domain_id mapping)")
    parser.add_argument("--target_per_domain", type=int, required=True,
                        help="Total target samples to draw per domain (train+test), must be even (strict 1:1 steg/cover)")
    parser.add_argument("--seq_len", type=int, default=100,
                        help="Expected sequence length (number of rows per txt)")
    parser.add_argument("--sample_length_ms", type=int, default=1000,
                        help="Sample length in milliseconds for compatibility suffix (e.g., 1000 -> 1s)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (defaults to <data_root>/combined_multi)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train split ratio for standard dataset")
    parser.add_argument("--min_samples_per_bucket", type=int, default=1,
                        help="Minimum samples required per (domain,label) bucket; otherwise skip splitting for that bucket")
    return parser.parse_args()


def domain_id_map_from_list(domains: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(domains)}


def parse_txt_file(file_path: str) -> np.ndarray:
    with open(file_path, "r") as f:
        lines = f.readlines()
    sample = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        sample.append([int(x) for x in parts])
    return np.asarray(sample, dtype=np.int32)


def _find_single_pkl(data_root: str, domain: str, er: float, sample_length_ms: int) -> str:
    """Locate a single-domain PKL robustly under {data_root}/{domain}_{er}/pklfile.

    Preferred canonical name: {alg_lower}_{er}.pkl
    Also supports legacy names like *_1s_all.pkl, *_all.pkl.
    """
    base_dir = os.path.join(data_root, f"{domain}_{er}", "pklfile")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"pklfile dir not found: {base_dir}")
    candidates = []
    er_strs = [str(er), f"{er:.1f}"]
    alg_lower = domain.lower()
    for ers in er_strs:
        # Preferred modern naming
        candidates.append(os.path.join(base_dir, f"{alg_lower}_{ers}.pkl"))
        # Legacy names
        candidates.append(os.path.join(base_dir, f"{alg_lower}_{ers}_1s_all.pkl"))
        candidates.append(os.path.join(base_dir, f"{alg_lower}_{ers}_all.pkl"))
    # Fallback: any pkl in dir
    if not any(os.path.exists(p) for p in candidates):
        for fn in sorted(os.listdir(base_dir)):
            if fn.endswith('.pkl'):
                candidates.append(os.path.join(base_dir, fn))
                break
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No PKL found under {base_dir} for {domain} {er}")


def _load_single_dataset_as_pool(pkl_path: str, domain_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load single-dataset PKL (4 or 6 tuple) and return merged pools: features, labels([domain_id,steg]), algo_labels."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    def _normalize_labels_to_domain_steg(y: np.ndarray, dom_id: int) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 1:
            steg = y.astype(int)
        elif y.ndim == 2:
            if y.shape[1] >= 2:
                steg = y[:, -1].astype(int)
            elif y.shape[1] == 1:
                steg = y[:, 0].astype(int)
            else:
                raise ValueError(f"Empty label array with shape {y.shape}")
        else:
            raise ValueError(f"Unsupported label shape: {y.shape}")
        n = steg.shape[0]
        return np.stack([np.full((n,), dom_id, dtype=int), steg], axis=1)
    if isinstance(data, tuple) and len(data) == 6:
        x_tr, y_tr, x_te, y_te, a_tr, a_te = data
        feats = np.concatenate([x_tr, x_te], axis=0)
        labs_concat = np.concatenate([y_tr, y_te], axis=0)
        algs = np.concatenate([a_tr, a_te], axis=0)
        labs = _normalize_labels_to_domain_steg(labs_concat, domain_id)
        return feats, labs, algs
    elif isinstance(data, tuple) and len(data) == 4:
        x_tr, y_tr, x_te, y_te = data
        feats = np.concatenate([x_tr, x_te], axis=0)
        labs_concat = np.concatenate([y_tr, y_te], axis=0)
        labs = _normalize_labels_to_domain_steg(labs_concat, domain_id)
        algs = np.full((feats.shape[0],), domain_id, dtype=int)
        return feats, labs, algs
    else:
        raise ValueError(f"Unsupported single-dataset PKL format at {pkl_path}")


def _balanced_sample_indices_by_label(labels_bin: np.ndarray, target_each: int, seed: int) -> np.ndarray:
    """Return indices balanced by binary label (0/1)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels_bin == 1)[0]
    neg_idx = np.where(labels_bin == 0)[0]
    if len(pos_idx) < target_each or len(neg_idx) < target_each:
        raise ValueError(f"Insufficient samples for strict balance: pos={len(pos_idx)}, neg={len(neg_idx)}, need {target_each} each")
    sel_pos = rng.choice(pos_idx, target_each, replace=False)
    sel_neg = rng.choice(neg_idx, target_each, replace=False)
    return np.sort(np.concatenate([sel_pos, sel_neg], axis=0))


def stratified_split_by_domain_label(features: np.ndarray, labels: np.ndarray, algo_labels: np.ndarray,
                                     train_ratio: float, min_samples_per_bucket: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 按 (domain_id, binary_label) 进行分桶后分别 8:2 划分，再合并
    idx_all = np.arange(len(features))
    buckets = {}
    for idx in idx_all:
        dom = int(algo_labels[idx])
        bin_label = int(labels[idx][1])
        buckets.setdefault((dom, bin_label), []).append(idx)

    train_idx, test_idx = [], []
    for key, id_list in buckets.items():
        if len(id_list) < max(2, min_samples_per_bucket):
            # 样本太少，全部放入训练集，避免过度切分
            train_idx.extend(id_list)
            continue
        random.shuffle(id_list)
        split = int(train_ratio * len(id_list))
        train_idx.extend(id_list[:split])
        test_idx.extend(id_list[split:])

    x_train = features[train_idx]
    y_train = labels[train_idx]
    x_test = features[test_idx] if len(test_idx) > 0 else np.empty((0,), dtype=object)
    y_test = labels[test_idx] if len(test_idx) > 0 else np.empty((0, 2), dtype=np.int64)
    return x_train, y_train, x_test, y_test


def main():
    args = parse_args()
    random.seed(42)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    dom2id = domain_id_map_from_list(domains)

    # Name combined dataset by its composing single datasets (domains) and embedding rate
    er_str = f"{args.embedding_rate:.1f}" if isinstance(args.embedding_rate, float) else str(args.embedding_rate)
    dataset_id = f"{'+'.join(domains)}_{er_str}"
    out_dir = args.output_dir or os.path.join(args.data_root, "combined_multi")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Domains: {domains} -> {dom2id}")
    print(f"Embedding rate: {args.embedding_rate}")
    print(f"Seq len (strict): {args.seq_len}")
    print(f"Output dir: {out_dir}")

    all_features: List[np.ndarray] = []
    all_labels: List[List[int]] = []
    all_algo_labels: List[int] = []

    if args.target_per_domain % 2 != 0:
        raise ValueError("--target_per_domain must be even to enforce strict 1:1 steg/cover")
    each_label = args.target_per_domain // 2

    for idx, domain in enumerate(tqdm(domains, desc="Sampling per domain")):
        domain_id = dom2id[domain]
        single_pkl = _find_single_pkl(args.data_root, domain, args.embedding_rate, args.sample_length_ms)
        feats, labs, algs = _load_single_dataset_as_pool(single_pkl, domain_id)
        # Filter by expected seq_len (strict)
        keep = np.array([arr.shape[0] == args.seq_len for arr in feats])
        feats = feats[keep]
        labs = labs[keep]
        algs = algs[keep]
        sel_idx = _balanced_sample_indices_by_label(labs[:, 1], each_label, seed=42 + idx)
        all_features.extend(list(feats[sel_idx]))
        all_labels.extend(list(labs[sel_idx]))
        all_algo_labels.extend(list(algs[sel_idx]))

    if len(all_features) == 0:
        print("[ERROR] No samples collected. Abort.")
        return

    features = np.array(all_features, dtype=object)  # 保持原始列数；训练阶段自行取前7列
    labels = np.array(all_labels, dtype=np.int64)
    algo_labels = np.array(all_algo_labels, dtype=np.int64)

    # 分桶分层切分并输出 6 元组（覆盖标准版文件名）
    sample_len_suffix = f"_{int(args.sample_length_ms/1000)}s"
    x_train, y_train, x_test, y_test = stratified_split_by_domain_label(
        features, labels, algo_labels, args.train_ratio, args.min_samples_per_bucket
    )
    # Derive algo_labels splits according to the same indices via recomputation
    # Build index arrays by reconstructing bucket logic
    idx_all = np.arange(len(features))
    # Rebuild buckets
    buckets = {}
    for idx_i in idx_all:
        dom = int(algo_labels[idx_i])
        bin_label = int(labels[idx_i][1])
        buckets.setdefault((dom, bin_label), []).append(idx_i)
    train_idx, test_idx = [], []
    for key, id_list in buckets.items():
        if len(id_list) < max(2, args.min_samples_per_bucket):
            train_idx.extend(id_list)
            continue
        random.shuffle(id_list)
        split = int(args.train_ratio * len(id_list))
        train_idx.extend(id_list[:split])
        test_idx.extend(id_list[split:])
    algo_labels = np.array(all_algo_labels, dtype=np.int64)
    algo_train = algo_labels[train_idx]
    algo_test = algo_labels[test_idx]

    p_std = os.path.join(out_dir, f"{dataset_id}{sample_len_suffix}.pkl")
    with open(p_std, "wb") as f:
        pickle.dump((x_train, y_train, x_test, y_test, algo_train, algo_test), f)
    print(f"Saved (6-tuple): {p_std}")


if __name__ == "__main__":
    main()

