import argparse
import os
import pickle
import sys
import numpy as np

# Ensure project root is on sys.path for testing_utils import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from testing_utils import transfer_to_onehot


def _onehot_in_chunks(x, chunk_size):
    outputs = []
    for i in range(0, len(x), chunk_size):
        outputs.append(transfer_to_onehot(x[i:i + chunk_size]))
    return np.concatenate(outputs, axis=0) if outputs else np.array([])


def main():
    parser = argparse.ArgumentParser(description="Build one-hot cached PKL for FS-MDP.")
    parser.add_argument("--src_pkl", required=True, help="Source combined PKL (6-tuple) path")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: FS_MDP/onehot_cache)")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for one-hot conversion")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(script_dir, "onehot_cache")
    os.makedirs(out_dir, exist_ok=True)

    src_name = os.path.basename(args.src_pkl)
    if src_name.endswith(".pkl"):
        dst_name = src_name.replace(".pkl", "_onehot.pkl")
    else:
        dst_name = f"{src_name}_onehot.pkl"
    dst_path = os.path.join(out_dir, dst_name)

    with open(args.src_pkl, "rb") as f:
        data = pickle.load(f)
    if not (isinstance(data, tuple) and len(data) == 6):
        raise ValueError("Source PKL must be 6-tuple: (x_tr,y_tr,x_te,y_te,algo_tr,algo_te)")

    x_tr, y_tr, x_te, y_te, algo_tr, algo_te = data

    x_tr_oh = _onehot_in_chunks(x_tr, args.chunk_size)
    x_te_oh = _onehot_in_chunks(x_te, args.chunk_size)

    with open(dst_path, "wb") as f:
        pickle.dump((x_tr_oh, y_tr, x_te_oh, y_te, algo_tr, algo_te), f)

    print(f"Saved one-hot PKL: {dst_path}")
    print(f"x_tr_oh: {x_tr_oh.shape}, x_te_oh: {x_te_oh.shape}")


if __name__ == "__main__":
    main()

