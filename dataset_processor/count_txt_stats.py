import argparse
import os
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_tqdm():
    """优先使用 tqdm，若不可用则回退为普通迭代器。"""
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        def _noop_iter(iterable, *_, **__):
            return iterable
        return _noop_iter


def count_txt_files_in_directory(target_directory: Path) -> int:
    """递归统计目录中 .txt 文件数量。目录不存在时返回 0。"""
    if not target_directory.exists() or not target_directory.is_dir():
        return 0
    return sum(1 for p in target_directory.rglob("*.txt") if p.is_file())


def generate_pairs(steg_domains: List[str], embedding_rates: List[str]) -> List[str]:
    return [f"{d}_{r}" for d in steg_domains for r in embedding_rates]


def collect_counts(base_dir: Path,
                   steg_domains: List[str],
                   embedding_rates: List[str]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    tqdm = _get_tqdm()
    for pair in tqdm(generate_pairs(steg_domains, embedding_rates), desc="统计中", leave=False):
        pair_dir = base_dir / pair
        cover_dir = pair_dir / "Cover"
        steg_dir = pair_dir / "Steg"
        cover_count = count_txt_files_in_directory(cover_dir)
        steg_count = count_txt_files_in_directory(steg_dir)
        results.append({
            "pair": pair,
            "cover_count": cover_count,
            "steg_count": steg_count,
            "cover_exists": cover_dir.exists(),
            "steg_exists": steg_dir.exists(),
        })
    return results


def print_table(results: List[Dict[str, object]]) -> None:
    header = (
        f"{'组合':<16} {'cover数':>8} {'steg数':>8} "
        f"{'cover在否':>10} {'steg在否':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        pair = str(item["pair"])  # type: ignore
        cover_count = int(item["cover_count"])  # type: ignore
        steg_count = int(item["steg_count"])  # type: ignore
        cover_exists = "Y" if bool(item["cover_exists"]) else "N"  # type: ignore
        steg_exists = "Y" if bool(item["steg_exists"]) else "N"  # type: ignore
        print(f"{pair:<16} {cover_count:>8} {steg_count:>8} {cover_exists:>10} {steg_exists:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 cover 与 steg 子目录下 .txt 文件数量")
    parser.add_argument(
        "--base",
        default=os.environ.get("DASM_DATA_ROOT", os.path.join(PROJECT_ROOT, "dataset", "model_train")),
        help="基础目录，包含 {steg_domains}_{embedding_rate} 结构的子目录",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=["QIM", "PMS", "LSB", "AHCM"],
        help="steg_domains 列表",
    )
    parser.add_argument(
        "--rates",
        nargs="*",
        default=[
            "0.1", "0.2", "0.3", "0.4", "0.5",
            "0.6", "0.7", "0.8", "0.9", "1.0",
        ],
        help="embedding_rate 列表",
    )
    args = parser.parse_args()

    base_dir = Path(os.path.abspath(args.base))
    results = collect_counts(base_dir, list(args.domains), list(args.rates))
    print_table(results)


if __name__ == "__main__":
    main()
