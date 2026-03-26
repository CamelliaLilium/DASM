import os
import glob
import pickle
from typing import Iterable, List, Tuple
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iter_pkl_paths(root: str) -> List[str]:
    patterns = [
        os.path.join(root, "**", "pklfile", "*.pkl"),
        os.path.join(root, "**", "pkl", "*.pkl"),
    ]
    seen = set()
    paths: List[str] = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if p not in seen:
                seen.add(p)
                paths.append(p)
    paths.sort()
    return paths


def normalize_tuple(data):
    # Returns (x_train, y_train, x_test, y_test) or None if unsupported
    if isinstance(data, tuple) and len(data) == 6:
        xtr, ytr, xte, yte, _, _ = data
        return xtr, ytr, xte, yte
    if isinstance(data, tuple) and len(data) == 4:
        return data
    if isinstance(data, tuple) and len(data) == 3:
        x, y, _ = data
        return x, y, x, y
    return None


def count_label(seq) -> Tuple[int, int, int]:
    # Interpret labels where the last element per label is 0/1 -> cover/steg
    cover = steg = unknown = 0
    try:
        for item in seq:
            v = item
            if isinstance(v, (list, tuple)) and len(v) > 0:
                v = v[-1]
            if v == 0:
                cover += 1
            elif v == 1:
                steg += 1
            else:
                unknown += 1
    except Exception:
        unknown = len(seq)
    return cover, steg, unknown


def scan(root: str = os.environ.get("DASM_ROOT", PROJECT_ROOT)) -> None:
    paths = iter_pkl_paths(root)
    print(f"files={len(paths)}")
    print("path,total,train,test,cover,steg,unknown")
    Ttot=Ttr=Tte=Tcv=Tst=Tunk=0
    for p in tqdm(paths, desc="scan pkl", dynamic_ncols=True):
        try:
            with open(p, 'rb') as f:
                data = pickle.load(f)
            tpl = normalize_tuple(data)
            if tpl is None:
                continue
            xtr, ytr, xte, yte = tpl
            Ntr = len(xtr)
            Nte = len(xte)
            c1,s1,u1 = count_label(ytr)
            c2,s2,u2 = count_label(yte)
            cover = c1 + c2
            steg  = s1 + s2
            unk   = u1 + u2
            total = Ntr + Nte
            print(f"{p},{total},{Ntr},{Nte},{cover},{steg},{unk}")
            Ttot+=total; Ttr+=Ntr; Tte+=Nte; Tcv+=cover; Tst+=steg; Tunk+=unk
        except Exception as e:
            print(f"{p},ERROR,{e}")
    print(f"SUMMARY,{Ttot},{Ttr},{Tte},{Tcv},{Tst},{Tunk}")


if __name__ == "__main__":
    scan()

