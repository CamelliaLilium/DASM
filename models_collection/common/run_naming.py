from typing import List

from models_collection.common.domains import normalize_domain_name

FIXED_DOMAIN_ORDER: List[str] = ["AHCM", "LSB", "PMS", "QIM"]


def _format_domains(domains_csv: str) -> str:
    tokens = [normalize_domain_name(t) for t in str(domains_csv).split(",") if t]
    tokens = [t for t in tokens if t]
    if not tokens:
        return "UNKNOWN"
    token_set = set(tokens)
    ordered = [d for d in FIXED_DOMAIN_ORDER if d in token_set]
    extras = sorted(token_set.difference(FIXED_DOMAIN_ORDER))
    return "_".join(ordered + extras)


def get_optimizer_type(args) -> str:
    if getattr(args, "use_dgsam", False):
        return "dgsam"
    if getattr(args, "use_dbsm", False):
        return "dbsm"
    if getattr(args, "use_dasm", False):
        return "dasm"
    if getattr(args, "use_disam", False):
        return "disam"
    if getattr(args, "use_sam", False):
        return "sam"
    return "adam"


def build_run_tag(args, optimizer_type: str = None) -> str:
    opt = optimizer_type or get_optimizer_type(args)
    train_names = _format_domains(getattr(args, "train_domains", ""))
    test_names = _format_domains(getattr(args, "test_domains", ""))
    # 与 get_optimizer_type 一致：adam → adam_train，sam → sam_train，dasm → dasm_train（勿把 Adam 写成 dasm_train）
    prefix = f"{opt}_train"
    return f"{prefix}_{train_names}_to_{test_names}_{args.steg_algorithm}_bs{args.batch_size}"

