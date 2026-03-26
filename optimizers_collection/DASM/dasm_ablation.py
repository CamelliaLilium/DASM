#!/usr/bin/env python3
"""
DASM ablation runner.

Implements the ablation logic described in the provided figure:
1) remove SAM (use base Adam only)            -> rho=0.0 (keeps contrastive loss)
2) adaptive perturbation on/off               -> adaptive False vs True
3) remove contrastive loss                    -> use_contrast False
4) rho sweep                                  -> 0, 0.01, 0.05, 0.1
5) contrast_tau sweep                         -> 0.03, 0.07, 0.1, 0.2
6) lr sweep                                   -> configurable list
7) weight_decay sweep                         -> 0, 1e-4, 1e-3

This script generates commands for model_dasm_DomainGap.py and can
run them sequentially. It can also summarize results and emit paper-ready tables
from train_logs_*.json outputs.
"""
import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DASM_TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "model_dasm_DomainGap.py")


@dataclass
class Experiment:
    name: str
    overrides: Dict[str, object] = field(default_factory=dict)

    def build_args(self, base_args: Dict[str, object]) -> List[str]:
        args = dict(base_args)
        args.update(self.overrides)
        return _args_dict_to_list(args)


def _parse_float_list(value: str) -> List[float]:
    return [float(v) for v in value.split(",") if v.strip() != ""]


def _args_dict_to_list(args: Dict[str, object]) -> List[str]:
    flags: List[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                flags.append(flag)
            continue
        flags.append(f"{flag}={value}")
    return flags


def build_base_args(args: argparse.Namespace) -> Dict[str, object]:
    base = {
        "dataset_id": args.dataset_id,
        "embedding_rate": args.embedding_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_domains": args.train_domains,
        "test_domains": args.test_domains,
        "steg_algorithm": args.steg_algorithm,
        "gpu": args.gpu,
        "device": args.device,
        "eval_step": args.eval_step,
        "domain_test_interval": args.domain_test_interval,
        "data_root": args.data_root,
        "test_data_root": args.test_data_root,
        "result_path": args.result_path,
        "save_model": True,
        "use_dasm": args.use_dasm,
        "rho": args.rho,
        "adaptive": args.adaptive,
        "contrast_tau": args.contrast_tau,
        "use_contrast": args.use_contrast,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    return base


def build_experiments(args: argparse.Namespace) -> List[Experiment]:
    experiments: List[Experiment] = []
    if args.ablation_mode == "core3":
        # 3-group ablation for DASM + contrast
        # 1) Adam baseline: no DASM
        experiments.append(
            Experiment(
                name="core_adam_baseline",
                overrides={
                    "use_dasm": False,
                    "use_contrast": False,
                },
            )
        )
        # 2) DASM without contrast
        experiments.append(
            Experiment(
                name="core_dasm_no_contrast",
                overrides={
                    "use_dasm": True,
                    "use_contrast": False,
                },
            )
        )
        # 3) Full DASM (with contrast)
        experiments.append(
            Experiment(
                name="core_dasm_full",
                overrides={
                    "use_dasm": True,
                    "use_contrast": True,
                },
            )
        )
        return experiments

    # 1) Remove SAM: use rho=0 to disable perturbation while keeping contrastive loss.
    experiments.append(
        Experiment(
            name="ablate_no_sam_rho0",
            overrides={"rho": 0.0, "adaptive": False},
        )
    )

    # 2) Adaptive perturbation (ASAM) on/off
    experiments.append(
        Experiment(
            name="ablate_adaptive_false",
            overrides={"adaptive": False},
        )
    )
    experiments.append(
        Experiment(
            name="ablate_adaptive_true",
            overrides={"adaptive": True},
        )
    )

    # 3) Remove contrastive loss
    experiments.append(
        Experiment(
            name="ablate_no_contrast",
            overrides={"use_contrast": False},
        )
    )

    # 4) Rho sweep
    for rho in args.rho_list:
        experiments.append(
            Experiment(
                name=f"ablate_rho_{rho}",
                overrides={"rho": rho},
            )
        )

    # 5) Contrast temperature sweep
    for tau in args.tau_list:
        experiments.append(
            Experiment(
                name=f"ablate_tau_{tau}",
                overrides={"contrast_tau": tau},
            )
        )

    # 6) Learning rate sweep
    for lr in args.lr_list:
        experiments.append(
            Experiment(
                name=f"ablate_lr_{lr}",
                overrides={"lr": lr},
            )
        )

    # 7) Weight decay sweep
    for wd in args.wd_list:
        experiments.append(
            Experiment(
                name=f"ablate_wd_{wd}",
                overrides={"weight_decay": wd},
            )
        )

    if args.groups:
        selected = set(g.strip() for g in args.groups.split(",") if g.strip())
        experiments = [e for e in experiments if any(e.name.startswith(g) for g in selected)]

    return experiments


def _scan_train_logs(root: Path) -> Dict[str, List[Path]]:
    logs: Dict[str, List[Path]] = {}
    if not root.exists():
        return logs
    for log_path in root.rglob("train_logs_*.json"):
        try:
            rel = log_path.relative_to(root)
        except ValueError:
            continue
        if not rel.parts:
            continue
        exp_name = rel.parts[0]
        logs.setdefault(exp_name, []).append(log_path)
    return logs


def _extract_metrics(log_path: Path) -> Dict[str, float]:
    with log_path.open("r") as f:
        data = json.load(f)
    domain_test_acc = data.get("domain_test_acc", [])
    max_acc = {"QIM": 0.0, "PMS": 0.0, "LSB": 0.0, "AHCM": 0.0}
    for epoch_data in domain_test_acc:
        if isinstance(epoch_data, dict):
            for domain in max_acc:
                value = epoch_data.get(domain, None)
                if isinstance(value, (int, float)):
                    max_acc[domain] = max(max_acc[domain], float(value))
    avg = sum(max_acc.values()) / len(max_acc) if max_acc else 0.0
    val_acc = data.get("val_acc", [])
    epoch_acc = data.get("epoch_acc", [])
    val_best = max(val_acc) if val_acc else 0.0
    train_best = max(epoch_acc) if epoch_acc else 0.0
    metrics = {
        "avg": avg,
        "QIM": max_acc["QIM"],
        "PMS": max_acc["PMS"],
        "LSB": max_acc["LSB"],
        "AHCM": max_acc["AHCM"],
        "val_best": val_best,
        "train_best": train_best,
    }
    return metrics


def _write_latex_table(rows: List[Tuple[str, Dict[str, float]]], out_path: Path,
                       caption: str, label: str) -> None:
    header = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Setting & Avg & QIM & PMS & LSB & AHCM & Val \\",
        r"\midrule",
    ]
    body = []
    for name, metrics in rows:
        body.append(
            f"{name} & {metrics['avg']:.4f} & {metrics['QIM']:.4f} & "
            f"{metrics['PMS']:.4f} & {metrics['LSB']:.4f} & "
            f"{metrics['AHCM']:.4f} & {metrics['val_best']:.4f} \\\\"
        )
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header + body + footer))


def _write_csv(rows: List[Tuple[str, Dict[str, float]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["setting,avg,qim,pms,lsb,ahcm,val_best,train_best"]
    for name, metrics in rows:
        lines.append(
            f"{name},{metrics['avg']:.6f},{metrics['QIM']:.6f},{metrics['PMS']:.6f},"
            f"{metrics['LSB']:.6f},{metrics['AHCM']:.6f},{metrics['val_best']:.6f},"
            f"{metrics['train_best']:.6f}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def generate_tables(args: argparse.Namespace) -> None:
    root = Path(args.result_path)
    if args.ablation_root:
        root = root / args.ablation_root
    logs_by_exp = _scan_train_logs(root)
    if not logs_by_exp:
        print(f"No train_logs_*.json found under {root}")
        return
    rows: List[Tuple[str, Dict[str, float]]] = []
    for exp_name, log_paths in sorted(logs_by_exp.items()):
        # pick the latest log for this exp
        log_path = max(log_paths, key=lambda p: p.stat().st_mtime)
        metrics = _extract_metrics(log_path)
        rows.append((exp_name, metrics))

    _write_latex_table(rows, Path(args.table_out), args.table_caption, args.table_label)
    if args.table_csv:
        _write_csv(rows, Path(args.table_csv))
    print(f"Wrote LaTeX table to: {args.table_out}")
    if args.table_csv:
        print(f"Wrote CSV summary to: {args.table_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DASM ablation runner")
    parser.add_argument(
        "--train_script",
        type=str,
        default=DASM_TRAIN_SCRIPT,
        help="Training entry script path (default: model_dasm_DomainGap.py)",
    )
    parser.add_argument("--dataset_id", required=True, help="Combined dataset id or path")
    parser.add_argument("--embedding_rate", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=260)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_domains", type=str, default="QIM,PMS,LSB,AHCM")
    parser.add_argument("--test_domains", type=str, default="QIM,PMS,LSB,AHCM")
    parser.add_argument("--steg_algorithm", type=str, default="Transformer")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--domain_test_interval", type=int, default=5)
    parser.add_argument("--data_root", type=str, default=os.environ.get("DASM_DATA_ROOT", os.path.join(PROJECT_ROOT, "dataset", "model_train")))
    parser.add_argument("--test_data_root", type=str, default=os.environ.get("DASM_TEST_DATA_ROOT", os.path.join(PROJECT_ROOT, "dataset", "model_test")))
    parser.add_argument(
        "--result_path",
        type=str,
        default=os.environ.get("DASM_ABLATION_RESULT_ROOT", os.path.join(PROJECT_ROOT, "optimizers_collection", "DASM", "Ablation")),
    )
    parser.add_argument(
        "--ablation_root",
        type=str,
        default="",
        help="Subfolder under result_path to store ablation runs (set empty to disable)",
    )

    # Base DASM hyperparameters
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--contrast_tau", type=float, default=0.07)
    parser.add_argument("--use_contrast", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_dasm", action="store_true", default=True)

    # Sweep lists
    parser.add_argument("--rho_list", type=_parse_float_list, default="0,0.01,0.05,0.1")
    parser.add_argument("--tau_list", type=_parse_float_list, default="0.03,0.07,0.1,0.2")
    parser.add_argument("--lr_list", type=_parse_float_list, default="0.0001,0.0005,0.001")
    parser.add_argument("--wd_list", type=_parse_float_list, default="0,0.0001,0.001")
    parser.add_argument("--groups", type=str, default=None, help="Prefix filter, e.g. 'ablate_rho,ablate_tau'")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default=None,
        choices=[None, "core3"],
        help="If set to core3, generate only the 3 core ablation groups.",
    )

    # Execution
    parser.add_argument("--run", action="store_true", help="Execute commands (default: dry-run)")
    parser.add_argument("--max_runs", type=int, default=None, help="Limit number of runs")
    parser.add_argument("--table_only", action="store_true", help="Only generate tables and exit")
    parser.add_argument("--table_out", type=str, default="dasm_ablation_table.tex")
    parser.add_argument("--table_csv", type=str, default=None)
    parser.add_argument("--table_caption", type=str, default="DASM ablation results.")
    parser.add_argument("--table_label", type=str, default="tab:dasm_ablation")
    args = parser.parse_args()

    if args.table_only:
        generate_tables(args)
        return

    base_args = build_base_args(args)
    experiments = build_experiments(args)

    if args.max_runs is not None:
        experiments = experiments[: args.max_runs]

    for idx, exp in enumerate(experiments, start=1):
        run_args = dict(base_args)
        if args.ablation_root:
            run_args["result_path"] = os.path.join(args.result_path, args.ablation_root, exp.name)
        else:
            run_args["result_path"] = os.path.join(args.result_path, exp.name)
        cmd = ["python", args.train_script] + exp.build_args(run_args)
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        print(f"[{idx}/{len(experiments)}] {exp.name}")
        print(cmd_str)
        if args.run:
            subprocess.run(cmd, check=True)
        else:
            print("(dry-run; add --run to execute)")
        print("-" * 80)


if __name__ == "__main__":
    main()
