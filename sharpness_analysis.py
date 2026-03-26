#!/usr/bin/env python3
"""
Zeroth-order sharpness analysis on test domains.

Metric:
  Sharpness = max_{||eps||_2 <= rho} L(theta + eps) - L(theta)

We approximate the worst-case direction with the normalized gradient of the
average test loss (same evaluation metric for all models).
"""
import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


DOMAIN_NAMES = ["QIM", "PMS", "LSB", "AHCM"]
DOMAIN_IDS = {name: idx for idx, name in enumerate(DOMAIN_NAMES)}


@dataclass
class ModelSpec:
    name: str
    key: str
    checkpoint: str


def add_repo_paths():
    transformer_dir = os.path.join(PROJECT_ROOT, "models_collection", "Transformer")
    if transformer_dir not in sys.path:
        sys.path.append(transformer_dir)


add_repo_paths()
from transformer import Classifier1  # noqa: E402


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(path))

    def _resolve_repo_relative(value: str) -> str:
        if not isinstance(value, str):
            return value
        if os.path.isabs(value):
            return value
        candidate = os.path.join(config_dir, value)
        if os.path.exists(candidate):
            return candidate
        return os.path.join(PROJECT_ROOT, value)

    if "dataset" in config and isinstance(config["dataset"], dict):
        data_root = config["dataset"].get("data_root")
        if isinstance(data_root, str):
            config["dataset"]["data_root"] = _resolve_repo_relative(data_root)

    metrics_summary_csv = config.get("metrics_summary_csv")
    if isinstance(metrics_summary_csv, str):
        config["metrics_summary_csv"] = _resolve_repo_relative(metrics_summary_csv)

    for model in config.get("models", []):
        checkpoint = model.get("checkpoint")
        if isinstance(checkpoint, str):
            model["checkpoint"] = _resolve_repo_relative(checkpoint)

    return config


def get_pkl_path(data_root: str, dataset_id: str) -> str:
    if dataset_id.endswith(".pkl"):
        return os.path.join(data_root, dataset_id)
    return os.path.join(data_root, f"{dataset_id}.pkl")


def load_pkl(pkl_path: str):
    import pickle

    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Missing pkl: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, tuple) or len(data) not in (4, 6):
        raise ValueError(f"Unsupported pkl format: {type(data)}")
    if len(data) == 6:
        return data  # x_train, y_train, x_test, y_test, algo_train, algo_test
    return (*data, None, None)


def preprocess_features(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_train = x_train[:, :, 0:7]
    x_test = x_test[:, :, 0:7]
    x_train = np.where(x_train == -1, 200, x_train)
    x_test = np.where(x_test == -1, 200, x_test)
    return x_train, x_test


def extract_binary_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1 and y.shape[1] > 1:
        return y[:, 1].astype(np.int64)
    return y.astype(np.int64)


def build_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> DataLoader:
    x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))
    dataset = TensorDataset(x_tensor, y_tensor)

    if max_batches is not None:
        max_samples = min(len(dataset), batch_size * max_batches)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(dataset))[:max_samples]
        dataset = Subset(dataset, indices.tolist())

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    pe_key = "model1.position_embedding.pe"
    if pe_key in state_dict:
        ckpt_pe = state_dict[pe_key]
        model_pe = model.state_dict().get(pe_key)
        if model_pe is not None and ckpt_pe.shape != model_pe.shape:
            if ckpt_pe.ndim == 3 and ckpt_pe.transpose(0, 1).shape == model_pe.shape:
                state_dict[pe_key] = ckpt_pe.transpose(0, 1)

    model.load_state_dict(state_dict, strict=False)


def average_loss_and_backward(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.zero_grad()
    total_loss = 0.0
    total_samples = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def average_loss_no_grad(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def compute_sharpness(
    model: nn.Module,
    loader: DataLoader,
    rho: float,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    base_loss = average_loss_and_backward(model, loader, criterion, device)

    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_norm_sq += torch.sum(p.grad ** 2).item()
    grad_norm = float(np.sqrt(grad_norm_sq))

    if grad_norm == 0.0:
        pert_loss = base_loss
        sharpness = 0.0
        return {
            "base_loss": base_loss,
            "pert_loss": pert_loss,
            "sharpness": sharpness,
            "grad_norm": grad_norm,
        }

    scale = rho / (grad_norm + 1e-12)
    perturbations = []
    for p in model.parameters():
        if p.grad is None:
            perturbations.append(None)
            continue
        eps = p.grad * scale
        p.data.add_(eps)
        perturbations.append(eps)

    pert_loss = average_loss_no_grad(model, loader, criterion, device)

    for p, eps in zip(model.parameters(), perturbations):
        if eps is None:
            continue
        p.data.sub_(eps)

    sharpness = pert_loss - base_loss
    return {
        "base_loss": base_loss,
        "pert_loss": pert_loss,
        "sharpness": sharpness,
        "grad_norm": grad_norm,
    }


def format_mean_std(values: List[float]) -> str:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, std, f"{mean:.3f} ({std:.3f})"


def save_outputs(output_dir: str, records: List[Dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "sharpness_analysis_results.json")
    csv_path = os.path.join(output_dir, "sharpness_analysis_results.csv")
    md_path = os.path.join(output_dir, "sharpness_analysis_table.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "QIM",
                "PMS",
                "LSB",
                "AHCM",
                "mean",
                "std",
                "total",
                "rho",
                "checkpoint",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Algorithm | QIM | PMS | LSB | AHCM | Mean (Std) | Total |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in records:
            f.write(
                f"| {row['algorithm']} | {row['QIM']:.3f} | {row['PMS']:.3f} | "
                f"{row['LSB']:.3f} | {row['AHCM']:.3f} | "
                f"{row['mean']:.3f} ({row['std']:.3f}) | {row['total']:.3f} |\n"
            )

        f.write("\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\caption{Zeroth-order sharpness on individual domains and total test set.}\n")
        f.write("\\label{tab:zeroth_order_sharpness}\n")
        f.write("\\vskip 0.1in\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{small}\n")
        f.write("\\begin{sc}\n")
        f.write("\\begin{tabular}{l|cccc|c|c}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{4}{c|}{Individual domains} & Mean (Std) & Total \\\\\n")
        f.write("Algorithm & QIM & PMS & LSB & AHCM &  &  \\\\\n")
        f.write("\\midrule\n")
        for row in records:
            f.write(
                f"{row['algorithm']} & {row['QIM']:.3f} & {row['PMS']:.3f} & "
                f"{row['LSB']:.3f} & {row['AHCM']:.3f} & "
                f"{row['mean']:.3f} ({row['std']:.3f}) & {row['total']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{sc}\n")
        f.write("\\end{small}\n")
        f.write("\\end{center}\n")
        f.write("\\vskip -0.1in\n")
        f.write("\\end{table}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Zeroth-order sharpness analysis")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--output_dir", default=os.environ.get("DASM_SHARPNESS_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "sharpness_analysis")), help="Output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Computation device")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    rho = float(config.get("rho", 0.05))
    seed = int(config.get("seed", 42))

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_root = dataset_cfg["data_root"]
    dataset_id = dataset_cfg["dataset_id"]
    batch_size = int(dataset_cfg.get("batch_size", 128))
    max_batches_raw = dataset_cfg.get("max_batches", 20)
    max_batches = None if max_batches_raw is None else int(max_batches_raw)

    pkl_path = get_pkl_path(data_root, dataset_id)
    x_train, y_train, x_test, y_test, algo_train, algo_test = load_pkl(pkl_path)
    x_train, x_test = preprocess_features(x_train, x_test)

    y_test_bin = extract_binary_labels(y_test)
    if algo_test is None:
        raise ValueError("algo_labels_test missing in pkl; cannot split by domain.")

    loaders = {}
    for name in DOMAIN_NAMES:
        domain_id = DOMAIN_IDS[name]
        indices = np.where(algo_test == domain_id)[0]
        loaders[name] = build_loader(
            x_test[indices],
            y_test_bin[indices],
            batch_size=batch_size,
            max_batches=max_batches,
            seed=seed + domain_id,
        )

    loaders["Total"] = build_loader(
        x_test,
        y_test_bin,
        batch_size=batch_size,
        max_batches=max_batches,
        seed=seed + 99,
    )

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model_specs = [ModelSpec(**item) for item in config["models"]]
    records = []

    for spec in model_specs:
        model_args = SimpleNamespace(
            d_model=dataset_cfg.get("d_model", 64),
            num_heads=dataset_cfg.get("num_heads", 8),
            d_ff=dataset_cfg.get("d_ff", 256),
            max_len=dataset_cfg.get("max_len", 100),
            num_layers=dataset_cfg.get("num_layers", 2),
            dropout=dataset_cfg.get("dropout", 0.5),
            num_class=2,
        )
        model = Classifier1(model_args).to(device)
        load_checkpoint(model, spec.checkpoint)

        start = time.time()
        domain_vals = {}
        for name in DOMAIN_NAMES:
            result = compute_sharpness(model, loaders[name], rho=rho, device=device)
            domain_vals[name] = result["sharpness"]

        total_result = compute_sharpness(model, loaders["Total"], rho=rho, device=device)
        mean_val, std_val, _ = format_mean_std([domain_vals[n] for n in DOMAIN_NAMES])
        runtime_sec = round(time.time() - start, 2)

        records.append(
            {
                "algorithm": spec.name,
                "QIM": float(domain_vals["QIM"]),
                "PMS": float(domain_vals["PMS"]),
                "LSB": float(domain_vals["LSB"]),
                "AHCM": float(domain_vals["AHCM"]),
                "mean": float(mean_val),
                "std": float(std_val),
                "total": float(total_result["sharpness"]),
                "rho": rho,
                "checkpoint": spec.checkpoint,
                "runtime_sec": runtime_sec,
            }
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_outputs(args.output_dir, records)
    print(f"Saved results to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Sharpness Analysis (Hessian-based) for Transformer variants.

This script computes:
  - Top-1 Hessian eigenvalue (lambda_max)
  - Hessian trace (Hutchinson)
  - Test accuracy and train-test gap (from summary CSV)

Outputs are saved under the output directory as JSON/CSV/Markdown.
"""
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset


def add_repo_paths():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in os.sys.path:
        os.sys.path.append(repo_root)
    hessian_dir = os.path.join(repo_root, "hessian")
    if hessian_dir not in os.sys.path:
        os.sys.path.append(hessian_dir)


add_repo_paths()
from hessian_new import hessian  # noqa: E402


@dataclass
class ModelSpec:
    name: str
    key: str
    checkpoint: str


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summary_csv(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return summary[json_name][metric_type][domain] -> value."""
    if not os.path.isfile(path):
        return {}

    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            lines.append(line)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    reader = csv.DictReader(lines)
    for row in reader:
        json_name = row["json_name"].strip()
        metric_type = row["metric_type"].strip()
        domain = row["domain"].strip()
        try:
            value = float(row["value"])
        except (ValueError, TypeError):
            continue
        summary.setdefault(json_name, {}).setdefault(metric_type, {})[domain] = value
    return summary


def get_metric(summary: Dict, json_key: str, metric: str, domains: Tuple[str, ...]) -> Optional[float]:
    data = summary.get(json_key, {}).get(metric, {})
    for domain in domains:
        if domain in data:
            return data[domain]
    return None


def get_pkl_path(data_root: str, dataset_id: str) -> str:
    if dataset_id.endswith(".pkl"):
        return os.path.join(data_root, dataset_id)
    return os.path.join(data_root, f"{dataset_id}.pkl")


def load_dataset(pkl_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pickle

    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Missing pkl: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, tuple) or len(data) not in (4, 6):
        raise ValueError(f"Unsupported pkl format: {type(data)}")

    if len(data) == 6:
        x_train, y_train, x_test, y_test, _, _ = data
    else:
        x_train, y_train, x_test, y_test = data

    return x_train, y_train, x_test, y_test


def preprocess_features(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_train = x_train[:, :, 0:7]
    x_test = x_test[:, :, 0:7]
    x_train = np.where(x_train == -1, 200, x_train)
    x_test = np.where(x_test == -1, 200, x_test)
    return x_train, x_test


def extract_binary_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1 and y.shape[1] > 1:
        return y[:, 1].astype(np.int64)
    return y.astype(np.int64)


def build_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, max_batches: Optional[int]) -> DataLoader:
    x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))
    dataset = TensorDataset(x_tensor, y_tensor)

    if max_batches is not None:
        max_samples = min(len(dataset), batch_size * max_batches)
        subset = Subset(dataset, list(range(max_samples)))
        return DataLoader(subset, batch_size=batch_size, shuffle=False)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_model(d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int, dropout: float) -> nn.Module:
    import math
    import torch.nn.functional as F

    class HessianCompatibleMultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
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
            Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.w_o(attn_output)

    class HessianCompatibleTransformerLayer(nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int = 256, dropout: float = 0.1):
            super().__init__()
            self.self_attn = HessianCompatibleMultiHeadAttention(d_model, num_heads, dropout)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            attn_output = self.self_attn(x)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            return x

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 100):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len, :]
            return x + pe

    class Model1(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, d_model)
            self.position_embedding = PositionalEncoding(d_model, max_len)
            self.transformer_layers = nn.ModuleList(
                [HessianCompatibleTransformerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
            )
            self.pooling = nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            x = x.long()
            emb_x = self.embedding(x)
            if emb_x.dim() == 4:
                emb_x = emb_x.mean(dim=2)
            elif emb_x.dim() != 3:
                raise ValueError(f"Unexpected embedding shape: {emb_x.shape}")
            emb_x = self.position_embedding(emb_x)
            for layer in self.transformer_layers:
                emb_x = layer(emb_x)
            outputs = self.pooling(emb_x.permute(0, 2, 1)).squeeze(2)
            return outputs

    class Classifier1(nn.Module):
        def __init__(self):
            super().__init__()
            self.model1 = Model1()
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, 2)

        def forward(self, x):
            features = self.model1(x)
            x = self.dropout(features)
            return self.fc(x)

    return Classifier1()


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    pe_key = "model1.position_embedding.pe"
    if pe_key in state_dict:
        ckpt_pe = state_dict[pe_key]
        model_pe = model.state_dict().get(pe_key)
        if model_pe is not None and ckpt_pe.shape != model_pe.shape:
            if ckpt_pe.ndim == 3 and ckpt_pe.transpose(0, 1).shape == model_pe.shape:
                state_dict[pe_key] = ckpt_pe.transpose(0, 1)

    model.load_state_dict(state_dict, strict=False)


def compute_hessian_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eig_iter: int,
    trace_iter: int,
    tol: float,
) -> Tuple[float, float]:
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    hess = hessian(model, criterion, data=None, dataloader=loader, cuda=device.type == "cuda")
    eigvals, _ = hess.eigenvalues(maxIter=eig_iter, tol=tol, top_n=1)
    trace_vals = hess.trace(maxIter=trace_iter, tol=tol)
    lambda_max = float(eigvals[0]) if eigvals else float("nan")
    trace_mean = float(np.mean(trace_vals)) if trace_vals else float("nan")
    return lambda_max, trace_mean


def save_outputs(output_dir: str, records: List[Dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "sharpness_analysis_results.json")
    csv_path = os.path.join(output_dir, "sharpness_analysis_results.csv")
    md_path = os.path.join(output_dir, "sharpness_analysis_table.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "lambda_max",
                "trace",
                "test_acc",
                "train_acc",
                "gap",
                "checkpoint",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Algorithm | lambda_max (↓) | Trace (↓) | Test Acc. (↑) | Gap (Train-Test) |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for row in records:
            def fmt(val):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                if isinstance(val, float):
                    return f"{val:.3f}"
                return str(val)

            test_acc = row["test_acc"]
            gap = row["gap"]
            test_acc_str = f"{test_acc:.2f}%" if isinstance(test_acc, float) else "N/A"
            gap_str = f"{gap:.2f}%" if isinstance(gap, float) else "N/A"
            f.write(
                f"| {row['algorithm']} | {fmt(row['lambda_max'])} | {fmt(row['trace'])} | {test_acc_str} | {gap_str} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sharpness Analysis (Hessian-based)")
    parser.add_argument("--config", required=True, help="Path to sharpness analysis config JSON")
    parser.add_argument("--output_dir", default=os.environ.get("DASM_SHARPNESS_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "sharpness_analysis")), help="Output directory")
    parser.add_argument("--eig_iter", type=int, default=50, help="Max iterations for top eigenvalue")
    parser.add_argument("--trace_iter", type=int, default=50, help="Max iterations for trace")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for eigen/trace")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Computation device")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    summary_path = config["metrics_summary_csv"]
    model_specs = [ModelSpec(**item) for item in config["models"]]

    pkl_path = get_pkl_path(dataset_cfg["data_root"], dataset_cfg["dataset_id"])
    x_train, y_train, x_test, y_test = load_dataset(pkl_path)
    x_train, x_test = preprocess_features(x_train, x_test)
    y_train_bin = extract_binary_labels(y_train)
    y_test_bin = extract_binary_labels(y_test)

    loader = build_dataloader(
        x_test,
        y_test_bin,
        batch_size=dataset_cfg.get("batch_size", 128),
        max_batches=dataset_cfg.get("max_batches"),
    )

    summary = load_summary_csv(summary_path)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    records = []
    for spec in model_specs:
        record = {
            "algorithm": spec.name,
            "lambda_max": None,
            "trace": None,
            "test_acc": None,
            "train_acc": None,
            "gap": None,
            "checkpoint": spec.checkpoint,
        }

        json_key = f"{spec.key}.json"
        train_acc = get_metric(summary, json_key, "epoch_acc_best", ("NA", "ALL"))
        test_acc = get_metric(summary, json_key, "domain_test_acc_best_avg", ("ALL", "NA"))
        record["train_acc"] = train_acc
        record["test_acc"] = test_acc
        if train_acc is not None and test_acc is not None:
            record["gap"] = train_acc - test_acc

        if not os.path.isfile(spec.checkpoint):
            records.append(record)
            continue

        model = create_model(
            d_model=dataset_cfg.get("d_model", 64),
            num_heads=dataset_cfg.get("num_heads", 8),
            d_ff=dataset_cfg.get("d_ff", 256),
            num_layers=dataset_cfg.get("num_layers", 2),
            max_len=dataset_cfg.get("max_len", 100),
            dropout=dataset_cfg.get("dropout", 0.5),
        )

        load_checkpoint(model, spec.checkpoint)

        start = time.time()
        lambda_max, trace = compute_hessian_metrics(
            model, loader, device, eig_iter=args.eig_iter, trace_iter=args.trace_iter, tol=args.tol
        )
        elapsed = time.time() - start
        record["lambda_max"] = lambda_max
        record["trace"] = trace
        record["runtime_sec"] = round(elapsed, 2)
        records.append(record)

    save_outputs(args.output_dir, records)
    print(f"Saved results to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
