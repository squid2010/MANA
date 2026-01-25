#!/usr/bin/env python3
"""
Evaluate MANA model on lambda and phi datasets and compare train+val vs test.

Creates six plots (saved under results/untrained/<dataset_stem>/):
 - lambda_rmse_mae.png        (RMSE & MAE; indigo = train+val, orange = test)
 - lambda_spearman_pearson.png
 - lambda_rank_accuracy.png
 - phi_rmse_mae.png
 - phi_spearman_pearson.png
 - phi_rank_accuracy.png

Also writes a CSV + JSON summary of metrics under results/untrained/.

Defaults (per your request):
 - model: models/phi/best_model.pth
 - lambda dataset: data/lambda/lambdamax_data.h5
 - phi dataset:    data/phi/phidelta_data_backup.h5

This script expects the project's Python environment to have:
 - torch, numpy, pandas, matplotlib, tqdm
 - DatasetConstructor and MANA available in the project (imports from project)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from model.mana_model import MANA

    from data.dataset import DatasetConstructor
except ImportError:
    # Import errors are left silent so the script can still be linted in
    # environments without full project deps. Runtime will fail early if needed.
    pass

# ---------------------------------------------------------------------
# Colors & style (match scripts/evalution/evaluate.py)
# ---------------------------------------------------------------------
C_PRIMARY = "#565AA2"  # Indigo (used for train+val)
C_ACCENT = "#F6A21C"  # Orange (used for test)
C_BLACK = "#000000"
C_GRAY = "#B5BCBE"


def set_style():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.edgecolor"] = C_BLACK
    plt.rcParams["text.color"] = C_BLACK
    plt.rcParams["grid.color"] = C_GRAY
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["font.family"] = "sans-serif"


# ---------------------------------------------------------------------
# Metrics (reuse same semantics as evaluate.py)
# ---------------------------------------------------------------------
def rmse(pred, true):
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    return (
        float(np.sqrt(np.mean((pred - true) ** 2)))
        if pred.size and true.size
        else float("nan")
    )


def mae(pred, true):
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    return (
        float(np.mean(np.abs(pred - true))) if pred.size and true.size else float("nan")
    )


def pairwise_accuracy(y_true, y_pred):
    """
    Proportion of unordered pairs (i,j) where ordering in y_pred matches y_true.
    Ties in y_true are ignored (pairs with dt == 0).
    Returns nan if no valid pairs.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = y_true.size
    if n < 2:
        return float("nan")
    diff_true = y_true.reshape(-1, 1) - y_true.reshape(1, -1)
    diff_pred = y_pred.reshape(-1, 1) - y_pred.reshape(1, -1)
    iu = np.triu_indices(n, k=1)
    dt = diff_true[iu]
    dp = diff_pred[iu]
    valid = dt != 0
    n_valid = np.sum(valid)
    if n_valid == 0:
        return float("nan")
    concordant = np.sum(np.sign(dt[valid]) == np.sign(dp[valid]))
    return float(concordant) / float(n_valid)


def spearman_corr(a, b):
    # fallback implementation using ranks via pandas
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    except Exception:
        return float("nan")


def pearson_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        return float(pd.Series(a).corr(pd.Series(b), method="pearson"))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------
# Utilities for inference
# ---------------------------------------------------------------------
def safe_to_device(x, device):
    try:
        if isinstance(x, torch.Tensor):
            return x.to(device)
        # some objects (torch_geometric.data.Data) have .to()
        if hasattr(x, "to"):
            return x.to(device)
    except Exception:
        pass
    return x


def tensor_to_numpy_flat(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    arr = np.asarray(x)
    return arr.reshape(-1)


def extract_target_from_batch(batch, candidates):
    """
    Try extracting the target array from batch by checking attributes/keys.
    candidates: list of names to try, e.g. ['lambda_max', 'lambda']
    """
    # attr access
    for name in candidates:
        if hasattr(batch, name):
            return tensor_to_numpy_flat(getattr(batch, name))
    # mapping-like
    try:
        if isinstance(batch, dict):
            for name in candidates:
                if name in batch:
                    return tensor_to_numpy_flat(batch[name])
    except Exception:
        pass
    # fallback: try 'y'
    if hasattr(batch, "y"):
        return tensor_to_numpy_flat(batch.y)
    # if loader returns tuple (data, label)
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return tensor_to_numpy_flat(batch[1])
    raise RuntimeError(f"Could not extract target from batch; tried {candidates}")


def model_predict_on_loader(model, loader, device, predict_key):
    """
    Run model over loader and return (preds, targets) as 1D numpy arrays.
    predict_key: 'lambda' or 'phi'
    """
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {predict_key}", leave=False):
            batch_dev = safe_to_device(batch, device)
            out = model(batch_dev)
            # extract prediction
            if isinstance(out, dict):
                if predict_key in out:
                    pred_tensor = out[predict_key]
                else:
                    # try fallback names
                    pred_tensor = out.get(f"{predict_key}_pred", None) or out.get(
                        f"{predict_key}_mean", None
                    )
                    if pred_tensor is None:
                        # take first tensor-like value
                        vals = [v for v in out.values() if isinstance(v, torch.Tensor)]
                        if len(vals) > 0:
                            pred_tensor = vals[0]
                        else:
                            raise RuntimeError("Model returned no tensor outputs")
            else:
                pred_tensor = out
            pred_np = tensor_to_numpy_flat(pred_tensor)
            preds.append(pred_np)

            # extract target
            if predict_key == "lambda":
                t = extract_target_from_batch(batch, ["lambda_max", "lambda"])
            else:
                t = extract_target_from_batch(batch, ["phi_delta", "phi"])
            targets.append(t)

    if len(preds) == 0:
        return np.array([]), np.array([])
    preds_all = np.concatenate([p.reshape(-1) for p in preds])
    targets_all = np.concatenate([t.reshape(-1) for t in targets])
    return preds_all, targets_all


# ---------------------------------------------------------------------
# Dataset evaluation orchestration
# ---------------------------------------------------------------------
def combined_loader_from(train_loader, val_loader):
    """Return an iterable that yields batches from train_loader then val_loader (if present)."""

    class Chained:
        def __init__(self, loaders):
            self.loaders = [l for l in loaders if l is not None]

        def __iter__(self):
            for l in self.loaders:
                for b in l:
                    yield b

    return Chained([train_loader, val_loader])


def evaluate_dataset(model, device, h5path, task_key):
    """
    Load dataset via DatasetConstructor (split_by_mol_id=True), combine train+val,
    run inference and compute metrics including pairwise accuracy.
    Returns dict with trainval/test metrics and preds/targets.
    """
    ds = DatasetConstructor(str(h5path), split_by_mol_id=True)
    train_loader, val_loader, test_loader = ds.get_dataloaders()
    trainval_loader = combined_loader_from(train_loader, val_loader)

    preds_tv, targets_tv = model_predict_on_loader(
        model, trainval_loader, device, task_key
    )
    preds_test, targets_test = model_predict_on_loader(
        model, test_loader, device, task_key
    )

    # Ensure minimal alignment
    def compute_bundle(p, t):
        if p.size == 0 or t.size == 0:
            return {
                "rmse": float("nan"),
                "mae": float("nan"),
                "spearman": float("nan"),
                "pearson": float("nan"),
                "pairwise": float("nan"),
                "n": int(0),
            }
        m = min(len(p), len(t))
        p = p[:m]
        t = t[:m]
        return {
            "rmse": rmse(p, t),
            "mae": mae(p, t),
            "spearman": spearman_corr(p, t),
            "pearson": pearson_corr(p, t),
            "pairwise": pairwise_accuracy(t, p),
            "n": int(len(p)),
        }

    metrics_tv = compute_bundle(preds_tv, targets_tv)
    metrics_test = compute_bundle(preds_test, targets_test)

    return {
        "trainval": metrics_tv,
        "test": metrics_test,
        "preds_trainval": preds_tv,
        "targets_trainval": targets_tv,
        "preds_test": preds_test,
        "targets_test": targets_test,
    }


# ---------------------------------------------------------------------
# Plotting helpers (dataset-specific, separate lambda/phi)
# ---------------------------------------------------------------------
def plot_rmse_mae(metrics, out_path, title):
    """
    metrics: {'trainval': {...}, 'test': {...}}
    Creates a 1x2 figure: [RMSE] [MAE] where each subplot shows two bars: trainval (indigo), test (orange)
    """
    set_style()
    labels = ["train+val", "test"]
    rmse_vals = [metrics["trainval"]["rmse"], metrics["test"]["rmse"]]
    mae_vals = [metrics["trainval"]["mae"], metrics["test"]["mae"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # RMSE
    axes[0].bar(0, rmse_vals[0], color=C_PRIMARY, label="train+val")
    axes[0].bar(1, rmse_vals[1], color=C_ACCENT, label="test")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title(f"{title} — RMSE")
    for i, v in enumerate(rmse_vals):
        axes[0].text(
            i, v + (abs(v) * 0.01 if not np.isnan(v) else 0.0), f"{v:.3f}", ha="center"
        )

    # MAE
    axes[1].bar(0, mae_vals[0], color=C_PRIMARY, label="train+val")
    axes[1].bar(1, mae_vals[1], color=C_ACCENT, label="test")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("MAE")
    axes[1].set_title(f"{title} — MAE")
    for i, v in enumerate(mae_vals):
        axes[1].text(
            i, v + (abs(v) * 0.01 if not np.isnan(v) else 0.0), f"{v:.3f}", ha="center"
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spearman_pearson(metrics, out_path, title):
    set_style()
    labels = ["train+val", "test"]
    spear_vals = [metrics["trainval"]["spearman"], metrics["test"]["spearman"]]
    pear_vals = [metrics["trainval"]["pearson"], metrics["test"]["pearson"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(0, spear_vals[0], color=C_PRIMARY)
    axes[0].bar(1, spear_vals[1], color=C_ACCENT)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Spearman")
    axes[0].set_title(f"{title} — Spearman")
    for i, v in enumerate(spear_vals):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

    axes[1].bar(0, pear_vals[0], color=C_PRIMARY)
    axes[1].bar(1, pear_vals[1], color=C_ACCENT)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Pearson")
    axes[1].set_title(f"{title} — Pearson")
    for i, v in enumerate(pear_vals):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pairwise_accuracy(metrics, out_path, title):
    set_style()
    labels = ["train+val", "test"]
    vals = [metrics["trainval"]["pairwise"], metrics["test"]["pairwise"]]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(0, vals[0], color=C_PRIMARY)
    ax.bar(1, vals[1], color=C_ACCENT)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Pairwise rank accuracy")
    ax.set_title(f"{title} — Pairwise Rank Accuracy")
    for i, v in enumerate(vals):
        text = "nan" if np.isnan(v) else f"{v:.3f}"
        ax.text(i, (v if not np.isnan(v) else 0.0) + 0.02, text, ha="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare train+val vs test for lambda and phi datasets, plot metrics separately."
    )
    parser.add_argument(
        "--model",
        default="models/phi/best_model.pth",
        help="Path to trained model weights (relative to project root).",
    )
    parser.add_argument(
        "--lambda_h5",
        default="data/lambda/lambdamax_data.h5",
        help="Lambda H5 file (default: data/lambda/lambdamax_data.h5)",
    )
    parser.add_argument(
        "--phi_h5",
        default="data/phi/phidelta_data_backup.h5",
        help="Phi H5 file (default: data/phi/phidelta_data_backup.h5)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    # Project root resolution (file lives in scripts/miscellaneous/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    model_path = project_root / args.model
    lambda_path = project_root / args.lambda_h5
    phi_path = project_root / args.phi_h5

    if not model_path.exists():
        print(f"Model weights not found at {model_path}. Aborting.")
        return
    if not lambda_path.exists():
        print(f"Lambda dataset not found at {lambda_path}. Aborting.")
        return
    if not phi_path.exists():
        print(f"Phi dataset not found at {phi_path}. Aborting.")
        return

    device = (
        torch.device("cpu")
        if args.cpu or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    # Import model + dataset constructor from project path
    # (Ensure project root is on path)
    import sys

    sys.path.insert(0, str(project_root))
    try:
        from model.mana_model import MANA

        from data.dataset import DatasetConstructor
    except Exception as e:
        print("Failed to import project modules:", e)
        return

    # Initialize model and load weights (non-strict)
    model = MANA(num_atom_types=118, hidden_dim=128, tasks=["lambda", "phi"]).to(device)
    try:
        sd = torch.load(str(model_path), map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()
    except Exception as e:
        print("Warning: failed to fully load model weights:", e)
        # proceed with whatever is loaded

    # Evaluate each dataset separately
    out_base = project_root / "results" / "untrained"
    out_base.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("lambda", lambda_path, "lambda"),
        ("phi", phi_path, "phi"),
    ]

    metrics_summary = {}

    for short, path, task_key in datasets:
        print(f"Evaluating {short} dataset at {path} ...")
        try:
            res = evaluate_dataset(model, device, path, task_key)
        except Exception as e:
            print(f"Error evaluating {short}: {e}")
            continue

        metrics_summary[short] = {"trainval": res["trainval"], "test": res["test"]}

        out_dir = out_base / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save per-dataset CSV
        csv_path = out_dir / "metrics_summary.csv"
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(
                ["split", "n", "rmse", "mae", "spearman", "pearson", "pairwise"]
            )
            for split in ["trainval", "test"]:
                m = metrics_summary[short][split]
                writer.writerow(
                    [
                        split,
                        m.get("n", 0),
                        m.get("rmse", "nan"),
                        m.get("mae", "nan"),
                        m.get("spearman", "nan"),
                        m.get("pearson", "nan"),
                        m.get("pairwise", "nan"),
                    ]
                )

        # Save JSON
        with open(out_dir / "metrics_summary.json", "w") as jf:
            json.dump(metrics_summary[short], jf, indent=2)

        # Plots: separate for lambda and phi
        # 1) mae+rmse
        plot_rmse_mae(
            metrics_summary[short],
            out_dir / f"{short}_rmse_mae.png",
            title=short.upper(),
        )
        # 2) spearman & pearson
        plot_spearman_pearson(
            metrics_summary[short],
            out_dir / f"{short}_spearman_pearson.png",
            title=short.upper(),
        )
        # 3) pairwise rank accuracy
        plot_pairwise_accuracy(
            metrics_summary[short],
            out_dir / f"{short}_rank_accuracy.png",
            title=short.upper(),
        )

        # Save preds/targets arrays (optional, helpful)
        try:
            np.save(
                out_dir / "preds_trainval.npy", res.get("preds_trainval", np.array([]))
            )
            np.save(
                out_dir / "targets_trainval.npy",
                res.get("targets_trainval", np.array([])),
            )
            np.save(out_dir / "preds_test.npy", res.get("preds_test", np.array([])))
            np.save(out_dir / "targets_test.npy", res.get("targets_test", np.array([])))
        except Exception:
            pass

        print(f"Saved outputs to {out_dir}")

    # Consolidated CSV at top-level
    top_csv = out_base / "metrics_summary_all.csv"
    with open(top_csv, "w", newline="") as tf:
        writer = csv.writer(tf)
        writer.writerow(
            ["dataset", "split", "n", "rmse", "mae", "spearman", "pearson", "pairwise"]
        )
        for dsname in metrics_summary:
            for split in ["trainval", "test"]:
                m = metrics_summary[dsname][split]
                writer.writerow(
                    [
                        dsname,
                        split,
                        m.get("n", 0),
                        m.get("rmse", "nan"),
                        m.get("mae", "nan"),
                        m.get("spearman", "nan"),
                        m.get("pearson", "nan"),
                        m.get("pairwise", "nan"),
                    ]
                )

    # Save JSON
    with open(out_base / "metrics_summary_all.json", "w") as jf:
        json.dump(metrics_summary, jf, indent=2)

    print("\nAll done. Outputs under:", out_base)


if __name__ == "__main__":
    main()
