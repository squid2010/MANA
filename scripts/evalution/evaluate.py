#!/usr/bin/env python3
"""
Evaluation script for MANA models.

This file fixes solvent handling and reporting so the evaluation follows the
same semantics as the screening code (i.e. a "solvated" forward pass and a
clean "vacuum" forward pass where solvent tensors are empty). It also
improves solvent-shift reporting so lambda and phi solvent shifts are aligned
per-sample in a stable, predictable DataFrame (NaNs where a shift is not
available).
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# ---------------------------------------------------------------------
# Path & Import Setup
# ---------------------------------------------------------------------
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

try:
    from rdkit import Chem  # noqa: F401

    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

# ---------------------------------------------------------------------
# Configuration & Colors
# ---------------------------------------------------------------------
NUM_ATOM_TYPES = 118
C_PRIMARY = "#565AA2"  # Indigo
C_ACCENT = "#F6A21C"  # Orange
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
# Plotting / Metrics Helpers
# ---------------------------------------------------------------------
def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def pairwise_accuracy(y_true, y_pred):
    """
    Compute pairwise ranking accuracy: proportion of unordered pairs (i,j)
    where the relative ordering in y_pred matches that in y_true.
    Ties in y_true are ignored.
    Returns None if no valid pairs.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    if n < 2:
        return None
    diff_true = y_true.reshape(-1, 1) - y_true.reshape(1, -1)
    diff_pred = y_pred.reshape(-1, 1) - y_pred.reshape(1, -1)
    iu = np.triu_indices(n, k=1)
    dt = diff_true[iu]
    dp = diff_pred[iu]
    valid = dt != 0
    n_valid = np.sum(valid)
    if n_valid == 0:
        return None
    concordant = np.sum(np.sign(dt[valid]) == np.sign(dp[valid]))
    return float(concordant) / float(n_valid)


def parity_plot(y_true, y_pred, title, xlabel, ylabel, path):
    if len(y_true) == 0:
        return
    set_style()
    plt.figure(figsize=(6, 6))
    min_val, max_val = (
        float(min(y_true.min(), y_pred.min())),
        float(max(y_true.max(), y_pred.max())),
    )
    buffer = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
    lims = [min_val - buffer, max_val + buffer]
    plt.plot(lims, lims, color=C_BLACK, linestyle="--", alpha=0.7, zorder=1)
    plt.scatter(
        y_true, y_pred, s=20, alpha=0.6, c=C_PRIMARY, edgecolors="none", zorder=2
    )
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, bins, labels, title, path):
    if len(y_true) == 0:
        return
    set_style()
    y_true_cat = pd.cut(y_true, bins=bins, labels=labels).astype(str)
    y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels).astype(str)
    valid = (y_true_cat != "nan") & (y_pred_cat != "nan")
    y_true_cat, y_pred_cat = y_true_cat[valid], y_pred_cat[valid]
    if len(y_true_cat) == 0:
        return

    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1%",
        cmap=sns.light_palette(C_PRIMARY, as_cmap=True),
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(title, fontweight="bold")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_solvent_effects(df, out_dir):
    if df.empty:
        return
    set_style()

    # Lambda shift histogram (drop NaNs)
    if "lambda_shift" in df.columns and not df["lambda_shift"].dropna().empty:
        plt.figure(figsize=(6, 4))
        plt.hist(
            df["lambda_shift"].dropna(),
            bins=30,
            color=C_ACCENT,
            edgecolor=C_BLACK,
            alpha=0.8,
        )
        plt.title("Predicted Solvatochromism (Solvent - Vacuum)", fontweight="bold")
        plt.xlabel("$\\Delta \\lambda_{max}$ (nm)", fontweight="bold")
        plt.ylabel("Count", fontweight="bold")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / "solvatochromism_hist.png")
        plt.close()

    # Phi shift histogram (drop NaNs)
    if "phi_shift" in df.columns and not df["phi_shift"].dropna().empty:
        plt.figure(figsize=(6, 4))
        plt.hist(
            df["phi_shift"].dropna(),
            bins=30,
            color=C_PRIMARY,
            edgecolor=C_BLACK,
            alpha=0.8,
        )
        plt.title("Predicted Solvent Effect on Yield", fontweight="bold")
        plt.xlabel("$\\Delta \\Phi_{\\Delta}$ (Solvated - Vacuum)", fontweight="bold")
        plt.ylabel("Count", fontweight="bold")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / "phi_solvent_effect_hist.png")
        plt.close()


# ---------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------
def _to_numpy_flat(tensor):
    """Utility: make sure tensor -> 1d numpy array (float)"""
    if isinstance(tensor, np.ndarray):
        return tensor.ravel()
    if tensor is None:
        return np.array([], dtype=float)
    try:
        return tensor.detach().cpu().numpy().ravel()
    except Exception:
        return np.array([], dtype=float)


def _make_vacuum_batch(batch, device):
    """
    Return a shallow clone of `batch` with solvent-related tensors replaced by
    empty tensors on the same device and with appropriate dtypes/shapes.

    The model forward checks `hasattr(data, "x_s") and data.x_s.numel() > 0`.
    So ensuring `x_s.numel() == 0` is sufficient to indicate a vacuum pass.
    """
    # Many PyG Data implementations support .clone(); use it to avoid mutating input.
    try:
        batch_vac = batch.clone()
    except Exception:
        # As a fallback, try a shallow copy of attributes
        batch_vac = batch

    # Use device argument (provided by caller)
    long_dtype = torch.long
    float_dtype = torch.float32

    # Replace solvent attributes with empty tensors on the correct device/dtype.
    # Keep shapes consistent (1-D for x_s and batch_s; 2-D for positions; 2x0 for edge_index_s).
    batch_vac.x_s = torch.empty((0,), dtype=long_dtype, device=device)
    batch_vac.pos_s = torch.empty((0, 3), dtype=float_dtype, device=device)
    batch_vac.edge_index_s = torch.empty((2, 0), dtype=long_dtype, device=device)
    # Edge attributes are usually shape (num_edges, attr_dim). Use attr_dim 4 fallback if unknown.
    batch_vac.edge_attr_s = torch.empty((0, 4), dtype=float_dtype, device=device)
    batch_vac.batch_s = torch.empty((0,), dtype=long_dtype, device=device)

    return batch_vac


def run_inference(model, dataset_path, device, results_dict, active_tasks):
    """
    Runs inference on a dataset and APPENDS results to the shared results_dict.

    active_tasks: List of strings (e.g. ["lambda"] or ["phi"]) determining which
                  heads to record.
    """
    print(f"--- Processing: {dataset_path.name} (Tasks: {active_tasks}) ---")
    dataset = DatasetConstructor(str(dataset_path), split_by_mol_id=True)
    _, _, loader = dataset.get_dataloaders(num_workers=0)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {dataset_path.stem}"):
            batch = batch.to(device)

            # 1) Solvated pass (original batch)
            preds = model(batch)
            l_pred = _to_numpy_flat(preds.get("lambda"))
            p_pred = _to_numpy_flat(preds.get("phi"))

            # 2) Vacuum pass (explicitly empty solvent tensors)
            batch_vac = _make_vacuum_batch(batch, device)
            # ensure moved to device (if clone returned a Python reference)
            try:
                batch_vac = batch_vac.to(device)
            except Exception:
                pass

            preds_vac = model(batch_vac)
            l_vac = _to_numpy_flat(preds_vac.get("lambda"))
            p_vac = _to_numpy_flat(preds_vac.get("phi"))

            # 3) True targets (if present) -> make flat numpy arrays
            if "lambda" in active_tasks and hasattr(batch, "lambda_max"):
                l_true = _to_numpy_flat(batch.lambda_max)
                # extend lists with per-sample scalars
                results_dict["true_l"].extend([float(x) for x in l_true])
                results_dict["pred_l"].extend([float(x) for x in l_pred])
                results_dict["vac_l"].extend([float(x) for x in l_vac])

            if "phi" in active_tasks and hasattr(batch, "phi_delta"):
                p_true = _to_numpy_flat(batch.phi_delta)
                results_dict["true_p"].extend([float(x) for x in p_true])
                results_dict["pred_p"].extend([float(x) for x in p_pred])
                results_dict["vac_p"].extend([float(x) for x in p_vac])


def generate_report(results, out_dir):
    """Generates plots and metrics from the accumulated results."""
    print(f"\nGenerating Report in {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)

    metrics_log = []
    # Build solvent DataFrame with per-sample rows; columns will be 'lambda_shift' and 'phi_shift'
    df_solvent = pd.DataFrame()

    # --- LAMBDA ANALYSIS ---
    if len(results.get("true_l", [])) > 0:
        y_t = np.array(results["true_l"], dtype=float)
        y_p = np.array(results["pred_l"], dtype=float)
        y_v = np.array(results["vac_l"], dtype=float)

        # Basic mask for finite true values
        mask = np.isfinite(y_t)
        if mask.any():
            rmse_val = rmse(y_p[mask], y_t[mask])
            mae_val = mae(y_p[mask], y_t[mask])
            metrics_log.append(f"Lambda RMSE: {rmse_val:.4f} nm")
            metrics_log.append(f"Lambda MAE:  {mae_val:.4f} nm")

            if np.sum(mask) >= 2:
                try:
                    rho_val, _ = spearmanr(y_t[mask], y_p[mask])
                    tau_val, _ = kendalltau(y_t[mask], y_p[mask])
                    metrics_log.append(f"Lambda Spearman rho: {rho_val:.4f}")
                    metrics_log.append(f"Lambda Kendall tau: {tau_val:.4f}")
                    pw_acc = pairwise_accuracy(y_t[mask], y_p[mask])
                    if pw_acc is not None:
                        metrics_log.append(f"Lambda Pairwise Acc: {pw_acc:.4f}")
                except Exception:
                    pass

            parity_plot(
                y_t[mask],
                y_p[mask],
                "Absorption Max",
                "True",
                "Pred",
                out_dir / "lambda_parity.png",
            )

            l_bins = [-np.inf, 400, 500, 600, 700, np.inf]
            l_lbls = ["UV", "Blue", "Green", "Red", "NIR"]
            plot_confusion_matrix(
                y_t[mask],
                y_p[mask],
                l_bins,
                l_lbls,
                "Spectral Class",
                out_dir / "lambda_confusion.png",
            )

            # Compute lambda solvent shift (solvated - vacuum) per-sample
            # Place into df_solvent aligned by row
            shift = y_p - y_v
            # Ensure df_solvent has enough rows
            if df_solvent.shape[0] < len(shift):
                df_solvent = df_solvent.reindex(range(len(shift)))
            df_solvent.loc[: len(shift) - 1, "lambda_shift"] = shift

    # --- PHI ANALYSIS ---
    if len(results.get("true_p", [])) > 0:
        y_t = np.array(results["true_p"], dtype=float)
        y_p = np.array(results["pred_p"], dtype=float)
        y_v = np.array(results["vac_p"], dtype=float)

        # Exclude any dataset phi values > 1 from evaluation
        mask = np.isfinite(y_t) & (y_t >= 0) & (y_t <= 1)
        n_total = len(y_t)
        n_excluded = np.sum(np.isfinite(y_t) & (y_t > 1))
        if n_excluded > 0:
            metrics_log.append(
                f"Excluded {int(n_excluded)} phi entries > 1 from evaluation (out of {int(n_total)})"
            )

        if mask.any():
            rmse_val = rmse(y_p[mask], y_t[mask])
            mae_val = mae(y_p[mask], y_t[mask])
            metrics_log.append(f"Phi RMSE: {rmse_val:.4f}")
            metrics_log.append(f"Phi MAE:  {mae_val:.4f}")

            if np.sum(mask) >= 2:
                try:
                    rho_val, _ = spearmanr(y_t[mask], y_p[mask])
                    tau_val, _ = kendalltau(y_t[mask], y_p[mask])
                    metrics_log.append(f"Phi Spearman rho: {rho_val:.4f}")
                    metrics_log.append(f"Phi Kendall tau: {tau_val:.4f}")
                    pw_acc = pairwise_accuracy(y_t[mask], y_p[mask])
                    if pw_acc is not None:
                        metrics_log.append(f"Phi Pairwise Acc: {pw_acc:.4f}")
                except Exception:
                    pass

            parity_plot(
                y_t[mask],
                y_p[mask],
                "Quantum Yield",
                "True",
                "Pred",
                out_dir / "phi_parity.png",
            )

            p_bins = [-np.inf, 0.24, 0.62, np.inf]
            p_lbls = ["Low", "Med", "High"]
            plot_confusion_matrix(
                y_t[mask],
                y_p[mask],
                p_bins,
                p_lbls,
                "Yield Class",
                out_dir / "phi_confusion.png",
            )

            # Compute phi solvent shift (solvated - vacuum) for evaluated entries only.
            shift_full = y_p - y_v
            # Create array of NaNs and populate only for evaluated entries.
            phi_shift_arr = np.full_like(shift_full, np.nan, dtype=float)
            phi_shift_arr[mask] = shift_full[mask]

            # Ensure df_solvent has enough rows
            if df_solvent.shape[0] < len(phi_shift_arr):
                df_solvent = df_solvent.reindex(range(len(phi_shift_arr)))
            df_solvent.loc[: len(phi_shift_arr) - 1, "phi_shift"] = phi_shift_arr

    # --- SOLVENT PLOTS ---
    if not df_solvent.empty:
        plot_solvent_effects(df_solvent, out_dir)

    # --- SAVE METRICS ---
    with open(out_dir / "metrics.txt", "w") as f:
        f.write("\n".join(metrics_log))
    if metrics_log:
        print("\n  > " + "\n  > ".join(metrics_log))
    else:
        print("No metrics computed.")


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MANA Evaluation (Aggregated)")
    parser.add_argument("--model", required=True, help="Path to model .pth file")
    parser.add_argument(
        "--lambda_dataset", default=None, help="Path to Lambda H5 dataset"
    )
    parser.add_argument("--phi_dataset", default=None, help="Path to Phi H5 dataset")
    parser.add_argument("--output", required=True, help="Directory to save results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_path = Path(args.model)
    out_dir = Path(args.output)

    # 1. Load Model
    print(f"\nLoading Model: {model_path.name}")
    model = MANA(
        num_atom_types=NUM_ATOM_TYPES, hidden_dim=128, tasks=["lambda", "phi"]
    ).to(device)
    try:
        # Some saved state dicts include extra keys; load permissively
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Shared Container
    results = {
        "true_l": [],
        "pred_l": [],
        "vac_l": [],
        "true_p": [],
        "pred_p": [],
        "vac_p": [],
    }

    # 3. Process Datasets (With Strict Task Isolation)
    if args.lambda_dataset:
        run_inference(
            model, Path(args.lambda_dataset), device, results, active_tasks=["lambda"]
        )

    if args.phi_dataset:
        run_inference(
            model, Path(args.phi_dataset), device, results, active_tasks=["phi"]
        )

    if not args.lambda_dataset and not args.phi_dataset:
        print("Error: No datasets provided. Use --lambda_dataset or --phi_dataset.")
        return

    # 4. Generate Final Report
    generate_report(results, out_dir)


if __name__ == "__main__":
    main()
