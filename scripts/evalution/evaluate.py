import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau
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
    pass

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ---------------------------------------------------------------------
# Configuration & Colors
# ---------------------------------------------------------------------
NUM_ATOM_TYPES = 118
C_PRIMARY = "#565AA2"   # Indigo
C_ACCENT  = "#F6A21C"   # Orange
C_BLACK   = "#000000"
C_GRAY    = "#B5BCBE"

def set_style():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.edgecolor'] = C_BLACK
    plt.rcParams['text.color'] = C_BLACK
    plt.rcParams['grid.color'] = C_GRAY
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------
def rmse(pred, true): 
    return np.sqrt(np.mean((pred - true) ** 2))

def parity_plot(y_true, y_pred, title, xlabel, ylabel, path):
    if len(y_true) == 0: return
    set_style()
    plt.figure(figsize=(6, 6))
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    buffer = (max_val - min_val) * 0.05
    lims = [min_val - buffer, max_val + buffer]
    plt.plot(lims, lims, color=C_BLACK, linestyle="--", alpha=0.7, zorder=1)
    plt.scatter(y_true, y_pred, s=20, alpha=0.6, c=C_PRIMARY, edgecolors="none", zorder=2)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel(xlabel, fontweight='bold'); plt.ylabel(ylabel, fontweight='bold')
    plt.title(title, fontweight='bold'); plt.grid(True); plt.tight_layout()
    plt.savefig(path); plt.close()

def plot_confusion_matrix(y_true, y_pred, bins, labels, title, path):
    if len(y_true) == 0: return
    set_style()
    y_true_cat = pd.cut(y_true, bins=bins, labels=labels).astype(str)
    y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels).astype(str)
    valid = (y_true_cat != 'nan') & (y_pred_cat != 'nan')
    y_true_cat, y_pred_cat = y_true_cat[valid], y_pred_cat[valid]
    if len(y_true_cat) == 0: return

    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".1%", cmap=sns.light_palette(C_PRIMARY, as_cmap=True), 
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title, fontweight='bold'); plt.ylabel("True Class"); plt.xlabel("Predicted Class")
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_solvent_effects(df, out_dir):
    if df.empty: return
    set_style()

    # 1. Lambda Shift Histogram
    if 'lambda_shift' in df.columns and not df['lambda_shift'].dropna().empty:
        plt.figure(figsize=(6, 4))
        plt.hist(df['lambda_shift'].dropna(), bins=30, color=C_ACCENT, edgecolor=C_BLACK, alpha=0.8)
        plt.title("Predicted Solvatochromism (Solvent - Vacuum)", fontweight='bold')
        plt.xlabel("$\Delta \lambda_{max}$ (nm)", fontweight='bold')
        plt.ylabel("Count", fontweight='bold')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(out_dir / "solvatochromism_hist.png"); plt.close()

    # 2. Phi Change Histogram
    if 'phi_shift' in df.columns and not df['phi_shift'].dropna().empty:
        plt.figure(figsize=(6, 4))
        plt.hist(df['phi_shift'].dropna(), bins=30, color=C_PRIMARY, edgecolor=C_BLACK, alpha=0.8)
        plt.title("Predicted Solvent Effect on Yield", fontweight='bold')
        plt.xlabel("$\Delta \Phi_{\Delta}$ (Solvated - Vacuum)", fontweight='bold')
        plt.ylabel("Count", fontweight='bold')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(out_dir / "phi_solvent_effect_hist.png"); plt.close()

# ---------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------
def run_inference(model, dataset_path, device, results_dict, active_tasks):
    """
    Runs inference on a dataset and APPENDS results to the shared results_dict.
    active_tasks: List of strings (e.g. ["lambda"] or ["phi"]) determining which heads to record.
    """
    print(f"--- Processing: {dataset_path.name} (Tasks: {active_tasks}) ---")
    dataset = DatasetConstructor(str(dataset_path), split_by_mol_id=True)
    _, _, loader = dataset.get_dataloaders(num_workers=0) 

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {dataset_path.stem}"):
            batch = batch.to(device)

            # 1. Solvated Pass
            preds = model(batch)
            l_pred = preds["lambda"].cpu().numpy()
            p_pred = preds["phi"].cpu().numpy()

            # 2. Vacuum Pass
            batch_vac = batch.clone() 
            if hasattr(batch_vac, 'x_s'):
                batch_vac.x_s = torch.tensor([], dtype=torch.long, device=device)
                batch_vac.pos_s = torch.tensor([], dtype=torch.float32, device=device)
                batch_vac.edge_index_s = torch.empty((2, 0), dtype=torch.long, device=device)
                batch_vac.edge_attr_s = torch.empty((0, 4), device=device)
                batch_vac.batch_s = torch.tensor([], dtype=torch.long, device=device)

            preds_vac = model(batch_vac)
            l_vac = preds_vac["lambda"].cpu().numpy()
            p_vac = preds_vac["phi"].cpu().numpy()

            # 3. Store Data (Strictly based on active_tasks)

            # --- LAMBDA STORAGE ---
            if "lambda" in active_tasks and hasattr(batch, "lambda_max"):
                l_true = batch.lambda_max.cpu().numpy()
                results_dict["true_l"].extend(l_true)
                results_dict["pred_l"].extend(l_pred)
                results_dict["vac_l"].extend(l_vac)

            # --- PHI STORAGE ---
            if "phi" in active_tasks and hasattr(batch, "phi_delta"):
                p_true = batch.phi_delta.cpu().numpy()
                results_dict["true_p"].extend(p_true)
                results_dict["pred_p"].extend(p_pred)
                results_dict["vac_p"].extend(p_vac)

def generate_report(results, out_dir):
    """Generates plots and metrics from the accumulated results."""
    print(f"\nGenerating Report in {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)

    metrics_log = []
    df_solvent = pd.DataFrame() 

    # --- LAMBDA ANALYSIS ---
    if len(results["true_l"]) > 0:
        y_t = np.array(results["true_l"])
        y_p = np.array(results["pred_l"])
        y_v = np.array(results["vac_l"])

        mask = np.isfinite(y_t)
        if mask.any():
            rmse_val = rmse(y_p[mask], y_t[mask])
            metrics_log.append(f"Lambda RMSE: {rmse_val:.4f} nm")

            # Correlations for lambda (if enough points)
            if np.sum(mask) >= 2:
                try:
                    r_val, _ = pearsonr(y_t[mask], y_p[mask])
                    rho_val, _ = spearmanr(y_t[mask], y_p[mask])
                    tau_val, _ = kendalltau(y_t[mask], y_p[mask])
                    metrics_log.append(f"Lambda Pearson r: {r_val:.4f}")
                    metrics_log.append(f"Lambda Spearman rho: {rho_val:.4f}")
                    metrics_log.append(f"Lambda Kendall tau: {tau_val:.4f}")
                except Exception:
                    pass

            parity_plot(y_t[mask], y_p[mask], "Absorption Max", "True", "Pred", out_dir / "lambda_parity.png")

            l_bins = [-np.inf, 400, 500, 600, 700, np.inf]
            l_lbls = ["UV", "Blue", "Green", "Red", "NIR"]
            plot_confusion_matrix(y_t[mask], y_p[mask], l_bins, l_lbls, "Spectral Class", out_dir / "lambda_confusion.png")


            # Append to solvent dataframe
            shift = y_p - y_v
            df_temp = pd.DataFrame({'lambda_shift': shift})
            df_solvent = pd.concat([df_solvent, df_temp], axis=1)

    # --- PHI ANALYSIS ---
    if len(results["true_p"]) > 0:
        y_t = np.array(results["true_p"])
        y_p = np.array(results["pred_p"])
        y_v = np.array(results["vac_p"])

        # Exclude any dataset phi values > 1 from evaluation
        mask = np.isfinite(y_t) & (y_t >= 0) & (y_t <= 1)
        n_total = len(y_t)
        n_excluded = np.sum(np.isfinite(y_t) & (y_t > 1))
        if n_excluded > 0:
            metrics_log.append(f"Excluded {int(n_excluded)} phi entries > 1 from evaluation (out of {int(n_total)})")

        if mask.any():
            rmse_val = rmse(y_p[mask], y_t[mask])
            metrics_log.append(f"Phi RMSE:    {rmse_val:.4f}")

            # Correlations for phi (if enough points)
            if np.sum(mask) >= 2:
                try:
                    r_val, _ = pearsonr(y_t[mask], y_p[mask])
                    rho_val, _ = spearmanr(y_t[mask], y_p[mask])
                    tau_val, _ = kendalltau(y_t[mask], y_p[mask])
                    metrics_log.append(f"Phi Pearson r: {r_val:.4f}")
                    metrics_log.append(f"Phi Spearman rho: {rho_val:.4f}")
                    metrics_log.append(f"Phi Kendall tau: {tau_val:.4f}")
                except Exception:
                    pass

            parity_plot(y_t[mask], y_p[mask], "Quantum Yield", "True", "Pred", out_dir / "phi_parity.png")

            p_bins = [-np.inf, 0.33, 0.66, np.inf]
            p_lbls = ["Low", "Med", "High"]
            plot_confusion_matrix(y_t[mask], y_p[mask], p_bins, p_lbls, "Yield Class", out_dir / "phi_confusion.png")


            # Use only the evaluated entries for solvent shift histogram (align to evaluation)
            shift = (y_p - y_v)[mask]
            df_temp = pd.DataFrame({'phi_shift': shift})

            # Align lengths if needed (simple concat for histograms)
            if 'lambda_shift' in df_solvent.columns:
                if len(df_temp) > len(df_solvent):
                    df_solvent = df_solvent.reindex(range(len(df_temp)))
                elif len(df_solvent) > len(df_temp):
                    df_temp = df_temp.reindex(range(len(df_solvent)))
                df_solvent['phi_shift'] = df_temp['phi_shift']
            else:
                df_solvent = pd.concat([df_solvent, df_temp], axis=1)

    # --- SOLVENT PLOTS ---
    if not df_solvent.empty:
        plot_solvent_effects(df_solvent, out_dir)

    # --- SAVE METRICS ---
    with open(out_dir / "metrics.txt", "w") as f:
        f.write("\n".join(metrics_log))
    print("\n  > " + "\n  > ".join(metrics_log))

# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MANA Evaluation (Aggregated)")
    parser.add_argument("--model", required=True, help="Path to model .pth file")
    parser.add_argument("--lambda_dataset", default=None, help="Path to Lambda H5 dataset")
    parser.add_argument("--phi_dataset", default=None, help="Path to Phi H5 dataset")
    parser.add_argument("--output", required=True, help="Directory to save results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    out_dir = Path(args.output)

    # 1. Load Model
    print(f"\nLoading Model: {model_path.name}")
    model = MANA(num_atom_types=NUM_ATOM_TYPES, hidden_dim=128, tasks=["lambda", "phi"]).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}"); return

    # 2. Shared Container
    results = {"true_l": [], "pred_l": [], "vac_l": [], 
               "true_p": [], "pred_p": [], "vac_p": []}

    # 3. Process Datasets (With Strict Task Isolation)
    if args.lambda_dataset:
        # STRICTLY process only Lambda
        run_inference(model, Path(args.lambda_dataset), device, results, active_tasks=["lambda"])

    if args.phi_dataset:
        # STRICTLY process only Phi
        run_inference(model, Path(args.phi_dataset), device, results, active_tasks=["phi"])

    if not args.lambda_dataset and not args.phi_dataset:
        print("Error: No datasets provided. Use --lambda_dataset or --phi_dataset.")
        return

    # 4. Generate Final Report
    generate_report(results, out_dir)

if __name__ == "__main__":
    main()