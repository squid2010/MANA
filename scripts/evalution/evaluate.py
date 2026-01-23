import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Try importing RDKit, handle gracefully if missing (though required for Family analysis)
try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not found. Chemical family analysis will be skipped.")

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent.parent
project_root = script_dir.parent
sys.path.insert(0, str(script_dir))

# Attempt imports from local project structure
try:
    from model.mana_model import MANA
    from data.dataset import DatasetConstructor
except ImportError:
    # Mocking for standalone syntax checking if project structure is missing
    pass

# ---------------------------------------------------------------------
# Configuration & Colors (ISEF Scheme)
# ---------------------------------------------------------------------
NUM_ATOM_TYPES = 118

C_PRIMARY   = "#565AA2"   # Purple/Indigo (Data/True)
C_ACCENT    = "#F6A21C"   # Orange (Predictions/Fit)
C_BLACK     = "#000000"
C_GRAY      = "#B5BCBE"

def set_style():
    """Applies the custom color scheme to global plot settings"""
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.edgecolor'] = C_BLACK
    plt.rcParams['axes.labelcolor'] = C_BLACK
    plt.rcParams['xtick.color'] = C_BLACK
    plt.rcParams['ytick.color'] = C_BLACK
    plt.rcParams['text.color'] = C_BLACK
    plt.rcParams['grid.color'] = C_GRAY
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------------------
# Chemical Families Logic
# ---------------------------------------------------------------------
FAMILY_PATTERNS = {
    "Porphyrin": "[n]1cc2[nH]c(cc3[n]c(cc4[nH]c(cc1c2)cc4)cc3)c",
    "BODIPY": "[B-](F)(F)[n]1cc2cc[n+](c2c1)",
    "Phthalocyanine": "c12c(n3c(n4c(n5c(n1)c1ccccc15)c1ccccc14)c1ccccc13)ccccc2",
    "Xanthene": "O=C1C=CC2=C(O1)C=CC(=O)C2",
    "Coumarin": "O=C1OC=CC2=CC=CC=C12",
    "Flavone": "O=C1C=C(O2)C(C3=CC=CC=C3)=CC2=CC1",
    "Other": "*"
}

def identify_family(smiles):
    if not HAS_RDKIT: 
        return "Unknown"
    if not isinstance(smiles, str): 
        return "Invalid"
    mol = Chem.MolFromSmiles(smiles) # pyright: ignore[reportAttributeAccessIssue, reportPossiblyUnboundVariable]
    if not mol: 
        return "Invalid"
    for name, smarts in FAMILY_PATTERNS.items():
        if name == "Other": 
            continue
        pattern = Chem.MolFromSmarts(smarts) # pyright: ignore[reportAttributeAccessIssue, reportPossiblyUnboundVariable]
        if pattern and mol.HasSubstructMatch(pattern): 
            return name
    return "Other"

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def mae(pred, true):
    return np.mean(np.abs(pred - true))

def get_pairwise_accuracy(y_true, y_pred):
    n = len(y_true)
    if n < 2: 
        return 0.0
    # Subsample if large to prevent O(N^2) memory issues
    if n > 2000:
        indices = np.random.choice(n, 2000, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    diff_true = y_true[:, None] - y_true[None, :]
    diff_pred = y_pred[:, None] - y_pred[None, :]
    
    # Valid pairs have different ground truth values
    valid_pairs = diff_true != 0
    # Concordant pairs have the same sign difference
    concordant = (np.sign(diff_true) == np.sign(diff_pred)) & valid_pairs
    
    if valid_pairs.sum() == 0: 
        return 0.0
    return concordant.sum() / valid_pairs.sum()

# ---------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------
def parity_plot(y_true, y_pred, title, xlabel, ylabel, path):
    if len(y_true) == 0: 
        return
    set_style()
    plt.figure(figsize=(6, 6))

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    buffer = (max_val - min_val) * 0.05
    lims = [min_val - buffer, max_val + buffer]

    plt.plot(lims, lims, color=C_BLACK, linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    plt.scatter(y_true, y_pred, s=20, alpha=0.6, c=C_PRIMARY, edgecolors="none", zorder=2)

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def residual_hist(residuals, title, xlabel, path):
    if len(residuals) == 0: 
        return
    set_style()
    plt.figure(figsize=(6, 4))
    
    plt.hist(residuals, bins=50, color=C_PRIMARY, alpha=0.8, edgecolor=C_BLACK, linewidth=0.8)
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_generic_confusion_matrix(y_true, y_pred, bins, labels, title, path):
    """
    Generic function to plot confusion matrix for continuous variables by binning them.
    """
    if len(y_true) == 0: 
        return
    set_style()

    # Convert continuous values to categorical strings
    y_true_cat = pd.cut(y_true, bins=bins, labels=labels).astype(str) # pyright: ignore[reportAttributeAccessIssue]
    y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels).astype(str) # pyright: ignore[reportAttributeAccessIssue]
    
    # Filter out 'nan' strings if any values fell outside bins
    valid_mask = (y_true_cat != 'nan') & (y_pred_cat != 'nan')
    y_true_cat = y_true_cat[valid_mask]
    y_pred_cat = y_pred_cat[valid_mask]

    if len(y_true_cat) == 0: 
        return

    # Compute CM
    # We use labels=labels to ensure the order of the matrix matches the bins
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels, normalize='true')
    
    plt.figure(figsize=(8, 6))
    cmap = sns.light_palette(C_PRIMARY, as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt=".1%", cmap=cmap, 
                xticklabels=labels, yticklabels=labels, 
                linecolor=C_BLACK, linewidths=0.5, cbar=False)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("True Class", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_family_performance(df, task_name, out_dir):
    if 'family' not in df.columns or df.empty: 
        return
    set_style()
    
    families = df['family'].unique()
    family_stats = []
    
    for fam in families:
        subset = df[df['family'] == fam]
        if len(subset) < 5: 
            continue
        err = np.sqrt(((subset['true'] - subset['pred']) ** 2).mean())
        family_stats.append({'Family': fam, 'RMSE': err, 'Count': len(subset)})
    
    if not family_stats: 
        return

    stats_df = pd.DataFrame(family_stats).sort_values('RMSE')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=stats_df, x='Family', y='RMSE', color=C_PRIMARY, edgecolor=C_BLACK)
    
    plt.title(f"{task_name} Error by Chemical Family", fontsize=14, fontweight='bold')
    plt.ylabel("RMSE", fontsize=12, fontweight='bold')
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    
    # Annotate counts
    for i, row in enumerate(stats_df.itertuples()):
        plt.text(i, row.RMSE, f"n={row.Count}", ha='center', va='bottom', color=C_BLACK, fontsize=9) # pyright: ignore[reportAttributeAccessIssue]
        
    plt.tight_layout()
    plt.savefig(out_dir / f"{task_name.lower()}_rmse_by_family.png")
    plt.close()

# ---------------------------------------------------------------------
# Core Evaluation Logic
# ---------------------------------------------------------------------
def run_evaluation(device, dataset_path, model_path, tasks, phase_name, out_dir, deep_analysis=False):
    """
    Main driver function that handles data loading, inference, and plotting.
    """
    print(f"\n[{phase_name}] Starting evaluation...")
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return []
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return []

    # Load Dataset
    dataset = DatasetConstructor(str(dataset_path), split_by_mol_id=True) # pyright: ignore[reportPossiblyUnboundVariable]
    _, _, test_loader = dataset.get_dataloaders(num_workers=0)

    # Stats for normalization
    l_mean = dataset.lambda_mean if not np.isnan(dataset.lambda_mean) else 500.0
    l_std = dataset.lambda_std if not np.isnan(dataset.lambda_std) else 100.0

    # Initialize Model
    model = MANA( # pyright: ignore[reportPossiblyUnboundVariable]
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=128,
        tasks=tasks,
        lambda_mean=l_mean,
        lambda_std=l_std,
    ).to(device)

    # Load Weights
    try:
        # strict=False allows loading a model trained on 2 tasks into a wrapper set for 1 task, etc.
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False) 
    except Exception as e:
        print(f"ERROR: Could not load weights: {e}")
        return []

    model.eval()

    # Containers
    data_log = {
        "lambda": {"true": [], "pred": [], "smiles": []},
        "phi":    {"true": [], "pred": [], "smiles": []}
    }

    print(f"Processing {len(test_loader.dataset)} samples...") # pyright: ignore[reportArgumentType]
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=phase_name, leave=False):
            batch = batch.to(device)
            preds = model(batch)
            
            # Helper to extract SMILES if available
            batch_smiles = batch.smiles if hasattr(batch, 'smiles') else [""] * batch.num_graphs

            # --- Lambda Extraction ---
            if "lambda" in tasks and hasattr(batch, "lambda_max") and "lambda" in preds:
                y_true = batch.lambda_max.squeeze().cpu().numpy()
                y_pred = preds["lambda"].squeeze().cpu().numpy()
                
                # Handle single item batches ensuring array shape
                if y_true.ndim == 0: 
                    y_true = y_true.reshape(1)
                    y_pred = y_pred.reshape(1)

                mask = np.isfinite(y_true)
                if mask.any():
                    data_log["lambda"]["true"].extend(y_true[mask])
                    data_log["lambda"]["pred"].extend(y_pred[mask])
                    data_log["lambda"]["smiles"].extend(np.array(batch_smiles)[mask])

            # --- Phi Extraction ---
            if "phi" in tasks and hasattr(batch, "phi_delta") and "phi" in preds:
                y_true = batch.phi_delta.squeeze().cpu().numpy()
                y_pred = preds["phi"].squeeze().cpu().numpy()

                if y_true.ndim == 0: 
                    y_true = y_true.reshape(1)
                    y_pred = y_pred.reshape(1)

                # Filter valid Phi (0 to 1 range typically, or just non-nan)
                mask = np.isfinite(y_true) & (y_true >= 0)
                if mask.any():
                    data_log["phi"]["true"].extend(y_true[mask])
                    data_log["phi"]["pred"].extend(y_pred[mask])
                    data_log["phi"]["smiles"].extend(np.array(batch_smiles)[mask])

    # Setup Output
    subdir = phase_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    phase_out_dir = out_dir / subdir
    os.makedirs(phase_out_dir, exist_ok=True)
    
    metrics_log = []

    # ==========================
    # PROCESS LAMBDA RESULTS
    # ==========================
    if len(data_log["lambda"]["true"]) > 0:
        y_t = np.array(data_log["lambda"]["true"])
        y_p = np.array(data_log["lambda"]["pred"])
        smiles = data_log["lambda"]["smiles"]

        l_rmse = rmse(y_p, y_t)
        l_mae = mae(y_p, y_t)
        metrics_log.append(f"[{phase_name}] Lambda RMSE: {l_rmse:.4f} nm")
        metrics_log.append(f"[{phase_name}] Lambda MAE:  {l_mae:.4f} nm")

        # Standard Plots
        parity_plot(y_t, y_p, f"{phase_name} - Absorption Max", "True (nm)", "Pred (nm)", phase_out_dir / "lambda_parity.png")
        residual_hist(y_p - y_t, f"{phase_name} - Lambda Residuals", "Error (nm)", phase_out_dir / "lambda_residuals.png")

        # --- Deep Analysis (Confusion Matrix) ---
        if deep_analysis:
            # 1. Confusion Matrix
            # Bins based on visible spectrum: UV, Blue/Violet, Green/Yellow, Orange/Red, NIR
            l_bins = [-np.inf, 400, 500, 600, 700, np.inf]
            l_labels = ["UV (<400)", "Blue (400-500)", "Green (500-600)", "Red (600-700)", "NIR (>700)"]
            plot_generic_confusion_matrix(y_t, y_p, l_bins, l_labels, "Absorption Region Classification", phase_out_dir / "lambda_confusion_matrix.png")

            # 2. Family Analysis
            if HAS_RDKIT:
                df_l = pd.DataFrame({'true': y_t, 'pred': y_p, 'smiles': smiles})
                df_l['family'] = df_l['smiles'].apply(identify_family)
                plot_family_performance(df_l, "Lambda", phase_out_dir)

    # ==========================
    # PROCESS PHI RESULTS
    # ==========================
    if len(data_log["phi"]["true"]) > 0:
        y_t = np.array(data_log["phi"]["true"])
        y_p = np.array(data_log["phi"]["pred"])
        smiles = data_log["phi"]["smiles"]

        p_rmse = rmse(y_p, y_t)
        #p_mae = mae(y_p, y_t)
        
        if len(y_t) > 1:
            p_r, _ = pearsonr(y_t, y_p)
            #s_rho, _ = spearmanr(y_t, y_p)
        else:
            p_r = 0.0
            #p_r, s_rho = 0.0, 0.0

        metrics_log.append(f"[{phase_name}] Phi RMSE:     {p_rmse:.4f}")
        metrics_log.append(f"[{phase_name}] Phi Pearson:  {p_r:.4f}")
        
        # Standard Plots
        parity_plot(y_t, y_p, f"{phase_name} - Quantum Yield", "True", "Pred", phase_out_dir / "phi_parity.png")
        residual_hist(y_p - y_t, f"{phase_name} - Phi Residuals", "Error", phase_out_dir / "phi_residuals.png")

        # --- Deep Analysis ---
        if deep_analysis:
            # 1. Pairwise Accuracy
            acc = get_pairwise_accuracy(y_t, y_p)
            metrics_log.append(f"[{phase_name}] Phi Ranking Acc: {acc:.4f}")

            # 2. Confusion Matrix
            # Bins: Low, Med, High
            p_bins = [-np.inf, 0.1, 0.5, np.inf]
            p_labels = ["Low (<0.1)", "Med (0.1-0.5)", "High (>0.5)"]
            plot_generic_confusion_matrix(y_t, y_p, p_bins, p_labels, "Quantum Yield Classification", phase_out_dir / "phi_confusion_matrix.png")

            # 3. Family Analysis
            if HAS_RDKIT:
                df_p = pd.DataFrame({'true': y_t, 'pred': y_p, 'smiles': smiles})
                df_p['family'] = df_p['smiles'].apply(identify_family)
                plot_family_performance(df_p, "Phi", phase_out_dir)

    # Write logs
    with open(phase_out_dir / "metrics.txt", "w") as f:
        f.write("\n".join(metrics_log))
    
    return metrics_log

# ---------------------------------------------------------------------
# Evaluation Runners
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comprehensive MANA Evaluation")
    parser.add_argument("--p1", action="store_true", help="Run Phase 1 (Lambda Only)")
    parser.add_argument("--p2", action="store_true", help="Run Phase 2 (Fluorescence)")
    parser.add_argument("--p3", action="store_true", help="Run Phase 3 (Singlet Oxygen / Final)")
    parser.add_argument("--deep", action="store_true", help="Enable deep analysis (Confusion Matrices, Chemical Families)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    # If no specific phase selected, run all
    if not (args.p1 or args.p2 or args.p3):
        print("No specific phase selected. Running all phases.")
        args.p1, args.p2, args.p3 = True, True, True

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = project_root / "results" / "combined_eval"
    os.makedirs(out_dir, exist_ok=True)
    
    all_metrics = []

    # --- Phase 1: Lambda Only ---
    if args.p1:
        metrics = run_evaluation(
            device,
            dataset_path=project_root / "data" / "lambda" / "lambda_only_data.h5",
            model_path=project_root / "models" / "lambda" / "best_model.pth",
            tasks=["lambda"],
            phase_name="Phase 1 (Absorption)",
            out_dir=out_dir,
            deep_analysis=args.deep
        )
        all_metrics.extend(metrics)

    # --- Phase 2: Fluorescence ---
    if args.p2:
        metrics = run_evaluation(
            device,
            dataset_path=project_root / "data" / "fluor" / "fluorescence_data.h5",
            model_path=project_root / "models" / "fluor" / "best_model.pth",
            tasks=["lambda", "phi"],
            phase_name="Phase 2 (Fluorescence)",
            out_dir=out_dir,
            deep_analysis=args.deep
        )
        all_metrics.extend(metrics)

    # --- Phase 3: Singlet Oxygen (Phi Focus) ---
    if args.p3:
        # Test on Phi Data
        metrics_phi = run_evaluation(
            device,
            dataset_path=project_root / "data" / "phi" / "phidelta_data.h5",
            model_path=project_root / "models" / "phi" / "best_model.pth",
            tasks=["phi"],
            phase_name="Phase 3 (Singlet Oxygen)",
            out_dir=out_dir,
            deep_analysis=args.deep
        )
        all_metrics.extend(metrics_phi)

        # Test Lambda retention in final model
        metrics_lam = run_evaluation(
            device,
            dataset_path=project_root / "data" / "lambda" / "lambda_all_data.h5",
            model_path=project_root / "models" / "phi" / "best_model.pth",
            tasks=["lambda"],
            phase_name="Phase 3 (Abs Retention)",
            out_dir=out_dir,
            deep_analysis=args.deep
        )
        all_metrics.extend(metrics_lam)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for m in all_metrics:
        print(m)
    
    with open(out_dir / "summary_metrics.txt", "w") as f:
        f.write("\n".join(all_metrics))

    print(f"\nResults saved to: {out_dir}")

if __name__ == "__main__":
    main()