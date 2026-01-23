import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr  # Added for correlation metrics

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent.parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(script_dir))

from model.mana_model import MANA  # noqa: E402
from data.dataset import DatasetConstructor  # noqa: E402

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# Must match the value used during training
NUM_ATOM_TYPES = 118


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def mae(pred, true):
    return np.mean(np.abs(pred - true))


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def parity_plot(y_true, y_pred, title, xlabel, ylabel, path):
    if len(y_true) == 0:
        return

    plt.figure(figsize=(6, 6))

    # Dynamic limits
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    buffer = (max_val - min_val) * 0.05
    lims = [min_val - buffer, max_val + buffer]

    plt.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    plt.scatter(y_true, y_pred, s=15, alpha=0.5, c="tab:blue", edgecolors="none")

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def residual_hist(residuals, title, xlabel, path):
    if len(residuals) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.hist(
        residuals,
        bins=50,
        color="tab:blue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Generic Evaluation Engine
# ---------------------------------------------------------------------
def _evaluate_model(device, dataset_path, model_path, tasks, phase_name, out_dir):
    """
    Generic evaluation loop that can handle any phase.
    """
    print("\n" + "=" * 80)
    print(f"TESTING {phase_name.upper()}")
    print("=" * 80)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return []

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return []

    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_path}")
    print(f"Tasks:   {tasks}")

    # Load Dataset (Always split by mol_id for rigorous testing)
    dataset = DatasetConstructor(
        str(dataset_path),
        cutoff_radius=5.0,
        batch_size=64,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
        split_by_mol_id=True,
    )

    _, _, test_loader = dataset.get_dataloaders(num_workers=0)

    # Handle Normalization Stats (Defaults if missing)
    l_mean = dataset.lambda_mean
    l_std = dataset.lambda_std
    if np.isnan(l_mean): l_mean = 500.0
    if np.isnan(l_std): l_std = 100.0

    # Load Model Structure
    model = MANA(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
        tasks=tasks,
        lambda_mean=l_mean,
        lambda_std=l_std,
    ).to(device)

    # Load Weights
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✓ Weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model weights: {e}")
        return []

    model.eval()

    # Results Accumulators
    results = {
        "lambda": {"true": [], "pred": []},
        "phi": {"true": [], "pred": []}
    }

    n_samples = len(test_loader.dataset) # pyright: ignore
    print(f"Evaluating on {n_samples} test samples...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {phase_name}", leave=False):
            batch = batch.to(device)
            preds = model(batch)

            # 1. Collect Lambda Results
            if "lambda" in tasks and hasattr(batch, "lambda_max"):
                y_true = batch.lambda_max.squeeze().cpu().numpy()
                y_pred = preds["lambda"].cpu().numpy()
                
                # Filter valid
                mask = np.isfinite(y_true)
                if mask.any():
                    results["lambda"]["true"].append(y_true[mask])
                    results["lambda"]["pred"].append(y_pred[mask])

            # 2. Collect Phi Results
            if "phi" in tasks and hasattr(batch, "phi_delta"):
                y_true = batch.phi_delta.squeeze().cpu().numpy()
                y_pred = preds["phi"].cpu().numpy()
                
                # Filter valid
                mask = np.isfinite(y_true) & (y_true >= 0)
                if mask.any():
                    results["phi"]["true"].append(y_true[mask])
                    results["phi"]["pred"].append(y_pred[mask])

    # Process Metrics & Plots
    metrics_log = []
    
    # Create subdirectory for this phase (e.g., results/phase_1_absorption)
    subdir = phase_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    phase_out_dir = out_dir / subdir
    os.makedirs(phase_out_dir, exist_ok=True)

    # --- Process Lambda ---
    if len(results["lambda"]["true"]) > 0:
        y_t = np.concatenate(results["lambda"]["true"])
        y_p = np.concatenate(results["lambda"]["pred"])
        
        l_rmse = rmse(y_p, y_t)
        l_mae = mae(y_p, y_t)
        
        metrics_log.append(f"[{phase_name}] Lambda RMSE: {l_rmse:.4f} nm")
        metrics_log.append(f"[{phase_name}] Lambda MAE:  {l_mae:.4f} nm")

        parity_plot(
            y_t, y_p, 
            f"{phase_name} - Absorption Max", 
            "True (nm)", "Pred (nm)", 
            phase_out_dir / "lambda_parity.png"
        )
        residual_hist(
            y_p - y_t, 
            f"{phase_name} - Lambda Residuals", 
            "Error (nm)", 
            phase_out_dir / "lambda_residuals.png"
        )
        print(f"✓ Lambda plots saved to {phase_out_dir}")

    # --- Process Phi ---
    if len(results["phi"]["true"]) > 0:
        y_t = np.concatenate(results["phi"]["true"])
        y_p = np.concatenate(results["phi"]["pred"])

        p_rmse = rmse(y_p, y_t)
        p_mae = mae(y_p, y_t)
        
        # Calculate Correlations (Need at least 2 points)
        if len(y_t) > 1:
            p_r, _ = pearsonr(y_t, y_p)
            s_rho, _ = spearmanr(y_t, y_p)
        else:
            p_r, s_rho = 0.0, 0.0

        metrics_log.append(f"[{phase_name}] Phi RMSE:     {p_rmse:.4f}")
        metrics_log.append(f"[{phase_name}] Phi MAE:      {p_mae:.4f}")
        metrics_log.append(f"[{phase_name}] Phi Pearson:  {p_r:.4f}")
        metrics_log.append(f"[{phase_name}] Phi Spearman: {s_rho:.4f}")

        parity_plot(
            y_t, y_p, 
            f"{phase_name} - Quantum Yield", 
            "True", "Pred", 
            phase_out_dir / "phi_parity.png"
        )
        residual_hist(
            y_p - y_t, 
            f"{phase_name} - Phi Residuals", 
            "Error", 
            phase_out_dir / "phi_residuals.png"
        )
        print(f"✓ Phi plots saved to {phase_out_dir}")

    if not metrics_log:
        print("WARNING: No valid targets found for evaluation.")

    return metrics_log


# ---------------------------------------------------------------------
# Phase Wrappers
# ---------------------------------------------------------------------

def test_phase1(device, out_dir):
    """Test Phase 1: General Absorption (Lambda Only)"""
    return _evaluate_model(
        device,
        dataset_path = project_root / "data" / "lambda" / "lambda_only_data.h5",
        model_path = project_root / "models" / "lambda" / "best_model.pth",
        tasks = ["lambda"],
        phase_name = "Phase 1 (Absorption)",
        out_dir = out_dir
    )

def test_phase2(device, out_dir):
    """Test Phase 2: Fluorescence (Lambda + Phi)"""
    return _evaluate_model(
        device,
        dataset_path = project_root / "data" / "fluor" / "fluorescence_data.h5",
        model_path = project_root / "models" / "fluor" / "best_model.pth",
        tasks = ["lambda", "phi"],
        phase_name = "Phase 2 (Fluorescence)",
        out_dir = out_dir
    )

def test_phase3(device, out_dir):
    """Test Phase 3: Singlet Oxygen (Phi Only)"""
    return _evaluate_model(
        device,
        dataset_path = project_root / "data" / "phi" / "phidelta_data.h5",
        model_path = project_root / "models" / "phi" / "best_model.pth",
        tasks = ["phi"],
        phase_name = "Phase 3 (Singlet Oxygen)",
        out_dir = out_dir
    )


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MANA Model Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --p1          # Test Phase 1 (Absorption)
  python test.py --p3          # Test Phase 3 (Singlet Oxygen)
  python test.py --cpu         # Force CPU usage
  python test.py               # Test ALL phases (default)
        """
    )
    
    parser.add_argument("--p1", action="store_true", help="Test Phase 1 Model (Absorption)")
    parser.add_argument("--p2", action="store_true", help="Test Phase 2 Model (Fluorescence)")
    parser.add_argument("--p3", action="store_true", help="Test Phase 3 Model (Singlet Oxygen)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU device")

    args = parser.parse_args()

    # Default: Test everything if no specific flag is set
    if not (args.p1 or args.p2 or args.p3):
        print("No phase selected. Defaulting to testing ALL phases.")
        args.p1 = True
        args.p2 = True
        args.p3 = True

    # Device Setup
    if args.cpu:
        device = torch.device("cpu")
        print("Device: cpu (forced)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print(f"Device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Device: {device}")
    
    # Output Directory
    out_dir = project_root / "results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Run Tests
    all_metrics = []

    if args.p1:
        all_metrics.extend(test_phase1(device, out_dir))
    
    if args.p2:
        all_metrics.extend(test_phase2(device, out_dir))

    if args.p3:
        all_metrics.extend(test_phase3(device, out_dir))

    # Summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    
    if all_metrics:
        for m in all_metrics:
            print(m)
        
        # Save summary
        with open(out_dir / "summary_metrics.txt", "w") as f:
            f.write("\n".join(all_metrics))
        print(f"\nSummary saved to: {out_dir / 'summary_metrics.txt'}")
    else:
        print("No metrics generated.")

if __name__ == "__main__":
    main()