import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
# Adjust this depending on where you run it. 
# If running from 'scripts/', parent.parent is root.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(script_dir.parent))

print(f"Project Root: {project_root}")

from model.mana_model import MANA  # noqa: E402
from data.dataset import DatasetConstructor  # noqa: E402

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
    if len(y_true) == 0: return # Skip if empty
    
    plt.figure(figsize=(6, 6))
    
    # Dynamic limits
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    buffer = (max_val - min_val) * 0.05
    lims = [min_val - buffer, max_val + buffer]

    plt.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    plt.scatter(y_true, y_pred, s=15, alpha=0.5, c='tab:blue', edgecolors='none')
    
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
    if len(residuals) == 0: return

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, color='tab:blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print("MANA MODEL – TEST METRICS & DIAGNOSTICS")
    print("=" * 80)

    # 1. Setup Paths (Works for Local & Colab)
    # Checks for Colab-style path first, then local structure
    if os.path.exists("deep4chem_data.h5"):
        dataset_path = "deep4chem_data.h5"
        model_dir = "models"
    else:
        dataset_path = project_root / "data" / "deep4chem_data.h5"
        model_dir = project_root / "models"

    model_path = os.path.join(model_dir, "best_model.pth")
    out_dir = os.path.join(model_dir, "test_results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_path}")

    # 2. Load Dataset (Only Test Split)
    # We set batch_size=1 to ensure no padding weirdness during evaluation logic, 
    # though >1 works fine too.
    dataset = DatasetConstructor(
        str(dataset_path),
        cutoff_radius=5.0,
        batch_size=64, 
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
    )

    _, _, test_loader = dataset.get_dataloaders(num_workers=0)

    # 3. Load Model
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    else:
        device = torch.device("cpu")
        
    print(f"Device: {device}")

    # Initialize model with correct architecture
    # IMPORTANT: 'tasks' must match what you trained with. 
    # If unsure, we enable both heads. The weights for the unused one won't matter.
    model = MANA(
        num_atom_types=dataset.num_atom_types,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
        tasks=["lambda", "phi"], 
    ).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Model weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model weights: {e}")
        return

    model.eval()

    # 4. Evaluation Loop
    lambda_true, lambda_pred = [], []
    phi_true, phi_pred = [], []

    print(f"Evaluating on {len(test_loader.dataset)} test samples...")
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            
            # --- Lambda Max ---
            # Get true values and masks
            if hasattr(batch, "lambda_max"):
                y_true = batch.lambda_max.squeeze().cpu().numpy()
                y_pred = preds["lambda"].cpu().numpy()
                
                # Filter NaNs (Crucial for real datasets)
                mask = np.isfinite(y_true)
                if mask.any():
                    lambda_true.append(y_true[mask])
                    lambda_pred.append(y_pred[mask])

            # --- Phi Delta ---
            if hasattr(batch, "phi_delta") and "phi" in preds:
                y_true = batch.phi_delta.squeeze().cpu().numpy()
                y_pred = preds["phi"].cpu().numpy()
                
                mask = np.isfinite(y_true)
                if mask.any():
                    phi_true.append(y_true[mask])
                    phi_pred.append(y_pred[mask])

    # 5. Process & Save Metrics
    metrics_str = []

    # --- Process Lambda ---
    if len(lambda_true) > 0:
        lambda_true = np.concatenate(lambda_true)
        lambda_pred = np.concatenate(lambda_pred)
        
        l_rmse = rmse(lambda_pred, lambda_true)
        l_mae = mae(lambda_pred, lambda_true)
        
        metrics_str.append(f"Lambda_max RMSE: {l_rmse:.4f}")
        metrics_str.append(f"Lambda_max MAE:  {l_mae:.4f}")
        
        parity_plot(lambda_true, lambda_pred, "Absorption Max (nm)", "True", "Predicted", 
                   os.path.join(out_dir, "lambda_parity.png"))
        residual_hist(lambda_pred - lambda_true, "Lambda Residuals", "Error (nm)", 
                     os.path.join(out_dir, "lambda_residuals.png"))
    else:
        print("WARNING: No valid Lambda_max data found in test set.")

    # --- Process Phi ---
    if len(phi_true) > 0:
        phi_true = np.concatenate(phi_true)
        phi_pred = np.concatenate(phi_pred)
        
        p_rmse = rmse(phi_pred, phi_true)
        p_mae = mae(phi_pred, phi_true)
        
        metrics_str.append(f"Phi RMSE:        {p_rmse:.4f}")
        metrics_str.append(f"Phi MAE:         {p_mae:.4f}")
        
        parity_plot(phi_true, phi_pred, "Singlet Oxygen Yield", "True", "Predicted", 
                   os.path.join(out_dir, "phi_parity.png"))
        residual_hist(phi_pred - phi_true, "Phi Residuals", "Error", 
                     os.path.join(out_dir, "phi_residuals.png"))
    else:
        # It's normal to have no phi data if you didn't train on it or if dataset is sparse
        pass 

    # 6. Report
    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    for line in metrics_str:
        print(line)
    
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(metrics_str))

    print(f"\nPlots saved to: {out_dir}")

if __name__ == "__main__":
    main()