import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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
    plt.figure(figsize=(5, 5))
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def residual_hist(residuals, title, xlabel, path):
    plt.figure(figsize=(5, 4))
    plt.hist(residuals, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print("MANA MODEL – TEST METRICS & DIAGNOSTICS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    dataset_path = (
        "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/data/"
        "photosensitizer_dataset.h5"
    )
    model_dir = (
        "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/models"
    )
    model_path = os.path.join(model_dir, "best_model.pth")

    out_dir = os.path.join(model_dir, "test_results")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = DatasetConstructor(
        dataset_path,
        cutoff_radius=5.0,
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
    )

    _, _, test_loader = dataset.get_dataloaders(num_workers=0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if dataset.energies_excited.ndim > 1:
        num_excited = dataset.energies_excited.shape[1]
    else:
        num_excited = 1
    num_states = 1 + num_excited

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MANA(
        num_atom_types=dataset.num_atom_types,
        num_states=num_states,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loaded model: {model_path}")
    print(f"Device: {device}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    E_true, E_pred = [], []
    nac_true, nac_pred = [], []
    lambda_true, lambda_pred = [], []
    phi_true, phi_pred = [], []

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            energies, _, nac, lambda_max, phi = model(batch)

            # Energies
            E_pred.append(energies.cpu().numpy())
            E_true.append(batch.energies.view(energies.shape).cpu().numpy())

            # NAC magnitudes
            nac_pred.append(torch.norm(nac, dim=-1).cpu().numpy())
            nac_true.append(torch.norm(batch.nac, dim=(-1, -2)).cpu().numpy())

            # Spectroscopy
            lambda_pred.append(lambda_max.squeeze(-1).cpu().numpy())
            lambda_true.append(batch.lambda_max.cpu().numpy())

            phi_pred.append(phi.squeeze(-1).cpu().numpy())
            phi_true.append(batch.phi_delta.cpu().numpy())

    # ------------------------------------------------------------------
    # Stack
    # ------------------------------------------------------------------
    E_true = np.vstack(E_true)
    E_pred = np.vstack(E_pred)

    nac_true = np.concatenate([x.reshape(-1) for x in nac_true])
    nac_pred = np.concatenate([x.reshape(-1) for x in nac_pred])


    lambda_true = np.concatenate(lambda_true)
    lambda_pred = np.concatenate(lambda_pred)

    phi_true = np.concatenate(phi_true)
    phi_pred = np.concatenate(phi_pred)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = {
        "Energy RMSE (Ha)": rmse(E_pred, E_true),
        "Energy MAE (Ha)": mae(E_pred, E_true),
        "NAC RMSE": rmse(nac_pred, nac_true),
        "NAC MAE": mae(nac_pred, nac_true),
        "Lambda_max RMSE": rmse(lambda_pred, lambda_true),
        "Lambda_max MAE": mae(lambda_pred, lambda_true),
        "Phi RMSE": rmse(phi_pred, phi_true),
        "Phi MAE": mae(phi_pred, phi_true),
    }

    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k:22s}: {v:.6f}")

    # Save metrics
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    parity_plot(
        E_true[:, 0],
        E_pred[:, 0],
        "Ground-State Energy",
        "True E₀ (Ha)",
        "Predicted E₀ (Ha)",
        os.path.join(out_dir, "energy_ground_parity.png"),
    )

    parity_plot(
        lambda_true,
        lambda_pred,
        "Absorption Maximum",
        "True λₘₐₓ",
        "Predicted λₘₐₓ",
        os.path.join(out_dir, "lambda_max_parity.png"),
    )

    parity_plot(
        phi_true,
        phi_pred,
        "Singlet Oxygen Yield",
        "True φ",
        "Predicted φ",
        os.path.join(out_dir, "phi_parity.png"),
    )

    residual_hist(
        lambda_pred - lambda_true,
        "λₘₐₓ Residuals",
        "Pred − True",
        os.path.join(out_dir, "lambda_residuals.png"),
    )

    residual_hist(
        phi_pred - phi_true,
        "φ Residuals",
        "Pred − True",
        os.path.join(out_dir, "phi_residuals.png"),
    )

    print(f"\nSaved plots and metrics to: {out_dir}")
    print("=" * 80)
    print("Testing completed successfully.")


if __name__ == "__main__":
    main()
