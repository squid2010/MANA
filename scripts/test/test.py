import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir.parent))

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
# Generic Lambda Test Function
# ---------------------------------------------------------------------
def _test_lambda_model(device, out_dir, model_path, output_subdir, title):
    """
    Generic function to test a lambda model.

    Args:
        device: torch device
        out_dir: base output directory
        model_path: path to the model weights
        output_subdir: subdirectory name for outputs (e.g., "lambda" or "lambda_pretrained")
        title: display title for logging
    """
    print("\n" + "=" * 80)
    print(f"TESTING {title}")
    print("=" * 80)

    dataset_path = project_root / "data" / "lambdamax_data.h5"

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return []

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return []

    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_path}")

    # Load Dataset (no mol_id splitting for lambda - same as training)
    dataset = DatasetConstructor(
        str(dataset_path),
        cutoff_radius=5.0,
        batch_size=64,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
        split_by_mol_id=False,
    )

    _, _, test_loader = dataset.get_dataloaders(num_workers=0)

    # Load Model
    model = MANA(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
        tasks=["lambda"],
        lambda_mean=dataset.lambda_mean,
        lambda_std=dataset.lambda_std,
    ).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✓ {title} weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model weights: {e}")
        return []

    model.eval()

    # Evaluation Loop
    lambda_true, lambda_pred = [], []

    n_samples = len(test_loader.dataset)  # pyright: ignore[reportArgumentType]
    print(f"Evaluating on {n_samples} test samples...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {title}", leave=False):
            batch = batch.to(device)
            preds = model(batch)

            if hasattr(batch, "lambda_max"):
                y_true = batch.lambda_max.squeeze().cpu().numpy()
                y_pred = preds["lambda"].cpu().numpy()

                mask = np.isfinite(y_true)
                if mask.any():
                    lambda_true.append(y_true[mask])
                    lambda_pred.append(y_pred[mask])

    # Process Metrics
    metrics_str = []

    if len(lambda_true) > 0:
        lambda_true = np.concatenate(lambda_true)
        lambda_pred = np.concatenate(lambda_pred)

        l_rmse = rmse(lambda_pred, lambda_true)
        l_mae = mae(lambda_pred, lambda_true)

        metrics_str.append(f"{title} RMSE: {l_rmse:.4f} nm")
        metrics_str.append(f"{title} MAE:  {l_mae:.4f} nm")

        lambda_out = out_dir / output_subdir
        os.makedirs(lambda_out, exist_ok=True)

        parity_plot(
            lambda_true,
            lambda_pred,
            f"{title} - Absorption Max (nm)",
            "True",
            "Predicted",
            lambda_out / "parity.png",
        )
        residual_hist(
            lambda_pred - lambda_true,
            f"{title} Residuals",
            "Error (nm)",
            lambda_out / "residuals.png",
        )

        print(f"✓ {title} plots saved to: {lambda_out}")
    else:
        print("WARNING: No valid Lambda_max data found in test set.")

    return metrics_str


# ---------------------------------------------------------------------
# Test Lambda Model (Fine-tuned)
# ---------------------------------------------------------------------
def test_lambda(device, out_dir):
    model_path = project_root / "models" / "phi" / "best_model.pth"
    return _test_lambda_model(device, out_dir, model_path, "lambda", "LAMBDA MODEL")


# ---------------------------------------------------------------------
# Test Lambda Model (Pretrained)
# ---------------------------------------------------------------------
def test_lambda_pretrained(device, out_dir):
    model_path = project_root / "models" / "lambda" / "best_model.pth"
    return _test_lambda_model(
        device, out_dir, model_path, "lambda_pretrained", "PRETRAINED LAMBDA MODEL"
    )


# ---------------------------------------------------------------------
# Test Phi Model
# ---------------------------------------------------------------------
def test_phi(device, out_dir):
    print("\n" + "=" * 80)
    print("TESTING PHI MODEL")
    print("=" * 80)

    dataset_path = project_root / "data" / "phidelta_data.h5"
    model_path = project_root / "models" / "phi" / "best_model.pth"

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return []

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return []

    print(f"Dataset: {dataset_path}")
    print(f"Model:   {model_path}")

    # Load Dataset (split_by_mol_id=True to match training and prevent data leakage)
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

    # Load Model
    model = MANA(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
        tasks=["phi"],
        lambda_mean=dataset.lambda_mean,
        lambda_std=dataset.lambda_std,
    ).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("✓ Phi model weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model weights: {e}")
        return []

    model.eval()

    # Evaluation Loop
    phi_true, phi_pred = [], []

    n_samples = len(test_loader.dataset)  # pyright: ignore[reportArgumentType]
    print(f"Evaluating on {n_samples} test samples...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating PHI MODEL", leave=False):
            batch = batch.to(device)
            preds = model(batch)

            if hasattr(batch, "phi_delta") and "phi" in preds:
                y_true = batch.phi_delta.squeeze().cpu().numpy()
                y_pred = preds["phi"].cpu().numpy()

                mask = np.isfinite(y_true)
                if mask.any():
                    phi_true.append(y_true[mask])
                    phi_pred.append(y_pred[mask])

    # Process Metrics
    metrics_str = []

    if len(phi_true) > 0:
        phi_true = np.concatenate(phi_true)
        phi_pred = np.concatenate(phi_pred)

        p_rmse = rmse(phi_pred, phi_true)
        p_mae = mae(phi_pred, phi_true)

        metrics_str.append(f"Phi RMSE: {p_rmse:.4f}")
        metrics_str.append(f"Phi MAE:  {p_mae:.4f}")

        phi_out = out_dir / "phi"
        os.makedirs(phi_out, exist_ok=True)

        parity_plot(
            phi_true,
            phi_pred,
            "Singlet Oxygen Yield",
            "True",
            "Predicted",
            phi_out / "parity.png",
        )
        residual_hist(
            phi_pred - phi_true, "Phi Residuals", "Error", phi_out / "residuals.png"
        )

        print(f"✓ Phi plots saved to: {phi_out}")
    else:
        print("WARNING: No valid Phi data found in test set.")

    return metrics_str


# ---------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="MANA Model Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --pretrain          # Test only pretrained lambda model
  python test.py --lambda            # Test only fine-tuned lambda model
  python test.py --phi               # Test only phi model
  python test.py --pretrain --phi    # Test pretrained lambda and phi
  python test.py                     # Test all models (default)
  python test.py --cpu               # Force CPU (recommended for macOS)
        """,
    )

    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Evaluate the pretrained lambda model (from lambda training phase)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_model",  # 'lambda' is a reserved keyword
        action="store_true",
        help="Evaluate the fine-tuned lambda model (from phi training phase)",
    )
    parser.add_argument(
        "--phi",
        action="store_true",
        help="Evaluate the phi (singlet oxygen yield) model",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU device (recommended for macOS to avoid MPS crashes)",
    )

    args = parser.parse_args()

    # If no flags specified, run all tests
    if not (args.pretrain or args.lambda_model or args.phi):
        args.pretrain = True
        args.lambda_model = True
        args.phi = True

    return args


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 80)
    print("MANA MODEL – TEST METRICS & DIAGNOSTICS")
    print("=" * 80)

    # Show which models will be tested
    models_to_test = []
    if args.pretrain:
        models_to_test.append("Pretrained Lambda")
    if args.lambda_model:
        models_to_test.append("Fine-tuned Lambda")
    if args.phi:
        models_to_test.append("Phi")
    print(f"Models to evaluate: {', '.join(models_to_test)}")

    # Setup output directory
    out_dir = project_root / "results"
    os.makedirs(out_dir, exist_ok=True)

    # Detect device
    if args.cpu:
        device = torch.device("cpu")
        print("Device: cpu (forced via --cpu flag)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device}")
    elif torch.backends.mps.is_available():
        # MPS can be unstable - warn user
        print(
            "WARNING: MPS detected but can be unstable. Use --cpu if you encounter crashes."
        )
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print(f"Device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Device: {device}")
    print(f"Project Root: {project_root}")

    # Run tests based on flags
    all_metrics = []

    if args.pretrain:
        pretrained_metrics = test_lambda_pretrained(device, out_dir)
        all_metrics.extend(pretrained_metrics)

    if args.lambda_model:
        lambda_metrics = test_lambda(device, out_dir)
        all_metrics.extend(lambda_metrics)

    if args.phi:
        phi_metrics = test_phi(device, out_dir)
        all_metrics.extend(phi_metrics)

    # Final Report
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    if len(all_metrics) > 0:
        for line in all_metrics:
            print(line)

        with open(out_dir / "metrics.txt", "w") as f:
            f.write("\n".join(all_metrics))

        print(f"\nAll results saved to: {out_dir}")
    else:
        print("No metrics computed. Check that models and datasets exist.")


if __name__ == "__main__":
    main()
