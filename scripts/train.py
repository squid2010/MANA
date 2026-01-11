#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import torch

# ------------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------------
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from model.mana_model import MANA  # noqa: E402
from model.training_engine import TrainingEngine  # noqa: E402
from data.dataset import DatasetConstructor  # noqa: E402


def check_conda_environment():
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env.lower() != "mana":
        print(
            f"WARNING: Not running in 'mana' conda environment (current: {conda_env})"
        )
        return False
    print(f"✓ Running in correct conda environment: {conda_env}")
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("MANA PHOTOSENSITIZER PROPERTY TRAINING (λmax, φΔ)")
    print("=" * 80)

    check_conda_environment()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    dataset_path = (
        "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/data/"
        "photosensitizer_dataset.h5"
    )
    save_dir = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/models"
    os.makedirs(save_dir, exist_ok=True)

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

    train_loader, val_loader, test_loader = dataset.get_dataloaders(num_workers=0)

    print(f"Atom types: {dataset.num_atom_types}")
    print(f"Training samples: {len(train_loader.dataset)}")  # pyright: ignore
    print(f"Validation samples: {len(val_loader.dataset)}")  # pyright: ignore

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    hyperparams = {
        "learning_rate": 5e-4,
        "max_epochs": 500,
        "early_stopping_patience": 80,
        "weight_decay": 1e-5,
    }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MANA(
        num_atom_types=dataset.num_atom_types,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ------------------------------------------------------------------
    # Training Engine
    # ------------------------------------------------------------------
    engine = TrainingEngine(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        hyperparams=hyperparams,
        save_dir=save_dir,
    )

    # ------------------------------------------------------------------
    # Configuration summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Learning rate: {hyperparams['learning_rate']}")
    print(f"Max epochs: {hyperparams['max_epochs']}")
    print(f"Weight decay: {hyperparams['weight_decay']}")
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    try:
        engine.train()

        print("\nTraining completed successfully.")
        print("Saved artifacts:")
        print("  - best_model.pth")
        print("  - loss_history.npz")
        print("  - loss.png")

    except Exception as e:
        print(f"\nTraining failed with error:\n{e}")
