#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import torch

# Add project root to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

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
    print("MANA NON-ADIABATIC PHOTODYNAMICS TRAINING")
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
    print(f"Training samples: {len(train_loader.dataset)}")  # pyright: ignore[reportArgumentType]
    print(f"Validation samples: {len(val_loader.dataset)}")  # pyright: ignore[reportArgumentType]

    # ------------------------------------------------------------------
    # Hyperparameters (aligned with MANA internal loss)
    # ------------------------------------------------------------------
    hyperparams = {
        "learning_rate": 5e-4,
        "max_epochs": 500,
        "early_stopping_patience": 80,
        "gradient_clip_norm": 1.0,
        "weight_decay": 1e-5,
        "loss_weights": {
            "lambda": 1.0,
            "phi": 2.0,
            "nac_reg": 0.1,
        },
    }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # Determine number of states from dataset
    if dataset.energies_excited.ndim > 1:
        num_excited = dataset.energies_excited.shape[1]
    else:
        num_excited = 1
    num_states = 1 + num_excited

    print(
        f"Model configuration: {num_states} states (1 ground + {num_excited} excited)"
    )

    model = MANA(
        num_atom_types=dataset.num_atom_types,
        num_states=num_states,
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
    print("Loss weights:")
    for k, v in hyperparams["loss_weights"].items():
        print(f"  {k}: {v}")
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
        print("  - loss curves")
        print("  - training summary")

    except Exception as e:
        print(f"\nTraining failed with error:\n{e}")
