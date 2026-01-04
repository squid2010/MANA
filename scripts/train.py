#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add the scripts directory to the Python path for direct execution
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from data.dataset import DatasetConstructor
from model.mana_model import MANA
from model.training_engine import TrainingEngine


def check_conda_environment():
    """Check if running in the correct conda environment"""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env.lower() != "mana":
        print(
            f"WARNING: Not running in 'mana' conda environment (current: {conda_env})"
        )
        print("Please run: conda activate mana")
        return False
    else:
        print(f"✓ Running in correct conda environment: {conda_env}")
        return True


if __name__ == "__main__":
    print("=" * 80)
    print("MANA MODEL TRAINING")
    print("=" * 80)

    # Check conda environment (optional)
    check_conda_environment()

    file_path = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/dataset_construction/qm_results.h5"
    directory = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/models"

    # Create dataset with DataLoader functionality
    dataset = DatasetConstructor(
        file_path,
        cutoff_radius=5,
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
    )

    # Get DataLoaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(num_workers=0)

    # Check atomic numbers in the dataset for model initialization
    print(
        f"Dataset contains {dataset.num_atom_types} unique atom types: {dataset.unique_atoms}"
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Optimized hyperparameters with adaptive NAC weight system
    hyperparams = {
        "learning_rate": 5e-4,
        "batch_size": 32,
        "max_epochs": 500,
        "early_stopping_patience": 100,  # More patience for adaptive system
        "gradient_clip_norm": 0.5,  # Tighter clipping for stability
        "weight_decay": 1e-5,
        "loss_weights": {
            "energy": 5.0,  # Base energy weight
            "force": 1.0,  # l
            "nac": 1.0,  # Starting NAC weight (will be adaptive)
            "dipole": 0.0,  # Low dipole weight
        },
        # Adaptive NAC weight system configuration
        "adaptive_nac": {
            "enabled": True,  # Enable adaptive NAC weight adjustment
            "min_weight": 0.1,  # Minimum NAC weight
            "reduction_factor": 0.8,  # Multiply by this when reducing weight
            "patience": 5,  # Epochs to wait before reducing weight
            "spike_threshold": 2.0,  # Reduce if val_nac > train_nac * threshold
            "restore_threshold": 0.9,  # Restore if val_nac < train_nac * threshold
        },
    }

    # Create model and training engine
    model = MANA(dataset.num_atom_types, 3)
    engine = TrainingEngine(
        model, "cpu", train_loader, val_loader, hyperparams, directory
    )

    # Print configuration
    print("=" * 60)
    print(
        f"Model Architecture: MANA with {dataset.num_atom_types} atom types, 3 singlet states"
    )
    print(f"Learning Rate: {hyperparams['learning_rate']}")
    print(f"Batch Size: {hyperparams['batch_size']}")
    print(f"Max Epochs: {hyperparams['max_epochs']}")
    print(f"Early Stopping Patience: {hyperparams['early_stopping_patience']}")
    print(f"Gradient Clipping: {hyperparams['gradient_clip_norm']}")
    print(f"Weight Decay: {hyperparams['weight_decay']}")
    print()
    print("Loss Weights:")
    print(f"  - Energy: {hyperparams['loss_weights']['energy']}")
    print(f"  - Force:  {hyperparams['loss_weights']['force']}")
    print(f"  - NAC:    {hyperparams['loss_weights']['nac']}")
    print(f"  - Dipole: {hyperparams['loss_weights']['dipole']}")
    print()
    print(f"Model will be saved to: {directory}")
    print("=" * 60)

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    try:
        engine.train()

        print("\n" + "=" * 80)
        print("=" * 80)
        print("Check the generated files for:")
        print("  Loss plots (PNG) - NAC spikes should be controlled")
        print("  Training summary (TXT) - includes adaptive weight history")
        print("  Best model weights (PTH)")
        print("  TensorBoard logs")
        print("  NAC weight adaptation log")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("The adaptive system may need further tuning.")

        # Save partial results if possible
        if hasattr(engine, "train_losses") and len(engine.train_losses) > 0:
            print("\nSaving partial training results...")
            try:
                engine.plot_losses()
                engine.save_training_summary(
                    len(engine.train_losses),
                    min(engine.val_losses) if engine.val_losses else float("inf"),
                    f"training_failed: {str(e)[:100]}",
                )
                print("Partial results saved.")
            except:
                print("Could not save partial results.")
