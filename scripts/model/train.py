import os
import sys
from pathlib import Path

# Add the scripts directory to the Python path for direct execution
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from data.dataset import DatasetConstructor
from model.mana_model import MANA
from model.training_engine import TrainingEngine

if __name__ == "__main__":
    file_path = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/dataset_construction/qm_results.h5"
    directory = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/models"

    # Create dataset with DataLoader functionality
    dataset = DatasetConstructor(
        file_path,
        cutoff_radius=5,
        batch_size=16,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
    )

    # Get DataLoaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(num_workers=2)

    # Check atomic numbers in the dataset for model initialization
    print(
        f"Dataset contains {dataset.num_atom_types} unique atom types: {dataset.unique_atoms}"
    )

    hyperparams = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "max_epochs": 200,
        "early_stopping_patience": 30,
        "gradient_clip_norm": 1.0,
        "loss_weights": {"energy": 1.0, "force": 100.0, "nac": 50.0, "dipole": 25.0},
    }

    model = MANA(dataset.num_atom_types, 3)
    engine = TrainingEngine(
        model, "cpu", train_loader, val_loader, hyperparams, directory
    )
    engine.train()
