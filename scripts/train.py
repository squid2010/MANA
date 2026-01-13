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
    if conda_env.lower() != "mana" and conda_env != "":
        print(
            f"WARNING: Not running in 'mana' conda environment (current: {conda_env})"
        )
        return False
    print(f"✓ Running in environment: {conda_env if conda_env else 'Base/System'}")
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("MANA PHOTOSENSITIZER PROPERTY TRAINING")
    print("=" * 80)

    check_conda_environment()

    # ------------------------------------------------------------------
    # Paths (Dynamic & Relative)
    # ------------------------------------------------------------------
    # Looks for data in: ProjectRoot/data/deep4chem_data.h5
    dataset_path = project_root / "data" / "deep4chem_data.h5"
    
    # Saves models in: ProjectRoot/models
    save_dir = project_root / "models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Dataset: {dataset_path}")
    print(f"Save Dir: {save_dir}")

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run parse_deep4chem.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = DatasetConstructor(
        str(dataset_path),
        cutoff_radius=5.0,
        batch_size=64,
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
        "tasks": ["lambda"],
    }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MANA(
        num_atom_types=dataset.num_atom_types,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
        tasks=hyperparams["tasks"],
    )

    # Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS Fallback warning for scatter operations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    else:
        device = torch.device("cpu")
    
    #device = torch.device("cpu")
    
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
        save_dir=str(save_dir),
    )

    # ------------------------------------------------------------------
    # Configuration summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Learning rate: {hyperparams['learning_rate']}")
    print(f"Max epochs: {hyperparams['max_epochs']}")
    print(f"Weight decay: {hyperparams['weight_decay']}")    
    print(f"Active Training Tasks: {hyperparams['tasks']}")
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
        print("  - loss_curves.png")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error:\n{e}")
        # Print full traceback for easier debugging
        import traceback
        traceback.print_exc()