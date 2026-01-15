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


if __name__ == "__main__":
    print("=" * 100)
    print("MANA PHOTOSENSITIZER PROPERTY TRAINING")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Paths (Dynamic & Relative)
    # ------------------------------------------------------------------
    # Looks for data in: ProjectRoot/data/deep4chem_data.h5
    lambda_dataset_path = project_root / "data" / "lambdamax_data.h5"
    phi_dataset_path = project_root / "data" / "phi_data.h5"
    
    # Saves models in: ProjectRoot/models
    save_dir_lambda = project_root / "models" / "lambda"
    save_dir_phi = project_root / "models" / "phi"
    os.makedirs(save_dir_lambda, exist_ok=True)
    os.makedirs(save_dir_phi, exist_ok=True)
    
    # --------------------------------------------------------------------------------------------------
    # TRAINING LAMBDA HEAD
    # --------------------------------------------------------------------------------------------------
    
    print("=" * 80)
    print("TRAINING LAMBDA HEAD")
    print("=" * 80)

    print(f"Dataset: {lambda_dataset_path}")
    print(f"Save Dir: {save_dir_lambda}")

    if not lambda_dataset_path.exists():
        print(f"ERROR: Dataset not found at {lambda_dataset_path}")
        print("Please run build_lambda_dataset.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = DatasetConstructor(
        str(lambda_dataset_path),
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
        "max_epochs": 300,
        "early_stopping_patience": 60,
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
        lambda_mean=dataset.lambda_mean,
        lambda_std=dataset.lambda_std,
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
        save_dir=str(save_dir_lambda),
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
        
    # --------------------------------------------------------------------------------------------------
    # TRAINING PHI HEAD
    # --------------------------------------------------------------------------------------------------

    print("=" * 80)
    print("TRAINING PHI HEAD")
    print("=" * 80)

    print(f"Dataset: {phi_dataset_path}")
    print(f"Save Dir: {save_dir_phi}")

    if not phi_dataset_path.exists():
        print(f"ERROR: Dataset not found at {phi_dataset_path}")
        print("Please run build_phi_dataset.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = DatasetConstructor(
        str(phi_dataset_path),
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
        "max_epochs": 250,
        "early_stopping_patience": 40,
        "weight_decay": 1e-5,
        "tasks": ["phi"],
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
        lambda_mean=dataset.lambda_mean,
        lambda_std=dataset.lambda_std,
    )
    
    model.load_state_dict(torch.load(save_dir_lambda / "best_model.pth"), strict=False)
    model = model.to(device)

    # Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS Fallback warning for scatter operations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    else:
        device = torch.device("cpu")
    
    #device = torch.device("cpu"))

    # ------------------------------------------------------------------
    # Training Engine
    # ------------------------------------------------------------------
    engine = TrainingEngine(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        hyperparams=hyperparams,
        save_dir=str(save_dir_phi),
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