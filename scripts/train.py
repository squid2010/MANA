import os
import sys
from pathlib import Path
import numpy as np
import torch

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NUM_ATOM_TYPES = 118  # Universal constant - covers H(1) through I(53)

# ------------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------------
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from model.mana_model import MANA  # noqa: E402
from model.training_engine import TrainingEngine  # noqa: E402
from data.dataset import DatasetConstructor  # noqa: E402

def train_head(hyperparams, dataset_path, save_dir, split_by_mol_id, load_model, model_path, freeze_backbone=False):
    head = hyperparams["tasks"][0]
    print("\n" + "=" * 80)
    print(f"TRAINING {head.upper()} HEAD")
    print("=" * 80)

    print(f"Dataset: {dataset_path}")
    print(f"Save Dir: {save_dir}")
    print(f"Freeze Backbone: {freeze_backbone}")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"Please run the corresponding build script first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    # FIX 1: Use the passed 'dataset_path' argument, not the global variable
    dataset = DatasetConstructor(
        str(dataset_path),
        cutoff_radius=5.0,
        batch_size=64,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
        split_by_mol_id=split_by_mol_id,
    )

    train_loader, val_loader, _ = dataset.get_dataloaders(num_workers=0)

    print(f"Atom types: {dataset.num_atom_types}")
    print(f"Training samples: {len(train_loader.dataset)}")  # pyright: ignore
    print(f"Validation samples: {len(val_loader.dataset)}")  # pyright: ignore

    # Handle NaN means (Phase 3 dataset has no valid Lambda Max, so mean is NaN)
    l_mean = dataset.lambda_mean
    l_std = dataset.lambda_std
    if np.isnan(l_mean):
        l_mean = 500.0 # Default fallback
        l_std = 100.0

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MANA(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
        tasks=hyperparams["tasks"],
        lambda_mean=l_mean,
        lambda_std=l_std,
    )
    
    if load_model:
        print(f"Loading weights from: {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')), strict=False
        )
    
    # FIX 3: Control freezing via argument
    if freeze_backbone:
        print("Freezing Backbone Layers...")
        model.freeze_backbone()
    
    # Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")

    model = model.to(device)

    # ------------------------------------------------------------------
    # Training Engine
    # ------------------------------------------------------------------
    # FIX 2: Use the passed 'save_dir' argument, not the global 'save_dir_lambda'
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
    print("-" * 60)
    print(f"Device: {device}")
    print(f"Learning rate: {hyperparams['learning_rate']}")
    print(f"Max epochs: {hyperparams['max_epochs']}")
    print(f"Active Tasks: {hyperparams['tasks']}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    try:
        engine.train()
        print(f"\nPhase complete. Best model saved to {save_dir}/best_model.pth")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error:\n{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 100)
    print("MANA PHOTOSENSITIZER PROPERTY TRAINING")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    lambda_dataset_path = project_root / "data" / "lambdamax_data.h5"
    fluor_dataset_path = project_root / "data" / "flourescence_data.h5" # Check spelling matches your build script
    phi_dataset_path = project_root / "data" / "phidelta_data.h5"

    save_dir_lambda = project_root / "models" / "lambda"
    save_dir_fluor = project_root / "models" / "fluor"
    save_dir_phi = project_root / "models" / "phi"
    
    os.makedirs(save_dir_lambda, exist_ok=True)
    os.makedirs(save_dir_fluor, exist_ok=True)
    os.makedirs(save_dir_phi, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Lambda (Deep4Chem Absorption)
    # ------------------------------------------------------------------
    lambda_hyperparams = {
        "learning_rate": 5e-4,
        "max_epochs": 300,
        "early_stopping_patience": 60,
        "weight_decay": 1e-5,
        "tasks": ["lambda"],
    }
    
    train_head(
        lambda_hyperparams, 
        lambda_dataset_path, 
        save_dir_lambda, 
        split_by_mol_id=False, 
        load_model=False, 
        model_path=None,
        freeze_backbone=False
    )
    
    # ------------------------------------------------------------------
    # Phase 2: Fluorescence (Deep4Chem Emission)
    # ------------------------------------------------------------------
    # Note: We set freeze_backbone=False to allow adaptation
    fluor_hyperparams = {
        "learning_rate": 1e-4, 
        "max_epochs": 250,
        "early_stopping_patience": 30,
        "weight_decay": 1e-4,
        "tasks": ["phi"],
    }
    
    if (save_dir_lambda / "best_model.pth").exists():
        train_head(
            fluor_hyperparams, 
            fluor_dataset_path, 
            save_dir_fluor, 
            split_by_mol_id=True, 
            load_model=True, 
            model_path=save_dir_lambda / "best_model.pth",
            freeze_backbone=False # Unfrozen to learn emission physics
        )
    else:
        print("Skipping Phase 2 (Phase 1 model not found)")

    # ------------------------------------------------------------------
    # Phase 3: Singlet Oxygen (Wilkinson)
    # ------------------------------------------------------------------
    # Note: We set freeze_backbone=True because dataset is small
    phi_hyperparams = {
        "learning_rate": 1e-4,
        "max_epochs": 250,
        "early_stopping_patience": 30,
        "weight_decay": 1e-4,
        "tasks": ["phi"],
    }
    
    if (save_dir_fluor / "best_model.pth").exists():
        train_head(
            phi_hyperparams, 
            phi_dataset_path, 
            save_dir_phi, 
            split_by_mol_id=True, 
            load_model=True, 
            model_path=save_dir_fluor / "best_model.pth",
            freeze_backbone=True # Frozen to prevent overfitting
        )
    else:
        print("Skipping Phase 3 (Phase 2 model not found)")