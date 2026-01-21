import os
import sys
from pathlib import Path
import numpy as np
import torch

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NUM_ATOM_TYPES = 118  # Universal constant

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
    # Using 'split_by_mol_id=True' is generally safer for chemistry to avoid 
    # data leakage between conformers of the same molecule.
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

    # Handle NaN means (Phase 3 dataset has no valid Lambda Max)
    l_mean = dataset.lambda_mean
    l_std = dataset.lambda_std
    if np.isnan(l_mean):
        l_mean = 500.0 
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
        # strict=False allows us to load the backbone even if we are switching heads
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')), strict=False
        )
    
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
    print(f"Weight Decay: {hyperparams['weight_decay']}")
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
    fluor_dataset_path = project_root / "data" / "flourescence_data.h5"
    phi_dataset_path = project_root / "data" / "phidelta_data.h5"

    save_dir_lambda = project_root / "models" / "lambda"
    save_dir_fluor = project_root / "models" / "fluor"
    save_dir_phi = project_root / "models" / "phi"
    
    os.makedirs(save_dir_lambda, exist_ok=True)
    os.makedirs(save_dir_fluor, exist_ok=True)
    os.makedirs(save_dir_phi, exist_ok=True)

    # =================================================================
    # PHASE 1: PRE-TRAINING (Absorption)
    # =================================================================
    lambda_hyperparams = {
        "learning_rate": 1e-3,     # Fast learning for large dataset
        "max_epochs": 200,         # 200 is usually enough for convergence
        "early_stopping_patience": 20, 
        "weight_decay": 1e-5,      # Standard regularization
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
    
    # =================================================================
    # PHASE 2: ADAPTATION (Fluorescence)
    # =================================================================
    fluor_hyperparams = {
        "learning_rate": 1e-4,     # 10x slower to preserve features
        "max_epochs": 200,
        "early_stopping_patience": 25,
        "weight_decay": 1e-5,      
        "tasks": ["phi"],          # Re-purposing the Phi head
    }
    
    p1_model_path = save_dir_lambda / "best_model.pth"
    if p1_model_path.exists():
        train_head(
            fluor_hyperparams, 
            fluor_dataset_path, 
            save_dir_fluor, 
            split_by_mol_id=True, 
            load_model=True, 
            model_path=p1_model_path,
            freeze_backbone=False  # Unfrozen: Allow backbone to learn Emission physics
        )
    else:
        print("Skipping Phase 2 (Phase 1 model not found)")

    # =================================================================
    # PHASE 3: SPECIALIZATION (Singlet Oxygen)
    # =================================================================
    phi_hyperparams = {
        "learning_rate": 5e-5,     # Very slow fine-tuning
        "max_epochs": 150,         # Shorter run for small data
        "early_stopping_patience": 20,
        "weight_decay": 1e-3,      # High regularization to prevent overfitting on small data
        "tasks": ["phi"],
    }
    
    p2_model_path = save_dir_fluor / "best_model.pth"
    if p2_model_path.exists():
        train_head(
            phi_hyperparams, 
            phi_dataset_path, 
            save_dir_phi, 
            split_by_mol_id=True, 
            load_model=True, 
            model_path=p2_model_path,
            freeze_backbone=True   # Frozen: Don't break the GNN on 1.4k samples
        )
    else:
        print("Skipping Phase 3 (Phase 2 model not found)")