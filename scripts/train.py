import os
import sys
from pathlib import Path
import numpy as np
import torch

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NUM_ATOM_TYPES = 118 

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from model.mana_model import MANA 
from model.training_engine import TrainingEngine 
from data.dataset import DatasetConstructor 

def train_phase(phase_name, hyperparams, dataset_path, save_dir, load_path=None, freeze_backbone=False):
    print("\n" + "=" * 80)
    print(f"STARTING PHASE: {phase_name.upper()}")
    print("=" * 80)
    print(f"Primary Dataset: {dataset_path}")
    print(f"Tasks: {hyperparams['tasks']}")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    # 1. Load Primary Dataset
    dataset = DatasetConstructor(
        str(dataset_path),
        cutoff_radius=5.0,
        batch_size=64,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
        split_by_mol_id=True, 
    )
    
    train_set = dataset.get_dataloaders(num_workers=0)[0].dataset
    val_set = dataset.get_dataloaders(num_workers=0)[1].dataset

    # Re-wrap in DataLoaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    # Handle Normalization Stats
    l_mean = dataset.lambda_mean if not np.isnan(dataset.lambda_mean) else 500.0
    l_std = dataset.lambda_std if not np.isnan(dataset.lambda_std) else 100.0

    # 2. Model
    model = MANA(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=128,
        num_layers=4,
        num_rbf=20,
        tasks=hyperparams["tasks"],
        lambda_mean=l_mean,
        lambda_std=l_std,
    )
    
    # 3. Load Weights
    if load_path:
        print(f"Loading weights from: {load_path}")
        # strict=False allows loading Phase 1 (Lambda only) into Phase 2 (Lambda + Phi)
        model.load_state_dict(torch.load(load_path, map_location='cpu'), strict=False)
    
    if freeze_backbone:
        model.freeze_backbone(hyperparams["tasks"])
    
    # 4. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    model = model.to(device)

    # 5. Train
    engine = TrainingEngine(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        hyperparams=hyperparams,
        save_dir=str(save_dir),
    )

    engine.train()
    print(f"\n{phase_name} complete. Best model saved to {save_dir}/best_model.pth")


if __name__ == "__main__":
    print("=" * 100)
    print("MANA PHOTOSENSITIZER TRAINING PIPELINE")
    print("=" * 100)

    # Paths
    lambda_data = project_root / "data" / "lambda" / "lambda_only_data.h5"
    phi_data    = project_root / "data" / "phi" / "phidelta_data.h5"

    dir_p1 = project_root / "models" / "phase1_lambda"
    dir_p2 = project_root / "models" / "phase2_singlet"
    
    os.makedirs(dir_p1, exist_ok=True)
    os.makedirs(dir_p2, exist_ok=True)

    # =================================================================
    # PHASE 1: Lambda Head
    # Objective: Build absorption representations
    # =================================================================
    p1_params = {
        "learning_rate": 1e-3,
        "max_epochs": 200,
        "early_stopping_patience": 40,
        "weight_decay": 1e-5,
        "tasks": ["lambda"], 
    }
    
    train_phase("Phase 1 (Absorption)", p1_params, lambda_data, dir_p1, load_path=None)
    
    # =================================================================
    # PHASE 2: Singlet Oxygen
    # Objective: Train Phi head + Gentle fine-tuning of backbone
    # =================================================================
    p2_params = {
        "learning_rate": 2e-4, # Lower LR to refine
        "max_epochs": 200,
        "early_stopping_patience": 50,
        "weight_decay": 1e-4,
        "tasks": ["phi"], # We focus only on phi loss here
    }
    
    p1_model = dir_p1 / "best_model.pth"
    
    if p1_model.exists():
        train_phase(
            "Phase 2 (Singlet Oxygen)", 
            p2_params, 
            phi_data, 
            dir_p2, 
            load_path=p1_model, 
            freeze_backbone=False # MUST be False for differential LR to work
        )
    else:
        print("Skipping Phase 2 (Phase 1 model missing)")