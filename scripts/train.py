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
    print(f"Dataset: {dataset_path}")
    print(f"Tasks: {hyperparams['tasks']}")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    # 1. Dataset
    # Split by mol_id ensures rigorous validation
    dataset = DatasetConstructor(
        str(dataset_path),
        cutoff_radius=5.0,
        batch_size=64,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
        split_by_mol_id=True, 
    )

    train_loader, val_loader, _ = dataset.get_dataloaders(num_workers=0)

    # Handle Normalization Stats
    l_mean = dataset.lambda_mean
    l_std = dataset.lambda_std
    l_mean = 500.0 if np.isnan(l_mean) else l_mean
    l_std = 100.0 if np.isnan(l_std) else l_std

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
        # strict=False allows us to load weights even if the "tasks" changed 
        # (e.g. adding a new head in Phase 2)
        model.load_state_dict(torch.load(load_path, map_location='cpu'), strict=False)
    
    if freeze_backbone:
        model.freeze_backbone()
    
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
    fluor_data  = project_root / "data" / "fluor" / "fluorescence_data.h5"
    phi_data    = project_root / "data" / "phi" / "phidelta_data.h5"

    dir_p1 = project_root / "models" / "phase1_lambda"
    dir_p2 = project_root / "models" / "phase2_fluor"
    dir_p3 = project_root / "models" / "phase3_singlet"
    
    os.makedirs(dir_p1, exist_ok=True)
    os.makedirs(dir_p2, exist_ok=True)
    os.makedirs(dir_p3, exist_ok=True)

    # =================================================================
    # PHASE 1: GENERAL PRE-TRAINING (Lambda Only)
    # Objective: Learn molecular representation and solvent interaction.
    # =================================================================
    p1_params = {
        "learning_rate": 1e-3, 
        "max_epochs": 150,
        "early_stopping_patience": 20, 
        "weight_decay": 1e-5,
        "tasks": ["lambda"], # Only train lambda head
    }
    
    train_phase("Phase 1 (Absorption)", p1_params, lambda_data, dir_p1, load_path=None)
    
    # =================================================================
    # PHASE 2: ADAPTATION (Fluorescence)
    # Objective: Learn emission physics (Phi_F) while retaining absorption.
    # =================================================================
    p2_params = {
        "learning_rate": 2e-4, # Lower LR to refine
        "max_epochs": 200,
        "early_stopping_patience": 25,
        "weight_decay": 1e-5,      
        "tasks": ["lambda", "phi"], # Train BOTH heads
    }
    
    p1_model = dir_p1 / "best_model.pth"
    if p1_model.exists():
        train_phase("Phase 2 (Fluorescence)", p2_params, fluor_data, dir_p2, load_path=p1_model)
    else:
        print("Skipping Phase 2 (Phase 1 model missing)")

    # =================================================================
    # PHASE 3: SPECIALIZATION (Singlet Oxygen)
    # Objective: Fine-tune Phi head for Phi_Delta.
    # =================================================================
    p3_params = {
        "learning_rate": 5e-5, # Very low LR
        "max_epochs": 150,
        "early_stopping_patience": 20,
        "weight_decay": 1e-3, 
        "tasks": ["phi"], # Focus on Phi head
    }
    
    p2_model = dir_p2 / "best_model.pth"
    if p2_model.exists():
        train_phase("Phase 3 (Singlet Oxygen)", p3_params, phi_data, dir_p3, load_path=p2_model, freeze_backbone=True)
    else:
        print("Skipping Phase 3 (Phase 2 model missing)")