import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Setup Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from mana_model import MANA

# CONFIG
PRETRAINED_PATH = project_root / "models" / "best_model.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    print(f"--- Loading Pre-Trained Model & Adding Solvent Head ---")
    
    # 1. Initialize the NEW architecture (with solvent enabled)
    # Note: We assume num_atom_types is the same as your saved model (e.g. 21)
    model = MANA(
        num_atom_types=21, 
        hidden_dim=128,
        tasks=["lambda", "phi"],
        use_solvent=True  # <--- This triggers the new architecture
    ).to(DEVICE)

    # 2. Load the OLD weights
    # We use strict=False. 
    # This will load the 'layers' and 'lambda_head' perfectly.
    # It will FAIL to load 'phi_head' (because sizes don't match), which is exactly what we want!
    pretrained_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    model_dict = model.state_dict()

    # Filter out unnecessary keys (like the old phi_head)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    
    print("✓ Backbone and Lambda Head weights loaded.")
    print("✓ Phi Head and Solvent Encoder initialized from scratch.")

    # 3. FREEZE the Backbone and Lambda Head
    for name, param in model.named_parameters():
        if "solvent_encoder" in name or "phi_head" in name:
            param.requires_grad = True # Train these
            print(f"  -> Trainable: {name}")
        else:
            param.requires_grad = False # Freeze these
            
    # 4. Create Optimizer
    # Only pass the trainable parameters to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    
    print(f"\nModel is ready. Training {len(trainable_params)} tensors out of {len(list(model.parameters()))}.")
    
    # ... (Insert your training loop here using your solvent dataset) ...
    # Remember: Your dataset must now provide `data.solvent` (dielectric constant)

if __name__ == "__main__":
    main()