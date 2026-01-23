import argparse
import sys
import os
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent.parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))

from model.mana_model import MANA
from data.dataset import DatasetConstructor

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NIR_TARGET_NM = 800.0  # Ideal target for deep tissue penetration (Start of NIR window)
IDEAL_PHI = 1.0        # Max theoretical singlet oxygen yield

def calculate_rankings(df):
    """
    Adds ranking scores to the DataFrame based on user criteria.
    """
    # 1. Lambda Score (Closeness to NIR Target)
    # Lower is better (closest distance)
    df['dist_lambda'] = np.abs(df['pred_lambda'] - NIR_TARGET_NM)
    
    # 2. Phi Score (Closeness to Ideal Yield)
    # Lower is better (distance from 1.0)
    df['dist_phi'] = np.abs(df['pred_phi'] - IDEAL_PHI)

    # 3. Combined Score (Weighted Equally by Divergence)
    # Problem: Lambda errors are in 100s (nm), Phi errors are 0.0-1.0.
    # Solution: Z-Score Normalization (Standardization) to make units comparable.
    
    lambda_z = (df['dist_lambda'] - df['dist_lambda'].mean()) / df['dist_lambda'].std()
    phi_z = (df['dist_phi'] - df['dist_phi'].mean()) / df['dist_phi'].std()
    
    # Combined Divergence (Lower is better)
    df['combined_score'] = (lambda_z + phi_z) / 2.0
    
    return df

def run_screening(input_path, model_path, output_path, device, batch_size=64):
    print(f"--- Loading Large Dataset: {input_path.name} ---")
    
    # split_by_mol_id=False ensures we load everything linearly without shuffling for train/test
    # We pass the input_path directly to the constructor
    dataset = DatasetConstructor(
        str(input_path), 
        split_by_mol_id=True,  # Keeps molecules intact
        batch_size=batch_size
    )
    
    # Use the test_loader concept to just iterate over everything, 
    # but since we want ALL data, we might need to access the internal list or join loaders.
    # For a screening script, it's safer to create a single loader for the whole dataset.
    # The DatasetConstructor creates splits automatically. 
    # To screen *everything*, we will combine indices manually here.
    
    all_indices = list(range(len(dataset)))
    # Create a custom loader for the full dataset
    from data.dataset import GeometricSubset
    from torch_geometric.loader import DataLoader
    
    full_loader = DataLoader(
        GeometricSubset(dataset, all_indices), 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0 # Set to 2 or 4 if on Linux/Windows, 0 often safer on Mac
    )

    # --- Load Model ---
    model = MANA(
        num_atom_types=118, 
        hidden_dim=128, 
        tasks=["lambda", "phi"],
    ).to(device)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✓ Model loaded.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- Inference Loop ---
    results = []
    print(f"--- Screening {len(dataset)} molecules ---")
    
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Inference"):
            batch = batch.to(device)
            preds = model(batch)
            
            # Move to CPU numpy
            l_preds = preds["lambda"].cpu().numpy()
            p_preds = preds["phi"].cpu().numpy()
            
            # Handle potential single-item batch dimensions
            if l_preds.ndim == 0: l_preds = [l_preds]
            if p_preds.ndim == 0: p_preds = [p_preds]
            
            # Extract Metadata
            # Note: batch.smiles is a list of strings
            smiles = batch.smiles
            mol_ids = batch.mol_id.cpu().numpy()

            for i in range(len(mol_ids)):
                results.append({
                    "mol_id": mol_ids[i],
                    "smiles": smiles[i],
                    "pred_lambda": float(l_preds[i]),
                    "pred_phi": float(p_preds[i])
                })

    # --- Analysis ---
    df = pd.DataFrame(results)
    df = calculate_rankings(df)
    
    # Save Full Results
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Full results saved to: {output_path}")

    # --- Display Top Candidates ---
    pd.set_option('display.max_colwidth', 40)
    
    print("\n" + "="*80)
    print(f"TOP 5: WAVELENGTH (Closest to {NIR_TARGET_NM} nm)")
    print("="*80)
    top_lambda = df.sort_values("dist_lambda", ascending=True).head(5)
    print(top_lambda[['mol_id', 'smiles', 'pred_lambda', 'pred_phi']])

    print("\n" + "="*80)
    print("TOP 5: QUANTUM YIELD (Highest Phi)")
    print("="*80)
    top_phi = df.sort_values("pred_phi", ascending=False).head(5)
    print(top_phi[['mol_id', 'smiles', 'pred_lambda', 'pred_phi']])

    print("\n" + "="*80)
    print("TOP 5: BALANCED (Best Combination)")
    print("="*80)
    # combined_score: lower is better (less divergence)
    top_combined = df.sort_values("combined_score", ascending=True).head(5)
    print(top_combined[['mol_id', 'smiles', 'pred_lambda', 'pred_phi', 'combined_score']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .h5 file")
    parser.add_argument("--output", type=str, default="results/screening/large_screen_results.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_path = Path(args.input)
    model_path = project_root / "models" / "fluor" / "best_model.pth" # Default to best model
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    run_screening(input_path, model_path, output_path, device, args.batch_size)

if __name__ == "__main__":
    main()