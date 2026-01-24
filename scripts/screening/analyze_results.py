import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
NIR_TARGET_NM = 800.0   # Ideal Wavelength (Start of NIR window)
IDEAL_PHI = 1.0         # Ideal Quantum Yield

def load_results(h5_path):
    """Reads the raw prediction data from the H5 file."""
    print(f"Loading results from {h5_path}...")
    
    with h5py.File(h5_path, "r") as f:
        # Load arrays
        mol_ids = f["mol_ids"][()]
        lambda_preds = f["lambda_max"][()] # stored as 'lambda_max' in your saver
        phi_preds = f["phi_delta"][()]     # stored as 'phi_delta' in your saver
        raw_smiles = f["smiles"][()]

    # Decode SMILES (H5 often stores strings as bytes)
    smiles = [s.decode("utf-8") if isinstance(s, bytes) else s for s in raw_smiles]

    df = pd.DataFrame({
        "mol_id": mol_ids,
        "smiles": smiles,
        "pred_lambda": lambda_preds,
        "pred_phi": phi_preds
    })
    
    print(f"Loaded {len(df)} molecules.")
    return df

def calculate_scores(df):
    """
    Calculates rankings based on absolute utility for PDT.
    """
    # --- 1. Calculate Distances (Keep this for the Display function) ---
    df["dist_lambda"] = np.abs(df["pred_lambda"] - NIR_TARGET_NM)

    # --- 2. Wavelength Score (Sigmoid-like preference) ---
    # We use a Gaussian curve centered at NIR_TARGET_NM
    sigma = 100.0 
    df["score_lambda"] = np.exp(-((df["pred_lambda"] - NIR_TARGET_NM)**2) / (2 * sigma**2))

    # Penalize < 600nm or > 900nm heavily (outside therapeutic window)
    df.loc[df["pred_lambda"] < 600, "score_lambda"] *= 0.1 
    df.loc[df["pred_lambda"] > 900, "score_lambda"] *= 0.1

    # --- 3. Yield Score ---
    # Clamp between 0 and 1
    df["score_phi"] = df["pred_phi"].clip(0, 1.0) 

    # --- 4. Combined Score (Product Method) ---
    df["combined_score"] = df["score_lambda"] * df["score_phi"]
    
    return df

def display_top_k(df, k=10):
    """Prints formatted tables for the top K molecules."""
    
    pd.set_option('display.max_colwidth', 40)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.3f}'.format)

    # --- 1. Top Lambda (Closest to Target) ---
    # Uses 'dist_lambda' which we just restored above
    print("\n" + "="*80)
    print(f"TOP {k}: WAVELENGTH (Closest to {NIR_TARGET_NM} nm)")
    print("="*80)
    top_lambda = df.sort_values("dist_lambda", ascending=True).head(k)
    print(top_lambda[["mol_id", "smiles", "pred_lambda", "pred_phi"]])

    # --- 2. Top Phi (Highest Yield) ---
    print("\n" + "="*80)
    print(f"TOP {k}: QUANTUM YIELD (Highest Phi)")
    print("="*80)
    top_phi = df.sort_values("pred_phi", ascending=False).head(k)
    print(top_phi[["mol_id", "smiles", "pred_lambda", "pred_phi"]])

    # --- 3. Top Combined (Weighted Utility) ---
    print("\n" + "="*80)
    print(f"TOP {k}: COMBINED (Weighted Utility)")
    print("Higher Score = Better Candidate (Max 1.0)")
    print("="*80)
    # Sort DESCENDING because higher score is better now
    top_combined = df.sort_values("combined_score", ascending=False).head(k)
    print(top_combined[["mol_id", "smiles", "pred_lambda", "pred_phi", "combined_score"]])

def main():
    parser = argparse.ArgumentParser(description="Analyze screening results from H5 file.")
    parser.add_argument("--input", required=True, help="Path to screening_results.h5")
    parser.add_argument("--top", type=int, default=10, help="Number of top molecules to display")
    parser.add_argument("--save_csv", default=None, help="Optional path to save the ranked CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        return

    # Run Pipeline
    df = load_results(input_path)
    df = calculate_scores(df)
    display_top_k(df, k=args.top)

    # Optional Save
    if args.save_csv:
        csv_path = Path(args.save_csv)
        print(f"\nSaving full ranked list to {csv_path}...")
        # Sort by combined score for the CSV
        df.sort_values("combined_score", ascending=False).to_csv(csv_path, index=False)
        print("Done.")

if __name__ == "__main__":
    main()