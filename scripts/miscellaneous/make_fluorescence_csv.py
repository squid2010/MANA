import pandas as pd
import os

# --- Configuration ---
INPUT_FILE = "data/lambdamax_dataset.csv"
OUTPUT_FILE = "data/fluorescence_dataset.csv"

def clean_csv():
    print(f"Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Read the CSV
    df = pd.read_csv(INPUT_FILE)

    # 1. Normalize Column Names
    # We rename 'Quantum yield' (if it exists) to 'Quantum Yield' to match your spec
    df = df.rename(columns={"Quantum yield": "Quantum Yield"})

    # 2. Select Columns
    # We want: Chromophore, Solvent, Quantum Yield, dielectric
    target_columns = ["Chromophore", "Solvent", "Quantum Yield", "dielectric"]
    
    # Check if columns exist
    missing_cols = [c for c in target_columns if c not in df.columns]
    
    # If 'dielectric' is missing from source, we can create it (filled with NaN)
    if "dielectric" in missing_cols:
        print("Note: 'dielectric' column missing in source. Creating it (empty).")
        df["dielectric"] = float("nan")
        missing_cols.remove("dielectric")

    # If other critical columns are missing, stop
    if missing_cols:
        print(f"Error: Source file missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Filter to only the columns we want
    df = df[target_columns]

    # 3. Filter Data (Remove N/A)
    # "only puts in rows with values of not n/a for those headers (except for dielectric)"
    initial_count = len(df)
    
    # We drop rows where Chromophore, Solvent, or Quantum Yield are NaN.
    # We do NOT include 'dielectric' in this subset check.
    df = df.dropna(subset=["Chromophore", "Solvent", "Quantum Yield"])

    removed_count = initial_count - len(df)
    print(f"Rows processed: {initial_count}")
    print(f"Rows removed (missing structure/solvent/yield): {removed_count}")
    print(f"Rows remaining: {len(df)}")

    # 4. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Cleaned data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_csv()