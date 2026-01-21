import pandas as pd

# --- MANUAL PATCHES FOR "FRAGMENT" ERRORS ---
# These correct the specific issues where the script grabbed a substituent instead of the core.
PATCH_FIXES = {
    # Benzophenones
    "Benzophenone, 4,4'-bis(dimethylamino)-": "CN(C)C1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)N(C)C", # Michler's Ketone
    "Benzophenone, 4-(trifluoromethyl)-": "C1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)C(F)(F)F",
    "Benzophenone, 4,4'-dimethoxy-": "COC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)OC",
    "Benzophenone, 4-fluoro-": "C1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)F",

    # Oligothiophenes / Furans (Previously grabbed "2-thienyl" fragment)
    "Furan, 2,5-di(2-thienyl)-": "C1=CSC(=C1)C2=CC=C(O2)C3=CC=CS3",
    "2,2':5',2''-Terthiophene, 5-bromo-": "C1=CSC(=C1)C2=CC=C(S2)C3=CC=C(S3)Br",
    "2,2':5',2''-Terthiophene, 5-cyano-": "C1=CSC(=C1)C2=CC=C(S2)C3=CC=C(S3)C#N",
    "2,2':5',2''-Terthiophene, 5,5''-dibromo-": "C1=C(SC(=C1)C2=CC=C(S2)C3=CC=C(S3)Br)Br",
    "2,2':5',2''-Terthiophene, 5-methyl-": "CC1=CC=C(S1)C2=CC=C(S2)C3=CC=CS3",
    "alpha-Terthienyl": "C1=CSC(=C1)C2=CC=C(S2)C3=CC=CS3",
    
    # Phthalocyanines / Naphthalocyanines (Previously grabbed "1,1-dimethylethyl" -> t-butyl)
    "Naphthalocyanine, 2,11,20,29-tetrakis(1,1-dimethylethyl)-": "CC(C)(C)C1=CC2=C3C=C1C=CC3=C4N=C5C6=CC(=CC7=C6C(=N5)N=C8C9=C(C=C(C=C9)C(C)(C)C)C(=N8)N=C2N4)C(C)(C)C", # Isomer approx
    "Phthalocyanine, 2,9,16,23-tetrakis(1,1-dimethylethyl)-": "CC(C)(C)C1=CC2=C(C=C1)C3=NC4=NC(=NC5=CC(=C(C=C5N5)C(C)(C)C)N=C(N3)C3=CC(=C(C=C3)C(C)(C)C)C(C)(C)C)C3=CC(=C(C=C34)C(C)(C)C)C(C)(C)C", # Isomer approx
    
    # Others
    "Acetophenone, 2'-methyl-, biradical": "CC1=CC=CC=C1C(=O)C", # 2-methylacetophenone (ignoring biradical state notation for SMILES)
    "9,10-Anthraquinone, 1-chloro-": "C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C(=CC=C3)Cl",
    "9,10-Anthraquinone, 2-chloro-": "C1=CC=C2C(=C1)C(=O)C3=CC(=CC=C3C2=O)Cl",
    "Pyridine, 2,6-bis(2-thienyl)-": "C1=CSC(=C1)C2=NC(=CC=C2)C3=CC=CS3",
}

def main():
    file_path = 'data/wilkinson_with_smiles.csv' # Or whatever your latest file is named
    print(f"Loading {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found. Please check the path.")
        return

    # 1. APPLY MANUAL PATCHES
    print("Applying manual patches...")
    initial_matches = 0
    for name, smiles in PATCH_FIXES.items():
        mask = df['Structure'] == name
        if mask.any():
            df.loc[mask, 'SMILES'] = smiles
            initial_matches += mask.sum()
    print(f"Patched {initial_matches} rows with correct structures.")

    # 2. AUDIT FOR REMAINING BAD SMILES
    # Logic: If SMILES is very short (< 8 chars) but the name is long (> 15 chars), it's likely a fragment error.
    # Excludes simple molecules like "Benzene" or solvents if they are in the Structure column.
    
    print("\nAuditing for suspicious fragments...")
    
    # Calculate lengths
    df['Smiles_Len'] = df['SMILES'].astype(str).apply(len)
    df['Name_Len'] = df['Structure'].astype(str).apply(len)
    
    # Define "Suspicious": Name is complex (>15 chars) but SMILES is tiny (<10 chars)
    # This catches things like "C[N]C" (len 5) for "Benzophenone..." (len 30+)
    suspicious_mask = (df['Smiles_Len'] < 10) & (df['Name_Len'] > 15) & (df['SMILES'].notna())
    
    bad_rows = df[suspicious_mask]
    
    if not bad_rows.empty:
        print(f"Found {len(bad_rows)} suspicious entries. Examples:")
        print(bad_rows[['Structure', 'SMILES']].head(10))
        
        # Action: Clear them
        print(f"\nClearing {len(bad_rows)} suspicious SMILES to prevent training on garbage data.")
        df.loc[suspicious_mask, 'SMILES'] = None
    else:
        print("No obvious fragment errors found.")

    # 3. FINAL SAVE
    # Drop temp columns
    df = df.drop(columns=['Smiles_Len', 'Name_Len'])
    
    output_file = 'data/final_training_set_clean.csv'
    
    # Ensure correct column order for training
    cols = ['Structure', 'Solvent', 'PhiDelta', 'SMILES', 'Dielectric', 'lambda_max']
    # Filter only rows that have valid SMILES now
    df_clean = df.dropna(subset=['SMILES'])[cols]
    
    df_clean.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"Audit Complete.")
    print(f"Total Valid Rows: {len(df_clean)}")
    print(f"Saved cleaned dataset to: {output_file}")

if __name__ == "__main__":
    main()