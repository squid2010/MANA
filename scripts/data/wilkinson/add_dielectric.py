import pandas as pd
import os

# --- Configuration ---
# We read the cleaned file from the previous step
INPUT_FILE = "data/fluorescence_dataset.csv" 
OUTPUT_FILE = "data/fluorescence_dataset.csv"

# --- Solvent SMILES to Dielectric Constant Map ---
# Values taken from standard chemical references at 20-25°C
SOLVENT_MAP = {
    # Polar Protic
    "O": 78.54,              # Water
    "[2H]O[2H]": 78.54,      # D2O (approx same as H2O)
    "CO": 32.7,              # Methanol
    "[2H]OC": 32.7,          # MeOD
    "CCO": 24.5,             # Ethanol
    "[2H]OCC": 24.5,         # EtOD
    "CCCO": 20.1,            # 1-Propanol
    "CC(C)O": 17.9,          # Isopropanol (2-Propanol)
    "CC(C)CO": 17.5,         # Isobutanol
    "CC(=O)O": 6.15,         # Acetic Acid
    
    # Polar Aprotic
    "CS(C)=O": 46.7,         # DMSO
    "CN(C)C=O": 36.7,        # DMF
    "CC(=O)N(C)C": 37.78,    # DMA (Dimethylacetamide)
    "CC#N": 37.5,            # Acetonitrile
    "CC(C)=O": 20.7,         # Acetone
    "CC(=O)C(C)(C)C": 13.0,  # Pinacolone
    "c1ccncc1": 12.4,        # Pyridine
    "CC(=O)OC(C)=O": 20.7,   # Acetic Anhydride
    
    # Chlorinated
    "ClCCl": 8.93,           # Dichloromethane (DCM)
    "ClCCCl": 10.36,         # 1,2-Dichloroethane (DCE)
    "ClC(Cl)Cl": 4.81,       # Chloroform
    "ClC(Cl)(Cl)Cl": 2.24,   # Carbon Tetrachloride
    "Clc1ccccc1": 5.62,      # Chlorobenzene
    "Clc1ccccc1Cl": 9.93,    # o-Dichlorobenzene (approx)

    # Ethers / Esters
    "C1CCOC1": 7.58,         # THF
    "C1COCCO1": 2.25,        # 1,4-Dioxane
    "CCOCC": 4.33,           # Diethyl Ether
    "COCCOC": 7.2,           # DME (Dimethoxyethane)
    "COC(=O)C": 6.02,        # Methyl Acetate (approx)
    "CCOC(C)=O": 6.02,       # Ethyl Acetate
    
    # Aromatics / Hydrocarbons
    "Cc1ccccc1": 2.38,       # Toluene
    "c1ccccc1": 2.27,        # Benzene
    "C1CCCCC1": 2.02,        # Cyclohexane
    "CCCCCC": 1.88,          # Hexane
    "CCCCCCC": 1.92,         # Heptane
    "CCCCCCCC": 1.95,        # Octane
    "S=C=S": 2.64,           # Carbon Disulfide
}

def add_dielectric_column():
    print(f"Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run clean_dataset.py first.")
        return

    df = pd.read_csv(INPUT_FILE)

    # Function to lookup dielectric
    def get_dielectric(solvent_smiles):
        # Clean string just in case
        if not isinstance(solvent_smiles, str):
            return 0.0
        
        s = solvent_smiles.strip()
        
        # Direct lookup
        if s in SOLVENT_MAP:
            return SOLVENT_MAP[s]
        
        # Fallback: Many of your entries are likely dyes (solutes) incorrectly in the solvent column
        # We return 0.0 for these. The model will handle 0.0 as "unknown environment".
        return 0.0

    # Apply mapping
    print("Mapping solvents to dielectric constants...")
    # 'dielectric' column might already exist (empty), we overwrite it
    df['dielectric'] = df['Solvent'].apply(get_dielectric)

    # Statistics
    non_zero = df[df['dielectric'] > 0.0]
    zero_count = len(df) - len(non_zero)
    
    print("-" * 40)
    print(f"Total Rows: {len(df)}")
    print(f"Successfully Mapped Solvents: {len(non_zero)}")
    print(f"Unmapped (likely dirty data/solutes): {zero_count}")
    print("-" * 40)
    
    # Show some examples
    print("Sample Mapped Rows:")
    print(non_zero[['Solvent', 'dielectric']].head(5))

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Final dataset with dielectrics saved to: {OUTPUT_FILE}")
    print("You should now update build_flourescence_dataset.py to point to this file.")

if __name__ == "__main__":
    add_dielectric_column()