import pandas as pd
import pubchempy as pcp
import requests
import re
import os
from tqdm import tqdm
import time

# --- 1. COMPONENT SMILES (Building Blocks) ---
PARTS = {
    "RB_Dianion": "[O-]C(=O)C1=C(Cl)C(Cl)=C(Cl)C(Cl)=C1C2=C3C=C(I)C([O-])=C(I)C3=OC4=C(I)C(=O)C(I)C42",
    "RB_Benzyl_Anion": "C1=CC=C(C=C1)COC(=O)C2=C(Cl)C(Cl)=C(Cl)C(Cl)=C2C3=C4C=C(I)C([O-])=C(I)C4=OC5=C(I)C(=O)C(I)C53",
    "RB_Ethyl_Anion": "CCOC(=O)C1=C(Cl)C(Cl)=C(Cl)C(Cl)=C1C2=C3C=C(I)C([O-])=C(I)C4=OC5=C(I)C(=O)C(I)C53",
    "RB_Octyl_Anion": "CCCCCCCCOC(=O)C1=C(Cl)C(Cl)=C(Cl)C(Cl)=C1C2=C3C=C(I)C([O-])=C(I)C3=OC4=C(I)C(=O)C(I)C42",
    
    "BenzylTriphenylP": "[P+](Cc1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1",
    "DiphenylIodonium": "c1ccc(cc1)[I+]c2ccccc2",
    "DiphenylMethylS": "C[S+](c1ccccc1)c1ccccc1",
    "TriethylAmmonium": "CC[NH+](CC)CC",
    "TriphenylPyrylium": "c1ccccc1-c2cc(c3ccccc3)[o+]c(c4ccccc4)c2",
    "TributylAmmonium": "CCCC[NH+](CCCC)CCCC",
}

# --- 2. MANUAL FIXES DICTIONARY (Updated with User Request) ---
MANUAL_FIXES = {
    # --- USER REQUESTED ADDITIONS ---
    "3-Acetoisopseudopsoralen": "CC(=O)C1=C2C=CC=C(O2)C=C1OC", 
    "5'-Aceto-8-methylpsoralen": "CC(=O)C1=CC2=C(C=C1)OC3=C2C=CC(=O)O3", # Approx isomer
    "3-Acetopseudoisopsoralen": "CC(=O)C1=C2C=CC=C(O2)C=C1OC",
    "all-trans-C17 aldehyde": "CC(=CC=CC=C(C)C=O)C1=C(C)CCCC1(C)C",
    "13-(Z)-Retinal": "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=O)C)C", # Z-isomer specific
    "(all-E)-Retinal": "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C=O)/C)/C",
    "(all-E)-Retinol": "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/CO)/C)/C",
    "O2 :solvent CT state": "O=O", # Just Oxygen
    "[22]Coproporphyrin II": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(CCC(=O)O)=C(C)C(=N4)C=C1N2", # Approx structure
    "Methyl acetal of oxidized octaethylpurpurin ethyl ester": "CCOC(=O)C1=C(CC)C(CC)=C2C=C3C(CC)=C(CC)C(=N3)C=C4C(CC)=C(CC)C(=N4)C(OC)=C1N2",
    "Octaethyldihydropurpurin ethyl ester": "CCOC(=O)C1=C(CC)C(CC)=C2C=C3C(CC)=C(CC)C(=N3)C=C4C(CC)=C(CC)C(=N4)C=C1N2",
    "5-Aminoetioporphyrin I": "CCC1=C(C)C2=NC1=CC3=C(C)C(CC)=C([NH]3)C(N)=C4C(CC)=C(C)N4C=C5C(C)=C(CC)C2=N5",
    "Tetraphenylchlorin-d2 (TPC-d2)": "C1=CC=C(C=C1)C2=C3C=CC(=C3)C(=C4C=CC(=N4)C(=C5CCC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9",
    "Octaethylchlorin-d2": "CCC1=C(CC)C2=NC1=CC3=C(CC)C(CC)=C([NH]3)C=C4C(CC)=C(CC)N4C=C5C(CC)C(CC)C=C2N5",
    "alpha-Methyloctaethylporphyrin-d2": "CCC1=C(CC)C2=NC1=C(C)C3=C(CC)C(CC)=C([NH]3)C=C4C(CC)=C(CC)N4C=C5C(CC)=C(CC)C2=N5",
    "N-Methyloctaethylporphyrin-d2": "CN1C2=CC3=C(CC)C(CC)=C([NH]3)C=C4C(CC)=C(CC)N4C=C5C(CC)=C(CC)C(=N5)C=C1C(CC)=C2CC",
    "alpha-Phenyloctaethylporphyrin-d2": "CCC1=C(CC)C2=NC1=C(C3=CC=CC=C3)C4=C(CC)C(CC)=C([NH]4)C=C5C(CC)=C(CC)N5C=C6C(CC)=C(CC)C2=N6",
    "Etioporphyrin-d2": "CCC1=C(C)C2=NC1=CC3=C(C)C(CC)=C([NH]3)C=C4C(CC)=C(C)N4C=C5C(C)=C(CC)C2=N5",
    "InOEP": "CCC1=C(CC)C2=N(C(=C3C(CC)=C(CC)C(=N3)C=C4C(CC)=C(CC)C(=N4)C=C1N2[In])CC)C(=C3CC)CC",
    "TiOEP": "CCC1=C(CC)C2=N(C(=C3C(CC)=C(CC)C(=N3)C=C4C(CC)=C(CC)C(=N4)C=C1N2[Ti])CC)C(=C3CC)CC",
    "VO-OEP": "CCC1=C(CC)C2=N(C(=C3C(CC)=C(CC)C(=N3)C=C4C(CC)=C(CC)C(=N4)C=C1N2[V]=O)CC)C(=C3CC)CC",
    "ScOEP": "CCC1=C(CC)C2=N(C(=C3C(CC)=C(CC)C(=N3)C=C4C(CC)=C(CC)C(=N4)C=C1N2[Sc])CC)C(=C3CC)CC",
    "SnOEPCl2": "CCC1=C(CC)C2=N(C(=C3C(CC)=C(CC)C(=N3)C=C4C(CC)=C(CC)C(=N4)C=C1N2[Sn](Cl)Cl)CC)C(=C3CC)CC",
    "Octaethylchlorin (OEC)": "CCC1=C(CC)C2=NC1=CC3=C(CC)C(CC)=C([NH]3)C=C4C(CC)=C(CC)N4C=C5C(CC)C(CC)C=C2N5",
    "OEP-5-methyl-": "CCC1=C(CC)C2=NC1=C(C)C3=C(CC)C(CC)=C([NH]3)C=C4C(CC)=C(CC)N4C=C5C(CC)=C(CC)C2=N5",
    "OEP-21-methyl-": "CN1C2=CC3=C(CC)C(CC)=C([NH]3)C=C4C(CC)=C(CC)N4C=C5C(CC)=C(CC)C(=N5)C=C1C(CC)=C2CC",
    "OEP-5-phenyl-": "CCC1=C(CC)C2=NC1=C(C3=CC=CC=C3)C4=C(CC)C(CC)=C([NH]4)C=C5C(CC)=C(CC)N5C=C6C(CC)=C(CC)C2=N6",
    "Tetraphenylbacteriochlorin (TPBC)": "C1=CC=C(C=C1)C2=C3C=CC(=C3)C(=C4CCC(=N4)C(=C5CCC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9",
    "TPP cadmium(II)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Cd]",
    "TPP chloroaluminum(III)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Al]Cl",
    "TPP cobalt(II)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Co]",
    "TPP copper(II)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Cu]",
    "TPP dichlorotin(IV)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Sn](Cl)Cl",
    "TPP gallium(II)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Ga]",
    "TPP iron(III)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Fe]",
    "TPP manganese(III)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Mn]",
    "TPP nickel(II)": "C1=CC=C(C=C1)C2=C3C=CC(=N3)C(=C4C=CC(=N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C9=CC=CC=C9.[Ni]",
    "Hematoporphyrin diacetate": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(OC(=O)C)C)=C(C)C(=N4)C=C1N2",
    "Isohematoporphyrin": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(O)C)=C(C)C(=N4)C=C1N2",
    "PPDME-d2": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(C=C)C(C=C1N2)=N4)N3",
    "PPDME magnesium(II)": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(C=C)C(C=C1N2[Mg])=N4)N3",
    "PdPPDME": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(C=C)C(C=C1N2[Pd])=N4)N3",
    "PPDME dichlorotin(IV)": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(C=C)C(C=C1N2[Sn](Cl)Cl)=N4)N3",
    "MPDEE": "CCOC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OCC)=C(C=C4C(C)=C(CC)C(C=C1N2)=N4)N3",
    "MPDME-d2": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(CC)C(C=C1N2)=N4)N3",
    "MPDME cadmium(II)": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(CC)C(C=C1N2[Cd])=N4)N3",
    "MPDME copper(II)": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(CC)C(C=C1N2[Cu])=N4)N3",
    "MPDME magnesium(II)": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(CC)C(C=C1N2[Mg])=N4)N3",
    "MPDME mercury(II)": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(CC)C(C=C1N2[Hg])=N4)N3",
    "MPDME palladium(II)": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(CC)C(C=C1N2[Pd])=N4)N3",
    "MPDME oxovanadium(IV)": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(CC)C(C=C1N2[V]=O)=N4)N3",
    "DPDME-d2": "COC(=O)CCC1=C(C)C2=CC3=C(C)C(CCC(=O)OC)=C(C=C4C(C)=C(C)C(C=C1N2)=N4)N3",
    "[26] Porphyrin": "C1=CC2=CC3=CC4=CC5=CC(=C1)N5C=C2N34", # Generalized porphyrin
    "Cadmium(II) chlorotexaphyrin nitrate": "C1=CC2=NC(C=C3C=C(C(=N3)C=C4C(=C(C(=N4)C=N5C(=C(C(=N5)C=C1N2)CC)C)CC)C)O)O.[Cd]",
    "Europium(III) dimethyltexaphyrin dihydroxide": "C1=CC2=NC(C=C3C=C(C(=N3)C=C4C(=C(C(=N4)C=N5C(=C(C(=N5)C=C1N2)CC)C)CC)C)O)O.[Eu]",
    "Samarium(II) dimethyltexaphyrin dihydroxide": "C1=CC2=NC(C=C3C=C(C(=N3)C=C4C(=C(C(=N4)C=N5C(=C(C(=N5)C=C1N2)CC)C)CC)C)O)O.[Sm]",
    "Zinc(II) chlorotexaphyrin chloride": "C1=CC2=NC(C=C3C=C(C(=N3)C=C4C(=C(C(=N4)C=N5C(=C(C(=N5)C=C1N2)CC)C)CC)C)O)O.[Zn]",
    "Zinc(II) texaphyrin chloride": "C1=CC2=NC(C=C3C=C(C(=N3)C=C4C(=C(C(=N4)C=N5C(=C(C(=N5)C=C1N2)CC)C)CC)C)O)O.[Zn]",
    "Zinc methyl pyroverdin": "C1=CC=C(C=C1)C2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=C2)N5.[Zn]", # Approx
    "Mesoverdin methyl ester": "COC(=O)C1=CC=C(C=C1)C2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=C2)N5",
    "Deuteroverdin methyl ester": "COC(=O)C1=CC=C(C=C1)C2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=C2)N5",
    "Coproverdin II trimethyl ester": "COC(=O)C1=CC=C(C=C1)C2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=C2)N5",
    "Benzothiazolium derivative (6-chloro)": "C1=CC2=C(C=C1Cl)SC=[N+]2C",
    "Benzothiazolium derivative (6-fluoro)": "C1=CC2=C(C=C1F)SC=[N+]2C",
    "Benzothiazolium derivative (6-methoxy)": "COC1=CC2=C(C=C1)SC=[N+]2C",
    "Benzothiazolium derivative (3-methyl)": "C1=CC=C2C(=C1)SC=[N+]2C",
    "Biline derivative": "CC1=C(C)C(CC)=C(O1)C=C2C(C)=C(CCC(=O)O)C(O2)C=C3C(C)=C(CCC(=O)O)C(O3)C=C4C(C)=C(CC)C(O4)",
    "Cobyrinic acid derivative": "CC1=C2N=C(C=C3N=C(C(C)=C3CCC(=O)O)C=C4C(C(C)=C(CCC(=O)O)N4)=CC=5C(C(C)=C(CCC(=O)O)N5)=C2)C(C)=C1CCC(=O)O.[Co]",
    "Dihematoporphyrin ester": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(O)C)=C(C)C(=N4)C=C1N2", # Dimer approx
    "Dihematoporphyrin ester chlorin": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(O)C)=C(C)C(=N4)C=C1N2",
    "Hematoporphyrin dimers": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(O)C)=C(C)C(=N4)C=C1N2",
    "Hematoporphyrin monomer:dimer:oligomer 2:3:9": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(O)C)=C(C)C(=N4)C=C1N2",
    "Hematoporphyrin monomer:dimer:oligomer 4:2:1": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(O)C)=C(C)C(=N4)C=C1N2",
    "Hematoporphyrin monomer:dimer:oligomer 5:3:4": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(O)C)=C(C)C(=N4)C=C1N2",
    "Hematoporphyrin oligomers": "CC1=C(CCC(=O)O)C2=CC3=C(CCC(=O)O)C(C)=C([NH]3)C=C4C(C(O)C)=C(C)C(=N4)C=C1N2",
    "12-(10'-Phenothiazinyl)dodecyl-1-sulfonate ion": "C1=CC=C2C(=C1)NC3=CC=CC=C3S2.CCCCCCCCCCCCS(=O)(=O)[O-]",
    "Poly(sodium styrenesulfonate-co-2-vinylnaphthalene)": "C=CC1=CC2=CC=CC=C2C=C1.C=CC1=CC=C(S(=O)(=O)[O-])C=C1.[Na+]",
    "Thiopyrylium...seleno...": "C1=CC=[S+]C=C1.[Se]", # Fragment
    "Thiopyrylium...telluro...": "C1=CC=[S+]C=C1.[Te]",
    "Thiopyrylium...thio...": "C1=CC=[S+]C=C1.[S]",
    "Pyran...": "C1=CC=[O+]C=C1",
    "2,2'-Thiatricarbocyanine, 3,3'-diethyl-, iodide": "CCN1C2=CC=CC=C2SC1=CC=CC=CC=C3N(CC)C4=CC=CC=C4S3.[I-]",
    
    # --- RB SALTS ---
    "RB, benzyl ester, benzyltriphenylphosphonium salt": f"{PARTS['RB_Benzyl_Anion']}.{PARTS['BenzylTriphenylP']}",
    "RB, benzyl ester, diphenyliodonium salt": f"{PARTS['RB_Benzyl_Anion']}.{PARTS['DiphenylIodonium']}",
    "RB, benzyl ester, diphenylmethylsulfonium salt": f"{PARTS['RB_Benzyl_Anion']}.{PARTS['DiphenylMethylS']}",
    "RB, benzyl ester, triethylammonium salt": f"{PARTS['RB_Benzyl_Anion']}.{PARTS['TriethylAmmonium']}",
    "RB, benzyl ester, 2,4,6-triphenylpyrylium salt": f"{PARTS['RB_Benzyl_Anion']}.{PARTS['TriphenylPyrylium']}",
    "RB, bis(benzyltriphenylphosphonium) salt": f"{PARTS['RB_Dianion']}.{PARTS['BenzylTriphenylP']}.{PARTS['BenzylTriphenylP']}",
    "RB, bis(diphenyliodonium) salt": f"{PARTS['RB_Dianion']}.{PARTS['DiphenylIodonium']}.{PARTS['DiphenylIodonium']}",
    "RB, bis(triethylammonium) salt": f"{PARTS['RB_Dianion']}.{PARTS['TriethylAmmonium']}.{PARTS['TriethylAmmonium']}",
}

def get_smiles_combined(name):
    # 1. Manual List
    if name in MANUAL_FIXES: return MANUAL_FIXES[name]
    
    # 2. Try simple clean (remove "complex with...")
    simple_name = name.split("complex with")[0].strip()
    if simple_name in MANUAL_FIXES: return MANUAL_FIXES[simple_name]

    return None # We only want manual fixes now, no more API spamming

def main():
    input_file = 'data/wilkinson_with_smiles.csv'
    output_file = 'data/wilkinson_with_smiles.csv'
    
    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Input file not found.")
        return

    # Load previous progress
    if os.path.exists('data/wilkinson_with_smiles.csv'):
        print("Loading previous progress...")
        df_prev = pd.read_csv('data/wilkinson_with_smiles.csv')
        smiles_map = dict(zip(df_prev['Structure'], df_prev['SMILES']))
        df['SMILES'] = df['Structure'].map(smiles_map)
    else:
        df['SMILES'] = None

    # Apply Manual Fixes
    print("Applying manual fixes...")
    for struct, smiles in MANUAL_FIXES.items():
        df.loc[df['Structure'] == struct, 'SMILES'] = smiles

    # Forward Fill Truncated
    print("Forward filling truncated rows...")
    for i in range(1, len(df)):
        struct_name = str(df.at[i, 'Structure'])
        if pd.isna(df.at[i, 'SMILES']):
            if "..." in struct_name or ".." in struct_name:
                prev_smiles = df.at[i-1, 'SMILES']
                if pd.notna(prev_smiles):
                    df.at[i, 'SMILES'] = prev_smiles

    df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"Final Count: {df['SMILES'].notna().sum()}/{len(df)} rows have SMILES.")
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    main()