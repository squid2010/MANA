import pandas as pd
import numpy as np

def process_wilkinson_data(input_file, output_file):
    # Dictionary of dielectric constants
    dielectric_dict = {
        'C6H6': 2.28, 'CD3OD': 32.7, 'MeOH': 32.7, 'CH3CN': 37.5, 'CDCl3': 4.81,
        'H2O': 78.4, 'CH3COCH3': 20.7, 'CCl4': 2.24, 'D2O': 78.4, 'iso-BuOH': 16.7,
        '1-BuOH': 17.5, '(C2H5)2O': 4.33, 'i-C5H11OH': 14.7, 'n-C6H14': 1.88,
        'C6H5CH3': 2.38, 'C6H5CN': 26.0, 'C6H5Cl': 5.62, 'n-C7H16': 1.92,
        'CH2Cl2': 8.93, 'CHCl3': 4.81, 'CS2': 2.64, 'DMF': 36.7, 'EtOH': 24.5,
        'HCONH2': 109.0, '1-PrOH': 20.3, '2-PrOH': 19.9, 'c-C4H8O': 7.58,
        'c-C6H12': 2.02, 'CD3CN': 37.5, 'ClCF2CCl2F': 2.41, 'CH3CO2C2H5': 6.02,
        'C5H5N': 12.4, 'H2O (mic)': 78.4, 'DMSO': 46.7, 'diox': 2.25, 'C6D6': 2.28,
        'n-C5H12': 1.84, 'C6H5CH2OH': 13.0, 'm-Cresol': 11.5, 'D2O (mic)': 78.4,
        'tert-BuOH': 12.4, 'c-C6H12 (mic)': 2.02, 'MeOD': 32.7, 'Hexanes': 1.88,
        '2-Methoxyethanol': 16.9, 'Propylene carbonate': 64.9, 'i-octane': 1.94,
        'air': 1.0, '(CH3)2CO': 20.7, 'Decalin': 2.17, 'c-C8H16': 2.0,
        '1,3,5-C6H3(CH3)3': 2.3, '1,2-C6H4(CH3)2': 2.57, '1,4-C6H4(CH3)2': 2.27,
        'c-C6H11OH': 15.0, 'D2O/EtOH (90:10)': 0.9*78.4 + 0.1*24.5,
        'D2O/EtOH (95:5)': 0.95*78.4 + 0.05*24.5, 'D2O (ves)': 78.4,
        'H2O (ves)': 78.4, 'n-C3H7I': 7.1, 'C6D5Br': 5.4, 'C6F6': 2.05,
        'C6H5Br': 5.4, 'C6H5F': 5.42, 'C6H5I': 4.6, '2-EtNp': 2.56, '1-MeNp': 2.55,
        'EtOD': 24.5, 'CD2Cl2': 8.93, 'H2O (cells)': 78.4, 'DMAA': 37.8,
        'air (mic)': 1.0
    }

    # Load data
    df = pd.read_csv(input_file)

    # Rename columns to match target format
    df = df.rename(columns={
        'Structure Name': 'Structure',
        'Phi_Delta': 'PhiDelta'
    })

    # Add dummy columns for SMILES and lambda_max
    df['SMILES'] = ""
    df['lambda_max'] = ""

    # Map dielectric values
    df['Dielectric'] = df['Solvent'].map(dielectric_dict)

    # Count skipped rows
    skipped_rows_count = df['Dielectric'].isna().sum()

    # Drop rows with missing dielectric values
    df_clean = df.dropna(subset=['Dielectric'])

    # Reorder columns
    df_final = df_clean[['Structure', 'Solvent', 'PhiDelta', 'SMILES', 'Dielectric', 'lambda_max']]

    # Save file
    df_final.to_csv(output_file, index=False)

    print(f"Rows skipped due to not having values: {skipped_rows_count}")

# Example usage (assuming file is in current directory)
process_wilkinson_data('data/wilkinson_raw.csv', 'data/wilkinson_cleaned.csv')