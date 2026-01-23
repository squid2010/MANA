import pandas as pd


def extract_unique_smiles(csv_path, output_txt_path):
    """
    Extracts unique SMILES strings from the 'Chromophore' column of a CSV file
    and saves them as a comma-separated string in a text file.

    Args:
        csv_path (str): The path to the input CSV file.
        output_txt_path (str): The path to the output text file.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if "Chromophore" not in df.columns:
        print("Error: The 'Chromophore' column was not found in the CSV file.")
        return

    unique_smiles = df["Chromophore"].dropna().unique()

    smiles_string = ",".join(unique_smiles)

    try:
        with open(output_txt_path, "w") as f:
            f.write(smiles_string)
        print(f"Unique SMILES extracted and saved to '{output_txt_path}'")
    except Exception as e:
        print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    input_csv = "data/lambda/lambdamax_dataset.csv"
    output_txt = "unique_photosensitizers_smiles.txt"
    extract_unique_smiles(input_csv, output_txt)
