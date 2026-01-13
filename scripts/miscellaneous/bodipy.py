import pdfplumber
import pandas as pd
import re

# File path (ensure this matches your local file name)
pdf_path = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/notes/chem202100922-sup-0001-misc_information.pdf"
output_csv = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/data/bodipy_singlet_oxygen_data.csv"

# Pages containing Table S1 (Based on the document structure S3-S11)
# Adjust these indices if the page mapping shifts (0-indexed)
start_page = 2  # Page S3 is usually the 3rd page (index 2)
end_page = 10   # Page S11 is usually the 11th page (index 10)

all_rows = []

print(f"Processing {pdf_path}...")

with pdfplumber.open(pdf_path) as pdf:
    # Iterate through the relevant pages
    for i in range(start_page, end_page + 1):
        page = pdf.pages[i]
        
        # Extract table data
        # pdfplumber's extract_table() tries to guess the grid
        table = page.extract_table()
        
        if table:
            # Clean up rows
            for row in table:
                # Remove None values and newlines
                clean_row = [cell.replace('\n', ' ').strip() if cell else "" for cell in row]
                
                # Simple filter to remove empty rows or header repetitions
                if any(clean_row) and "Structure" not in clean_row[0]:
                    all_rows.append(clean_row)
        
        print(f"Extracted data from Page {i+1}")

# Create DataFrame
columns = ["Structure", "Solvent", "Phi_Delta", "Comments", "Reference"]
# Note: Extracted tables might vary in column count; strict handling ensures no crashes
df = pd.DataFrame(all_rows)

# Attempt to assign proper column names if the shape matches
if df.shape[1] == 5:
    df.columns = columns
else:
    print(f"Warning: Extracted {df.shape[1]} columns. Saving without strict headers.")

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"\nSuccess! Data saved to {output_csv}")
print(df.head())