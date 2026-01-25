import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem

# --- REFINED COLOR PALETTE ---
C_BACKGROUND = "#FFFFFF"  # White
C_CARBON     = "#565AA2"  # Indigo (Primary structure)
C_HYDROGEN   = "#B5BCBE"  # Light Gray
C_HETERO     = "#F6A21C"  # Bright Orange (N, O, S, P)

# --- DISTINCT SPECIAL ATOMS ---
C_BORON      = "#8E44AD"  # Deep Purple (Distinct from Indigo)
C_IODINE     = "#A93226"  # Dark Crimson (Distinct from Boron)
C_HALOGEN    = "#D35400"  # Burnt Sienna (F, Cl, Br - warm tone, NO GREEN)

def show_molecule(smiles, width=600, height=400):
    """
    Generates a 3D visualization where Boron and Iodine are distinct colors,
    fitting the Indigo/Orange/Purple theme.
    """
    # 1. Generate 3D Structure using RDKit
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("Error: Invalid SMILES string.")
        return None
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG()) 
    AllChem.UFFOptimizeMolecule(mol) 
    mol_block = Chem.MolToMolBlock(mol)

    # 2. Setup Viewer
    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_block, 'mol')

    # 3. Define Atom Mapping
    element_map = {
        'C': C_CARBON,
        'H': C_HYDROGEN,
        
        # Standard Heteroatoms -> Orange
        'N': C_HETERO, 'O': C_HETERO, 'S': C_HETERO, 'P': C_HETERO,
        
        # Special Atoms -> Distinct Colors
        'B': C_BORON,    # Purple
        'I': C_IODINE,   # Crimson
        
        # Other Halogens -> Burnt Sienna (No Green)
        'F': C_HALOGEN, 'Cl': C_HALOGEN, 'Br': C_HALOGEN
    }

    # 4. Apply Style
    view.setStyle({
        'stick': {
            'colorscheme': {'prop': 'elem', 'map': element_map},
            'radius': 0.15 
        },
        'sphere': {
            'colorscheme': {'prop': 'elem', 'map': element_map},
            'scale': 0.3
        }
    })

    # 5. Final Polish
    view.setBackgroundColor(C_BACKGROUND)
    view.zoomTo()
    
    return view

# --- EXAMPLE USAGE ---
# Your Iodinated Fused Bis(benzo-BODIPY)
target_smiles = "CC1=c2cc3c(cc2=[N+]2C1=Cc1c4cc(I)ccc4c(C)n1[B-]2(F)F)C1=Cc2c4cc(I)ccc4c(C)n2[B-](F)(F)[N+]1=C3C"

print(f"Visualizing: {target_smiles}")
viewer = show_molecule(target_smiles)
viewer.show()