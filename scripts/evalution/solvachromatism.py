import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent.parent
project_root = script_dir.parent
sys.path.insert(0, str(script_dir))

from model.mana_model import MANA 
from data.dataset import DatasetConstructor 

# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------
C_PRIMARY = "#565AA2"   # Purple
C_SECONDARY = "#F6A21C" # Orange
C_BLACK = "#000000"
C_GRAY = "#B5BCBE"

NUM_ATOM_TYPES = 118

ANALYSIS_JOBS = [
    {
        "name": "Singlet Oxygen (Phi)",
        "id": "phi",
        "csv": project_root / "data" / "phi" / "phidelta_dataset.csv",
        "h5": project_root / "data" / "phi" / "phidelta_data.h5",
        "model": project_root / "models" / "phi" / "best_model.pth",
        "task": "phi",
        "target_attr": "phi_delta",
        "y_label": "Singlet Oxygen Yield (ΦΔ)",
        "y_lim": (0, 1.05)
    },
    {
        "name": "Absorption Max (Lambda)",
        "id": "lambda",
        "csv": project_root / "data" / "lambda" / "lambdamax_dataset.csv",
        "h5": project_root / "data" / "lambda" / "lambda_all_data.h5", 
        "model": project_root / "models" / "phi" / "best_model.pth", 
        "task": "lambda",
        "target_attr": "lambda_max",
        "y_label": "Absorption Max (λmax) [nm]",
        "y_lim": None 
    }
]

def get_solvent_map(csv_path):
    if not csv_path.exists(): return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        solv_col = next((c for c in ["Solvent", "solvent", "Solvent_SMILES"] if c in df.columns), None)
        if not solv_col: return None
        return df[solv_col].to_dict()
    except: return None

def plot_combined_grid(df_results, job, out_dir):
    counts = df_results.groupby('smiles')['solvent'].nunique()
    complex_mols = counts[counts >= 3].index.tolist()
    if not complex_mols: complex_mols = counts[counts >= 2].index.tolist()
    if not complex_mols: return

    interesting = []
    for smi in complex_mols:
        subset = df_results[df_results['smiles'] == smi]
        var = subset['True'].var()
        interesting.append((smi, var))
    interesting.sort(key=lambda x: x[1], reverse=True)
    top_mols = [x[0] for x in interesting[:9]]
    
    n_mols = len(top_mols)
    if n_mols == 0: return

    cols = 3
    rows = (n_mols + cols - 1) // cols
    fig_height = 4 * rows
    
    # Setup Style
    plt.rcParams['axes.edgecolor'] = C_BLACK
    plt.rcParams['grid.color'] = C_GRAY
    plt.rcParams['grid.alpha'] = 0.5
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, fig_height), constrained_layout=True)
    if rows*cols == 1: axes = [axes]
    else: axes = axes.flatten()

    sns.set_theme(style="whitegrid") # Resets params, so we might need to enforce colors again manually or via rcParams context

    for i, smi in enumerate(top_mols):
        ax = axes[i]
        subset = df_results[df_results['smiles'] == smi].copy()
        
        melted = subset.melt(
            id_vars=['solvent'], 
            value_vars=['True', 'Predicted'], 
            var_name='Type', 
            value_name='Value'
        )
        
        # Color Mapping: True = Purple, Pred = Orange
        custom_palette = {'True': C_PRIMARY, 'Predicted': C_SECONDARY}
        
        sns.barplot(
            data=melted, 
            x='solvent', 
            y='Value', 
            hue='Type',
            palette=custom_palette,
            alpha=0.9, 
            edgecolor=C_BLACK, 
            ax=ax
        )
        
        ax.set_title(f"Molecule {i+1}", fontsize=12, fontweight='bold', color=C_BLACK)
        ax.set_xlabel("")
        ax.set_ylabel(job['y_label'] if i % cols == 0 else "")
        if job['y_lim']: ax.set_ylim(job['y_lim'])
        ax.tick_params(axis='x', rotation=45, colors=C_BLACK)
        ax.tick_params(axis='y', colors=C_BLACK)
        if ax.get_legend(): ax.get_legend().remove()

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels() if n_mols > 0 else ([], [])
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12)
    
    plt.suptitle(f"{job['name']} - Solvatochromism Analysis", fontsize=16, fontweight='bold', y=1.03, color=C_BLACK)
    
    filename = f"summary_solvatochromism_{job['id']}.png"
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {filename}")

def run_analysis(job, device, out_dir):
    print(f"\n--- Analyzing: {job['name']} ---")
    if not job['h5'].exists() or not job['model'].exists(): return
    id_to_solvent = get_solvent_map(job['csv'])
    if not id_to_solvent: return

    dataset = DatasetConstructor(str(job['h5']), split_by_mol_id=True)
    _, _, test_loader = dataset.get_dataloaders(num_workers=0)

    l_mean = dataset.lambda_mean if not np.isnan(dataset.lambda_mean) else 500.0
    l_std = dataset.lambda_std if not np.isnan(dataset.lambda_std) else 100.0

    model = MANA(
        num_atom_types=NUM_ATOM_TYPES,
        hidden_dim=128,
        tasks=[job['task']], 
        lambda_mean=l_mean,
        lambda_std=l_std
    ).to(device)

    try:
        model.load_state_dict(torch.load(job['model'], map_location=device, weights_only=True), strict=False)
    except: return
    model.eval()

    data_records = []
    target_attr = job['target_attr']
    task_key = job['task']

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            batch = batch.to(device)
            preds = model(batch)
            if not hasattr(batch, target_attr): continue
            y_pred = preds[task_key].cpu().numpy()
            y_true = getattr(batch, target_attr).cpu().numpy()
            mol_ids = batch.mol_id.cpu().numpy()
            smiles = batch.smiles
            for i in range(len(y_pred)):
                val_true = y_true[i]
                if np.isnan(val_true) or val_true < 0: continue
                m_id = mol_ids[i]
                solvent_name = id_to_solvent.get(m_id, "Unknown")
                solvent_name = str(solvent_name).strip().strip('"').strip("'")
                if solvent_name.lower() in ['unknown', 'nan', 'none']: continue
                data_records.append({
                    "smiles": smiles[i],
                    "solvent": solvent_name,
                    "True": float(val_true),
                    "Predicted": float(y_pred[i])
                })

    if not data_records: return
    df_results = pd.DataFrame(data_records)
    plot_combined_grid(df_results, job, out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = project_root / "results" / "solvatochromism"
    os.makedirs(out_dir, exist_ok=True)
    for job in ANALYSIS_JOBS: run_analysis(job, device, out_dir)

if __name__ == "__main__":
    main()