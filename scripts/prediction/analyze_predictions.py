import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_and_clean(csv_path, target_col, true_col, pred_col, desc):
    """Loads CSV, ensures columns exist, and drops NaNs for the specific target."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Check required columns
    if true_col not in df.columns or pred_col not in df.columns:
        print(f"Warning: {desc} missing columns {true_col} or {pred_col}")
        return pd.DataFrame()
    
    # Filter valid rows (NaNs and zeros/empty if that indicates missing)
    # For Lambda: valid > 10 nm
    # For Phi: valid 0 <= x <= 1
    
    df_clean = df.dropna(subset=[true_col, pred_col])
    
    if target_col == 'lambda':
        # Ignore phi data in lambda file, assume rows with lambda < 10 are invalid/dummy
        df_clean = df_clean[df_clean[true_col] > 10.0]
    elif target_col == 'phi':
        # Ignore lambda data in phi file
        # Phi must be between 0 and 1
        df_clean = df_clean[(df_clean[true_col] >= 0.0) & (df_clean[true_col] <= 1.0)]
        
    print(f"Loaded {desc}: {len(df_clean)} valid rows")
    return df_clean

def main():
    parser = argparse.ArgumentParser(description="Analyze Performance Drop (Trained vs New Families)")
    parser.add_argument("--pred_lambda", required=True, help="Predictions on Lambda Trained Set (CSV)")
    parser.add_argument("--pred_phi", required=True, help="Predictions on Phi Trained Set (CSV)")
    parser.add_argument("--pred_new", required=True, help="Predictions on New/Novel Families (CSV)")
    parser.add_argument("--output_dir", default="analysis_results", help="Directory to save plots")
    args = parser.parse_args()

    # 1. Load Data for LAMBDA Comparison
    # "Trained" comes from the lambda file, "New" comes from the new families file
    print("\n--- Loading Lambda Data ---")
    df_lambda_trained = load_and_clean(args.pred_lambda, 'lambda', 'lambda_true', 'lambda_pred', "Lambda Trained")
    df_lambda_new = load_and_clean(args.pred_new, 'lambda', 'lambda_true', 'lambda_pred', "Lambda New")

    # 2. Load Data for PHI Comparison
    # "Trained" comes from the phi file, "New" comes from the new families file
    print("\n--- Loading Phi Data ---")
    df_phi_trained = load_and_clean(args.pred_phi, 'phi', 'phi_true', 'phi_pred', "Phi Trained")
    df_phi_new = load_and_clean(args.pred_new, 'phi', 'phi_true', 'phi_pred', "Phi New")

    # 3. Calculate Metrics
    results = []

    # Lambda Metrics
    if not df_lambda_trained.empty:
        r = rmse(df_lambda_trained['lambda_true'], df_lambda_trained['lambda_pred'])
        results.append({"Task": "Absorption (Lambda)", "Family": "Trained", "RMSE": r})
    
    if not df_lambda_new.empty:
        r = rmse(df_lambda_new['lambda_true'], df_lambda_new['lambda_pred'])
        results.append({"Task": "Absorption (Lambda)", "Family": "New", "RMSE": r})

    # Phi Metrics
    if not df_phi_trained.empty:
        r = rmse(df_phi_trained['phi_true'], df_phi_trained['phi_pred'])
        results.append({"Task": "Quantum Yield (Phi)", "Family": "Trained", "RMSE": r})

    if not df_phi_new.empty:
        r = rmse(df_phi_new['phi_true'], df_phi_new['phi_pred'])
        results.append({"Task": "Quantum Yield (Phi)", "Family": "New", "RMSE": r})

    df_res = pd.DataFrame(results)
    print("\n--- Results Summary ---")
    print(df_res)

    # 4. Calculate Drops and Plot
    print("\n--- Performance Drops ---")
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # --- Lambda Plot & Drop ---
    l_rows = df_res[df_res['Task'] == "Absorption (Lambda)"]
    if len(l_rows) == 2:
        trained_err = l_rows[l_rows['Family']=='Trained']['RMSE'].values[0]
        new_err = l_rows[l_rows['Family']=='New']['RMSE'].values[0]
        print(f"Lambda Drop: {new_err - trained_err:.2f} nm (Train: {trained_err:.2f}, New: {new_err:.2f})")
        
        plt.figure(figsize=(6, 5))
        ax = sns.barplot(data=l_rows, x='Family', y='RMSE', palette=['#565AA2', '#F6A21C'])
        plt.title("Absorption Generalization Gap", fontsize=14, fontweight='bold')
        plt.ylabel("RMSE (nm)", fontsize=12)
        # Add labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/lambda_gap.png", dpi=300)
        plt.close()
    
    # --- Phi Plot & Drop ---
    p_rows = df_res[df_res['Task'] == "Quantum Yield (Phi)"]
    if len(p_rows) == 2:
        trained_err = p_rows[p_rows['Family']=='Trained']['RMSE'].values[0]
        new_err = p_rows[p_rows['Family']=='New']['RMSE'].values[0]
        print(f"Phi Drop:    {new_err - trained_err:.4f} (Train: {trained_err:.4f}, New: {new_err:.4f})")

        plt.figure(figsize=(6, 5))
        ax = sns.barplot(data=p_rows, x='Family', y='RMSE', palette=['#565AA2', '#F6A21C'])
        plt.title("Quantum Yield Generalization Gap", fontsize=14, fontweight='bold')
        plt.ylabel("RMSE (Unitless)", fontsize=12)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/phi_gap.png", dpi=300)
        plt.close()

    print(f"\nAnalysis complete. Plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()