#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the scripts directory to the Python path for direct execution
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from data.dataset import DatasetConstructor
from model.mana_model import MANA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ValidationEngine:
    """
    Comprehensive validation and analysis engine for the MANA model.
    Evaluates model performance across all predicted properties and generates
    detailed analysis plots and reports.
    """

    def __init__(self, model_path, dataset_path, analysis_dir="analysis", device="cpu"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(exist_ok=True)
        self.device = device

        # Create timestamp for this validation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.analysis_dir / f"validation_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        # Load dataset and model
        self.dataset = None
        self.model = None
        self.test_loader = None

        # Results storage
        self.results = {}
        self.predictions = {}
        self.targets = {}
        self.metrics = {}

    def load_dataset(self, batch_size=32, num_workers=0):
        """Load and prepare the dataset"""
        print("Loading dataset...")

        self.dataset = DatasetConstructor(
            self.dataset_path,
            cutoff_radius=5,
            batch_size=batch_size,
            train_split=0.8,
            val_split=0.1,
            random_seed=42,
        )

        # Get test loader
        _, _, self.test_loader = self.dataset.get_dataloaders(num_workers=num_workers)

        print(f"✓ Dataset loaded: {len(self.test_loader.dataset)} test samples")
        print(
            f"  - Atom types: {self.dataset.num_atom_types} ({self.dataset.unique_atoms})"
        )

    def load_model(self):
        """Load the trained model"""
        print("Loading trained model...")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Initialize model with same architecture as training
        self.model = MANA(
            num_atom_types=self.dataset.num_atom_types,
            num_singlet_states=3,  # Standard configuration
            hidden_dim=128,
            num_layers=4,
            num_rbf=20,
        )

        # Load model weights
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        # Set model to train mode for gradient computation during validation
        self.model.train()

        print(f"✓ Model loaded from: {self.model_path}")

        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  - Total parameters: {total_params:,}")

    def run_predictions(self):
        """Run predictions on test set and collect results"""
        print("Running predictions on test set...")

        all_predictions = {"energies": [], "forces": [], "dipoles": [], "nac": []}
        all_targets = {"energies": [], "forces": [], "dipoles": [], "nac": []}
        batch_count = 0
        force_computation_errors = 0

        # Set model to train mode to ensure gradient flow
        print("Setting model to training mode for gradient computation...")
        self.model.train()

        for batch in self.test_loader:
            batch = batch.to(self.device)
            batch_count += 1

            # Clear any existing gradients
            if batch.pos.grad is not None:
                batch.pos.grad.zero_()

            # Enable gradients for positions
            batch.pos.requires_grad_(True)

            # Debug: Check if positions have gradients enabled
            if not batch.pos.requires_grad:
                print(
                    f"ERROR: batch.pos.requires_grad is False for batch {batch_count}"
                )
                continue

            try:
                # Forward pass with gradient tracking
                pred_energies, pred_dipoles, pred_nac = self.model(batch)

                # Debug: Check if predictions have gradients
                if not pred_energies.requires_grad:
                    print(
                        f"Warning: Predicted energies don't require grad for batch {batch_count}"
                    )

                # Compute forces from ground state energy
                ground_energies = pred_energies[:, 0]  # Ground state only

                # Ensure we have a scalar for backward pass
                total_energy = ground_energies.sum()

                # Debug gradient computation
                if total_energy.requires_grad:
                    # Compute gradients for forces
                    gradients = torch.autograd.grad(
                        outputs=total_energy,
                        inputs=batch.pos,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False,  # Changed to False to catch issues
                    )

                    if gradients[0] is not None:
                        pred_forces = -gradients[0]
                    else:
                        print(f"Warning: No gradients returned for batch {batch_count}")
                        pred_forces = torch.zeros_like(batch.pos)
                        force_computation_errors += 1
                else:
                    print(
                        f"Warning: Total energy doesn't require gradients for batch {batch_count}"
                    )
                    pred_forces = torch.zeros_like(batch.pos)
                    force_computation_errors += 1

            except Exception as e:
                print(f"Error in batch {batch_count}: {type(e).__name__}: {e}")
                print("Creating zero tensors for this batch...")

                # Create fallback predictions with correct shapes
                try:
                    # Try to get at least the energy prediction
                    with torch.no_grad():
                        pred_energies, pred_dipoles, pred_nac = self.model(batch)
                    pred_forces = torch.zeros_like(batch.pos)
                    force_computation_errors += 1
                except Exception as e2:
                    print(f"Complete failure for batch {batch_count}: {e2}")
                    continue

            # Store predictions (detach to avoid memory issues)
            all_predictions["energies"].append(pred_energies.detach().cpu())
            all_predictions["dipoles"].append(pred_dipoles.detach().cpu())
            all_predictions["nac"].append(pred_nac.detach().cpu())
            all_predictions["forces"].append(pred_forces.detach().cpu())

            # Store targets
            batch_size = pred_energies.shape[0]
            num_states = pred_energies.shape[1]
            target_energies = batch.energies.view(batch_size, num_states).cpu()
            all_targets["energies"].append(target_energies)

            if hasattr(batch, "transition_dipoles"):
                all_targets["dipoles"].append(batch.transition_dipoles.cpu())
            else:
                all_targets["dipoles"].append(
                    torch.zeros_like(pred_dipoles.detach()).cpu()
                )

            if hasattr(batch, "nac"):
                # Process NAC targets
                num_coupling_pairs = pred_nac.shape[1]
                num_atoms = batch.nac.shape[1]
                target_nac_per_atom = batch.nac.view(
                    batch_size, num_coupling_pairs, num_atoms, 3
                )
                target_nac = target_nac_per_atom.sum(dim=2).cpu()
                all_targets["nac"].append(target_nac)
            else:
                all_targets["nac"].append(torch.zeros_like(pred_nac.detach()).cpu())

            # Process force targets
            num_molecules = batch.batch.max().item() + 1
            num_atoms_per_molecule = batch.forces.shape[1]
            total_atoms = num_molecules * num_atoms_per_molecule

            ground_state_indices = torch.arange(0, batch.forces.shape[0], num_states)
            target_forces_ground = batch.forces[ground_state_indices]
            target_forces_flat = target_forces_ground.reshape(total_atoms, 3).cpu()
            all_targets["forces"].append(target_forces_flat)

        # Set model back to eval mode
        self.model.eval()

        # Report force computation issues
        if force_computation_errors > 0:
            print(
                f"Warning: Force computation failed for {force_computation_errors}/{batch_count} batches"
            )

        # Concatenate all predictions and targets
        try:
            self.predictions = {
                key: torch.cat(values, dim=0) for key, values in all_predictions.items()
            }
            self.targets = {
                key: torch.cat(values, dim=0) for key, values in all_targets.items()
            }
        except Exception as e:
            print(f"Error concatenating results: {e}")
            print("Checking shapes...")
            for key in all_predictions.keys():
                shapes = [t.shape for t in all_predictions[key]]
                print(f"{key} prediction shapes: {shapes}")
            raise

        print(f"✓ Predictions completed on {batch_count} batches")
        print(f"  - Energies shape: {self.predictions['energies'].shape}")
        print(f"  - Forces shape: {self.predictions['forces'].shape}")
        print(f"  - Dipoles shape: {self.predictions['dipoles'].shape}")
        print(f"  - NAC shape: {self.predictions['nac'].shape}")

    def compute_metrics(self):
        """Compute comprehensive metrics for all properties"""
        print("Computing validation metrics...")

        self.metrics = {}

        # Energy metrics (per state)
        num_states = self.predictions["energies"].shape[1]
        state_names = ["Ground"] + [f"S{i}" for i in range(1, num_states)]

        self.metrics["energies"] = {}
        for i, state in enumerate(state_names):
            pred_state = self.predictions["energies"][:, i].numpy()
            target_state = self.targets["energies"][:, i].numpy()

            self.metrics["energies"][state] = {
                "mae": float(mean_absolute_error(target_state, pred_state)),
                "mse": float(mean_squared_error(target_state, pred_state)),
                "rmse": float(np.sqrt(mean_squared_error(target_state, pred_state))),
                "r2": float(r2_score(target_state, pred_state)),
                "mean_error": float(np.mean(pred_state - target_state)),
                "std_error": float(np.std(pred_state - target_state)),
            }

        # Force metrics
        pred_forces = self.predictions["forces"].numpy()
        target_forces = self.targets["forces"].numpy()

        # Ensure shapes match for forces
        min_size = min(pred_forces.shape[0], target_forces.shape[0])
        pred_forces = pred_forces[:min_size]
        target_forces = target_forces[:min_size]

        self.metrics["forces"] = {
            "mae": float(
                mean_absolute_error(target_forces.flatten(), pred_forces.flatten())
            ),
            "mse": float(
                mean_squared_error(target_forces.flatten(), pred_forces.flatten())
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(target_forces.flatten(), pred_forces.flatten())
                )
            ),
            "r2": float(r2_score(target_forces.flatten(), pred_forces.flatten())),
            "component_mae": [
                float(mean_absolute_error(target_forces[:, i], pred_forces[:, i]))
                for i in range(3)
            ],
            "component_rmse": [
                float(
                    np.sqrt(mean_squared_error(target_forces[:, i], pred_forces[:, i]))
                )
                for i in range(3)
            ],
        }

        # Dipole metrics
        pred_dipoles = self.predictions["dipoles"].numpy()
        target_dipoles = self.targets["dipoles"].numpy()

        self.metrics["dipoles"] = {
            "mae": float(
                mean_absolute_error(target_dipoles.flatten(), pred_dipoles.flatten())
            ),
            "mse": float(
                mean_squared_error(target_dipoles.flatten(), pred_dipoles.flatten())
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(target_dipoles.flatten(), pred_dipoles.flatten())
                )
            ),
            "r2": float(r2_score(target_dipoles.flatten(), pred_dipoles.flatten())),
            "component_mae": [
                float(mean_absolute_error(target_dipoles[:, i], pred_dipoles[:, i]))
                for i in range(3)
            ],
            "magnitude_mae": float(
                mean_absolute_error(
                    np.linalg.norm(target_dipoles, axis=1),
                    np.linalg.norm(pred_dipoles, axis=1),
                )
            ),
        }

        # NAC metrics
        pred_nac = self.predictions["nac"].numpy()
        target_nac = self.targets["nac"].numpy()

        num_coupling_pairs = pred_nac.shape[1]
        coupling_names = [f"S{i}-S{i + 1}" for i in range(num_coupling_pairs)]

        self.metrics["nac"] = {"overall": {}, "coupling_pairs": {}}

        # Overall NAC metrics
        self.metrics["nac"]["overall"] = {
            "mae": float(mean_absolute_error(target_nac.flatten(), pred_nac.flatten())),
            "mse": float(mean_squared_error(target_nac.flatten(), pred_nac.flatten())),
            "rmse": float(
                np.sqrt(mean_squared_error(target_nac.flatten(), pred_nac.flatten()))
            ),
            "r2": float(r2_score(target_nac.flatten(), pred_nac.flatten())),
        }

        # Per coupling pair metrics
        for i, coupling in enumerate(coupling_names):
            pred_coupling = pred_nac[:, i, :].flatten()
            target_coupling = target_nac[:, i, :].flatten()

            self.metrics["nac"]["coupling_pairs"][coupling] = {
                "mae": float(mean_absolute_error(target_coupling, pred_coupling)),
                "mse": float(mean_squared_error(target_coupling, pred_coupling)),
                "rmse": float(
                    np.sqrt(mean_squared_error(target_coupling, pred_coupling))
                ),
                "r2": float(r2_score(target_coupling, pred_coupling)),
            }

        print("✓ Metrics computed successfully")

    def generate_plots(self):
        """Generate comprehensive visualization plots"""
        print("Generating analysis plots...")

        # Set matplotlib style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Energy parity plots
        self._plot_energy_parity()

        # 2. Force analysis plots
        self._plot_force_analysis()

        # 3. Dipole analysis plots
        self._plot_dipole_analysis()

        # 4. NAC analysis plots
        self._plot_nac_analysis()

        # 5. Error distribution plots
        self._plot_error_distributions()

        # 6. Correlation matrix
        self._plot_correlation_matrix()

        # 7. Summary dashboard
        self._plot_summary_dashboard()

        print(f"✓ All plots saved to: {self.run_dir}")

    def _plot_energy_parity(self):
        """Generate energy parity plots for each electronic state"""
        num_states = self.predictions["energies"].shape[1]
        state_names = ["Ground"] + [f"S{i}" for i in range(1, num_states)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for i, (state, ax) in enumerate(zip(state_names, axes)):
            if i >= num_states:
                ax.axis("off")
                continue

            pred = self.predictions["energies"][:, i].numpy()
            target = self.targets["energies"][:, i].numpy()

            # Parity plot
            ax.scatter(target, pred, alpha=0.6, s=20)

            # Perfect prediction line
            min_val, max_val = (
                min(target.min(), pred.min()),
                max(target.max(), pred.max()),
            )
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect",
            )

            # Metrics
            r2 = self.metrics["energies"][state]["r2"]
            mae = self.metrics["energies"][state]["mae"]

            ax.set_xlabel("Target Energy (Hartree)")
            ax.set_ylabel("Predicted Energy (Hartree)")
            ax.set_title(f"{state} State Energy\nR² = {r2:.4f}, MAE = {mae:.4f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.run_dir / "energy_parity_plots.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_force_analysis(self):
        """Generate force analysis plots"""
        pred_forces = self.predictions["forces"].numpy()
        target_forces = self.targets["forces"].numpy()

        # Ensure shapes match
        min_size = min(pred_forces.shape[0], target_forces.shape[0])
        pred_forces = pred_forces[:min_size]
        target_forces = target_forces[:min_size]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Overall force parity
        ax = axes[0, 0]
        ax.scatter(target_forces.flatten(), pred_forces.flatten(), alpha=0.6, s=10)
        min_val, max_val = target_forces.min(), target_forces.max()
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        r2 = self.metrics["forces"]["r2"]
        mae = self.metrics["forces"]["mae"]
        ax.set_xlabel("Target Forces (Hartree/Bohr)")
        ax.set_ylabel("Predicted Forces (Hartree/Bohr)")
        ax.set_title(f"Force Parity Plot\nR² = {r2:.4f}, MAE = {mae:.4f}")
        ax.grid(True, alpha=0.3)

        # Component-wise MAE
        ax = axes[0, 1]
        components = ["X", "Y", "Z"]
        mae_components = self.metrics["forces"]["component_mae"]
        bars = ax.bar(components, mae_components)
        ax.set_ylabel("MAE (Hartree/Bohr)")
        ax.set_title("Force Component MAE")
        ax.grid(True, alpha=0.3)
        for bar, mae in zip(bars, mae_components):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{mae:.4f}",
                ha="center",
                va="bottom",
            )

        # Force magnitude correlation
        ax = axes[1, 0]
        target_mag = np.linalg.norm(target_forces, axis=1)
        pred_mag = np.linalg.norm(pred_forces, axis=1)
        ax.scatter(target_mag, pred_mag, alpha=0.6, s=20)
        min_val, max_val = (
            min(target_mag.min(), pred_mag.min()),
            max(target_mag.max(), pred_mag.max()),
        )
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        r2_mag = r2_score(target_mag, pred_mag)
        ax.set_xlabel("Target Force Magnitude")
        ax.set_ylabel("Predicted Force Magnitude")
        ax.set_title(f"Force Magnitude\nR² = {r2_mag:.4f}")
        ax.grid(True, alpha=0.3)

        # Error distribution
        ax = axes[1, 1]
        errors = (pred_forces - target_forces).flatten()
        ax.hist(errors, bins=50, alpha=0.7, density=True)
        ax.axvline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Force Error (Hartree/Bohr)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Force Error Distribution\nMean: {errors.mean():.4f}±{errors.std():.4f}"
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.run_dir / "force_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_dipole_analysis(self):
        """Generate dipole moment analysis plots"""
        pred_dipoles = self.predictions["dipoles"].numpy()
        target_dipoles = self.targets["dipoles"].numpy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Overall dipole parity
        ax = axes[0, 0]
        ax.scatter(target_dipoles.flatten(), pred_dipoles.flatten(), alpha=0.6, s=20)
        min_val, max_val = target_dipoles.min(), target_dipoles.max()
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        r2 = self.metrics["dipoles"]["r2"]
        mae = self.metrics["dipoles"]["mae"]
        ax.set_xlabel("Target Dipole (Debye)")
        ax.set_ylabel("Predicted Dipole (Debye)")
        ax.set_title(f"Dipole Parity Plot\nR² = {r2:.4f}, MAE = {mae:.4f}")
        ax.grid(True, alpha=0.3)

        # Component-wise analysis
        ax = axes[0, 1]
        components = ["X", "Y", "Z"]
        mae_components = self.metrics["dipoles"]["component_mae"]
        ax.bar(components, mae_components)
        ax.set_ylabel("MAE (Debye)")
        ax.set_title("Dipole Component MAE")
        ax.grid(True, alpha=0.3)

        # Dipole magnitude
        ax = axes[1, 0]
        target_mag = np.linalg.norm(target_dipoles, axis=1)
        pred_mag = np.linalg.norm(pred_dipoles, axis=1)
        ax.scatter(target_mag, pred_mag, alpha=0.6, s=20)
        min_val, max_val = 0, max(target_mag.max(), pred_mag.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        r2_mag = r2_score(target_mag, pred_mag)
        mae_mag = self.metrics["dipoles"]["magnitude_mae"]
        ax.set_xlabel("Target Dipole Magnitude (Debye)")
        ax.set_ylabel("Predicted Dipole Magnitude (Debye)")
        ax.set_title(f"Dipole Magnitude\nR² = {r2_mag:.4f}, MAE = {mae_mag:.4f}")
        ax.grid(True, alpha=0.3)

        # Vector angle analysis
        ax = axes[1, 1]
        # Calculate angles between predicted and target dipole vectors
        dot_products = np.sum(pred_dipoles * target_dipoles, axis=1)
        pred_norms = np.linalg.norm(pred_dipoles, axis=1)
        target_norms = np.linalg.norm(target_dipoles, axis=1)

        # Avoid division by zero
        valid_mask = (pred_norms > 1e-6) & (target_norms > 1e-6)

        if np.sum(valid_mask) > 0:
            cos_angles = dot_products[valid_mask] / (
                pred_norms[valid_mask] * target_norms[valid_mask]
            )
            cos_angles = np.clip(cos_angles, -1, 1)  # Numerical stability
            angles = np.arccos(cos_angles) * 180 / np.pi

            ax.hist(angles, bins=30, alpha=0.7, density=True)
            ax.set_xlabel("Angle between vectors (degrees)")
            ax.set_ylabel("Density")
            ax.set_title(
                f"Dipole Vector Angles\nMean: {angles.mean():.1f}°±{angles.std():.1f}°"
            )
        else:
            ax.set_xlabel("Angle between vectors (degrees)")
            ax.set_ylabel("Density")
            ax.set_title("Dipole Vector Angles\nNo valid vectors found")

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.run_dir / "dipole_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_nac_analysis(self):
        """Generate NAC analysis plots"""
        pred_nac = self.predictions["nac"].numpy()
        target_nac = self.targets["nac"].numpy()

        num_coupling_pairs = pred_nac.shape[1]
        coupling_names = [f"S{i}-S{i + 1}" for i in range(num_coupling_pairs)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Overall NAC parity
        ax = axes[0, 0]
        ax.scatter(target_nac.flatten(), pred_nac.flatten(), alpha=0.6, s=10)
        min_val, max_val = target_nac.min(), target_nac.max()
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        r2 = self.metrics["nac"]["overall"]["r2"]
        mae = self.metrics["nac"]["overall"]["mae"]
        ax.set_xlabel("Target NAC")
        ax.set_ylabel("Predicted NAC")
        ax.set_title(f"NAC Parity Plot\nR² = {r2:.4f}, MAE = {mae:.4f}")
        ax.grid(True, alpha=0.3)

        # Per coupling pair MAE
        ax = axes[0, 1]
        mae_values = [
            self.metrics["nac"]["coupling_pairs"][coupling]["mae"]
            for coupling in coupling_names
        ]
        ax.bar(coupling_names, mae_values)
        ax.set_ylabel("MAE")
        ax.set_title("NAC MAE per Coupling Pair")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)

        # NAC magnitude distribution
        ax = axes[1, 0]
        target_magnitudes = np.linalg.norm(target_nac, axis=2)
        pred_magnitudes = np.linalg.norm(pred_nac, axis=2)

        ax.scatter(
            target_magnitudes.flatten(), pred_magnitudes.flatten(), alpha=0.6, s=10
        )
        min_val, max_val = 0, max(target_magnitudes.max(), pred_magnitudes.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        ax.set_xlabel("Target NAC Magnitude")
        ax.set_ylabel("Predicted NAC Magnitude")
        ax.set_title("NAC Magnitude Correlation")
        ax.grid(True, alpha=0.3)

        # Error distribution
        ax = axes[1, 1]
        errors = (pred_nac - target_nac).flatten()
        ax.hist(errors, bins=50, alpha=0.7, density=True)
        ax.axvline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("NAC Error")
        ax.set_ylabel("Density")
        ax.set_title(
            f"NAC Error Distribution\nMean: {errors.mean():.4f}±{errors.std():.4f}"
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.run_dir / "nac_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_error_distributions(self):
        """Plot error distributions for all properties"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Energy errors (ground state)
        ax = axes[0, 0]
        pred_energy = self.predictions["energies"][:, 0]
        target_energy = self.targets["energies"][:, 0]

        # Convert to numpy if needed
        if hasattr(pred_energy, "numpy"):
            pred_energy = pred_energy.numpy()
        if hasattr(target_energy, "numpy"):
            target_energy = target_energy.numpy()

        energy_errors = pred_energy - target_energy
        ax.hist(energy_errors, bins=50, alpha=0.7, density=True, color="blue")
        ax.axvline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Energy Error (Hartree)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Ground State Energy Errors\nMean: {energy_errors.mean():.6f}±{energy_errors.std():.6f}"
        )
        ax.grid(True, alpha=0.3)

        # Force errors
        ax = axes[0, 1]
        pred_forces = self.predictions["forces"]
        target_forces = self.targets["forces"]

        # Convert to numpy if needed
        if hasattr(pred_forces, "numpy"):
            pred_forces = pred_forces.numpy()
        if hasattr(target_forces, "numpy"):
            target_forces = target_forces.numpy()

        min_size = min(pred_forces.shape[0], target_forces.shape[0])
        force_errors = (pred_forces[:min_size] - target_forces[:min_size]).flatten()
        ax.hist(force_errors, bins=50, alpha=0.7, density=True, color="green")
        ax.axvline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Force Error (Hartree/Bohr)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Force Errors\nMean: {force_errors.mean():.6f}±{force_errors.std():.6f}"
        )
        ax.grid(True, alpha=0.3)

        # Dipole errors
        ax = axes[1, 0]
        pred_dipoles = self.predictions["dipoles"]
        target_dipoles = self.targets["dipoles"]

        # Convert to numpy if needed
        if hasattr(pred_dipoles, "numpy"):
            pred_dipoles = pred_dipoles.numpy()
        if hasattr(target_dipoles, "numpy"):
            target_dipoles = target_dipoles.numpy()

        dipole_errors = (pred_dipoles - target_dipoles).flatten()
        ax.hist(dipole_errors, bins=50, alpha=0.7, density=True, color="orange")
        ax.axvline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Dipole Error (Debye)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Dipole Errors\nMean: {dipole_errors.mean():.6f}±{dipole_errors.std():.6f}"
        )
        ax.grid(True, alpha=0.3)

        # NAC errors
        ax = axes[1, 1]
        pred_nac = self.predictions["nac"]
        target_nac = self.targets["nac"]

        # Convert to numpy if needed
        if hasattr(pred_nac, "numpy"):
            pred_nac = pred_nac.numpy()
        if hasattr(target_nac, "numpy"):
            target_nac = target_nac.numpy()

        nac_errors = (pred_nac - target_nac).flatten()
        ax.hist(nac_errors, bins=50, alpha=0.7, density=True, color="purple")
        ax.axvline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("NAC Error")
        ax.set_ylabel("Density")
        ax.set_title(
            f"NAC Errors\nMean: {nac_errors.mean():.6f}±{nac_errors.std():.6f}"
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.run_dir / "error_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_correlation_matrix(self):
        """Plot correlation matrix between different properties"""
        # Prepare data for correlation analysis
        data_dict = {}

        # Add energy errors for each state
        num_states = self.predictions["energies"].shape[1]
        state_names = ["Ground"] + [f"S{i}" for i in range(1, num_states)]

        for i, state in enumerate(state_names):
            pred_energy = self.predictions["energies"][:, i]
            target_energy = self.targets["energies"][:, i]

            # Convert to numpy if needed
            if hasattr(pred_energy, "numpy"):
                pred_energy = pred_energy.numpy()
            if hasattr(target_energy, "numpy"):
                target_energy = target_energy.numpy()

            energy_errors = np.abs(pred_energy - target_energy)
            data_dict[f"{state}_Energy_Error"] = energy_errors

        # Add force magnitude errors
        pred_forces = self.predictions["forces"]
        target_forces = self.targets["forces"]

        # Convert to numpy if needed
        if hasattr(pred_forces, "numpy"):
            pred_forces = pred_forces.numpy()
        if hasattr(target_forces, "numpy"):
            target_forces = target_forces.numpy()

        min_size = min(pred_forces.shape[0], target_forces.shape[0])

        force_mag_errors = np.abs(
            np.linalg.norm(pred_forces[:min_size], axis=1)
            - np.linalg.norm(target_forces[:min_size], axis=1)
        )
        data_dict["Force_Magnitude_Error"] = force_mag_errors[
            : len(data_dict[f"{state_names[0]}_Energy_Error"])
        ]

        # Add dipole magnitude errors
        pred_dipoles = self.predictions["dipoles"]
        target_dipoles = self.targets["dipoles"]

        # Convert to numpy if needed
        if hasattr(pred_dipoles, "numpy"):
            pred_dipoles = pred_dipoles.numpy()
        if hasattr(target_dipoles, "numpy"):
            target_dipoles = target_dipoles.numpy()

        dipole_mag_errors = np.abs(
            np.linalg.norm(pred_dipoles, axis=1)
            - np.linalg.norm(target_dipoles, axis=1)
        )
        data_dict["Dipole_Magnitude_Error"] = dipole_mag_errors

        # Add NAC magnitude errors
        pred_nac = self.predictions["nac"]
        target_nac = self.targets["nac"]

        # Convert to numpy if needed
        if hasattr(pred_nac, "numpy"):
            pred_nac = pred_nac.numpy()
        if hasattr(target_nac, "numpy"):
            target_nac = target_nac.numpy()

        nac_mag_errors = np.abs(
            np.linalg.norm(pred_nac, axis=(1, 2))
            - np.linalg.norm(target_nac, axis=(1, 2))
        )
        data_dict["NAC_Magnitude_Error"] = nac_mag_errors

        # Create DataFrame and correlation matrix
        df = pd.DataFrame(data_dict)
        correlation_matrix = df.corr()

        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(correlation_matrix.corr())
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
            cbar_kws={"shrink": 0.8},
            mask=mask,
        )
        plt.title("Property Error Correlation Matrix")
        plt.tight_layout()
        plt.savefig(
            self.run_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_summary_dashboard(self):
        """Create a summary dashboard with key metrics"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(
            "MANA Model Validation Summary Dashboard", fontsize=16, fontweight="bold"
        )

        # Energy R² scores
        ax1 = fig.add_subplot(gs[0, 0])
        num_states = self.predictions["energies"].shape[1]
        state_names = ["Ground"] + [f"S{i}" for i in range(1, num_states)]
        r2_values = [self.metrics["energies"][state]["r2"] for state in state_names]
        bars = ax1.bar(state_names, r2_values, color="skyblue")
        ax1.set_ylabel("R² Score")
        ax1.set_title("Energy R² by State")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Force metrics
        ax2 = fig.add_subplot(gs[0, 1])
        force_metrics = ["MAE", "RMSE", "R²"]
        force_values = [
            self.metrics["forces"]["mae"],
            self.metrics["forces"]["rmse"],
            self.metrics["forces"]["r2"],
        ]
        bars = ax2.bar(force_metrics, force_values, color="lightgreen")
        ax2.set_ylabel("Metric Value")
        ax2.set_title("Force Metrics")
        ax2.grid(True, alpha=0.3)

        # Dipole metrics
        ax3 = fig.add_subplot(gs[0, 2])
        dipole_metrics = ["MAE", "RMSE", "R²"]
        dipole_values = [
            self.metrics["dipoles"]["mae"],
            self.metrics["dipoles"]["rmse"],
            self.metrics["dipoles"]["r2"],
        ]
        bars = ax3.bar(dipole_metrics, dipole_values, color="lightcoral")
        ax3.set_ylabel("Metric Value")
        ax3.set_title("Dipole Metrics")
        ax3.grid(True, alpha=0.3)

        # NAC metrics
        ax4 = fig.add_subplot(gs[0, 3])
        nac_metrics = ["MAE", "RMSE", "R²"]
        nac_values = [
            self.metrics["nac"]["overall"]["mae"],
            self.metrics["nac"]["overall"]["rmse"],
            self.metrics["nac"]["overall"]["r2"],
        ]
        bars = ax4.bar(nac_metrics, nac_values, color="plum")
        ax4.set_ylabel("Metric Value")
        ax4.set_title("NAC Metrics")
        ax4.grid(True, alpha=0.3)

        # Energy MAE comparison
        ax5 = fig.add_subplot(gs[1, :2])
        mae_values = [self.metrics["energies"][state]["mae"] for state in state_names]
        bars = ax5.bar(state_names, mae_values, color="lightblue")
        ax5.set_ylabel("MAE (Hartree)")
        ax5.set_title("Energy MAE by Electronic State")
        ax5.grid(True, alpha=0.3)
        for bar, mae in zip(bars, mae_values):
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0001,
                f"{mae:.5f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Overall performance radar chart
        ax6 = fig.add_subplot(gs[1, 2:], projection="polar")

        # Normalize metrics to 0-1 scale for radar chart
        categories = ["Energy\n(Ground)", "Forces", "Dipoles", "NAC"]
        values = [
            self.metrics["energies"]["Ground"]["r2"],
            min(self.metrics["forces"]["r2"], 1.0),
            min(self.metrics["dipoles"]["r2"], 1.0),
            min(self.metrics["nac"]["overall"]["r2"], 1.0),
        ]

        # Complete the circle
        values += [values[0]]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += [angles[0]]

        ax6.plot(angles, values, "o-", linewidth=2, color="red")
        ax6.fill(angles, values, alpha=0.25, color="red")
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title("Overall Performance (R² Scores)", y=1.08)
        ax6.grid(True)

        # Summary statistics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis("tight")
        ax7.axis("off")

        # Create summary table
        summary_data = [
            ["Property", "MAE", "RMSE", "R²", "Best State/Component"],
            [
                "Energy (Ground)",
                f"{self.metrics['energies']['Ground']['mae']:.5f}",
                f"{self.metrics['energies']['Ground']['rmse']:.5f}",
                f"{self.metrics['energies']['Ground']['r2']:.4f}",
                "Ground State",
            ],
            [
                "Forces",
                f"{self.metrics['forces']['mae']:.5f}",
                f"{self.metrics['forces']['rmse']:.5f}",
                f"{self.metrics['forces']['r2']:.4f}",
                f"Component {np.argmin(self.metrics['forces']['component_mae'])}",
            ],
            [
                "Dipoles",
                f"{self.metrics['dipoles']['mae']:.5f}",
                f"{self.metrics['dipoles']['rmse']:.5f}",
                f"{self.metrics['dipoles']['r2']:.4f}",
                f"Component {np.argmin(self.metrics['dipoles']['component_mae'])}",
            ],
            [
                "NAC",
                f"{self.metrics['nac']['overall']['mae']:.5f}",
                f"{self.metrics['nac']['overall']['rmse']:.5f}",
                f"{self.metrics['nac']['overall']['r2']:.4f}",
                "Overall",
            ],
        ]

        table = ax7.table(cellText=summary_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style the header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax7.set_title("Summary Metrics Table", fontsize=14, pad=20)

        plt.savefig(
            self.run_dir / "validation_dashboard.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def save_results(self):
        """Save detailed results and metrics to files"""
        print("Saving validation results...")

        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, "item"):  # NumPy scalar
                return obj.item()
            elif hasattr(obj, "tolist"):  # NumPy array
                return obj.tolist()
            else:
                return obj

        # Save metrics as JSON
        metrics_path = self.run_dir / "validation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(convert_numpy_types(self.metrics), f, indent=2)

        # Save predictions and targets as numpy files
        def to_numpy(tensor):
            """Convert tensor to numpy array safely"""
            if hasattr(tensor, "numpy"):
                return tensor.numpy()
            else:
                return tensor

        np.savez(
            self.run_dir / "predictions.npz",
            energies=to_numpy(self.predictions["energies"]),
            forces=to_numpy(self.predictions["forces"]),
            dipoles=to_numpy(self.predictions["dipoles"]),
            nac=to_numpy(self.predictions["nac"]),
        )

        np.savez(
            self.run_dir / "targets.npz",
            energies=to_numpy(self.targets["energies"]),
            forces=to_numpy(self.targets["forces"]),
            dipoles=to_numpy(self.targets["dipoles"]),
            nac=to_numpy(self.targets["nac"]),
        )

        # Generate comprehensive text report
        self._generate_text_report()

        print(f"✓ Results saved to: {self.run_dir}")

    def _generate_text_report(self):
        """Generate comprehensive text report"""
        report_path = self.run_dir / "validation_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MANA MODEL VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                "Validation Date: {}\n".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Dataset Path: {self.dataset_path}\n")
            f.write(f"Test Samples: {len(self.test_loader.dataset)}\n")
            f.write(f"Device: {self.device}\n\n")

            # Energy metrics
            f.write("ENERGY VALIDATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            num_states = self.predictions["energies"].shape[1]
            state_names = ["Ground"] + [f"S{i}" for i in range(1, num_states)]

            for state in state_names:
                metrics = self.metrics["energies"][state]
                f.write(f"{state} State:\n")
                f.write(f"  MAE:  {metrics['mae']:.6f} Hartree\n")
                f.write(f"  RMSE: {metrics['rmse']:.6f} Hartree\n")
                f.write(f"  R²:   {metrics['r2']:.6f}\n")
                f.write(
                    f"  Mean Error: {metrics['mean_error']:.6f} ± {metrics['std_error']:.6f}\n\n"
                )

            # Force metrics
            f.write("FORCE VALIDATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            force_metrics = self.metrics["forces"]
            f.write(f"Overall:\n")
            f.write(f"  MAE:  {force_metrics['mae']:.6f} Hartree/Bohr\n")
            f.write(f"  RMSE: {force_metrics['rmse']:.6f} Hartree/Bohr\n")
            f.write(f"  R²:   {force_metrics['r2']:.6f}\n\n")

            f.write("Component-wise MAE:\n")
            for i, component in enumerate(["X", "Y", "Z"]):
                f.write(
                    f"  {component}: {force_metrics['component_mae'][i]:.6f} Hartree/Bohr\n"
                )
            f.write("\n")

            # Dipole metrics
            f.write("DIPOLE VALIDATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            dipole_metrics = self.metrics["dipoles"]
            f.write("Overall:\n")
            f.write(f"  MAE:  {dipole_metrics['mae']:.6f} Debye\n")
            f.write(f"  RMSE: {dipole_metrics['rmse']:.6f} Debye\n")
            f.write(f"  R²:   {dipole_metrics['r2']:.6f}\n")
            f.write(f"  Magnitude MAE: {dipole_metrics['magnitude_mae']:.6f} Debye\n\n")

            f.write("Component-wise MAE:\n")
            for i, component in enumerate(["X", "Y", "Z"]):
                f.write(
                    f"  {component}: {dipole_metrics['component_mae'][i]:.6f} Debye\n"
                )
            f.write("\n")

            # NAC metrics
            f.write("NON-ADIABATIC COUPLING VALIDATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            nac_metrics = self.metrics["nac"]
            f.write("Overall:\n")
            f.write(f"  MAE:  {nac_metrics['overall']['mae']:.6f}\n")
            f.write(f"  RMSE: {nac_metrics['overall']['rmse']:.6f}\n")
            f.write(f"  R²:   {nac_metrics['overall']['r2']:.6f}\n\n")

            f.write("Per Coupling Pair:\n")
            for coupling, metrics in nac_metrics["coupling_pairs"].items():
                f.write(f"  {coupling}:\n")
                f.write(f"    MAE:  {metrics['mae']:.6f}\n")
                f.write(f"    RMSE: {metrics['rmse']:.6f}\n")
                f.write(f"    R²:   {metrics['r2']:.6f}\n")
            f.write("\n")

            # Model information
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 40 + "\n")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(
                f"Atom Types: {self.dataset.num_atom_types} ({self.dataset.unique_atoms})\n"
            )
            f.write(f"Electronic States: {num_states}\n\n")

            # Summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Best Energy R²: {max(self.metrics['energies'][state]['r2'] for state in state_names):.4f}\n"
            )
            f.write(f"Force R²: {self.metrics['forces']['r2']:.4f}\n")
            f.write(f"Dipole R²: {self.metrics['dipoles']['r2']:.4f}\n")
            f.write(f"NAC R²: {self.metrics['nac']['overall']['r2']:.4f}\n")

            # Calculate overall performance score
            scores = [
                self.metrics["energies"]["Ground"]["r2"],
                self.metrics["forces"]["r2"],
                self.metrics["dipoles"]["r2"],
                self.metrics["nac"]["overall"]["r2"],
            ]
            overall_score = np.mean(
                [max(0, score) for score in scores]
            )  # Handle negative R²
            f.write("\nOverall Performance Score: {:.4f}\n".format(overall_score))

            f.write("\n" + "=" * 80 + "\n")

    def run_validation(self, batch_size=32, num_workers=0):
        """Run complete validation pipeline"""
        print("=" * 80)
        print("MANA MODEL VALIDATION ENGINE")
        print("=" * 80)

        # Load dataset and model
        self.load_dataset(batch_size, num_workers)
        self.load_model()

        # Run predictions
        self.run_predictions()

        # Compute metrics
        self.compute_metrics()

        # Generate plots
        self.generate_plots()

        # Save results
        self.save_results()

        print("\n" + "=" * 80)
        print("VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved to: {self.run_dir}")
        print(f"Key metrics:")
        print(
            f"  - Ground State Energy R²: {self.metrics['energies']['Ground']['r2']:.4f}"
        )
        print(f"  - Force R²: {self.metrics['forces']['r2']:.4f}")
        print(f"  - Dipole R²: {self.metrics['dipoles']['r2']:.4f}")
        print(f"  - NAC R²: {self.metrics['nac']['overall']['r2']:.4f}")

        # Calculate and display overall performance
        scores = [
            self.metrics["energies"]["Ground"]["r2"],
            self.metrics["forces"]["r2"],
            self.metrics["dipoles"]["r2"],
            self.metrics["nac"]["overall"]["r2"],
        ]
        overall_score = np.mean([max(0, score) for score in scores])
        print(f"  - Overall Performance Score: {overall_score:.4f}")
        print("=" * 80)


def find_latest_model():
    """Find the most recently trained model"""
    models_dir = Path("models")
    if not models_dir.exists():
        return None

    # Look for timestamped directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        return None

    # Sort by creation time and get the most recent
    latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)

    # Look for .pth files in the directory
    pth_files = list(latest_dir.glob("*.pth"))
    if pth_files:
        # Prefer best_model.pth if it exists
        best_model = latest_dir / "best_model.pth"
        if best_model.exists():
            return str(best_model)
        else:
            return str(pth_files[0])

    return None


def check_conda_environment():
    """Check if running in the correct conda environment"""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env.lower() != "mana":
        print(
            f"WARNING: Not running in 'mana' conda environment (current: {conda_env})"
        )
        print("Please run: conda activate mana")
        return False
    else:
        print(f"✓ Running in correct conda environment: {conda_env}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MANA Model Validation Engine")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to trained model (.pth file). If not provided, will use the latest model.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="dataset_construction/qm_results.h5",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--analysis-dir",
        "-a",
        type=str,
        default="analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for validation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run validation on",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers for data loading"
    )

    args = parser.parse_args()

    # Check conda environment
    check_conda_environment()

    # Find model path
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print(
                "❌ No trained model found. Please train a model first or specify --model path."
            )
            sys.exit(1)
        else:
            print(f"Using latest model: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Create validation engine and run
    try:
        engine = ValidationEngine(
            model_path=model_path,
            dataset_path=args.dataset,
            analysis_dir=args.analysis_dir,
            device=args.device,
        )

        engine.run_validation(batch_size=args.batch_size, num_workers=args.num_workers)

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
