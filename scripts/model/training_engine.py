import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter


class TrainingEngine:
    def __init__(self, model, device, train_loader, val_loader, hyperparams, directory):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Core logic
        self.model = model.to(device)
        self.device = device
        self.directory = directory + "/" + timestamp
        os.makedirs(self.directory, exist_ok=True)
        self.writer = SummaryWriter(self.directory + "/verbose")

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Hyperparameters
        self.lr = hyperparams["learning_rate"]
        self.batch_size = hyperparams["batch_size"]
        self.max_epochs = hyperparams["max_epochs"]
        self.patience = hyperparams["early_stopping_patience"]
        self.gradient_clip_norm = hyperparams["gradient_clip_norm"]
        self.energy_weight = hyperparams["loss_weights"]["energy"]
        self.force_weight = hyperparams["loss_weights"]["force"]
        self.dipole_weight = hyperparams["loss_weights"]["dipole"]

        # NAC weight management
        self.original_nac_weight = hyperparams["loss_weights"]["nac"]
        self.current_nac_weight = self.original_nac_weight

        # Adaptive NAC parameters
        self.adaptive_nac_config = hyperparams.get("adaptive_nac", {})
        self.adaptive_enabled = self.adaptive_nac_config.get("enabled", True)
        self.min_nac_weight = self.adaptive_nac_config.get("min_weight", 0.0001)
        self.reduction_factor = self.adaptive_nac_config.get("reduction_factor", 0.8)
        self.nac_patience = self.adaptive_nac_config.get("patience", 5)
        self.spike_threshold = self.adaptive_nac_config.get("spike_threshold", 2.0)
        self.restore_threshold = self.adaptive_nac_config.get("restore_threshold", 0.9)

        # Adaptive tracking
        self.nac_patience_counter = 0
        self.nac_reductions = 0
        self.nac_restorations = 0
        self.nac_weight_history = []
        self.recent_val_nac_losses = []
        self.recent_train_nac_losses = []

        # Optimizer with weight decay for regularization
        weight_decay = hyperparams.get("weight_decay", 0.0)
        self.optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=weight_decay)

        # Loss tracking for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_losses_E = []
        self.train_losses_F = []
        self.train_losses_NAC = []
        self.train_losses_mu = []
        self.val_losses_E = []
        self.val_losses_F = []
        self.val_losses_NAC = []
        self.val_losses_mu = []
        self.hyperparams = hyperparams

        # Store initial model parameters for debugging
        self.initial_params = {}
        for name, param in self.model.named_parameters():
            self.initial_params[name] = param.data.clone()

    def update_nac_weight(self, train_nac, val_nac, epoch):
        """Adaptively update NAC weight based on training dynamics"""
        if not self.adaptive_enabled:
            return False

        # Store recent losses for trend analysis
        self.recent_train_nac_losses.append(train_nac)
        self.recent_val_nac_losses.append(val_nac)

        # Keep only recent history (last 10 epochs)
        if len(self.recent_train_nac_losses) > 10:
            self.recent_train_nac_losses.pop(0)
            self.recent_val_nac_losses.pop(0)

        # Record current weight
        self.nac_weight_history.append(
            {
                "epoch": epoch,
                "weight": self.current_nac_weight,
                "train_nac": train_nac,
                "val_nac": val_nac,
                "ratio": val_nac / train_nac if train_nac > 0 else float("inf"),
            }
        )

        # Check for spikes (validation >> training)
        if train_nac > 0 and val_nac > train_nac * self.spike_threshold:
            self.nac_patience_counter += 1

            if self.nac_patience_counter >= self.nac_patience:
                if self.current_nac_weight > self.min_nac_weight:
                    old_weight = self.current_nac_weight
                    self.current_nac_weight = max(
                        self.current_nac_weight * self.reduction_factor,
                        self.min_nac_weight,
                    )
                    self.nac_reductions += 1
                    self.nac_patience_counter = 0

                    print(
                        f"  🔽 NAC weight reduced: {old_weight:.6f} → {self.current_nac_weight:.6f}"
                    )
                    print(
                        f"     Reason: val_nac ({val_nac:.4f}) > {self.spike_threshold}x train_nac ({train_nac:.4f})"
                    )
                    return True

        # Check for stabilization (validation stable relative to training)
        elif train_nac > 0 and val_nac < train_nac * self.restore_threshold:
            self.nac_patience_counter = max(0, self.nac_patience_counter - 1)

            # Consider restoration if we've been stable for a while
            if (
                len(self.recent_val_nac_losses) >= 5
                and self.current_nac_weight < self.original_nac_weight * 0.5
            ):
                # Check if recent validation NAC losses are stable
                recent_val_ratios = [
                    v / t
                    for v, t in zip(
                        self.recent_val_nac_losses[-5:],
                        self.recent_train_nac_losses[-5:],
                    )
                    if t > 0
                ]
                if (
                    recent_val_ratios
                    and max(recent_val_ratios) < self.restore_threshold
                ):
                    old_weight = self.current_nac_weight
                    self.current_nac_weight = min(
                        self.current_nac_weight / self.reduction_factor,
                        self.original_nac_weight,
                    )
                    self.nac_restorations += 1

                    print(
                        f"  🔼 NAC weight restored: {old_weight:.6f} → {self.current_nac_weight:.6f}"
                    )
                    print(f"     Reason: Stable validation NAC for 5 epochs")
                    return True
        else:
            self.nac_patience_counter = max(0, self.nac_patience_counter - 1)

        return False

    @property
    def nac_weight(self):
        """Current NAC weight (for compatibility)"""
        return self.current_nac_weight

    def loss_function(self, batch, energies, mu, nac):
        # Reshape batch.energies to match predicted energies shape
        batch_size = energies.shape[0]
        num_states = energies.shape[1]

        try:
            target_energies = batch.energies.view(batch_size, num_states)
        except Exception:
            # Fallback for shape issues
            target_energies = batch.energies.reshape(batch_size, num_states)

        loss_E = F.mse_loss(energies, target_energies)

        loss_mu = 0.0
        if hasattr(batch, "transition_dipoles"):
            loss_mu = F.mse_loss(mu, batch.transition_dipoles)

        loss_nac = 0.0
        if hasattr(batch, "nac"):
            try:
                num_coupling_pairs = nac.shape[1]
                num_atoms = batch.nac.shape[1]

                # Reshape to (batch_size, num_coupling_pairs, num_atoms, 3)
                target_nac_per_atom = batch.nac.view(
                    batch_size, num_coupling_pairs, num_atoms, 3
                )

                # Sum over atoms to get molecule-level NAC
                target_nac = target_nac_per_atom.sum(dim=2)

                # Conservative NAC loss with NaN safety
                if torch.isfinite(nac).all() and torch.isfinite(target_nac).all():
                    loss_nac = F.mse_loss(nac, target_nac)

                    # Clamp to prevent extreme values
                    loss_nac = torch.clamp(loss_nac, max=100.0)
                else:
                    loss_nac = torch.tensor(0.0, device=energies.device)

            except Exception as _:
                loss_nac = torch.tensor(0.0, device=energies.device)

        # Forces from ground state only - FIXED VERSION
        loss_F = torch.tensor(0.0, device=energies.device)

        # Ensure positions require gradients for force computation
        if batch.pos.grad is not None:
            batch.pos.grad.zero_()

        batch.pos.requires_grad_(True)

        try:
            # Use ground state energy (first column) for force computation
            ground_energies = energies[:, 0]  # Shape: (batch_size,)
            total_energy = ground_energies.sum()  # Scalar

            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=total_energy,
                inputs=batch.pos,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            if gradients[0] is not None:
                predicted_forces = -gradients[0]  # Forces are negative gradients

                # Handle PyTorch Geometric batching structure
                # batch.forces shape: (batch_size * num_states, num_atoms_per_molecule, 3)
                # predicted_forces shape: (total_atoms_in_batch, 3)

                # Determine the number of molecules in the batch
                num_molecules = batch.batch.max().item() + 1
                num_atoms_per_molecule = batch.forces.shape[1]
                total_atoms = num_molecules * num_atoms_per_molecule

                # Extract ground state forces only (every num_states-th entry starting from 0)
                ground_state_indices = torch.arange(
                    0, batch.forces.shape[0], num_states
                )
                target_forces_ground = batch.forces[
                    ground_state_indices
                ]  # Shape: (num_molecules, num_atoms_per_molecule, 3)

                # Flatten target forces to match predicted forces structure
                target_forces_flat = target_forces_ground.reshape(total_atoms, 3)

                # Ensure shapes match
                if predicted_forces.shape[0] == total_atoms:
                    loss_F = F.mse_loss(predicted_forces, target_forces_flat)
                else:
                    # If shapes don't match, try to fix by trimming/padding
                    min_atoms = min(
                        predicted_forces.shape[0], target_forces_flat.shape[0]
                    )
                    if min_atoms > 0:
                        loss_F = F.mse_loss(
                            predicted_forces[:min_atoms], target_forces_flat[:min_atoms]
                        )
                    else:
                        loss_F = torch.tensor(0.0, device=energies.device)
            else:
                loss_F = torch.tensor(0.0, device=energies.device)

        except Exception as _:
            # If force computation fails silently set to zero
            loss_F = torch.tensor(0.0, device=energies.device)

        # Compute total loss using current adaptive NAC weight
        loss = (
            self.energy_weight * loss_E
            + self.dipole_weight * loss_mu
            + self.force_weight * loss_F
            + self.current_nac_weight * loss_nac
        )

        # Simple NaN check on final loss only
        if not torch.isfinite(loss):
            print("Warning: Non-finite total loss, using energy fallback")
            loss = self.energy_weight * loss_E

        return loss, loss_E, loss_F, loss_mu, loss_nac

    def training_step(self, batch):
        """
        Single training step for the MANA model.
        """
        # Ensure positions require gradients
        batch.pos.requires_grad_(True)

        self.optimizer.zero_grad()

        energies, mu, nac = self.model(batch)

        loss, loss_E, loss_F, loss_mu, loss_nac = self.loss_function(
            batch, energies, mu, nac
        )

        loss.backward()

        # Simple gradient clipping without excessive monitoring

        # Apply standard gradient clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )

        self.optimizer.step()

        return loss, loss_E, loss_F, loss_mu, loss_nac

    def validation_step(self, batch):
        """
        Single validation step for the MANA model.
        """
        # Ensure positions require gradients for force computation in validation too
        batch.pos.requires_grad_(True)

        # Forward pass with gradients enabled for force computation
        energies, mu, nac = self.model(batch)

        loss, loss_E, loss_F, loss_mu, loss_nac = self.loss_function(
            batch, energies, mu, nac
        )

        return loss, loss_E, loss_F, loss_mu, loss_nac

    def train_epoch(self, index):
        """
        Trains one full epoch
        """
        total_loss = 0.0
        total_loss_E = 0.0
        total_loss_F = 0.0
        total_loss_mu = 0.0
        total_loss_nac = 0.0
        batch_count = 0

        for i, data in enumerate(self.train_loader):
            loss, loss_E, loss_F, loss_mu, loss_nac = self.training_step(data)
            total_loss += loss.item()
            total_loss_E += (
                loss_E.item() if isinstance(loss_E, torch.Tensor) else loss_E
            )
            total_loss_F += (
                loss_F.item() if isinstance(loss_F, torch.Tensor) else loss_F
            )
            total_loss_mu += (
                loss_mu if isinstance(loss_mu, (int, float)) else loss_mu.item()
            )
            total_loss_nac += (
                loss_nac.item() if isinstance(loss_nac, torch.Tensor) else loss_nac
            )
            batch_count += 1

        # Return average losses for the entire epoch
        epoch_avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        epoch_avg_loss_E = total_loss_E / batch_count if batch_count > 0 else 0.0
        epoch_avg_loss_F = total_loss_F / batch_count if batch_count > 0 else 0.0
        epoch_avg_loss_mu = total_loss_mu / batch_count if batch_count > 0 else 0.0
        epoch_avg_loss_nac = total_loss_nac / batch_count if batch_count > 0 else 0.0

        return (
            epoch_avg_loss,
            epoch_avg_loss_E,
            epoch_avg_loss_F,
            epoch_avg_loss_mu,
            epoch_avg_loss_nac,
        )

    def validate_epoch(self):
        """
        Validates one full epoch
        """
        total_vloss = 0.0
        total_vloss_E = 0.0
        total_vloss_F = 0.0
        total_vloss_mu = 0.0
        total_vloss_nac = 0.0
        batch_count = 0

        self.model.eval()
        for i, vdata in enumerate(self.val_loader):
            vloss, vloss_E, vloss_F, vloss_mu, vloss_nac = self.validation_step(vdata)
            total_vloss += vloss.item()
            total_vloss_E += (
                vloss_E.item() if isinstance(vloss_E, torch.Tensor) else vloss_E
            )
            total_vloss_F += (
                vloss_F.item() if isinstance(vloss_F, torch.Tensor) else vloss_F
            )
            total_vloss_mu += (
                vloss_mu if isinstance(vloss_mu, (int, float)) else vloss_mu.item()
            )
            total_vloss_nac += (
                vloss_nac.item() if isinstance(vloss_nac, torch.Tensor) else vloss_nac
            )
            batch_count += 1

        # Return average losses for the entire validation epoch
        avg_vloss = total_vloss / batch_count if batch_count > 0 else 0.0
        avg_vloss_E = total_vloss_E / batch_count if batch_count > 0 else 0.0
        avg_vloss_F = total_vloss_F / batch_count if batch_count > 0 else 0.0
        avg_vloss_mu = total_vloss_mu / batch_count if batch_count > 0 else 0.0
        avg_vloss_nac = total_vloss_nac / batch_count if batch_count > 0 else 0.0

        return avg_vloss, avg_vloss_E, avg_vloss_F, avg_vloss_mu, avg_vloss_nac

    def plot_losses(self):
        """
        Create and save loss plots using matplotlib with adaptive NAC weight indicators
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_losses) + 1)

        # Total loss
        ax1.plot(epochs, self.train_losses, "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, self.val_losses, "r-", label="Validation Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Total Loss")
        ax1.set_title("Total Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Energy loss
        ax2.plot(
            epochs, self.train_losses_E, "b-", label="Training Energy Loss", linewidth=2
        )
        ax2.plot(
            epochs, self.val_losses_E, "r-", label="Validation Energy Loss", linewidth=2
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Energy Loss")
        ax2.set_title("Energy Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Force loss
        ax3.plot(
            epochs, self.train_losses_F, "b-", label="Training Force Loss", linewidth=2
        )
        ax3.plot(
            epochs, self.val_losses_F, "r-", label="Validation Force Loss", linewidth=2
        )
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Force Loss")
        ax3.set_title("Force Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # NAC loss with adaptive weight changes
        ax4.plot(
            epochs, self.train_losses_NAC, "b-", label="Training NAC Loss", linewidth=2
        )
        ax4.plot(
            epochs, self.val_losses_NAC, "r-", label="Validation NAC Loss", linewidth=2
        )

        # Mark NAC weight changes
        for entry in self.nac_weight_history:
            if entry["epoch"] > 1:  # Skip first epoch
                prev_weight = (
                    self.nac_weight_history[entry["epoch"] - 2]["weight"]
                    if entry["epoch"] > 1
                    else self.original_nac_weight
                )
                if abs(entry["weight"] - prev_weight) > 1e-6:  # Weight changed
                    ax4.axvline(
                        x=entry["epoch"], color="orange", linestyle="--", alpha=0.7
                    )

        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("NAC Loss")
        ax4.set_title("Non-Adiabatic Coupling Loss (Adaptive Weights)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        plot_path = f"{self.directory}/training_losses.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Loss plots saved to: {plot_path}")

        # Also display the plot
        plt.show()

    def save_training_summary(self, final_epoch, best_vloss, reason="completed"):
        """
        Save comprehensive training summary with adaptive NAC information
        """
        summary_path = f"{self.directory}/training_summary.txt"

        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ADAPTIVE MANA MODEL TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Training configuration
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Learning Rate: {self.lr}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Max Epochs: {self.max_epochs}\n")
            f.write(f"Early Stopping Patience: {self.patience}\n")
            f.write(f"Gradient Clip Norm: {self.gradient_clip_norm}\n")
            f.write(f"Weight Decay: {self.hyperparams.get('weight_decay', 0.0)}\n\n")

            # Loss weights (adaptive)
            f.write("LOSS WEIGHTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Energy Weight: {self.energy_weight}\n")
            f.write(f"Force Weight: {self.force_weight}\n")
            f.write(f"Original NAC Weight: {self.original_nac_weight}\n")
            f.write(f"Final NAC Weight: {self.current_nac_weight}\n")
            f.write(f"Dipole Weight: {self.dipole_weight}\n\n")

            # Adaptive NAC summary
            f.write("ADAPTIVE NAC SYSTEM:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Enabled: {self.adaptive_enabled}\n")
            f.write(f"Weight Reductions: {self.nac_reductions}\n")
            f.write(f"Weight Restorations: {self.nac_restorations}\n")
            f.write(f"Minimum Weight: {self.min_nac_weight}\n")
            f.write(f"Reduction Factor: {self.reduction_factor}\n")
            f.write(f"Spike Threshold: {self.spike_threshold}\n\n")

            # Training results
            f.write("TRAINING RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Epochs Completed: {final_epoch}\n")
            f.write(f"Training Termination Reason: {reason}\n")
            f.write(f"Best Validation Loss: {best_vloss:.6f}\n")
            f.write(f"Final Training Loss: {self.train_losses[-1]:.6f}\n")
            f.write(f"Final Validation Loss: {self.val_losses[-1]:.6f}\n\n")

            # Final loss breakdown
            f.write("FINAL LOSS BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Training - Energy: {self.train_losses_E[-1]:.6f}, Force: {self.train_losses_F[-1]:.6f}, NAC: {self.train_losses_NAC[-1]:.6f}, Dipole: {self.train_losses_mu[-1]:.6f}\n"
            )
            f.write(
                f"Validation - Energy: {self.val_losses_E[-1]:.6f}, Force: {self.val_losses_F[-1]:.6f}, NAC: {self.val_losses_NAC[-1]:.6f}, Dipole: {self.val_losses_mu[-1]:.6f}\n\n"
            )

            # Model information
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 40 + "\n")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")

            # NAC weight history
            f.write("NAC WEIGHT HISTORY:\n")
            f.write("-" * 40 + "\n")
            for i, entry in enumerate(self.nac_weight_history):
                if (
                    i == 0
                    or i == len(self.nac_weight_history) - 1
                    or abs(
                        entry["weight"]
                        - (
                            self.nac_weight_history[i - 1]["weight"]
                            if i > 0
                            else self.original_nac_weight
                        )
                    )
                    > 1e-6
                ):
                    f.write(
                        f"Epoch {entry['epoch']:3d}: weight={entry['weight']:.6f}, "
                        f"train_nac={entry['train_nac']:.4f}, val_nac={entry['val_nac']:.4f}, "
                        f"ratio={entry['ratio']:.2f}\n"
                    )

            # Data information
            f.write(f"\nTraining Samples: {len(self.train_loader.dataset)}\n")
            f.write(f"Validation Samples: {len(self.val_loader.dataset)}\n\n")

            # Files created
            f.write("FILES CREATED:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model Directory: {self.directory}\n")
            f.write(f"Best Model: {self.directory}/best_model.pth\n")
            f.write(f"Loss Plots: {self.directory}/training_losses.png\n")
            f.write(f"TensorBoard Logs: {self.directory}/verbose\n")
            f.write(f"Training Summary: {summary_path}\n")

        print(f"Training summary saved to: {summary_path}")

    def train(self):
        epoch_number = 0
        best_vloss = 1_000_000.0
        patience_counter = 0

        print("Starting training...")

        for epoch in range(self.max_epochs):
            print(f"EPOCH {epoch_number + 1}")

            # Training
            self.model.train()
            avg_loss, avg_loss_E, avg_loss_F, avg_loss_mu, avg_loss_nac = (
                self.train_epoch(epoch_number)
            )

            # Validation
            avg_vloss, avg_vloss_E, avg_vloss_F, avg_vloss_mu, avg_vloss_nac = (
                self.validate_epoch()
            )

            # Adaptive NAC weight adjustment
            weight_changed = self.update_nac_weight(
                avg_loss_nac, avg_vloss_nac, epoch_number + 1
            )

            # Store losses for plotting
            self.train_losses.append(avg_loss)
            self.val_losses.append(avg_vloss)
            self.train_losses_E.append(avg_loss_E)
            self.train_losses_F.append(avg_loss_F)
            self.train_losses_NAC.append(avg_loss_nac)
            self.train_losses_mu.append(avg_loss_mu)
            self.val_losses_E.append(avg_vloss_E)
            self.val_losses_F.append(avg_vloss_F)
            self.val_losses_NAC.append(avg_vloss_nac)
            self.val_losses_mu.append(avg_vloss_mu)

            # Print detailed loss breakdown with percentages
            print(f"LOSS: train {avg_loss:.4f}, validation {avg_vloss:.4f}")
            print(
                f"  TRAIN  - Energy: {avg_loss_E:.4f}, Force: {avg_loss_F:.4f}, NAC: {avg_loss_nac:.4f}, Dipole: {avg_loss_mu:.4f}"
            )
            print(
                f"  VALID  - Energy: {avg_vloss_E:.4f}, Force: {avg_vloss_F:.4f}, NAC: {avg_vloss_nac:.4f}, Dipole: {avg_vloss_mu:.4f}"
            )

            # Show weighted contributions and percentages
            if avg_loss > 0:
                train_weighted_E = self.energy_weight * avg_loss_E
                train_weighted_F = self.force_weight * avg_loss_F
                train_weighted_NAC = self.current_nac_weight * avg_loss_nac

                print(
                    f"  WEIGHTED TRAIN - Energy: {train_weighted_E:.4f} ({100 * train_weighted_E / avg_loss:.1f}%), "
                    f"Force: {train_weighted_F:.4f} ({100 * train_weighted_F / avg_loss:.1f}%), "
                    f"NAC: {train_weighted_NAC:.4f} ({100 * train_weighted_NAC / avg_loss:.1f}%)"
                )

            # Show current NAC weight if it changed
            if weight_changed or epoch_number == 0:
                print(f"  NAC weight: {self.current_nac_weight:.6f}")

            self.writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number + 1,
            )
            self.writer.flush()

            # Early stopping and best model saving
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                patience_counter = 0
                torch.save(self.model.state_dict(), f"{self.directory}/best_model.pth")
                print(f" Best model saved with validation loss: {best_vloss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(
                        f" Early stopping triggered after {self.patience} epochs without improvement"
                    )
                    self.plot_losses()
                    self.save_training_summary(
                        epoch_number + 1, best_vloss, "early_stopping"
                    )
                    return
                print(f" No improvement for {patience_counter}/{self.patience} epochs")

            # Overfitting detection
            if avg_vloss > 1.1 * best_vloss and epoch_number > 10:
                print(
                    f" WARNING: Potential overfitting detected (val_loss {avg_vloss:.4f} > 1.1 * best {best_vloss:.4f})"
                )

            epoch_number += 1

        # Training completed normally
        print("\n" + "=" * 80)
        print("ADAPTIVE TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Final Training Loss: {self.train_losses[-1]:.6f}")
        print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
        print(f"Best Validation Loss: {best_vloss:.6f}")
        print(
            f"Final NAC Weight: {self.current_nac_weight:.6f} (started at {self.original_nac_weight:.6f})"
        )
        print(f"NAC Weight Reductions: {self.nac_reductions}")
        print(f"NAC Weight Restorations: {self.nac_restorations}")

        # Create plots and save summary
        self.plot_losses()
        self.save_training_summary(epoch_number, best_vloss, "max_epochs_reached")

        print(f"\nAll results saved to: {self.directory}")
        print("Files created:")
        print(f"  - Best model: {self.directory}/best_model.pth")
        print(f"  - Loss plots: {self.directory}/training_losses.png")
        print(f"  - Training summary: {self.directory}/training_summary.txt")
        print(f"  - TensorBoard logs: {self.directory}/verbose")
