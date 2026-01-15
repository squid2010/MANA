import os
import matplotlib
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Use non-interactive backend so training on headless machines works
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingEngine:
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        hyperparams,
        save_dir,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir

        self.max_epochs = hyperparams["max_epochs"]
        self.patience = hyperparams["early_stopping_patience"]

        self.optimizer = Adam(
            self.model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=True, # pyright: ignore[reportArgumentType]
        )

        os.makedirs(save_dir, exist_ok=True)

        # History for plotting and analysis
        self.history = {
            "train_total": [],
            "val_total": [],
            "train_lambda": [],
            "train_phi": [],
            "val_lambda": [],
            "val_phi": [],
        }

    def train(self):
        best_val = float("inf")
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            train_total, train_comps = self._train_epoch()
            val_total, val_comps = self._validate()

            # store history
            self.history["train_total"].append(train_total)
            self.history["val_total"].append(val_total)

            self.history["train_lambda"].append(train_comps.get("loss_lambda", 0))
            self.history["train_phi"].append(train_comps.get("loss_phi", 0))
            self.history["val_lambda"].append(val_comps.get("loss_lambda", 0))
            self.history["val_phi"].append(val_comps.get("loss_phi", 0))

            # Step the learning rate scheduler
            self.scheduler.step(val_total)

            # Print totals and components for transparency
            current_lr = self.optimizer.param_groups[0]["lr"]
            lam_str = f"λ={train_comps.get('loss_lambda', 0):.2f}"
            phi_str = f"φ={train_comps.get('loss_phi', 0):.4f}"

            val_lam_str = f"λ={val_comps.get('loss_lambda', 0):.2f}"
            val_phi_str = f"φ={val_comps.get('loss_phi', 0):.4f}"

            print(
                f"Epoch {epoch:4d} | "
                f"Train: {train_total:.4f} ({lam_str}, {phi_str}) | "
                f"Val: {val_total:.4f} ({val_lam_str}, {val_phi_str}) | "
                f"LR: {current_lr:.2e}"
            )

            # checkpointing based on validation total loss
            if val_total < best_val:
                best_val = val_total
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, "best_model.pth"),
                )
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        # After training, save and plot loss curves
        try:
            self._plot_losses()
            print(f"Saved loss history and plots to: {self.save_dir}")
        except Exception as e:
            # don't crash training if plotting fails; just report
            print(f"Warning: failed to save/plot losses: {e}")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        accumulators = {}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for i, batch in enumerate(pbar):
            batch = batch.to(self.device)

            preds = self.model(batch)

            loss, metrics = self.model.loss_fn(preds, batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in metrics.items():
                accumulators[k] = accumulators.get(k, 0.0) + v
            n_batches += 1

            current_lam = metrics.get("loss_lambda", 0.0)
            current_phi = metrics.get("loss_phi", 0.0)
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.2f}",
                    "λ": f"{current_lam:.1f}",
                    "φ": f"{current_phi:.4f}",
                }
            )

        if n_batches == 0:
            return 0.0, {}

        self._save_history()

        avg_metrics = {k: v / n_batches for k, v in accumulators.items()}
        return total_loss / n_batches, avg_metrics

    @torch.no_grad()
    def _validate(self):
        self.model.eval()

        total_loss = 0.0

        accumulators = {}

        n_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            preds = self.model(batch)
            loss, metrics = self.model.loss_fn(preds, batch)

            total_loss += loss.item()

            for k, v in metrics.items():
                accumulators[k] = accumulators.get(k, 0.0) + v

            n_batches += 1

        if n_batches == 0:
            return 0.0, {}

        self._save_history()

        # Average out
        avg_metrics = {k: v / n_batches for k, v in accumulators.items()}
        return total_loss / n_batches, avg_metrics

    def _save_history(self):
        # Convert lists to numpy arrays and save
        save_dict = {k: np.array(v) for k, v in self.history.items()}
        np.savez_compressed(
            os.path.join(self.save_dir, "loss_history.npz"),
            **save_dict,  # pyright:ignore[reportArgumentType]
        )

    def _plot_losses(self):
        epochs = np.arange(1, len(self.history["train_total"]) + 1)

        # Create 2 subplots so the scales don't mess each other up
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Lambda Max (High values)
        ax1.plot(
            epochs,
            self.history["train_lambda"],
            label="Train Lambda",
            color="tab:purple",
        )
        ax1.plot(
            epochs,
            self.history["val_lambda"],
            "--",
            label="Val Lambda",
            color="tab:purple",
        )
        ax1.set_title("Absorption (Lambda) Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Huber Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Phi (Low values)
        ax2.plot(
            epochs, self.history["train_phi"], label="Train Phi", color="tab:brown"
        )
        ax2.plot(
            epochs, self.history["val_phi"], "--", label="Val Phi", color="tab:brown"
        )
        ax2.set_title("Quantum Yield (Phi) Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # FIX: Just use a relative filename. It will save inside self.save_dir automatically.
        fig_path = os.path.join(self.save_dir, "loss_curves.png")
        plt.savefig(fig_path)
        plt.close()
