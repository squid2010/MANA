import os

import matplotlib

# plotting and saving utilities
import numpy as np
import torch
from torch.optim.adam import Adam

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
            
            self.history["train_lambda"].append(train_comps["lambda"])
            self.history["train_phi"].append(train_comps["phi"])
            
            self.history["val_lambda"].append(val_comps["lambda"])
            self.history["val_phi"].append(val_comps["phi"])

            # Print totals and components for transparency
            print(
                f"Epoch {epoch:4d} | "
                f"Train: {train_total:.6f} "
                f"(λ={train_comps['lambda']:.6f}, φ={train_comps['phi']:.6f}) | "
                f"Val: {val_total:.6f} "
                f"(λ={val_comps['lambda']:.6f}, φ={val_comps['phi']:.6f})"
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
            self._save_history()
            self._plot_losses()
            print(f"Saved loss history and plots to: {self.save_dir}")
        except Exception as e:
            # don't crash training if plotting fails; just report
            print(f"Warning: failed to save/plot losses: {e}")

    def _train_epoch(self):
        self.model.train()
    
        total_loss = 0.0
        total_lambda = 0.0
        total_phi = 0.0
        n_batches = 0
    
        for batch in self.train_loader:
            batch = batch.to(self.device)
            preds = self.model(batch)
            loss, metrics = self.model.loss_fn(preds, batch)
    
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
    
            total_loss += loss.item()
            total_lambda += metrics.get("loss_lambda", 0.0)
            total_phi += metrics.get("loss_phi", 0.0)
            n_batches += 1
    
        if n_batches == 0:
            return 0.0, {"lambda": 0.0, "phi": 0.0}
    
        return (
            total_loss / n_batches,
            {
                "lambda": total_lambda / n_batches,
                "phi": total_phi / n_batches,
            },
        )


    @torch.no_grad()
    def _validate(self):
        self.model.eval()
    
        total_loss = 0.0
        total_lambda = 0.0
        total_phi = 0.0
        n_batches = 0
    
        for batch in self.val_loader:
            batch = batch.to(self.device)
            preds = self.model(batch)
            loss, metrics = self.model.loss_fn(preds, batch)
    
            total_loss += loss.item()
            total_lambda += metrics.get("loss_lambda", 0.0)
            total_phi += metrics.get("loss_phi", 0.0)
            n_batches += 1
    
        if n_batches == 0:
            return 0.0, {"lambda": 0.0, "phi": 0.0}
    
        return (
            total_loss / n_batches,
            {
                "lambda": total_lambda / n_batches,
                "phi": total_phi / n_batches,
            },
        )


    def _save_history(self):
        # Convert lists to numpy arrays and save
        save_dict = {k: np.array(v) for k, v in self.history.items()}
        np.savez_compressed(
            os.path.join(self.save_dir, "loss_history.npz"), **save_dict #pyright:ignore[reportArgumentType]
        )

    def _plot_losses(self):
        epochs = np.arange(1, len(self.history["train_total"]) + 1)

        plt.figure(figsize=(10, 6))
        # total losses
        plt.plot(
            epochs, self.history["train_total"], label="Train Total", color="tab:blue"
        )
        plt.plot(
            epochs, self.history["val_total"], label="Val Total", color="tab:orange"
        )

        # components (train dashed, val dotted)
        plt.plot(
            epochs,
            self.history["train_lambda"],
            "--",
            label="Train Lambda",
            color="tab:purple",
        )
        plt.plot(
            epochs,
            self.history["val_lambda"],
            ":",
            label="Val Lambda",
            color="tab:purple",
        )
        plt.plot(
            epochs,
            self.history["train_phi"],
            "--",
            label="Train Phi",
            color="tab:brown",
        )
        plt.plot(
            epochs, self.history["val_phi"], ":", label="Val Phi", color="tab:brown"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fig_path = os.path.join(self.save_dir, "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/models/loss.png")
        plt.savefig(fig_path)
        plt.close()
