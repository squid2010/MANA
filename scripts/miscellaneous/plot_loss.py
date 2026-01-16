#!/usr/bin/env python3
"""
Standalone script to plot loss curves from loss_history.npz

Usage:
    python plot_loss.py

This will read loss_history.npz from the same directory and generate loss.png
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt


def plot_loss_curves(npz_path: str, output_path: str) -> None:
    """
    Load loss history from npz file and create loss curve plots.

    Args:
        npz_path: Path to the loss_history.npz file
        output_path: Path where the output PNG will be saved
    """
    # Load the loss history
    data = np.load(npz_path)
    
    # Extract arrays
    train_total = data.get("train_total", None)
    val_total = data.get("val_total", None)
    train_lambda = data.get("train_lambda", None)
    val_lambda = data.get("val_lambda", None)
    train_phi = data.get("train_phi", None)
    val_phi = data.get("val_phi", None)
    
    print(train_lambda)

    # Determine number of epochs from available data
    num_epochs = 0
    for arr in [train_total, val_total, train_lambda, val_lambda, train_phi, val_phi]:
        if arr is not None:
            num_epochs = max(num_epochs, len(arr))

    if num_epochs == 0:
        raise ValueError("No valid loss data found in npz file")

    epochs = np.arange(1, num_epochs + 1)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Total Loss
    ax1 = axes[0]
    if train_total is not None:
        ax1.plot(epochs, train_total, label="Train Total", color="tab:blue")
    if val_total is not None:
        ax1.plot(epochs, val_total, "--", label="Val Total", color="tab:orange")
    ax1.set_title("Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Lambda (Absorption) Loss
    ax2 = axes[1]
    if train_lambda is not None:
        ax2.plot(epochs, train_lambda, label="Train Lambda", color="tab:purple")
    if val_lambda is not None:
        ax2.plot(
            epochs, val_lambda, "--", label="Val Lambda", color="tab:purple", alpha=0.7
        )
    ax2.set_title("Absorption (Lambda) Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Huber Loss")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Phi (Quantum Yield) Loss
    ax3 = axes[2]
    if train_phi is not None:
        ax3.plot(epochs, train_phi, label="Train Phi", color="tab:brown")
    if val_phi is not None:
        ax3.plot(epochs, val_phi, "--", label="Val Phi", color="tab:brown", alpha=0.7)
    ax3.set_title("Quantum Yield (Phi) Loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MSE Loss")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Loss plot saved to: {output_path}")


def main():
    # Default paths
    npz_path = "models/phi/loss_history.npz"
    output_path = "models/phi/loss.png"

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find loss_history.npz at: {npz_path}")

    plot_loss_curves(npz_path, output_path)


if __name__ == "__main__":
    main()
