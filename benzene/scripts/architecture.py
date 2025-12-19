import os
import torch
from torch import nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Check if accelerator is available
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")