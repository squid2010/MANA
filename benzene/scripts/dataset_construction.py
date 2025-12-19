import h5py
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class DatasetConstructor(Dataset):
    def _load_hdf5(self, filename):
        """
        Recursively loads all datasets from an HDF5 file into a dictionary.
        """
        data_dict = {}

        def recursively_load(hdf5_object, current_path, data_dict):
            for key, item in hdf5_object.items():
                new_path = f"{current_path}/{key}" if current_path else key
                if isinstance(item, h5py.Dataset):
                    # Load the dataset into a NumPy array in memory
                    data_dict[new_path] = item[()]
                elif isinstance(item, h5py.Group):
                    # Recurse into the group
                    recursively_load(item, new_path, data_dict)

        with h5py.File(filename, "r") as f:
            recursively_load(f, "", data_dict)

        return data_dict

    def __init__(
        self,
        hdf5_file,
        cutoff_radius=None,
        transform=None,
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
    ):
        super().__init__(transform=transform)

        # Data stuff
        data_dict = self._load_hdf5(hdf5_file)
        self.atomic_numbers = np.array(data_dict["atomic_numbers"])
        self.couplings = np.array(data_dict["couplings_nacv"])
        self.energies_ground = np.array(data_dict["energies_ground"])
        self.energies_excited = np.array(data_dict["energies_excited"])
        self.forces_ground = np.array(data_dict["forces_ground"])
        self.forces_excited = np.array(data_dict["forces_excited"])
        self.positions = np.array(data_dict["geometries"])
        self.metadata = np.array(data_dict.get("metadata", {}))
        self.oscillator_strengths = np.array(
            data_dict.get("oscillator_strengths", None)
        )

        # Graph Stuff
        self.cutoff_radius = cutoff_radius

        # Dataloader stuff
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.random_seed = random_seed

        # Preprocess the dataset
        self.preprocess()

        # Update n_structures after preprocessing (outlier removal)
        self.n_structures = len(self.positions)

        # Create train/val/test splits
        self._create_splits()

        # Store normalization parameters
        self._store_normalization_params()

    # Functions to preprocess the dataset
    # 1. Remove outliers based on z-score thresholding
    # 2. Normalize energies by shifting minimum to zero
    # 3. Normalize forces using z-score normalization
    # 4. Normalize couplings using typical magnitudes (median)
    def _remove_outliers(self, threshold=3.0):
        """
        Removes outlier structures based on a z-score threshold of the excited and ground energies.
        """
        # Calculate z-scores for ground state energies
        ground_mean = np.mean(self.energies_ground)
        ground_std = np.std(self.energies_ground)
        ground_z_scores = np.abs((self.energies_ground - ground_mean) / ground_std)

        # Calculate z-scores for excited state energies
        # Handle case where excited energies might be 2D (multiple excited states per structure)
        if self.energies_excited.ndim > 1:
            # Flatten to calculate overall statistics, then reshape z-scores
            excited_flat = self.energies_excited.flatten()
            excited_mean = np.mean(excited_flat)
            excited_std = np.std(excited_flat)
            excited_z_scores = np.abs(
                (self.energies_excited - excited_mean) / excited_std
            )
            # Take max z-score across excited states for each structure
            excited_z_scores_max = np.max(excited_z_scores, axis=1)
        else:
            excited_mean = np.mean(self.energies_excited)
            excited_std = np.std(self.energies_excited)
            excited_z_scores_max = np.abs(
                (self.energies_excited - excited_mean) / excited_std
            )

        # Create mask for non-outliers (both ground and excited energies must be within threshold)
        valid_mask = (ground_z_scores < threshold) & (excited_z_scores_max < threshold)

        # Apply mask to all arrays that correspond to per-structure data
        self.couplings = self.couplings[valid_mask]
        self.energies_excited = self.energies_excited[valid_mask]
        self.energies_ground = self.energies_ground[valid_mask]
        self.forces_excited = self.forces_excited[valid_mask]
        self.forces_ground = self.forces_ground[valid_mask]
        self.positions = self.positions[valid_mask]
        if self.oscillator_strengths is not None:
            self.oscillator_strengths = self.oscillator_strengths[valid_mask]

        # atomic_numbers and metadata are kept as-is since they don't correspond to per-structure data

    def _normalize_energies(self):
        """
        Normalize ground state energies and excited state energies through shifting the minimum to 0
        """

        # normalize ground state energies
        min_ground = np.min(self.energies_ground, axis=0)
        self.energies_ground -= min_ground

        # normalize excited state energies
        min_excited = np.min(self.energies_excited, axis=0)
        self.energies_excited -= min_excited

    def _normalize_forces(self):
        """
        Normalize the Numpy array of forces_ground and forces_excited from the data_dict using z-score
        """

        # Function to normalize all forces
        def normalize_force(forces):
            mean_force = np.mean(forces, axis=0)
            std_force = np.std(forces, axis=0)
            return (forces - mean_force) / std_force

        # Normalize forces_ground and forces_excited
        self.forces_ground = normalize_force(self.forces_ground)
        self.forces_excited = normalize_force(self.forces_excited)

    def _normalize_couplings(self):
        """
        Normalize the Numpy array of couplings from the data_dict using typical magnitudes (median)
        """

        nonzero_mask = np.abs(self.couplings) > 1e-10
        if np.any(nonzero_mask):
            coupling_scale = np.median(np.abs(self.couplings[nonzero_mask]))
        else:
            coupling_scale = 1.0  # fallback

        # Scale by typical magnitude
        self.couplings = self.couplings / coupling_scale
        self.coupling_scale = coupling_scale

    def preprocess(self):
        """
        Process the dataset by removing outliers and normalizing energies, forces, and couplings.
        """
        self._remove_outliers(threshold=3.0)
        self._normalize_energies()
        self._normalize_forces()
        self._normalize_couplings()

    def _create_splits(self):
        """
        Create train/validation/test splits with shuffling.
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Create shuffled indices
        indices = np.arange(self.n_structures)
        np.random.shuffle(indices)

        # Calculate split points
        train_end = int(self.train_split * self.n_structures)
        val_end = train_end + int(self.val_split * self.n_structures)

        # Create splits
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

        print("Dataset splits created:")
        print(f"  Train: {len(self.train_indices)} samples")
        print(f"  Validation: {len(self.val_indices)} samples")
        print(f"  Test: {len(self.test_indices)} samples")

    def _store_normalization_params(self):
        """
        Store normalization parameters for later use.
        """
        # Calculate energy normalization parameters (min values used for shifting)
        self.energy_ground_min = np.min(self.energies_ground)
        self.energy_excited_min = np.min(self.energies_excited, axis=0)

        # Force normalization parameters (mean and std)
        self.force_ground_mean = np.mean(self.forces_ground, axis=0)
        self.force_ground_std = np.std(self.forces_ground, axis=0)
        self.force_excited_mean = np.mean(self.forces_excited, axis=0)
        self.force_excited_std = np.std(self.forces_excited, axis=0)

        # Coupling scale already stored in _normalize_couplings

        self.normalization_params = {
            "energy_ground_min": self.energy_ground_min,
            "energy_excited_min": self.energy_excited_min.tolist()
            if hasattr(self.energy_excited_min, "tolist")
            else self.energy_excited_min,
            "force_ground_mean": self.force_ground_mean.tolist(),
            "force_ground_std": self.force_ground_std.tolist(),
            "force_excited_mean": self.force_excited_mean.tolist(),
            "force_excited_std": self.force_excited_std.tolist(),
            "coupling_scale": self.coupling_scale,
        }

    def get_subset(self, indices):
        """
        Create a subset dataset with specific indices.
        """
        subset = Subset(self, indices)
        return subset

    def get_dataloaders(self, num_workers=0):
        """
        Create DataLoader objects for train, validation, and test sets.

        Args:
            num_workers: Number of workers for data loading (default: 0 for single-threaded)

        Returns:
            train_loader, val_loader, test_loader
        """
        # Create subset datasets
        train_dataset = self.get_subset(self.train_indices)
        val_dataset = self.get_subset(self.val_indices)
        test_dataset = self.get_subset(self.test_indices)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader, test_loader

    def save_normalization_params(self, filepath):
        """
        Save normalization parameters to a JSON file.
        """
        import json

        with open(filepath, "w") as f:
            json.dump(self.normalization_params, f, indent=2)
        print(f"Normalization parameters saved to {filepath}")

    def get_dataset_stats(self):
        """
        Get statistics about the dataset.
        """
        stats = {
            "total_structures": self.n_structures,
            "train_size": len(self.train_indices),
            "val_size": len(self.val_indices),
            "test_size": len(self.test_indices),
            "n_atoms": len(self.atomic_numbers),
            "energy_ground_range": [
                float(np.min(self.energies_ground)),
                float(np.max(self.energies_ground)),
            ],
            "energy_excited_shape": list(self.energies_excited.shape),
            "forces_shape": list(self.forces_ground.shape),
            "couplings_shape": list(self.couplings.shape),
            "normalization_params": self.normalization_params,
        }
        return stats

    def len(self):
        return self.n_structures

    def get(self, idx):
        # Get structure data
        pos = torch.tensor(
            self.positions[idx], dtype=torch.float
        )  # Shape: (n_atoms, 3)
        z = torch.tensor(self.atomic_numbers, dtype=torch.long)  # Shape: (n_atoms,)

        # Create edges based on cutoff
        edge_index, edge_attr = self._create_edges(pos)

        # Prepare targets - combine ground and excited states as per architecture guide
        # Combine energies: [ground_state, excited_state_1, excited_state_2, ...]
        energy_ground = self.energies_ground[idx : idx + 1]  # Shape: (1,)
        if self.energies_excited.ndim > 1:
            energy_excited = self.energies_excited[idx]  # Shape: (n_excited,)
        else:
            energy_excited = self.energies_excited[idx : idx + 1]  # Shape: (1,)
        energies = torch.cat(
            [
                torch.tensor(energy_ground, dtype=torch.float32),
                torch.tensor(energy_excited, dtype=torch.float32),
            ]
        )  # Shape: (n_states,) where n_states = 1 + n_excited

        # Combine forces: [ground_forces, excited_forces_1, excited_forces_2, ...]
        forces = torch.stack(
            [
                torch.tensor(
                    self.forces_ground[idx], dtype=torch.float32
                ),  # Shape: (12, 3)
                *[
                    torch.tensor(self.forces_excited[idx][i], dtype=torch.float32)
                    for i in range(self.forces_excited.shape[1])
                ],  # Each shape: (12, 3)
            ]
        )  # Shape: (n_states, 12, 3)

        # Non-adiabatic couplings
        nac = torch.tensor(
            self.couplings[idx], dtype=torch.float32
        )  # Shape: (n_couplings, 12, 3)

        return Data(
            x=z,  # Node features (atomic numbers)
            pos=pos,  # Node positions
            edge_index=edge_index,  # Edge indices
            edge_attr=edge_attr,  # Edge attributes (distances and unit vectors)
            energies=energies,  # Combined energies (n_states,)
            forces=forces,  # Combined forces (n_states, 12, 3)
            nac=nac,  # Non-adiabatic couplings (n_couplings, 12, 3)
            idx=idx,
        )

    def _create_edges(self, positions):
        n_atoms = positions.size(0)

        if self.cutoff_radius is None:
            # Full connectivity (all pairs except self-loops)
            row, col = torch.meshgrid(
                torch.arange(n_atoms), torch.arange(n_atoms), indexing="ij"
            )
            mask = row != col
            edge_index = torch.stack([row[mask], col[mask]], dim=0)
        else:
            # Cutoff-based connectivity
            dist_matrix = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0))[0]
            mask = (dist_matrix < self.cutoff_radius) & (dist_matrix > 0)
            edge_index = mask.nonzero().t().contiguous()

        # Calculate edge attributes (distances, unit vectors)
        if edge_index.size(1) > 0:
            row, col = edge_index
            diff = (
                positions[col] - positions[row]
            )  # Shape: (n_edges, 3) for displacement vectors
            distances = torch.norm(diff, dim=1, keepdim=True)  # Shape: (n_edges, 1)
            unit_vectors = diff / (
                distances + 1e-8
            )  # Shape: (n_edges, 3), avoiding division by 0

            edge_attr = torch.cat([distances, unit_vectors], dim=1)
        else:
            edge_attr = torch.empty((0, 4), dtype=torch.float)

        return edge_index, edge_attr


if __name__ == "__main__":
    file_path = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/benzene/dataset_construction/qm_results.h5"

    # Create dataset with DataLoader functionality
    dataset = DatasetConstructor(
        file_path,
        cutoff_radius=5,
        batch_size=16,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
    )

    # Get DataLoaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(num_workers=2)

    # Print dataset statistics
    stats = dataset.get_dataset_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save normalization parameters
    dataset.save_normalization_params(
        "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/benzene/dataset_construction/normalization_params.json"
    )

    # Test iteration through DataLoader
    print("\nTesting DataLoader iteration:")
    for batch_idx, batch in enumerate(train_loader):
        print(
            f"  Batch {batch_idx}: {batch.x.size(0)} atoms, {batch.batch.max().item() + 1} graphs"
        )
        if batch_idx >= 2:  # Just test first few batches
            break
