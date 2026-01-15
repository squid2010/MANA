import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm  # Add progress bar for startup


class GeometricSubset:
    """
    Wrapper to handle train/val/test splits for the dataset.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Fetch directly from the parent dataset's pre-computed list
        return self.dataset[self.indices[idx]]


class DatasetConstructor(Dataset):
    def __init__(
        self,
        hdf5_file,
        cutoff_radius=5.0,
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
        num_atom_types=None,
    ):
        super().__init__()

        print(f"Loading raw data from {hdf5_file}...")
        with h5py.File(hdf5_file, "r") as f:
            self.atomic_numbers = f["atomic_numbers"][()]  # pyright: ignore[reportIndexIssue]
            self.positions = f["geometries"][()]  # pyright: ignore[reportIndexIssue]
            self.lambda_max = f["lambda_max"][()]  # pyright: ignore[reportIndexIssue]
            self.phi_delta = f["phi_delta"][()]  # pyright: ignore[reportIndexIssue]
            self.mol_ids = f["mol_ids"][()]  # pyright: ignore[reportIndexIssue]
            self.dielectric = f["dielectric"][()]  # pyright: ignore[reportIndexIssue]

            raw_smiles = f["smiles"][()]  # pyright: ignore[reportIndexIssue]
            self.smiles = [s.decode("utf-8") if isinstance(s, bytes) else s for s in raw_smiles]  # pyright: ignore[reportGeneralTypeIssues]

        # 1. Build Vocabulary
        unique_atoms = set()
        for z in self.atomic_numbers:  # pyright: ignore[reportGeneralTypeIssues]
            unique_atoms.update(z[z > 0])  # Ignore padding (0)

        self.unique_atoms = sorted(list(unique_atoms))
        self.atom_to_index = {a: i + 1 for i, a in enumerate(self.unique_atoms)}

        self.num_atom_types = (
            num_atom_types if num_atom_types is not None else len(self.unique_atoms) + 1
        )

        self.cutoff_radius = cutoff_radius
        self.batch_size = batch_size
        self.n_structures = self.atomic_numbers.shape[0]  # pyright: ignore[reportAttributeAccessIssue]

        # 2. PRE-COMPUTE GRAPHS (The Speed Fix)
        # We process all molecules into PyG Data objects right now
        print(
            f"Pre-processing {self.n_structures} molecular graphs (this speeds up training)..."
        )
        self.data_list = []

        for idx in tqdm(range(self.n_structures)):
            data_obj = self._process_one(idx)
            self.data_list.append(data_obj)

        # 3. Create Splits
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        idx = np.random.permutation(self.n_structures)

        n_train = int(train_split * self.n_structures)
        n_val = int(val_split * self.n_structures)

        self.train_indices = idx[:n_train]
        self.val_indices = idx[n_train : n_train + n_val]
        self.test_indices = idx[n_train + n_val :]

        train_lambda = self.lambda_max[self.train_indices]  # pyright: ignore[reportIndexIssue]

        self.lambda_mean = np.mean(train_lambda)  # pyright: ignore[reportCallIssue, reportArgumentType]
        self.lambda_std = np.std(train_lambda)  # pyright: ignore[reportCallIssue, reportArgumentType]

    def _process_one(self, idx):
        """Internal helper to process a single molecule index into a Data object"""
        z_raw = self.atomic_numbers[idx]  # pyright: ignore[reportIndexIssue]

        # Map atoms
        z = torch.tensor(
            [self.atom_to_index[a] if a > 0 else 0 for a in z_raw],  # pyright: ignore[reportGeneralTypeIssues]
            dtype=torch.long,
        )

        pos = torch.tensor(self.positions[idx], dtype=torch.float32)  # pyright: ignore[reportIndexIssue]
        atom_mask = torch.tensor(z_raw > 0, dtype=torch.bool)  # pyright: ignore[reportOperatorIssue]

        # Squeeze out padding NOW so the model doesn't process ghost atoms
        # This makes the tensors smaller and faster
        real_mask = atom_mask
        z = z[real_mask]
        pos = pos[real_mask]

        # Generate Edges
        # Since we removed padding, we don't need to filter creating edges
        # We just compute distances on the real atoms
        if pos.size(0) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4))
        else:
            dist = torch.cdist(pos, pos)
            mask = (dist < self.cutoff_radius) & (dist > 0)
            row, col = mask.nonzero(as_tuple=True)
            edge_index = torch.stack([row, col], dim=0)

            diff = pos[col] - pos[row]
            d = torch.norm(diff, dim=1, keepdim=True)
            u = diff / (d + 1e-8)
            edge_attr = torch.cat([d, u], dim=1)

        return Data(
            x=z,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            lambda_max=torch.tensor([self.lambda_max[idx]], dtype=torch.float32),  # pyright: ignore[reportIndexIssue]
            phi_delta=torch.tensor([self.phi_delta[idx]], dtype=torch.float32),  # pyright: ignore[reportIndexIssue]
            mol_id=torch.tensor([self.mol_ids[idx]], dtype=torch.int32),  # pyright: ignore[reportIndexIssue]
            dielectric=torch.tensor([self.dielectric[idx]], dtype=torch.float32).view(1, 1),  # pyright: ignore[reportIndexIssue]
            smiles=self.smiles[idx],
        )

    def len(self):
        return self.n_structures

    def get(self, idx):
        return self.data_list[idx]

    # Required for the subset wrapper to work nicely via indexing
    def __getitem__(self, idx):
        return self.data_list[idx]  # pyright: ignore[reportArgumentType, reportCallIssue]

    def get_dataloaders(self, num_workers=0):
        # Since data is pre-loaded in RAM, num_workers=0 is usually fastest!
        return (
            DataLoader(
                GeometricSubset(self, self.train_indices),  # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            DataLoader(
                GeometricSubset(self, self.val_indices),  # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            DataLoader(
                GeometricSubset(self, self.test_indices),  # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        )
