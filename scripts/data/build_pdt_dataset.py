#!/usr/bin/env python3
"""
Dataset builder for MANA (photosensitizer PDT task).

Generates a molecular graph dataset with targets:
- lambda_max (absorption maximum, nm)
- phi_delta (singlet oxygen quantum yield)

Sources:
- QM9 (geometry + atom types)
- HOPV15 (organic chromophores, when available)

All photodynamic targets are synthetic but physically motivated.
"""

import os
import h5py
import numpy as np
import warnings
from typing import Dict, List
warnings.filterwarnings("ignore")


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from torch_geometric.datasets import QM9
    TORCH_GEOM_AVAILABLE = True
except ImportError:
    TORCH_GEOM_AVAILABLE = False


class PhotosensitizerDatasetBuilder:
    def __init__(self, output_path: str, num_molecules: int = 1000):
        self.output_path = output_path
        self.num_molecules = num_molecules
        self.molecules: List[Dict] = []

    # ------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------
    def download_qm9(self, root="./data/qm9") -> List[Dict]:
        if not TORCH_GEOM_AVAILABLE:
            return []

        dataset = QM9(root=root) #pyright: ignore[reportPossiblyUnboundVariable]
        mols = []

        for i, data in enumerate(dataset):
            if i >= self.num_molecules:
                break

            mols.append(
                {
                    "positions": data.pos.numpy(),
                    "atomic_numbers": data.z.numpy(),
                    "gap": data.y[0, 7].item(),  # HOMO-LUMO gap
                    "mol_id": f"qm9_{i}",
                    "smiles": "",
                }
            )

        return mols

    def download_hopv15(self) -> List[Dict]:
        if not RDKIT_AVAILABLE:
            return []

        import pandas as pd

        url = (
            "https://raw.githubusercontent.com/"
            "aspuru-guzik-group/photovoltaic-discovery/"
            "master/data/hopv15.csv"
        )
        df = pd.read_csv(url)

        mols = []
        for i, row in df.iterrows():
            if len(mols) >= self.num_molecules:
                break

            smi = row.get("SMILES", None)
            if not isinstance(smi, str):
                continue

            mol = Chem.MolFromSmiles(smi) #pyright: ignore[reportPossiblyUnboundVariable, reportAttributeAccessIssue]
            if mol is None:
                continue

            mol = Chem.AddHs(mol) #pyright: ignore[reportPossiblyUnboundVariable, reportAttributeAccessIssue]
            AllChem.EmbedMolecule(mol, randomSeed=42) #pyright: ignore[reportPossiblyUnboundVariable, reportAttributeAccessIssue]
            AllChem.MMFFOptimizeMolecule(mol) #pyright: ignore[reportPossiblyUnboundVariable, reportAttributeAccessIssue]

            conf = mol.GetConformer()
            positions = conf.GetPositions()
            atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

            gap = row.get("gap_eV", 2.0)

            mols.append(
                {
                    "positions": positions,
                    "atomic_numbers": atomic_numbers,
                    "gap": gap / 27.211,  #  pyright: ignore[reportOptionalOperand] (eV → Hartree)
                    "mol_id": f"hopv_{i}",
                    "smiles": smi,
                }
            )

        return mols

    # ------------------------------------------------------------
    # Synthetic photodynamics
    # ------------------------------------------------------------
    def assign_photodynamics(self, mol: Dict) -> Dict:
        gap_ev = abs(mol["gap"]) * 27.211

        lambda_max = 1240.0 / max(gap_ev, 1.0)
        lambda_max += np.random.normal(0.0, 20.0)
        lambda_max = np.clip(lambda_max, 300.0, 800.0)

        energy_factor = np.exp(-((gap_ev - 2.0) ** 2))
        phi_delta = energy_factor * (0.6 + np.random.normal(0, 0.15))
        phi_delta = np.clip(phi_delta, 0.05, 0.95)

        mol["lambda_max"] = lambda_max
        mol["phi_delta"] = phi_delta
        return mol

    # ------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------
    def build(self):
        print("Collecting molecules...")

        self.molecules.extend(self.download_qm9())
        if len(self.molecules) < self.num_molecules:
            self.molecules.extend(self.download_hopv15())

        self.molecules = self.molecules[: self.num_molecules]

        if not self.molecules:
            raise RuntimeError("No molecules collected.")

        print(f"Collected {len(self.molecules)} molecules")

        for i, mol in enumerate(self.molecules):
            self.molecules[i] = self.assign_photodynamics(mol)

        self._write_hdf5()

    # ------------------------------------------------------------
    # HDF5
    # ------------------------------------------------------------
    def _write_hdf5(self):
        max_atoms = max(len(m["atomic_numbers"]) for m in self.molecules)
        n = len(self.molecules)

        atomic_numbers = np.zeros((n, max_atoms), dtype=np.int32)
        geometries = np.zeros((n, max_atoms, 3), dtype=np.float32)
        lambda_max = np.zeros(n, dtype=np.float32)
        phi_delta = np.zeros(n, dtype=np.float32)

        for i, mol in enumerate(self.molecules):
            na = len(mol["atomic_numbers"])
            atomic_numbers[i, :na] = mol["atomic_numbers"]
            geometries[i, :na] = mol["positions"]
            lambda_max[i] = mol["lambda_max"]
            phi_delta[i] = mol["phi_delta"]

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with h5py.File(self.output_path, "w") as f:
            f.create_dataset("atomic_numbers", data=atomic_numbers)
            f.create_dataset("geometries", data=geometries)
            f.create_dataset("lambda_max", data=lambda_max)
            f.create_dataset("phi_delta", data=phi_delta)
            f.create_dataset(
                "mol_ids",
                data=np.array([m["mol_id"].encode() for m in self.molecules]),
            )
            f.create_dataset(
                "smiles",
                data=np.array([(m["smiles"] or "").encode() for m in self.molecules]),
            )

            f.attrs["task"] = "PDT photodynamics"
            f.attrs["targets"] = "lambda_max, phi_delta"
            f.attrs["units_lambda_max"] = "nm"

        print(f"Dataset written to {self.output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_molecules", type=int, default=1000)
    args = parser.parse_args()

    builder = PhotosensitizerDatasetBuilder(
        output_path=args.output,
        num_molecules=args.num_molecules,
    )
    builder.build()


if __name__ == "__main__":
    main()
