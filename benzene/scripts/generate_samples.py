# pyright: basic

import os

import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import BFGSLineSearch
from ase.vibrations import Vibrations
from xtb.ase.calculator import XTB


def generate_geometry():
    carbon_positions = []
    hydrogen_positions = []
    bond_length_C_C = 1.395  # Approximate C-C bond length in benzene in angstroms
    bond_length_C_H = 1.084  # Approximate C-H bond length in benzene in angstroms
    unit_cell_lengths = [7.287, 9.20, 6.688] # Lengths of the unit cell in Angstroms

    # Create initial, unoptimized guess
    for i in range(6):
        angle = i * (2 * np.pi / 6)

        c_x = bond_length_C_C * np.cos(angle)
        c_y = bond_length_C_C * np.sin(angle)
        carbon_positions.append([c_x, c_y, 0.0])

        # Hydrogen positions are slightly further out along the same angle
        h_x = bond_length_C_C * np.cos(angle) + bond_length_C_H * np.cos(angle)
        h_y = bond_length_C_C * np.sin(angle) + bond_length_C_H * np.sin(angle)
        hydrogen_positions.append([h_x, h_y, 0.0])

    calculator = XTB(
        method="GFN2-xTB",
        electronic_temperature=300.0,
        max_iterations=500,  # More SCF iterations
        accuracy=0.1,  # Tighter convergence criteria
    )  # DFT based calculator for optimization

    # Benzene molecule
    benzene = Atoms(
        "C6H6", 
        positions=carbon_positions + hydrogen_positions, 
        cell=unit_cell_lengths,
        calculator=calculator
    )
    
    benzene.center()

    # Start with slightly perturbed geometry, because for some reason it helps a lot
    positions = benzene.positions + np.random.normal(0, 0.1, benzene.positions.shape)
    benzene.positions = positions

    # Optimize benzene
    dyn = BFGSLineSearch(benzene)
    dyn.run(fmax=0.01)

    return benzene

def get_vibration_data(mol):
    vib = Vibrations(mol, name="geometry", delta=0.005)

    vib.run()

    # Get vibrational frequencies and normal modes
    vib_data = vib.get_vibrations()

    frequencies = vib_data.get_frequencies()
    modes = vib_data.get_modes()

    print("\nVibrational frequencies (cm⁻¹):")
    for i, freq in enumerate(frequencies):
        print(f"Mode {i}: {freq:.2f} cm⁻¹")

    vib_modes = [
        (i, freq.real, mode)
        for i, (freq, mode) in enumerate(zip(frequencies, modes))
        if freq.real > 50  # Filter out translation/rotation and very soft modes
    ]

    # Generate animation files for Ovito
    generate_mode_animations(mol, vib_modes)

    vib.clean()

    return vib_modes

def generate_mode_animations(mol, vib_modes, n_frames=20):
    """Generate grid-based multi-trajectory file showing all vibrational modes."""

    os.makedirs("geometry", exist_ok=True)

    # Filter valid modes
    valid_modes = [
        (i, freq, mode)
        for (i, freq, mode) in vib_modes
        if freq is not None and freq > 0
    ]

    n_modes = len(valid_modes)
    grid_size = int(np.ceil(np.sqrt(n_modes)))
    spacing = 8.0  # Spacing between molecules in grid (Angstrom)

    all_frames = []

    for frame in range(n_frames):
        combined_atoms = []
        combined_symbols = []

        for grid_idx, (mode_idx, freq, mode) in enumerate(valid_modes):
            # Calculate grid position
            row = grid_idx // grid_size
            col = grid_idx % grid_size
            grid_offset = np.array([col * spacing, row * spacing, 0])

            # Phase of oscillation
            phase = 2 * np.pi * frame / n_frames
            displacement = 0.3 * np.cos(phase)  # 0.3 Å amplitude

            # Create displaced geometry
            displaced_positions = mol.positions + displacement * mode + grid_offset

            combined_atoms.extend(displaced_positions)
            combined_symbols.extend(mol.get_chemical_symbols())

        # Create combined molecule for this frame
        if combined_atoms:
            combined_mol = Atoms(symbols=combined_symbols, positions=combined_atoms)
            combined_mol.info["comment"] = (
                f"All modes frame {frame}, {n_modes} modes in {grid_size}x{grid_size} grid"
            )
            all_frames.append(combined_mol)

    # Write single multi-trajectory file
    filename = "geometry/all_modes_grid.xyz"
    write(filename, all_frames)
    print(
        f"Generated grid animation: {filename} ({n_modes} modes in {grid_size}x{grid_size} grid)"
    )


def displace_mol(mol, vib_modes, min_amplitude, max_amplitude, num_points):
    amplitudes = np.linspace(min_amplitude, max_amplitude, num_points)

    mols = []
    for mode_idx, freq, mode in vib_modes:
        for amplitude in amplitudes:
            new_positions = mol.positions + amplitude * mode
            new_mol = mol.copy()
            new_mol.set_positions(new_positions)
            mols.append(new_mol)

    return mols

def check_samples(mols, eq_mol, min_dist=0.8, max_dist=2.0, rmsd_min=0.01):
    valid_mols = []
    eq_pos = eq_mol.get_positions()

    for mol_1 in mols:
        mol_1.cell = eq_mol.cell  # Ensure consistent cell
        mol_1.pbc = True  # Enable periodic boundary conditions
        
        pos1 = mol_1.get_positions()
        is_valid = True

        # Check for atomic collisions (minimum distance between any two atoms > min_dist)
        n_atoms = len(pos1)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = np.linalg.norm(pos1[i] - pos1[j])
                if distance < min_dist:
                    is_valid = False
                    break
            if not is_valid:
                break

        # Check for atoms too far from equilibrium (each atom within max_dist of equilibrium)
        if is_valid:
            distances = np.linalg.norm(pos1 - eq_pos, axis=1)
            if np.any(distances > max_dist):
                is_valid = False

        # Check for duplicates (RMSD with previously accepted molecules > rmsd_min)
        if is_valid:
            for mol_2 in valid_mols:
                pos2 = mol_2.get_positions()
                squared_diff = np.sum((pos1 - pos2) ** 2, axis=1)
                rmsd = np.sqrt(np.mean(squared_diff))
                if rmsd < rmsd_min:
                    is_valid = False
                    break

        if is_valid:
            valid_mols.append(mol_1)

    return valid_mols

def generate_sample_file(mols, path="geometry/samples.extxyz"):
    """Generate multi-frame file showing all molecules."""

    dir, _ = os.path.split(path)

    os.makedirs(dir, exist_ok=True)

    write(path, mols, format="extxyz")
    print(f"Generated samples: {path}")

def generate_samples(
    min_amplitude=0.05, 
    max_amplitude=1.0, 
    num_samples=1000, 
    min_dist=0.8, 
    max_dist=2.0, 
    rmsd_min=0.01, 
    path="geometry/samples.extxyz"
):
    mol = generate_geometry()
    vib_data = get_vibration_data(mol)
    num_points = int(num_samples/len(vib_data))
    mols = displace_mol(mol, vib_data, min_amplitude, max_amplitude, num_points)
    valid_mols = check_samples(mols, mol, min_dist, max_dist, rmsd_min)
    generate_sample_file(valid_mols, path)
    
    return valid_mols
    
if __name__ == "__main__":
    generate_samples()