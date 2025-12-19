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
    unit_cell_lengths = [7.287, 9.20, 6.688]  # Lengths of the unit cell in Angstroms

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
        calculator=calculator,
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


def check_samples(mols, eq_mol, min_dist=0.6, max_dist=3.0, rmsd_min=0.005):
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


def generate_random_displacements(mol, num_samples=500, displacement_scale=0.3):
    """Generate random displacements around equilibrium geometry."""
    mols = []
    for _ in range(num_samples):
        # Random displacement with decreasing probability for larger displacements
        displacement = np.random.normal(0, displacement_scale, mol.positions.shape)
        new_positions = mol.positions + displacement
        new_mol = mol.copy()
        new_mol.set_positions(new_positions)
        mols.append(new_mol)
    return mols


def generate_temperature_sampling(mol, num_samples=300, temperature=300):
    """Generate samples based on thermal motion at given temperature."""
    # Approximate thermal energy kT (in eV, then convert to motion)
    kT_eV = 8.617e-5 * temperature  # kT in eV
    # Convert to approximate displacement (very rough estimate)
    thermal_displacement = np.sqrt(kT_eV / 0.1) * 0.1  # ~0.05-0.1 Å

    mols = []
    for _ in range(num_samples):
        displacement = np.random.normal(0, thermal_displacement, mol.positions.shape)
        new_positions = mol.positions + displacement
        new_mol = mol.copy()
        new_mol.set_positions(new_positions)
        mols.append(new_mol)
    return mols


def generate_mixed_displacements(mol, vib_modes, num_samples=200):
    """Generate samples by mixing multiple vibrational modes."""
    mols = []
    n_modes = len(vib_modes)

    for _ in range(num_samples):
        combined_displacement = np.zeros_like(mol.positions)
        # Randomly select 2-4 modes to combine
        n_active_modes = np.random.randint(2, min(5, n_modes + 1))
        selected_modes = np.random.choice(n_modes, n_active_modes, replace=False)

        for mode_idx in selected_modes:
            _, freq, mode = vib_modes[mode_idx]
            # Random amplitude with bias toward smaller values
            amplitude = np.random.exponential(0.2)  # Exponential distribution
            phase = np.random.uniform(0, 2 * np.pi)
            combined_displacement += amplitude * np.cos(phase) * mode

        new_positions = mol.positions + combined_displacement
        new_mol = mol.copy()
        new_mol.set_positions(new_positions)
        mols.append(new_mol)

    return mols


def generate_sample_file(mols, path="/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/benzene/geometry"):
    """Generate multi-frame file showing all molecules."""

    dir, _ = os.path.split(path)

    os.makedirs(dir, exist_ok=True)

    write(path, mols, format="extxyz")
    print(f"Generated samples: {path}")


def generate_samples(
    min_amplitude=0.02,
    max_amplitude=0.8,
    num_samples=2000,
    min_dist=0.6,
    max_dist=3.0,
    rmsd_min=0.005,
    path="/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/benzene/geometry",
):
    mol = generate_geometry()
    vib_data = get_vibration_data(mol)

    print(f"Generating {num_samples} samples using multiple strategies...")

    # Strategy 1: Traditional vibrational mode displacement (40% of samples)
    num_vib_samples = int(0.4 * num_samples)
    num_points = max(
        10, int(num_vib_samples / len(vib_data))
    )  # At least 10 points per mode
    vib_mols = displace_mol(mol, vib_data, min_amplitude, max_amplitude, num_points)
    print(f"Generated {len(vib_mols)} vibrational samples")

    # Strategy 2: Random displacements (30% of samples)
    num_random_samples = int(0.3 * num_samples)
    random_mols = generate_random_displacements(
        mol, num_random_samples, displacement_scale=0.25
    )
    print(f"Generated {len(random_mols)} random displacement samples")

    # Strategy 3: Temperature-based sampling (20% of samples)
    num_temp_samples = int(0.2 * num_samples)
    temp_mols = generate_temperature_sampling(mol, num_temp_samples, temperature=300)
    print(f"Generated {len(temp_mols)} temperature-based samples")

    # Strategy 4: Mixed mode displacements (10% of samples)
    num_mixed_samples = int(0.1 * num_samples)
    mixed_mols = generate_mixed_displacements(mol, vib_data, num_mixed_samples)
    print(f"Generated {len(mixed_mols)} mixed-mode samples")

    # Combine all samples
    all_mols = vib_mols + random_mols + temp_mols + mixed_mols
    print(f"Total generated: {len(all_mols)} samples")

    # Filter with relaxed constraints
    valid_mols = check_samples(all_mols, mol, min_dist, max_dist, rmsd_min)
    print(f"Valid samples after filtering: {len(valid_mols)}")

    # If still not enough, generate more with even more relaxed constraints
    if len(valid_mols) < 0.8 * num_samples:
        print("Generating additional samples with relaxed constraints...")
        extra_random = generate_random_displacements(mol, 300, displacement_scale=0.15)
        extra_valid = check_samples(
            extra_random, mol, min_dist * 0.9, max_dist * 1.2, rmsd_min * 0.5
        )
        valid_mols.extend(extra_valid)
        print(f"Final sample count: {len(valid_mols)}")

    generate_sample_file(valid_mols, path)

    return valid_mols


if __name__ == "__main__":
    generate_samples()
