#!/usr/bin/env python3
"""
Analysis script for benzene quantum chemistry results.
Shows how to access and use the extracted data from GPAW calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os

def load_results(results_file='scf_calc/results.npz'):
    """Load the saved quantum chemistry results"""
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Run quantum_chemistry.py first.")
        return None

    data = np.load(results_file, allow_pickle=True)
    return data

def analyze_energies(results):
    """Analyze ground state energies"""
    print("\n" + "="*60)
    print("GROUND STATE ENERGY ANALYSIS")
    print("="*60)

    energies = results['ground_state_energies']

    print(f"Number of molecules: {len(energies)}")
    print(f"Mean energy: {np.mean(energies):.6f} eV")
    print(f"Standard deviation: {np.std(energies):.6f} eV")
    print(f"Energy range: {np.min(energies):.6f} to {np.max(energies):.6f} eV")

    # Plot energy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(energies, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Ground State Energy (eV)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ground State Energies')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scf_calc/energy_distribution.png', dpi=300)
    print("Energy distribution plot saved as: scf_calc/energy_distribution.png")

    return energies

def analyze_homo_lumo_gaps(results):
    """Analyze HOMO-LUMO gaps"""
    print("\n" + "="*60)
    print("HOMO-LUMO GAP ANALYSIS")
    print("="*60)

    gaps = results['homo_lumo_gaps']

    print(f"Number of molecules with gap data: {len(gaps)}")
    print(f"Mean HOMO-LUMO gap: {np.mean(gaps):.6f} eV")
    print(f"Standard deviation: {np.std(gaps):.6f} eV")
    print(f"Gap range: {np.min(gaps):.6f} to {np.max(gaps):.6f} eV")

    # Plot gap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(gaps, bins=20, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('HOMO-LUMO Gap (eV)')
    plt.ylabel('Frequency')
    plt.title('Distribution of HOMO-LUMO Gaps')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scf_calc/gap_distribution.png', dpi=300)
    print("Gap distribution plot saved as: scf_calc/gap_distribution.png")

    return gaps

def analyze_molecular_orbitals(results):
    """Analyze molecular orbital energies"""
    print("\n" + "="*60)
    print("MOLECULAR ORBITAL ANALYSIS")
    print("="*60)

    mo_energies_list = results['mo_energies']

    if len(mo_energies_list) == 0:
        print("No MO energy data found.")
        return

    # Take the first molecule as an example
    mo_energies = mo_energies_list[0]
    print(f"Example molecule - Number of molecular orbitals: {len(mo_energies)}")

    # Assume benzene has 42 electrons (6 carbons * 6 + 6 hydrogens * 1 = 42)
    n_electrons = 42
    homo_idx = n_electrons // 2 - 1

    print(f"\nFirst 15 molecular orbital energies (eV):")
    print("Orbital   Energy (eV)   Type")
    print("-" * 35)

    for i, energy in enumerate(mo_energies[:15]):
        if i <= homo_idx:
            orbital_type = "Occupied"
        else:
            orbital_type = "Virtual"
        print(f"{i+1:3d}       {energy:8.4f}     {orbital_type}")

    # Plot MO energy diagram for first molecule
    plt.figure(figsize=(12, 8))

    occupied_energies = mo_energies[:homo_idx+1]
    virtual_energies = mo_energies[homo_idx+1:homo_idx+11]  # Show first 10 virtual

    # Plot occupied orbitals
    for i, energy in enumerate(occupied_energies):
        plt.hlines(energy, i-0.4, i+0.4, colors='blue', linewidth=3, label='Occupied' if i == 0 else "")

    # Plot virtual orbitals
    for i, energy in enumerate(virtual_energies):
        idx = len(occupied_energies) + i
        plt.hlines(energy, idx-0.4, idx+0.4, colors='red', linewidth=3, label='Virtual' if i == 0 else "")

    plt.xlabel('Molecular Orbital Index')
    plt.ylabel('Energy (eV)')
    plt.title('Molecular Orbital Energy Diagram (Example Molecule)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scf_calc/mo_diagram.png', dpi=300)
    print("MO energy diagram saved as: scf_calc/mo_diagram.png")

def analyze_wavefunctions(results):
    """Analyze wavefunction data"""
    print("\n" + "="*60)
    print("WAVEFUNCTION ANALYSIS")
    print("="*60)

    wavefunctions = results['wavefunctions']

    if len(wavefunctions) == 0:
        print("No wavefunction data found.")
        return

    # Analyze first available wavefunction
    wf = wavefunctions[0]
    print(f"Wavefunction shape: {wf.shape}")
    print(f"Wavefunction data type: {wf.dtype}")
    print(f"Min/Max values: {np.min(wf):.6f} / {np.max(wf):.6f}")
    print(f"Norm (should be ~1): {np.sum(np.abs(wf)**2) * (0.25**3):.6f}")  # h^3 volume element

    # Plot wavefunction slice
    if len(wf.shape) == 3:
        # Take a slice through the middle of the z-direction
        z_mid = wf.shape[2] // 2
        wf_slice = wf[:, :, z_mid]

        plt.figure(figsize=(10, 8))
        plt.imshow(wf_slice.real, cmap='RdBu', origin='lower')
        plt.colorbar(label='Wavefunction amplitude')
        plt.title(f'HOMO Wavefunction (z-slice at index {z_mid})')
        plt.xlabel('Grid points (x)')
        plt.ylabel('Grid points (y)')
        plt.tight_layout()
        plt.savefig('scf_calc/wavefunction_slice.png', dpi=300)
        print("Wavefunction slice plot saved as: scf_calc/wavefunction_slice.png")

def analyze_electron_density(results):
    """Analyze electron density data"""
    print("\n" + "="*60)
    print("ELECTRON DENSITY ANALYSIS")
    print("="*60)

    densities = results['electron_densities']

    if len(densities) == 0:
        print("No electron density data found.")
        return

    # Analyze first density
    density = densities[0]
    print(f"Density shape: {density.shape}")
    print(f"Total electron count: {np.sum(density) * (0.25**3):.2f}")  # Should be ~42 for benzene
    print(f"Max density: {np.max(density):.6f}")

    # Plot density slice
    if len(density.shape) == 3:
        z_mid = density.shape[2] // 2
        density_slice = density[:, :, z_mid]

        plt.figure(figsize=(10, 8))
        plt.imshow(density_slice, cmap='viridis', origin='lower')
        plt.colorbar(label='Electron density')
        plt.title(f'Electron Density (z-slice at index {z_mid})')
        plt.xlabel('Grid points (x)')
        plt.ylabel('Grid points (y)')
        plt.tight_layout()
        plt.savefig('scf_calc/density_slice.png', dpi=300)
        print("Electron density slice plot saved as: scf_calc/density_slice.png")

def correlation_analysis(results):
    """Analyze correlations between different properties"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    energies = results['ground_state_energies']
    gaps = results['homo_lumo_gaps']

    if len(energies) > 1 and len(gaps) > 1:
        correlation = np.corrcoef(energies, gaps)[0, 1]
        print(f"Correlation between energy and HOMO-LUMO gap: {correlation:.4f}")

        # Scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(energies, gaps, alpha=0.7)
        plt.xlabel('Ground State Energy (eV)')
        plt.ylabel('HOMO-LUMO Gap (eV)')
        plt.title(f'Energy vs HOMO-LUMO Gap (correlation: {correlation:.3f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('scf_calc/energy_gap_correlation.png', dpi=300)
        print("Correlation plot saved as: scf_calc/energy_gap_correlation.png")

def main():
    """Main analysis function"""
    print("BENZENE QUANTUM CHEMISTRY RESULTS ANALYSIS")
    print("=" * 60)

    # Load results
    results = load_results()
    if results is None:
        return

    print(f"Loaded data with keys: {list(results.keys())}")

    # Create output directory for plots
    os.makedirs('scf_calc', exist_ok=True)

    # Run all analyses
    try:
        energies = analyze_energies(results)
        #gaps = analyze_homo_lumo_gaps(results)
        analyze_molecular_orbitals(results)
        analyze_wavefunctions(results)
        analyze_electron_density(results)
        correlation_analysis(results)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("All plots and analysis results saved in scf_calc/ directory")

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure you have run quantum_chemistry.py successfully first.")

if __name__ == "__main__":
    main()
