#pyright: basic
import os
import numpy as np
from ase.io import read
from gpaw import GPAW


def calculate_scf(samples_file, basis='dzp', convergence={'energy': 1e-5, 'density': 1e-4, 'eigenstates': 1e-6}):
    """
    Calculate SCF energies and extract molecular orbital information for benzene molecules.

    Returns:
        results (dict): Dictionary containing:
            - ground_state_energies: List of ground state energies
            - mo_energies: List of molecular orbital energies for each molecule
            - mo_coefficients: List of molecular orbital coefficients
            - homo_lumo_gaps: List of HOMO-LUMO gaps
            - wavefunctions: List of ground state wavefunctions
    """

    # Create output directory if it doesn't exist
    os.makedirs('scf_calc', exist_ok=True)

    samples = read(samples_file, format="extxyz", index=":")

    results = {
        'ground_state_energies': [],
        'mo_energies': [],
        'mo_coefficients': [],
        'homo_lumo_gaps': [],
        'wavefunctions': [],
        'electron_densities': []
    }

    print(f"Starting SCF calculations for {len(samples)} benzene molecules...")
    print("=" * 70)

    for i, atoms in enumerate(samples):
        print(f"\nProcessing molecule {i+1}/{len(samples)}...")

        try:
            # Set up GPAW calculator with optimized settings for benzene
            calc = GPAW(
                mode='fd',  # Finite difference mode
                xc='PBE',   # PBE functional (fast and reliable)
                h=0.25,     # Grid spacing in Angstrom
                spinpol=False,  # No spin polarization for closed-shell benzene
                convergence=convergence,
                maxiter=100,
                occupations={
                    'name': 'fermi-dirac',
                    'width': 0.01  # Small smearing for molecules
                },
                symmetry={'point_group': False},
                txt=f'scf_calc/gpaw/benzene_{i+1}.txt',
                basis=basis
            )

            atoms.calc = calc

            # 1. Get ground state energy
            energy = atoms.get_potential_energy()
            results['ground_state_energies'].append(energy)
            print(f"Ground State Energy: {energy:.6f} eV")

            # 2. Get molecular orbital energies (eigenvalues)
            try:
                mo_energies = calc.get_eigenvalues()

                # Handle different return types (could be array or list of arrays for spin)
                if isinstance(mo_energies, list):
                    mo_energies = mo_energies[0]  # Take first spin channel for closed shell

                # Ensure it's a numpy array
                mo_energies = np.array(mo_energies)
                results['mo_energies'].append(mo_energies.copy())

                print(f"MO energies shape: {mo_energies.shape}")

            except Exception as e:
                print(f"Error getting MO energies: {e}")
                results['mo_energies'].append(None)
                results['mo_coefficients'].append(None)
                results['homo_lumo_gaps'].append(None)
                results['wavefunctions'].append(None)
                results['electron_densities'].append(None)
                continue

            # 3. Get number of electrons to find HOMO/LUMO
            try:
                n_electrons = calc.get_number_of_electrons()
                print(f"Number of electrons: {n_electrons}")

                if n_electrons % 2 == 0 and len(mo_energies) > n_electrons // 2:  # Closed shell system
                    homo_idx = int(n_electrons // 2 - 1)
                    lumo_idx = int(n_electrons // 2)

                    if homo_idx >= 0 and lumo_idx < len(mo_energies):
                        homo_energy = float(mo_energies[homo_idx])
                        lumo_energy = float(mo_energies[lumo_idx])
                        homo_lumo_gap = lumo_energy - homo_energy

                        results['homo_lumo_gaps'].append(homo_lumo_gap)

                        print(f"HOMO energy (orbital {homo_idx+1}): {homo_energy:.6f} eV")
                        print(f"LUMO energy (orbital {lumo_idx+1}): {lumo_energy:.6f} eV")
                        print(f"HOMO-LUMO gap: {homo_lumo_gap:.6f} eV")
                    else:
                        results['homo_lumo_gaps'].append(None)
                        print(f"Invalid HOMO/LUMO indices: HOMO={homo_idx}, LUMO={lumo_idx}, n_orbitals={len(mo_energies)}")
                else:
                    results['homo_lumo_gaps'].append(None)
                    print(f"Open shell or insufficient orbitals: {n_electrons} electrons, {len(mo_energies)} orbitals")

            except Exception as e:
                print(f"Error analyzing HOMO/LUMO: {e}")
                results['homo_lumo_gaps'].append(None)
                n_electrons = 0
                homo_idx = -1

            # 4. Get molecular orbital coefficients/wavefunctions
            mo_coeffs = []
            try:
                n_bands = len(mo_energies)
                max_orbitals = int(min(n_bands, max(10, n_electrons // 2 + 3)))  # Get at least 10 or occupied + 3 virtual

                print(f"Extracting wavefunctions for {max_orbitals} orbitals...")

                for band in range(max_orbitals):
                    try:
                        # Get pseudo-wavefunction for this band
                        psi = calc.get_pseudo_wave_function(band=band, kpt=0, spin=0)
                        mo_coeffs.append(psi.copy())
                    except Exception as band_error:
                        print(f"Could not extract wavefunction for band {band}: {band_error}")
                        break

                results['mo_coefficients'].append(mo_coeffs)
                print(f"Successfully extracted wavefunctions for {len(mo_coeffs)} molecular orbitals")

            except Exception as e:
                print(f"Error extracting MO wavefunctions: {e}")
                results['mo_coefficients'].append([])

            # 5. Get ground state wavefunction (HOMO or highest occupied orbital)
            try:
                if homo_idx >= 0 and len(mo_coeffs) > homo_idx:
                    # Use already extracted HOMO from mo_coeffs
                    ground_state_wf = mo_coeffs[homo_idx]
                    results['wavefunctions'].append(ground_state_wf.copy())
                    print(f"Ground state wavefunction extracted (shape: {ground_state_wf.shape})")
                else:
                    results['wavefunctions'].append(None)
                    print("Could not extract ground state wavefunction - invalid HOMO index")
            except Exception as e:
                print(f"Error extracting ground state wavefunction: {e}")
                results['wavefunctions'].append(None)

            # 6. Get electron density
            try:
                electron_density = calc.get_pseudo_density()
                results['electron_densities'].append(electron_density.copy())
                print(f"Electron density extracted (shape: {electron_density.shape})")
            except Exception as e:
                print(f"Error extracting electron density: {e}")
                results['electron_densities'].append(None)

            # Print orbital energy summary
            try:
                print(f"\nMolecular Orbital Energies (first 10 orbitals):")
                for j in range(min(10, len(mo_energies))):
                    occ_status = "occupied" if j < n_electrons//2 else "virtual"
                    print(f"  MO {j+1:2d}: {float(mo_energies[j]):8.4f} eV ({occ_status})")

                if len(mo_energies) > 10:
                    print(f"  ... and {len(mo_energies)-10} more orbitals")
            except Exception as e:
                print(f"Error printing orbital summary: {e}")

        except Exception as e:
            print(f"Error calculating molecule {i+1}: {e}")
            # Append None values for failed calculations
            results['ground_state_energies'].append(None)
            results['mo_energies'].append(None)
            results['mo_coefficients'].append(None)
            results['homo_lumo_gaps'].append(None)
            results['wavefunctions'].append(None)
            results['electron_densities'].append(None)

        print("-" * 50)

    return results


def save_results(results, output_file='scf_calc/results.npz'):
    """Save calculation results to a compressed numpy file"""

    # Convert lists to arrays where possible, handling None values
    save_dict = {}

    # Save energies as arrays
    energies = [e for e in results['ground_state_energies'] if e is not None]
    if energies:
        save_dict['ground_state_energies'] = np.array(energies)

    gaps = [g for g in results['homo_lumo_gaps'] if g is not None]
    if gaps:
        save_dict['homo_lumo_gaps'] = np.array(gaps)

    # Save MO energies (these are arrays for each molecule)
    mo_energies_valid = [mo for mo in results['mo_energies'] if mo is not None]
    if mo_energies_valid:
        save_dict['mo_energies'] = mo_energies_valid

    # Save other data
    save_dict['mo_coefficients'] = [mo for mo in results['mo_coefficients'] if mo is not None]
    save_dict['wavefunctions'] = [wf for wf in results['wavefunctions'] if wf is not None]
    save_dict['electron_densities'] = [ed for ed in results['electron_densities'] if ed is not None]

    np.savez_compressed(output_file, **save_dict)
    print(f"\nResults saved to {output_file}")


def print_summary(results):
    """Print a summary of the calculation results"""

    n_total = len(results['ground_state_energies'])
    n_successful = len([e for e in results['ground_state_energies'] if e is not None])

    print("\n" + "=" * 70)
    print("CALCULATION SUMMARY")
    print("=" * 70)
    print(f"Total molecules processed: {n_total}")
    print(f"Successful calculations: {n_successful}")
    print(f"Failed calculations: {n_total - n_successful}")

    if n_successful > 0:
        valid_energies = [e for e in results['ground_state_energies'] if e is not None]
        valid_gaps = [g for g in results['homo_lumo_gaps'] if g is not None]

        print(f"\nEnergy Statistics:")
        print(f"  Mean energy: {np.mean(valid_energies):.6f} eV")
        print(f"  Energy range: {np.min(valid_energies):.6f} to {np.max(valid_energies):.6f} eV")
        print(f"  Energy std dev: {np.std(valid_energies):.6f} eV")

        if valid_gaps:
            print(f"\nHOMO-LUMO Gap Statistics:")
            print(f"  Mean gap: {np.mean(valid_gaps):.6f} eV")
            print(f"  Gap range: {np.min(valid_gaps):.6f} to {np.max(valid_gaps):.6f} eV")
            print(f"  Gap std dev: {np.std(valid_gaps):.6f} eV")


if __name__ == "__main__":
    geometry_file = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/NAQMD/benzene/geometry/samples.extxyz"

    print("BENZENE QUANTUM CHEMISTRY ANALYSIS")
    print("Extracting: Ground state energy, wavefunctions, and molecular orbitals")
    print("=" * 70)

    # Run the calculations
    results = calculate_scf(geometry_file)

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)

    print(f"\nCalculations complete!")
    print(f"Output files saved in: scf_calc/")
    print(f"- Individual calculation logs: benzene_*.txt")
    print(f"- Compiled results: results.npz")
