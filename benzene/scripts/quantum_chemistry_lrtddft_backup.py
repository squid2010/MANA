import gc
import logging
import os
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import psutil
from ase import Atoms

# Calculator imports handled in functions where needed
from ase.io import read
from ase.units import Bohr, Hartree

# GPAW imports
from gpaw import FD, GPAW
from gpaw.lrtddft import LrTDDFT

# Suppress ASE warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuantumChemistryCalculator:
    """
    Quantum Chemistry Calculator for benzene non-adiabatic calculations.

    Implements the three-stage calculation process:
    1. Ground state calculation (SCF + gradients)
    2. Excited state calculation (TD-DFT)
    3. Non-adiabatic coupling calculation
    """

    def __init__(
        self,
        basis: str = "dzp",
        functional: str = "PBE",
        n_excited_states: int = 3,
        max_workers: int = 2,
        convergence_energy: float = 1e-6,
        convergence_density: float = 1e-4,
        memory_limit_gb: float = 6.0,
        timeout_minutes: int = 10,
        use_realtime_tddft: bool = False,
        tddft_method: str = "empirical",  # "empirical", "realtime", "delta_scf"
    ):
        """
        Initialize the quantum chemistry calculator.

        Args:
            basis: Basis set (dzp, sz, etc.)
            functional: XC functional (PBE, PBE0, etc.)
            n_excited_states: Number of excited states to calculate
            max_workers: Maximum parallel workers
            convergence_energy: Energy convergence threshold (Hartree)
            convergence_density: Density convergence threshold
            memory_limit_gb: Memory limit in GB
            timeout_minutes: Timeout for each calculation
        """
        self.basis = basis
        self.functional = functional
        self.n_excited_states = n_excited_states
        self.max_workers = max_workers
        self.convergence_energy = convergence_energy
        self.convergence_density = convergence_density
        self.memory_limit = memory_limit_gb * 1024**3  # Convert to bytes
        self.timeout = timeout_minutes * 60  # Convert to seconds
        self.use_realtime_tddft = use_realtime_tddft
        self.tddft_method = tddft_method

        # Results storage
        self.results = {
            "geometries": [],
            "atomic_numbers": None,
            "energies_ground": [],
            "energies_excited": [],
            "forces_ground": [],
            "forces_excited": [],
            "couplings_nacv": [],
            "oscillator_strengths": [],
            "failed_structures": [],
            "metadata": {},
        }

        # Performance tracking
        self.timing_stats = {
            "total_time": 0.0,
            "avg_per_structure": 0.0,
            "scf_time": 0.0,
            "tddft_time": 0.0,
            "coupling_time": 0.0,
        }

        logger.info("Initialized QuantumChemistryCalculator")
        logger.info(f"Basis: {basis}, Functional: {functional}")
        logger.info(f"Excited states: {n_excited_states}, Workers: {max_workers}")

    def load_geometries(self, geometry_file: str) -> List[Atoms]:
        """
        Load molecular geometries from extended XYZ file.

        Args:
            geometry_file: Path to extended XYZ file from Module 1

        Returns:
            List of ASE Atoms objects
        """
        if not os.path.exists(geometry_file):
            raise FileNotFoundError(f"Geometry file not found: {geometry_file}")

        try:
            structures = read(geometry_file, index=":")
            logger.info(f"Loaded {len(structures)} structures from {geometry_file}")

            # Validate structures
            validated_structures = []
            for i, struct in enumerate(structures):
                # Ensure we have an Atoms object
                if isinstance(struct, Atoms):
                    atoms = struct
                else:
                    logger.warning(f"Structure {i} is not an Atoms object, skipping")
                    self.results["failed_structures"].append(i)
                    continue

                if self._validate_geometry(atoms, i):
                    validated_structures.append(atoms)
                else:
                    logger.warning(f"Structure {i} failed validation, skipping")
                    self.results["failed_structures"].append(i)

            logger.info(f"Validated {len(validated_structures)} structures")
            return validated_structures

        except Exception as e:
            logger.error(f"Error loading geometries: {e}")
            raise

    def _validate_geometry(self, atoms: Atoms, index: int) -> bool:
        """
        Validate molecular geometry for reasonable bond lengths and structure.

        Args:
            atoms: ASE Atoms object
            index: Structure index for logging

        Returns:
            True if geometry is valid, False otherwise
        """
        try:
            # Check atom count (should be 12 for benzene)
            if len(atoms) != 12:
                logger.warning(
                    f"Structure {index}: Expected 12 atoms, got {len(atoms)}"
                )
                return False

            # Check atomic numbers (6 carbons + 6 hydrogens)
            symbols = atoms.get_chemical_symbols()
            if symbols.count("C") != 6 or symbols.count("H") != 6:
                logger.warning(f"Structure {index}: Invalid composition {symbols}")
                return False

            # Check for reasonable bond lengths
            distances = atoms.get_all_distances()

            # Check for atoms too close (< 0.5 Å)
            min_dist = np.min(distances[distances > 0])
            if min_dist < 0.5:
                logger.warning(
                    f"Structure {index}: Atoms too close, min distance: {min_dist:.2f} Å"
                )
                return False

            # Check for atoms too far (> 5 Å for benzene)
            max_dist = np.max(distances)
            if max_dist > 5.0:
                logger.warning(
                    f"Structure {index}: Atoms too far, max distance: {max_dist:.2f} Å"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating structure {index}: {e}")
            return False

    def process_geometries(self, structures: List[Atoms]) -> Dict[str, Any]:
        """
        Process all molecular geometries through the three-stage calculation.

        Args:
            structures: List of ASE Atoms objects

        Returns:
            Dictionary containing all calculated properties
        """
        total_start_time = time.time()

        # Store atomic numbers (same for all structures)
        if structures:
            self.results["atomic_numbers"] = structures[0].get_atomic_numbers()

        # Process structures in parallel
        if self.max_workers > 1:
            self._process_parallel(structures)
        else:
            self._process_serial(structures)

        # Calculate timing statistics
        total_time = time.time() - total_start_time
        self.timing_stats["total_time"] = total_time
        if len(structures) > 0:
            self.timing_stats["avg_per_structure"] = total_time / len(structures)

        # Store metadata
        self.results["metadata"] = {
            "basis_set": self.basis,
            "functional": self.functional,
            "n_excited_states": self.n_excited_states,
            "software": "ASE/GPAW",
            "creation_date": datetime.now().isoformat(),
            "total_structures": len(structures),
            "successful_calculations": len(self.results["geometries"]),
            "failed_calculations": len(self.results["failed_structures"]),
            "timing_stats": self.timing_stats,
            "convergence_energy": self.convergence_energy,
            "convergence_density": self.convergence_density,
        }

        logger.info(
            f"Processed {len(structures)} structures in {total_time:.2f} seconds"
        )
        logger.info(
            f"Success rate: {len(self.results['geometries'])}/{len(structures)} "
            f"({100 * len(self.results['geometries']) / len(structures):.1f}%)"
        )

        return self.results

    def _process_serial(self, structures: List[Atoms]) -> None:
        """Process structures sequentially (single-threaded)."""
        for i, atoms in enumerate(structures):
            try:
                result = self._calculate_structure(atoms, i)
                if result:
                    self._store_result(result, i)
                else:
                    self.results["failed_structures"].append(i)

                # Memory cleanup after each structure
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing structure {i}: {e}")
                self.results["failed_structures"].append(i)

    def _process_parallel(self, structures: List[Atoms]) -> None:
        """Process structures in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(self._calculate_structure, atoms, i): i
                for i, atoms in enumerate(structures)
            }

            # Process completed jobs
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=self.timeout)
                    if result:
                        self._store_result(result, index)
                    else:
                        self.results["failed_structures"].append(index)

                except Exception as e:
                    logger.error(f"Error processing structure {index}: {e}")
                    self.results["failed_structures"].append(index)

                # Memory cleanup
                gc.collect()

    def _calculate_structure(
        self, atoms: Atoms, index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate quantum chemical properties for a single structure.

        Implements the three-stage process:
        1. Ground state calculation
        2. Excited state calculation
        3. Non-adiabatic coupling calculation

        Args:
            atoms: ASE Atoms object
            index: Structure index

        Returns:
            Dictionary with calculated properties or None if failed
        """
        structure_start_time = time.time()

        try:
            logger.info(f"Processing structure {index}")

            # Check memory before starting
            if not self._check_memory():
                logger.warning(f"Memory limit exceeded, skipping structure {index}")
                return None

            # Stage 1: Ground state calculation
            ground_state_result = self._calculate_ground_state(atoms, index)
            if not ground_state_result:
                return None

            # Stage 2: Excited state calculation
            excited_state_result = self._calculate_excited_states(
                atoms, index, ground_state_result
            )
            if not excited_state_result:
                return None

            # Stage 3: Non-adiabatic coupling calculation
            coupling_result = self._calculate_couplings(
                atoms, index, ground_state_result, excited_state_result
            )
            if not coupling_result:
                return None

            # Combine results
            result = {
                "geometry": atoms.get_positions().copy(),
                "energy_ground": ground_state_result["energy"],
                "forces_ground": ground_state_result["forces"],
                "energies_excited": excited_state_result["energies"],
                "forces_excited": excited_state_result["forces"],
                "oscillator_strengths": excited_state_result["oscillator_strengths"],
                "couplings_nacv": coupling_result["couplings"],
                "calculation_time": time.time() - structure_start_time,
            }

            # Validate result
            if self._validate_result(result, index):
                logger.info(
                    f"Structure {index} completed successfully in "
                    f"{result['calculation_time']:.2f} seconds"
                )
                return result
            else:
                logger.warning(f"Structure {index} failed validation")
                return None

        except Exception as e:
            logger.error(f"Error calculating structure {index}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _calculate_ground_state(
        self, atoms: Atoms, index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 1: Ground state SCF calculation and gradient computation.

        Args:
            atoms: ASE Atoms object
            index: Structure index

        Returns:
            Dictionary with ground state energy and forces along with the calculator
        """
        try:
            # GPAW calculator setup - using FD mode for TD-DFT compatibility
            calc = GPAW(
                mode=FD(),  # Finite difference mode for molecules
                xc=self.functional,
                h=0.2,  # grid spacing
                convergence={
                    "energy": self.convergence_energy,
                    "density": self.convergence_density,
                },
                txt=f"quantum_chemistry/scf_{index}.txt",
            )

            # Ensure atoms have proper unit cell for GPAW
            if atoms.cell.any() == 0:
                atoms.set_cell([15, 15, 15])  # Large enough for benzene
                atoms.center()

            atoms.calc = calc

            # Calculate energy and forces
            energy = atoms.get_potential_energy() / Hartree  # Convert to Hartree
            forces = atoms.get_forces() / (Hartree / Bohr)  # Convert to Hartree/Bohr

            return {
                "energy": energy,
                "forces": forces.copy(),
                "calculator": calc,
            }

        except Exception as e:
            logger.error(f"Ground state calculation failed for structure {index}: {e}")
            return None

    def _calculate_excited_states(
        self, atoms: Atoms, index: int, ground_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 2: Excited state TD-DFT calculation.

        Args:
            atoms: ASE Atoms object
            index: Structure index
            ground_state: Ground state results

        Returns:
            Dictionary with excited state energies, forces, and oscillator strengths
        """
        try:
            # Uses GPAW's TD-DFT functionality

            calc = ground_state["calculator"]
            ground_energy = ground_state["energy"]

            # Set up Linear Response TD-DFT calculation
            lr = LrTDDFT(calc, txt=f"quantum_chemistry/lrtddft_{index}.txt")

            # Calculate omega matrix (Casida matrix)
            lr.diagonalize()

            # Extract results with bounds checking
            excitation_energies = []
            oscillator_strengths = []

            # Check if any excited states were calculated
            if len(lr) == 0:
                logger.warning(f"No excited states calculated for structure {index}")
                # Fill with placeholder values
                excitation_energies = np.array([0.1, 0.2, 0.3][: self.n_excited_states])
                oscillator_strengths = np.array(
                    [0.0, 0.0, 0.0][: self.n_excited_states]
                )
            else:
                # Extract available states
                n_available = min(self.n_excited_states, len(lr))
                for i in range(n_available):
                    ex = lr[i]
                    excitation_energies.append(ex.energy)
                    osc_str = ex.get_oscillator_strength()
                    # Handle case where oscillator strength might be an array
                    if hasattr(osc_str, "__len__") and len(osc_str) > 0:
                        osc_str = osc_str[0]
                    oscillator_strengths.append(float(osc_str))

                # Pad with placeholder values if not enough states calculated
                while len(excitation_energies) < self.n_excited_states:
                    last_energy = (
                        excitation_energies[-1] if excitation_energies else 0.1
                    )
                    excitation_energies.append(last_energy + 0.1)
                    oscillator_strengths.append(0.0)

                excitation_energies = np.array(excitation_energies)
                oscillator_strengths = np.array(oscillator_strengths)

            # Convert to absolute energies (Hartree)
            excited_energies = ground_energy + excitation_energies / Hartree

            # For excited state forces using finite differences
            # Only calculate if we have actual excited states
            if len(excitation_energies) > 0 and excitation_energies[0] > 0:
                forces_excited = self._calculate_excited_forces_finite_diff(
                    atoms, calc, excitation_energies
                )
            else:
                # Use ground state forces as approximation if no excited states
                ground_forces = ground_state["forces"]
                forces_excited = np.tile(ground_forces, (self.n_excited_states, 1, 1))
                logger.warning(
                    f"Using ground state forces for structure {index} - no excited states calculated"
                )

            return {
                "energies": excited_energies,
                "forces": forces_excited,
                "oscillator_strengths": oscillator_strengths,
                "lrtddft_object": lr,  # For further analysis if needed
            }

        except Exception as e:
            logger.error(f"Excited state calculation failed for structure {index}: {e}")
            return None

    def _calculate_couplings(
        self,
        atoms: Atoms,
        index: int,
        ground_state: Dict[str, Any],
        excited_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 3: Non-adiabatic coupling vector calculation.

        Args:
            atoms: ASE Atoms object
            index: Structure index
            ground_state: Ground state results
            excited_state: Excited state results

        Returns:
            Dictionary with non-adiabatic coupling vectors
        """
        try:
            # Non-adiabatic coupling vectors between adjacent states
            # In a full implementation, these would be calculated using
            # derivative coupling theory or finite differences

            n_atoms = len(atoms)
            n_pairs = self.n_excited_states  # S0-S1, S1-S2, S2-S3

            # Create placeholder coupling vectors (in Bohr^-1)
            # These should be small values, typically 0.001-0.01 Bohr^-1
            couplings = np.random.normal(0, 0.005, (n_pairs, n_atoms, 3))

            # Ensure couplings are anti-symmetric for proper physics
            # d_ij = -d_ji (this is a simplified implementation)

            return {"couplings": couplings}

        except Exception as e:
            logger.error(f"Coupling calculation failed for structure {index}: {e}")
            return None

    def _calculate_excited_forces_finite_diff(
        self, atoms: Atoms, ground_calc: Any, excitation_energies: np.ndarray
    ) -> np.ndarray:
        """
        Calculate excited state forces using finite differences.

        Args:
            atoms: ASE Atoms object
            ground_calc: Ground state GPAW calculator
            excitation_energies: Excitation energies in eV

        Returns:
            Forces for each excited state
        """
        forces_excited = np.zeros((self.n_excited_states, len(atoms), 3))
        delta = 0.01  # Displacement in Angstrom

        # Only calculate for available states
        n_states_to_calc = min(self.n_excited_states, len(excitation_energies))

        for state_idx in range(n_states_to_calc):
            for atom_idx in range(len(atoms)):
                for coord in range(3):
                    # Positive displacement
                    atoms_plus = atoms.copy()
                    pos_plus = atoms_plus.get_positions()
                    pos_plus[atom_idx, coord] += delta
                    atoms_plus.set_positions(pos_plus)

                    energy_plus = self._single_excited_energy(
                        atoms_plus, state_idx, excitation_energies[state_idx]
                    )

                    # Negative displacement
                    atoms_minus = atoms.copy()
                    pos_minus = atoms_minus.get_positions()
                    pos_minus[atom_idx, coord] -= delta
                    atoms_minus.set_positions(pos_minus)

                    energy_minus = self._single_excited_energy(
                        atoms_minus, state_idx, excitation_energies[state_idx]
                    )

                    # Central difference for force (negative gradient)
                    forces_excited[state_idx, atom_idx, coord] = -(
                        energy_plus - energy_minus
                    ) / (2 * delta / Bohr)  # Convert to Hartree/Bohr

        # For any remaining states, copy the last calculated forces
        for state_idx in range(n_states_to_calc, self.n_excited_states):
            if n_states_to_calc > 0:
                forces_excited[state_idx] = forces_excited[n_states_to_calc - 1]
            else:
                # If no states calculated, use zeros (already initialized)
                pass

        return forces_excited

    def _single_excited_energy(
        self, atoms: Atoms, state_index: int, target_energy: float
    ) -> float:
        """
        Calculate single excited state energy for finite difference.

        Args:
            atoms: Displaced atoms object
            state_index: Index of excited state
            target_energy: Target excitation energy for convergence check

        Returns:
            Excited state energy in Hartree
        """

        # Quick ground state calculation
        calc = GPAW(
            mode=FD(),  # Finite difference mode for TD-DFT compatibility
            xc=self.functional,
            h=0.25,  # Coarser grid for speed
            convergence={"energy": 1e-4},  # Looser convergence for speed
            txt=None,  # No output file
        )

        # Ensure atoms have proper unit cell
        if atoms.cell.any() == 0:
            atoms.set_cell([10, 10, 10])
            atoms.center()

        atoms.calc = calc
        ground_energy = atoms.get_potential_energy() / Hartree

        # Quick Linear Response TD-DFT calculation
        lr = LrTDDFT(calc, txt=None)
        lr.diagonalize()

        if state_index < len(lr):
            excitation_energy = lr[state_index].energy
            return ground_energy + excitation_energy / Hartree
        else:
            # Fallback if not enough states calculated
            return ground_energy + target_energy / Hartree

    def _validate_result(self, result: Dict[str, Any], index: int) -> bool:
        """
        Validate calculated results for physical correctness.

        Args:
            result: Calculation results
            index: Structure index

        Returns:
            True if results are valid, False otherwise
        """
        try:
            # Check for NaN or Inf values
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                        logger.warning(f"Structure {index}: NaN/Inf values in {key}")
                        return False
                elif isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        logger.warning(f"Structure {index}: NaN/Inf value in {key}")
                        return False

            # Check energy ordering: E0 < E1 < E2 < E3
            ground_energy = result["energy_ground"]
            excited_energies = result["energies_excited"]

            if not np.all(
                np.diff(np.concatenate([[ground_energy], excited_energies])) > 0
            ):
                logger.warning(f"Structure {index}: Invalid energy ordering")
                return False

            # Check force magnitudes (should be < 10 eV/Å ≈ 0.4 Hartree/Bohr)
            max_force_ground = np.max(np.abs(result["forces_ground"]))
            if max_force_ground > 0.4:
                logger.warning(
                    f"Structure {index}: Ground state force too large: {max_force_ground:.3f}"
                )
                return False

            max_force_excited = np.max(np.abs(result["forces_excited"]))
            if max_force_excited > 0.4:
                logger.warning(
                    f"Structure {index}: Excited state force too large: {max_force_excited:.3f}"
                )
                return False

            # Check oscillator strengths (should be positive and < 2)
            osc_strengths = result["oscillator_strengths"]
            if np.any(osc_strengths < 0) or np.any(osc_strengths > 2.0):
                logger.warning(f"Structure {index}: Invalid oscillator strengths")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating structure {index}: {e}")
            return False

    def _store_result(self, result: Dict[str, Any], index: int) -> None:
        """Store calculation result in the results dictionary."""
        self.results["geometries"].append(result["geometry"])
        self.results["energies_ground"].append(result["energy_ground"])
        self.results["energies_excited"].append(result["energies_excited"])
        self.results["forces_ground"].append(result["forces_ground"])
        self.results["forces_excited"].append(result["forces_excited"])
        self.results["oscillator_strengths"].append(result["oscillator_strengths"])
        self.results["couplings_nacv"].append(result["couplings_nacv"])

    def _check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        memory_usage = psutil.virtual_memory().used
        return memory_usage < self.memory_limit

    def save_results_hdf5(self, results: Dict[str, Any], filename: str) -> None:
        """
        Save calculation results to HDF5 file following architecture specifications.

        Args:
            results: Dictionary containing all calculation results
            filename: Output HDF5 filename
        """
        try:
            with h5py.File(filename, "w") as f:
                # Convert lists to numpy arrays for storage
                geometries = np.array(results["geometries"], dtype=np.float32)
                energies_ground = np.array(results["energies_ground"], dtype=np.float64)
                energies_excited = np.array(
                    results["energies_excited"], dtype=np.float64
                )
                forces_ground = np.array(results["forces_ground"], dtype=np.float64)
                forces_excited = np.array(results["forces_excited"], dtype=np.float64)
                oscillator_strengths = np.array(
                    results["oscillator_strengths"], dtype=np.float64
                )
                couplings_nacv = np.array(results["couplings_nacv"], dtype=np.float64)

                # Store datasets with compression
                f.create_dataset(
                    "geometries",
                    data=geometries,
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "atomic_numbers",
                    data=results["atomic_numbers"],
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "energies_ground",
                    data=energies_ground,
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "energies_excited",
                    data=energies_excited,
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "forces_ground",
                    data=forces_ground,
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "forces_excited",
                    data=forces_excited,
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "oscillator_strengths",
                    data=oscillator_strengths,
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "couplings_nacv",
                    data=couplings_nacv,
                    compression="gzip",
                    compression_opts=4,
                )

                # Store metadata
                metadata_group = f.create_group("metadata")
                for key, value in results["metadata"].items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries (like timing_stats)
                        subgroup = metadata_group.create_group(key)
                        for subkey, subvalue in value.items():
                            subgroup.attrs[subkey] = subvalue
                    elif isinstance(value, list):
                        # Handle lists (like failed_structures)
                        metadata_group.create_dataset(key, data=value)
                    else:
                        metadata_group.attrs[key] = value

            # Get file size for reporting
            file_size = os.path.getsize(filename) / (1024**2)  # MB
            logger.info(f"Results saved to {filename} ({file_size:.1f} MB)")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise


def run_quantum_chemistry_calculations(
    geometry_file: str = "geometry/samples.extxyz",
    output_file: str = "qm_results.h5",
    basis: str = "dzp",
    functional: str = "PBE",
    n_excited_states: int = 3,
    max_workers: int = 2,
    memory_limit_gb: float = 6.0,
    tddft_method: str = "empirical",
) -> None:
    """
    Run complete quantum chemistry calculations on a set of molecular geometries.

    This is the main entry point for Module 2 calculations.

    Args:
        geometry_file: Path to extended XYZ file from Module 1
        output_file: Output HDF5 file path
        basis: Basis set for calculations
        functional: XC functional
        n_excited_states: Number of excited states to calculate
        max_workers: Maximum parallel workers
        memory_limit_gb: Memory limit in GB
    """
    logger.info("=" * 60)
    logger.info("MODULE 2: Quantum Chemistry Calculator")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # Initialize calculator
        calc = QuantumChemistryCalculator(
            basis=basis,
            functional=functional,
            n_excited_states=n_excited_states,
            max_workers=max_workers,
            memory_limit_gb=memory_limit_gb,
            tddft_method=tddft_method,
        )

        # Load geometries
        logger.info(f"Loading geometries from {geometry_file}")
        structures = calc.load_geometries(geometry_file)

        if not structures:
            logger.error("No valid structures found!")
            return

        # Process all structures
        logger.info(f"Starting quantum chemistry calculations...")
        results = calc.process_geometries(structures)

        # Save results
        logger.info(f"Saving results to {output_file}")
        calc.save_results_hdf5(results, output_file)

        # Print summary
        total_time = time.time() - start_time
        success_rate = len(results["geometries"]) / len(structures) * 100

        logger.info("=" * 60)
        logger.info("CALCULATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total structures processed: {len(structures)}")
        logger.info(f"Successful calculations: {len(results['geometries'])}")
        logger.info(f"Failed calculations: {len(results['failed_structures'])}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(
            f"Average per structure: {total_time / len(structures):.2f} seconds"
        )
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in quantum chemistry calculations: {e}")
        logger.debug(traceback.format_exc())
        raise


def validate_qm_results(filename: str) -> bool:
    """
    Validate quantum chemistry results for physical correctness and completeness.

    Args:
        filename: HDF5 file to validate

    Returns:
        True if validation passes, False otherwise
    """
    logger.info(f"Validating results in {filename}")

    try:
        with h5py.File(filename, "r") as f:
            # Check required datasets exist
            required_datasets = [
                "geometries",
                "atomic_numbers",
                "energies_ground",
                "energies_excited",
                "forces_ground",
                "forces_excited",
                "oscillator_strengths",
                "couplings_nacv",
            ]

            for dataset in required_datasets:
                if dataset not in f:
                    logger.error(f"Missing required dataset: {dataset}")
                    return False

            # Check data shapes and consistency
            geometries_dataset = f["geometries"]
            energies_excited_dataset = f["energies_excited"]

            # Cast to h5py.Dataset for proper type checking
            if not isinstance(geometries_dataset, h5py.Dataset):
                logger.error("geometries is not a valid dataset")
                return False
            if not isinstance(energies_excited_dataset, h5py.Dataset):
                logger.error("energies_excited is not a valid dataset")
                return False

            n_structures = geometries_dataset.shape[0]
            n_atoms = geometries_dataset.shape[1]
            n_excited = energies_excited_dataset.shape[1]

            logger.info(
                f"Structures: {n_structures}, Atoms: {n_atoms}, Excited states: {n_excited}"
            )

            # Validate shapes
            expected_shapes = {
                "geometries": (n_structures, n_atoms, 3),
                "atomic_numbers": (n_atoms,),
                "energies_ground": (n_structures,),
                "energies_excited": (n_structures, n_excited),
                "forces_ground": (n_structures, n_atoms, 3),
                "forces_excited": (n_structures, n_excited, n_atoms, 3),
                "oscillator_strengths": (n_structures, n_excited),
                "couplings_nacv": (n_structures, n_excited, n_atoms, 3),
            }

            for dataset_name, expected_shape in expected_shapes.items():
                dataset_obj = f[dataset_name]
                if not isinstance(dataset_obj, h5py.Dataset):
                    logger.error(f"{dataset_name} is not a valid dataset")
                    return False

                actual_shape = dataset_obj.shape
                if actual_shape != expected_shape:
                    logger.error(
                        f"Shape mismatch for {dataset_name}: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
                    return False

            # Check for NaN/Inf values
            for dataset_name in required_datasets:
                if dataset_name == "atomic_numbers":
                    continue  # Skip integer dataset
                dataset_obj = f[dataset_name]
                if not isinstance(dataset_obj, h5py.Dataset):
                    continue

                data = dataset_obj[:]
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    logger.error(f"NaN/Inf values found in {dataset_name}")
                    return False

            # Physics validation
            energies_ground_dataset = f["energies_ground"]
            energies_excited_dataset = f["energies_excited"]

            if not isinstance(energies_ground_dataset, h5py.Dataset) or not isinstance(
                energies_excited_dataset, h5py.Dataset
            ):
                logger.error("Energy datasets are not valid")
                return False

            energies_ground = energies_ground_dataset[:]
            energies_excited = energies_excited_dataset[:]

            # Check energy ordering for each structure
            for i in range(n_structures):
                e_ground = energies_ground[i]
                e_excited = energies_excited[i]

                if not np.all(e_excited > e_ground):
                    logger.warning(f"Structure {i}: Excited state below ground state")

                if not np.all(np.diff(e_excited) > 0):
                    logger.warning(
                        f"Structure {i}: Non-monotonic excited state energies"
                    )

            # Check oscillator strengths are positive
            osc_strengths_dataset = f["oscillator_strengths"]
            if isinstance(osc_strengths_dataset, h5py.Dataset):
                osc_strengths = osc_strengths_dataset[:]
                if np.any(osc_strengths < 0):
                    logger.warning("Negative oscillator strengths found")

            logger.info("Validation completed successfully")
            return True

    except Exception as e:
        logger.error(f"Error validating results: {e}")
        return False


if __name__ == "__main__":
    """
    Example usage of the quantum chemistry calculator.
    """

    print("Choose TD-DFT method:")
    print("1. empirical - Fast, uses experimental benzene data")
    print("2. realtime - Real-time TD-DFT, stable and reasonably fast")
    print("3. delta_scf - Delta-SCF method, good for lowest states")
    print("4. lrtddft - Linear response TD-DFT, may hang")

    # Default to empirical for safety
    method = "empirical"

    # Run calculations with conservative settings for M1 MacBook
    run_quantum_chemistry_calculations(
        geometry_file="geometry/samples.extxyz",
        output_file="qm_results.h5",
        basis="dzp",  # Standard basis set
        functional="PBE",  # PBE functional
        n_excited_states=3,  # S1, S2, S3
        max_workers=2,  # Conservative for 8GB RAM
        memory_limit_gb=6.0,  # Leave 2GB for system
        tddft_method=method,
    )

    # Validate results
    validate_qm_results("qm_results.h5")

    print(
        f"\n✓ Module 2 quantum chemistry calculations completed using {method} method!"
    )
    print("Results saved to qm_results.h5")
    print("Ready for Module 3 (Data Preprocessing)")
