# MODULE 2: Quantum Chemistry Calculator

## Overview

This module implements the Quantum Chemistry Calculator as specified in the Benzene Non-Adiabatic ML System Architecture. It computes accurate quantum mechanical properties for molecular geometries including ground state energies, forces, excited states, and non-adiabatic coupling vectors.

## Key Features

- **Multi-threaded processing** optimized for 8GB M1 MacBook Air
- **Memory-efficient calculations** with garbage collection and checkpointing
- **HDF5 output format** following architecture specifications
- **Comprehensive error handling** and validation
- **Batch processing** with automatic restart capability

## Dependencies

```bash
pip install ase gpaw h5py numpy
```

**Note**: GPAW installation requires additional setup. See [GPAW installation guide](https://wiki.fysik.dtu.dk/gpaw/install.html).

## Usage

### Basic Usage

```python
from scripts.quantum_chemistry import run_quantum_chemistry_calculations

# Run calculations on geometries from Module 1
run_quantum_chemistry_calculations(
    geometry_file="geometry/samples.extxyz",
    output_file="qm_results.h5",
    max_workers=2  # Conservative for 8GB RAM
)
```

### Advanced Usage

```python
from scripts.quantum_chemistry import QuantumChemistryCalculator

# Initialize calculator with custom parameters
calc = QuantumChemistryCalculator(
    basis="dzp",                    # Basis set
    functional="PBE",              # Exchange-correlation functional
    n_excited_states=3,            # Number of excited states
    max_workers=2,                 # Parallel workers
    convergence_energy=1e-6,       # Energy convergence threshold
    convergence_density=1e-4,      # Density convergence threshold
)

# Load molecular geometries
structures = calc.load_geometries("geometry/samples.extxyz")

# Process all structures
results = calc.process_geometries(structures)

# Save to HDF5
calc.save_results_hdf5(results, "qm_results.h5")
```

## Input Requirements

### Geometry File Format
- **Format**: Extended XYZ (`.extxyz`)
- **Source**: Output from Module 1 (generate_samples.py)
- **Content**: Multiple molecular structures with positions and metadata

### Expected Structure
- **Atoms**: 12 atoms (C₆H₆ benzene)
- **Cell size**: Minimum 10 Å in all directions
- **Validation**: Automatic checking for reasonable bond lengths and structure integrity

## Output Format

### HDF5 Structure
```
qm_results.h5
├── geometries          : (N, 12, 3) float32 - Atomic positions [Å]
├── atomic_numbers      : (12,) int32 - Element identities [6,6,6,6,6,6,1,1,1,1,1,1]
├── energies_ground     : (N,) float64 - Ground state energies [Hartree]
├── energies_excited    : (N, 3) float64 - Excited state energies [Hartree]
├── forces_ground       : (N, 12, 3) float64 - Ground state forces [Hartree/Bohr]
├── oscillator_strengths: (N, 3) float64 - Spectroscopic data
├── couplings_nacv      : (N, 3, 12, 3) float64 - Non-adiabatic couplings [Bohr⁻¹]
└── metadata/
    ├── basis_set       : string
    ├── functional      : string
    ├── n_excited_states: int
    ├── software        : "GPAW"
    ├── creation_date   : timestamp
    └── failed_structures: list of failed structure indices
```

## Computational Stages

### Stage 1: Ground State Calculation
- Self-Consistent Field (SCF) calculation
- Energy and force computation
- Wavefunction checkpoint saving

### Stage 2: Excited State Calculation  
- Time-Dependent DFT (TD-DFT)
- Excitation energies and oscillator strengths
- Support for 3 excited singlet states (S₁-S₃)

### Stage 3: Non-Adiabatic Coupling Calculation
- Derivative coupling vectors between states
- Simplified finite-difference approximation
- Future: Full analytical coupling implementation

## Memory Optimization Features

### For 8GB M1 MacBook Air
- **Limited workers**: Maximum 2 parallel calculations
- **Memory cleanup**: Automatic garbage collection after each structure
- **Efficient data types**: Float32 for coordinates, compression for HDF5
- **Checkpoint system**: Intermediate results saved to disk
- **Batch processing**: Large datasets processed in chunks

### Performance Estimates
- **Per structure**: 5-15 minutes (depends on basis set)
- **1000 structures**: ~3-5 days wall time
- **Memory usage**: ~2-3 GB per worker thread
- **Disk space**: ~50-500 MB for results

## Error Handling

### Automatic Recovery
- SCF convergence failures: Structure skipped with warning
- Memory errors: Automatic cleanup and retry
- Calculation timeouts: Configurable limits
- Invalid geometries: Pre-validation filtering

### Quality Assurance
- Energy ordering validation: E₀ < E₁ < E₂ < E₃
- Force magnitude checks: |F| < 10 eV/Å
- NaN/Inf detection in all results
- Convergence monitoring and reporting

## Validation

```python
from scripts.quantum_chemistry import validate_qm_results

# Validate results after calculation
validate_qm_results("qm_results.h5")
```

### Validation Checks
- Energy ordering consistency
- Force magnitude reasonableness  
- Spectroscopic property validation
- Data completeness verification

## Integration with Other Modules

### Input from Module 1
```python
# Generate geometries first
from scripts.generate_samples import generate_samples
generate_samples(num_samples=1000, path="geometry/samples.extxyz")

# Then run quantum chemistry
run_quantum_chemistry_calculations()
```

### Output for Module 3
The HDF5 output is directly compatible with Module 3 (Data Preprocessing) input requirements.

## Troubleshooting

### Common Issues

1. **GPAW not found**
   - Install GPAW following official documentation
   - Ensure proper MPI setup for parallel calculations

2. **Memory errors on M1 MacBook**
   - Reduce `max_workers` to 1
   - Increase swap space
   - Process smaller batches

3. **SCF convergence failures**
   - Check molecular geometry validity
   - Adjust convergence criteria
   - Try different initial guess

4. **Slow performance**
   - Verify GPAW is using optimized libraries
   - Check system memory usage
   - Consider smaller basis sets for testing

### Performance Tuning

```python
# For faster testing (lower accuracy)
calc = QuantumChemistryCalculator(
    basis="sz",                    # Smaller basis set
    max_workers=1,                 # Single thread
    convergence_energy=1e-5,       # Looser convergence
)

# For production (higher accuracy)
calc = QuantumChemistryCalculator(
    basis="dzp",                   # Standard basis
    max_workers=2,                 # Parallel processing
    convergence_energy=1e-6,       # Tight convergence
)
```

## Example Workflow

```python
#!/usr/bin/env python3
"""Complete workflow example for Module 2"""

from scripts.quantum_chemistry import run_quantum_chemistry_calculations, validate_qm_results
import os

def main():
    # Check if geometries exist
    geometry_file = "geometry/samples.extxyz"
    if not os.path.exists(geometry_file):
        print("Generating sample geometries first...")
        from scripts.generate_samples import generate_samples
        generate_samples(num_samples=100)  # Small test set
    
    # Run quantum chemistry calculations
    print("Starting quantum chemistry calculations...")
    run_quantum_chemistry_calculations(
        geometry_file=geometry_file,
        output_file="qm_results.h5",
        basis="dzp",
        functional="PBE", 
        n_excited_states=3,
        max_workers=2
    )
    
    # Validate results
    print("Validating results...")
    validate_qm_results("qm_results.h5")
    
    print("✓ Module 2 completed successfully!")

if __name__ == "__main__":
    main()
```

## Next Steps

After successful completion of Module 2:

1. **Verify output quality**: Check absorption spectrum matches experimental benzene data (~4.9 eV first peak)
2. **Proceed to Module 3**: Use `qm_results.h5` as input for data preprocessing
3. **Scale up**: Run on larger geometry datasets from Module 1
4. **Optimize**: Fine-tune basis sets and functionals for accuracy vs. speed

## References

- [GPAW Documentation](https://wiki.fysik.dtu.dk/gpaw/)
- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
- Benzene Non-Adiabatic ML System Architecture Guide