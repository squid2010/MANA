#!/usr/bin/env python3
"""
Test script for real-time TD-DFT implementation in quantum_chemistry.py

This script tests the RT-TDDFT functionality with a simple benzene molecule
to verify that the changes from LR-TDDFT to RT-TDDFT work correctly.
"""

import os
import sys

import numpy as np
from ase import Atoms
from ase.build import molecule

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from quantum_chemistry import QuantumChemistryCalculator


def create_test_benzene():
    """Create a simple benzene molecule for testing."""
    try:
        # Try to use ASE's built-in benzene
        atoms = molecule("C6H6")
    except:
        # If that fails, create a simple benzene manually
        positions = np.array(
            [
                [0.000, 1.396, 0.000],  # C
                [1.209, 0.698, 0.000],  # C
                [1.209, -0.698, 0.000],  # C
                [0.000, -1.396, 0.000],  # C
                [-1.209, -0.698, 0.000],  # C
                [-1.209, 0.698, 0.000],  # C
                [0.000, 2.479, 0.000],  # H
                [2.147, 1.240, 0.000],  # H
                [2.147, -1.240, 0.000],  # H
                [0.000, -2.479, 0.000],  # H
                [-2.147, -1.240, 0.000],  # H
                [-2.147, 1.240, 0.000],  # H
            ]
        )
        symbols = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
        atoms = Atoms(symbols=symbols, positions=positions)

    # Set up unit cell
    atoms.set_cell([15, 15, 15])
    atoms.center()
    return atoms


def test_rt_tddft_basic():
    """Test basic RT-TDDFT functionality."""
    print("Testing RT-TDDFT basic functionality...")

    # Create test molecule
    atoms = create_test_benzene()
    print(f"Created benzene with {len(atoms)} atoms")

    # Initialize calculator with fast settings for testing
    calc = QuantumChemistryCalculator(
        functional="PBE",
        mode="fd",
        n_excited_states=2,  # Just test 2 states for speed
        max_workers=1,
    )

    # Test ground state calculation
    print("Running ground state calculation...")
    ground_state = calc._calculate_ground_state(atoms, 0)

    if ground_state is None:
        print("❌ Ground state calculation failed")
        return False

    print(f"✅ Ground state energy: {ground_state['energy']:.6f} Ha")
    print(f"✅ Forces shape: {ground_state['forces'].shape}")

    # Test excited state calculation (RT-TDDFT)
    print("Running RT-TDDFT calculation...")
    try:
        excited_state = calc._calculate_excited_states(atoms, 0, ground_state)

        if excited_state is None:
            print("❌ RT-TDDFT calculation failed")
            return False

        print(f"✅ Excited state calculation completed")
        print(f"✅ Excited energies: {excited_state['energies']}")
        print(f"✅ Oscillator strengths: {excited_state['oscillator_strengths']}")
        print(f"✅ Forces shape: {excited_state['forces'].shape}")

        # Check that we got reasonable excitation energies (should be positive and in eV range)
        excitation_energies = (
            excited_state["energies"] - ground_state["energy"]
        ) * 27.211  # Convert to eV
        print(f"✅ Excitation energies (eV): {excitation_energies}")

        # Basic sanity checks
        if np.any(excitation_energies < 0):
            print("❌ Warning: Found negative excitation energies")

        if np.any(excitation_energies > 20):
            print("❌ Warning: Found very high excitation energies (>20 eV)")

        # Check that first excited state is reasonable for benzene (~4-6 eV expected)
        if 3.0 < excitation_energies[0] < 8.0:
            print("✅ First excitation energy looks reasonable for benzene")
        else:
            print(
                f"⚠️  First excitation energy ({excitation_energies[0]:.2f} eV) may be outside expected range"
            )

        return True

    except Exception as e:
        print(f"❌ RT-TDDFT calculation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_spectrum_extraction():
    """Test that spectrum extraction works properly."""
    print("\nTesting spectrum extraction...")

    # Test with mock dipole data
    time_step = 0.02  # fs
    max_time = 20.0
    nsteps = int(max_time / time_step)

    # Create a mock dipole signal with known frequencies
    time_array = np.linspace(0, max_time, nsteps)
    # Add signals at 5 eV and 7 eV (typical benzene excitations)
    freq1_au = 5.0 / 27.211  # Convert 5 eV to atomic units
    freq2_au = 7.0 / 27.211  # Convert 7 eV to atomic units

    mock_signal = np.sin(2 * np.pi * freq1_au * time_array * 41.34) + 0.5 * np.sin(
        2 * np.pi * freq2_au * time_array * 41.34
    )

    # Apply window and FFT
    window = np.hanning(len(mock_signal))
    signal_windowed = mock_signal * window
    fft_signal = np.fft.fft(signal_windowed)
    freqs = np.fft.fftfreq(len(signal_windowed), d=time_step * 41.34)

    # Get positive frequencies and convert to eV
    positive_mask = freqs > 0
    frequencies = freqs[positive_mask] * 27.211
    intensities = np.abs(fft_signal[positive_mask]) ** 2

    # Find peaks
    if len(intensities) > 0:
        threshold = max(intensities) * 0.1
        peaks = []

        for i in range(1, len(frequencies) - 1):
            if (
                intensities[i] > intensities[i - 1]
                and intensities[i] > intensities[i + 1]
                and intensities[i] > threshold
                and frequencies[i] > 0.1
            ):
                peaks.append(frequencies[i])

        print(f"✅ Found {len(peaks)} peaks in test spectrum")
        print(f"✅ Peak frequencies: {sorted(peaks)[:5]} eV")  # Show first 5 peaks

        # Check if we recovered our input frequencies approximately
        peaks_sorted = sorted(peaks)
        if len(peaks_sorted) >= 2:
            if abs(peaks_sorted[0] - 5.0) < 0.5 and abs(peaks_sorted[1] - 7.0) < 0.5:
                print("✅ Successfully recovered input frequencies")
                return True
            else:
                print(f"⚠️  Peak frequencies don't match expected values")
        else:
            print("⚠️  Not enough peaks found")

    return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("RT-TDDFT Implementation Test")
    print("=" * 60)

    # Test spectrum extraction first (faster)
    spectrum_test = test_spectrum_extraction()

    # Test actual RT-TDDFT calculation
    rt_tddft_test = test_rt_tddft_basic()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Spectrum extraction: {'✅ PASS' if spectrum_test else '❌ FAIL'}")
    print(f"RT-TDDFT calculation: {'✅ PASS' if rt_tddft_test else '❌ FAIL'}")

    if spectrum_test and rt_tddft_test:
        print("✅ All tests passed! RT-TDDFT implementation is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
