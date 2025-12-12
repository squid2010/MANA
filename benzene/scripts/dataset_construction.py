import h5py
import numpy as np


def load_hdf5(filename):
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

def remove_outliers(data, threshold=3.0):
    """
    Removes outliers from a dictionary of NumPy arrays based on a z-score threshold of the excited and ground energies.
    """
    # Extract ground and excited state energies
    energies_ground = data["energies_ground"]
    energies_excited = data["energies_excited"]

    # Calculate z-scores for ground state energies
    ground_mean = np.mean(energies_ground)
    ground_std = np.std(energies_ground)
    ground_z_scores = np.abs((energies_ground - ground_mean) / ground_std)

    # Calculate z-scores for excited state energies
    # Handle case where excited energies might be 2D (multiple excited states per structure)
    if energies_excited.ndim > 1:
        # Flatten to calculate overall statistics, then reshape z-scores
        excited_flat = energies_excited.flatten()
        excited_mean = np.mean(excited_flat)
        excited_std = np.std(excited_flat)
        excited_z_scores = np.abs((energies_excited - excited_mean) / excited_std)
        # Take max z-score across excited states for each structure
        excited_z_scores_max = np.max(excited_z_scores, axis=1)
    else:
        excited_mean = np.mean(energies_excited)
        excited_std = np.std(energies_excited)
        excited_z_scores_max = np.abs((energies_excited - excited_mean) / excited_std)

    # Create mask for non-outliers (both ground and excited energies must be within threshold)
    valid_mask = (ground_z_scores < threshold) & (excited_z_scores_max < threshold)

    # Apply mask to all arrays in the data dictionary
    filtered_data = {}
    for key, array in data.items():
        if isinstance(array, np.ndarray) and len(array) == len(valid_mask):
            # Apply mask to arrays that have the same length as the number of structures
            filtered_data[key] = array[valid_mask]
        else:
            # Keep arrays that don't correspond to per-structure data (like atomic_numbers)
            filtered_data[key] = array

    return filtered_data

# Example Usage:
file_path = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/benzene/qm_results.h5"
all_data = load_hdf5(file_path)

print(all_data.keys())
