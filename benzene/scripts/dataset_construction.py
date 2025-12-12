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

    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return data[z_scores < threshold]


# Example Usage:
file_path = "/Users/sumerchaudhary/Documents/QuantumProjects/Projects/MANA/benzene/qm_results.h5"
all_data = load_hdf5(file_path)

print(all_data.keys())