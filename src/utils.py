import h5py
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def save_to_hdf5(simulation, filename):
    """
    Save simulation data and parameters to an HDF5 file.
    
    Args:
    simulation (SphericalCollapse): The simulation object to save
    filename (str): The name of the output HDF5 file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, 'w') as hf:
        # Save parameters as attributes in the root group
        for key, value in simulation.get_parameters_dict().items():
            if isinstance(value, (int, float, str, bool, np.number)):
                hf.attrs[key] = value
            else:
                hf.attrs[key] = str(value)

        # Create a group for snapshots
        snapshots_group = hf.create_group('snapshots')

        # Prepare data for saving
        results = {key: [] for key in simulation.snapshots[0].keys()}
        for snapshot in simulation.snapshots:
            for key, value in snapshot.items():
                results[key].append(value)

        # Save snapshot data
        for key, value_list in results.items():
            if not value_list or all(v is None for v in value_list):
                logger.warning(f"Skipping {key} because all values are None")
                continue

            try:
                if np.isscalar(value_list[0]):
                    dataset = snapshots_group.create_dataset(key, data=np.array(value_list))
                else:
                    dataset = snapshots_group.create_dataset(key, data=np.array(value_list))
            except Exception as e:
                logger.error(f"Error creating dataset for {key}: {e}")
                continue

            logger.debug(f"Saved dataset {key} with shape {dataset.shape}")

    logger.info(f"Saved simulation data and parameters to {filename}")

def load_simulation_data(filename):
    """
    Load simulation data and parameters from an HDF5 file.
    
    Args:
    filename (str): The name of the HDF5 file to load
    
    Returns:
    tuple: (params, snapshots) where params is a dict of simulation parameters
           and snapshots is a dict of time series data
    """
    with h5py.File(filename, 'r') as hf:
        # Load parameters
        params = dict(hf.attrs)
        
        # Load snapshot data
        snapshots = {}
        for key in hf['snapshots']:
            snapshots[key] = np.array(hf['snapshots'][key])
    
    return params, snapshots