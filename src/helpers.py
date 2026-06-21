import numpy as np
from scipy.spatial.transform import Rotation as R
from huggingface_hub import hf_hub_download
import h5py

def extract_7dof_state(obs):
    """
    Converts raw robosuite observations into a standardized 7-DoF state representation.

    Args:
        obs (dict): Dictionary containing the observation arrays.
                    Required keys: 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'

    Returns:
        np.ndarray: An array of shape (N, 7) containing (x, y, z, roll, pitch, yaw, gripper_width)
    """
    # 1. Translation: (x, y, z)
    pos = np.atleast_2d(obs['robot0_eef_pos'])

    # 2. Rotation: (roll, pitch, yaw)
    quat = np.atleast_2d(obs['robot0_eef_quat'])
    # Robosuite outputs (x, y, z, w), which is exactly what Scipy expects
    rpy = R.from_quat(quat).as_euler('xyz', degrees=False)

    # 3. Gripper Width: Left Finger (0) - Right Finger (1)
    gripper_qpos = np.atleast_2d(obs['robot0_gripper_qpos'])
    width = gripper_qpos[:, 0] - gripper_qpos[:, 1]

    # Expand width dimensions from (N,) to (N, 1) to match the other arrays
    width = np.expand_dims(width, axis=-1)

    # 4. Concatenate into a single (N, 7) state vector
    state_7dof = np.concatenate([pos, rpy, width], axis=-1)

    # If the input was a single timestep, return a 1D array instead of (1, 7)
    if state_7dof.shape[0] == 1 and obs['robot0_eef_pos'].ndim == 1:
        return state_7dof[0]

    return state_7dof




def download_and_inspect_hdf5_files(
        repo_id = "aboguszewski/robomimic",
        filenames = ["dino3-embeddings/block-lifting/ph.hdf5", "dino3-embeddings/block-lifting/mh-better.hdf5", "dino3-embeddings/block-lifting/mh-okay.hdf5", "dino3-embeddings/block-lifting/mh-worse.hdf5"]
) -> list[str]:
    """
    Downloads HDF5 files from the Hugging Face Hub and inspects their structure.
    """
    hdf5_paths = []
    print("Downloading/Locating HDF5 file from Hugging Face...")
    # This will download the file and cache it locally.
    # Next time you run this, it will just load from the cache instantly.
    for filename in filenames:
        hdf5_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        hdf5_paths.append(hdf5_path)

        print(f"File {filename} located at: {hdf5_path}\n")

        def print_h5_structure(group, indent=0):
            """Recursively prints the structure of an HDF5 group."""
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"{'  ' * indent}- {key} | Shape: {item.shape} | Dtype: {item.dtype}")
                elif isinstance(item, h5py.Group):
                    print(f"{'  ' * indent}+ {key}/")
                    print_h5_structure(item, indent + 1)

        # Open the file and inspect the first demonstration
        with h5py.File(hdf5_path, 'r') as f:
            print(f"=== HDF5 Schema for {filename} (First Demonstration) ===")

            # Robomimic datasets usually store everything under the 'data' key
            if 'data' in f:
                demo_keys = list(f['data'].keys())
                first_demo = demo_keys[0]
                print(f"Total demonstrations: {len(demo_keys)}")
                print(f"Inspecting {first_demo}:\n")

                # Print the structure of just the first demo so we don't flood the console
                print_h5_structure(f[f'data/{first_demo}'])
            else:
                # Fallback just in case it's structured differently
                print_h5_structure(f)

    return hdf5_paths


def resolve_hdf5_path(
    filename: str,
    repo_id: str = "aboguszewski/robomimic"
) -> str:
    """
    Downloads/Locates a single HDF5 file from the Hugging Face Hub and returns the cached local path.
    """
    print(f"Resolving HF dataset: {filename} from repo: {repo_id}...")
    hdf5_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    print(f"Resolved path: {hdf5_path}")
    return hdf5_path