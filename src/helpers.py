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
    if gripper_qpos.shape[1] == 2:
        width = gripper_qpos[:, 0] - gripper_qpos[:, 1]
    elif gripper_qpos.shape[1] == 6:
        width = 0.08 - 0.1 * np.mean(np.abs(gripper_qpos), axis=1)

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


def get_dataset_stats(
    filenames: list[str] | str,
    repo_id: str = "aboguszewski/robomimic"
) -> dict:
    """
    Calculates trajectory statistics (min, max, mean trajectory length, etc.)
    for the specified HDF5 dataset(s).

    Args:
        filenames (list[str] | str): One or more local dataset paths or Hugging Face filenames.
        repo_id (str): Hugging Face repo ID to resolve paths if filenames are HF keys.

    Returns:
        dict: A dictionary of statistics for each dataset, and overall stats.
    """
    import os

    if isinstance(filenames, str):
        filenames = [filenames]

    stats_dict = {}

    for filename in filenames:
        # Resolve the file path (either local or HF download)
        if os.path.exists(filename):
            hdf5_path = filename
        else:
            hdf5_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

        print(f"Calculating stats for: {filename} (resolved: {hdf5_path})")

        with h5py.File(hdf5_path, 'r') as f:
            base_grp = f['data'] if 'data' in f else f
            demo_keys = list(base_grp.keys())

            # Filter demo keys (typically start with 'demo_')
            demo_keys = [k for k in demo_keys if k.startswith('demo_') or isinstance(base_grp[k], h5py.Group)]

            trajectory_lengths = []

            for demo_key in demo_keys:
                demo_grp = base_grp[demo_key]

                # Find any dataset in the demo group to get the trajectory length from its first dimension
                demo_len = None

                # Check direct children first (e.g. actions)
                for name, item in demo_grp.items():
                    if isinstance(item, h5py.Dataset):
                        demo_len = item.shape[0]
                        break

                # If not found, look deeper recursively or check 'obs'
                if demo_len is None and 'obs' in demo_grp:
                    obs_grp = demo_grp['obs']
                    for name, item in obs_grp.items():
                        if isinstance(item, h5py.Dataset):
                            demo_len = item.shape[0]
                            break

                if demo_len is not None:
                    trajectory_lengths.append(demo_len)

            if not trajectory_lengths:
                print(f"Warning: No trajectories found in {filename}.")
                continue

            total_demos = len(trajectory_lengths)
            total_timesteps = sum(trajectory_lengths)
            min_len = min(trajectory_lengths)
            max_len = max(trajectory_lengths)
            mean_len = float(np.mean(trajectory_lengths))

            file_stats = {
                "total_demos": total_demos,
                "total_timesteps": total_timesteps,
                "min_trajectory_len": min_len,
                "max_trajectory_len": max_len,
                "mean_trajectory_len": mean_len
            }

            print(f"=== Stats for {filename} ===")
            print(f"  Total Demonstrations: {total_demos}")
            print(f"  Total Timesteps:       {total_timesteps}")
            print(f"  Min Trajectory Len:    {min_len}")
            print(f"  Max Trajectory Len:    {max_len}")
            print(f"  Mean Trajectory Len:   {mean_len:.2f}\n")

            stats_dict[filename] = file_stats

    # If multiple files, calculate aggregated stats
    if len(stats_dict) > 1:
        all_min = min(s["min_trajectory_len"] for s in stats_dict.values())
        all_max = max(s["max_trajectory_len"] for s in stats_dict.values())
        all_demos = sum(s["total_demos"] for s in stats_dict.values())
        all_timesteps = sum(s["total_timesteps"] for s in stats_dict.values())
        all_mean = all_timesteps / all_demos if all_demos > 0 else 0.0

        aggregated = {
            "total_demos": all_demos,
            "total_timesteps": all_timesteps,
            "min_trajectory_len": all_min,
            "max_trajectory_len": all_max,
            "mean_trajectory_len": all_mean
        }
        stats_dict["overall"] = aggregated

        print("=== Aggregated Stats ===")
        print(f"  Total Demonstrations: {all_demos}")
        print(f"  Total Timesteps:       {all_timesteps}")
        print(f"  Min Trajectory Len:    {all_min}")
        print(f"  Max Trajectory Len:    {all_max}")
        print(f"  Mean Trajectory Len:   {all_mean:.2f}\n")

    return stats_dict