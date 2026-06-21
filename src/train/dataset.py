import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset

from src.helpers import extract_7dof_state
from src.config import Config


class MicroVLADataset(Dataset):
    def __init__(self, hdf5_paths: list[str], config: "Config", use_icl: bool = True, task_to_demos: dict = None):
        """
        Args:
            hdf5_paths: List of paths to task HDF5 files.
            config: Configuration object containing dataset stats and context_len.
            use_icl: If True, fetches a context demo + query demo (length 2K).
                     If False, fetches only the query demo (length K).
            task_to_demos: Optional dict mapping {hdf5_path: [list_of_demo_keys]}.
                           Used to strictly separate train/test splits.
        """
        self.config = config
        self.context_len = config.context_len
        self.hdf5_paths = hdf5_paths
        self.use_icl = use_icl

        self.files = {}
        self.valid_starts = []
        self.task_to_starts = {i: [] for i in range(len(self.hdf5_paths))}

        for task_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, 'r') as f:
                base_grp = f['data'] if 'data' in f else f

                # Use provided split keys if available, otherwise use all demos in the file
                if task_to_demos is not None and path in task_to_demos:
                    demos = task_to_demos[path]
                else:
                    demos = [k for k in base_grp.keys() if k.startswith('demo_')]

                for demo in demos:
                    ep_len = base_grp[demo]['actions'].shape[0]
                    for i in range(ep_len):
                        self.valid_starts.append((task_idx, demo, i, ep_len))
                        self.task_to_starts[task_idx].append((demo, i, ep_len))

    def __len__(self):
        return len(self.valid_starts)

    def _pad_tensor(self, tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len == 0:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[0] = pad_len
        zeros = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, zeros], dim=0)

    def _get_chunk(self, task_idx: int, demo: str, start: int, ep_len: int):
        """Extracts an UNPADDED trajectory chunk capped at context_len."""
        actual_K = min(self.context_len, ep_len - start)
        end = start + actual_K

        file_handle = self.files[task_idx]
        base_grp = file_handle['data'] if 'data' in file_handle else file_handle
        ep_grp = base_grp[demo]

        obs_slice = {
            'robot0_eef_pos': ep_grp['obs']['robot0_eef_pos'][start:end],
            'robot0_eef_quat': ep_grp['obs']['robot0_eef_quat'][start:end],
            'robot0_gripper_qpos': ep_grp['obs']['robot0_gripper_qpos'][start:end]
        }

        # NOTE: Ensure extract_7dof_state is imported/defined
        state_features = extract_7dof_state(obs_slice)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)

        state_mean = self.config.dataset_stats["observation.state"]["mean"]
        state_std = self.config.dataset_stats["observation.state"]["std"]
        normalized_state = (state_tensor - state_mean) / (state_std + 1e-6)

        dino_rv = torch.tensor(ep_grp['obs']['robot0_robotview_image'][start:end], dtype=torch.float32)
        dino_eye = torch.tensor(ep_grp['obs']['robot0_eye_in_hand_image'][start:end], dtype=torch.float32)
        actions = torch.tensor(ep_grp['actions'][start:end], dtype=torch.float32)

        # -------------------------------------------------------------
        # MODIFIED: Conditionally calculate RTG or use step rewards
        # -------------------------------------------------------------
        if getattr(self.config, 'use_rtg', True): # Defaults to True if flag is missing
            # Returns-to-go (Cumulative future rewards)
            all_future_rewards = ep_grp['rewards'][start:]
            processed_rewards = np.cumsum(all_future_rewards[::-1])[::-1][:actual_K].copy()
        else:
            # Standard single-step rewards
            processed_rewards = ep_grp['rewards'][start:end].copy()
            
        rewards_tensor = torch.tensor(processed_rewards, dtype=torch.float32).unsqueeze(-1)

        return {
            "eef_state": normalized_state,
            "dino_robotview": dino_rv,
            "dino_eye": dino_eye,
            "actions": actions,
            "rewards": rewards_tensor, # Renamed from "rtg"
            "actual_K": actual_K
        }

    def __getitem__(self, idx):
        if not self.files:
            for i, path in enumerate(self.hdf5_paths):
                self.files[i] = h5py.File(path, 'r')

        task_idx, query_demo, query_start, query_ep_len = self.valid_starts[idx]
        query_chunk = self._get_chunk(task_idx, query_demo, query_start, query_ep_len)

        # Updated key list to use "rewards" instead of "rtg"
        batch_keys = ["eef_state", "dino_robotview", "dino_eye", "actions", "rewards"]

        if self.use_icl:
            # --- ICL MODE: Context + Query ---
            context_demo, context_start, context_ep_len = random.choice(self.task_to_starts[task_idx])
            context_chunk = self._get_chunk(task_idx, context_demo, context_start, context_ep_len)

            total_max_len = 2 * self.context_len
            combined_valid_len = context_chunk["actual_K"] + query_chunk["actual_K"]
            tail_pad_len = total_max_len - combined_valid_len

            combined_batch = {}
            for key in batch_keys:
                unpadded_combined = torch.cat([context_chunk[key], query_chunk[key]], dim=0)
                combined_batch[key] = self._pad_tensor(unpadded_combined, tail_pad_len)

            ctx_time = torch.arange(0, context_chunk["actual_K"], dtype=torch.long)
            query_time = torch.arange(0, query_chunk["actual_K"], dtype=torch.long)
            unpadded_time = torch.cat([ctx_time, query_time], dim=0)
            combined_batch["timesteps"] = self._pad_tensor(unpadded_time, tail_pad_len)

            # Loss Mask: 0 for context, 1 for query, 0 for tail padding
            context_mask = torch.zeros(context_chunk["actual_K"], dtype=torch.float32)
            query_mask = torch.ones(query_chunk["actual_K"], dtype=torch.float32)
            padding_mask = torch.zeros(tail_pad_len, dtype=torch.float32)
            combined_batch["loss_mask"] = torch.cat([context_mask, query_mask, padding_mask], dim=0)

            combined_batch["task_idx"] = torch.tensor(task_idx, dtype=torch.long)

            return combined_batch

        else:
            # --- BASELINE MODE: Query Only ---
            total_max_len = self.context_len
            tail_pad_len = total_max_len - query_chunk["actual_K"]

            baseline_batch = {}
            for key in batch_keys:
                baseline_batch[key] = self._pad_tensor(query_chunk[key], tail_pad_len)

            query_time = torch.arange(0, query_chunk["actual_K"], dtype=torch.long)
            baseline_batch["timesteps"] = self._pad_tensor(query_time, tail_pad_len)

            # Loss Mask: 1 for query, 0 for padding
            query_mask = torch.ones(query_chunk["actual_K"], dtype=torch.float32)
            padding_mask = torch.zeros(tail_pad_len, dtype=torch.float32)
            baseline_batch["loss_mask"] = torch.cat([query_mask, padding_mask], dim=0)

            baseline_batch["task_idx"] = torch.tensor(task_idx, dtype=torch.long)

            return baseline_batch