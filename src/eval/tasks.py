import robosuite
import h5py
import random
import torch
import numpy as np

from src.helpers import extract_7dof_state


def build_standard_env(
    env,
    hdf5_path: str,
    cfg,
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    control_freq=20,
    horizon=300
):
    """
    Builds the standard Robosuite Lift environment and extracts a random
    demonstration trajectory from the dataset to serve as the ICL context.
    """
    # 1. Instantiate the Environment
    env = robosuite.make(
        env,
        robots="Panda",
        has_renderer=False,           # Headless for evaluation loops
        has_offscreen_renderer=True,  # Required to grab camera frames
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        camera_heights=512,
        camera_widths=512,
        reward_shaping=True,
    )

    # 2. Extract ICL Context from Dataset
    with h5py.File(hdf5_path, 'r') as f:
        base_grp = f['data'] if 'data' in f else f
        demos = [k for k in base_grp.keys() if k.startswith('demo_')]

        # Pick a random demonstration
        demo = random.choice(demos)
        ep_grp = base_grp[demo]

        ep_len = ep_grp['actions'].shape[0]
        # We extract from the beginning of the episode up to context_len
        K = min(cfg.context_len, ep_len)

        # --- A. Proprioception (State) ---
        obs_slice = {
            'robot0_eef_pos': ep_grp['obs']['robot0_eef_pos'][:K],
            'robot0_eef_quat': ep_grp['obs']['robot0_eef_quat'][:K],
            'robot0_gripper_qpos': ep_grp['obs']['robot0_gripper_qpos'][:K]
        }

        # Ensure extract_7dof_state is in scope
        state_features = extract_7dof_state(obs_slice)  # Shape: (K, 7)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)

        # Normalize State using Config Stats
        state_mean = cfg.dataset_stats["observation.state"]["mean"].clone().detach().to(torch.float32)
        state_std = cfg.dataset_stats["observation.state"]["std"].clone().detach().to(torch.float32)
        normalized_state = (state_tensor - state_mean) / (state_std + 1e-6)

        # --- B. Vision (Dino embeddings)
        raw_rv = torch.tensor(ep_grp['obs']['robot0_robotview_image'][:K], dtype=torch.float32)
        raw_eye = torch.tensor(ep_grp['obs']['robot0_eye_in_hand_image'][:K], dtype=torch.float32)

        # --- C. Actions & Returns-To-Go ---
        actions = torch.tensor(ep_grp['actions'][:K], dtype=torch.float32)

        if cfg.use_time_based_rewards:
            dones = ep_grp['dones'][:]
            rewards = np.where(dones == 1, cfg.time_based_success_reward, -1.0)
        else:
            rewards = ep_grp['rewards'][:]
        # Calculate full trajectory RTG, then slice the first K elements
        rtg_array = np.cumsum(rewards[::-1])[::-1][:K].copy()
        rtg = torch.tensor(rtg_array, dtype=torch.float32).unsqueeze(-1)

    # 3. Package Context Dictionary
    context = {
        "eef_state": normalized_state,
        "img_robotview": raw_rv,
        "img_eye": raw_eye,
        "actions": actions,
        "rtg": rtg
    }

    return env, context

import robosuite
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_composite_controller_config

class VariantLift(Lift):
    """
    An advanced subclass of Lift that injects a target object alongside
    multiple distractor objects to test language-driven visual grounding.
    """
    def __init__(self, target_shape="cube", target_rgba=[1, 0, 0, 1], distractors=None, **kwargs):
        self.target_shape = target_shape
        self.target_rgba = target_rgba
        # List of dicts, e.g., [{"shape": "cube", "rgba": [0,0,1,1]}]
        self.distractors = distractors if distractors is not None else []
        super().__init__(**kwargs)

    def _load_model(self):
        super(Lift, self)._load_model()

        # 1. Setup the Arena
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # 2. Build the Target Object (MUST be named "cube" for success tracking)
        if self.target_shape == "cube":
            self.cube = BoxObject(
                name="cube", size_min=[0.02, 0.02, 0.02], size_max=[0.022, 0.022, 0.022],
                rgba=self.target_rgba, material=None
            )
        elif self.target_shape == "puck":
            self.cube = CylinderObject(
                name="cube", size_min=[0.025, 0.010], size_max=[0.030, 0.015],
                rgba=self.target_rgba, material=None
            )

        all_objects = [self.cube]

        # 3. Build the Distractor Objects
        for i, dist in enumerate(self.distractors):
            d_shape = dist.get("shape", "cube")
            d_rgba = dist.get("rgba", [0.5, 0.5, 0.5, 1])
            d_name = f"distractor_{i}"  # Unique names prevent XML clashes

            if d_shape == "cube":
                d_obj = BoxObject(
                    name=d_name, size_min=[0.02, 0.02, 0.02], size_max=[0.022, 0.022, 0.022],
                    rgba=d_rgba, material=None
                )
            elif d_shape == "puck":
                d_obj = CylinderObject(
                    name=d_name, size_min=[0.025, 0.010], size_max=[0.030, 0.015],
                    rgba=d_rgba, material=None
                )
            all_objects.append(d_obj)

        # 4. Standard Placement and Compilation
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=all_objects,
            # Widened the spawn area slightly from the default so
            # multiple objects have room to spawn without colliding
            x_range=[-0.03, 0.03],
            y_range=[-0.03, 0.03],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

def build_color_varied_lift(
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    color_name: str = "blue",
    rgba: list = [0, 0, 1, 1],
    control_freq=20,
    horizon=300,
):
    """Builds a Lift task with a uniquely colored cube."""

    env = robosuite.make(
        "VariantLift",                  # <--- Using our new subclass!
        target_shape="cube",
        target_rgba=rgba,               # Injecting the color
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        camera_heights=512,
        camera_widths=512,
        reward_shaping=True,
    )

    context = {
        "eef_state": None,
        "img_robotview": None,
        "img_eye": None,
        "actions": None,
        "rtg": None,
    }
    return env, context


def build_standard_lift(
    hdf5_path: str,
    cfg,
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    control_freq=20,
    horizon=300
):
    """
    Builds the standard Robosuite Lift environment and extracts a random
    demonstration trajectory from the dataset to serve as the ICL context.
    """
    # 1. Instantiate the Environment
    env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,           # Headless for evaluation loops
        has_offscreen_renderer=True,  # Required to grab camera frames
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        camera_heights=512,
        camera_widths=512,
        reward_shaping=True,
    )

    # 2. Extract ICL Context from Dataset
    with h5py.File(hdf5_path, 'r') as f:
        base_grp = f['data'] if 'data' in f else f
        demos = [k for k in base_grp.keys() if k.startswith('demo_')]

        # Pick a random demonstration
        demo = random.choice(demos)
        ep_grp = base_grp[demo]

        ep_len = ep_grp['actions'].shape[0]
        # We extract from the beginning of the episode up to context_len
        K = min(cfg.context_len, ep_len)

        # --- A. Proprioception (State) ---
        obs_slice = {
            'robot0_eef_pos': ep_grp['obs']['robot0_eef_pos'][:K],
            'robot0_eef_quat': ep_grp['obs']['robot0_eef_quat'][:K],
            'robot0_gripper_qpos': ep_grp['obs']['robot0_gripper_qpos'][:K]
        }

        # Ensure extract_7dof_state is in scope
        state_features = extract_7dof_state(obs_slice)  # Shape: (K, 7)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)

        # Normalize State using Config Stats
        state_mean = cfg.dataset_stats["observation.state"]["mean"].clone().detach().to(torch.float32)
        state_std = cfg.dataset_stats["observation.state"]["std"].clone().detach().to(torch.float32)
        normalized_state = (state_tensor - state_mean) / (state_std + 1e-6)

        # --- B. Vision (Dino embeddings)
        raw_rv = torch.tensor(ep_grp['obs']['robot0_robotview_image'][:K], dtype=torch.float32)
        raw_eye = torch.tensor(ep_grp['obs']['robot0_eye_in_hand_image'][:K], dtype=torch.float32)

        # --- C. Actions & Returns-To-Go ---
        actions = torch.tensor(ep_grp['actions'][:K], dtype=torch.float32)

        if cfg.use_time_based_rewards:
            dones = ep_grp['dones'][:]
            rewards = np.where(dones == 1, cfg.time_based_success_reward, -1.0)
        else:
            rewards = ep_grp['rewards'][:]
        # Calculate full trajectory RTG, then slice the first K elements
        rtg_array = np.cumsum(rewards[::-1])[::-1][:K].copy()
        rtg = torch.tensor(rtg_array, dtype=torch.float32).unsqueeze(-1)

    # 3. Package Context Dictionary
    context = {
        "eef_state": normalized_state,
        "img_robotview": raw_rv,
        "img_eye": raw_eye,
        "actions": actions,
        "rtg": rtg
    }

    return env, context


def build_standard_stack(
    hdf5_path: str,
    cfg,
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    control_freq=20,
    horizon=300
):
    """
    Builds the standard Robosuite Stack environment and extracts a random
    demonstration trajectory from the dataset to serve as the ICL context.
    """
    # 1. Instantiate the Environment
    env = robosuite.make(
        "Stack",
        robots="Panda",
        has_renderer=False,           # Headless for evaluation loops
        has_offscreen_renderer=True,  # Required to grab camera frames
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        camera_heights=512,
        camera_widths=512,
        reward_shaping=True,
    )

    # 2. Extract ICL Context from Dataset
    with h5py.File(hdf5_path, 'r') as f:
        base_grp = f['data'] if 'data' in f else f
        demos = [k for k in base_grp.keys() if k.startswith('demo_')]

        # Pick a random demonstration
        demo = random.choice(demos)
        ep_grp = base_grp[demo]

        ep_len = ep_grp['actions'].shape[0]
        # We extract from the beginning of the episode up to context_len
        K = min(cfg.context_len, ep_len)

        # --- A. Proprioception (State) ---
        obs_slice = {
            'robot0_eef_pos': ep_grp['obs']['robot0_eef_pos'][:K],
            'robot0_eef_quat': ep_grp['obs']['robot0_eef_quat'][:K],
            'robot0_gripper_qpos': ep_grp['obs']['robot0_gripper_qpos'][:K]
        }

        # Ensure extract_7dof_state is in scope
        state_features = extract_7dof_state(obs_slice)  # Shape: (K, 7)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)

        # Normalize State using Config Stats
        state_mean = cfg.dataset_stats["observation.state"]["mean"].clone().detach().to(torch.float32)
        state_std = cfg.dataset_stats["observation.state"]["std"].clone().detach().to(torch.float32)
        normalized_state = (state_tensor - state_mean) / (state_std + 1e-6)

        # --- B. Vision (Dino embeddings)
        raw_rv = torch.tensor(ep_grp['obs']['robot0_robotview_image'][:K], dtype=torch.float32)
        raw_eye = torch.tensor(ep_grp['obs']['robot0_eye_in_hand_image'][:K], dtype=torch.float32)

        # --- C. Actions & Returns-To-Go ---
        actions = torch.tensor(ep_grp['actions'][:K], dtype=torch.float32)

        if cfg.use_time_based_rewards:
            dones = ep_grp['dones'][:]
            rewards = np.where(dones == 1, cfg.time_based_success_reward, -1.0)
        else:
            rewards = ep_grp['rewards'][:]
        # Calculate full trajectory RTG, then slice the first K elements
        rtg_array = np.cumsum(rewards[::-1])[::-1][:K].copy()
        rtg = torch.tensor(rtg_array, dtype=torch.float32).unsqueeze(-1)

    # 3. Package Context Dictionary
    context = {
        "eef_state": normalized_state,
        "img_robotview": raw_rv,
        "img_eye": raw_eye,
        "actions": actions,
        "rtg": rtg
    }

    return env, context


def build_color_varied_lift_with_distractor(
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    color_name: str = "blue",
    rgba: list = [0, 0, 1, 1],
    distractor_rgba = [1, 0, 0, 1],
    control_freq=20,
    horizon=300,
):
    """Builds a Lift task with two uniquely ocloured cubes - and asks to pick up one of them."""

    env = robosuite.make(
        "VariantLift",
        target_shape="cube",
        target_rgba=rgba,
        distractors=[
            {"shape": "cube", "rgba": distractor_rgba}
        ],
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        reward_shaping=True,
        camera_heights=512,
        camera_widths=512,
    )

    context = {
        "eef_state": None,
        "img_robotview": None,
        "img_eye": None,
        "actions": None,
        "rtg": None,
    }
    return env, context


def build_puck_lift(
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    control_freq=20,
    horizon=300
):
    """
    Tests if the robot can pick up a puck shaped object
    """

    env = robosuite.make(
        "VariantLift",
        target_shape="puck",
        target_rgba=[1, 0, 0, 1],         # Target: Red Puck
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        reward_shaping=True,
        camera_heights=512,
        camera_widths=512,
    )

    context = {
        "eef_state": None,
        "img_robotview": None,
        "img_eye": None,
        "actions": None,
        "rtg": None,
    }
    return env, context


def build_nut_assembly_square(
    hdf5_path: str,
    cfg,
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    control_freq=20,
    horizon=300
):
    """
    Builds the standard Robosuite NutAssemblySquare environment and extracts a random
    demonstration trajectory from the dataset to serve as the ICL context.
    """
    # 1. Instantiate the Environment
    env = robosuite.make(
        "NutAssemblySquare",
        robots="Panda",
        has_renderer=False,           # Headless for evaluation loops
        has_offscreen_renderer=True,  # Required to grab camera frames
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        camera_heights=512,
        camera_widths=512,
        reward_shaping=True,
    )

    # 2. Extract ICL Context from Dataset
    with h5py.File(hdf5_path, 'r') as f:
        base_grp = f['data'] if 'data' in f else f
        demos = [k for k in base_grp.keys() if k.startswith('demo_')]

        # Pick a random demonstration
        demo = random.choice(demos)
        ep_grp = base_grp[demo]

        ep_len = ep_grp['actions'].shape[0]
        # We extract from the beginning of the episode up to context_len
        K = min(cfg.context_len, ep_len)

        # --- A. Proprioception (State) ---
        obs_slice = {
            'robot0_eef_pos': ep_grp['obs']['robot0_eef_pos'][:K],
            'robot0_eef_quat': ep_grp['obs']['robot0_eef_quat'][:K],
            'robot0_gripper_qpos': ep_grp['obs']['robot0_gripper_qpos'][:K]
        }

        # Ensure extract_7dof_state is in scope
        state_features = extract_7dof_state(obs_slice)  # Shape: (K, 7)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)

        # Normalize State using Config Stats
        state_mean = cfg.dataset_stats["observation.state"]["mean"].clone().detach().to(torch.float32)
        state_std = cfg.dataset_stats["observation.state"]["std"].clone().detach().to(torch.float32)
        normalized_state = (state_tensor - state_mean) / (state_std + 1e-6)

        # --- B. Vision (Dino embeddings)
        raw_rv = torch.tensor(ep_grp['obs']['robot0_robotview_image'][:K], dtype=torch.float32)
        raw_eye = torch.tensor(ep_grp['obs']['robot0_eye_in_hand_image'][:K], dtype=torch.float32)

        # --- C. Actions & Returns-To-Go ---
        actions = torch.tensor(ep_grp['actions'][:K], dtype=torch.float32)

        if cfg.use_time_based_rewards:
            dones = ep_grp['dones'][:]
            rewards = np.where(dones == 1, cfg.time_based_success_reward, -1.0)
        else:
            rewards = ep_grp['rewards'][:]
        # Calculate full trajectory RTG, then slice the first K elements
        rtg_array = np.cumsum(rewards[::-1])[::-1][:K].copy()
        rtg = torch.tensor(rtg_array, dtype=torch.float32).unsqueeze(-1)

    # 3. Package Context Dictionary
    context = {
        "eef_state": normalized_state,
        "img_robotview": raw_rv,
        "img_eye": raw_eye,
        "actions": actions,
        "rtg": rtg
    }

    return env, context


def build_iiwa_lift(
    hdf5_path: str,
    cfg,
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    control_freq=20,
    horizon=300
):
    """
    Builds the standard Robosuite Lift environment with the IIWA robot and extracts a random
    demonstration trajectory from the dataset to serve as the ICL context.
    """
    # 1. Instantiate the Environment
    env = robosuite.make(
        "Lift",
        robots="IIWA",
        has_renderer=False,           # Headless for evaluation loops
        has_offscreen_renderer=True,  # Required to grab camera frames
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        camera_heights=512,
        camera_widths=512,
        reward_shaping=True,
    )

    # 2. Extract ICL Context from Dataset
    with h5py.File(hdf5_path, 'r') as f:
        base_grp = f['data'] if 'data' in f else f
        demos = [k for k in base_grp.keys() if k.startswith('demo_')]

        # Pick a random demonstration
        demo = random.choice(demos)
        ep_grp = base_grp[demo]

        ep_len = ep_grp['actions'].shape[0]
        # We extract from the beginning of the episode up to context_len
        K = min(cfg.context_len, ep_len)

        # --- A. Proprioception (State) ---
        obs_slice = {
            'robot0_eef_pos': ep_grp['obs']['robot0_eef_pos'][:K],
            'robot0_eef_quat': ep_grp['obs']['robot0_eef_quat'][:K],
            'robot0_gripper_qpos': ep_grp['obs']['robot0_gripper_qpos'][:K]
        }

        # Ensure extract_7dof_state is in scope
        state_features = extract_7dof_state(obs_slice)  # Shape: (K, 7)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)

        # Normalize State using Config Stats
        state_mean = cfg.dataset_stats["observation.state"]["mean"].clone().detach().to(torch.float32)
        state_std = cfg.dataset_stats["observation.state"]["std"].clone().detach().to(torch.float32)
        normalized_state = (state_tensor - state_mean) / (state_std + 1e-6)

        # --- B. Vision (Dino embeddings)
        raw_rv = torch.tensor(ep_grp['obs']['robot0_robotview_image'][:K], dtype=torch.float32)
        raw_eye = torch.tensor(ep_grp['obs']['robot0_eye_in_hand_image'][:K], dtype=torch.float32)

        # --- C. Actions & Returns-To-Go ---
        actions = torch.tensor(ep_grp['actions'][:K], dtype=torch.float32)

        if cfg.use_time_based_rewards:
            dones = ep_grp['dones'][:]
            rewards = np.where(dones == 1, cfg.time_based_success_reward, -1.0)
        else:
            rewards = ep_grp['rewards'][:]
        # Calculate full trajectory RTG, then slice the first K elements
        rtg_array = np.cumsum(rewards[::-1])[::-1][:K].copy()
        rtg = torch.tensor(rtg_array, dtype=torch.float32).unsqueeze(-1)

    # 3. Package Context Dictionary
    context = {
        "eef_state": normalized_state,
        "img_robotview": raw_rv,
        "img_eye": raw_eye,
        "actions": actions,
        "rtg": rtg
    }

    return env, context


def build_iiwa_stack(
    hdf5_path: str,
    cfg,
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    control_freq=20,
    horizon=300
):
    """
    Builds the standard Robosuite Stack environment with the IIWA robot and extracts a random
    demonstration trajectory from the dataset to serve as the ICL context.
    """
    # 1. Instantiate the Environment
    env = robosuite.make(
        "Stack",
        robots="IIWA",
        has_renderer=False,           # Headless for evaluation loops
        has_offscreen_renderer=True,  # Required to grab camera frames
        use_camera_obs=True,
        camera_names=camera_names,
        control_freq=control_freq,
        horizon=horizon,
        camera_heights=512,
        camera_widths=512,
        reward_shaping=True,
    )

    # 2. Extract ICL Context from Dataset
    with h5py.File(hdf5_path, 'r') as f:
        base_grp = f['data'] if 'data' in f else f
        demos = [k for k in base_grp.keys() if k.startswith('demo_')]

        # Pick a random demonstration
        demo = random.choice(demos)
        ep_grp = base_grp[demo]

        ep_len = ep_grp['actions'].shape[0]
        # We extract from the beginning of the episode up to context_len
        K = min(cfg.context_len, ep_len)

        # --- A. Proprioception (State) ---
        obs_slice = {
            'robot0_eef_pos': ep_grp['obs']['robot0_eef_pos'][:K],
            'robot0_eef_quat': ep_grp['obs']['robot0_eef_quat'][:K],
            'robot0_gripper_qpos': ep_grp['obs']['robot0_gripper_qpos'][:K]
        }

        # Ensure extract_7dof_state is in scope
        state_features = extract_7dof_state(obs_slice)  # Shape: (K, 7)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)

        # Normalize State using Config Stats
        state_mean = cfg.dataset_stats["observation.state"]["mean"].clone().detach().to(torch.float32)
        state_std = cfg.dataset_stats["observation.state"]["std"].clone().detach().to(torch.float32)
        normalized_state = (state_tensor - state_mean) / (state_std + 1e-6)

        # --- B. Vision (Dino embeddings)
        raw_rv = torch.tensor(ep_grp['obs']['robot0_robotview_image'][:K], dtype=torch.float32)
        raw_eye = torch.tensor(ep_grp['obs']['robot0_eye_in_hand_image'][:K], dtype=torch.float32)

        # --- C. Actions & Returns-To-Go ---
        actions = torch.tensor(ep_grp['actions'][:K], dtype=torch.float32)

        if cfg.use_time_based_rewards:
            dones = ep_grp['dones'][:]
            rewards = np.where(dones == 1, cfg.time_based_success_reward, -1.0)
        else:
            rewards = ep_grp['rewards'][:]
        # Calculate full trajectory RTG, then slice the first K elements
        rtg_array = np.cumsum(rewards[::-1])[::-1][:K].copy()
        rtg = torch.tensor(rtg_array, dtype=torch.float32).unsqueeze(-1)

    # 3. Package Context Dictionary
    context = {
        "eef_state": normalized_state,
        "img_robotview": raw_rv,
        "img_eye": raw_eye,
        "actions": actions,
        "rtg": rtg
    }

    return env, context