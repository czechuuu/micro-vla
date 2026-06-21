import abc
import numpy as np
import torch

from src.eval.dino import DinoV3Wrapper
from src.transformer import MicroVLADecisionTransformer
from src.helpers import extract_7dof_state



class BasePolicy(abc.ABC):
    """
    Abstract base class defining the contract for any policy being evaluated.
    The Evaluation Engine will only interact with these methods.
    """

    @abc.abstractmethod
    def reset(self):
        """
        Clears any internal memory, action-chunking queues, or hidden states
        at the start of a new episode.
        """
        pass

    @abc.abstractmethod
    def get_action(self, obs: dict, context: dict) -> np.ndarray:
        """
        Takes the raw Robosuite observation dictionary and an ICL context dictionary,
        and returns a fully formatted action array ready for the environment.
        """
        pass

    def register_reward(self, reward: float):
        """
        Register a reward from the environment to track returns for RTG calculation.
        """
        pass



class DecisionTransformerWrapper(BasePolicy):
    """
    A unified evaluation wrapper for the MicroVLADecisionTransformer.
    Supports both standard autoregressive inference and In-Context Learning (ICL)
    while embedding raw images on the fly via DINOv3 to conserve VRAM.
    """
    def __init__(self, cfg, uses_icl: bool = False):
        self.cfg = cfg
        self.device = cfg.device
        self.uses_icl = uses_icl
        self.current_rtg = cfg.target_return

        # Safely extract the action dim
        self.action_dim = cfg.action_dim[0] if isinstance(cfg.action_dim, (tuple, list)) else cfg.action_dim

        # Extract and cache the normalization stats locally
        self.state_mean = self.cfg.dataset_stats["observation.state"]["mean"].clone().detach().to(torch.float32).to(self.device)
        self.state_std = self.cfg.dataset_stats["observation.state"]["std"].clone().detach().to(torch.float32).to(self.device)

        # 1. Instantiate the Vision Encoder
        print("Loading DINOv3 Vision Encoder...")
        self.vision_encoder = DinoV3Wrapper(cfg)

        # 2. Instantiate the Decision Transformer
        mode_str = "ICL" if self.uses_icl else "Standard"
        print(f"Loading {mode_str} Decision Transformer from checkpoint: {cfg.model_id}")

        # use_precomputed_vision MUST be True since we are passing DINO embeddings
        self.model = MicroVLADecisionTransformer(cfg)

        state_dict = torch.load(cfg.model_id, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        self.reset()

    def reset(self):
        """Clears all rolling sequence buffers at the start of a new episode."""
        self.state_history = []
        self.action_history = []
        self.rtg_history = []
        self.img_rv_history = []
        self.img_eye_history = []
        self.current_timestep = 0
        self.current_rtg = self.cfg.target_return

    def register_reward(self, reward: float):
        self.current_rtg -= reward

    @torch.no_grad()
    def get_action(self, obs: dict, context: dict = None) -> np.ndarray:
        # --- 1. Process and Embed Live Images ---
        raw_rv = np.flipud(obs["robot0_robotview_image"]).copy()
        rv_embedding = self.vision_encoder(images=raw_rv)
        torch_rv_emb = torch.from_numpy(rv_embedding).squeeze(0).float().to(self.device)
        self.img_rv_history.append(torch_rv_emb)

        raw_eye = np.flipud(obs["robot0_eye_in_hand_image"]).copy()
        eye_embedding = self.vision_encoder(images=raw_eye)
        torch_eye_emb = torch.from_numpy(eye_embedding).squeeze(0).float().to(self.device)
        self.img_eye_history.append(torch_eye_emb)

        # --- 2. Process Proprioception & RTG ---
        proprio = extract_7dof_state(obs)
        state_tensor = torch.tensor(proprio, dtype=torch.float32, device=self.device)
        normalized_state = (state_tensor - self.state_mean) / (self.state_std + 1e-6)

        self.state_history.append(normalized_state)
        self.rtg_history.append(torch.tensor([self.current_rtg], dtype=torch.float32, device=self.device))

        # --- 3. Sliding Window on Query ---
        cap = self.cfg.context_len

        query_states_valid = torch.stack(self.state_history[-cap:])
        query_rtgs_valid = torch.stack(self.rtg_history[-cap:])
        query_img_rvs_valid = torch.stack(self.img_rv_history[-cap:])
        query_img_eyes_valid = torch.stack(self.img_eye_history[-cap:])

        # Setup Query Actions (Shifted with causal placeholder)
        dummy_action = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        if len(self.action_history) == 0:
            query_actions_valid = dummy_action.unsqueeze(0)
        else:
            query_actions_valid = torch.stack(self.action_history[-(cap-1):] + [dummy_action])

        # --- 4. Modality Assembly (ICL vs Standard) ---
        if self.uses_icl and context is not None:
            # Note: `context["img_robotview"]` and `context["img_eye"]` must be pre-embedded
            states = torch.cat([context["eef_state"].to(self.device), query_states_valid], dim=0).unsqueeze(0)
            rtgs = torch.cat([context["rtg"].to(self.device), query_rtgs_valid], dim=0).unsqueeze(0)
            actions = torch.cat([context["actions"].to(self.device), query_actions_valid], dim=0).unsqueeze(0)
            img_rvs = torch.cat([context["img_robotview"].to(self.device), query_img_rvs_valid], dim=0).unsqueeze(0)
            img_eyes = torch.cat([context["img_eye"].to(self.device), query_img_eyes_valid], dim=0).unsqueeze(0)

            # Align Timesteps to match ICL dataset (both start at 0)
            ctx_K = context["eef_state"].shape[0]
            qry_K = query_states_valid.shape[0]
            ctx_time = torch.arange(0, ctx_K, dtype=torch.long, device=self.device)
            qry_time = torch.arange(0, qry_K, dtype=torch.long, device=self.device)
            timesteps = torch.cat([ctx_time, qry_time], dim=0).unsqueeze(0)

        else:
            # Standard Autoregressive Forward
            states = query_states_valid.unsqueeze(0)
            rtgs = query_rtgs_valid.unsqueeze(0)
            actions = query_actions_valid.unsqueeze(0)
            img_rvs = query_img_rvs_valid.unsqueeze(0)
            img_eyes = query_img_eyes_valid.unsqueeze(0)

            # Standard rolling timesteps
            qry_K = query_states_valid.shape[0]
            timesteps = torch.arange(0, qry_K, dtype=torch.long, device=self.device).unsqueeze(0)

        # --- 5. Forward Pass ---
        predicted_actions = self.model(
            eef_state=states,
            rewards=rtgs,
            timesteps=timesteps,
            actions=actions,
            dino_robotview=img_rvs,
            dino_eye=img_eyes
        )

        # --- 6. Extract Action and Cache ---
        action_to_take = predicted_actions[0, -1, :].cpu().numpy()

        self.action_history.append(torch.from_numpy(action_to_take).to(self.device))
        self.current_timestep += 1

        return action_to_take