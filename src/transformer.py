import torch
import torch.nn as nn

class MicroVLADecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.dino_dim = 384  # DINOv3 ViT-Small base feature dimension

        # ---------------------------------------------------------
        # 1. Base Embeddings (Action, Reward, Timestep)
        # ---------------------------------------------------------
        if self.config.use_reward_tokens:
            self.embed_reward = nn.Linear(1, self.hidden_dim)
            
        self.embed_action = nn.Linear(config.action_dim[0], self.hidden_dim)
        self.embed_timestep = nn.Embedding(config.max_ep_len, self.hidden_dim)

        # ---------------------------------------------------------
        # 2. Modality Embeddings (Conditional based on Config)
        # ---------------------------------------------------------
        if self.config.fuse_observations:
            # EARLY FUSION: Concatenate all modalities into one large vector, 
            # then process through a small MLP down to hidden_dim.
            concat_dim = (2 * self.dino_dim) + config.obs_dim[0]
            self.embed_obs_fused = nn.Sequential(
                nn.Linear(concat_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        else:
            # LATE FUSION: Project each modality independently to the hidden_dim
            self.embed_rv = nn.Linear(self.dino_dim, self.hidden_dim)
            self.embed_eye = nn.Linear(self.dino_dim, self.hidden_dim)
            self.embed_proprio = nn.Linear(config.obs_dim[0], self.hidden_dim)

        # ---------------------------------------------------------
        # 3. Core Transformer Architecture
        # ---------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN variant for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        # ---------------------------------------------------------
        # 4. Policy Prediction Head
        # ---------------------------------------------------------
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, config.action_dim[0])
        )

    def forward(
        self,
        eef_state: torch.Tensor,
        timesteps: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor = None,
        dino_robotview: torch.Tensor = None,
        dino_eye: torch.Tensor = None,
    ) -> torch.Tensor:

        B, K, _ = eef_state.shape

        # Step 1: Base & Timestep Embeddings
        time_emb = self.embed_timestep(timesteps)       # Shape: (B, K, D)
        action_emb = self.embed_action(actions) + time_emb 

        if self.config.use_reward_tokens:
            assert rewards is not None, "rewards tensor must be provided when use_reward_tokens is True."
            reward_emb = self.embed_reward(rewards) + time_emb

        # Step 2: Modality Processing & Interleaving
        if self.config.fuse_observations:
            # --- EARLY FUSION MODE (Concatenated Tokens) ---
            obs_concat = torch.cat([dino_eye, dino_robotview, eef_state], dim=-1)
            obs_emb = self.embed_obs_fused(obs_concat) + time_emb

            if self.config.use_reward_tokens:
                # Sequence: [R, O_fused, A]
                stacked_inputs = torch.stack((reward_emb, obs_emb, action_emb), dim=2)
                seq_len_per_timestep = 3
            else:
                # Sequence: [O_fused, A]
                stacked_inputs = torch.stack((obs_emb, action_emb), dim=2)
                seq_len_per_timestep = 2
                
            sequence_inputs = stacked_inputs.reshape(B, seq_len_per_timestep * K, self.hidden_dim)

        else:
            # --- LATE FUSION MODE (Separate Tokens) ---
            eye_emb = self.embed_eye(dino_eye) + time_emb
            rv_emb = self.embed_rv(dino_robotview) + time_emb
            proprio_emb = self.embed_proprio(eef_state) + time_emb

            if self.config.use_reward_tokens:
                # Sequence: [R, Eye, RV, Proprio, A]
                stacked_inputs = torch.stack((reward_emb, eye_emb, rv_emb, proprio_emb, action_emb), dim=2)
                seq_len_per_timestep = 5
            else:
                # Sequence: [Eye, RV, Proprio, A]
                stacked_inputs = torch.stack((eye_emb, rv_emb, proprio_emb, action_emb), dim=2)
                seq_len_per_timestep = 4

            sequence_inputs = stacked_inputs.reshape(B, seq_len_per_timestep * K, self.hidden_dim)

        # Step 3: Generate Dynamic Causal Masking
        total_seq_len = seq_len_per_timestep * K
        mask = torch.nn.Transformer.generate_square_subsequent_mask(total_seq_len, device=eef_state.device)

        # Step 4: Pass through the core Transformer Layers
        transformer_outputs = self.transformer(sequence_inputs, mask=mask, is_causal=True)

        # Step 5: Extract the Correct Tokens for Action Prediction
        # The model must predict the action from the token IMMEDIATELY PRECEDING the action token
        # so it can causally attend to all current timestep modalities.
        if self.config.fuse_observations:
            if self.config.use_reward_tokens:
                # [R, O, A] -> O is at index 1
                obs_outputs = transformer_outputs[:, 1::seq_len_per_timestep, :]
            else:
                # [O, A] -> O is at index 0
                obs_outputs = transformer_outputs[:, 0::seq_len_per_timestep, :]
        else:
            if self.config.use_reward_tokens:
                # [R, O_eye, O_rv, O_proprio, A] -> O_proprio is at index 3
                obs_outputs = transformer_outputs[:, 3::seq_len_per_timestep, :]
            else:
                # [O_eye, O_rv, O_proprio, A] -> O_proprio is at index 2
                obs_outputs = transformer_outputs[:, 2::seq_len_per_timestep, :]

        # Predict continuous actions: Shape (B, K, action_dim)
        predicted_actions = self.predict_action(obs_outputs)

        return predicted_actions