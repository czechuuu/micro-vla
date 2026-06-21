from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    model_id: str | None = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim: tuple[int] = (7,)
    obs_dim: tuple[int] = (7,)

    num_epochs: int = 1
    steps_per_epoch: int = 9000
    batch_size: int = 8

    context_len: int = 75       # How many timesteps of history of a single trajectory the model looks at - with ICL the model can attend to more than context len
    max_ep_len: int = 200         # Max episode length (needed for positional embeddings, Robosuite default is 500)
    hidden_dim: int = 256         # Transformer embedding dimension
    num_layers: int = 3           # Number of Transformer blocks
    nhead: int = 4                # Number of attention heads
    dropout: float = 0.1
    image_size: int = 512
    vision_embed_dim: int = 512
    target_return: float = 50.0
    use_reward_tokens: bool = False
    use_rtg: bool = False # Only valid if use_reward_tokens = True. Decides whether to use RTG or standard step-wise rewards.
    fuse_observations: bool = False # Whether we should fuse cameras and proprio into one latent token or let them be separate.
    use_time_based_rewards: bool = False
    time_based_success_reward: float = 100.0


    dataset_stats: dict = field(
        default_factory=lambda: {
            "observation.state": {
                "mean": torch.tensor([
                    -0.02250661411056904,
                    -0.0017509463337428896,
                    0.8927278452815375,
                    0.14772878979109655,
                    -0.09987207042179344,
                    0.028312488183443292,
                    0.06784597426917023
                ], dtype=torch.float32),
                "std": torch.tensor([
                    0.04077795730302498,
                    0.0156866066296251,
                    0.0650977811584224,
                    3.117419007186853,
                    0.05621472595042969,
                    0.23287363452579218,
                    0.014721579731309775
                ], dtype=torch.float32)
            }
        }
    )

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["device"] = str(d["device"])
        if "dataset_stats" in d and "observation.state" in d["dataset_stats"]:
            obs_state = d["dataset_stats"]["observation.state"]
            # Convert tensors to lists for JSON serialization
            if "mean" in obs_state and hasattr(obs_state["mean"], "tolist"):
                obs_state["mean"] = obs_state["mean"].tolist()
            if "std" in obs_state and hasattr(obs_state["std"], "tolist"):
                obs_state["std"] = obs_state["std"].tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        if "device" in d and isinstance(d["device"], str):
            d["device"] = torch.device(d["device"])
        if "action_dim" in d and isinstance(d["action_dim"], list):
            d["action_dim"] = tuple(d["action_dim"])
        if "obs_dim" in d and isinstance(d["obs_dim"], list):
            d["obs_dim"] = tuple(d["obs_dim"])
        if "dataset_stats" in d and "observation.state" in d["dataset_stats"]:
            obs_state = d["dataset_stats"]["observation.state"]
            if "mean" in obs_state and isinstance(obs_state["mean"], list):
                obs_state["mean"] = torch.tensor(obs_state["mean"], dtype=torch.float32)
            if "std" in obs_state and isinstance(obs_state["std"], list):
                obs_state["std"] = torch.tensor(obs_state["std"], dtype=torch.float32)
        return cls(**d)

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> "Config":
        import json
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)