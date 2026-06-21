import argparse
import sys
import os
import torch

# Ensure the root of the project is in the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.helpers import resolve_hdf5_path
from src.eval.policy import DecisionTransformerWrapper
from src.eval.eval_runner import BenchmarkSuite
from src.eval.tasks import (
    build_standard_lift,
    build_standard_stack,
    build_color_varied_lift,
    build_color_varied_lift_with_distractor,
    build_puck_lift
)

def main():
    parser = argparse.ArgumentParser(description="Run Micro-VLA evaluation from the command line.")
    
    # Required arguments
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to the trained PyTorch model checkpoint (.pt or .weights)."
    )

    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to the config.json file (if omitted, checks checkpoint directory)."
    )

    
    # Task evaluation arguments
    parser.add_argument(
        "--tasks", 
        nargs="+", 
        default=["Standard_Lift", "Red_Cube_Lift"],
        help="Tasks to evaluate (Standard_Lift, Standard_Stack, Red_Cube_Lift, Blue_Cube_Lift, Green_Cube_Lift, Blue_Cube_With_Distractor_Lift, Puck_Lift)."
    )
    parser.add_argument(
        "--lift-dataset", 
        type=str, 
        default="dino3-embeddings/block-lifting/ph.hdf5",
        help="HF or local HDF5 path to extract context demonstrations for Standard_Lift."
    )
    parser.add_argument(
        "--stack-dataset", 
        type=str, 
        default="dino3-embeddings/block-lifting/mh-better.hdf5",
        help="HF or local HDF5 path to extract context demonstrations for Standard_Stack."
    )
    
    # Evaluation run settings
    parser.add_argument("--num-episodes", type=int, default=8, help="Number of rollouts per evaluation task.")
    parser.add_argument("--horizon", type=int, default=100, help="Maximum timesteps per rollout episode.")
    parser.add_argument("--save-videos", action="store_true", default=True, help="Save evaluation execution videos.")
    parser.add_argument("--no-save-videos", action="store_false", dest="save_videos", help="Do not save execution videos.")
    parser.add_argument("--output-dir", type=str, default="./vla_evaluations", help="Directory where metrics and videos are stored.")
    parser.add_argument("--policy-name", type=str, default=None, help="Name of the evaluated policy (defaults to checkpoint name).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch computing device (cpu, cuda, cuda:0, etc.).")
    
    # Model configuration overrides (must match the trained model architecture)
    parser.add_argument("--context-len", type=int, default=75, help="Transformer context length.")
    parser.add_argument("--max-ep-len", type=int, default=200, help="Maximum episode length.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads.")
    
    # Model behavior flags
    parser.add_argument("--use-icl", action="store_true", default=False, help="Run model using In-Context Learning context demo.")
    parser.add_argument("--use-reward-tokens", action="store_true", default=False, help="Model trained with reward tokens.")
    parser.add_argument("--use-rtg", action="store_true", default=False, help="Model trained with returns-to-go.")
    parser.add_argument("--fuse-observations", action="store_true", default=False, help="Model trained with early fused observations.")
    
    parser.add_argument(
        "--from-file", "--from_file",
        type=str,
        default=None,
        help="Path to a JSON file containing argument overrides."
    )

    args = parser.parse_args()

    # Load arguments from file if specified
    if args.from_file is not None:
        if not os.path.exists(args.from_file):
            print(f"Error: JSON file '{args.from_file}' does not exist.", file=sys.stderr)
            sys.exit(1)
        try:
            import json
            with open(args.from_file, "r") as f:
                file_args = json.load(f)
            print(f"Loading CLI arguments from file: {args.from_file}")
            for k, v in file_args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    print(f"Warning: Unknown argument '{k}' in JSON file - ignoring.")
        except Exception as e:
            print(f"Error reading JSON file '{args.from_file}': {e}", file=sys.stderr)
            sys.exit(1)

    # 1. Validate checkpoint exists
    if args.checkpoint is None:
        print("Error: --checkpoint is required (must be specified on command line or via JSON file).", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint '{args.checkpoint}' does not exist.", file=sys.stderr)
        sys.exit(1)

        
    # 2. Resolve HDF5 dataset paths if standard tasks are evaluated
    lift_h5 = args.lift_dataset
    if "Standard_Lift" in args.tasks:
        if not os.path.exists(args.lift_dataset):
            try:
                lift_h5 = resolve_hdf5_path(args.lift_dataset)
            except Exception as e:
                print(f"Error: Could not resolve lift dataset '{args.lift_dataset}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            lift_h5 = os.path.abspath(args.lift_dataset)
            
    stack_h5 = args.stack_dataset
    if "Standard_Stack" in args.tasks:
        if not os.path.exists(args.stack_dataset):
            try:
                stack_h5 = resolve_hdf5_path(args.stack_dataset)
            except Exception as e:
                print(f"Error: Could not resolve stack dataset '{args.stack_dataset}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            stack_h5 = os.path.abspath(args.stack_dataset)
            
    # 3. Create model configuration object
    config_path = args.config
    if config_path is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        possible_config = os.path.join(checkpoint_dir, "config.json")
        if checkpoint_dir and os.path.exists(possible_config):
            config_path = possible_config
            print(f"Auto-detected config file at: {config_path}")

    if config_path is not None:
        if not os.path.exists(config_path):
            print(f"Error: Config file '{config_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
        print(f"Loading model architecture configuration from: {config_path}")
        cfg = Config.load(config_path)
        cfg.model_id = args.checkpoint
        cfg.device = torch.device(args.device)
    else:
        print("No config file specified or auto-detected. Using command-line defaults/overrides.")
        cfg = Config(
            model_id=args.checkpoint,
            device=torch.device(args.device),
            context_len=args.context_len,
            max_ep_len=args.max_ep_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            nhead=args.nhead,
            use_reward_tokens=args.use_reward_tokens,
            use_rtg=args.use_rtg,
            fuse_observations=args.fuse_observations
        )

    
    # 4. Map task names to their respective builders
    all_possible_tasks = {
        "Standard_Lift": lambda: build_standard_lift(cfg=cfg, hdf5_path=lift_h5, horizon=args.horizon),
        "Standard_Stack": lambda: build_standard_stack(cfg=cfg, hdf5_path=stack_h5, horizon=args.horizon),
        "Red_Cube_Lift": lambda: build_color_varied_lift(horizon=args.horizon, rgba=[1, 0, 0, 1]),
        "Blue_Cube_Lift": lambda: build_color_varied_lift(horizon=args.horizon, rgba=[0, 0, 1, 1]),
        "Green_Cube_Lift": lambda: build_color_varied_lift(horizon=args.horizon, rgba=[0, 1, 0, 1]),
        "Blue_Cube_With_Distractor_Lift": lambda: build_color_varied_lift_with_distractor(horizon=args.horizon, rgba=[0, 0, 1, 1], distractor_rgba=[1, 0, 0, 1]),
        "Puck_Lift": lambda: build_puck_lift(horizon=args.horizon),
    }
    
    # Filter requested tasks
    tasks_to_evaluate = {}
    for task_name in args.tasks:
        if task_name in all_possible_tasks:
            tasks_to_evaluate[task_name] = all_possible_tasks[task_name]
        else:
            print(f"Warning: Task '{task_name}' is not recognized and will be skipped.", file=sys.stderr)
            
    if not tasks_to_evaluate:
        print("Error: No valid tasks selected for evaluation.", file=sys.stderr)
        sys.exit(1)
        
    # 5. Determine policy name
    policy_name = args.policy_name
    if policy_name is None:
        policy_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        
    # 6. Instantiate wrapper and start evaluation
    print(f"Instantiating policy wrapper for checkpoint: {args.checkpoint} (using device: {args.device})")
    policy = DecisionTransformerWrapper(cfg, uses_icl=args.use_icl)
    
    print(f"Initializing benchmark suite on tasks: {list(tasks_to_evaluate.keys())}")
    benchmark = BenchmarkSuite(
        tasks_dict=tasks_to_evaluate,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        save_videos=args.save_videos
    )
    
    print("Starting evaluation rollouts...")
    results = benchmark.evaluate_policy(policy, policy_name=policy_name)
    
    print("\nEvaluation results:")
    for task_name, metrics in results.items():
        print(f"  {task_name}: Success Rate = {metrics.get('success_rate', 0.0):.1%}")

if __name__ == "__main__":
    main()
