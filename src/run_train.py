import argparse
import sys
import os
import torch


# Ensure the root of the project is in the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.helpers import resolve_hdf5_path
from src.train.training import train_micro_vla

def main():
    parser = argparse.ArgumentParser(description="Run Micro-VLA training from the command line.")
    
    # Dataset arguments
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["dino3-embeddings/block-lifting/panda/ph.hdf5"],
        help="List of HF dataset paths or local paths to HDF5 files."
    )
    parser.add_argument(
        "--weights", 
        nargs="+", 
        type=float, 
        default=None,
        help="List of weights for the sample weights sampler (must match the number of datasets)."
    )
    
    # Training configuration arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=9000, help="Steps per epoch.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--checkpoint-name", type=str, default="micro_vla_weights.pt", help="Filename of the saved checkpoint.")
    
    # Model configuration overrides
    parser.add_argument("--context-len", type=int, default=75, help="Transformer context length.")
    parser.add_argument("--max-ep-len", type=int, default=200, help="Maximum episode length.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers.")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    
    # Modality and task flags
    parser.add_argument("--use-icl", action="store_true", default=True, help="Train in ICL mode (context + query).")
    parser.add_argument("--no-icl", action="store_false", dest="use_icl", help="Train in standard mode (query only).")
    parser.add_argument("--use-reward-tokens", action="store_true", default=False, help="Incorporate reward tokens in sequence.")
    parser.add_argument("--use-rtg", action="store_true", default=False, help="Use returns-to-go instead of step rewards.")
    parser.add_argument("--use-time-based-rewards", action="store_true", default=False, help="Use time-based rewards (-1 step reward, time_based_success_reward success).")
    parser.add_argument("--time-based-success-reward", type=float, default=100.0, help="Reward value on task success for time-based rewards.")
    parser.add_argument("--target-return", type=float, default=50.0, help="Target return value for RTG conditioning.")
    parser.add_argument("--fuse-observations", action="store_true", default=False, help="Fuse all observation modalities early.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch computing device.")
    parser.add_argument(
        "--from-file", "--from_file",
        type=str,
        default=None,
        help="Path to a JSON file containing argument overrides."
    )
    parser.add_argument(
        "--from-pretrained", "--from_pretrained",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint to finetune from."
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

    
    # 1. Resolve dataset paths
    resolved_paths = []
    for dataset in args.datasets:
        if os.path.exists(dataset):
            resolved_paths.append(os.path.abspath(dataset))
        else:
            # Fallback to downloading/resolving from Hugging Face
            try:
                resolved_paths.append(resolve_hdf5_path(dataset))
            except Exception as e:
                print(f"Error: Could not find local file or resolve HF dataset '{dataset}': {e}", file=sys.stderr)
                sys.exit(1)
                
    print(f"Resolved dataset paths: {resolved_paths}")
    
    # 2. Process sample weights
    task_weights = None
    if args.weights is not None:
        if len(args.weights) != len(args.datasets):
            print(f"Error: Number of weights ({len(args.weights)}) does not match number of datasets ({len(args.datasets)}).", file=sys.stderr)
            sys.exit(1)
        task_weights = {path: w for path, w in zip(resolved_paths, args.weights)}
        print(f"Using manual task weights: {task_weights}")
    else:
        print("Using automatic task balancing (inverse frequency).")
        
    # 3. Build training config
    config = Config(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        context_len=args.context_len,
        max_ep_len=args.max_ep_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        use_reward_tokens=args.use_reward_tokens,
        use_rtg=args.use_rtg,
        fuse_observations=args.fuse_observations,
        use_time_based_rewards=args.use_time_based_rewards,
        time_based_success_reward=args.time_based_success_reward,
        target_return=args.target_return,
        from_pretrained=args.from_pretrained
    )
    
    # 4. Trigger training
    train_micro_vla(
        config=config,
        hdf5_paths=resolved_paths,
        use_icl=args.use_icl,
        checkpoint_name=args.checkpoint_name,
        task_weights=task_weights
    )

if __name__ == "__main__":
    main()
