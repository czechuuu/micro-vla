import os
import json
import h5py
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm


from src.config import Config
from src.transformer import MicroVLADecisionTransformer
from src.train.dataset import MicroVLADataset

def train_micro_vla(
    config: Config,
    hdf5_paths: list[str],
    use_icl: bool = True,
    checkpoint_name: str = "micro_vla_weights.pt",
    task_weights: dict[str, float] | None = None
):
    # Create run directory and dump config
    run_dir = checkpoint_name
    for ext in ['.pt', '.pth', '.weights']:
        if checkpoint_name.endswith(ext):
            run_dir = checkpoint_name[:-len(ext)]
            break
    os.makedirs(run_dir, exist_ok=True)
    config.save(os.path.join(run_dir, "config.json"))
    print(f"Created run directory '{run_dir}' and dumped configuration to '{run_dir}/config.json'")

    # -----------------------------------------
    # 1. Create Strict Episode-Level Splits
    # -----------------------------------------
    train_splits = {}
    test_splits = {}

    for path in hdf5_paths:
        with h5py.File(path, 'r') as f:
            base_grp = f['data'] if 'data' in f else f
            all_demos = [k for k in base_grp.keys() if k.startswith('demo_')]

        # Shuffle in-place and calculate the 80/20 index
        all_demos = list(all_demos)
        random.shuffle(all_demos)
        split_idx = int(0.9 * len(all_demos))

        train_splits[path] = all_demos[:split_idx]
        test_splits[path] = all_demos[split_idx:]

    # -----------------------------------------
    # 2. Instantiate Datasets & DataLoaders
    # -----------------------------------------
    train_dataset = MicroVLADataset(hdf5_paths, config, use_icl=use_icl, task_to_demos=train_splits)
    test_dataset = MicroVLADataset(hdf5_paths, config, use_icl=use_icl, task_to_demos=test_splits)

    # Create Sample Weights for the Sampler
    sample_weights = []
    if task_weights is not None:
        # Strategy A: Use explicit weights provided by you
        for task_idx, _, _, _ in train_dataset.valid_starts:
            path = hdf5_paths[task_idx]
            weight = task_weights.get(path, 1.0)
            sample_weights.append(weight)
    else:
        # Strategy B: Automatic Inverse Frequency (Perfectly balances tasks)
        # First, count how many valid starting indices belong to each task
        task_counts = {i: 0 for i in range(len(hdf5_paths))}
        for task_idx, _, _, _ in train_dataset.valid_starts:
            task_counts[task_idx] += 1

        print(f"TASK COUNTS: {task_counts}")

        # Then, assign a weight that is inversely proportional to its frequency
        for task_idx, _, _, _ in train_dataset.valid_starts:
            # E.g., if task 0 has 10,000 samples, weight is 1/10000.
            weight = 1.0 / (task_counts[task_idx] + 1e-8)
            sample_weights.append(weight)

    # Instantiate the PyTorch Sampler
    if config.steps_per_epoch < len(sample_weights):
        print("NOTE: will omit some trajectories to respect the config steps_per_epoch. Review if this is intended or if you are accidentally truncating your epochs")
    num_samples = min(config.steps_per_epoch, len(sample_weights))
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples, # Number of steps per epoch can be configured.
        replacement=True         # Must be True to allow oversampling smaller tasks
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # -----------------------------------------
    # 3. Initialize Model & Optimizer
    # -----------------------------------------
    model = MicroVLADecisionTransformer(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    mode_str = "ICL" if use_icl else "Baseline"
    print(f"Starting {mode_str} training on {config.device}...")
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # -----------------------------------------
    # 4. Training & Validation Loop
    # -----------------------------------------
    losses_history = []
    for epoch in range(config.num_epochs):

        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")

        for batch_idx, batch in enumerate(progress_bar):
            eef_state = batch["eef_state"].to(config.device)
            dino_robotview = batch["dino_robotview"].to(config.device)
            dino_eye = batch["dino_eye"].to(config.device)
            actions = batch["actions"].to(config.device)
            rewards = batch["rewards"].to(config.device)
            timesteps = batch["timesteps"].to(config.device)

            # The unified mask automatically handles ICL context filtering vs baseline padding
            loss_mask = batch["loss_mask"].to(config.device)

            # Forward Pass
            predicted_actions = model(
                eef_state=eef_state,
                rewards=rewards,
                timesteps=timesteps,
                actions=actions,
                dino_robotview=dino_robotview,
                dino_eye=dino_eye
            )

            # Masked Loss Calculation
            raw_loss = F.mse_loss(predicted_actions, actions, reduction='none')
            loss_mask = loss_mask.unsqueeze(-1)
            action_dim = raw_loss.shape[-1]
            masked_loss = (raw_loss * loss_mask).sum() / (loss_mask.sum() * action_dim + 1e-8)

            # Backpropagation
            optimizer.zero_grad()
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()

            # Logging
            total_train_loss += masked_loss.item()
            progress_bar.set_postfix({'loss': f"{masked_loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # --- EVALUATION PHASE ---
        model.eval()
        total_test_loss = 0.0

        # Initialize tracking dictionaries for per-task losses
        num_tasks = len(hdf5_paths)
        task_test_losses = {i: 0.0 for i in range(num_tasks)}
        task_test_counts = {i: 0 for i in range(num_tasks)}

        with torch.no_grad():
            for batch in test_loader:
                eef_state = batch["eef_state"].to(config.device)
                dino_robotview = batch["dino_robotview"].to(config.device)
                dino_eye = batch["dino_eye"].to(config.device)
                actions = batch["actions"].to(config.device)
                rewards = batch["rewards"].to(config.device)
                timesteps = batch["timesteps"].to(config.device)
                loss_mask = batch["loss_mask"].to(config.device)
                task_indices = batch["task_idx"].tolist() # Convert to list for easy iteration

                predicted_actions = model(
                    eef_state=eef_state,
                    rewards=rewards,
                    timesteps=timesteps,
                    actions=actions,
                    dino_robotview=dino_robotview,
                    dino_eye=dino_eye
                )

                # Raw loss shape: [Batch, Time, Action_Dim]
                raw_loss = F.mse_loss(predicted_actions, actions, reduction='none')
                loss_mask_expanded = loss_mask.unsqueeze(-1)

                # 1. Calculate global batch loss (same as before)
                action_dim = raw_loss.shape[-1]
                global_masked_loss = (raw_loss * loss_mask_expanded).sum() / (loss_mask_expanded.sum() * action_dim + 1e-8)
                total_test_loss += global_masked_loss.item()

                # 2. Calculate loss per trajectory in the batch
                # Sum across Time (dim=1) and Action (dim=2) to get shape: [Batch]
                item_masked_loss = (raw_loss * loss_mask_expanded).sum(dim=(1, 2))
                item_valid_elements = loss_mask.sum(dim=1) * action_dim + 1e-8
                item_loss = item_masked_loss / item_valid_elements # Shape: [Batch]

                # 3. Route the loss to the correct task tally
                for b_idx in range(len(task_indices)):
                    t_idx = task_indices[b_idx]
                    task_test_losses[t_idx] += item_loss[b_idx].item()
                    task_test_counts[t_idx] += 1

        avg_test_loss = total_test_loss / len(test_loader)

        # --- LOGGING ---
        print(f"\n--- Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Global Test Loss: {avg_test_loss:.4f} ---")

        # Print the breakdown per task
        for i in range(num_tasks):
            if task_test_counts[i] > 0:
                avg_task_loss = task_test_losses[i] / task_test_counts[i]
                # Extracting just the filename for cleaner logging
                task_name = hdf5_paths[i].split("/")[-1]
                print(f"    Task {i} ({task_name}): {avg_task_loss:.4f}")

        # Save epoch checkpoint
        epoch_checkpoint = os.path.join(run_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), epoch_checkpoint)
        print(f"Saved epoch checkpoint: {epoch_checkpoint}")

        # Save losses history to JSON
        epoch_task_losses = {}
        for i in range(num_tasks):
            if task_test_counts[i] > 0:
                avg_task_loss = task_test_losses[i] / task_test_counts[i]
                task_name = hdf5_paths[i].split("/")[-1]
                epoch_task_losses[task_name] = avg_task_loss

        losses_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_test_loss,
            "task_val_losses": epoch_task_losses
        })

        losses_file = os.path.join(run_dir, "losses.json")
        with open(losses_file, "w") as f:
            json.dump(losses_history, f, indent=4)
        print(f"Saved losses history to {losses_file}")

    # -----------------------------------------
    # 5. Save Model Checkpoint
    # -----------------------------------------
    final_checkpoint = os.path.join(run_dir, os.path.basename(checkpoint_name))
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Training complete. Final epoch model saved as: {final_checkpoint}")

    return model
