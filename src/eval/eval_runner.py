"""
Example usage:
# 1. Define the Benchmark Suite (The Tasks)
tasks_to_evaluate = {
    "Standard_Lift": lambda: build_standard_lift(cfg=Config(), hdf5_path=hdf5_paths[0], horizon=100),
    #"Standard_Stack": lambda: build_standard_stack(cfg=Config(), hdf5_path=hdf5_paths[1], horizon=150),
    "Red_Cube_Lift": lambda: build_color_varied_lift(horizon=100, rgba=[1,0,0,1]),
    #"Blue_Cube_Lift": lambda: build_color_varied_lift(horizon=100, rgba=[0,0,1,1]),
}

# 2. Initialize the Benchmark Engine
benchmark = BenchmarkSuite( 
    tasks_dict=tasks_to_evaluate,
    output_dir="./vla_evaluations",
    num_episodes=8,
    save_videos=True
)

# 3. Define the Models to Evaluate
# In practice, your teammate might populate this list dynamically from a checkpoints folder
policies_to_test = {
    # "ICL_TRAINED": lambda: DecisionTransformerWrapper(
    #     Config(model_id="./lift_ph.pt", ), uses_icl=False),
    "NON_ICL_TRAINED": lambda: DecisionTransformerWrapper(
        Config(model_id="./lift_ph_no_icl.pt", ), uses_icl=False),
    # "ICL": lambda: DecisionTransformerICLWrapper(
    #     Config(model_id="./icl_weights.pt", context_len=100)
    # )
}

# 4. Run the Evaluations
all_results = {}

for policy_name, policy_ctor in policies_to_test.items():
    # Instantiate the brain
    policy = policy_ctor()

    # Test the brain against the suite
    results = benchmark.evaluate_policy(policy, policy_name=policy_name)
    all_results[policy_name] = results

    # Optional: Delete the policy from memory to prevent VRAM overflow when loading the next one
    del policy

print("\nAll models evaluated successfully!")

"""

import os
import json
import imageio
import numpy as np
from datetime import datetime

class BenchmarkSuite:
    """
    A standardized suite of tasks to evaluate ICL-conditioned policies.
    """
    def __init__(self, tasks_dict, output_dir="eval_results", num_episodes=10, save_videos=True, success_hold_steps=8):
        self.tasks_dict = tasks_dict
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.save_videos = save_videos
        # This now acts as our recording buffer rather than a strict success criteria
        self.success_hold_steps = success_hold_steps

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"benchmark_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

    def _run_episode(self, env, policy, context, episode_idx):
        """Runs a single episode conditioned on a demonstration context."""
        obs = env.reset()
        policy.reset()

        frames = []
        success = False
        slipped = False
        post_success_frames = 0

        for step in range(env.horizon):
            action = policy.get_action(obs, context)
            obs, reward, done, info = env.step(action)

            is_current_success = env._check_success()

            if hasattr(policy, "register_reward"):
                policy.register_reward(reward, is_success=is_current_success)

            if self.save_videos:
                frames.append(np.flipud(obs["robot0_robotview_image"]))

            # Trigger success on a single frame
            if not success and is_current_success:
                success = True
                # \n added to avoid overwriting the \r progress bar from the parent loop
                print(f"\n      -> Success reached at step {step}! Recording {self.success_hold_steps} trailing frames...")

            # If we've hit success at least once, start tracking buffer and slippage
            if success:
                if not is_current_success and not slipped:
                    slipped = True
                    print(f"      -> Warning: Success condition slipped at step {step}!")

                post_success_frames += 1
                if post_success_frames >= self.success_hold_steps:
                    break

        return success, frames, slipped

    def _evaluate_task(self, policy, task_name, task_builder, policy_dir):
        """Evaluates a policy on a single task and saves the outputs."""
        print(f"\n  --- Task: {task_name} ---")

        # task_builder must now return the environment and the context dictionary
        env, context = task_builder()

        task_dir = os.path.join(policy_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        successes = 0
        slips = 0

        for ep_idx in range(self.num_episodes):
            print(f"    [Ep {ep_idx+1}/{self.num_episodes}] Rolling out with ICL Context...", end="\r")
            
            # Unpack the newly added slipped variable
            success, frames, slipped = self._run_episode(env, policy, context, ep_idx)

            print(f"    [Ep {ep_idx+1}/{self.num_episodes}] Status:", end="")
            if success:
                successes += 1
                if slipped:
                    slips += 1
                    print("      -> Success (but slipped).")
                else:
                    print("      -> Success.")
            else:
                print("      -> Failed.")

            if self.save_videos:
                # Add "slipped" to the filename for easy debugging later
                status = "success" if success else "fail"
                if success and slipped:
                    status = "success_slipped"
                    
                video_path = os.path.join(task_dir, f"ep_{ep_idx:02d}_{status}.mp4")
                writer = imageio.get_writer(video_path, fps=env.control_freq)
                for f in frames:
                    writer.append_data(f)
                writer.close()

        env.close()
        
        success_rate = successes / self.num_episodes
        slip_rate = (slips / successes) if successes > 0 else 0.0
        
        print(f"  --- {task_name} | Success Rate: {success_rate:.0%} | Slip Rate (of successes): {slip_rate:.0%} ---")

        return {
            "success_rate": success_rate,
            "total_episodes": self.num_episodes,
            "successful_episodes": successes,
            "slipped_episodes": slips,
            "slip_rate_among_successes": slip_rate
        }

    def evaluate_policy(self, policy, policy_name: str):
        """Runs a specific policy through the entire task suite."""
        print(f"\n{'='*50}\nEvaluating Policy: {policy_name}\n{'='*50}")

        policy_dir = os.path.join(self.run_dir, policy_name)
        os.makedirs(policy_dir, exist_ok=True)

        policy_results = {}
        for task_name, task_builder in self.tasks_dict.items():
            task_metrics = self._evaluate_task(policy, task_name, task_builder, policy_dir)
            policy_results[task_name] = task_metrics

        report_path = os.path.join(policy_dir, f"{policy_name}_report.json")
        with open(report_path, "w") as f:
            json.dump(policy_results, f, indent=4)

        return policy_results
    
