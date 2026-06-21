import matplotlib.pyplot as plt
import numpy as np

# Data grouped by epoch for the 3 runs
epochs = [15, 16, 17, 18, 19]

# Stack data from the 3 runs
stack_data = {
    15: [25.0, 0.0, 12.5],
    16: [0.0, 12.5, 12.5],
    17: [0.0, 0.0, 0.0],
    18: [0.0, 12.5, 0.0],
    19: [25.0, 12.5, 0.0] 
}

# Lift data from the 3 runs
lift_data = {
    15: [100.0, 37.5, 75.0],
    16: [87.5, 75.0, 37.5],
    17: [100.0, 87.5, 75.0],
    18: [37.5, 37.5, 50.0],
    19: [100.0, 87.5, 87.5]
}

# Calculate statistics for Lift
lift_means = [np.mean(lift_data[e]) for e in epochs]
lift_mins = [np.min(lift_data[e]) for e in epochs]
lift_maxs = [np.max(lift_data[e]) for e in epochs]

# Calculate statistics for Stack
stack_means = [np.mean(stack_data[e]) for e in epochs]
stack_mins = [np.min(stack_data[e]) for e in epochs]
stack_maxs = [np.max(stack_data[e]) for e in epochs]

# Plotting setup - 1 row, 2 columns side-by-side subgraphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 1. Plot Lift Success Rate (Left Subgraph)
ax1.plot(epochs, lift_means, label='Mean Success Rate', color='blue', marker='o', linewidth=2)
ax1.fill_between(epochs, lift_mins, lift_maxs, color='blue', alpha=0.15, label='Min-Max Range')
ax1.plot(epochs, lift_mins, label='Min Success Rate', color='red', linestyle='--', marker='v')
ax1.plot(epochs, lift_maxs, label='Max Success Rate', color='green', linestyle='--', marker='^')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Success Rate (%)')
ax1.set_title('Standard_Lift Success Rate')
ax1.set_xticks(epochs)
ax1.set_ylim(-5, 105)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right')

# 2. Plot Stack Success Rate (Right Subgraph)
ax2.plot(epochs, stack_means, label='Mean Success Rate', color='blue', marker='o', linewidth=2)
ax2.fill_between(epochs, stack_mins, stack_maxs, color='blue', alpha=0.15, label='Min-Max Range')
ax2.plot(epochs, stack_mins, label='Min Success Rate', color='red', linestyle='--', marker='v')
ax2.plot(epochs, stack_maxs, label='Max Success Rate', color='green', linestyle='--', marker='^')
ax2.set_xlabel('Epoch')
ax2.set_title('Standard_Stack Success Rate')
ax2.set_xticks(epochs)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower right')

# Overall title and layout adjustments
plt.suptitle('DT (w/o ICL) on PH Lift + 10 Trajectories of Stack', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the visualization
plt.savefig('lift_and_stack.png', dpi=300)