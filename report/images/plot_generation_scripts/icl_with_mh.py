"""These are the results from ICL carried out on PH+MH lift task with negative time based rewards (-1 at every step, +100 on success - meant to help discriminate between the better and worse trajectories)."""
import matplotlib.pyplot as plt
import numpy as np

# Data grouped by epoch
epochs = [15, 16, 17, 18, 19]
data = {
    15: [50.0, 12.5, 50.0],
    16: [25.0, 25.0, 25.0],
    17: [62.5, 25.0, 75.5],
    18: [50.0, 50.0, 87.5],
    19: [50.0, 25.0, 37.5] 
}

# Calculate statistics
means = [np.mean(data[e]) for e in epochs]
mins = [np.min(data[e]) for e in epochs]
maxs = [np.max(data[e]) for e in epochs]

# Plotting setup
fig, ax = plt.subplots(figsize=(8, 5))

# Plot mean line
ax.plot(epochs, means, label='Mean Success Rate', color='blue', marker='o', linewidth=2)

# Shaded area representing the range between min and max
ax.fill_between(epochs, mins, maxs, color='blue', alpha=0.15, label='Min-Max Range')

# Plot min and max markers/lines
ax.plot(epochs, mins, label='Min Success Rate', color='red', linestyle='--', marker='v')
ax.plot(epochs, maxs, label='Max Success Rate', color='green', linestyle='--', marker='^')

# Graph customization
ax.set_xlabel('Epoch')
ax.set_ylabel('Success Rate (%)')
ax.set_title('PH+MH Lift Success Rate Metrics per Epoch')
ax.set_xticks(epochs)
ax.set_ylim(0, 100)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right')

# Adjust layout and save the visualization
plt.tight_layout()
plt.savefig('icl_with_mh.png', dpi=300)