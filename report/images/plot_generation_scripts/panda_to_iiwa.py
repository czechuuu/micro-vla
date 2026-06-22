import matplotlib.pyplot as plt
import numpy as np

# Data grouped by epoch
epochs = [15, 16, 17, 18, 19]
data = {
    15: [75.0, 87.5, 87.5],
    16: [62.5, 87.5, 87.5],
    17: [62.5, 75.0, 87.5],
    18: [100.0, 100.0, 87.5],
    19: [100.0, 75.0, 62.5]
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
ax.set_title('IIWA Lift success rate after training on 200 Panda and 10 IIWA demonstrations', fontsize=14, pad=15)
ax.set_xticks(epochs)
ax.set_ylim(0, 100)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right')

# Adjust layout and save the visualization
plt.tight_layout()
plt.savefig('panda_to_iiwa.png', dpi=300)