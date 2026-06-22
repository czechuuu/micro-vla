"""These are the results from finetuning the baseline non-ICL token-fusing token on the IIWA lift dataset. One epoch - one gradient update on 8 demonstrations."""
import matplotlib.pyplot as plt
import numpy as np

# Data grouped by epoch
epochs = list(range(20))
data = {
    0: [0.0, 0.0, 0.0],
    1: [0.0, 0.0, 0.0],
    2: [25.0, 25.0, 0.0],
    3: [37.5, 50.0, 25.0],
    4: [50.0, 25.0, 12.5],
    5: [37.5, 37.5, 87.5],
    6: [50.0, 87.5, 0.0],
    7: [50.0, 50.0, 75.0],
    8: [37.5, 50.0, 25.0],
    9: [12.5, 50.0, 37.5],
    10: [37.5, 12.5, 25.0],
    11: [25.0, 75.0, 37.5],
    12: [50.0, 62.5, 37.5],
    13: [37.5, 37.5, 37.5],
    14: [0.0, 25.0, 12.5],
    15: [0.0, 0.0, 25.0],
    16: [0.0, 0.0, 12.5],
    17: [0.0, 0.0, 12.5],
    18: [0.0, 0.0, 0.0],
    19: [0.0, 0.0, 0.0]
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
ax.set_xlabel('Num grad updates')
ax.set_ylabel('Success Rate (%)')
ax.set_title('IIWA_Lift Success Rate Metrics after N gradient updates on finetuning the Panda model')
ax.set_xticks(epochs)
ax.set_ylim(0, 100)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right')

# Adjust layout and save the visualization
plt.tight_layout()
plt.savefig('finetuned_iiwa.png', dpi=300)