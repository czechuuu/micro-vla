"""These are the results from the baseline ICL carried out on the Lift task with fusing all the observations into one token."""
import matplotlib.pyplot as plt
import numpy as np

# Data grouped by epoch
epochs = list(range(20))

# Previous Data (finetuned_iiwa)
data_batch_8 = {
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

# New Data (more_finetuned_iiwa)
data_batch_64 = {
    0: [0.0, 0.0, 0.0],
    1: [0.0, 0.0, 0.0],
    2: [37.5, 0.0, 37.5],
    3: [50.0, 25.0, 62.5],
    4: [25.0, 37.5, 37.5],
    5: [0.0, 0.0, 25.0],
    6: [12.5, 0.0, 12.5],
    7: [0.0, 0.0, 12.5],
    8: [25.0, 12.5, 37.5],
    9: [37.5, 25.0, 0.0],
    10: [0.0, 50.0, 37.5],
    11: [0.0, 0.0, 0.0],
    12: [0.0, 0.0, 0.0],
    13: [0.0, 0.0, 0.0],
    14: [0.0, 0.0, 0.0],
    15: [12.5, 0.0, 0.0],
    16: [0.0, 0.0, 0.0],
    17: [0.0, 0.0, 12.5],
    18: [0.0, 0.0, 0.0],
    19: [0.0, 0.0, 25.0]
}

# Helper function to plot a single subplot
def plot_subplot(ax, data, title):
    # Calculate statistics
    means = [np.mean(data[e]) for e in epochs]
    mins = [np.min(data[e]) for e in epochs]
    maxs = [np.max(data[e]) for e in epochs]

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
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right') # Moved to upper right to prevent obscuring zero values at the bottom

# Plotting setup (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot the individual data sets on their respective axes
plot_subplot(ax1, data_batch_8, 'Batch Size 8')
plot_subplot(ax2, data_batch_64, 'Batch Size 64')

# Add the main title across the entire figure
fig.suptitle('IIWA_Lift Success Rate against gradient updates fine-tuning Panda on 10 demonstrations of IIWA', fontsize=12, y=0.98)

# Adjust layout and save the visualization
plt.tight_layout()
plt.savefig('finetuned_iiwa.png', dpi=300)