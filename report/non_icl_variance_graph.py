import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Parsing the training and validation loss data
loss_data = [
    {"epoch": 1, "train_loss": 0.029203769743203295, "val_loss": 0.022445643557423428},
    {"epoch": 2, "train_loss": 0.01826112123736001, "val_loss": 0.021570634623175534},
    {"epoch": 3, "train_loss": 0.01585067859466333, "val_loss": 0.021892147891990402},
    {"epoch": 4, "train_loss": 0.01452467820795099, "val_loss": 0.023052454836071788},
    {"epoch": 5, "train_loss": 0.013426401685718277, "val_loss": 0.023380962636775703},
    {"epoch": 6, "train_loss": 0.01235801539118957, "val_loss": 0.024702113160879172},
    {"epoch": 7, "train_loss": 0.01148943832067243, "val_loss": 0.024926002006273436},
    {"epoch": 8, "train_loss": 0.01076629633348811, "val_loss": 0.025592640629273},
    {"epoch": 9, "train_loss": 0.009872608727117607, "val_loss": 0.027504168380396207},
    {"epoch": 10, "train_loss": 0.009052072568591514, "val_loss": 0.028529690819236786},
    {"epoch": 11, "train_loss": 0.00825527527480634, "val_loss": 0.028123074745321098},
    {"epoch": 12, "train_loss": 0.007809398083204249, "val_loss": 0.030799554897660092},
    {"epoch": 13, "train_loss": 0.007458395548506614, "val_loss": 0.02968626585006401},
    {"epoch": 14, "train_loss": 0.006794543894981846, "val_loss": 0.02930686964277773},
    {"epoch": 15, "train_loss": 0.006345759107252805, "val_loss": 0.027365636141156825},
    {"epoch": 16, "train_loss": 0.006006875757444137, "val_loss": 0.030574576938994416},
    {"epoch": 17, "train_loss": 0.005557028780939109, "val_loss": 0.02900420900575128},
    {"epoch": 18, "train_loss": 0.005421083252620669, "val_loss": 0.02855018129916627},
    {"epoch": 19, "train_loss": 0.005060133872630605, "val_loss": 0.029572003002536772},
    {"epoch": 20, "train_loss": 0.004864256034195082, "val_loss": 0.031075242278753083}
]

# 2. Parsing the success rate measurements (aligned with epochs 1 to 20)
success_rates = [
    [0.375, 0.375, 0.625], [0.625, 0.375, 0.125], [0.625, 0.5, 0.5],
    [0.625, 0.75, 0.625],  [0.75, 0.5, 0.875],   [0.75, 0.625, 0.75],
    [0.625, 0.5, 0.75],   [0.625, 0.75, 0.75],   [0.75, 0.75, 0.25],
    [0.875, 0.75, 1.0],   [0.75, 0.875, 0.625],  [0.75, 0.75, 0.875],
    [0.75, 0.625, 0.5],   [0.5, 0.5, 0.625],     [0.75, 0.75, 0.625],
    [1.0, 0.875, 0.875],   [0.75, 0.625, 0.625],  [0.625, 0.875, 0.375],
    [0.75, 0.625, 1.0],   [0.625, 0.75, 0.75]
]

# Extract lists for plotting
epochs = [d["epoch"] for d in loss_data]
train_loss = [d["train_loss"] for d in loss_data]
val_loss = [d["val_loss"] for d in loss_data]

# Compute mean, min, and max values for the success rate interval
sr_means = [np.mean(sr) for sr in success_rates]
sr_mins = [np.min(sr) for sr in success_rates]
sr_maxs = [np.max(sr) for sr in success_rates]

# Initialize the figure and the first (left) axis for Losses
fig, ax1 = plt.subplots(figsize=(10, 6))

color_loss = 'tab:red'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color=color_loss, fontsize=12)

# Plot training and validation losses
line1, = ax1.plot(epochs, train_loss, label='Train Loss', color='tab:red', linestyle='--')
line2, = ax1.plot(epochs, val_loss, label='Validation Loss', color='tab:orange')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_xticks(epochs)
ax1.grid(True, linestyle=':', alpha=0.5)

# Create a second (right) axis for the Success Rate
ax2 = ax1.twinx()  
color_sr = 'tab:blue'
ax2.set_ylabel('Success Rate', color=color_sr, fontsize=12)  

# Plot the success rate mean line and the shaded min-max interval
line3, = ax2.plot(epochs, sr_means, label='Success Rate (Mean)', color='tab:blue', linewidth=2)
ax2.fill_between(epochs, sr_mins, sr_maxs, color='tab:blue', alpha=0.15, label='Success Rate Range')
ax2.tick_params(axis='y', labelcolor=color_sr)

# Consolidate legends from both axes into a single box
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title('Training/Validation Loss and Success Rate vs. Epoch', fontsize=14, pad=15)
fig.tight_layout()  

# Save the plot
plt.savefig('training_metrics.png')