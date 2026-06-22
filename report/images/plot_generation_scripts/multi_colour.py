import matplotlib.pyplot as plt
import numpy as np

# Data Definition
data = {
    'Blue': [62.5, 62.5, 50.0],
    'Red': [75.0, 37.5, 50.0],
    'Green': [37.5, 62.5, 50.0]
}

# Calculate Statistics
colors = list(data.keys())
mins = [min(data[c]) for c in colors]
maxs = [max(data[c]) for c in colors]
means = [np.mean(data[c]) for c in colors]

# Sorting colors by mean performance (descending order)
sorted_indices = np.argsort(means)[::-1]
colors = [colors[i] for i in sorted_indices]
mins = [mins[i] for i in sorted_indices]
maxs = [maxs[i] for i in sorted_indices]
means = [means[i] for i in sorted_indices]

# Color map matching the cube colors
color_map = {'Red': '#ff4d4d', 'Blue': '#4da6ff', 'Green': '#5cd65c'}
bar_colors = [color_map[c] for c in colors]

# Create Figure
fig, ax = plt.subplots(figsize=(7, 5))

# Calculate error ranges relative to the mean
lower_err = np.array(means) - np.array(mins)
upper_err = np.array(maxs) - np.array(means)
yerr = np.array([lower_err, upper_err])

# Plot bars with min/max error bounds
bars = ax.bar(colors, means, yerr=yerr, color=bar_colors, capsize=10, 
               alpha=0.85, edgecolor='black', zorder=2)

# Styling details
ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Success Rate by Cube Color (Mean with Min/Max Range)', fontsize=12, fontweight='bold', pad=15)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)

# Add exact value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save plot to PNG file
output_filename = 'multi_colour.png'
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"Success: Saved to {output_filename}")