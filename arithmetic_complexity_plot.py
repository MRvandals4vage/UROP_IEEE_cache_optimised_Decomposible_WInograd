import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up IEEE-style formatting
plt.style.use('default')

# Customize to match IEEE style
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'serif',
    'mathtext.bf': 'serif:bold',
    'mathtext.it': 'serif:italic',
    'mathtext.sf': 'sans',
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.figsize': (3.5, 2.5)
})

# Calculate normalized complexities
m = np.arange(1, 6)  # Stack depths from 1 to 5
C_norm = m * (4/49)  # Normalized arithmetic complexity

# Create the plot with a single subplot
fig, ax = plt.subplots(figsize=(4, 3))

# Plot the data
ax.plot(m, C_norm, color='navy', linewidth=1.5)

# Formatting
ax.set_title('Arithmetic Complexity Reduction as a Function of Stacked Kernel Count', 
             fontsize=10, pad=10)
ax.set_xlabel('Stack Depth ($m$)', fontsize=9)
ax.set_ylabel('Normalized Arithmetic Complexity ($C_{norm}$)', fontsize=9)
ax.set_xticks(m)

# Set grid and adjust layout
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('arithmetic_complexity_reduction.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved as 'arithmetic_complexity_reduction.png' in the current directory.")
