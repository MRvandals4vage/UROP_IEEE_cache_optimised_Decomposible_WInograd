import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up plot style with IEEE specifications
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'figure.figsize': (3.5, 2.5),  # Single column width
    'font.size': 10,                # Base font size
    'axes.labelsize': 10,           # Axis labels
    'axes.titlesize': 10,           # Title size
    'xtick.labelsize': 9,           # X-tick labels
    'ytick.labelsize': 9,           # Y-tick labels
    'legend.fontsize': 9,           # Legend font size
    'lines.linewidth': 1.5,         # Line width
    'lines.markersize': 6,          # Marker size
    'grid.alpha': 0.3,              # Grid transparency
    'savefig.dpi': 300,             # Figure resolution
    'savefig.bbox': 'tight',        # Tight layout
    'savefig.pad_inches': 0.05,     # Padding
    'figure.autolayout': True       # Auto layout
})

# Data points
stack_depth = np.array([1, 2, 3, 4, 5])

# Normalized complexity values based on MAC count reduction
# Following the formula: C_norm = (0.5 + 0.5/m) / 2.25
# Where:
# - 1/2.25 accounts for Winograd F(2,3) benefit
# - (0.5 + 0.5/m) accounts for kernel stacking benefit
C_norm = (0.5 + 0.5/stack_depth) / 2.25

# Normalize to m=1
C_norm = C_norm / C_norm[0]

# Create figure and axis
fig, ax = plt.subplots()

# Plot the data with different markers
markers = ['o', 's', '^', 'D', 'v']  # Different markers for each point
for i, (x, y, marker) in enumerate(zip(stack_depth, C_norm, markers)):
    ax.plot(x, y, marker=marker, color='#1f77b4', 
            markerfacecolor='white', markeredgewidth=1.5)

# Connect points with a line
ax.plot(stack_depth, C_norm, color='#1f77b4', linestyle='-', 
        label='Theoretical Complexity')

# Set axis labels and title
ax.set_xlabel('Stack Depth (m)')
ax.set_ylabel('Normalized Complexity $C_{norm}$')
ax.set_title('Normalized Arithmetic Complexity vs. Kernel Stack Depth', 
             pad=10)

# Set x-ticks to exact integer values
ax.set_xticks(stack_depth)
ax.set_xticklabels([f'{x}' for x in stack_depth])

# Add grid
ax.grid(True, linestyle='--', alpha=0.3)

# Add legend
ax.legend(loc='upper right', frameon=True, fancybox=False, 
          edgecolor='black', framealpha=1)

# Save the figure
plt.savefig('arith_reduction.png', dpi=300, bbox_inches='tight', 
            pad_inches=0.05, transparent=False)

# Show the plot
plt.show()
