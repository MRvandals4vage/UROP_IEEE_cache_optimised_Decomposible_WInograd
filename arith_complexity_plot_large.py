import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up plot style with larger dimensions and better spacing
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'figure.figsize': (6, 4.5),     # Increased figure size (50% larger)
    'font.size': 12,                # Slightly larger base font size
    'axes.labelsize': 12,           # Axis labels
    'axes.titlesize': 13,           # Title size
    'xtick.labelsize': 11,          # X-tick labels
    'ytick.labelsize': 11,          # Y-tick labels
    'legend.fontsize': 11,          # Legend font size
    'lines.linewidth': 2,           # Slightly thicker lines
    'lines.markersize': 9,          # Larger markers
    'grid.alpha': 0.3,              # Grid transparency
    'savefig.dpi': 300,             # High resolution
    'savefig.bbox': 'tight',        
    'savefig.pad_inches': 0.2,      # Increased padding
    'figure.autolayout': True,
    'figure.subplot.left': 0.15,    # More left margin
    'figure.subplot.right': 0.92,   # More right margin
    'figure.subplot.bottom': 0.15,  # More bottom margin
    'figure.subplot.top': 0.90      # More top margin
})

# Data points
stack_depth = np.array([1, 2, 3, 4, 5])

# Normalized complexity values
C_norm = (0.5 + 0.5/stack_depth) / 2.25
C_norm = C_norm / C_norm[0]  # Normalize to m=1

# Create figure and axis with more control
fig, ax = plt.subplots(figsize=(6, 4.5))

# Plot the data with different markers and more spacing
markers = ['o', 's', '^', 'D', 'v']
for i, (x, y, marker) in enumerate(zip(stack_depth, C_norm, markers)):
    ax.plot(x, y, marker=marker, color='#1f77b4', 
            markersize=10,           # Larger markers
            markerfacecolor='white', 
            markeredgewidth=2,      # Thicker marker edges
            markeredgecolor='#1f77b4')

# Connect points with a line
ax.plot(stack_depth, C_norm, color='#1f77b4', 
        linestyle='-', linewidth=2.5,
        label='Theoretical Complexity')

# Set axis labels and title with more padding
ax.set_xlabel('Stack Depth (m)', labelpad=10)
ax.set_ylabel('Normalized Complexity $C_{norm}$', labelpad=10)
ax.set_title('Normalized Arithmetic Complexity vs. Kernel Stack Depth', 
             pad=15, fontweight='bold')

# Set x-ticks with more padding
ax.set_xticks(stack_depth)
ax.set_xticklabels([f'{x}' for x in stack_depth])
ax.tick_params(axis='both', which='major', pad=8)  # More padding for ticks

# Add grid with better visibility
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)

# Add legend with more padding and better visibility
legend = ax.legend(loc='upper right', frameon=True, 
                  fancybox=True, framealpha=0.9,
                  edgecolor='#666666', facecolor='white')
legend.get_frame().set_linewidth(0.8)

# Adjust layout with more padding
plt.tight_layout(pad=2.0)

# Save the figure with higher quality
plt.savefig('arith_reduction_large.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.25,  # More padding around the plot
            facecolor='white',
            edgecolor='none')

# Show the plot
plt.show()
