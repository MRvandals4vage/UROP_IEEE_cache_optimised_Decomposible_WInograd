import matplotlib.pyplot as plt
import numpy as np

# Data
stack_depth = [1, 2, 3, 4, 5]
dram_dominant = [1.00, 0.82, 0.68, 0.61, 0.57]
cache_optimized = [1.00, 0.78, 0.59, 0.52, 0.49]
onchip_buffer = [1.00, 0.74, 0.56, 0.48, 0.45]

# Set up the figure with IEEE style
plt.figure(figsize=(6.5, 5))
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#808080',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# Plot the data with specified colors and markers
plt.plot(stack_depth, dram_dominant, 'o-', color='#1f77b4', 
         markerfacecolor='white', markeredgewidth=1, 
         label='DRAM-Dominant')
plt.plot(stack_depth, cache_optimized, 's-', color='#ff7f0e',
         markerfacecolor='white', markeredgewidth=1,
         label='Cache-Optimized')
plt.plot(stack_depth, onchip_buffer, '^-', color='#2ca02c',
         markerfacecolor='white', markeredgewidth=1,
         label='Full On-Chip Buffering')

# Set title and labels
plt.title('Normalized Inference Energy under Memory Access Policies', 
          fontweight='bold', pad=12)
plt.xlabel('Stack Depth (m)', fontweight='bold')
plt.ylabel('Normalized Energy (E_norm)', fontweight='bold')

# Set axis limits and ticks
plt.ylim(0.40, 1.05)
plt.xlim(0.8, 5.2)
plt.xticks(stack_depth)
plt.yticks(np.arange(0.4, 1.1, 0.1))

# Add legend
legend = plt.legend(loc='upper right', frameon=True, 
                   edgecolor='black', fancybox=False)
legend.get_frame().set_alpha(1)

# Ensure grid is below other elements
plt.grid(True, linestyle='-', alpha=0.3, color='#808080')

# Save the figure with high resolution
plt.savefig('energy_reduction.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved as 'energy_reduction.png' with 300 DPI")
