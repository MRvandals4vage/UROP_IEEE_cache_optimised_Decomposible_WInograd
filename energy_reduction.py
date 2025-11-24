import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# Set up plot style
plt.style.use('seaborn-v0_8-white')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['axes.linewidth'] = 1.0
rcParams['grid.alpha'] = 0.3
rcParams['grid.linewidth'] = 0.5
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1

# Generate data
stack_depth = np.array([1, 2, 3, 4, 5])

# Example data (replace with actual values if available)
computation_energy = 1.0 / (0.7 + 0.3 * stack_depth)  # Example computation energy reduction
memory_energy = 1.0 - 0.15 * (stack_depth - 1)         # Example memory energy reduction

# Create figure
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(stack_depth, computation_energy, 
         color='#009E73', linewidth=2.5, 
         label='Computation Energy')
plt.plot(stack_depth, memory_energy, 
         color='#E69F00', linestyle='--', 
         linewidth=2.5, label='Memory Energy')

# Add labels and title
plt.xlabel('Stack Depth (m)', fontsize=12, labelpad=10)
plt.ylabel('Normalized Energy (Baseline = 1.0)', fontsize=12, labelpad=10)
plt.title('Energy Reduction Across Stacked Kernel Depths', 
          fontsize=14, pad=15, fontweight='bold')

# Customize the plot
plt.xticks(stack_depth)
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.3)

# Add legend
legend = plt.legend(frameon=True, framealpha=0.9, 
                   edgecolor='0.8', fontsize=10)
legend.get_frame().set_linewidth(0.8)

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the figure
output_path = os.path.join('graphs', 'energy_reduction_stacked_kernel.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Energy reduction graph saved as '{output_path}'")
