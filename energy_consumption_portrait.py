import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# Data for energy consumption (normalized)
policies = ['Baseline DRAM', 'Shared Memory', 'On-Chip Cache', 'Hybrid Access']
monolithic = [1.0, 0.8, 0.6, 0.45]  # Example values for monolithic kernels
stacked = [0.9, 0.65, 0.4, 0.3]     # Example values for stacked kernels

# Create figure and axis with portrait orientation
plt.figure(figsize=(8, 10))

# Set width of each bar
bar_width = 0.35
index = np.arange(len(policies))

# Plot bars with adjusted positions for better spacing
plt.bar(index - bar_width/2, monolithic, bar_width, label='Monolithic Kernels', color='#1f77b4', alpha=0.8)
plt.bar(index + bar_width/2, stacked, bar_width, label='Stacked Kernels', color='#ff7f0e', alpha=0.8)

# Customize the plot
plt.xlabel('Memory Access Policy', fontsize=14, labelpad=12)
plt.ylabel('Normalized Energy Consumption (E_norm)', fontsize=14, labelpad=12)
plt.title('Energy Efficiency: Monolithic vs. Stacked Kernels', 
          fontsize=16, pad=20, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(index, policies, rotation=15, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1.2)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add legend with larger font
plt.legend(fontsize=12, frameon=True, framealpha=0.9, loc='upper right')

# Add value labels on top of each bar with improved positioning
for i, (m, s) in enumerate(zip(monolithic, stacked)):
    plt.text(i - bar_width/2, m + 0.02, f'{m:.2f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.text(i + bar_width/2, s + 0.02, f'{s:.2f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add horizontal line at y=1.0 for reference
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure with high resolution
output_path = os.path.join('graphs', 'energy_consumption_portrait.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Print confirmation
print(f"\nPortrait-style visualization saved as '{output_path}'")
