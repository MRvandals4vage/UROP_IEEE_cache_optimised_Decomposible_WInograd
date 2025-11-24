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

# Create figure and axis
plt.figure(figsize=(12, 6))

# Set width of each bar
bar_width = 0.35
index = np.arange(len(policies))

# Plot bars
plt.bar(index, monolithic, bar_width, label='Monolithic Kernels', color='#1f77b4', alpha=0.8)
plt.bar(index + bar_width, stacked, bar_width, label='Stacked Kernels', color='#ff7f0e', alpha=0.8)

# Add labels, title, and legend
plt.xlabel('Memory Access Policy', fontsize=12, labelpad=10)
plt.ylabel('Normalized Energy Consumption (E_norm)', fontsize=12, labelpad=10)
plt.title('Energy Efficiency: Monolithic vs. Stacked Kernels', fontsize=14, pad=15)
plt.xticks(index + bar_width/2, policies, rotation=15, ha='right')
plt.ylim(0, 1.2)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.legend(fontsize=10, frameon=True, framealpha=0.9)

# Add value labels on top of each bar
for i, (m, s) in enumerate(zip(monolithic, stacked)):
    plt.text(i, m + 0.02, f'{m:.2f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + bar_width, s + 0.02, f'{s:.2f}', ha='center', va='bottom', fontsize=9)

# Add horizontal line at y=1.0 for reference
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# Adjust layout and save
plt.tight_layout()
output_path = os.path.join('graphs', 'energy_consumption_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Print summary
print("Energy Consumption Analysis:")
print("-" * 80)
print(f"{'Policy':<20} {'Monolithic':<15} {'Stacked':<15} 'Savings'")
print("-" * 80)
for i, policy in enumerate(policies):
    savings = ((monolithic[i] - stacked[i]) / monolithic[i]) * 100
    print(f"{policy:<20} {monolithic[i]:<15.2f} {stacked[i]:<15.2f} {savings:.1f}%")

print(f"\nVisualization saved as '{output_path}'")
