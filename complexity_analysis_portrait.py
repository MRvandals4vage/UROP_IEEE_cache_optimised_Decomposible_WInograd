import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# Sample data
kernel_counts = [1, 2, 3, 4, 5]
flops = [0.5, 2.1, 5.3, 12.8, 28.4]
parameters = [0.1, 0.5, 1.8, 5.2, 15.7]
test_accuracy = [0.25, 0.38, 0.45, 0.48, 0.49]

# Calculate normalized arithmetic complexity
max_flops = max(flops)
normalized_complexity = [f/max_flops for f in flops]

# Create figure with portrait orientation
plt.figure(figsize=(8, 10))

# Plot 1: Test Accuracy vs Number of Kernels
plt.subplot(3, 1, 1)
plt.plot(kernel_counts, test_accuracy, 'o-', color='#1f77b4', linewidth=2, markersize=8)
plt.xlabel('Stacked Kernel Count (m)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Model Performance vs Architecture Complexity', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(kernel_counts)
for i, acc in enumerate(test_accuracy):
    plt.text(kernel_counts[i], acc + 0.01, f'{acc:.2f}', 
             ha='center', va='bottom', fontsize=10)

# Plot 2: Arithmetic Complexity (FLOPs) vs Number of Kernels
plt.subplot(3, 1, 2)
plt.plot(kernel_counts, flops, 's-', color='#ff7f0e', linewidth=2, markersize=8)
plt.xlabel('Stacked Kernel Count (m)', fontsize=12)
plt.ylabel('Arithmetic Complexity (MFLOPs)', fontsize=12)
plt.title('Arithmetic Complexity vs Architecture', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(kernel_counts)
for i, flop in enumerate(flops):
    plt.text(kernel_counts[i], flop + 0.5, f'{flop:.1f}M', 
             ha='center', va='bottom', fontsize=10)

# Plot 3: Normalized Arithmetic Complexity vs Number of Kernels
plt.subplot(3, 1, 3)
plt.plot(kernel_counts, normalized_complexity, 'D-', color='#2ca02c', linewidth=2, markersize=8)
plt.xlabel('Stacked Kernel Count (m)', fontsize=12)
plt.ylabel('Normalized Arithmetic Complexity (C_norm)', fontsize=12)
plt.title('Normalized Complexity vs Architecture', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(kernel_counts)
for i, norm in enumerate(normalized_complexity):
    plt.text(kernel_counts[i], norm + 0.01, f'{norm:.2f}', 
             ha='center', va='bottom', fontsize=10)

# Adjust layout and save
plt.tight_layout()
output_path = os.path.join('graphs', 'complexity_analysis_portrait.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Portrait-style complexity analysis saved as '{output_path}'")
