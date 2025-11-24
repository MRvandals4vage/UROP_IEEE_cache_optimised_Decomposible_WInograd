import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Sample data for demonstration
# In a real scenario, these would be calculated from actual model training
kernel_counts = [1, 2, 3, 4, 5]  # Number of convolutional layers
flops = [0.5, 2.1, 5.3, 12.8, 28.4]  # FLOPs in millions
parameters = [0.1, 0.5, 1.8, 5.2, 15.7]  # Parameters in millions
test_accuracy = [0.25, 0.38, 0.45, 0.48, 0.49]  # Test accuracy

# Create the figure with three subplots
plt.figure(figsize=(20, 6))

# Calculate normalized arithmetic complexity
max_flops = max(flops)
normalized_complexity = [f/max_flops for f in flops]

# Plot 1: Test Accuracy vs Number of Kernels
plt.subplot(1, 3, 1)
plt.plot(kernel_counts, test_accuracy, 'o-', color='#1f77b4', linewidth=2, markersize=8)
plt.xlabel('Stacked Kernel Count (m)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Model Performance vs Architecture Complexity', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(kernel_counts)

# Add value labels
for i, acc in enumerate(test_accuracy):
    plt.text(kernel_counts[i], acc + 0.01, f'{acc:.2f}', 
             ha='center', va='bottom', fontsize=10)

# Plot 2: Arithmetic Complexity (FLOPs) vs Number of Kernels
plt.subplot(1, 3, 2)
plt.plot(kernel_counts, flops, 's-', color='#ff7f0e', linewidth=2, markersize=8)
plt.xlabel('Stacked Kernel Count (m)', fontsize=12)
plt.ylabel('Arithmetic Complexity (MFLOPs)', fontsize=12)
plt.title('Arithmetic Complexity vs Architecture', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(kernel_counts)

# Add value labels
for i, flop in enumerate(flops):
    plt.text(kernel_counts[i], flop + 0.5, f'{flop:.1f}M', 
             ha='center', va='bottom', fontsize=10)

# Plot 3: Normalized Arithmetic Complexity vs Number of Kernels
plt.subplot(1, 3, 3)
plt.plot(kernel_counts, normalized_complexity, 'D-', color='#2ca02c', linewidth=2, markersize=8)
plt.xlabel('Stacked Kernel Count (m)', fontsize=12)
plt.ylabel('Normalized Arithmetic Complexity (C_norm)', fontsize=12)
plt.title('Normalized Complexity vs Architecture', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(kernel_counts)

# Add value labels
for i, norm in enumerate(normalized_complexity):
    plt.text(kernel_counts[i], norm + 0.01, f'{norm:.2f}', 
             ha='center', va='bottom', fontsize=10)

# Adjust layout and save
plt.tight_layout()
plt.savefig('complexity_analysis.png', dpi=300, bbox_inches='tight')
print("Analysis complete! The plot has been saved as 'complexity_analysis.png'")

# Print results summary
print("\nSummary of Results:")
print("-" * 80)
print(f"{'Layers':<10} {'FLOPs (M)':<15} {'Params (M)':<15} {'Test Acc'}")
print("-" * 80)
for i in range(len(kernel_counts)):
    print(f"{kernel_counts[i]:<10} {flops[i]:<15.1f} {parameters[i]:<15.1f} {test_accuracy[i]:.4f}")
