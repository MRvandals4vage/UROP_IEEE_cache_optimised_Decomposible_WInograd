import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# Set Helvetica font and IEEE style
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['text.usetex'] = False  # Disable LaTeX for better compatibility
rcParams['axes.linewidth'] = 0.8  # Thinner axis lines
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['axes.edgecolor'] = 'black'  # Black axis lines
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.15  # Very light grid
rcParams['grid.linewidth'] = 0.5
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1

# Data for the performance comparison
models = ['VGG-16', 'ResNet-34', 'MobileNetV2']
baseline = [1.0, 1.0, 1.0]  # Normalized baseline = 1.0
stacked = [0.85, 0.72, 0.99]  # Example normalized values for stacked kernels

# Create figure with appropriate size
plt.figure(figsize=(8, 6))

# Set positions of bars on X-axis
bar_width = 0.35
x = np.arange(len(models))

# Create bars with specified colors and styles
baseline_bars = plt.bar(x - bar_width/2, baseline, bar_width, 
                       color='#1f77b4', alpha=0.8, label='Baseline',
                       edgecolor='black', linewidth=0.7)

stacked_bars = plt.bar(x + bar_width/2, stacked, bar_width, 
                      color='#ff7f0e', alpha=0.8, label='Stacked-Kernel',
                      edgecolor='black', linewidth=0.7, hatch='//')

# Add value labels on top of each bar
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

add_value_labels(baseline_bars)
add_value_labels(stacked_bars)

# Customize the plot
plt.xlabel('Model Architecture', fontsize=12, labelpad=10)
plt.ylabel('Normalized Metric', fontsize=12, labelpad=10)
plt.title('Performance Comparison of Baseline vs Stacked-Kernel CNNs', 
          fontsize=14, pad=15, fontweight='bold')

# Set x-ticks and labels
plt.xticks(x, models, fontsize=11)
plt.yticks(fontsize=10)
plt.ylim(0, 1.2)  # Leave some space above the highest bar

# Add a horizontal line at y=1.0 for reference
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# Add legend with a frame
legend = plt.legend(frameon=True, fontsize=10, 
                   facecolor='white', framealpha=0.9,
                   edgecolor='0.8', loc='upper right')
legend.get_frame().set_linewidth(0.8)

# Remove top and right spines for cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = os.path.join('graphs', 'cnn_performance_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Performance comparison graph saved as '{output_path}'")
