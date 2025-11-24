import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# Set up plot style for IEEE
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.autolayout': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})

# Data
variants = ['Baseline\nConvolution', 'Separable\nConvolution', 'Stacked\n(m=3)']
metrics = ['Energy', 'Latency', 'Accuracy']

# Normalized values (1.0 = baseline)
data = {
    'Baseline': [1.00, 1.00, 1.00],    # Energy, Latency, Accuracy
    'Separable': [0.85, 0.90, 0.95],   # 15% better, 10% better, 5% worse
    'Stacked': [0.70, 0.75, 1.02]      # 30% better, 25% better, 2% better
}

# Colors for each variant
colors = {
    'Baseline': ['#1f77b4', '#4c8cb5', '#7aa9c9'],
    'Separable': ['#ff7f0e', '#ff9b4d', '#ffb77c'],
    'Stacked': ['#2ca02c', '#5cb85c', '#8fd18f']
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set width of bars
bar_width = 0.25
index = np.arange(len(variants))

# Plot bars for each metric
for i, metric in enumerate(metrics):
    # Baseline (blue)
    ax.bar(index - bar_width + i*bar_width, 
           [data['Baseline'][i]] * len(variants),
           bar_width, 
           color=colors['Baseline'][i],
           edgecolor='white',
           linewidth=0.5,
           label=f'Baseline {metric}' if i == 0 else "")
    
    # Separable (orange)
    ax.bar(index - bar_width + i*bar_width, 
           [data['Separable'][i]] * len(variants),
           bar_width, 
           color=colors['Separable'][i],
           edgecolor='white',
           linewidth=0.5,
           label=f'Separable {metric}' if i == 0 else "")
    
    # Stacked (green)
    bars = ax.bar(index - bar_width + i*bar_width, 
                 [data['Stacked'][i]] * len(variants),
                 bar_width, 
                 color=colors['Stacked'][i],
                 edgecolor='white',
                 linewidth=0.5,
                 label=f'Stacked {metric}' if i == 0 else "")
    
    # Add value labels
    for j, variant in enumerate(variants):
        val = data[['Baseline', 'Separable', 'Stacked'][j]][i]
        if j > 0:  # Show percentage improvement over baseline
            improvement = (1 - val/data['Baseline'][i]) * 100
            if i == 2:  # Accuracy (higher is better)
                improvement = (val/1.0 - 1) * 100
            label = f"{improvement:+.1f}%"
            va = 'bottom' if val > 0.1 else 'top'
            y_pos = val + 0.02 if va == 'bottom' else val - 0.02
            ax.text(j - bar_width + i*bar_width, y_pos, label,
                   ha='center', va=va, fontsize=8, weight='bold')

# Customize the plot
ax.set_xticks(index)
ax.set_xticklabels(variants, fontsize=10)
ax.set_ylabel('Normalized Metric (Baseline = 1.0)', fontsize=10)
ax.set_ylim(0, 1.4)  # Extra space for labels
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.5)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['Baseline'][0], label='Baseline Convolution'),
    Patch(facecolor=colors['Separable'][0], label='Separable Convolution'),
    Patch(facecolor=colors['Stacked'][0], label='Stacked Convolution (m=3)'),
    Patch(facecolor='white', edgecolor='black', linewidth=0.5, 
          label='Darker: Energy\nMedium: Latency\nLighter: Accuracy')
]

ax.legend(handles=legend_elements, loc='upper right', 
          bbox_to_anchor=(1.0, 1.0), fontsize=9)

plt.title('CNN Variant Comparison: Normalized Metrics', 
          fontsize=12, pad=15, fontweight='bold')

# Save the figure
output_path = os.path.join('graphs', 'cnn_variant_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"CNN variant comparison plot saved as '{output_path}'")
