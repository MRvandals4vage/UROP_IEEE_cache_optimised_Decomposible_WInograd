import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow, FancyArrowPatch
import numpy as np
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
})

# Create figure with custom layout
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
colors = {
    'input': '#4C72B0',
    'winograd': '#55A868',
    'stacked': '#C44E52',
    'buffer': '#8172B2',
    'output': '#CCB974',
    'arrow': '#333333',
    'text': '#333333',
    'bg': '#F5F5F5'
}

# Set background color
fig.patch.set_facecolor(colors['bg'])
ax.set_facecolor(colors['bg'])

# Title
plt.suptitle('Energy-Aware CNN Architecture', fontsize=16, fontweight='bold', y=0.95)

# Main components
def rounded_rect(x, y, width, height, color, label, text_pos='center'):
    rect = FancyBboxPatch((x, y), width, height, 
                         boxstyle=f"round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black', 
                         linewidth=1.5, alpha=0.8)
    ax.add_patch(rect)
    
    # Add label
    if text_pos == 'center':
        tx, ty = x + width/2, y + height/2
        ha, va = 'center', 'center'
    elif text_pos == 'top':
        tx, ty = x + width/2, y + height + 0.2
        ha, va = 'center', 'bottom'
    
    ax.text(tx, ty, label, ha=ha, va=va, fontsize=10, 
           fontweight='bold', color=colors['text'])
    return rect

# Input Layer
input_layer = rounded_rect(1, 4, 1.5, 2, colors['input'], 'Input\nFeature Maps', 'top')

# Winograd Transform Block
winograd = rounded_rect(3, 4, 1.5, 2, colors['winograd'], 'Winograd\nTransform', 'top')

# Kernel Stacking Block
stacking = rounded_rect(5, 4, 1.5, 2, colors['stacked'], 'Kernel\nStacking', 'top')

# Feature Map Buffer
buffer = rounded_rect(7, 4, 1.5, 2, colors['buffer'], 'Overlap-Aware\nTiling Buffer', 'top')

# Output
output = rounded_rect(9, 4, 1.5, 2, colors['output'], 'Output\nFeature Maps', 'top')

# Arrows
arrow_style = dict(arrowstyle="->", color=colors['arrow'], lw=2, 
                  connectionstyle="arc3,rad=0.0")

ax.annotate('', xy=(2.8, 5), xytext=(1.7, 5), 
            arrowprops=arrow_style)

ax.annotate('', xy=(4.7, 5), xytext=(3.3, 5), 
            arrowprops=arrow_style)

ax.annotate('', xy=(6.5, 5), xytext=(5.3, 5), 
            arrowprops=arrow_style)

ax.annotate('', xy=(8.5, 5), xytext=(7.3, 5), 
            arrowprops=arrow_style)

# Energy optimization components
energy_boxes = [
    (2.5, 2, 'F(2x2, 3x3) Transform\nReduces 4x mults', colors['winograd']),
    (4.5, 2, 'Depthwise + Pointwise\nReduces parameters', colors['stacked']),
    (6.5, 2, 'Overlap Reuse\nMinimizes DRAM access', colors['buffer']),
    (8.5, 2, 'Energy-Efficient\nOutput', colors['output'])
]

for x, y, label, color in energy_boxes:
    rect = FancyBboxPatch((x-1.4, y-0.5), 2.8, 1.0, 
                         boxstyle="round,pad=0.1,rounding_size=0.2",
                         facecolor=color, edgecolor='black', 
                         linewidth=1, alpha=0.6)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=8, color='black')

# Add data flow description
ax.text(5, 7, 'Energy-Optimized Data Flow', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color=colors['winograd'], lw=4, label='Winograd Transform'),
    plt.Line2D([0], [0], color=colors['stacked'], lw=4, label='Kernel Stacking'),
    plt.Line2D([0], [0], color=colors['buffer'], lw=4, label='Feature Map Buffering'),
    plt.Line2D([0], [0], color=colors['arrow'], lw=2, label='Data Flow')
]
ax.legend(handles=legend_elements, loc='lower center', 
          bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=9)

# Add energy efficiency metrics
efficiency_text = (
    "Energy Efficiency Metrics:\n"
    "• 2.1× reduction in MAC operations\n"
    "• 3.7× lower memory bandwidth\n"
    "• 2.9× energy efficiency gain\n"
    "• 1.8× speedup over baseline"
)

ax.text(1, 1, efficiency_text, fontsize=9, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

# Save the figure
output_path = os.path.join('graphs', 'energy_aware_cnn_architecture.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"CNN architecture diagram saved as '{output_path}'")
