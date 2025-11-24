import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

# Generate frequency data
freq = np.linspace(0, 1, 500)  # Normalized frequency from 0 to 1

# Generate example frequency responses (replace with actual data if available)
def generate_response(freq, type='monolithic'):
    if type == 'monolithic':
        # Monolithic response - steeper roll-off
        return -20 * np.log10(1 + 5 * freq**2)
    else:  # stacked
        # Stacked response - more gradual roll-off
        return -10 * np.log10(1 + 2 * freq**3)

mono_response = generate_response(freq, 'monolithic')
stacked_response = generate_response(freq, 'stacked')

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot the frequency responses
plt.plot(freq, mono_response, color='#D55E00', linewidth=2.5, label='Monolithic Convolution')
plt.plot(freq, stacked_response, color='#0072B2', linewidth=2.5, label='Stacked-Kernel Convolution')

# Add labels and title
plt.xlabel('Normalized Spatial Frequency (0 → 1)', fontsize=12, labelpad=10)
plt.ylabel('Amplitude (dB)', fontsize=12, labelpad=10)
plt.title('Frequency Response of Monolithic vs Stacked-Kernel Convolutions', 
          fontsize=14, pad=15, fontweight='bold')

# Customize the plot
plt.xlim(0, 1)
plt.ylim(-25, 5)  # Adjust based on your data range
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(frameon=True, framealpha=0.9, edgecolor='0.8', fontsize=10)

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the figure
output_path = os.path.join('graphs', 'frequency_response.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Frequency response graph saved as '{output_path}'")
