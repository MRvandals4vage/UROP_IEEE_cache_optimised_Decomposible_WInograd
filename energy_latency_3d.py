import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure graphs directory exists
os.makedirs('graphs', exist_ok=True)

# Set up plot style
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
    'figure.autolayout': True
})

# Create data
m = np.linspace(1, 5, 50)  # Stacking depth
latency = np.linspace(0, 25, 50)  # Latency in ms
M, L = np.meshgrid(m, latency)

# Define energy functions for CO and SM
E_co = 0.5 * (1 - 0.1 * (M-3)**2) + 0.5 * np.exp(-0.1 * L)
E_sm = 0.4 * (1 - 0.08 * (M-3.5)**2) + 0.6 * np.exp(-0.08 * L)

# Find optimal point
opt_idx = np.argmin(E_co)
opt_m, opt_l = np.unravel_index(opt_idx, E_co.shape)
opt_energy = E_co[opt_m, opt_l]

# Create figure and 3D axis
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surfaces
surf1 = ax.plot_surface(M, L, E_co, cmap=cm.Blues, alpha=0.7, label='Cache-Optimized (CO)')
surf2 = ax.plot_surface(M, L, E_sm, cmap=cm.Oranges, alpha=0.7, label='Shared-Memory (SM)')

# Add color bars for both surfaces
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, label='Normalized Energy (CO)')
fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=10, label='Normalized Energy (SM)')

# Mark optimal point
ax.scatter(m[opt_m], latency[opt_l], opt_energy, color='red', s=100, 
           label=f'Optimal Configuration\n(m={m[opt_m]:.1f}, L={latency[opt_l]:.1f} ms)')

# Customize the z axis
ax.set_zlim(0, 1.0)
ax.zaxis.set_major_locator(LinearLocator(6))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Set labels and title
ax.set_xlabel('Stacking Depth (m)', labelpad=10)
ax.set_ylabel('Latency (ms)', labelpad=10)
ax.set_zlabel('Normalized Energy', labelpad=10)
ax.set_title('Energy-Latency Trade-off vs. Stacking Depth', y=1.02, fontsize=14, fontweight='bold')

# Add legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=10)

# Set viewing angle for better visualization
ax.view_init(elev=25, azim=-45)

# Save the figure
output_path = os.path.join('graphs', 'energy_latency_3d_surface.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"3D surface plot saved as '{output_path}'")
