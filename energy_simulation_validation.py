import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from scipy import stats

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

# Generate sample data (replace with actual data)
theoretical_energy = np.linspace(10, 100, 30)
# Add some noise to simulated data
np.random.seed(42)
noise = np.random.normal(0, 3, len(theoretical_energy))

# Cache-optimized (CO) data
simulated_energy_co = theoretical_energy * 0.95 + noise
# Shared-memory (SM) data
simulated_energy_sm = theoretical_energy * 0.98 + noise * 1.2

# Calculate regression line for CO
slope_co, intercept_co, r_value_co, _, _ = stats.linregress(
    theoretical_energy, simulated_energy_co)
line_co = slope_co * theoretical_energy + intercept_co

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot data points
co_scatter = ax.scatter(theoretical_energy, simulated_energy_co, 
                       c='#0072B2', s=60, alpha=0.8, label='Cache-Optimized (CO)', 
                       edgecolors='w', linewidth=0.5)
sm_scatter = ax.scatter(theoretical_energy, simulated_energy_sm, 
                       c='#E69F00', s=60, marker='s', alpha=0.8, 
                       label='Shared-Memory (SM)', edgecolors='w', linewidth=0.5)

# Plot regression line
regression_line, = ax.plot(theoretical_energy, line_co, 'k--', 
                          linewidth=1.5, label=f'Regression (R² = {r_value_co**2:.3f})')

# Plot 1:1 reference line
min_val = min(min(theoretical_energy), min(simulated_energy_co), min(simulated_energy_sm))
max_val = max(max(theoretical_energy), max(simulated_energy_co), max(simulated_energy_sm))
ax.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.3, linewidth=1, 
        label='Ideal 1:1 Line')

# Label outliers (points far from regression line)
residuals_co = np.abs(simulated_energy_co - line_co)
outlier_threshold = np.percentile(residuals_co, 90)  # Top 10% as outliers
for i, (x, y, res) in enumerate(zip(theoretical_energy, simulated_energy_co, residuals_co)):
    if res > outlier_threshold:
        ax.annotate(f'{i+1}', (x, y), 
                   xytext=(5, 5), textcoords='offset points',
                   color='red', fontsize=8, weight='bold')

# Set labels and title
ax.set_xlabel('Theoretical Energy (mJ)', fontsize=11, labelpad=8)
ax.set_ylabel('Simulated Energy (mJ)', fontsize=11, labelpad=8)
ax.set_title('Analytical vs. Hardware-Simulated Energy for Stacked CNN Kernels', 
             fontsize=12, pad=15, fontweight='bold')

# Set axis limits with some padding
padding = 0.1 * (max_val - min_val)
ax.set_xlim(min_val - padding, max_val + padding)
ax.set_ylim(min_val - padding, max_val + padding)

# Add grid
ax.grid(True, linestyle='--', alpha=0.3)

# Add legend
legend = ax.legend(loc='upper left', frameon=True, framealpha=0.9, 
                  edgecolor='0.8', fontsize=9)
legend.get_frame().set_linewidth(0.8)

# Add R² text
r_squared = r_value_co**2
ax.text(0.02, 0.98, f'$R^2 = {r_squared:.3f}$', 
        transform=ax.transAxes, ha='left', va='top', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

# Save the figure
output_path = os.path.join('graphs', 'energy_simulation_validation.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Energy simulation validation plot saved as '{output_path}'")
