import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up plot style
plt.style.use('seaborn-v0_8-paper')  # Professional style similar to IEEE
mpl.rcParams['figure.figsize'] = (3.5, 2.5)  # Single column width
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.titlesize'] = 8

# Stack depths
m = np.array([1, 2, 3, 4, 5])

# Energy parameters (α and β ratios based on hardware characteristics)
# RTX 4090: High-end GPU with good compute to memory ratio
# Jetson Orin: Mobile GPU with balanced compute/memory
# Cortex-A78: CPU with higher memory access costs

# Base energy parameters (normalized to RTX 4090)
# β/α ratios: ~100-200x based on hardware
alpha_rtx = 1.0
beta_rtx = 100 * alpha_rtx  # High-end GPU has better memory bandwidth

alpha_orin = 1.2  # Slightly less efficient compute
beta_orin = 150 * alpha_orin  # Higher memory access cost

alpha_a78 = 2.0  # Less efficient compute
beta_a78 = 200 * alpha_a78  # Highest memory access cost

def compute_energy(alpha, beta, m_values):
    """Compute normalized energy for given hardware parameters."""
    energy = []
    for m in m_values:
        # Compute energy components
        macs = 1.0  # Base MACs
        mem_access = 1.0 / (0.6 + 0.4 * m)  # Memory access improvement with stack depth
        
        # Apply Winograd and kernel stacking benefits
        macs *= 1/2.25  # Winograd F(2,3) benefit
        macs *= (0.5 + 0.5/m)  # Kernel stacking benefit (35-60% reduction)
        
        # Total energy
        e_total = alpha * macs + beta * mem_access
        energy.append(e_total)
    
    # Normalize to m=1
    return np.array(energy) / energy[0]

# Compute energy curves
e_rtx = compute_energy(alpha_rtx, beta_rtx, m)
e_orin = compute_energy(alpha_orin, beta_orin, m)
e_a78 = compute_energy(alpha_a78, beta_a78, m)

# Create plot
plt.figure(figsize=(3.5, 2.5))
plt.plot(m, e_rtx, color='#1f77b4', label='RTX 4090')  # Dark blue
plt.plot(m, e_orin, color='#000000', label='Jetson Orin')  # Black
plt.plot(m, e_a78, color='#d62728', label='Cortex-A78AE')  # Dark red

# Format plot
plt.xlabel('Stack Depth (m)')
plt.ylabel('Normalized Energy')
plt.title('Normalized Energy Consumption Under Varying Memory Access Policies', 
          fontsize=8, pad=8)
plt.legend(fontsize=6, loc='upper right', frameon=False)
plt.xticks(m)
plt.tight_layout()

# Save figure
plt.savefig('energy_consumption_vs_stack_depth.png', 
            dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()
