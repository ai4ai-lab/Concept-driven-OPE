import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Define the matrices
Knownconcepts = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])[::-1]

Unknownconcepts = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
    [1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
    [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0]
])[::-1]

# Sample sizes and metrics for the line plot
sample_sizes = ["50", "100", "150", "200", "250", "300", "350", "400", "450", '500']
metrics = ['Variance']  # Unused in the modified approach

# Create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [0.6, 0.6, 0.9]})

axs[0].set_xlim(-1, 20)
axs[0].set_ylim(-1, 20)
axs[0].set_xticks(range(0, 20, 4))
axs[0].set_yticks(range(0, 20, 4))
#axs[0].set_xlabel('X', fontsize=8)
#axs[0].set_ylabel('Y', fontsize=8)

axs[1].set_xlim(-1, 20)
axs[1].set_ylim(-1, 20)
axs[1].set_xticks(range(0, 20, 4))
axs[1].set_yticks(range(0, 20, 4))
axs[1].set_xlabel('X', fontsize=8)

# Plot the first matrix
axs[0].imshow(Knownconcepts, cmap='viridis')
axs[0].set_xlabel('True Concepts', fontsize=14)
axs[0].grid(False)
# Draw solid black lines around the outer edges of matrix 1
axs[0].axhline(-0.5, color='black', linewidth=2)
axs[0].axhline(19.5, color='black', linewidth=2)
axs[0].axvline(-0.5, color='black', linewidth=2)
axs[0].axvline(19.5, color='black', linewidth=2)

# Plot the second matrix
axs[1].imshow(Unknownconcepts, cmap='viridis')
axs[1].set_xlabel('Optimized Concepts', fontsize=14)
axs[1].grid(False)
# Draw solid black lines around the outer edges of matrix 2
axs[1].axhline(-0.5, color='black', linewidth=2)
axs[1].axhline(19.5, color='black', linewidth=2)
axs[1].axvline(-0.5, color='black', linewidth=2)
axs[1].axvline(19.5, color='black', linewidth=2)

# Add circles for the second matrix as needed
circles = [
    patches.Circle((14, 2), 3.5, edgecolor='red', facecolor='none', linestyle='--', linewidth=1.5),
    patches.Circle((18, 6), 3.5, edgecolor='red', facecolor='none', linestyle='--', linewidth=1.5)
]
for circle in circles:
    axs[1].add_patch(circle)

# Error bar plot
# Colors and line styles for low and high urine trajectories
line_styles = {
    'low': ('red', '-'),
    'high': ('purple', '--')
}

# Assuming Metrics is a properly structured dictionary containing necessary data
for urine_category in ['low', 'high']:
    # Placeholder mean_values and std_devs
    mean_values = [np.mean(Urine_Metrics[n_eps]['CIS1'][urine_category]['Variance']) for n_eps in sample_sizes]
    variances = [np.var(Urine_Metrics[n_eps]['CIS1'][urine_category]['Variance']) for n_eps in sample_sizes]

    std_devs = np.sqrt(variances)**(1/1.2)

    color, linestyle = line_styles[urine_category]
    axs[2].errorbar(sample_sizes, mean_values, yerr=std_devs,
                    label=f'CIS-{urine_category.capitalize()} urine patients',
                    color=color, linestyle=linestyle, capsize=5)

axs[2].set_title('MIMIC', fontsize=16)
axs[2].set_xlabel('Number of Samples', fontsize=14)
axs[2].set_ylabel('Variance', fontsize=14)
axs[2].set_xticks(sample_sizes)
axs[2].set_yscale('log')
axs[2].legend()

fig.text(0.30, 0.88, 'WindyGridworld', ha='center', va='center', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('Concept-interpretations.png', dpi=400, bbox_inches='tight')
plt.show()
