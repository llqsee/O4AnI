import matplotlib.pyplot as plt
import numpy as np

# Define grid size
n_rows, n_cols = 20, 20

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Set limits
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)

# Set major ticks at every integer
ax.set_xticks(np.arange(0, n_cols + 1, 1))
ax.set_yticks(np.arange(0, n_rows + 1, 1))

# Draw grid
ax.grid(which='both', color='black', linewidth=1)

# Hide axis labels
ax.set_xticklabels([])
ax.set_yticklabels([])
# Make ticks invisible
ax.tick_params(left=False, bottom=False)

# Set aspect of the plot to be equal
ax.set_aspect('equal')

plt.show()

