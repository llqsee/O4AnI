import numpy as np
import matplotlib.pyplot as plt

# Function to generate data and plot
def generate_and_plot(correlation, ax):
    # Number of samples
    n_samples = 500
    
    # Mean vector and covariance matrix
    mean = [0, 0]
    covariance = [[1, correlation], [correlation, 1]]  # Covariance matrix
    
    # Generate the data
    x, y = np.random.multivariate_normal(mean, covariance, n_samples).T
    
    # Plot
    ax.scatter(x, y)
    ax.set_title(f'Correlation: {correlation}')
    ax.grid(True)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Generate and plot data with different correlations
correlations = [-0.8, 0, 0.8]
for ax, corr in zip(axes, correlations):
    generate_and_plot(corr, ax)

plt.tight_layout()
plt.show()
