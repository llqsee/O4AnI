import matplotlib.pyplot as plt
import numpy as np

# Generating sample data for different patterns
np.random.seed(0)

# Positive linear relationship
x1 = np.linspace(0, 10, 100)
y1 = 2 * x1 + np.random.normal(0, 1, 100)

# Negative linear relationship
x2 = np.linspace(0, 10, 100)
y2 = -2 * x2 + np.random.normal(0, 1, 100)

# No correlation
x3 = np.random.uniform(0, 10, 100)
y3 = np.random.uniform(0, 10, 100)

# More obvious curvilinear relationship
x4_obvious = np.linspace(0, 10, 100)
y4_obvious = (x4_obvious - 5)**3 + np.random.normal(0, 10, 100)

# Clusters
x5_1 = np.random.normal(2, 0.5, 50)
y5_1 = np.random.normal(2, 0.5, 50)
x5_2 = np.random.normal(8, 0.5, 50)
y5_2 = np.random.normal(8, 0.5, 50)

# Outliers
x6 = np.linspace(0, 10, 100)
y6 = 2 * x6 + np.random.normal(0, 1, 100)
x6_outliers = np.array([2, 8])
y6_outliers = np.array([20, -10])

# Heteroscedasticity
x7 = np.linspace(0, 10, 100)
y7 = 2 * x7 + np.random.normal(0, 0.5 * x7, 100)

# Cyclical Patterns
x9 = np.linspace(0, 4 * np.pi, 100)
y9 = np.sin(x9) + np.random.normal(0, 0.2, 100)

# Plotting the patterns
fig, axs = plt.subplots(4, 2, figsize=(5, 15))
fig.suptitle('Visual Patterns in Scatterplots', fontsize=16)

# Common color for all points
color = 'black'

# Positive linear relationship
axs[0, 0].scatter(x1, y1, color=color)
axs[0, 0].set_title('Positive Linear Relationship')

# Negative linear relationship
axs[0, 1].scatter(x2, y2, color=color)
axs[0, 1].set_title('Negative Linear Relationship')

# No correlation
axs[1, 0].scatter(x3, y3, color=color)
axs[1, 0].set_title('No Correlation')

# More obvious curvilinear relationship
axs[1, 1].scatter(x4_obvious, y4_obvious, color=color)
axs[1, 1].set_title('Curvilinear Relationship')

# Clusters
axs[2, 0].scatter(x5_1, y5_1, color=color)
axs[2, 0].scatter(x5_2, y5_2, color=color)
axs[2, 0].set_title('Clusters')

# Outliers
axs[2, 1].scatter(x6, y6, color=color)
axs[2, 1].scatter(x6_outliers, y6_outliers, color=color)
axs[2, 1].set_title('Outliers')

# Heteroscedasticity
axs[3, 0].scatter(x7, y7, color=color)
axs[3, 0].set_title('Heteroscedasticity')

# Cyclical Patterns
axs[3, 1].scatter(x9, y9, color=color)
axs[3, 1].set_title('Cyclical Patterns')

for ax in axs.flat:
    ax.label_outer()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
