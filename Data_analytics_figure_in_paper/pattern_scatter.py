import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.stats import linregress

# Let's create synthetic data for 4 different patterns: grouping, correlation, outlier, and trend.

# Grouping: Create clustered data
X_grouping, _ = make_blobs(n_samples=30, centers=2, cluster_std=0.60, random_state=0)

# Correlation: Create a positive linear correlation
np.random.seed(0)
x_corr = np.random.rand(30)
y_corr = 2 * x_corr + np.random.normal(0, 0.1, 30)  # Strong positive correlation

# Outlier: Create data with an outlier
x_outlier = np.random.rand(30)
y_outlier = 2 * x_outlier + np.random.normal(0, 0.1, 30)

# Add an outlier
x_outlier = np.append(x_outlier, 1.5)
y_outlier = np.append(y_outlier, 0.5)

# Trend: Generate a trend line with some noise
x_trend = np.random.rand(30)
y_trend = 2 * x_trend + np.random.normal(0, 0.1, 30)


# Homoscedastic data (constant spread)
x_homo = np.random.rand(30)
y_homo = 2 * x_homo + np.random.normal(0, 0.1, 30)

# Heteroscedastic data (spread increases with x)
x_hetero = np.random.rand(30)
y_hetero = 2 * x_hetero + np.random.normal(0, x_hetero * 0.5, 30)  # moderate increase in spread

# Plotting the scatter plots
fig, axs = plt.subplots(2, 3, figsize=(9, 6))  # Adjusting for 3 columns

# Grouping
axs[0, 0].scatter(X_grouping[:, 0], X_grouping[:, 1], color=['orange' if i < 15 else 'black' for i in range(30)])
axs[0, 0].set_title("Clusters")

# Correlation
axs[0, 1].scatter(x_corr, y_corr, color='black')
axs[0, 1].set_title("Correlation")

# Outlier
axs[1, 0].scatter(x_outlier[:-1], y_outlier[:-1], color='black')
axs[1, 0].scatter(x_outlier[-1], y_outlier[-1], color='orange')  # Outlier point
axs[1, 0].set_title("Outlier")

# Trend
axs[0, 2].scatter(x_trend, y_trend, color='black')
slope, intercept, _, _, _ = linregress(x_trend, y_trend)
axs[0, 2].plot(x_trend, intercept + slope*x_trend, color='orange')
axs[0, 2].set_title("Trend")

# Homoscedastic spread
axs[1, 1].scatter(x_homo, y_homo, color='black')
axs[1, 1].set_title("Homoscedastic Spread")

# Heteroscedastic spread
axs[1, 2].scatter(x_hetero, y_hetero, color='black')
axs[1, 2].set_title("Heteroscedastic Spread")

# Layout adjustments
plt.tight_layout()
plt.show()