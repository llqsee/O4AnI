import torch
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
import os

from datasets.generateData import load_and_sample_data

# ----------------------------------------------------------
# We extract the Sample_Superstore_Order data
file_location = 'datasets/Sample_Superstore_Orders.csv'
percentage_to_extract = 0.01
data, x, y, z = load_and_sample_data(percentage_to_extract, file_location)

# Find the range of your data
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# Expand the range slightly to ensure all data points are covered
# x_range = x_max - x_min
# y_range = y_max - y_min
# x_min -= 0.1 * x_range
# x_max += 0.1 * x_range
# y_min -= 0.1 * y_range
# y_max += 0.1 * y_range






# ====================================================
# Visualize the scatterplots
# Create a scatterplot
fig = plt.figure(figsize=(10, 6), dpi = 100)
# Set the scale for both x-axis and y-axis
plt.xlim(x_min, x_max)  # Specify the range for the x-axis
plt.ylim(y_min, y_max)  # Specify the range for the y-axis
# Adjust the subplot parameters to set the size of the plot area (excluding labels)
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.scatter(x, y, c=z, cmap='viridis', marker='o', alpha=0.6)
# sns.scatterplot(x=x, y=y, hue=z, palette='viridis', marker='o', alpha=0.6, s=100)
# Add labels and a colorbar
plt.xlabel('x')
plt.ylabel('y')
# plt.colorbar(label='Pixel Height')
# Customize plot appearance as needed
plt.title('Scatterplot of '+ 'x' + ' and ' + 'y')
plt.grid(True)
# Show the plot
plt.show()