import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data  # Ensure this module and function are correctly defined

# Load the data
file_location = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file_location_1 = os.path.join(os.path.dirname(__file__), '..', 'datasets/DimRed_data/metadata_metrics_DimRed.csv')
data_1 = load_data(file_location)
data_2 = load_data(file_location_1)

# Concatenate the two datasets
df = pd.concat([data_1, data_2], ignore_index=True)
# df = data_2

# List of quality metric columns (NOT the time columns)
quality_metrics = [
    'M_distance', 'LOF Distance', 'Isolation Forest Distance',
    'Centroid Distance', 'Average Linkage Method Distance',
    'Complete Linkage Method Distance', 'Single Linkage Method Distance',
    'Leverage Score Distance', 'Cook Distance',
    'Orthogonal Distance to Lowess Line', 'Vertical Distance to Lowess Line',
    'Horizontal Distance to Lowess Line', 'Visibility Index',
    'Grid Density Overlap Degree', 'Pairwise Bounding Box Overlap Degree',
    'Kernel Density Overlap Degree', 'MPix', 'Nearest Neighbor Distance'
]

# Convert all method columns to numeric
df[quality_metrics] = df[quality_metrics].apply(pd.to_numeric, errors='coerce')

# Filter data: well-separated + multiple Marker Sizes
df = df[df['well-separated'] == 'yes']
df = df[df['Scatterplot Name'].str.contains('ascending', na=False)]


# df_filtered = df_filtered[df_filtered['Dataset Name'].isin(
#     [d for d in df_filtered["Dataset Name"].unique()
#      if df_filtered[df_filtered["Dataset Name"] == d]["Marker Size"].nunique() > 1]
# )]

df_filtered = df

# Identify the method columns
method_cols = quality_metrics

# Get unique datasets
datasets = df_filtered["Dataset Name"].unique()
methods = method_cols

# Create subplot matrix
fig, axes = plt.subplots(len(datasets), len(methods), figsize=(15, 20))

# Ensure correct indexing structure
if len(datasets) == 1:
    axes = [axes]
if len(methods) == 1:
    axes = [[ax] for ax in axes]

# Plotting
for i, dataset in enumerate(datasets):
    df_dataset = df_filtered[df_filtered["Dataset Name"] == dataset]
    for j, method in enumerate(methods):
        ax = axes[i][j]

        # Sort by Marker Size
        df_sorted = df_dataset.sort_values(by=["Marker Size"])
        x = df_sorted["Marker Size"]
        y = df_sorted[method]

        ax.plot(x, y, linestyle='-')
        ax.set_xticks([])
        ax.set_yticks([])

        if i == len(datasets) - 1:
            ax.set_xlabel(method, fontsize=8, rotation=90, ha='right')
        if j == 0:
            ax.set_ylabel(dataset, fontsize=8, rotation=0, labelpad=20, ha='right')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.suptitle("Line plots across Marker Size (well-separated only)", y=1.02)
plt.show()
