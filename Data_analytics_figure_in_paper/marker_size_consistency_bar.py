import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data

# Define quality metric columns (non-time)
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

our_methods = [
    'M_distance', 'LOF Distance', 'Isolation Forest Distance',
    'Centroid Distance', 'Average Linkage Method Distance',
    'Complete Linkage Method Distance', 'Single Linkage Method Distance',
    'Leverage Score Distance', 'Cook Distance',
    'Orthogonal Distance to Lowess Line', 'Vertical Distance to Lowess Line',
    'Horizontal Distance to Lowess Line'
]

positive_change_methods = ['Pairwise Bounding Box Overlap Degree', 'MPix']
negative_change_methods = our_methods + [
    'Visibility Index', 'Grid Density Overlap Degree',
    'Kernel Density Overlap Degree', 'Nearest Neighbor Distance'
]

# Load data
file_location = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file_location_1 = os.path.join(os.path.dirname(__file__), '..', 'datasets/DimRed_data/metadata_metrics_DimRed.csv')
data_1 = load_data(file_location)
data_2 = load_data(file_location_1)

# Combine datasets
df = pd.concat([data_1, data_2], ignore_index=True)

# Ensure all method columns are numeric
df[quality_metrics] = df[quality_metrics].apply(pd.to_numeric, errors='coerce')

# Filter only scatterplots with 'category_based' in Scatterplot Name
df = df[df['Scatterplot Name'].str.contains('category_based', na=False)]

# Keep only datasets that have at least 2 unique marker sizes
filtered_data = []
for dataset in df["Dataset Name"].unique():
    df_dataset = df[df["Dataset Name"] == dataset]
    if df_dataset["Marker Size"].nunique() >= 2:
        filtered_data.append(df_dataset)

df = pd.concat(filtered_data)

# Count total datasets used for consistency checking
total_datasets = df["Dataset Name"].nunique()

# Check consistency for each method
method_consistency_percentage = {}

for method in quality_metrics:
    count = 0
    for dataset in df["Dataset Name"].unique():
        df_dataset = df[df["Dataset Name"] == dataset].sort_values(by="Marker Size")

        x = df_dataset["Marker Size"].values
        y = df_dataset[method].values

        if method in positive_change_methods:
            if np.all(np.diff(y) > 0):
                count += 1
        elif method in negative_change_methods:
            if np.all(np.diff(y) < 0):
                count += 1

    # Convert count to percentage
    percentage = (count / total_datasets) * 100
    method_consistency_percentage[method] = percentage

# Prepare data for bar chart
methods = list(method_consistency_percentage.keys())
percentages = list(method_consistency_percentage.values())
clean_labels = [m.replace(" Distance", "").replace("M_", "M ") for m in methods]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(methods))

bars = ax.bar(x, percentages, color='steelblue')
ax.set_xticks(x)
ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=10)
ax.set_ylabel("Percentage of Consistent Data Items (%)", fontsize=12)
ax.set_xlabel("Method", fontsize=12)
ax.set_title("Consistency with Marker Size Trend (category_based only)", fontsize=14)
ax.set_ylim(0, 100)  # Set y-axis to 0–100%

# Annotate bars
for i, v in enumerate(percentages):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

fig.tight_layout()
plt.show()
