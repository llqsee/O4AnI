import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data

# Load the data
file_location = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file_location_1 = os.path.join(os.path.dirname(__file__), '..', 'datasets/DimRed_data/metadata_metrics_DimRed.csv')
data_1 = load_data(file_location)
data_2 = load_data(file_location_1)

# Concatenate datasets
df = pd.concat([data_1, data_2], ignore_index=True)

# List of quality metrics (non-time columns)
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

our_methods = ['M_distance', 'LOF Distance', 'Isolation Forest Distance',
    'Centroid Distance', 'Average Linkage Method Distance',
    'Complete Linkage Method Distance', 'Single Linkage Method Distance',
    'Leverage Score Distance', 'Cook Distance',
    'Orthogonal Distance to Lowess Line', 'Vertical Distance to Lowess Line',
    'Horizontal Distance to Lowess Line']

compare_methods = ['Visibility Index',
    'Grid Density Overlap Degree', 'Pairwise Bounding Box Overlap Degree',
    'Kernel Density Overlap Degree', 'MPix', 'Nearest Neighbor Distance']

positive_change_methods = ['Pairwise Bounding Box Overlap Degree', 'Mpix']

negative_change_methods = our_methods + ['Visibility Index', 'Grid Density Overlap Degree', 'Kernel Density Overlap Degree', 'Nearest Neighbor Distance']

# Ensure all method columns are numeric
df[quality_metrics] = df[quality_metrics].apply(pd.to_numeric, errors='coerce')



# Filter to marker size = 50 and well-separated datasets
# df = df[df['Shape'].str.contains('linear', na=False) | df['Shape'].str.contains('three-order', na=False)]  # Filter for linear shapes
# df_50 = df[df["Marker Size"] == 10]
df =df[df['well-separated'] == 'yes']  # Optional: uncomment to filter well-separated datasets
df_50 = df

# Filter based on unique values of 'No. Covered Data Points by Different Classes'
filtered_data = []

for dataset in df_50["Dataset Name"].unique():
    df_dataset = df_50[df_50["Dataset Name"] == dataset]
    unique_values = df_dataset["No. Covered Data Points by Different Classes"].unique()

    if unique_values.size >1:
        for val in unique_values:
            filtered_data.append(df_dataset[df_dataset["No. Covered Data Points by Different Classes"] == val].iloc[[0]])
    else:
        pass

df_50 = pd.concat(filtered_data)

# Total number of datasets being evaluated
total_datasets = df_50["Dataset Name"].nunique()

# Check consistency for each method and calculate percentage
method_consistency_percentage = {}

for method in quality_metrics:
    count = 0
    for dataset in df_50["Dataset Name"].unique():
        df_dataset = df_50[df_50["Dataset Name"] == dataset]
        df_sorted = df_dataset.sort_values(by="No. Covered Data Points by Different Classes")

        x = df_sorted["No. Covered Data Points by Different Classes"].values
        y = df_sorted[method].values

        if method in positive_change_methods:
            if np.all(np.diff(y) > 0):
                count += 1
        elif method in negative_change_methods:
            if np.all(np.diff(y) < 0):
                count += 1

    percentage = (count / total_datasets) * 100
    method_consistency_percentage[method] = percentage

# Prepare data for bar chart
methods = list(method_consistency_percentage.keys())
percentages = list(method_consistency_percentage.values())

# Clean method names
# clean_labels = [m.replace(" Distance", "").replace("M_", "M ") for m in methods]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
x = np.arange(len(methods))

ax.bar(x, percentages, color='steelblue')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
ax.set_ylabel("Percentage of data size being consitency (%) ", fontsize=14)
ax.set_xlabel("Methods", fontsize=14)
ax.set_title("Consistency with number of covered data points", fontsize=14)
ax.set_ylim(0, 105)  # Show full percentage range

# Annotate bars
for i, v in enumerate(percentages):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)

fig.tight_layout()
plt.show()
