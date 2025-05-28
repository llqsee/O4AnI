import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust path to access load_data
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.generateData import load_data

# Load the data
file_location = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file_location_1 = os.path.join(os.path.dirname(__file__), '..', 'datasets/DimRed_data/metadata_metrics_DimRed.csv')
# file_location = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'simulated_datasets', 'metadata_metrics_sepme.csv')
data_1 = load_data(file_location)  # Ensure load_data is properly defined and the file path is correct
data_2 = load_data(file_location_1)  # Ensure load_data is properly defined and the file path is correct

data = pd.concat([data_1, data_2], ignore_index=True)

# Time cost attributes
time_cost_attributes = [
    'M_distance Calculation Time', 'LOF Distance Calculation Time', 'Isolation Forest Distance Calculation Time',
    'Centroid Distance Calculation Time', 'Average Linkage Method Distance Calculation Time',
    'Complete Linkage Method Distance Calculation Time', 'Single Linkage Method Distance Calculation Time',
    'Leverage Score Distance Calculation Time', 'Cook Distance Calculation Time',
    'Orthogonal Distance to Lowess Line Calculation Time', 'Vertical Distance to Lowess Line Calculation Time',
    'Horizontal Distance to Lowess Line Calculation Time', 'Visibility Index Calculation Time',
    'Grid Density Overlap Degree Calculation Time', 'Pairwise Bounding Box Overlap Degree Calculation Time', 
    'Kernel Density Overlap Degree Calculation Time', 'MPix Calculation Time', 
    'Nearest Neighbor Distance Calculation Time'
]

# Combine and bin data
bin_edges = np.arange(0, 21, 1)  # time cost bins: 0–1s, 1–2s, ..., 20s
bin_labels = [f'{int(b)}–{int(b+1)}' for b in bin_edges[:-1]]

# Initialize DataFrame to collect binned counts
binned_df = pd.DataFrame(index=bin_labels, columns=time_cost_attributes)

for method in time_cost_attributes:
    # Bin the data for this method
    counts, _ = np.histogram(data[method].dropna(), bins=bin_edges)
    binned_df[method] = counts

# Convert to numeric in case of dtype issues
binned_df = binned_df.fillna(0).astype(int)

# Transpose for heatmap: rows = methods, columns = time bins
heatmap_data = binned_df.T

# Plot heat bar (2D histogram)
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={'label': 'Number of Instances'})

plt.title("Heat Bar: Instance Counts by Time Cost and Method", fontsize=16)
plt.xlabel("Time Cost Bin (seconds)", fontsize=14)
plt.ylabel("Method", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
