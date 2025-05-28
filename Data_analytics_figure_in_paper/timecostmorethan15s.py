import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data  # Ensure this module and function are correctly defined

# Load the data
file_location = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
data = load_data(file_location)  # Ensure load_data is properly defined and the file path is correct

# Time cost attributes
time_cost_attributes = [
    'M_distance Calculation Time', 'LOF Distance Calculation Time', 'Isolation Forest Distance Calculation Time',
    'Centroid Distance Calculation Time', 'Average Linkage Method Distance Calculation Time',
    'Complete Linkage Method Distance Calculation Time', 'Single Linkage Method Distance Calculation Time',
    'Leverage Score Distance Calculation Time', 'Cook Distance Calculation Time',
    'Orthogonal Distance to Lowess Line Calculation Time', 'Vertical Distance to Lowess Line Calculation Time',
    'Horizontal Distance to Lowess Line Calculation Time', 'Visibility Index Calculation Time',
    'Grid Density Overlap Degree Calculation Time', 'Pairwise Bounding Box Overlap Degree Calculation Time', 'Kernel Density Overlap Degree Calculation Time',
    'MPix Calculation Time', 'Nearest Neighbor Distance Calculation Time'
]

# Count how many times the calculation time exceeds 15 for each method
exceed_counts = {}
for attribute in time_cost_attributes:
    exceed_counts[attribute] = (data[attribute] >= 15).sum()

# Prepare data for plotting
methods = list(exceed_counts.keys())
counts = list(exceed_counts.values())

# Plot the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(methods))

ax.bar(x, counts, color='skyblue')
ax.set_xlabel('Method', fontsize=14)
ax.set_ylabel('Count of Calculation Time > 15', fontsize=14)
ax.set_title('Comparison of Methods by Calculation Time Exceeding 15', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=12)

fig.tight_layout()
plt.show()
