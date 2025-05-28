import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets.generateData import load_data  # Ensure this module and function are correctly defined

# Load the data
file_location = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file_location_1 = os.path.join(os.path.dirname(__file__), '..', 'datasets/DimRed_data/metadata_metrics_DimRed.csv')

data_1 = load_data(file_location)
data_2 = load_data(file_location_1)
data = pd.concat([data_1, data_2], ignore_index=True)

# Attributes
time_cost_attributes = [
    'M_distance Calculation Time', 'LOF Distance Calculation Time', 'Isolation Forest Distance Calculation Time',
    'Centroid Distance Calculation Time', 'Average Linkage Method Distance Calculation Time',
    'Complete Linkage Method Distance Calculation Time', 'Single Linkage Method Distance Calculation Time',
    'Leverage Score Distance Calculation Time', 'Cook Distance Calculation Time',
    'Orthogonal Distance to Lowess Line Calculation Time', 'Vertical Distance to Lowess Line Calculation Time',
    'Horizontal Distance to Lowess Line Calculation Time', 'Visibility Index Calculation Time',
    'Grid Density Overlap Degree Calculation Time', 'Pairwise Bounding Box Overlap Degree Calculation Time',
    'Kernel Density Overlap Degree Calculation Time', 'MPix Calculation Time', 'Nearest Neighbor Distance Calculation Time'
]

cleaned_labels = [attr.replace(' Calculation Time', '') for attr in time_cost_attributes]

# Group by 'The Entire Data Number' and compute the mean for each method
grouped = data.groupby('Number of Entire Data')[time_cost_attributes].mean().reset_index()

# Plot multi-line chart
plt.figure(figsize=(12, 8))
for attr, label in zip(time_cost_attributes, cleaned_labels):
    plt.plot(grouped['Number of Entire Data'], grouped[attr], label=label)

plt.xlabel('Number of Data Items', fontsize=14)
plt.ylabel('Average Calculation Time', fontsize=14)
plt.title('Time Cost vs. Number of Data Items for Each Method', fontsize=16)
plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()
