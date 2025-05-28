import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from datasets.generateData import load_data
import seaborn as sns
import matplotlib.cm as cm


# ========================================================================================
# Load the data
# Assuming your notebook is in the root directory of the project
file_location = os.path.join(os.getcwd(), 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file_location_1 = os.path.join(os.getcwd(), 'datasets', 'DimRed_data', 'metadata_metrics_DimRed.csv')
data_1 = load_data(file_location)
data_2 = load_data(file_location_1)

# Concatenate datasets
df = pd.concat([data_1, data_2], ignore_index=True)

method_abbreviations = {
    "M_distance": "MD",
    "LOF Distance": "LOFD",
    "Isolation Forest Distance": "IFD",
    "Centroid Distance": "CED",
    "Average Linkage Method Distance": "ALMD",
    "Complete Linkage Method Distance": "CLMD",
    "Single Linkage Method Distance": "SLMD",
    # "Leverage Score Distance": "LSD",
    # "Cook Distance": "COD",
    # "Orthogonal Distance to Lowess Line": "ODLL",
    # "Vertical Distance to Lowess Line": "VDLL",
    # "Horizontal Distance to Lowess Line": "HDLL",
    "Visibility Index": "VI",
    "Grid Density Overlap Degree": "GDOD",
    "Pairwise Bounding Box Overlap Degree": "PBBOD",
    "MPix": "MP",
    "Nearest Neighbor Distance": "NND",
    "Kernel Density Overlap Degree": "KDOD"
}


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


# List of quality metrics
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
               'Complete Linkage Method Distance', 'Single Linkage Method Distance'
]

other_methods = ['Leverage Score Distance', 'Cook Distance',
               'Orthogonal Distance to Lowess Line', 'Vertical Distance to Lowess Line',
               'Horizontal Distance to Lowess Line']

compare_methods = ['Visibility Index',
                   'Grid Density Overlap Degree', 'Pairwise Bounding Box Overlap Degree',
                   'Kernel Density Overlap Degree', 'MPix', 'Nearest Neighbor Distance']

positive_change_methods = ['Pairwise Bounding Box Overlap Degree', 'MPix']
negative_change_methods = our_methods + ['Visibility Index', 'Grid Density Overlap Degree',
                                         'Kernel Density Overlap Degree', 'Nearest Neighbor Distance']

# Ensure all method columns are numeric
# df[quality_metrics] = df[quality_metrics].apply(pd.to_numeric, errors='coerce')
df[quality_metrics] = df[quality_metrics].apply(pd.to_numeric, errors='coerce').round(2)




# Step 1: Filter scatterplots based on the marker size
marker_size_data = df[df['Marker Size'] == 20]

# Step 3: Calculate deviation across marker sizes
# Group by dataset
deviation_dict_marker_size = {metric: [] for metric in method_abbreviations.keys()}
for dataset, group in marker_size_data.groupby('Dataset Name'):
    for metric in method_abbreviations.keys():
        deviation_dict_marker_size[metric].append(group[metric].std(ddof=0))
        

# Step 4: Calculate average deviation across datasets
avg_deviation_values = []
for metric in method_abbreviations.keys():
    # Calculate the average deviation across all datasets
    avg_deviation = np.nanmean(deviation_dict_marker_size[metric])
    avg_deviation_values.append(avg_deviation)

# Prepare y-axis labels
y_labels = [method_abbreviations.get(name, name) for name in method_abbreviations.keys()]

# Step 5: Plot
fig, ax = plt.subplots(figsize=(6, 7))

y = np.arange(len(avg_deviation_values))
height = 0.6

bars = ax.barh(y, avg_deviation_values, height=height, color='black')

ax.set_xlabel('Average Deviation Across Rendering Orders', fontsize=14)
ax.set_ylabel('Methods', fontsize=14)
ax.set_yticks(y)
ax.set_yticklabels(y_labels, fontsize=12)

# ✅ Use log scale here:
ax.set_xscale('log')
ax.set_xlim(left=max(min(avg_deviation_values), 1e-4), right=max(avg_deviation_values) * 50)


# Annotate bars
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.4f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha='left', va='center', fontsize=10)

# plt.title('Average Deviation per Method', fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.5)
fig.tight_layout()
plt.show()