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
file_location_2 = os.path.join(os.path.dirname(__file__), '..', 'Data_analytics_figure_in_paper', 'dataset_abb.csv')
data_1 = load_data(file_location)  # Ensure load_data is properly defined and the file path is correct
data_2 = load_data(file_location_1)  # Ensure load_data is properly defined and the file path is correct
data_abb = load_data(file_location_2)  # Load the dataset abbreviation mapping

# Concatenate the two datasets
df = pd.concat([data_1, data_2], ignore_index=True)
# df = data_2  # Use only the DimRed data for this analysis


# Create the dictionary where key = DatasetName and value = DatasetID
dataset_abb = dict(zip(data_abb["DatasetName"], data_abb["DatasetID"]))



method_abbreviations = {
    "M_distance": "MD",
    "LOF Distance": "LOFD",
    "Isolation Forest Distance": "IFD",
    "Centroid Distance": "CED",
    "Average Linkage Method Distance": "ALMD",
    "Complete Linkage Method Distance": "CLMD",
    "Single Linkage Method Distance": "SLMD",
    "Visibility Index": "VI",
    "Grid Density Overlap Degree": "GDOD",
    "Pairwise Bounding Box Overlap Degree": "PBBOD",
    "MPix": "MP",
    "Nearest Neighbor Distance": "NND",
    "Kernel Density Overlap Degree": "KDOD"
}

# List of quality metric columns (NOT the time columns)
quality_metrics = list(method_abbreviations.keys())


our_methods = ['M_distance', 'LOF Distance', 'Isolation Forest Distance',
    'Centroid Distance', 'Average Linkage Method Distance',
    'Complete Linkage Method Distance', 'Single Linkage Method Distance',
    'Leverage Score Distance', 'Cook Distance',
    'Orthogonal Distance to Lowess Line', 'Vertical Distance to Lowess Line',
    'Horizontal Distance to Lowess Line']

positive_change_methods = ['Pairwise Bounding Box Overlap Degree', 'MPix']

negative_change_methods = our_methods + ['Visibility Index', 'Grid Density Overlap Degree', 'Kernel Density Overlap Degree', 'Nearest Neighbor Distance']


# Convert all method columns to numeric, non-numeric entries become NaN
df[quality_metrics] = df[quality_metrics].apply(pd.to_numeric, errors='coerce')


# Filter data where Marker Size == 50
# df = df[(df["Marker Size"] == 50)]
# df = df[df['well-separated'] == '1-class-1-cluster']  # Filter for well-separated datasets
df = df[df['Scatterplot Name'].str.contains('ascending', na=False)]
# df = df[df['Shape'].str.contains('linear', na=False) | df['Shape'].str.contains('three-order', na=False)]  # Filter for linear shapes
# Filter data based on the conditions for 'No. Covered Data Points by Different Classes'
filtered_data = []

for dataset in df["Dataset Name"].unique():
    df_dataset = df[df["Dataset Name"] == dataset]
    unique_values = df_dataset["No. Covered Data Points by Different Classes"].unique()
    
    if unique_values.size >1:
        for val in unique_values:
            filtered_data.append(df_dataset[df_dataset["No. Covered Data Points by Different Classes"] == val].iloc[[0]])
    else:
        pass

# Combine the filtered data back into a single DataFrame
df = pd.concat(filtered_data)

# Identify the method columns by finding those ending with "Distance"
method_cols = quality_metrics

# Get unique datasets
datasets = df["Dataset Name"].unique()
methods = method_cols

# Create a matrix of plots
fig, axes = plt.subplots(len(datasets), len(methods), figsize=(15, 18))

# If there's only one dataset or method, reshape axes for consistent indexing
if len(datasets) == 1:
    axes = [axes]
if len(methods) == 1:
    axes = [[ax] for ax in axes]

# Plotting
for i, dataset in enumerate(datasets):
    df_dataset = df[df["Dataset Name"] == dataset]
    for j, method in enumerate(methods):
        ax = axes[i][j]
        # Sort by "No. Covered Data Points by Different Classes" and then by the method values
        if method in positive_change_methods:
            df_dataset_sorted = df_dataset.sort_values(by=["No. Covered Data Points by Different Classes", method])
        elif method in negative_change_methods:
            df_dataset_sorted = df_dataset.sort_values(by=["No. Covered Data Points by Different Classes", method], ascending=False)
            
        x = df_dataset_sorted["No. Covered Data Points by Different Classes"]
        y = df_dataset_sorted[method]
        ax.plot(x, y, linestyle='-')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == len(datasets) - 1:  # Add x-axis label only for the last row
            ax.set_xlabel(method_abbreviations.get(method, method), fontsize=8, rotation=90, ha='right')
        # if j == 0:  # Add y-axis label only for the first column
        #     ax.set_ylabel(dataset_abb.get(dataset, dataset), fontsize=8, rotation=0, labelpad=20, ha='right')
        if j == 0:  # Add y-axis label only for the first column
            ax.text(-0.3, 0.5, dataset_abb.get(dataset, dataset), fontsize=8,
                    rotation=0, ha='right', va='center', transform=ax.transAxes)

# Adjust margins and spacing
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Increase the distance between plots
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Add margins around the entire figure
plt.suptitle("Line plots for Marker Size = 50", y=1.02)


output_file = os.path.join(os.path.dirname(__file__), 'matrix.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
