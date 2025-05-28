import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Function to load CSV data
def load_data(file_location):
    data = pd.read_csv(file_location)
    return data

# Get all CSV files in the specified directory
file_location_pattern = 'datasets/DimRed_data/Data_GlimmerMDS/*.csv'
# file_location_pattern = 'datasets/DimRed_data/Data_TSNE/*.csv'
file_locations = glob.glob(file_location_pattern)

for file_location in file_locations:
    data = load_data(file_location)  # Load data from file
    
    unique_groups = data['class'].unique()
    colors = plt.cm.get_cmap('viridis', len(unique_groups))
    
    plt.figure(figsize=(8, 6))  # Create a new figure for each file
    
    for i, group in enumerate(unique_groups):
        group_data = data[data['class'] == group]
        plt.scatter(group_data['1'], group_data['2'], c=[colors(i)], 
                    label=f"{group}", marker='o', s=50, alpha=0.5, edgecolor='black')
    
    plt.legend(title='Class')
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    
    # Ensure the directory exists
    output_dir = 'datasets/DimRed_data/Data_GlimmerMDS_figures'
    # output_dir = 'datasets/DimRed_data/Data_TSNE_figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    output_file = os.path.join(output_dir, f"{os.path.basename(file_location).replace('.csv', '.png')}")
    plt.title(f"Scatter Plot for {os.path.basename(file_location)}")
    plt.savefig(output_file)
