import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("Current Working Directory:", os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets.generateData import load_data  # Ensure this module and function are correctly defined

# Load the main metadata file
file_location = os.path.join(os.path.dirname(__file__), 'metadata_metrics_DimRed_fu.csv')
data = load_data(file_location)  # Ensure load_data is properly defined and the file path is correct

# Load the well-separated labels
index_location_1 = os.path.join(os.path.dirname(__file__), 'seperation_label_Data_GlimmerMDS.csv')
index_location_2 = os.path.join(os.path.dirname(__file__), 'seperation_label_Data_TSNE.csv')
index_data_1 = load_data(index_location_1)
index_data_2 = load_data(index_location_2)

index_data = pd.concat([index_data_1, index_data_2], ignore_index=True)

# Load the shape labels
shape_label_location = os.path.join(os.path.dirname(__file__), 'shape_label.csv')
shape_data = load_data(shape_label_location)

# Ensure required columns exist in the datasets
if (
    'Dataset Name' in data.columns and
    'dataset name' in index_data.columns and
    'well-separated' in index_data.columns and
    'Dataset Name' in shape_data.columns and
    'Shape' in shape_data.columns
):
    # Process well-separated labels
    index_data['dataset name'] = index_data['dataset name'].str.replace('.csv', '', regex=False)
    well_separated_map = index_data.set_index('dataset name')['well-separated'].to_dict()
    data['well-separated'] = data['Dataset Name'].map(well_separated_map)
    
    # Process shape labels
    shape_data['Dataset Name'] = shape_data['Dataset Name'].str.replace('.csv', '', regex=False)
    shape_map = shape_data.set_index('Dataset Name')['Shape'].to_dict()
    data['Shape'] = data['Dataset Name'].map(shape_map)
else:
    print("Required columns are missing in the datasets.")

# Save the updated data to metadata_metrics_DimRed_1.csv
output_file_location = os.path.join(os.path.dirname(__file__), 'metadata_metrics_DimRed.csv')
data.to_csv(output_file_location, index=False)
print(f"Data saved to {output_file_location}")