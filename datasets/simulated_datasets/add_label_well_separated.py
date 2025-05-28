import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("Current Working Directory:", os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data  # Ensure this module and function are correctly defined

file_location = os.path.join(os.path.dirname(__file__), 'metadata_metrics.csv')
# file_location = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'simulated_datasets', 'metadata_metrics_sepme.csv')
data = load_data(file_location)  # Ensure load_data is properly defined and the file path is correct

index_location = os.path.join(os.path.dirname(__file__), 'seperation_label.csv')
index_data = load_data(index_location)

# Ensure 'dataset name' exists in both datasets and 'well-separated' exists in index_data
if 'Dataset Name' in data.columns and 'dataset name' in index_data.columns and 'well-separated' in index_data.columns:
    # Create a dictionary mapping 'dataset name' to 'well-separated' from index_data
    index_data['dataset name'] = index_data['dataset name'].str.replace('.csv', '', regex=False)
    well_separated_map = index_data.set_index('dataset name')['well-separated'].to_dict()
    
    # Update the 'well-separated' attribute in data based on the mapping
    data['well-separated'] = data['Dataset Name'].map(well_separated_map)
else:
    print("Required columns are missing in the datasets.")
    


# Save the updated data to metadata_metrics_1.csv
output_file_location = os.path.join(os.path.dirname(__file__), 'metadata_metrics.csv')
data.to_csv(output_file_location, index=False)
print(f"Data saved to {output_file_location}")