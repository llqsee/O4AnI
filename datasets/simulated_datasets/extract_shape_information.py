import os
import sys
import pandas as pd

# Ensure relative import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datasets.generateData import load_data  # Assumes this works as intended

# Load metadata file
file_location = os.path.join(os.path.dirname(__file__), 'metadata_metrics_1.csv')
data = load_data(file_location)

# Extract unique dataset names
unique_datasets = data['Dataset Name'].unique()

# Create a DataFrame with dataset names and dummy shape (or replace with actual logic)
output_rows = []
for name in unique_datasets:
    # Example: If you want to actually load dataset files and get shapes, you'd load them here
    # dataset_df = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
    # shape = dataset_df.shape

    # Assuming 'Shape' column exists in the metadata and contains the shape information
    shape = data[data['Dataset Name'] == name]['Shape'].iloc[0] if 'Shape' in data.columns else 'Unknown'
    output_rows.append({'Dataset Name': name, 'Shape': shape})

output_df = pd.DataFrame(output_rows)

# Save the result
output_file_location = os.path.join(os.path.dirname(__file__), 'shape_label.csv')
output_df.to_csv(output_file_location, index=False)
print(f"Saved dataset names and shapes to: {output_file_location}")
