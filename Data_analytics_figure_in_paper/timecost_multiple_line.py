import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cm import get_cmap
from datasets.generateData import load_data  # Ensure this module and function are correctly defined

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
    "Average Linkage Method Distance": "ALMD",
    "Visibility Index": "VI",
    "Grid Density Overlap Degree": "GDOD",
    "Pairwise Bounding Box Overlap Degree": "PBBOD",
    "MPix": "MP",
    "Nearest Neighbor Distance": "NND",
    "Kernel Density Overlap Degree": "KDOD"
}
# Ensure all method columns are numeric
columns_to_convert = [col for col in method_abbreviations.keys() if col in df.columns]
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce').round(2)
cleaned_labels = method_abbreviations.values()

# Group by 'The Entire Data Number' and compute the mean for each method
time_cost_attributes = [f"{x} Calculation Time" for x in method_abbreviations.keys()]
grouped = df.groupby('Number of Entire Data')[time_cost_attributes].mean().reset_index()

# Plot multi-line chart with a better colormap

plt.figure(figsize=(7, 8))
colormap = get_cmap('tab20')  # Use a colormap with distinct colors
colors = colormap(np.linspace(0, 1, len(time_cost_attributes)))

for attr, label, color in zip(time_cost_attributes, cleaned_labels, colors):
    plt.plot(grouped['Number of Entire Data'], grouped[attr], label=label, linewidth=3, color=color)

plt.xlabel('Number of Data Items', fontsize=14)
plt.ylabel('Average Calculation Time', fontsize=14)
plt.title('Time Cost vs. Number of Data Items for Each Method', fontsize=16)
plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
for y in range(0, 15, 2):  # Use range to simplify the loop
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=1)
plt.show()
