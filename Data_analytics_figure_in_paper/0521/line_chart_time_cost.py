import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print("Current Working Directory:", os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cm import get_cmap
from datasets.generateData import load_data  # Ensure this module and function are correctly defined

# ========================================================================================
# Load the data (UNTOUCHED)
# ========================================================================================
file_location   = os.path.join(os.getcwd(), 'datasets', 'simulated_datasets',    'metadata_metrics.csv')
file_location_1 = os.path.join(os.getcwd(), 'datasets', 'DimRed_data',          'metadata_metrics_DimRed.csv')
data_1 = load_data(file_location)
data_2 = load_data(file_location_1)

df = pd.concat([data_1, data_2], ignore_index=True)

# ========================================================================================
# Mapping & Conversion
# ========================================================================================
method_abbreviations = {
    "M_distance": "MD-AnI",
    "LOF Distance": "LOF-AnI",
    "Isolation Forest Distance": "IFD-AnI",
    # "Centroid Distance": "CED",
    "Average Linkage Method Distance": "ALMD-AnI",
    # "Complete Linkage Method Distance": "CLMD",
    # "Single Linkage Method Distance": "SLMD",
    "Visibility Index": "VI",
    "Grid Density Overlap Degree": "GDOD",
    "Pairwise Bounding Box Overlap Degree": "PBBOD",
    "MPix": "MP",
    "Nearest Neighbor Distance": "NND",
    "Kernel Density Overlap Degree": "KDOD"
}

# 1) build calc-time column → abbrev map, but only for existing columns
calc_map = {
    f"{method} Calculation Time": abbrev
    for method, abbrev in method_abbreviations.items()
    if f"{method} Calculation Time" in df.columns
}

# 2) convert & round those columns
df[list(calc_map.keys())] = (
    df[list(calc_map.keys())]
    .apply(pd.to_numeric, errors='coerce')
    .round(2)
)

# 3) rename to short labels
df = df.rename(columns=calc_map)
short_labels = list(calc_map.values())

# ========================================================================================
# Aggregate
# ========================================================================================
grouped = (
    df
    .groupby('Number of Entire Data', as_index=False)[short_labels]
    .mean()
)

# ========================================================================================
# Plot
# ========================================================================================
plt.figure(figsize=(10, 8))
cmap   = get_cmap('tab10')
colors = cmap(np.linspace(0, 1, len(short_labels)))
x      = grouped['Number of Entire Data']

for lbl, col in zip(short_labels, colors):
    plt.plot(x, grouped[lbl], label=lbl, linewidth=3, color=col)

plt.xlabel('Number of Data Items',     fontsize=18)
plt.ylabel('Average Calculation Time', fontsize=18)
plt.title ('Time Cost vs. Number of Data Items', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=9, bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()

# horizontal gridlines every 2 units
for y in range(0, 15, 2):
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=1)

plt.show()
