import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data

# --- Load the data ----------------------------------------------------------
base = os.getcwd()
file1 = os.path.join(base, 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file2 = os.path.join(base, 'datasets', 'DimRed_data', 'metadata_metrics_DimRed.csv')
data_1 = load_data(file1)
data_2 = load_data(file2)
df = pd.concat([data_1, data_2], ignore_index=True)

# --- Define your methods and abbreviations ---------------------------------
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

# pick columns that actually exist in df
cols = [m for m in method_abbreviations.keys() if m in df.columns]

# --- 1) Count how many “-” strings appear per method column ---------------
missing_counts = df[cols].apply(lambda col: col.astype(str).eq('-').sum())
print("Counts of ‘-’ per method column:\n", missing_counts)

# --- 2) Replace “-” with NaN, then convert to float -----------------------
df[cols] = df[cols].replace('-', np.nan).apply(pd.to_numeric, errors='coerce')

# (… now carry on with your rounding, filtering, correlation, plotting, etc. …)

# e.g. if you want to round to two decimals:
df[cols] = df[cols].round(2)

# Print the number of items (rows) in the dataframe
print(f"Number of items in the dataframe: {len(df)}")


# ===========================
# Plotting the missing counts ----------------------------------------------


# assuming you’ve already done:
# cols = [m for m in method_abbreviations.keys() if m in df.columns]
# missing_counts = df[cols].apply(lambda col: col.astype(str).eq('-').sum())

methods = missing_counts.index.tolist()
counts  = missing_counts.values

plt.figure(figsize=(10, 6))
plt.bar(methods, counts)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Method', fontsize=14)
plt.ylabel('Number of Missing (“-”) Entries', fontsize=14)
plt.title('Missing Computations per Method', fontsize=16)
plt.tight_layout()
plt.show()