import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data
import ast



# ========================================================================================
# Load the data
# Assuming your notebook is in the root directory of the project
file_location = os.path.join(os.getcwd(), 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file_location_1 = os.path.join(os.getcwd(), 'datasets', 'DimRed_data', 'metadata_metrics_DimRed.csv')
data_1 = load_data(file_location)
data_2 = load_data(file_location_1)

# Concatenate datasets
df = pd.concat([data_1, data_2], ignore_index=True)

# method_abbreviations = {
#     "M_distance": "MD",
#     "LOF Distance": "LOFD",
#     "Isolation Forest Distance": "IFD",
#     "Centroid Distance": "CED",
#     "Average Linkage Method Distance": "ALMD",
#     "Complete Linkage Method Distance": "CLMD",
#     "Single Linkage Method Distance": "SLMD",
#     "Leverage Score Distance": "LSD",
#     "Cook Distance": "COD",
#     "Orthogonal Distance to Lowess Line": "ODLL",
#     "Vertical Distance to Lowess Line": "VDLL",
#     "Horizontal Distance to Lowess Line": "HDLL",
#     "Visibility Index": "VI",
#     "Grid Density Overlap Degree": "GDOD",
#     "Pairwise Bounding Box Overlap Degree": "PBBOD",
#     "MPix": "MP",
#     "Nearest Neighbor Distance": "NND",
#     "Kernel Density Overlap Degree": "KDOD"
# }


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



our_methods = ['M_distance', 'LOF Distance', 'Isolation Forest Distance',
               'Centroid Distance', 'Average Linkage Method Distance',
               'Complete Linkage Method Distance', 'Single Linkage Method Distance',
               'Leverage Score Distance', 'Cook Distance',
               'Orthogonal Distance to Lowess Line', 'Vertical Distance to Lowess Line',
               'Horizontal Distance to Lowess Line'
]

compare_methods = ['Visibility Index',
                   'Grid Density Overlap Degree', 'Pairwise Bounding Box Overlap Degree',
                   'Kernel Density Overlap Degree', 'MPix', 'Nearest Neighbor Distance']

positive_change_methods = ['Pairwise Bounding Box Overlap Degree', 'MPix']
negative_change_methods = our_methods + ['Visibility Index', 'Grid Density Overlap Degree',
                                         'Kernel Density Overlap Degree', 'Nearest Neighbor Distance']


# Ensure all method columns are numeric
columns_to_convert = [col for col in method_abbreviations.keys() if col in df.columns]
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce').round(2)
cleaned_labels = method_abbreviations.values()
# ==========================================================================================


# ========================================================================================
# Functions to compute monotonicity scores
def monotonicity_score_positive(x, y):
    n = len(x)
    count = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] <= x[j]:
                total += 1
                if y[i] < y[j]:
                    count += 1
    return count, total

def monotonicity_score_negative(x, y):
    n = len(x)
    count = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] <= x[j]:
                total += 1
                if y[i] > y[j]:
                    count += 1
    return count, total



def compute_consistency(df_subset):
    result = {}
    for method in method_abbreviations.keys():
        count_consistency = 0
        count_total = 0
        for dataset in df_subset["Dataset Name"].unique():
            df_dataset = df_subset[df_subset["Dataset Name"] == dataset]
            if method in positive_change_methods:
                df_sorted = df_dataset.sort_values(
                    by=["No. Covered Data Points by Different Classes", method],
                    ascending=[True, True]
                )
            elif method in negative_change_methods:
                df_sorted = df_dataset.sort_values(
                    by=["No. Covered Data Points by Different Classes", method],
                    ascending=[True, False]
                )
            x = df_sorted["No. Covered Data Points by Different Classes"].values
            y = df_sorted[method].values

            if method in positive_change_methods:
                count, total = monotonicity_score_positive(x, y)
            elif method in negative_change_methods:
                count, total = monotonicity_score_negative(x, y)

            count_consistency += count
            count_total += total
        percentage = (count_consistency / count_total) * 100 if count_total > 0 else 0
        result[method] = percentage
    return result
# ========================================================================================



# ===========================================================================================
# Filter to well-separated datasets
df_1class1cluster = df[df['well-separated'] == '1-class-1-cluster']
df_1classmcluster = df[df['well-separated'] == '1-class-m-cluster']
# ========================================================================================


# ========================================================================================
# Compute consistency for both shape groups for the both well-separated data and not-well-separted data
shape_result_cluster = compute_consistency(df_1class1cluster)
shape_result_other = compute_consistency(df_1classmcluster)
# shape_result_cluster = compute_consistency(df_only_cluster)
# shape_result_other = compute_consistency(df_contains_other)

# Prepare for horizontal grouped bar chart
methods = list(method_abbreviations.keys())
cluster_scores = [shape_result_cluster[m] for m in methods]
other_scores = [shape_result_other[m] for m in methods]

# ========================================================================================
# Plot horizontal grouped bar chart
fig, ax = plt.subplots(figsize=(8, 10))

y = np.arange(len(methods))
height = 0.35

bars1 = ax.barh(y - height/2, cluster_scores, height, label='1-class-1-cluster', color='cornflowerblue')
bars2 = ax.barh(y + height/2, other_scores, height, label='1-class-m-cluster', color='darkorange')

ax.set_xlabel('Consistency (%)', fontsize=18)
ax.set_ylabel('Methods', fontsize=18)
ax.set_yticks(y)
ax.set_yticklabels([method_abbreviations[m] for m in methods], fontsize=16)
ax.set_xlim(0, 120)
ax.legend(fontsize=14)

# Annotate bars
for bars in [bars1, bars2]:
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12)

fig.tight_layout()
plt.show()