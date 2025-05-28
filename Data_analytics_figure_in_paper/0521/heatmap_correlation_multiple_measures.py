import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print("Current Working Directory:", os.getcwd())

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

import seaborn as sns
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
    result_pearson = {}
    result_spearman = {}
    result_kendall = {}

    for method in method_abbreviations:
        total_count, total_pairs = 0, 0
        pearson_rs, spearman_rs, kendall_ts = [], [], []

        for dataset in df_subset["Dataset Name"].unique():
            sub = df_subset[df_subset["Dataset Name"] == dataset]
            # sort ascending/descending as before...
            x = sub["No. Covered Data Points by Different Classes"].values
            y = sub[method].values

            # --- NEW: filter out bad entries ---
            if np.isfinite(x).any():
                pass
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            # if not enough data or constant array, skip correlations
            if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
                r = rho = tau = np.nan
            else:
                r, _   = pearsonr(x, y)
                rho, _ = spearmanr(x, y)
                tau, _ = kendalltau(x, y)

            pearson_rs.append(r)
            spearman_rs.append(rho)
            kendall_ts.append(tau)

            # compute your monotonicity counts as before...
            if method in positive_change_methods:
                c, t = monotonicity_score_positive(x, y)
            else:
                c, t = monotonicity_score_negative(x, y)
            total_count  += c
            total_pairs  += t

        result[method]           = (total_count / total_pairs) * 100 if total_pairs else 0
        result_pearson[method]   = np.nanmean(pearson_rs)
        result_spearman[method]  = np.nanmean(spearman_rs)
        result_kendall[method]   = np.nanmean(kendall_ts)

    return result, result_pearson, result_spearman, result_kendall

# ========================================================================================



# ========================================================================
# Remove the data that don't change the hidden data points
# Filter based on unique values of 'No. Covered Data Points by Different Classes'
filtered_data = []

for dataset in df["Dataset Name"].unique():
    df_dataset = df[df["Dataset Name"] == dataset]
    unique_values = df_dataset["No. Covered Data Points by Different Classes"].unique()

    if unique_values.size > 1:
        # for val in unique_values:
        #     filtered_data.append(df_dataset[df_dataset["No. Covered Data Points by Different Classes"] == val].iloc[[0]])
        filtered_data.append(df_dataset)
    else:
        pass

# Concatenate the filtered data into a single DataFrame
# df = pd.concat(filtered_data, ignore_index=True)
# ============================================================================



# ========================================================================================
# Compute correlation for all methods
result_cluster,result_pearson, result_spearman, result_kendall  = compute_consistency(df)
methods = list(method_abbreviations.keys())

# --- zero out any NaNs in those dicts ---
for d in (result_pearson, result_spearman, result_kendall):
    for method, val in d.items():
        # if it’s NaN, replace with 0.0
        if np.isnan(val):
            d[method] = 0.0
# =========================================================================================



# ========================================================================================

# … after compute_consistency(df) and zeroing out NaNs in the dicts …

methods = list(method_abbreviations.keys())
abbrevs = [method_abbreviations[m] for m in methods]

scores = {
    "Pearson":  [ result_pearson[m]   for m in methods ],
    "Spearman": [ result_spearman[m]  for m in methods ],
    "Kendall":  [ result_kendall[m]   for m in methods ],
}

# ensure no empty‐slice means
for name, vec in scores.items():
    if all(np.isnan(vec)):
        scores[name] = [0.0] * len(vec)

# build DataFrame
df_scores = pd.DataFrame(scores, index=abbrevs).T

# bump seaborn base font
sns.set(font_scale=1.2)

plt.figure(figsize=(14, 4))
heat = sns.heatmap(
    df_scores,
    annot=True,
    fmt=".2f",
    vmin=-1, vmax=1,  # Set colorbar (legend) range from -1 to 1
    cmap="coolwarm",
    cbar_kws={"label": "Rank Correlation"}  # only 'label' here
)

# enlarge annotation text
for text in heat.texts:
    text.set_fontsize(14)

# axis labels & title
heat.set_xlabel("Method",  fontsize=16)
heat.set_ylabel("Measure", fontsize=16)
heat.set_title("Correlation Heatmap", fontsize=18, pad=12)

# tick labels
heat.set_xticklabels(heat.get_xticklabels(), rotation=45, ha="right", fontsize=14)
heat.set_yticklabels(heat.get_yticklabels(), rotation=0,            fontsize=14)

# now grab and resize the colorbar
cbar = heat.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)        # tick labels
cbar.ax.yaxis.label.set_size(14)         # the colorbar label

plt.tight_layout()
plt.show()
# =========================================================================================