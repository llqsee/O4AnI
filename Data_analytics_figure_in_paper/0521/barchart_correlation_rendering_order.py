import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# so that `datasets.generateData` is importable:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datasets.generateData import load_data

# -----------------------------------------------------------------------------------
# 1) Load & concatenate
# ------------------------------------------------------------------------------------
cwd   = os.getcwd()
file1 = os.path.join(cwd, 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file2 = os.path.join(cwd, 'datasets', 'DimRed_data',       'metadata_metrics_DimRed.csv')
df    = pd.concat([load_data(file1), load_data(file2)], ignore_index=True)
# df.replace('-', np.nan, inplace=True)


# -----------------------------------------------------------------------------------
# 2) Remove datasets with no covered data points by different classes
#    and keep only ascending‐order renderings
# -----------------------------------------------------------------------------------
# # 2a) Filter out datasets with no covered points by different classes
# max_covered = df.groupby("Dataset Name")["No. Covered Data Points by Different Classes"].max()
# valid = max_covered[max_covered > 1].index
# df = df[df["Dataset Name"].isin(valid)]
# 2b) Keep only 20 marker size
df = df[df['Marker Size'] == 20]


# -----------------------------------------------------------------------------------
# 3) Abbreviations & numeric conversion
# -----------------------------------------------------------------------------------
method_abbrevs = {
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
to_num = [c for c in method_abbrevs if c in df.columns]
df[to_num] = df[to_num].apply(pd.to_numeric, errors='coerce').round(2)


# -----------------------------------------------------------------------------------
# 4) Compute per‐dataset Kendall τ, then take the mean across datasets
# -----------------------------------------------------------------------------------
taus = {}
for method in method_abbrevs:
    if method not in df.columns:
        taus[method] = np.nan
        continue

    tau_list = []
    for ds in df["Dataset Name"].unique():
        sub = df[df["Dataset Name"] == ds]
        x = sub["No. Covered Data Points by Different Classes"].to_numpy()
        y = sub[method].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # need at least two varying points
        if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
            continue
        if method == 'Nearest Neighbor Distance' and not all(y == y[0]):
            pass
        
        t, _ = kendalltau(x, y)
        tau_list.append(t)

    # average over all datasets (or NaN if none)
    taus[method] = np.nanmean(tau_list) if tau_list else 0



# -----------------------------------------------------------------------------------
# 5) Bar chart (monochrome) of those raw τ values
# -----------------------------------------------------------------------------------
abbrevs = [method_abbrevs[m] for m in method_abbrevs]
values  = [taus[m] for m in method_abbrevs]

plt.figure(figsize=(5, 5))
plt.bar(abbrevs, values, color='gray')

plt.ylim(-1, 1)
plt.xlabel("Method", fontsize=14)
plt.ylabel("Rank Correlation", fontsize=14)
# plt.title("Rank Correlation by Rendering Order and Method", fontsize=16, pad=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
