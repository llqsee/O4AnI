import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# so that `datasets.generateData` is importable:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datasets.generateData import load_data


# -------------------------------------------------------
# 1) Load & concatenate both CSVs
# -------------------------------------------------------
cwd = os.getcwd()
print("Current Working Directory:", cwd)

file1 = os.path.join(cwd, 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file2 = os.path.join(cwd, 'datasets', 'DimRed_data',       'metadata_metrics_DimRed.csv')

df1 = load_data(file1)
df2 = load_data(file2)
df  = pd.concat([df1, df2], ignore_index=True)


# -------------------------------------------------------
# 2) Remove datasets with no covered data points by different classes
# -------------------------------------------------------
max_covered = (
    df
    .groupby("Dataset Name")["No. Covered Data Points by Different Classes"]
    .max()
)
valid_datasets = max_covered[max_covered > 1].index
df = df[df["Dataset Name"].isin(valid_datasets)]


# -------------------------------------------------------
# 3) method abbreviations & ensure numeric
# -------------------------------------------------------
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
df[to_num] = df[to_num].apply(pd.to_numeric, errors='coerce').round(6)


# -------------------------------------------------------
# 4) Bin each Dataset into 4 equal‐count groups by Number of Entire Data
#    and label bins with their exact min–max ranges
# -------------------------------------------------------
# 4a) get each dataset’s single "Number of Entire Data" value
dataset_sizes = df.groupby("Dataset Name")["Number of Entire Data"].first()

# 4b) qcut into 4 bins
sizes_qcut = pd.qcut(dataset_sizes, q=4, precision=0, duplicates="drop")

# 4c) map Dataset Name → Interval object, then to a "min–max" label
ds_to_interval = sizes_qcut.to_dict()
df["SizeInterval"] = df["Dataset Name"].map(ds_to_interval)

interval_labels = {
    interval: f"{int(interval.left)}–{int(interval.right)}"
    for interval in sizes_qcut.cat.categories
}
df["SizeGroup"] = df["SizeInterval"].map(interval_labels)

# the exact labels in ascending order
group_labels = [interval_labels[iv] for iv in sizes_qcut.cat.categories]


# -------------------------------------------------------
# 5) Compute mean Kendall’s τ for each method within each size‐group
# -------------------------------------------------------
def compute_kendall_for_subset(subdf):
    out = {}
    for method in method_abbrevs:
        taus = []
        for ds in subdf["Dataset Name"].unique():
            part = subdf[subdf["Dataset Name"] == ds]
            x = part["No. Covered Data Points by Different Classes"].to_numpy()
            y = part[method].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
                taus.append(np.nan)
            else:
                tau, _ = kendalltau(x, y)
                taus.append(tau)
        out[method] = np.nanmean(taus)
    return out

group_corr = {
    grp: compute_kendall_for_subset(df[df["SizeGroup"] == grp])
    for grp in group_labels
}


# -------------------------------------------------------
# 6) Assemble into a DataFrame and plot heatmap
# -------------------------------------------------------
methods = list(method_abbrevs.keys())
abbrevs = [method_abbrevs[m] for m in methods]

df_tau = pd.DataFrame(
    {grp: [group_corr[grp][m] for m in methods]
     for grp in group_labels},
    index=abbrevs
).T.fillna(0.0)


sns.set(font_scale=1.2)
plt.figure(figsize=(len(abbrevs) * 1, 5))
plt.gcf().set_dpi(150)

heat = sns.heatmap(
    df_tau,
    annot=True, fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,  # Set colorbar (legend) range from -1 to 1
    center=0,
    cbar_kws={"label": "Rank Correlation"},
)

heat.set_xlabel("Method",             fontsize=14)
heat.set_ylabel("Data-Size Range",    fontsize=14)
heat.set_title("Impact of Data Size on Rank Correlation",
               fontsize=16, pad=12)

heat.set_xticklabels(
    heat.get_xticklabels(),
    rotation=45, ha="right", fontsize=12
)
heat.set_yticklabels(
    heat.get_yticklabels(),
    rotation=0,            fontsize=12
)

# Reduce the number of ticks in the colorbar (legend)
cbar = heat.collections[0].colorbar
cbar.set_ticks(np.linspace(-1, 1, 5))  # e.g., 5 ticks: -1, -0.5, 0, 0.5, 1

plt.tight_layout()
plt.show()
