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


# 1) Load & concatenate
cwd = os.getcwd()
file1 = os.path.join(cwd, 'datasets', 'simulated_datasets', 'metadata_metrics.csv')
file2 = os.path.join(cwd, 'datasets', 'DimRed_data',       'metadata_metrics_DimRed.csv')
df = pd.concat([load_data(file1), load_data(file2)], ignore_index=True)


# 2) Filter out datasets with no covered points by different classes
max_covered = df.groupby("Dataset Name")["No. Covered Data Points by Different Classes"].max()
valid = max_covered[max_covered > 1].index
df = df[df["Dataset Name"].isin(valid)]


# 3) Method abbreviations & numeric conversion
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


# 4) Quartile-bin by No. Covered Data Points and label with exact ranges
#    (here we use q=5 to get five bins, just as your previous code did)
dataset_covered = df.groupby("Dataset Name")["No. Covered Data Points by Different Classes"].first()
covered_qcut    = pd.qcut(dataset_covered, q=5, precision=0, duplicates="drop")
ds_to_interval  = covered_qcut.to_dict()


# # 4a) print the maximum and minimum number of covered points
# max_covered_points = dataset_covered.max()
# min_covered_points = dataset_covered.min()
# min_index = dataset_covered.idxmin()
# print(f"Maximum No. Covered Data Points: {max_covered_points}")
# print(f"Minimum No. Covered Data Points: {min_covered_points} (Dataset: {min_index})")

# map back to each row
df["CoveredInterval"] = df["Dataset Name"].map(ds_to_interval)

# build labels like "200-500", "501-1000", etc.
interval_labels = {}
for i, interval in enumerate(covered_qcut.cat.categories):
    left = 0 if i == 0 else int(interval.left)
    right = int(interval.right)
    interval_labels[interval] = f"{left}-{right}"
df["CoveredGroup"] = df["CoveredInterval"].map(interval_labels)

# keep the ordered list of group labels
group_labels = [interval_labels[iv] for iv in covered_qcut.cat.categories]


# 5) Compute mean Kendall’s τ per method per covered-points-group
def compute_kendall_for_subset(subdf):
    out = {}
    for method in method_abbrevs:
        taus = []
        for ds in subdf["Dataset Name"].unique():
            part = subdf[subdf["Dataset Name"] == ds]
            x, y  = (
                part["No. Covered Data Points by Different Classes"].to_numpy(),
                part[method].to_numpy()
            )
            mask  = np.isfinite(x) & np.isfinite(y)
            x, y  = x[mask], y[mask]
            if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
                taus.append(np.nan)
            else:
                tau, _ = kendalltau(x, y)
                taus.append(tau)
        out[method] = np.nanmean(taus)
    return out

group_corr = {
    grp: compute_kendall_for_subset(df[df["CoveredGroup"] == grp])
    for grp in group_labels
}


# 6) Assemble into df and plot heatmap
methods = list(method_abbrevs.keys())
abbrevs = [method_abbrevs[m] for m in methods]

df_tau = (
    pd.DataFrame(
        {grp: [group_corr[grp][m] for m in methods]
         for grp in group_labels},
        index=abbrevs
    )
    .T
    .fillna(0.0)
)

sns.set(font_scale=1.2)
plt.figure(figsize=(len(abbrevs) * 1, 5))
plt.gcf().set_dpi(150)

heat = sns.heatmap(
    df_tau,
    annot=True, fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    cbar_kws={"label": "Rank Correlation"}
)

heat.set_xlabel("Method",                   fontsize=14)
heat.set_ylabel("Number of Anomalies Range", fontsize=14)
heat.set_title("Impact of Anomalies on Rank Correlation",
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

# tighten and show
plt.tight_layout()
plt.show()
