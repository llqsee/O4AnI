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
    "M_distance": "MD",
    "LOF Distance": "LOFD",
    "Isolation Forest Distance": "IFD",
    # "Centroid Distance": "CED",
    "Average Linkage Method Distance": "ALMD",
    # ...
    "Visibility Index": "VI",
    "Grid Density Overlap Degree": "GDOD",
    "Pairwise Bounding Box Overlap Degree": "PBBOD",
    "MPix": "MP",
    "Nearest Neighbor Distance": "NND",
    "Kernel Density Overlap Degree": "KDOD"
}
to_num = [c for c in method_abbrevs if c in df.columns]
df[to_num] = df[to_num].apply(pd.to_numeric, errors='coerce').round(6)


# 4) Split into two groups: those whose Shape includes 'three_order' vs. those that don't
#    (Shape is assumed to be a list-like column)
df["HasThreeOrder"] = df["Shape"].apply(lambda x: "three_order" in x)

# We'll give them human-readable labels:
group_keys   = [True, False]
group_labels = ["Includes curve-shaped clusters", "Excludes curve-shaped clusters"]


# 5) Compute mean Kendall’s τ per method in each of these two groups
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
    grp_label: compute_kendall_for_subset(df[df["HasThreeOrder"] == grp_key])
    for grp_key, grp_label in zip(group_keys, group_labels)
}


# 6) Assemble into df and plot heatmap
methods = list(method_abbrevs.keys())
abbrevs = [method_abbrevs[m] for m in methods]

df_tau = (
    pd.DataFrame(
        {lbl: [group_corr[lbl][m] for m in methods]
         for lbl in group_labels},
        index=abbrevs
    )
    .T
    .fillna(0.0)
)

sns.set(font_scale=1.2)
plt.figure(figsize=(len(abbrevs) * 1, 4))
plt.gcf().set_dpi(150)

heat = sns.heatmap(
    df_tau,
    annot=True, fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    cbar_kws={"label": "Rank Correlation"},
)

heat.set_xlabel("Method",                 fontsize=14)
heat.set_ylabel("Shape Group",            fontsize=14)
heat.set_title("Impact of Shape on Rank Correlation",
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
