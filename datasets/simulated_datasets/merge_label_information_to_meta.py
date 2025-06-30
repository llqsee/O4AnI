import pandas as pd

# 1. Load both CSVs
cluster_df = pd.read_csv("datasets/simulated_datasets/LabelInformationSimulated_1.csv")
scatter_df = pd.read_csv("datasets/simulated_datasets/metadata_metrics_simulated_0623_evening.csv")

# 2. Create a “base” DatasetName without the .csv
cluster_df["DatasetName_base"] = cluster_df["DatasetName"] \
    .str.replace(r"\.csv$", "", regex=True)

# 3. Build lookup dicts keyed on the base name
shape_map = cluster_df.set_index("DatasetName_base")["Shape"].to_dict()
cc_map    = cluster_df.set_index("DatasetName_base")["ClusterCorrelation"].to_dict()

# 4. Map into the scatter‐metrics frame (which already has no .csv in its names)
scatter_df["Shape"]              = scatter_df["DatasetName"].map(shape_map)
scatter_df["ClusterCorrelation"] = scatter_df["DatasetName"].map(cc_map)

# 5. (Optional) report any unmatched names
missing = scatter_df[scatter_df["Shape"].isna()]
if not missing.empty:
    print("Warning: these DatasetNames weren’t found in your label file:")
    print(missing["DatasetName"].unique())

# 6. Save the merged DataFrame
scatter_df.to_csv(
    "datasets/simulated_datasets/merged_scatter_with_shape_and_correlation.csv",
    index=False
)
print("Done – saved to merged_scatter_with_shape_and_correlation.csv")
