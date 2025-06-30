import os
import glob
import shutil
import pandas as pd

# ————————————————
# 1. Load the label file & parse base names
# ———————————————— 
base_dir = "datasets/simulated_datasets"
csv_folder_name = 'csv_files_1'
scatterplot_folder_name = 'figure_files_0623_evening'

cluster_df = pd.read_csv(
    os.path.join(base_dir, "LabelInformationSimulated_1.csv")
)

# strip “.csv” so that matches your dataset filenames (without path)
cluster_df["DatasetName_base"] = (
    cluster_df["DatasetName"]
    .str.replace(r"\.csv$", "", regex=True)
)

# detect if 'separated' appears in the ClusterCorrelation list
cluster_df["IsSeparated"] = (
    cluster_df["ClusterCorrelation"]
    .str.contains(r"'separated'")
)

# ————————————————
# 2. Prepare your output folders (under the same root)
# ————————————————
csv_dest_root  = os.path.join(base_dir, csv_folder_name + "_groups")
fig_dest_root  = os.path.join(base_dir, scatterplot_folder_name + "_groups")

for root in (csv_dest_root, fig_dest_root):
    for sub in ("separated_well", "not_separated"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

# ————————————————
# 3. Copy the CSV files into csv_file_1/
# ————————————————
for _, row in cluster_df.iterrows():
    base = row["DatasetName_base"]
    group = "separated_well" if row["IsSeparated"] else "not_separated"
    src = os.path.join(base_dir, csv_folder_name, base + ".csv")
    dst = os.path.join(csv_dest_root, group, base + ".csv")

    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"⚠️ CSV not found: {src}")

# ————————————————
# 4. Copy the scatter‐plot files into figure_files_0623_evening/
# ————————————————
# adjust extensions as needed
EXTS = ("png", "pdf", "svg", "jpg", "jpeg")

for _, row in cluster_df.iterrows():
    base = row["DatasetName_base"]
    group = "separated_well" if row["IsSeparated"] else "not_separated"
    found = False

    for ext in EXTS:
        pattern = os.path.join(base_dir, "figure_files_0623_evening", f"*{base}*.{ext}")
        for src_fig in glob.glob(pattern):
            dst_fig = os.path.join(fig_dest_root, group, os.path.basename(src_fig))
            shutil.copy2(src_fig, dst_fig)
            found = True

    if not found:
        print(f"⚠️ No figure found for: {base} (tried {EXTS})")

print("Done! All CSVs and figures have been split under datasets/simulated_datasets/.")
