import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set the path to access parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("Current Working Directory:", os.getcwd())

# Define folder paths
data_folder = os.path.join(os.getcwd(), 'datasets/simulated_datasets/csv_files_1')
output_folder = os.path.join(os.getcwd(), 'datasets/simulated_datasets/example_scatterplots')
os.makedirs(output_folder, exist_ok=True)

def plot_scatter_from_csv(file_path):
    data = pd.read_csv(file_path)
    if data.shape[1] < 2:
        print(f"File {file_path} does not have enough columns to plot.")
        return

    # Make sure column names are strings
    required_columns = ['1', '2', 'class']
    if not all(col in data.columns for col in required_columns):
        print(f"File {file_path} does not have the required columns: {required_columns}")
        return

    x = data['1']
    y = data['2']
    labels = data['class'].astype(str)

    # Encode labels to integers 0,1,2,...
    unique_labels = list(pd.Categorical(labels).categories)
    label_to_num = {lab: i for i, lab in enumerate(unique_labels)}
    numeric_labels = labels.map(label_to_num)

    # Create a discrete colormap with exactly as many colors as classes
    cmap = ListedColormap(plt.get_cmap('tab10').colors[:len(unique_labels)])

    plt.figure()
    scatter = plt.scatter(x, y,
                          c=numeric_labels,
                          cmap=cmap,
                          alpha=0.6,
                          edgecolor='k',
                          linewidth=0.2)

    plt.title(f"Scatterplot for {os.path.basename(file_path)}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Build a legend that matches the discrete colors exactly
    handles = []
    for i, lab in enumerate(unique_labels):
        handles.append(
            plt.Line2D([0], [0],
                       marker='o',
                       color='w',
                       label=lab,
                       markerfacecolor=cmap(i),
                       markersize=8,
                       markeredgecolor='k',
                       markeredgewidth=0.2)
        )
    plt.legend(handles=handles, title="Class", loc='best', frameon=True)

    # Save out
    output_file = os.path.join(output_folder,
                               os.path.basename(file_path).replace('.csv', '.png'))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

# Process all CSV files one by one
csv_files = [os.path.join(data_folder, f)
             for f in os.listdir(data_folder) if f.lower().endswith('.csv')]

for csv_file in csv_files:
    plot_scatter_from_csv(csv_file)
