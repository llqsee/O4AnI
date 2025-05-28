import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the path to access parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("Current Working Directory:", os.getcwd())

# Define folder paths
data_folder = os.path.join(os.getcwd(), 'datasets/DimRed_data/Data_TSNE')
output_folder = os.path.join(os.getcwd(), 'datasets/DimRed_data/example_scatterplots_Data_TSNE')
os.makedirs(output_folder, exist_ok=True)

# Function to plot scatterplot for a single CSV file
def plot_scatter_from_csv(file_path):
    data = pd.read_csv(file_path)
    if data.shape[1] < 2:
        print(f"File {file_path} does not have enough columns to plot.")
        return
    required_columns = ['1', '2', 'class']
    if not all(col in data.columns for col in required_columns):
        print(f"File {file_path} does not have the required columns: {required_columns}")
        return

    x, y = data['1'], data['2']
    labels = data['class']
    
    # Map string labels to numeric IDs for coloring
    unique_labels = labels.unique()
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    colors = labels.map(label_to_num)

    plt.figure()
    scatter = plt.scatter(x, y, c=colors, cmap='tab10', alpha=0.5)
    plt.title(f"Scatterplot for {os.path.basename(file_path)}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=plt.cm.tab10(label_to_num[label]), markersize=8)
               for label in unique_labels]
    plt.legend(handles=handles, title="Class")

    output_file = os.path.join(output_folder, os.path.basename(file_path).replace('.csv', '.png'))
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


# Process all CSV files one by one
csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    plot_scatter_from_csv(csv_file)


# Create a CSV file to store dataset names and their "well-separated" status
summary_file = os.path.join(output_folder, 'seperation_label_Data_TSNE.csv')

with open(summary_file, 'w') as f:
    f.write('dataset name,well-separated\n')
    for csv_file in csv_files:
        dataset_name = os.path.basename(csv_file)
        well_separated = 'no'  # Placeholder value, adjust logic as needed
        f.write(f'{dataset_name},{well_separated}\n')

print(f"Summary file created: {summary_file}")