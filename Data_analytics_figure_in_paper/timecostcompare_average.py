import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets.generateData import load_data  # Ensure this module and function are correctly defined

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

## Apply abbreviations to labels
abbreviated_labels = [method_abbreviations.get(label, label) for label in cleaned_labels]

time_cost_attributes = [f"{x} Calculation Time" for x in method_abbreviations.keys()]
avg_costs = [df[attr].mean() for attr in time_cost_attributes]


# Plot the horizontal bar chart
fig, ax = plt.subplots(figsize=(7, 8))
y = np.arange(len(abbreviated_labels))

bars = ax.barh(y, avg_costs, color='black')
ax.set_yticks(y)
ax.set_yticklabels(abbreviated_labels, fontsize=12)
ax.set_xlabel('Average Calculation Time (s)', fontsize=14)
ax.set_ylabel('Method', fontsize=14)
ax.set_title('Average Time Cost per Method', fontsize=16)
ax.set_xlim([0, max(avg_costs) * 1.1])  # Set x-limit to avoid long bars touching the edge
ax.invert_yaxis()

# Annotate each bar with its value
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + max(avg_costs) * 0.01,  # slight padding from the end of the bar
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}', va='center', fontsize=11)

fig.tight_layout()
# plt.grid(True)
plt.show()