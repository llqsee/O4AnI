import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data  # Ensure this module and function are correctly defined
# from utils.importance_calculation import *
# from utils.calApproach import *
# from utils.mapping_function import *
# from utils.density_transform import *



file_location = 'collect_scatterplots/collect_simulated_scatterplots/metadata_metrics.csv'
data = load_data(file_location)  # Make sure load_data is properly defined


# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data["hidden_point_number"]/data['Misclassified Data Number'], data["Visibility Index"], color='blue', alpha=0.7, label="Visibility Index")

# Labels and title
plt.xlabel("The Number of Hidden Points")
plt.ylabel("Visibility Index")
plt.title("Scatter Plot of Visibility Index vs. How many percentage of misclassified data points are hidden")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
