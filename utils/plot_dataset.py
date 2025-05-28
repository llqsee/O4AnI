import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

from datasets.generateData import load_and_sample_data, load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture



# Input the dataset from the datasets folder
# file_location = 'datasets/simuliti/case_correlation/negative_strong.csv'
# file_location = 'datasets/simuliti/functionbasedDataset/overplottingSimulatedUserStudyData/clusterBig.csv'
# file_location = 'datasets/simuliti/functionbasedDataset/overplottingSimulatedUserStudyData/CubicFunction.csv'
# file_location = 'datasets/simuliti/functionbasedDataset/overplottingSimulatedUserStudyData/linearFunction.csv'
# file_location = 'datasets/simuliti/functionbasedDataset/overplottingSimulatedUserStudyData/lipCluster_45.csv'
# file_location = 'datasets/simuliti/functionbasedDataset/overplottingSimulatedUserStudyData/lipCluster_135.csv'
# file_location = 'datasets/simuliti/functionbasedDataset/overplottingSimulatedUserStudyData/1312data/csvData/none_entireNumber50_testNumber20_class3_class_touch_typenon_touched.csv'
# file_location = 'datasets/simuliti/functionbasedDataset/overplottingSimulatedUserStudyData/1312data/csvData/none_entireNumber200_testNumber3_class5_class_touch_typenon_touched.csv'

file_location = 'datasets/20newsgroups/20newsgroups.csv'


data = load_data(file_location)  # Make sure load_data is properly defined


# Plotting the generated samples
plt.figure(figsize=(8, 6))

# plt.scatter(data['feature_1'], data['feature_2'], c='black', marker='o', edgecolor='k', s=50, alpha=1)

# plt.scatter(data['X coordinate'], data['Y coordinate'], c=data['group'], marker='o', s=50, alpha=1)
unique_groups = data['group'].unique()
# colors = plt.cm.get_cmap('viridis', len(unique_groups))
colors = plt.cm.get_cmap('plasma', len(unique_groups))

for i, group in enumerate(unique_groups):
    group_data = data[data['group'] == group]
    plt.scatter(group_data['X coordinate'], group_data['Y coordinate'], c=[colors(i)], label=group, marker='o', s=50, alpha=1, edgecolor='black')

plt.legend(title='Group')

# plt.title('Scatterplot of Weak Negative Correlation', fontsize = 20)
# plt.title('Scatterplot of Weak Positive Correlation', fontsize = 20)
# plt.title('Scatterplot of Strong Positive Correlation', fontsize = 20)
# plt.title('Scatterplot of Strong Negative Correlation', fontsize = 20)
plt.xlabel('X Coordinate', fontsize = 18)
plt.ylabel('Y Coordinate',fontsize = 18)
plt.show()
