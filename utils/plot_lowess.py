import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

# Re-importing necessary libraries and redefining data for LOWESS plot after code execution reset
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datasets.generateData import load_and_sample_data, load_data


# We test a single method
dataset = 'negative_strong'
folder_location = 'datasets/simuliti/case_correlation/' 

file_location = folder_location + dataset + '.csv'
data = load_data(file_location)  # Make sure load_data is properly defined

x = data['feature_1']
y = data['feature_2']

# Generate LOWESS smoothed line
lowess = sm.nonparametric.lowess
z = lowess(y, x, frac=1./5.)  # frac controls the degree of smoothing

# Plot the original data
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the LOWESS line
plt.plot(z[:,0], z[:,1], color='red', label='LOWESS Line', linewidth=2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points with LOWESS Trend Line')
plt.legend()
plt.show()
