import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# file_location = 'datasets/DimRed_data/Data_GlimmerMDS/bbdm13_GlimmerMDS_2.csv'

def load_data(file_location):
    data = pd.read_csv(file_location)
    return data


def delete_non_matching_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and not filename.endswith('_2.csv'):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

directory = 'datasets/DimRed_data/Data_TSNE'
delete_non_matching_files(directory)
