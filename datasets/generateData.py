import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

# https://www.kaggle.com/code/alexisbcook/scatter-plots/data



# Load the datasets and return: 1) x,y,z that are 
def load_and_sample_data(percentage_to_extract, file_location, d1, d2, d3):
    
    current_directory = os.getcwd()
    # Load the CSV data into a Pandas DataFrame
    df = pd.read_csv(os.path.join(current_directory, file_location))

    # Use the 'sample' method to randomly extract rows
    df_sampled = df.sample(frac=percentage_to_extract, random_state=42)

    # Parameters
    dimension_1 = d1
    dimension_2 = d2
    dimension_3 = d3

    x = df_sampled[dimension_1]
    y = df_sampled[dimension_2]
    z = df_sampled[dimension_3]

    # Stack x, y, and z to form a 2D array
    data = np.column_stack((x, y, z))

    return data, x, y, z, d1,d2,d3


# Just load the datasets
def load_data(file_location):
    current_directory = os.getcwd()
    # Load the CSV data into a Pandas DataFrame
    df = pd.read_csv(os.path.join(current_directory, file_location))
    return df
    

if __name__ == "__main__":
    # file_location = 'datasets/Sample_Superstore_Orders.csv'
    # percentage_to_extract = 0.01
    # data, x, y, z = load_and_sample_data(percentage_to_extract, file_location, 'Sales', 'Profit', 'Cluster')
    
    file_location = 'datasets/World_Happiness.csv'
    load_data(file_location)