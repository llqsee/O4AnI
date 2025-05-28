
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())
import pandas as pd


directory = 'datasets/mnist/'
# Load the CSV file
file_path = 'mnist_pred_morning07062024.csv'
df = pd.read_csv(directory  + file_path)

# Display the first few rows of the dataframe
df.head()

# Modify the is_correct column values
df['PredictedResults'] = df['PredictedResults'].replace({True: 'is_true', False: 'is_false'})

# Save the updated dataframe to a new CSV file
file_path_1 = directory + 'mnist_pred_afternoon12062024.csv'
df.to_csv(file_path_1, index=False)