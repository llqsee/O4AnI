import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datasets.generateData import load_data
import seaborn as sns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean') # or median, or most_frequent



def shap_cal(data):
    # Preparing the data
    features = ['GDP per capita', 'Social support', 'Healthy life', 'Freedom', 'Generosity', 'Corruption']
    X = data[features]
    y = data['Happiness Score']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Creating a Random Forest model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_imputed, y_train)

    # Computing SHAP values on the test set
    explainer = shap.Explainer(model, X_train_imputed)
    shap_values = explainer(X_test_imputed)
    
    # shap.summary_plot(shap_values, X_test_imputed)
    shap.summary_plot(shap_values, X_test_imputed, max_display=X.shape[1], feature_names=features)


    # Access the current axes
    ax = plt.gca()

    # Initialize a list to store display coordinates
    display_coordinates = []

    # Iterate through each point in shap_values
    for i in range(shap_values.shape[0]):
        for j in range(shap_values.shape[1]):
            # Convert data coordinates (j, shap_values[i, j]) to display coordinates
            display_coords = ax.transData.transform((j, shap_values.values[i, j]))
            display_coordinates.append(display_coords)

    # Convert to numpy array for easier handling
    display_coordinates = np.array(display_coordinates)
    print(display_coordinates)

    # Show the plot
    plt.show()


def pearson_corr(data):

    features = ['GDP per capita', 'Social support', 'Healthy life', 'Freedom', 'Generosity', 'Corruption']
    X = data[features]
    # Calculate the correlation matrix
    correlation_matrix = X.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Pearson Correlation Coefficient Between Features')
    plt.show()

if __name__ == "__main__":
    # Load the data:
    file_location = 'datasets/World_Happiness.csv'
    data = load_data(file_location)
    shap_cal(data)
    pearson_corr(data)