import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap

# Load the dataset
X, y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)

# Create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)

# Train the model
params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss",
}
model = xgboost.train(
    params,
    d_train,
    5000,
    evals=[(d_test, "test")],
    verbose_eval=100,
    early_stopping_rounds=20,
)

# Explain prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# If you want to add predictions to X_test for other purposes, use a copy:
X_test_with_predictions = X_test.copy()
predictions = model.predict(d_test)
X_test_with_predictions['predicted'] = predictions

# Convert probabilities to boolean labels based on the 0.5 threshold
boolean_labels = predictions >= 0.5
X_test_with_predictions['predicted_class'] = boolean_labels

# Define the mappings for the categorical columns
labels_relationship = {0: 'Not-in-family', 1: 'Unmarried', 2: 'Other-relative', 3: 'Own-child', 4: 'Husband', 5: 'Wife'}
labels_occupation = {0: 'Adm-clerical', 1: 'Exec-managerial', 2: 'Handlers-cleaners', 3: 'Prof-specialty', 4: 'Other-service',
                     5: 'Sales', 6: 'Craft-repair', 7: 'Transport-moving', 8: 'Farming-fishing', 9: 'Machine-op-inspct',
                     10: 'Tech-support', 11: 'Protective-serv', 12: 'Armed-Forces', 13: 'Priv-house-serv'}
labels_race = {0: 'White', 1: 'Asian-Pac-Islander', 2: 'Amer-Indian-Eskimo', 3: 'Other', 4: 'Black'}
labels_sex = {0: 'Male', 1: 'Female'}

# Add SHAP values as new columns
for col_name in X.columns:
    col_index = X.columns.tolist().index(col_name)
    X_test_with_predictions[col_name + '_shap'] = shap_values[:, col_index]

# Apply the mappings to the categorical columns
X_test_with_predictions['Relationship'] = X_test_with_predictions['Relationship'].map(labels_relationship)
X_test_with_predictions['Occupation'] = X_test_with_predictions['Occupation'].map(labels_occupation)
X_test_with_predictions['Race'] = X_test_with_predictions['Race'].map(labels_race)
X_test_with_predictions['Sex'] = X_test_with_predictions['Sex'].map(labels_sex)

# Process the t-SNE embedding
shap_embedded = TSNE(n_components=2, perplexity=50).fit_transform(shap_values[:1000, :])
X_abstract = X_test_with_predictions.iloc[:1000, :].copy()

# Add SHAP values as new columns to the abstract dataset
for col_name in X.columns:
    col_index = X.columns.tolist().index(col_name)
    X_abstract[col_name + '_shap'] = shap_values[:1000, col_index]

X_abstract['X coordinate'] = shap_embedded[:1000, 0]
X_abstract['Y coordinate'] = shap_embedded[:1000, 1]

# Apply the mappings to the categorical columns in the abstract dataset
X_abstract['Relationship'] = X_abstract['Relationship'].map(labels_relationship)
X_abstract['Occupation'] = X_abstract['Occupation'].map(labels_occupation)
X_abstract['Race'] = X_abstract['Race'].map(labels_race)
X_abstract['Sex'] = X_abstract['Sex'].map(labels_sex)

current_directory = os.getcwd()
# Save the datasets to CSV files
X_test_with_predictions.to_csv(os.path.join(current_directory, 'datasets/adult_income/adult_xgboost_v3.csv'), index=False)
X_abstract.to_csv(os.path.join(current_directory, 'datasets/adult_income/adult_xgboost_t-sne_v3.csv'), index=False)
