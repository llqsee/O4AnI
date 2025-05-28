import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import numpy as np

directory = 'datasets/mnist/'

# If saved as SavedModel
model = tf.keras.models.load_model(directory + 'my_model')
# Load the MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess test images
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Get predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Create a dictionary to hold activation data
activations = {}

# Iterate through each layer to get activations
i = 0
for layer in model.layers:
    intermediate_model = Model(inputs=model.input, outputs=layer.output)
    intermediate_output = intermediate_model.predict(test_images)
    if i >= len(model.layers) - 3:
        activations[layer.name] = intermediate_output
    i += 1

# Prepare data for CSV
data_for_csv = {
    'TrueLabel': test_labels,
    'PredictedLabel': predicted_classes
}

matches = test_labels == predicted_classes
data_for_csv['PredictedResults'] = matches

# Add probabilities for each class
for i in range(predictions.shape[1]):  # For each class
    data_for_csv[f'Prob_Class_{i}'] = predictions[:, i]

# Convert to DataFrame
results_df = pd.DataFrame(data_for_csv)

# Convert the 'TrueLabel' and 'PredictedLabel' columns to 'digit_' + number
results_df['TrueLabel'] = results_df['TrueLabel'].apply(lambda x: f'digit_{x}')
results_df['PredictedLabel'] = results_df['PredictedLabel'].apply(lambda x: f'digit_{x}')

# Function to apply dimensionality reduction
def apply_dr(features, method):
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    
    if method == 'tsne':
        dr_model = TSNE(n_components=2, perplexity=5, learning_rate=10, n_iter=3000, init='pca', random_state=500)
    elif method == 'pca':
        dr_model = PCA(n_components=2)
    elif method == 'umap':
        dr_model = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Unsupported dimensionality reduction method")
    
    return dr_model.fit_transform(features_standardized)

# Apply dimensionality reduction on the 'Prob_Class_' columns
prob_columns = [col for col in results_df.columns if col.startswith('Prob_Class_')]
features_prob = results_df[prob_columns].values

dr_methods = ['tsne', 'pca', 'umap']
for method in dr_methods:
    dr_results = apply_dr(features_prob, method)
    results_df[f'prob_{method}_x'] = dr_results[:, 0]
    results_df[f'prob_{method}_y'] = dr_results[:, 1]

# Add activation layers to the DataFrame
for layer_name, activation in activations.items():
    # Flatten activations if they are not already flat
    if len(activation.shape) > 2:
        activation = activation.reshape((activation.shape[0], -1))
    for i in range(activation.shape[1]):
        results_df[f'{layer_name}_act_{i}'] = activation[:, i]

# Apply dimensionality reduction on each activation layer
for layer_name, activation in activations.items():
    for method in dr_methods:
        dr_results = apply_dr(activation, method)
        results_df[f'{layer_name}_{method}_x'] = dr_results[:, 0]
        results_df[f'{layer_name}_{method}_y'] = dr_results[:, 1]

# Save the results to CSV
results_df.to_csv(directory + 'mnist_pred_morning07062024.csv', index=False)
