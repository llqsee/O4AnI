import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap

# Load the dataset
input_csv = 'datasets/mnist/mnist_pred_updated_str.csv'
output_csv = 'datasets/mnist/mnist_pred_updated_str_with_dr_results.csv'
data = pd.read_csv(input_csv)

# Select the columns for dimensionality reduction
prob_columns = [f'prob_{i}' for i in range(10)]
features_prob = data[prob_columns].values

# Ensure labels are available for LDA
if 'pred' not in data.columns:
    raise ValueError("The dataset must contain a 'pred' column for LDA.")
labels = data['pred']

# Function to apply dimensionality reduction
def apply_dr(features, method, labels=None):
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    
    if method == 'tsne':
        dr_model = TSNE(
            n_components=2,
            perplexity=10,
            learning_rate=200,
            n_iter=2000,
            init='pca',
            metric='euclidean',
            random_state=42
        )
        return dr_model.fit_transform(features_standardized)

    elif method == 'isomap':
        dr_model = Isomap(
            n_components=2,
            n_neighbors=10,
            metric='euclidean'
        )
        return dr_model.fit_transform(features_standardized)

    elif method == 'umap':
        dr_model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        return dr_model.fit_transform(features_standardized)

    elif method == 'lda':
        if labels is None:
            raise ValueError("Labels are required for LDA.")
        dr_model = LDA(n_components=2)
        return dr_model.fit_transform(features_standardized, labels)

    elif method == 'mds':
        dr_model = MDS(
            n_components=2,
            n_init=4,
            max_iter=300,
            random_state=42
        )
        return dr_model.fit_transform(features_standardized)

    else:
        raise ValueError("Unsupported dimensionality reduction method")

# Apply dimensionality reduction methods
dr_methods = ['tsne', 'isomap', 'umap', 'lda', 'mds']
for method in dr_methods:
    print(f"Applying {method.upper()}...")
    if method == 'lda':
        dr_results = apply_dr(features_prob, method, labels=labels)
    else:
        dr_results = apply_dr(features_prob, method)
    
    data[f'{method}_x'] = dr_results[:, 0]
    data[f'{method}_y'] = dr_results[:, 1]

# Save the updated dataset
data.to_csv(output_csv, index=False)
print("Dimensionality reduction results saved.")
