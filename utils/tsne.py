import pandas as pd
from sklearn.manifold import TSNE
from datasets.generateData import load_data

def process_data(file_location):
    # Load the data
    data = load_data(file_location)

    # Selecting the relevant attributes for dimension reduction
    selected_columns = ['prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4', 
                        'prob_5', 'prob_6', 'prob_7', 'prob_8', 'prob_9']
    data_subset = data[selected_columns]

    # Perform TSNE for dimension reduction
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(data_subset)

    # Add the results back to the original dataframe
    data['x'] = tsne_results[:, 0]
    data['y'] = tsne_results[:, 1]

    # Save the updated dataframe
    updated_file_path = 'datasets/mnist_pred_updated.csv'
    data.to_csv(updated_file_path, index=False)

if __name__ == "__main__":
    file_location = 'datasets/mnist_pred.csv'
    process_data(file_location)
