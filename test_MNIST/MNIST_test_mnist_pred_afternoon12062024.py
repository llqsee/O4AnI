import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.generateData import load_and_sample_data, load_data
from Our_metrics.Scatter_Metrics import Scatter_Metric
from Compared_metrics.CDM import CDM_Metric
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from datasets.generateData import load_data  # Ensure this module and function are correctly defined
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.parameters import mnist_pred_categories

import numpy as np


importance_index_methods = ['mahalanobis_distance', 'average_linkage_method', 
                            'complete_linkage_method', 'single_linkage_method', 'centroid_method', 'isolation_forest',
                            'leverage_score', 'cook_distance', 'euc_distance',
                            'orthogonal_distance_to_lowess_line', 'vertical_distance_to_lowess_line',
                            'horizontal_distance_to_lowess_line']


file_location = 'datasets/mnist/mnist_pred_afternoon12062024.csv'
data = load_data(file_location)  # Make sure load_data is properly defined

analysis = Scatter_Metric(data)

analysis = Scatter_Metric(data, 
                          margins = {'left':0.2, 'right': 0.7, 'top':0.8, 'bottom': 0.2},
                        marker = 'square', 
                        marker_size = 50, 
                        dpi = 100, 
                        figsize= (10, 6),
                        xvariable = 'dense_umap_x', 
                        yvariable = 'dense_umap_y',
                        zvariable='TrueLabel',
                        color_map='tab10'
                        )

analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=10, weight_same_class=0, order_variable='importance_index', asending=False)


analysis.result

max_value = np.nanmax(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))
min_value = np.nanmin(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))

if max_value != min_value:  # Avoid division by zero
    normalized_matrix = np.where(np.isnan(analysis.other_layer_matrix), np.nan, (analysis.other_layer_matrix - min_value) / (max_value - min_value))
else:
    normalized_matrix = np.zeros_like(analysis.other_layer_matrix)

# analysis.visualize_heat_map(normalized_matrix)
analysis.visualize_heat_map(analysis.overall_layer_matrix)

analysis.save_figure(filename = 'test_MNIST/output_MNIST_scatterplot.png')
analysis.save_heatmap(filename = 'test_MNIST/output_MNIST_heatmap.png')

analysis.bar_for_importance(analysis.data)
analysis.save_importance_bar(filename = 'test_MNIST/output_MNIST_importance_bar_distribution.png')

