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
import seaborn as sns

import numpy as np
import random


importance_index_methods = ['mahalanobis_distance', 'isolation_forest', 'lof_forest', 'average_linkage_method', 
                            'complete_linkage_method', 'single_linkage_method', 'centroid_method',  # for cluster-based method
                            'leverage_score', 'cook_distance', 'orthogonal_distance_to_lowess_line', 'vertical_distance_to_lowess_line', 'influence_score',
                            'horizontal_distance_to_lowess_line' # for distance-based method
                            ]


file_location = 'datasets/mnist/mnist_pred_updated_str.csv'
# file_location = 'datasets/mnist/mnist_pred_with_dr_results.csv'

data = load_data(file_location)  # Make sure load_data is properly defined

analysis = Scatter_Metric(data)

analysis = Scatter_Metric(data, 
                          margins = {'left':0.2, 'right': 0.7, 'top':0.8, 'bottom': 0.2},
                        marker = 'plus', 
                        marker_size = 10, 
                        dpi = 100, 
                        figsize= (12, 8),
                        xvariable = 'X coordinate',
                        yvariable = 'Y coordinate', 
                        zvariable='pred',
                        color_map='tab10'
                        )



# projected_labels = random.sample(data['pred'].unique().tolist(), len(data['pred'].unique()))

projected_labels = ['digit_2', 'digit_8', 'digit_5', 'digit_7', 'digit_3', 'digit_4', 'digit_1', 'digit_0', 'digit_6', 'digit_9']
# projected_labels = ['digit_7', 'digit_3', 'digit_4', 'digit_1', 'digit_0', 'digit_6', 'digit_9', 'digit_2', 'digit_8', 'digit_5']
analysis._sort_data(attribute = 'pred', order = projected_labels)

# Randomly shuffle the data
# shuffled_data = analysis.data.sample(frac=1, random_state=random.randint(0, 1000)).reset_index(drop=True)
# analysis.data = shuffled_data

analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=10, weight_same_class=1)





# analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=10, weight_same_class=1, order_variable='importance_index', asending=True)
# analysis.importance_metric(important_cal_method = 'lof_distance', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
# analysis.importance_metric(important_cal_method = 'average_linkage_method', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
# analysis.importance_metric(important_cal_method = 'complete_linkage_method', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
# analysis.importance_metric(important_cal_method = 'single_linkage_method', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
# analysis.importance_metric(important_cal_method = 'centroid_method', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
# analysis.importance_metric(important_cal_method = 'isolation_forest', weight_diff_class=20, weight_same_class=0)
# analysis.importance_metric(important_cal_method = 'cook_distance', weight_diff_class=20, weight_same_class=0, order_variable='importance_index', asending=True)
# analysis.importance_metric(important_cal_method = 'leverage_score', weight_diff_class=20, weight_same_class=0, order_variable='importance_index', asending=False)
# analysis.importance_metric(important_cal_method = 'influence_distance', weight_diff_class=20, weight_same_class=0, order_variable='importance_index', asending=True)

analysis.result


max_value = np.nanmax(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))
min_value = np.nanmin(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))

if max_value != min_value:  # Avoid division by zero
    normalized_matrix = np.where(np.isnan(analysis.other_layer_matrix), np.nan, (analysis.other_layer_matrix - min_value) / (max_value - min_value))
else:
    normalized_matrix = np.zeros_like(analysis.other_layer_matrix)

# analysis.visualize_heat_map(normalized_matrix)
# analysis.visualize_heat_map(analysis.other_layer_matrix)
analysis.visualize_heat_map(analysis.overall_layer_matrix)

analysis.save_figure(filename = 'test_MNIST/scatterplot_mnist_pred_updated_str.png')
analysis.save_heatmap(filename = 'test_MNIST/heat_map_mnist_pred_updated_str.png')

analysis.bar_for_importance(analysis.data)
analysis.save_importance_bar(filename = 'test_MNIST/importance_bar_distribution_mnist_pred_updated_str.png')


analysis.plot_density_heatmap(filename = 'test_MNIST/density_heatmap_mnist_pred_updated_str.png')


analysis.distance_consistency()