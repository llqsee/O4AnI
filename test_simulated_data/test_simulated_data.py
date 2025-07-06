import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.generateData import load_and_sample_data, load_data
from Our_metrics.Scatter_Metrics import Scatter_Metric
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from datasets.generateData import load_data  # Ensure this module and function are correctly defined
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.parameters import mnist_pred_categories

import numpy as np
import random
import json


importance_index_methods = ['mahalanobis_distance', 'isolation_forest', 'lof_forest', 'average_linkage_method', 
                            'complete_linkage_method', 'single_linkage_method', 'centroid_method',  # for cluster-based method
                            'leverage_score', 'cook_distance', 'orthogonal_distance_to_lowess_line', 'vertical_distance_to_lowess_line',
                            'horizontal_distance_to_lowess_line' # for distance-based method
                            ]


# dataset_name = 'clusternumber2_datanumber500_testnumbercategorylarge_repeatnumber0'   # It's the examples put in the paper
dataset_name = 'clusternumber3_datanumber200_testnumbercategorymedium_repeatnumber1'  # This is for the workflow chart in the paper (check the bugs based on this dataset))
# dataset_name = 'clusternumber3_datanumber200_testnumbercategorylarge_repeatnumber0'

# file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber2_datanumber200_testnumbercategorysmall_repeatnumber1.csv'
# file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber3_datanumber200_testnumbercategorylarge_repeatnumber0.csv'
# file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber5_datanumber100_testnumbercategorymedium_repeatnumber0.csv'
# file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber2_datanumber100_testnumbercategorymedium_repeatnumber2.csv'
# file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber5_datanumber500_testnumbercategorymedium_repeatnumber1.csv'
# file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber3_datanumber200_testnumbercategorysmall_repeatnumber1.csv'
# file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber3_datanumber200_testnumbercategorylarge_repeatnumber2.csv'
# file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber3_datanumber200_testnumbercategorylarge_repeatnumber3.csv'  # This is for the workflow chart in the paper
file_location = 'datasets/simulated_datasets/csv_files_1/' + dataset_name + '.csv'

# file_location = 'datasets/DimRed_data/Data_TSNE/wine_TSNE_2.csv'



data = load_data(file_location)  # Make sure load_data is properly defined

analysis = Scatter_Metric(data)

render_order_methods = ['category_based']
# render_order_methods = ['ascending']
marker_sizes = [110]

# render_order_methods = ['descending']
# marker_sizes = [50]

weight_diff_class = 10
weight_same_class = 0

for rm in render_order_methods:
    for mk in marker_sizes:
        analysis = Scatter_Metric(data, 
                                margins = {'left':0, 'right': 1, 'top':1, 'bottom': 0},
                                marker = 'square', 
                                marker_size = mk, 
                                dpi = 100, 
                                figsize= (10, 8),
                                xvariable = '1', 
                                yvariable = '2',
                                zvariable='class',
                                color_map='tab10'
                                )
        
        if rm == 'descending':
            analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=weight_diff_class, weight_same_class=weight_same_class, order_variable='importance_index', asending=False)
        elif rm == 'category_based':
            projected_labels = data['class'].unique().tolist()
            analysis._sort_data(attribute = 'class', order = projected_labels)
            analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=weight_diff_class, weight_same_class=weight_same_class)
        elif rm == 'ascending':
            analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=weight_diff_class, weight_same_class=weight_same_class, order_variable='importance_index', asending=True)
            # analysis.importance_metric(important_cal_method = 'lof_distance', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
            # analysis.importance_metric(important_cal_method = 'average_linkage_method', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
            # analysis.importance_metric(important_cal_method = 'complete_linkage_method', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
            # analysis.importance_metric(important_cal_method = 'single_linkage_method', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
            # analysis.importance_metric(important_cal_method = 'centroid_method', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
            # analysis.importance_metric(important_cal_method = 'isolation_forest', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)
        
        # analysis.plot_top_layer()

        analysis.cal_covered_data_points()
        
        analysis.calculate_pixels_covered_by_different_categories()

        # print('number_of_covered_data_points:', number_of_covered_data_points_different_class)

        max_value = np.nanmax(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))
        min_value = np.nanmin(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))

        if max_value != min_value:  # Avoid division by zero
            normalized_matrix = np.where(np.isnan(analysis.other_layer_matrix), np.nan, (analysis.other_layer_matrix - min_value) / (max_value - min_value))
        else:
            normalized_matrix = np.zeros_like(analysis.other_layer_matrix)

        # analysis.visualize_heat_map(normalized_matrix)
        # analysis.visualize_heat_map(analysis.other_layer_matrix)
        analysis.visualize_heat_map(analysis.overall_layer_matrix)

        # Convert numpy array to list for JSON serialization
        pixel_color_matrix_list = analysis.pixel_color_matrix.tolist()

        # Save to JSON file
        with open(f'test_simulated_data/pixel_color_matrix_{rm}markersize{mk}_{dataset_name}.json', 'w') as f:
            json.dump(pixel_color_matrix_list, f)
            

        analysis.data.to_csv(f'test_simulated_data/analysis_data_{rm}markersize{mk}_{dataset_name}.csv', index=False)

        analysis.save_figure(filename = f'test_simulated_data/figure_{rm}markersize{mk}_{dataset_name}.png')
        analysis.save_heatmap(filename = f'test_simulated_data/figure_{rm}markersize{mk}_{dataset_name}_heatmap.png')

        # analysis.bar_for_importance(analysis.data)
        # analysis.save_importance_bar(filename = 'test_simulated_data/importance_bar_distribution_simulated_data.png')

        analysis.plot_scatter_with_importance(filename = 'test_simulated_data/scatterplot_with_importance_' + dataset_name + '.png', grid=True, x_grid=50, y_grid=60)