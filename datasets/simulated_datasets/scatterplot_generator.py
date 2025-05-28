import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("Current Working Directory:", os.getcwd())

import numpy as np
import pandas as pd
import concurrent.futures

from Our_metrics.Scatter_Metrics import Scatter_Metric


def visualize_and_save(data, marker_size, figure_save_folder, metadata_save_folder , scatterplot_filename="", dataset_filename = '', metadata_csv="metadata_metrics.csv", order_method = None):  
    """Visualize and save the generated scatterplot along with metadata."""
    
    
    
    # ==================================================================================================
    # Compute the metrics for the scatterplot and save the scatterplot
    scatter_metric = Scatter_Metric(data, 
                            margins = {'left':0.15, 'right': 0.75, 'top':0.9, 'bottom': 0.1},
                            marker = 'square', 
                            marker_size = marker_size, 
                            dpi = 100, 
                            figsize= [10, 8],
                            xvariable = '1', 
                            yvariable = '2',
                            zvariable='class',
                            color_map='tab10'
                            )
    
    
    
    
    
    # ==================================================================================================
    # M_distance method calcuation
    if order_method == 'category_based':
        projected_labels = data['class'].unique().tolist()
        scatter_metric._sort_data(attribute = 'class', order = projected_labels)
        scatter_metric.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=20, weight_same_class=1)
    elif order_method == 'ascending':
        scatter_metric.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=20, weight_same_class=1, order_variable = 'importance_index', asending = True)
        
    else:
        scatter_metric.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=20, weight_same_class=1, order_variable = 'importance_index', asending = False)
        
    m_distance = scatter_metric.result
    m_distance_time = scatter_metric.calculation_time
    # ==================================================================================================
    
    # ==================================================================================================
    # LOF method calculation
    scatter_metric.importance_metric(important_cal_method = 'lof_distance', weight_diff_class=20, weight_same_class=1)
    lof_distance = scatter_metric.result
    lof_distance_time = scatter_metric.calculation_time
    # ==================================================================================================
    
    # ==================================================================================================
    # isolation_forest method calculation
    scatter_metric.importance_metric(important_cal_method = 'isolation_forest', weight_diff_class=20, weight_same_class=1)
    iso_distance = scatter_metric.result
    iso_distance_time = scatter_metric.calculation_time
    # ==================================================================================================
    
    # ==================================================================================================
    # Centroid  method calculation
    scatter_metric.importance_metric(important_cal_method = 'centroid_method', weight_diff_class=20, weight_same_class=1)
    centroid_distance = scatter_metric.result
    centoid_time = scatter_metric.calculation_time
    # ==================================================================================================
    
    # ==================================================================================================
    # average_linkage_method  method calculation
    scatter_metric.importance_metric(important_cal_method = 'average_linkage_method', weight_diff_class=20, weight_same_class=1)
    average_linkage_method_distance = scatter_metric.result
    average_linkage_method_time = scatter_metric.calculation_time
    # ==================================================================================================
    
    # ==================================================================================================
    # complete_linkage_method  method calculation
    scatter_metric.importance_metric(important_cal_method = 'complete_linkage_method', weight_diff_class=20, weight_same_class=1)
    complete_linkage_method_distance = scatter_metric.result
    complete_linkage_method_time = scatter_metric.calculation_time
    # ==================================================================================================
    
    
    # ==================================================================================================
    # single_linkage_method  method calculation
    scatter_metric.importance_metric(important_cal_method = 'single_linkage_method', weight_diff_class=20, weight_same_class=1)
    single_linkage_method_distance = scatter_metric.result
    single_linkage_method_time = scatter_metric.calculation_time
    # ==================================================================================================

    # ==================================================================================================
    # leverage_score  method calculation
    scatter_metric.importance_metric(important_cal_method = 'leverage_score', weight_diff_class=20, weight_same_class=1)
    leverage_score_distance = scatter_metric.result
    leverage_score_time = scatter_metric.calculation_time
    # ==================================================================================================
    
    # ==================================================================================================
    # cook_distance  method calculation
    scatter_metric.importance_metric(important_cal_method = 'cook_distance', weight_diff_class=20, weight_same_class=1)
    cook_distance = scatter_metric.result
    cook_distance_score_time = scatter_metric.calculation_time
    # ==================================================================================================


    # ==================================================================================================
    # orthogonal_distance_to_lowess_line  method calculation
    scatter_metric.importance_metric(important_cal_method = 'orthogonal_distance_to_lowess_line', weight_diff_class=20, weight_same_class=1)
    orthogonal_distance_to_lowess_line_distance = scatter_metric.result
    orthogonal_distance_to_lowess_line_time = scatter_metric.calculation_time
    # ==================================================================================================
    

    # ==================================================================================================
    # vertical_distance_to_lowess_line  method calculation
    scatter_metric.importance_metric(important_cal_method = 'vertical_distance_to_lowess_line', weight_diff_class=20, weight_same_class=1)
    vertical_distance_to_lowess_line_distance = scatter_metric.result
    vertical_distance_to_lowess_line_time = scatter_metric.calculation_time
    # ==================================================================================================
    

    # ==================================================================================================
    # horizontal_distance_to_lowess_line  method calculation
    scatter_metric.importance_metric(important_cal_method = 'horizontal_distance_to_lowess_line', weight_diff_class=20, weight_same_class=1)
    horizontal_distance_to_lowess_line_distance = scatter_metric.result
    horizontal_distance_to_lowess_line_time = scatter_metric.calculation_time
    # ==================================================================================================

    


    
    

    scatter_metric.visilibity_index()
    visibility_index = scatter_metric.result
    visibility_index_time = scatter_metric.calculation_time
    
    scatter_metric.mpix()
    mpix_result = scatter_metric.result
    mpix_time = scatter_metric.calculation_time
    
    scatter_metric.grid_based_density_overlap_degree(grid_pixel_ratio = 1)
    grid_density_overlap_degree = scatter_metric.result
    grid_density_overlap_degree_time = scatter_metric.calculation_time
    
    
    scatter_metric.kernel_density_estimation(bandwidth=0.1, gridsize=100)
    kernel_density_overlap_degree = scatter_metric.result
    kernel_density_overlap_degree_time = scatter_metric.calculation_time
    
    scatter_metric.nearest_neighbor_distance()
    nearest_neighbor_distance = scatter_metric.result
    nearest_neighbor_distance_time = scatter_metric.calculation_time
    
    
    # ==================================================================================================
    # Pairwise Bounding Box Overlap Degree
    scatter_metric.pairwise_bounding_box_based_overlap_degree()
    pairwise_bounding_box_overlap_degree = scatter_metric.result
    pairwise_bounding_box_overlap_degree_time = scatter_metric.calculation_time
    # ==================================================================================================
    
    
    number_covered_values, number_covered_values_different_classes, covered_values_different_classes = scatter_metric.cal_covered_data_points()


    # ==================================================================================================
            
    # ==================================================================================================
    # save figure
    plot_path = os.path.join(figure_save_folder, f"{scatterplot_filename}")
    scatter_metric.save_figure(filename=f'{plot_path}')
    # ==================================================================================================
    
    
    # ==================================================================================================
    # We save the metadata
    if 'shape' in data.keys():
        metadata = {
            "Scatterplot Name": scatterplot_filename,
            "Dataset Name": dataset_filename,
            "Number of Entire Data": len(data),
            'Number of Classes': len(data['class'].unique()),
            'Shape': data['shape'].unique(),
            "Marker Size": marker_size,
            "M_distance": m_distance,
            "M_distance Calculation Time": m_distance_time,
            "LOF Distance": lof_distance,
            "LOF Distance Calculation Time": lof_distance_time,
            "Isolation Forest Distance": iso_distance,
            "Isolation Forest Distance Calculation Time": iso_distance_time,
            "Centroid Distance": centroid_distance,
            "Centroid Distance Calculation Time": centoid_time,
            "Average Linkage Method Distance": average_linkage_method_distance,
            "Average Linkage Method Distance Calculation Time": average_linkage_method_time,
            "Complete Linkage Method Distance": complete_linkage_method_distance,
            "Complete Linkage Method Distance Calculation Time": complete_linkage_method_time,
            "Single Linkage Method Distance": single_linkage_method_distance,
            "Single Linkage Method Distance Calculation Time": single_linkage_method_time,
            "Leverage Score Distance": leverage_score_distance,
            "Leverage Score Distance Calculation Time": leverage_score_time,
            "Cook Distance": cook_distance,
            "Cook Distance Calculation Time": cook_distance_score_time,
            "Orthogonal Distance to Lowess Line": orthogonal_distance_to_lowess_line_distance,
            "Orthogonal Distance to Lowess Line Calculation Time": orthogonal_distance_to_lowess_line_time,
            "Vertical Distance to Lowess Line": vertical_distance_to_lowess_line_distance,
            "Vertical Distance to Lowess Line Calculation Time": vertical_distance_to_lowess_line_time,
            "Horizontal Distance to Lowess Line": horizontal_distance_to_lowess_line_distance,
            "Horizontal Distance to Lowess Line Calculation Time": horizontal_distance_to_lowess_line_time,
            "Visibility Index": visibility_index,
            "Visibility Index Calculation Time": visibility_index_time,
            "MPix": mpix_result,
            "MPix Calculation Time": mpix_time,
            "Grid Density Overlap Degree": grid_density_overlap_degree,
            "Grid Density Overlap Degree Calculation Time": grid_density_overlap_degree_time,
            "Kernel Density Overlap Degree": kernel_density_overlap_degree,
            "Kernel Density Overlap Degree Calculation Time": kernel_density_overlap_degree_time,
            "Nearest Neighbor Distance": nearest_neighbor_distance,
            "Nearest Neighbor Distance Calculation Time": nearest_neighbor_distance_time,
            "Pairwise Bounding Box Overlap Degree": pairwise_bounding_box_overlap_degree,
            "Pairwise Bounding Box Overlap Degree Calculation Time": pairwise_bounding_box_overlap_degree_time,
            "No. Covered Data Points": number_covered_values,
            "No. Covered Data Points by Different Classes": number_covered_values_different_classes,
            "Categories Covered Data Points by Different Classes": covered_values_different_classes
        }
    else:
        metadata = {
            "Scatterplot Name": scatterplot_filename,
            "Dataset Name": dataset_filename,
            "Number of Entire Data": len(data),
            'Number of Classes': len(data['class'].unique()),
            "Marker Size": marker_size,
            "M_distance": m_distance,
            "M_distance Calculation Time": m_distance_time,
            "LOF Distance": lof_distance,
            "LOF Distance Calculation Time": lof_distance_time,
            "Isolation Forest Distance": iso_distance,
            "Isolation Forest Distance Calculation Time": iso_distance_time,
            "Centroid Distance": centroid_distance,
            "Centroid Distance Calculation Time": centoid_time,
            "Average Linkage Method Distance": average_linkage_method_distance,
            "Average Linkage Method Distance Calculation Time": average_linkage_method_time,
            "Complete Linkage Method Distance": complete_linkage_method_distance,
            "Complete Linkage Method Distance Calculation Time": complete_linkage_method_time,
            "Single Linkage Method Distance": single_linkage_method_distance,
            "Single Linkage Method Distance Calculation Time": single_linkage_method_time,
            "Leverage Score Distance": leverage_score_distance,
            "Leverage Score Distance Calculation Time": leverage_score_time,
            "Cook Distance": cook_distance,
            "Cook Distance Calculation Time": cook_distance_score_time,
            "Orthogonal Distance to Lowess Line": orthogonal_distance_to_lowess_line_distance,
            "Orthogonal Distance to Lowess Line Calculation Time": orthogonal_distance_to_lowess_line_time,
            "Vertical Distance to Lowess Line": vertical_distance_to_lowess_line_distance,
            "Vertical Distance to Lowess Line Calculation Time": vertical_distance_to_lowess_line_time,
            "Horizontal Distance to Lowess Line": horizontal_distance_to_lowess_line_distance,
            "Horizontal Distance to Lowess Line Calculation Time": horizontal_distance_to_lowess_line_time,
            "Visibility Index": visibility_index,
            "Visibility Index Calculation Time": visibility_index_time,
            "MPix": mpix_result,
            "MPix Calculation Time": mpix_time,
            "Grid Density Overlap Degree": grid_density_overlap_degree,
            "Grid Density Overlap Degree Calculation Time": grid_density_overlap_degree_time,
            "Kernel Density Overlap Degree": kernel_density_overlap_degree,
            "Kernel Density Overlap Degree Calculation Time": kernel_density_overlap_degree_time,
            "Nearest Neighbor Distance": nearest_neighbor_distance,
            "Nearest Neighbor Distance Calculation Time": nearest_neighbor_distance_time,
            "Pairwise Bounding Box Overlap Degree": pairwise_bounding_box_overlap_degree,
            "Pairwise Bounding Box Overlap Degree Calculation Time": pairwise_bounding_box_overlap_degree_time,
            "No. Covered Data Points": number_covered_values,
            "No. Covered Data Points by Different Classes": number_covered_values_different_classes,
            "Categories Covered Data Points by Different Classes": covered_values_different_classes
        }

    metadata_csv_path = metadata_save_folder + metadata_csv
    metadata_df = pd.DataFrame([metadata])

    if os.path.exists(metadata_csv_path):
        existing_metadata_df = pd.read_csv(metadata_csv_path)
        if metadata["Scatterplot Name"] in existing_metadata_df["Scatterplot Name"].values:
            existing_metadata_df.update(metadata_df)
            existing_metadata_df.to_csv(metadata_csv_path, index=False)
        else:
            metadata_df.to_csv(metadata_csv_path, mode='a', header=False, index=False)
    else:
        metadata_df.to_csv(metadata_csv_path, index=False)
    # ==================================================================================================



# ==================================================================================================
# Define the parameters for generating scatterplots
marker_size = [10, 20, 50]
# marker_size = [10, 50]

# Define the folder for saving the generated scatterplots
figure_folder_name = os.getcwd() + '/datasets/simulated_datasets/figure_files_1_fu'
# figure_folder_name = os.getcwd() + '/datasets/simulated_datasets/figure_files_2'
# figure_folder_name = os.getcwd() + '/datasets/simulated_datasets/figures_test_folder'
# Ensure the folder for saving figures exists
os.makedirs(figure_folder_name, exist_ok=True)


# Load data
# data_folder = os.path.join(os.getcwd(), 'datasets/DimRed_data/abc')
data_folder = os.getcwd() + '/datasets/simulated_datasets/csv_files_1/fu'
# data_folder = os.getcwd() + '/datasets/simulated_datasets/csv_files_2'
# data_folder = os.getcwd() + '/datasets/simulated_datasets/test_folder'


csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]


# Define the folder for saving the metadata
metadata_save_folder = os.getcwd() + '/datasets/simulated_datasets/'

# Define the name of the metadata file
# metadata_csv = "metadata_metrics_2.csv" 
metadata_csv = "metadata_metrics_1_fu.csv"
# metadata_csv = "metadata_test_data.csv"

# define the order methods
order_method = ['category_based', 'ascending', 'descending']


for csv_file in csv_files:
    data_file = os.path.join(data_folder, csv_file)
    data_filename = os.path.basename(data_file).replace('.csv', '')
    data = pd.read_csv(data_file)
    
    for _ in order_method:
        for ms in marker_size:
            visualize_and_save(
                data,
                marker_size=ms,
                figure_save_folder=figure_folder_name,
                metadata_save_folder= metadata_save_folder,
                scatterplot_filename='figure_' + str(_) + 'markersize' + str(ms) + '_' + data_filename, 
                dataset_filename=data_filename,
                metadata_csv=metadata_csv,
                order_method = _
            )
# ==================================================================================================

