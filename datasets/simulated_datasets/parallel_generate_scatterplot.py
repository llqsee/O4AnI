import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import multiprocessing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC

import pandas as pd
from Our_metrics.Scatter_Metrics import Scatter_Metric
import json


def visualize_and_save(data, marker_size, figure_save_folder, metadata_save_folder, scatterplot_filename="", 
                       dataset_filename='', metadata_csv="metadata_metrics.csv", order_method=None):
    """Visualize and save the generated scatterplot along with metadata."""
    # ==================================================================================================
    # Initialize the scatter metric
    scatter_metric = Scatter_Metric(
        data,
        margins={'left': 0.15, 'right': 0.75, 'top': 0.9, 'bottom': 0.1},
        marker='square',
        marker_size=marker_size,
        dpi=100,
        figsize=[10, 8],
        xvariable='1',
        yvariable='2',
        zvariable='class',
        color_map='tab10'
    )

    weight_diff_class = 10
    weight_same_classes = [0, 5, 10]

    # Prepare dictionaries to hold results per weight
    m_distances = {}
    m_distances_time = {}
    lof_distances = {}
    lof_distances_time = {}
    iso_distances = {}
    iso_distances_time = {}
    centroid_distances = {}
    centroid_distances_time = {}
    avg_linkage_distances = {}
    avg_linkage_distances_time = {}

    # ==================================================================================================
    # Compute importance metrics for each weight_same_class
    for w in weight_same_classes:
        # Mahalanobis distance
        if order_method == 'category_based':
            projected_labels = data['class'].unique().tolist()
            scatter_metric._sort_data(attribute='class', order=projected_labels)
            scatter_metric.importance_metric(
                important_cal_method='mahalanobis_distance',
                weight_diff_class=weight_diff_class,
                weight_same_class=w
            )
        elif order_method == 'ascending':
            scatter_metric.importance_metric(
                important_cal_method='mahalanobis_distance',
                weight_diff_class=weight_diff_class,
                weight_same_class=w,
                order_variable='importance_index',
                asending=True
            )
        else:
            scatter_metric.importance_metric(
                important_cal_method='mahalanobis_distance',
                weight_diff_class=weight_diff_class,
                weight_same_class=w,
                order_variable='importance_index',
                asending=False
            )
        m_distances[w] = scatter_metric.result
        m_distances_time[w] = scatter_metric.calculation_time

        # LOF distance
        scatter_metric.importance_metric(
            important_cal_method='lof_distance',
            weight_diff_class=weight_diff_class,
            weight_same_class=w
        )
        lof_distances[w] = scatter_metric.result
        lof_distances_time[w] = scatter_metric.calculation_time

        # Isolation Forest
        scatter_metric.importance_metric(
            important_cal_method='isolation_forest',
            weight_diff_class=weight_diff_class,
            weight_same_class=w
        )
        iso_distances[w] = scatter_metric.result
        iso_distances_time[w] = scatter_metric.calculation_time

        # Centroid method
        scatter_metric.importance_metric(
            important_cal_method='centroid_method',
            weight_diff_class=weight_diff_class,
            weight_same_class=w
        )
        centroid_distances[w] = scatter_metric.result
        centroid_distances_time[w] = scatter_metric.calculation_time

        # Average linkage method
        scatter_metric.importance_metric(
            important_cal_method='average_linkage_method',
            weight_diff_class=weight_diff_class,
            weight_same_class=w
        )
        avg_linkage_distances[w] = scatter_metric.result
        avg_linkage_distances_time[w] = scatter_metric.calculation_time

    # ==================================================================================================
    # Compute additional metrics
    scatter_metric.visilibity_index()
    visibility_index = scatter_metric.result
    visibility_index_time = scatter_metric.calculation_time

    scatter_metric.mpix()
    mpix_result = scatter_metric.result
    mpix_time = scatter_metric.calculation_time

    scatter_metric.grid_based_density_overlap_degree(grid_pixel_ratio=1)
    grid_density_overlap_degree = scatter_metric.result
    grid_density_overlap_degree_time = scatter_metric.calculation_time

    scatter_metric.kernel_density_estimation(bandwidth=0.1, gridsize=100)
    kernel_density_overlap_degree = scatter_metric.result
    kernel_density_overlap_degree_time = scatter_metric.calculation_time

    scatter_metric.nearest_neighbor_distance()
    nearest_neighbor_distance = scatter_metric.result
    nearest_neighbor_distance_time = scatter_metric.calculation_time

    scatter_metric.pairwise_bounding_box_based_overlap_degree()
    pairwise_bounding_box_overlap_degree = scatter_metric.result
    pairwise_bounding_box_overlap_degree_time = scatter_metric.calculation_time

    # Covered points and pixels
    number_covered_values, number_covered_values_different_classes, covered_values_different_classes = scatter_metric.cal_covered_data_points()
    number_covered_different_pixels, _ = scatter_metric.calculate_pixels_covered_by_different_categories()
    number_covered_pixels, _ = scatter_metric.calculate_existing_pixel_coverage()
    
    # before you build metadata dict, serialize that list:
    covered_json = json.dumps(covered_values_different_classes)

    # ==================================================================================================
    # Save the figure
    plot_path = os.path.join(figure_save_folder, scatterplot_filename)
    scatter_metric.save_figure(filename=plot_path)

    # ==================================================================================================
    # Prepare metadata
    metadata = {
        "Scatterplot Name": scatterplot_filename,
        "Dataset Name": dataset_filename,
        "Number of Entire Data": len(data),
        "Number of Classes": len(data['class'].unique()),
        "Marker Size": marker_size,
        # Per-weight metrics
        "M_distance_10_0": m_distances[0],
        "M_distance_10_0_time": m_distances_time[0],
        "M_distance_10_5": m_distances[5],
        "M_distance_10_5_time": m_distances_time[5],
        "M_distance_10_10": m_distances[10],
        "M_distance_10_10_time": m_distances_time[10],
        "LOF Distance_10_0": lof_distances[0],
        "LOF Distance_10_0 Calculation Time": lof_distances_time[0],
        "LOF Distance_10_5": lof_distances[5],
        "LOF Distance_10_5 Calculation Time": lof_distances_time[5],
        "LOF Distance_10_10": lof_distances[10],
        "LOF Distance_10_10 Calculation Time": lof_distances_time[10],
        "Isolation Forest Distance_10_0": iso_distances[0],
        "Isolation Forest Distance_10_0 Calculation Time": iso_distances_time[0],
        "Isolation Forest Distance_10_5": iso_distances[5],
        "Isolation Forest Distance_10_5 Calculation Time": iso_distances_time[5],
        "Isolation Forest Distance_10_10": iso_distances[10],
        "Isolation Forest Distance_10_10 Calculation Time": iso_distances_time[10],
        "Centroid Distance_10_0": centroid_distances[0],
        "Centroid Distance_10_0 Calculation Time": centroid_distances_time[0],
        "Centroid Distance_10_5": centroid_distances[5],
        "Centroid Distance_10_5 Calculation Time": centroid_distances_time[5],
        "Centroid Distance_10_10": centroid_distances[10],
        "Centroid Distance_10_10 Calculation Time": centroid_distances_time[10],
        "Average Linkage Method Distance_10_0": avg_linkage_distances[0],
        "Average Linkage Method Distance_10_0 Calculation Time": avg_linkage_distances_time[0],
        "Average Linkage Method Distance_10_5": avg_linkage_distances[5],
        "Average Linkage Method Distance_10_5 Calculation Time": avg_linkage_distances_time[5],
        "Average Linkage Method Distance_10_10": avg_linkage_distances[10],
        "Average Linkage Method Distance_10_10 Calculation Time": avg_linkage_distances_time[10],
        # Overlap and density metrics
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
        # Coverage
        "No. Covered Data Points": number_covered_values,
        "No. Covered Data Points by Different Classes": number_covered_values_different_classes,
        "No. Covered Pixels": number_covered_pixels,
        "No. Covered Pixels by Different Classes": number_covered_different_pixels,
        "Categories Covered Data Points by Different Classes": covered_json
    }

    # ==================================================================================================
    # Write metadata to CSV
    metadata_csv_path = os.path.join(metadata_save_folder, metadata_csv)
    metadata_df = pd.DataFrame([metadata])
    if os.path.exists(metadata_csv_path):
        existing_metadata_df = pd.read_csv(metadata_csv_path)
        if scatterplot_filename in existing_metadata_df["Scatterplot Name"].values:
            existing_metadata_df.update(metadata_df)
            existing_metadata_df.to_csv(metadata_csv_path, index=False)
        else:
            metadata_df.to_csv(metadata_csv_path, mode='a', header=False, index=False)
    else:
        metadata_df.to_csv(metadata_csv_path, index=False)


def process_file(args):
    csv_file, marker_sizes, order_methods, figure_folder, metadata_folder, metadata_csv = args
    data_filename = os.path.splitext(os.path.basename(csv_file))[0]
    data = pd.read_csv(csv_file)
    for order in order_methods:
        for ms in marker_sizes:
            scatterplot_filename = f"figure_{order}_markersize{ms}_{data_filename}.png"
            visualize_and_save(
                data,
                marker_size=ms,
                figure_save_folder=figure_folder,
                metadata_save_folder=metadata_folder,
                scatterplot_filename=scatterplot_filename,
                dataset_filename=data_filename,
                metadata_csv=metadata_csv,
                order_method=order
            )


if __name__ == '__main__':
    
    
    def function_1():
        parser = argparse.ArgumentParser(description='Generate scatterplot metrics in parallel')
        parser.add_argument('--data-folder', type=str, default=os.getcwd() + '/datasets/simulated_datasets/csv_files_1/fu',
                            help='Folder containing CSV files')
        parser.add_argument('--figure-folder', type=str, default=os.getcwd() + '/datasets/simulated_datasets/figure_files_0610',
                            help='Folder to save generated figures')
        parser.add_argument('--metadata-folder', type=str, default=os.getcwd() + '/datasets/simulated_datasets/',
                            help='Folder to save metadata CSV')
        parser.add_argument('--metadata-csv', type=str, default='metadata_metrics_simulated_0610.csv',
                            help='Metadata CSV filename')
        parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                            help='Number of parallel worker processes')
        parser.add_argument('--file', type=str, help='Optional: path to a single CSV file to process')

        args = parser.parse_args()

        # Ensure output directories exist
        os.makedirs(args.figure_folder, exist_ok=True)
        os.makedirs(args.metadata_folder, exist_ok=True)

        # Define parameters
        marker_sizes = [10, 60, 110, 160]
        order_methods = ['category_based', 'ascending', 'descending']

        # Collect CSV files
        if args.file:
            csv_list = [args.file]
        else:
            csv_list = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder) if f.endswith('.csv')]

        # SLURM array support: pick one file if ARRAY_TASK_ID set
        slurm_id = os.getenv('SLURM_ARRAY_TASK_ID')
        if slurm_id is not None and not args.file:
            idx = int(slurm_id)
            csv_list = [csv_list[idx]]

        # Prepare tasks
        tasks = [
            (csv, marker_sizes, order_methods, args.figure_folder, args.metadata_folder, args.metadata_csv)
            for csv in csv_list
        ]

        # Run in parallel or sequentially if single task
        if len(tasks) > 1:
            with multiprocessing.Pool(processes=args.workers) as pool:
                pool.map(process_file, tasks)
        else:
            process_file(tasks[0])

        print("Processing complete.")
    
    function_1()