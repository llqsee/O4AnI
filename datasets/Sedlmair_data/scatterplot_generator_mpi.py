import os
import sys
import argparse
from mpi4py import MPI
import pandas as pd
import json
import numpy as np


# adjust import path as before
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Our_metrics.Scatter_Metrics import Scatter_Metric


def visualize_and_save(data, marker_size, figure_save_folder,
                       scatterplot_filename, dataset_filename, order_method=None):
    """Generate the scatterplot, compute metrics, save figure, and return metadata dict."""
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

    if order_method == 'category_based':
        projected_labels = data['class'].unique().tolist()
        scatter_metric._sort_data(attribute='class', order=projected_labels)
        # holders for results
        results = {}
        # compute weighted metrics
        for w in weight_same_classes:
            for method in [
                ('mahalanobis_distance', 'M'),
                ('lof_distance', 'LOF'),
                ('isolation_forest', 'IsolationForest'),
                ('centroid_method', 'Centroid'),
                ('average_linkage_method', 'AvgLinkage')
            ]:
                method_key, prefix = method[1], method[1]
                scatter_metric.importance_metric(
                    important_cal_method=method[0],
                    weight_diff_class=weight_diff_class,
                    weight_same_class=w
                )
                results[f"{prefix}_Dist_10_{w}"] = scatter_metric.result
                results[f"{prefix}_Dist_10_{w}_time"] = scatter_metric.calculation_time
    elif order_method == 'ascending':
        
        # holders for results
        results = {}
        # compute weighted metrics
        for w in weight_same_classes:
            for method in [
                ('mahalanobis_distance', 'M'),
                ('lof_distance', 'LOF'),
                ('isolation_forest', 'IsolationForest'),
                ('centroid_method', 'Centroid'),
                ('average_linkage_method', 'AvgLinkage')
            ]:
                method_key, prefix = method[1], method[1]
                scatter_metric.importance_metric(
                    important_cal_method=method[0],
                    weight_diff_class=weight_diff_class,
                    weight_same_class=w,
                    order_variable='importance_index',
                    asending=True
                )
                results[f"{prefix}_Dist_10_{w}"] = scatter_metric.result
                results[f"{prefix}_Dist_10_{w}_time"] = scatter_metric.calculation_time
    else:
        # holders for results
        results = {}
        # compute weighted metrics
        for w in weight_same_classes:
            for method in [
                ('mahalanobis_distance', 'M'),
                ('lof_distance', 'LOF'),
                ('isolation_forest', 'IsolationForest'),
                ('centroid_method', 'Centroid'),
                ('average_linkage_method', 'AvgLinkage')
            ]:
                method_key, prefix = method[1], method[1]
                scatter_metric.importance_metric(
                    important_cal_method=method[0],
                    weight_diff_class=weight_diff_class,
                    weight_same_class=w,
                    order_variable='importance_index',
                    asending=False
                )
                results[f"{prefix}_Dist_10_{w}"] = scatter_metric.result
                results[f"{prefix}_Dist_10_{w}_time"] = scatter_metric.calculation_time
                



    # other metrics
    for fn, key in [
        (scatter_metric.visilibity_index, 'VisibilityIndex'),
        (scatter_metric.mpix, 'MPix'),
        (lambda: scatter_metric.grid_based_density_overlap_degree(grid_pixel_ratio=1), 'GridOverlap'),
        (lambda: scatter_metric.kernel_density_estimation(bandwidth=0.1, gridsize=100), 'KDEOverlap'),
        (scatter_metric.nearest_neighbor_distance, 'NNDistance'),
        (scatter_metric.pairwise_bounding_box_based_overlap_degree, 'PBBOverlap'),
        (scatter_metric.distance_consistency, 'DistanceConsistency')
    ]:
        fn()
        results[key] = scatter_metric.result
        results[f"{key}_time"] = scatter_metric.calculation_time

    # coverage metrics
    num_diff, num_diff_px = scatter_metric.cal_covered_data_points()
    num_diff_px_general_method = scatter_metric.calculate_pixels_covered_by_different_categories()
    
    results.update({
        # "CoveredVals": num_vals,
        "CoveredValsDiffClasses": num_diff,
        "CoveredPixelsDiffClasses": num_diff_px,
        'CoveredPixelsDiffClassesGeneralMethod': num_diff_px_general_method
        # "CoveredValuesDiffJSON": json.dumps(covered_diff)
    })

    # save figure
    os.makedirs(figure_save_folder, exist_ok=True)
    plot_path = os.path.join(figure_save_folder, scatterplot_filename)
    scatter_metric.save_figure(filename=plot_path)

    # build metadata dict
    metadata = {
        "ScatterplotName": scatterplot_filename,
        "DatasetName": dataset_filename,
        "TotalPoints": len(data),
        "NumClasses": data['class'].nunique(),
        "MarkerSize": marker_size,
    }
    metadata.update(results)
    return metadata


def run_task(task):
    data = pd.read_csv(task['csv_file'])
    return visualize_and_save(
      data,
      marker_size=task['marker_size'],
      figure_save_folder=task['figure_folder'],
      scatterplot_filename=task['scatterplot_filename'],
      dataset_filename=task['dataset_filename'],
      order_method=task['order_method'] if 'order_method' in task else None
    )

def main():

    parser = argparse.ArgumentParser(
        description='Generate scatterplot metrics using multicore, distributed, or Slurm array')
    parser.add_argument('--data-folder', type=str,
                        default=os.getcwd() + '/datasets/Sedlmair_data/DR_data')
    parser.add_argument('--figure-folder', type=str,
                        default=os.getcwd() + '/datasets/Sedlmair_data/figure_files_0623_afternoon')
    parser.add_argument('--metadata-csv-path', type=str,
                        default=os.getcwd() + '/datasets/Sedlmair_data/metadata_metrics_sedlmair_0623_afternoon.csv')
    parser.add_argument('--marker-sizes', type=int, nargs='+',
                        default=[10, 60, 110, 160])
    parser.add_argument('--order-methods', type=str, nargs='+',
                        default=['category_based', 'ascending', 'descending'])
    parser.add_argument('--file', type=str,
                        help='Optional single CSV file to process')
    args = parser.parse_args()
    
    
    
    
    # 1) Build the full flat list of “tasks”
    if args.file:
        csvs = [args.file]
    else:
        csvs = [os.path.join(args.data_folder, f)
                for f in os.listdir(args.data_folder) if f.endswith('.csv')]

    all_tasks = []
    for csv_file in csvs:
        base = os.path.splitext(os.path.basename(csv_file))[0]
        for order in args.order_methods:
            for ms in args.marker_sizes:
                fname = f"figure_{order}_markersize{ms}_{base}.png"
                all_tasks.append({
                    'csv_file':         csv_file,
                    'marker_size':      ms,
                    'figure_folder':    args.figure_folder,
                    'scatterplot_filename': fname,
                    'dataset_filename': base,
                    'order_method':     order
                })

    # 2) MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 3) scatter the list of tasks
    if rank == 0:
        # split evenly into `size` chunks
        chunks = list(np.array_split(all_tasks, size))
    else:
        chunks = None

    my_tasks = comm.scatter(chunks, root=0)

    # 4) run your tasks
    my_results = [run_task(t) for t in my_tasks]

    # 5) gather back to rank 0
    gathered = comm.gather(my_results, root=0)

    if rank == 0:
        # flatten
        flat = [r for sub in gathered for r in sub]
        # write out the metadata CSV
        pd.DataFrame(flat).to_csv(args.metadata_csv_path, index=False)
        print(f"Wrote metadata for {len(flat)} figures to {args.metadata_csv_path}")

if __name__ == "__main__":
    main()