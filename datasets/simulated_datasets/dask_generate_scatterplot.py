import os
import sys
import argparse
import multiprocessing
import pandas as pd
import json

# Use dask.distributed for both local multicore and multi-node clusters
from dask.distributed import Client, LocalCluster

# adjust import path as before
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Our_metrics.Scatter_Metrics import Scatter_Metric


def visualize_and_save(data, marker_size, figure_save_folder,
                       scatterplot_filename, dataset_filename):
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

    # other metrics
    for fn, key in [
        (scatter_metric.visilibity_index, 'VisibilityIndex'),
        (scatter_metric.mpix, 'MPix'),
        (lambda: scatter_metric.grid_based_density_overlap_degree(grid_pixel_ratio=1), 'GridOverlap'),
        (lambda: scatter_metric.kernel_density_estimation(bandwidth=0.1, gridsize=100), 'KDEOverlap'),
        (scatter_metric.nearest_neighbor_distance, 'NNDistance'),
        (scatter_metric.pairwise_bounding_box_based_overlap_degree, 'PBBOverlap')
    ]:
        fn()
        results[key] = scatter_metric.result
        results[f"{key}_time"] = scatter_metric.calculation_time

    # coverage metrics
    num_vals, num_diff, covered_diff = scatter_metric.cal_covered_data_points()
    num_px, _ = scatter_metric.calculate_existing_pixel_coverage()
    num_diff_px, _ = scatter_metric.calculate_pixels_covered_by_different_categories()
    results.update({
        "CoveredVals": num_vals,
        "CoveredValsDiffClasses": num_diff,
        "CoveredPixels": num_px,
        "CoveredPixelsDiffClasses": num_diff_px,
        "CoveredValuesDiffJSON": json.dumps(covered_diff)
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
    """Unpack task dict, read CSV, and call visualize_and_save"""
    data = pd.read_csv(task['csv_file'])
    return visualize_and_save(
        data,
        marker_size=task['marker_size'],
        figure_save_folder=task['figure_folder'],
        scatterplot_filename=task['scatterplot_filename'],
        dataset_filename=task['dataset_filename']
    )

def main():

    parser = argparse.ArgumentParser(
        description='Generate scatterplot metrics using multicore, distributed, or Slurm array')
    
    parser.add_argument('--data-folder', type=str,
                        default=os.getcwd() + '/datasets/simulated_datasets/csv_files_1/fu')
    parser.add_argument('--figure-folder', type=str,
                        default=os.getcwd() + '/datasets/simulated_datasets/figure_files_0611')
    parser.add_argument('--metadata-csv-path', type=str,
                        default=os.getcwd() + '/datasets/simulated_datasets/metadata_metrics_simulated_0611.csv')
    parser.add_argument('--marker-sizes', type=int, nargs='+',
                        default=[10, 60, 110, 160])
    parser.add_argument('--order-methods', type=str, nargs='+',
                        default=['category_based', 'ascending', 'descending'])
    parser.add_argument('--file', type=str,
                        help='Optional single CSV file to process')
    parser.add_argument('--scheduler-address', type=str,
                        help='Dask scheduler address for distributed mode (e.g. tcp://host:8786)')
    parser.add_argument('--task-index', type=int,
                        help='Zero-based index when using Slurm array jobs')
    parser.add_argument('--n-tasks', type=int,
                        help='Total number of Slurm array tasks')
    args = parser.parse_args()

    # Gather CSV files
    if args.file:
        csvs = [args.file]
    else:
        csvs = [os.path.join(args.data_folder, f)
                for f in os.listdir(args.data_folder)
                if f.endswith('.csv')]

    # Build full task list
    tasks = []
    for csv_file in csvs:
        base = os.path.splitext(os.path.basename(csv_file))[0]
        for order in args.order_methods:
            for ms in args.marker_sizes:
                fname = f"figure_{order}_markersize{ms}_{base}.png"
                tasks.append({
                    'csv_file': csv_file,
                    'marker_size': ms,
                    'figure_folder': args.figure_folder,
                    'scatterplot_filename': fname,
                    'dataset_filename': base
                })

    # If running as a Slurm array, pick subset of tasks
    if args.task_index is not None and args.n_tasks:
        total = len(tasks)
        tasks = tasks[args.task_index::args.n_tasks]
        print(f"[SLURM ARRAY] Task {args.task_index}/{args.n_tasks} running {len(tasks)} of {total} total tasks")

    # Initialize Dask client (local or remote)
    if args.scheduler_address:
        client = Client(args.scheduler_address)
    else:
        cluster = LocalCluster(n_workers=multiprocessing.cpu_count(), threads_per_worker=1)
        client = Client(cluster)

    # Submit and gather
    futures = client.map(run_task, tasks)
    all_metadata = client.gather(futures)

    # Write metadata CSV
    os.makedirs(os.path.dirname(args.metadata_csv_path), exist_ok=True)
    pd.DataFrame(all_metadata).to_csv(args.metadata_csv_path, index=False)
    print(f"Wrote metadata for {len(all_metadata)} figures to {args.metadata_csv_path}")


if __name__ == '__main__':

    main()

