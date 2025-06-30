import os
import sys
import argparse
import multiprocessing
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for HPC

import pandas as pd
import json

# adjust import path as before
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Our_metrics.Scatter_Metrics import Scatter_Metric


def visualize_and_save(data, marker_size, figure_save_folder,
                       scatterplot_filename, dataset_filename):
    """Generate the scatterplot, compute metrics, save figure, and return metadata dict."""
    # --- initialize metric object ---
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

    # prepare holders
    m_distances = {}
    m_times     = {}
    lof_distances = {}
    lof_times     = {}
    iso_distances = {}
    iso_times     = {}
    cent_distances = {}
    cent_times     = {}
    avg_distances  = {}
    avg_times      = {}

    # loop over same-class weights
    for w in weight_same_classes:
        # mahalanobis
        scatter_metric.importance_metric(
            important_cal_method='mahalanobis_distance',
            weight_diff_class=weight_diff_class,
            weight_same_class=w,
            # use your desired ordering here
        )
        m_distances[w] = scatter_metric.result
        m_times[w]     = scatter_metric.calculation_time

        # LOF
        scatter_metric.importance_metric(
            important_cal_method='lof_distance',
            weight_diff_class=weight_diff_class,
            weight_same_class=w
        )
        lof_distances[w] = scatter_metric.result
        lof_times[w]     = scatter_metric.calculation_time

        # Isolation Forest
        scatter_metric.importance_metric(
            important_cal_method='isolation_forest',
            weight_diff_class=weight_diff_class,
            weight_same_class=w
        )
        iso_distances[w] = scatter_metric.result
        iso_times[w]     = scatter_metric.calculation_time

        # Centroid
        scatter_metric.importance_metric(
            important_cal_method='centroid_method',
            weight_diff_class=weight_diff_class,
            weight_same_class=w
        )
        cent_distances[w] = scatter_metric.result
        cent_times[w]     = scatter_metric.calculation_time

        # Average linkage
        scatter_metric.importance_metric(
            important_cal_method='average_linkage_method',
            weight_diff_class=weight_diff_class,
            weight_same_class=w
        )
        avg_distances[w] = scatter_metric.result
        avg_times[w]     = scatter_metric.calculation_time

    # other metrics
    scatter_metric.visilibity_index()
    visibility_index      = scatter_metric.result
    visibility_time       = scatter_metric.calculation_time

    scatter_metric.mpix()
    mpix_result           = scatter_metric.result
    mpix_time             = scatter_metric.calculation_time

    scatter_metric.grid_based_density_overlap_degree(grid_pixel_ratio=1)
    grid_overlap          = scatter_metric.result
    grid_overlap_time     = scatter_metric.calculation_time

    scatter_metric.kernel_density_estimation(bandwidth=0.1, gridsize=100)
    kde_overlap           = scatter_metric.result
    kde_overlap_time      = scatter_metric.calculation_time

    scatter_metric.nearest_neighbor_distance()
    nn_distance           = scatter_metric.result
    nn_distance_time      = scatter_metric.calculation_time

    scatter_metric.pairwise_bounding_box_based_overlap_degree()
    pbb_overlap           = scatter_metric.result
    pbb_overlap_time      = scatter_metric.calculation_time

    # coverage metrics
    number_covered_values, number_covered_diff, covered_values_diff = \
        scatter_metric.cal_covered_data_points()
    number_covered_pixels, _ = scatter_metric.calculate_existing_pixel_coverage()
    number_covered_diff_px, _ = scatter_metric.calculate_pixels_covered_by_different_categories()
    covered_json = json.dumps(covered_values_diff)

    # save figure
    os.makedirs(figure_save_folder, exist_ok=True)
    plot_path = os.path.join(figure_save_folder, scatterplot_filename)
    scatter_metric.save_figure(filename=plot_path)

    # build metadata dict
    metadata = {
        "Scatterplot Name": scatterplot_filename,
        "Dataset Name":    dataset_filename,
        "Number of Entire Data": len(data),
        "Number of Classes":      data['class'].nunique(),
        "Marker Size":            marker_size,
        # per-weight:
        **{
            f"M_distance_10_{w}":     m_distances[w]
            for w in weight_same_classes
        },
        **{
            f"M_distance_10_{w}_time": m_times[w]
            for w in weight_same_classes
        },
        **{
            f"LOF Distance_10_{w}":       lof_distances[w]
            for w in weight_same_classes
        },
        **{
            f"LOF Distance_10_{w}_time":  lof_times[w]
            for w in weight_same_classes
        },
        **{
            f"Isolation Forest Distance_10_{w}":      iso_distances[w]
            for w in weight_same_classes
        },
        **{
            f"Isolation Forest Distance_10_{w}_time": iso_times[w]
            for w in weight_same_classes
        },
        **{
            f"Centroid Distance_10_{w}":      cent_distances[w]
            for w in weight_same_classes
        },
        **{
            f"Centroid Distance_10_{w}_time": cent_times[w]
            for w in weight_same_classes
        },
        **{
            f"Average Linkage Method Distance_10_{w}":      avg_distances[w]
            for w in weight_same_classes
        },
        **{
            f"Average Linkage Method Distance_10_{w}_time": avg_times[w]
            for w in weight_same_classes
        },
        # other metrics:
        "Visibility Index":                    visibility_index,
        "Visibility Index Calculation Time":   visibility_time,
        "MPix":                                mpix_result,
        "MPix Calculation Time":               mpix_time,
        "Grid Density Overlap Degree":         grid_overlap,
        "Grid Density Overlap Degree Time":    grid_overlap_time,
        "Kernel Density Overlap Degree":       kde_overlap,
        "Kernel Density Overlap Degree Time":  kde_overlap_time,
        "Nearest Neighbor Distance":           nn_distance,
        "Nearest Neighbor Distance Time":      nn_distance_time,
        "Pairwise Bounding Box Overlap Degree":        pbb_overlap,
        "Pairwise Bounding Box Overlap Time":          pbb_overlap_time,
        # coverage:
        "No. Covered Data Points":                    number_covered_values,
        "No. Covered Data Points by Different Classes": number_covered_diff,
        "No. Covered Pixels":                          number_covered_pixels,
        "No. Covered Pixels by Different Classes":     number_covered_diff_px,
        "Categories Covered Data Points by Different Classes": covered_json
    }

    return metadata


def process_all(args):
    # gather CSV files
    if args.file:
        csvs = [args.file]
    else:
        csvs = [
            os.path.join(args.data_folder, f)
            for f in os.listdir(args.data_folder)
            if f.endswith('.csv')
        ]

    all_metadata = []
    for csv_file in csvs:
        data = pd.read_csv(csv_file)
        base = os.path.splitext(os.path.basename(csv_file))[0]

        for order in args.order_methods:
            for ms in args.marker_sizes:
                fname = f"figure_{order}_markersize{ms}_{base}.png"
                md = visualize_and_save(
                    data,
                    marker_size=ms,
                    figure_save_folder=args.figure_folder,
                    scatterplot_filename=fname,
                    dataset_filename=base
                )
                all_metadata.append(md)

    # write once
    os.makedirs(os.path.dirname(args.metadata_csv_path), exist_ok=True)
    pd.DataFrame(all_metadata).to_csv(args.metadata_csv_path, index=False)
    print(f"Wrote metadata for {len(all_metadata)} figures to {args.metadata_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate scatterplot metrics (single-pass metadata write)')
    parser.add_argument('--data-folder', type=str,
                        default=os.getcwd() + '/datasets/Sedlmair_data/DR_data/fu')
    parser.add_argument('--figure-folder', type=str,
                        default=os.getcwd() + '/datasets/Sedlmair_data/figure_files_0611')
    parser.add_argument('--metadata-csv-path', type=str,
                        default=os.getcwd() + '/datasets/Sedlmair_data/metadata_metrics_sedlmair_0611.csv')
    parser.add_argument('--marker-sizes', type=int, nargs='+',
                        default=[10, 60, 110, 160])
    parser.add_argument('--order-methods', type=str, nargs='+',
                        default=['category_based', 'ascending', 'descending'])
    parser.add_argument('--file', type=str,
                        help='Optional single CSV file to process')

    args = parser.parse_args()
    process_all(args)
