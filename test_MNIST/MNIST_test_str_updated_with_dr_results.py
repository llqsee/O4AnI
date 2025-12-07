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
import random


importance_index_methods = ['mahalanobis_distance', 'isolation_forest', 'lof_forest', 'average_linkage_method', 
                            'complete_linkage_method', 'single_linkage_method', 'centroid_method',  # for cluster-based method
                            'leverage_score', 'cook_distance', 'orthogonal_distance_to_lowess_line', 'vertical_distance_to_lowess_line', 'influence_score',
                            'horizontal_distance_to_lowess_line' # for distance-based method
                            ]


# file_location = 'datasets/mnist/mnist_pred_updated_str.csv'
file_location = 'datasets/mnist/mnist_pred_updated_str_with_dr_results_1027.csv'

data = load_data(file_location)  # Make sure load_data is properly defined

# Rendering orders to evaluate
render_orders = ['importance_index', 'category_based', 'random']

# Ensure output folders exist
scatter_out_dir = os.path.join('test_MNIST', 'generated_scatterplots')
figures_out_dir = os.path.join('test_MNIST', 'generated_figures')
os.makedirs(scatter_out_dir, exist_ok=True)
os.makedirs(figures_out_dir, exist_ok=True)

all_metadata = []

# Determine available DR coordinate pairs in the dataset (e.g., pca_x/pca_y, umap_x/umap_y)
dr_candidates = []
for col in data.columns:
    if col.endswith('_x'):
        prefix = col[:-2]
        if f'{prefix}_y' in data.columns:
            dr_candidates.append(prefix)

# If no DR columns are present, fall back to the original x/y columns used previously
if not dr_candidates:
    dr_candidates = [None]  # None => use original 'X coordinate'/'Y coordinate'

# Loop over dimensionality reductions and rendering orders
for dr in dr_candidates:
    dr_label = dr if dr is not None else 'original'
    print(f"Running DR: {dr_label}")
    for render_order in render_orders:
        print(f"  Running render order: {render_order}")
        # choose the x/y variable names depending on DR
        if dr is None:
            xvar = 'X coordinate'
            yvar = 'Y coordinate'
        else:
            xvar = f'{dr}_x'
            yvar = f'{dr}_y'

        analysis = Scatter_Metric(data, 
                                  margins = {'left':0.2, 'right': 0.7, 'top':0.8, 'bottom': 0.2},
                                marker = 'square', 
                                marker_size = 50, 
                                dpi = 100, 
                                figsize= (12, 8),
                                xvariable=xvar,
                                yvariable=yvar,
                                zvariable='pred',
                                color_map='tab10'
                                )

        # Apply the requested rendering order
        if render_order == 'importance_index':
            analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=10, weight_same_class=1, order_variable='importance_index', asending=True)
        elif render_order == 'category_based':
            projected_labels = data['pred'].unique().tolist()
            analysis._sort_data(attribute = 'pred', order = projected_labels)
            analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=10, weight_same_class=1)
        elif render_order == 'random':
            shuffled_data = analysis.data.sample(frac=1, random_state=random.randint(0, 1000)).reset_index(drop=True)
            analysis.data = shuffled_data
            analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=10, weight_same_class=1)

        max_value = np.nanmax(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))
        min_value = np.nanmin(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))

        if max_value != min_value:  # Avoid division by zero
            normalized_matrix = np.where(np.isnan(analysis.other_layer_matrix), np.nan, (analysis.other_layer_matrix - min_value) / (max_value - min_value))
        else:
            normalized_matrix = np.zeros_like(analysis.other_layer_matrix)

        # Visualize / save heatmap (kept in generated_figures)
        analysis.visualize_heat_map(analysis.overall_layer_matrix)
        heatmap_fname = os.path.join(figures_out_dir, f'heat_map_{os.path.splitext(os.path.basename(file_location))[0]}_{dr_label}_{render_order}.png')
        analysis.save_heatmap(filename = heatmap_fname)

        # Save scatterplot into dedicated scatter_out_dir
        scatter_fname = os.path.join(scatter_out_dir, f'scatterplot_{os.path.splitext(os.path.basename(file_location))[0]}_{dr_label}_{render_order}.png')
        analysis.save_figure(filename = scatter_fname)

        # Save importance bar in figures_out_dir
        analysis.bar_for_importance(analysis.data)
        bar_fname = os.path.join(figures_out_dir, f'importance_bar_{os.path.splitext(os.path.basename(file_location))[0]}_{dr_label}_{render_order}.png')
        analysis.save_importance_bar(filename = bar_fname)

        # Close figures to avoid memory warnings
        plt.close('all')

        # Compute a set of baseline and importance-based metrics (similar to mpi_scatterplot_generator)
        results = {}
        weight_diff_class = 10
        weight_same_classes = [0, 5, 10]

        methods = [
            ('mahalanobis_distance', 'M'),
            ('lof_distance', 'LOF'),
            ('isolation_forest', 'IsolationForest'),
            ('centroid_method', 'Centroid'),
            ('average_linkage_method', 'AvgLinkage')
        ]

        # Depending on the render_order, set ordering on analysis.data
        if render_order == 'category_based':
            projected_labels = data['pred'].unique().tolist()
            analysis._sort_data(attribute='pred', order=projected_labels)

        # Compute importance-based weighted metrics
        for w in weight_same_classes:
            for method in methods:
                m_name, prefix = method
                # Reuse the same analysis object; call importance_metric which sets analysis.result
                if render_order == 'importance_index':
                    analysis.importance_metric(important_cal_method=m_name, weight_diff_class=weight_diff_class, weight_same_class=w, order_variable='importance_index', asending=True)
                else:
                    analysis.importance_metric(important_cal_method=m_name, weight_diff_class=weight_diff_class, weight_same_class=w)
                results[f"{prefix}_Dist_10_{w}"] = analysis.result
                results[f"{prefix}_Dist_10_{w}_time"] = analysis.calculation_time

        # Other baseline metrics
        for fn, key in [
            (analysis.visilibity_index, 'VisibilityIndex'),
            (analysis.mpix, 'MPix'),
            (lambda: analysis.grid_based_density_overlap_degree(grid_pixel_ratio=1), 'GridOverlap'),
            (lambda: analysis.kernel_density_estimation(bandwidth=0.1, gridsize=100), 'KDEOverlap'),
            (analysis.nearest_neighbor_distance, 'NNDistance'),
            (analysis.pairwise_bounding_box_based_overlap_degree, 'PBBOverlap'),
            (analysis.distance_consistency, 'DistanceConsistency')
        ]:
            try:
                fn()
            except Exception as e:
                print(f"Warning: {key} failed: {e}")
            results[key] = analysis.result
            results[f"{key}_time"] = analysis.calculation_time

        # Coverage metrics
        try:
            num_diff_any, num_covered_by_diff_class, num_covered_pixels_anomaly = analysis.cal_covered_data_points()
        except Exception:
            num_diff_any, num_covered_by_diff_class, num_covered_pixels_anomaly = None, None, None
        try:
            num_diff_px_general_method = analysis.calculate_pixels_covered_by_different_categories()
        except Exception:
            num_diff_px_general_method = None

        results.update({
            "CoveredValsDiffClasses": num_covered_by_diff_class,
            "CoveredPixelsDiffClasses": num_covered_pixels_anomaly,
            'CoveredPixelsDiffClassesGeneralMethod': num_diff_px_general_method
        })

        # Build metadata-like dict and save to CSV (single-row per order)
        dataset_base = os.path.splitext(os.path.basename(file_location))[0]
        metadata = {
            "ScatterplotName": os.path.basename(scatter_fname),
            "DatasetName": dataset_base,
            "TotalPoints": len(data),
            "NumClasses": data['pred'].nunique() if 'pred' in data.columns else data['pred'].nunique(),
            "MarkerSize": analysis.marker_size,
            "RenderOrder": render_order,
            "DRmethod": dr_label,
            "XColumn": xvar,
            "YColumn": yvar
        }
        metadata.update(results)

        # collect metadata for this render order
        all_metadata.append(metadata)

# After processing all rendering orders, write a single CSV with all rows
out_folder = os.path.join('test_MNIST', 'output')
os.makedirs(out_folder, exist_ok=True)
out_csv = os.path.join(out_folder, f'metrics_{dataset_base}.csv')
import pandas as pd
pd.DataFrame(all_metadata).to_csv(out_csv, index=False)
print(f"Saved combined metrics CSV to {out_csv}")