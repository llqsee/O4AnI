from datasets.generateData import load_data
from Our_metrics.Scatter_Metrics import Scatter_Metric

# Load sample data (automatically resolves relative to project root)
data = load_data('datasets/mnist/mnist_pred_updated_str.csv')


# compute the VQM based on category-based order method
analysis_category_based = Scatter_Metric(
    data=data,
    marker = 'square', 
    marker_size = 100, 
    figsize= (12, 8),
    xvariable='X coordinate',
    yvariable='Y coordinate',
    zvariable='pred',
    dpi=100
)
projected_labels = ['digit_2', 'digit_8', 'digit_5', 'digit_7', 'digit_3', 'digit_4', 'digit_1', 'digit_0', 'digit_6', 'digit_9']
# projected_labels = ['digit_7', 'digit_3', 'digit_4', 'digit_1', 'digit_0', 'digit_6', 'digit_9', 'digit_2', 'digit_8', 'digit_5']
analysis_category_based._sort_data(attribute = 'pred', order = projected_labels)
score_category_based = analysis_category_based.importance_metric(important_cal_method = 'average_linkage_method', weight_diff_class=100, weight_same_class=0)

# Save results (directories are created automatically)
analysis_category_based.save_figure('figures/my_scatterplot_category_based_order.png')

# analysis_category_based.visualize_heat_map(analysis_category_based.overall_layer_matrix)
# analysis_category_based.save_heatmap('figures/my_heatmap_category_based_order.png')

# compute the VQM based on om4ani order method
analysis_om4ani_order = Scatter_Metric(
    data=data,
    marker = 'square', 
    marker_size = 100, 
    figsize= (12, 8),
    xvariable='X coordinate',
    yvariable='Y coordinate',
    zvariable='pred',
    dpi=100
)
score_om4ani_order = analysis_om4ani_order.importance_metric(
    important_cal_method='average_linkage_method', 
    weight_diff_class=100, 
    weight_same_class=0, 
    order_variable='importance_index', 
    asending=True
)

# Save results (directories are created automatically)
analysis_om4ani_order.save_figure('figures/my_scatterplot_om4ani_order.png')

# analysis_om4ani_order.visualize_heat_map(analysis_om4ani_order.overall_layer_matrix)
# analysis_om4ani_order.save_heatmap('figures/my_heatmap_om4ani_order.png')