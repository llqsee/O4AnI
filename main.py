from datasets.generateData import load_data
from Our_metrics.Scatter_Metrics import Scatter_Metric

# Load sample data (automatically resolves relative to project root)
data = load_data('datasets/mnist/mnist_pred_updated_str.csv')

# Configure the analyzer
analysis = Scatter_Metric(
    data=data,
    marker = 'square', 
    marker_size = 25, 
    figsize= (12, 8),
    xvariable='X coordinate',
    yvariable='Y coordinate',
    zvariable='pred',
    dpi=100
)

# Compute quality scores
score = analysis.importance_metric(
    important_cal_method='mahalanobis_distance', 
    weight_diff_class=100, 
    weight_same_class=0, 
    order_variable='importance_index', 
    asending=True
)

# Save results (directories are created automatically)
analysis.save_figure('output/my_scatterplot.png')

analysis.visualize_heat_map(analysis.overall_layer_matrix)
analysis.save_heatmap('output/my_heatmap.png')