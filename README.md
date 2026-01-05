# OM4AnI: Overlap Measure for Anomaly Identification

OM4AnI measures overplotting in multi-class scatterplots and highlight where important information is obscured. The core `Scatter_Metric` class builds pixel-level representations, computes quality metrics, produces scatterplots and heatmaps for highlighting where the anomalies are located.

## Workflow

Below is the computation pipeline of OM4AnI:

![OM4AnI workflow](figures/workflow.png)


## Highlights
- Pixel-accurate overplotting analysis for categorical or continuous classes.
- Multiple anomaly index computation methods (e.g., Mahalanobis distance, clustering, LOF, LOWESS-based distances).
- Configurable scatterplot/heatmap generation with legends, colorbars, and DPI-aware sizing.
- Works with provided datasets (MNIST, adult income, simulated) or your own tabular data.

## Requirements
- Python 3.9+ (tested with Anaconda)
- matplotlib, seaborn, scipy, numpy, pandas, scikit-learn

## Installation
```bash
git clone https://github.com/<your-org>/O4AnI_submission.git
cd O4AnI_submission
pip install -r requirements.txt  # or install the packages above manually
```

## Data
- Example CSVs live under [datasets/mnist](datasets/mnist), [datasets/adult_income](datasets/adult_income), and [datasets/simulated_datasets](datasets/simulated_datasets).
- Expected columns: `X coordinate`, `Y coordinate`, a class column such as `pred` or `label`, plus probability columns if you want richer importance calculations.

Example rows from [datasets/mnist/mnist_pred_updated_str.csv](datasets/mnist/mnist_pred_updated_str.csv):

| id | pred    | label   | eval | prob_7            | prob_9            | X coordinate | Y coordinate |
|----|---------|---------|------|-------------------|-------------------|--------------|--------------|
| 0  | digit_7 | digit_7 | True | 1.0               | 1.26573588e-21    | 13.364994    | -41.188786   |
| 1  | digit_9 | digit_9 | True | 9.79921998e-15    | 1.0               | -31.387823   | -60.370834   |
| 2  | digit_2 | digit_2 | True | 8.90068924e-26    | 7.58005401e-30    | 4.835193     | 65.88922     |
| 3  | digit_9 | digit_9 | True | 7.16228617e-13    | 1.0               | -47.098583   | -43.927925   |
| 4  | digit_1 | digit_1 | True | 2.86185333e-12    | 5.28650500e-15    | 48.73959     | -30.200827   |

## Quickstart
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.generateData import load_data
from Our_metrics.Scatter_Metrics import Scatter_Metric

# 1) Load sample data (MNIST embedding with predictions)
data = load_data('datasets/mnist/mnist_pred_updated_str.csv')

# 2) Configure the analyzer
metric = Scatter_Metric(
  data=data,
  margins={'left': 0.2, 'right': 0.7, 'top': 0.8, 'bottom': 0.2},
  marker='plus',
  marker_size=10,
  dpi=100,
  figsize=(12, 8),
  xvariable='X coordinate',
  yvariable='Y coordinate',
  zvariable='pred',
  color_map='tab10'
)

# 3) Compute importance-aware overplotting score
score = metric.importance_metric(
  important_cal_method='mahalanobis_distance',
  weight_diff_class=20,
  weight_same_class=1
)

# 4) Produce visuals
metric.plot_scatter_cal_matrix()
metric.visualize_pixel_matrix()
metric.save_figure('test_MNIST/scatterplot_mnist_pred_updated_str.png')
metric.save_heatmap('test_MNIST/heat_map_mnist_pred_updated_str.png')

print(f"OM4AnI score: {score:.2f}")
```

## Results
- Scatterplot: [test_MNIST/scatterplot_mnist_pred_updated_str.png](test_MNIST/scatterplot_mnist_pred_updated_str.png)
- Heatmap: [test_MNIST/heat_map_mnist_pred_updated_str.png](test_MNIST/heat_map_mnist_pred_updated_str.png)

The provided MNIST example yields an overplotting score of approximately 0.24 (higher implies more critical overlap).

## Key Methods (Scatter_Metric)
- `plot_scatter_cal_matrix`: render the scatterplot and compute per-pixel coverage.
- `visualize_pixel_matrix`: heatmap of pixel density/importance.
- `importance_metric`: compute overlap/importance score with configurable weighting.
- `save_figure` and `save_heatmap`: persist outputs for reports.

## Citation
If you build on OM4AnI, please cite the original paper:

> L. Liu, L. Bogachev, M. Rezaei, N. Ravikumar, A. Khara, and M. Azarmi, "OM4AnI: A Novel Overlap Measure for Anomaly Identification in Multi-Class Scatterplots," *IEEE Transactions on Visualization and Computer Graphics*, Early Access, 2025. [https://doi.org/10.1109/TVCG.2025.3642219](https://doi.org/10.1109/TVCG.2025.3642219)

The article is available via the IEEE Computer Society Digital Library: https://www.computer.org/csdl/journal/tg/5555/01/11295941/2ckEozKDT7G.