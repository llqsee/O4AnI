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

import numpy as np

importance_index_methods = ['mahalanobis_distance', 'average_linkage_method', 
                            'complete_linkage_method', 'single_linkage_method', 'centroid_method', 'lof_distance','isolation_forest',
                            'leverage_score', 'cook_distance',
                            'orthogonal_distance_to_lowess_line', 'vertical_distance_to_lowess_line',
                            'horizontal_distance_to_lowess_line', 'lof_distance']

columns = [
    "Age",                 # Numerical
    "Workclass",           # Categorical: ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
    "Education-Num",       # Numerical
    "Marital Status",      # Categorical: ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    "Occupation",          # Categorical: ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
    "Relationship",        # Categorical: ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    "Race",                # Categorical: ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    "Sex",                 # Categorical: ['Female', 'Male']
    "Capital Gain",        # Numerical
    "Capital Loss",        # Numerical
    "Hours per week",      # Numerical
    "Country",             # Categorical: ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    "predicted",           # Numerical (Prediction probability)
    "predicted_class"      # Categorical: [True, False] (Based on threshold 0.5)
]

file_location = 'datasets/adult_income/adult_xgboost_v3.csv'
data = load_data(file_location)  # Make sure load_data is properly defined

# ['Wife', 'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried']
# data_abstract = data[data['Relationship'].isin(['Not-in-family'])]
data_abstract = data[data['Relationship'].isin(['Own-child', 'Husband'])]


analysis = Scatter_Metric(data_abstract, 
                          margins = {'left':0.2, 'right': 0.7, 'top':0.8, 'bottom': 0.2},
                        marker = 'square', 
                        marker_size = 50, 
                        dpi = 100, 
                        figsize= (10, 6),
                        xvariable = 'Age', 
                        yvariable = 'Age_shap',
                        zvariable='Relationship',
                        color_map='tab10'
                        )

# analysis.importance_metric(important_cal_method = 'cook_distance', weight_diff_class=20, weight_same_class=1, order_variable='importance_index', asending=False)


# analysis.result

max_value = np.nanmax(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))
min_value = np.nanmin(np.nan_to_num(analysis.other_layer_matrix, nan=np.nan))

if max_value != min_value:  # Avoid division by zero
    normalized_matrix = np.where(np.isnan(analysis.other_layer_matrix), np.nan, (analysis.other_layer_matrix - min_value) / (max_value - min_value))
else:
    normalized_matrix = np.zeros_like(analysis.other_layer_matrix)

analysis.visualize_heat_map(analysis.overall_layer_matrix)

analysis.save_figure(filename = 'test_adult_income/output_adult_income_scatterplot.png')
analysis.save_heatmap(filename = 'test_adult_income/output_adult_income_heatmap.png')

analysis.bar_for_importance(analysis.data)
analysis.save_importance_bar(filename = 'test_adult_income/output_adult_income_importance_bar_distribution.png')


# =============================================================================



# data_1 = analysis.data.loc[data['Relationship'] == 'Husband'].copy()
# analysis.visualize_importance(data_1)
# analysis.save_importance_figure(filename = 'test_adult_income/output_emotion_importance_scatterplots_Husband.png')

# data_2 = analysis.data.loc[data['Relationship'] == 'Own-child'].copy()
# analysis.visualize_importance(data_2)
# analysis.save_importance_figure(filename = 'test_adult_income/output_emotion_importance_scatterplots_own-child.png')

# analysis.bar_for_importance(analysis.data)
# analysis.save_importance_bar(filename = 'test_adult_income/output_emotion_importance_bar_distribution.png')

# analysis.save_figure(filename = 'test_adult_income/output_adult_income_scatterplot.png')
# analysis.save_heatmap(filename = 'test_adult_income/output_adult_inceom_heatmap.png')

# analysis.plot_combined_heatmap_and_summaries()

# analysis.plot_depth_complexity_histogram()

# analysis.calculate_pixel_coverage()


