import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson
from scipy.special import expit as sigmoid  # More stable for sigmoid function

def normalization(series):
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def log_scale(series):
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    min_val = series[series > 0].min()
    max_val = series.max()
    scaled_series = (np.log(series.clip(lower=min_val)) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
    return scaled_series

# def sigmoid_scale(series):
#     return sigmoid(series)

def sigmoid_scale(series):
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    return sigmoid(10 * (series - 0.5))


def tanh_scale(series):
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    return 0.5 * (np.tanh(10 * (series - 0.5)) + 1)

def quantile_normalize(series):
    sorted_series = np.sort(series)
    uniform_distribution = np.linspace(0, 1, len(series))
    ranks = np.argsort(np.argsort(series))
    return pd.Series(uniform_distribution[ranks], index=series.index)

def boxcox_scale(series):
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    shifted_series = series + 1 - series.min()  # Shift series to ensure all values are positive
    scaled_data, _ = boxcox(shifted_series)
    return pd.Series((scaled_data - np.min(scaled_data)) / (np.max(scaled_data) - np.min(scaled_data)), index=series.index)

def yeojohnson_scale(series):
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    scaled_data, _ = yeojohnson(series)
    return pd.Series((scaled_data - np.min(scaled_data)) / (np.max(scaled_data) - np.min(scaled_data)), index=series.index)

def normalize_and_power_scale(series, exponent=2):
    """
    Normalizes a pandas Series and then applies a power scale transformation.
    
    Parameters:
    - series: pandas Series, the data to transform.
    - exponent: float, the exponent to use in the power transformation (default is 2).
    
    Returns:
    - Transformed data, with values first normalized and then power scaled.
    """
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    # Apply power scale transformation to the normalized data
    transformed_series = series ** exponent
    # Normalize the series
    normalized_series = (transformed_series - transformed_series.min()) / (transformed_series.max() - transformed_series.min())
    
    return normalized_series

def normalize_and_square_root_scale(series):
    """
    Normalizes a pandas Series and then applies a square root scale transformation.
    
    Parameters:
    - series: pandas Series, the data to transform.
    
    Returns:
    - Transformed data, with values first normalized and then square root scaled.
    """
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    # Normalize the series
    normalized_series = (series - series.min()) / (series.max() - series.min())
    
    # Apply square root scale transformation to the normalized data
    transformed_series = np.sqrt(normalized_series)
    
    return transformed_series
