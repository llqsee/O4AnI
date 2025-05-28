# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import boxcox, yeojohnson
# from scipy.special import expit as sigmoid  # More stable for sigmoid function

# # Sample data
# np.random.seed(0)  # For reproducibility
# data = np.random.normal(loc=0, scale=1, size=1000)
# data = pd.Series(data)

# # Functions
# def normalization(series):
#     return (series - series.min()) / (series.max() - series.min())

# def log_scale(series):
#     min_val = series[series > 0].min()
#     max_val = series.max()
#     return (np.log(series.clip(lower=min_val)) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))

# def sigmoid_scale(series):
#     return sigmoid(series)

# def tanh_scale(series):
#     return 0.5 * (np.tanh(series) + 1)

# def quantile_normalize(series):
#     sorted_series = np.sort(series)
#     uniform_distribution = np.linspace(0, 1, len(series))
#     ranks = np.argsort(np.argsort(series))
#     return pd.Series(uniform_distribution[ranks], index=series.index)

# def boxcox_scale(series):
#     shifted_series = series + 1 - series.min()  # Ensure all values are positive for Box-Cox
#     scaled_data, _ = boxcox(shifted_series)
#     return pd.Series((scaled_data - np.min(scaled_data)) / (np.max(scaled_data) - np.min(scaled_data)), index=series.index)

# def yeojohnson_scale(series):
#     scaled_data, _ = yeojohnson(series)
#     return pd.Series((scaled_data - np.min(scaled_data)) / (np.max(scaled_data) - np.min(scaled_data)), index=series.index)

# # Apply transformations
# data_sorted = data.sort_values().reset_index(drop=True)  # Sort data for plotting
# normalized = normalization(data_sorted)
# log_scaled = log_scale(data_sorted)
# sigmoid_scaled = sigmoid_scale(data_sorted)
# tanh_scaled = tanh_scale(data_sorted)
# quantile_normalized = quantile_normalize(data_sorted)
# boxcox_scaled = boxcox_scale(data_sorted.clip(lower=data_sorted.min() + 1e-9))  # Adjust to ensure positive for Box-Cox
# yeojohnson_scaled = yeojohnson_scale(data_sorted)

# # Plotting
# plt.figure(figsize=(12, 8))
# plt.plot(data_sorted, normalized, label='Normalization (Min-Max)')
# plt.plot(data_sorted, log_scaled, label='Log Scale', linestyle='--')
# plt.plot(data_sorted, sigmoid_scaled, label='Sigmoid Scale', linestyle='-.')
# plt.plot(data_sorted, tanh_scaled, label='Tanh Scale', linestyle=':')
# plt.plot(data_sorted, quantile_normalized, label='Quantile Normalize', linestyle='-.')
# plt.plot(data_sorted, boxcox_scaled, label='Box-Cox Scale', linestyle='--')
# plt.plot(data_sorted, yeojohnson_scaled, label='Yeo-Johnson Scale', linestyle='-')

# plt.title('Comparison of Scaling and Transformation Methods')
# plt.xlabel('Original Sorted Data')
# plt.ylabel('Transformed Data')
# plt.legend()
# plt.grid(True)
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # Generate x values
# x = np.linspace(0, 1, 400)

# # Define the functions
# def sigmoid(x):
#     return 1 / (1 + np.exp(-10 * (x - 0.5)))

# def scaled_tanh(x):
#     return 0.5 * (np.tanh(10 * (x - 0.5)) + 1)

# def gompertz(x):
#     return np.exp(-np.exp(-10 * (x - 0.5)))

# # Calculate function values
# sigmoid_values = sigmoid(x)
# scaled_tanh_values = scaled_tanh(x)
# gompertz_values = gompertz(x)

# # Plot the functions
# plt.figure(figsize=(14, 8))

# plt.plot(x, sigmoid_values, label='Sigmoid Function', color='blue')
# plt.plot(x, scaled_tanh_values, label='Scaled Hyperbolic Tangent', color='green')
# plt.plot(x, gompertz_values, label='Gompertz Function', color='red')

# # Add title and labels
# plt.title('Comparison of S-shaped Functions')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()
# plt.grid(True)

# plt.show()





import numpy as np
import matplotlib.pyplot as plt
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
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    transformed_series = series ** exponent
    normalized_series = (transformed_series - transformed_series.min()) / (transformed_series.max() - transformed_series.min())
    return normalized_series

def normalize_and_square_root_scale(series):
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    normalized_series = (series - series.min()) / (series.max() - series.min())
    transformed_series = np.sqrt(normalized_series)
    return transformed_series

# Generate example data
x = np.linspace(0, 1, 100)
series = pd.Series(x)

# Apply the transformations
y_normalization = normalization(series)
y_log_scale = log_scale(series)
y_sigmoid_scale = sigmoid_scale(series)
y_tanh_scale = tanh_scale(series)
# y_boxcox_scale = boxcox_scale(series)
# y_yeojohnson_scale = yeojohnson_scale(series)
y_power_scale = normalize_and_power_scale(series)
y_square_root_scale = normalize_and_square_root_scale(series)

# Plot the transformationsimport numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import boxcox, yeojohnson
from scipy.special import expit as sigmoid  # More stable for sigmoid function

def normalization(series):
    """
    Normalizes the series to the range [0, 1].
    
    Parameters:
    series (pd.Series): Input series to be normalized.
    
    Returns:
    pd.Series: Normalized series.
    """
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def log_scale(series):
    """
    Applies log scaling to the series.
    
    Parameters:
    series (pd.Series): Input series to be scaled.
    
    Returns:
    pd.Series: Log scaled series.
    """
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    min_val = series[series > 0].min()
    max_val = series.max()
    scaled_series = (np.log(series.clip(lower=min_val)) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
    return scaled_series

def sigmoid_scale(series):
    """
    Applies sigmoid scaling to the series.
    
    Parameters:
    series (pd.Series): Input series to be scaled.
    
    Returns:
    pd.Series: Sigmoid scaled series.
    """
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    return sigmoid(10 * (series - 0.5))

def tanh_scale(series):
    """
    Applies hyperbolic tangent scaling to the series.
    
    Parameters:
    series (pd.Series): Input series to be scaled.
    
    Returns:
    pd.Series: Tanh scaled series.
    """
    if series.min() == series.max():
        return pd.Series(0, index=series.index)
    return 0.5 * (np.tanh(10 * (series - 0.5)) + 1)

def quantile_normalize(series):
    """
    Applies quantile normalization to the series.
    
    Parameters:
    series (pd.Series): Input series to be normalized.
    
    Returns:
    pd.Series: Quantile normalized series.
    """
    sorted_series = np.sort(series)
    uniform_distribution = np.linspace(0, 1, len(series))
    ranks = np.argsort(np.argsort(series))
    return pd.Series(uniform_distribution[ranks], index=series.index)

def plot_series(series, title):
    """
    Plots the given series.
    
    Parameters:
    series (pd.Series): Series to be plotted.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(series, marker='o', linestyle='-', markersize=5)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def plot_transformed_series(original_series, transformed_series, title):
    """
    Plots the original and transformed series for comparison.
    
    Parameters:
    original_series (pd.Series): Original series.
    transformed_series (pd.Series): Transformed series.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(original_series, marker='o', linestyle='-', markersize=5, label='Original')
    plt.plot(transformed_series, marker='x', linestyle='--', markersize=5, label='Transformed')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

plt.figure(figsize=(20, 8))

# Plotting different scales
plt.plot(x, y_normalization, label='Normalization', linewidth=4)
plt.plot(x, y_log_scale, label='Log Scale', linewidth=4)
plt.plot(x, y_sigmoid_scale, label='Sigmoid Scale', linewidth=4)
plt.plot(x, y_tanh_scale, label='Tanh Scale', linewidth=4)
# plt.plot(x, y_boxcox_scale, label='Box-Cox Scale')
# plt.plot(x, y_yeojohnson_scale, label='Yeo-Johnson Scale')
plt.plot(x, y_power_scale, label='Power Scale', linewidth=4)
plt.plot(x, y_square_root_scale, label='Square Root Scale', linewidth=4)

# Enhancing plot aesthetics
plt.title('Mapping Functions', fontsize=30, fontweight='bold')
plt.xlabel('Input Value', fontsize=30)
plt.ylabel('Mapped Value', fontsize=30)
plt.legend(fontsize=20)
# plt.grid(True)
plt.show()

# Clear the plot
plt.clf()
plt.close()