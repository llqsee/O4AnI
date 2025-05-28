import pandas as pd
import numpy as np


# Simulating 100 data items for sample_data with a clear outlier
np.random.seed(42)  # For reproducibility

# Generate random data for Feature_1 and Feature_2
feature_1 = np.random.normal(10, 5, 100)  # mean=10, std=5
feature_2 = np.random.normal(15, 5, 100)  # mean=15, std=5

# Introduce clear outliers
feature_1[-1] = 100  # Last item as an outlier for Feature_1
feature_2[-1] = 300  # Last item as an outlier for Feature_2

# Create a pandas DataFrame
df_sample_simulated = pd.DataFrame({
    'Feature_1': feature_1,
    'Feature_2': feature_2
})