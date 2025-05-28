import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Current Working Directory:", os.getcwd())

from scipy.stats import gaussian_kde
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from utils.mapping_function import normalization, log_scale, sigmoid_scale, \
tanh_scale, normalize_and_power_scale, normalize_and_square_root_scale

from datasets.generateData import load_and_sample_data, load_data
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance, distance_matrix
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats.mstats import winsorize



importance_combine_method = 'density'
    

def normalize_coordinates(df, xvar, yvar, pixel_width, pixel_height):
    scaler_x = MinMaxScaler()
    df['normalized_x'] = scaler_x.fit_transform(df[[xvar]])
    df['normalized_y'] = scaler_x.fit_transform(df[[yvar]]) * (pixel_height / pixel_width)
    return df


def density(data, feature_1, feature_2):
    # Step 1: Determine the value of k
    n = len(data)
    if n < 5:
        k = n
    else:
        k = int(np.sqrt(n))  # Rule of thumb: k = sqrt(n)

    # Step 2: Use K-Nearest Neighbors to calculate density
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(data[[feature_1, feature_2]])
    distances, indices = neighbors.kneighbors(data[[feature_1, feature_2]])

    return distances[:, -1]



def balance_density_and_data_inter(density_value, data_inter, weight_density, weight_data_inter):
    # Normalize density_value and data_inter
    
    if importance_combine_method == 'distance':
        combine_value = data_inter
    elif importance_combine_method == 'density':
        combine_value = density_value
    elif importance_combine_method == 'distance_density':
        combine_value = density_value * data_inter
    
    # Combine normalized values with weights
    # combine_value = (weight_density * density_value_normalized) + (weight_data_inter * data_inter_normalized)
    # combine_value = density_value * data_inter
    # combine_value = density_value_normalized * data_inter_normalized
    
    # lower_limit = combine_value.quantile(0.001)
    # upper_limit = combine_value.quantile(0.999)
    # combine_value = np.clip(combine_value, lower_limit, upper_limit)
    
    # combine_value = winsorize(combine_value, limits=[0.001, 0.001])

    

    
    # combine_value = np.log(combine_value + 1)
    
    # combine_final_value = (combine_value - np.min(combine_value))/(np.max(combine_value)-np.min(combine_value))
    
    # return combine_value
    # return data_inter_normalized
    return data_inter
    # return density_value_normalized
    
    

def density_kde(data, feature_1, feature_2):
    epsilon=1e-6
    # Extract the relevant features
    values = data[[feature_1, feature_2]].values.T
    
    # Initialize and fit the KDE model
    kde = gaussian_kde(values)
    
    # Evaluate the density model on the data
    density_values = kde(values)
    
    return 1 / (density_values + epsilon)

def density_kde_color(data_all, data, feature_1, feature_2):
    epsilon = 1e-6
    
    # Extract the relevant features from all data for KDE fitting
    values_all = data_all[[feature_1, feature_2]].values.T
    
    # Initialize and fit the KDE model using all data
    kde_all = gaussian_kde(values_all)
    
    # Extract the relevant features from the specific group data
    values_group = data[[feature_1, feature_2]].values.T
    
    # Evaluate the fitted KDE model on the specific group data
    density_values_group = kde_all(values_group)
    
    # Return the inverse of the density values for the group, adding a small constant to avoid division by zero
    return 1 / (density_values_group + epsilon)



def lof_distance(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """This function is to calculate the LOF score and then add an attribute to original 
    data named importance_index
    
    Args:
        color_variable (_categorical_): a categorical variable
        data (_pd_): a pandas data format
        xvariable (_array_): it is an array that contains 1 or more than 1 variables, which is x-axis in the scatterplot
        yvariable (_array_): it is an array that contains 1 or more than 1 variables, which is y-axis in the scatterplot
        n_neighbors (int): number of neighbors to use for calculating LOF
        mapping_method (str): method to map importance_index values
    """
    n_neighbors=20
    
    def lof_importance(df_sample, n_neighbors=20):
        # Initialize the Local Outlier Factor model
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric='euclidean', contamination='auto')

        # Fit the model and calculate the LOF scores
        df_sample['LOF_Score'] = lof.fit_predict(df_sample)
        df_sample['LOF_Score'] = -lof.negative_outlier_factor_

        return df_sample['LOF_Score']
    
    if color_variable is None:
        lof_scores, normalized_df = lof_importance(data[[xvariable, yvariable]], n_neighbors)
        density_value = density(normalized_df, xvariable, yvariable)
        data.loc[:,'density'] = density_value
        data.loc[:,'distance'] = lof_scores
        # data.loc[:, 'importance_index'] = balance_density_and_data_inter(density_value, lof_scores)
    else:
        dff = pd.DataFrame()
        normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
        distance_list = []
        density_list = []
        for category in data[color_variable].unique():
            df = normalized_df[normalized_df[color_variable] == category]
            n_samples = len(df)
            if n_samples <= 1:
                continue
            else:
                effective_n_neighbors = min(n_neighbors, n_samples - 1)
                lof_scores = lof_importance(df[['normalized_x', 'normalized_y']], effective_n_neighbors)
                density_value = density(df, 'normalized_x', 'normalized_y')
                
                # Align with original data indices
                distance_list.append(pd.Series(lof_scores.values, index=df.index))
                density_list.append(pd.Series(density_value, index=df.index))
                
        # Combine results
        data['distance'] = pd.concat(distance_list).sort_index()
        data['density'] = pd.concat(density_list).sort_index()
        

    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)
    
    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass
    
    return data




def m_distance(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """This function is to calculate the m distance and then add an attribute to original 
    data named importance_index
    
    Args:
        color_variable (_categorical_): a categorical variable
        data (_pd_): a pandas data format
        xvariable (_array_): it is an array that contains 1 or more than 1 variables, which is x-axis in the scatterplot
        yvariable (_array_): it is an array that contains 1 or more than 1 variables, which is y-axis in the scatterplot
    """
    
    def mahalanobis_importance(df_sample):
        # Normalize x to [0, 1] and y to [0, pixel_height/pixel_width]
        
    
        # Calculate the mean and inverse of the covariance matrix
        mean_vector = df_sample.mean().values
        covariance_matrix = df_sample.cov().values
        # Add a small value to the diagonal elements of the covariance matrix to avoid singularity
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6
        covariance_matrix_inv = np.linalg.inv(covariance_matrix)

        # Calculate the Mahalanobis distance for each data point
        df_sample['Mahalanobis_Distance'] = df_sample.apply(
            lambda row: mahalanobis(row.values, mean_vector, covariance_matrix_inv), axis=1)

        return df_sample['Mahalanobis_Distance']
    
    if color_variable is None:
        distances, normalized_df = mahalanobis_importance(data[[xvariable, yvariable]])
        density_value = density(normalized_df, xvariable, yvariable)
        data.loc[:,'density'] = density_value
        data.loc[:,'distance'] = distances
    else:
        dff = pd.DataFrame()
        normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
        density_list = []
        distance_list = []
        for category in normalized_df[color_variable].unique():
            df = normalized_df[normalized_df[color_variable] == category]
            distances = mahalanobis_importance(df[['normalized_x', 'normalized_y']])
            density_value = density(df, 'normalized_x', 'normalized_y')
            
            # Align with original data indices
            distance_list.append(pd.Series(distances.values, index=df.index))
            density_list.append(pd.Series(density_value, index=df.index))

        # Combine results
        data['distance'] = pd.concat(distance_list).sort_index()
        data['density'] = pd.concat(density_list).sort_index()
    
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)
    
    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass
    
    return data


# Isolation Forest outlier detection
def isolation_forest(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """
    Applies the Isolation Forest algorithm to detect anomalies in a pandas DataFrame.
    
    Parameters:
    - color_variable: str or None, the name of the column containing categorical data.
    - data: pandas DataFrame, the dataset to analyze.
    - xvariable: list of str, the x-axis variables.
    - yvariable: list of str, the y-axis variables.
    - pixel_width: int, the width of the plot in pixels.
    - pixel_height: int, the height of the plot in pixels.
    - mapping_method: str, method to map importance_index values.
    
    Returns:
    - data: pandas DataFrame containing the original data with an additional 'importance_index' column.
    """
    
    # Initialize the Isolation Forest model
    clf = IsolationForest(random_state=42)
    
    if color_variable is None:
        df = data[[xvariable, yvariable]]
        
        # Normalize coordinates
        normalized_df = normalize_coordinates(df.copy(), xvariable, yvariable, pixel_width, pixel_height)
        
        clf.fit(normalized_df[['normalized_x', 'normalized_y']])
        
        # Get anomaly scores
        data_inter = -clf.decision_function(normalized_df[['normalized_x', 'normalized_y']])
        data_inter = data_inter + data_inter.min() + 1e-6
        
        density_value = density(normalized_df, 'normalized_x', 'normalized_y')
        data.loc[:,'density'] = density_value
        data.loc[:,'distance'] = data_inter
    else:
        dff = pd.DataFrame()
        normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
        distance_list = []
        density_list = []
        for category in normalized_df[color_variable].unique():
            df = normalized_df[normalized_df[color_variable] == category]

            # Fit the model
            clf.fit(df[['normalized_x', 'normalized_y']])
            
            # Get anomaly scores
            data_inter = -clf.decision_function(df[['normalized_x', 'normalized_y']])
            data_inter = data_inter + data_inter.min() + 1e-6
            
            density_value = density(df, 'normalized_x', 'normalized_y')
            
            # Align with original data indices
            distance_list.append(pd.Series(data_inter, index=df.index))
            density_list.append(pd.Series(density_value, index=df.index))
            
        # Combine results
        data['distance'] = pd.concat(distance_list).sort_index()
        data['density'] = pd.concat(density_list).sort_index()

        
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)

    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass
    
    return data



def cook_distance(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """
    Calculates Cook's Distance for detecting influential data points in a pandas DataFrame and adds an 'importance_index' attribute.

    Parameters:
    - color_variable: str or None, the name of the column containing categorical data.
    - data: pandas DataFrame, the dataset to analyze.
    - xvariable: list of str, the x-axis variables.
    - yvariable: list of str, the y-axis variables.
    - pixel_width: int, the width of the plot in pixels.
    - pixel_height: int, the height of the plot in pixels.
    - mapping_method: str, method to map importance_index values.

    Returns:
    - data: pandas DataFrame containing the original data with an additional 'importance_index' column.
    """

    if color_variable is None:
        df = data[[xvariable, yvariable]]
        
        # Normalize coordinates
        normalized_df = normalize_coordinates(df.copy(), xvariable, yvariable, pixel_width, pixel_height)
    
        # Add a constant to X for the intercept
        X = sm.add_constant(data[xvariable])
        
        # Fit a linear regression model
        model = sm.OLS(data[yvariable], X).fit()
        
        # Calculate Cook's Distance
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]

        # Update importance_index values by considering the density values
        density_value = density(normalized_df, xvariable, yvariable)
        
        data.loc[:,'density'] = density_value
        data.loc[:,'distance'] = cooks_d
        # data.loc[:, 'importance_index'] = balance_density_and_data_inter(density_value, cooks_d)
        
    else:
        normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
        distance_list = []
        density_list = []
        for category in normalized_df[color_variable].unique():
            df_sub = normalized_df[normalized_df[color_variable] == category]
            
            
            X = sm.add_constant(df_sub['normalized_x'])
            
            model = sm.OLS(df_sub['normalized_y'], X).fit()
            
            influence = model.get_influence()
            cooks_d = influence.cooks_distance[0]
            
            # Update importance_index values by considering the density values
            density_value = density(df_sub, 'normalized_x', 'normalized_y')
            
            # Align with original data indices
            distance_list.append(pd.Series(cooks_d, index=df_sub.index))
            density_list.append(pd.Series(density_value, index=df_sub.index))
            
        # Combine results
        data['distance'] = pd.concat(distance_list).sort_index()
        data['density'] = pd.concat(density_list).sort_index()
    

    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)
    
    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass
    
    return data




def leverage_score(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """
    Calculates leverage scores for detecting influential data points in a pandas DataFrame and adds an 'importance_index' attribute.

    Parameters:
    - color_variable: str or None, the name of the column containing categorical data.
    - data: pandas DataFrame, the dataset to analyze.
    - xvariable: list of str, the x-axis variables.
    - yvariable: list of str, the y-axis variables.
    - pixel_width: int, the width of the plot in pixels.
    - pixel_height: int, the height of the plot in pixels.
    - mapping_method: str, method to map importance_index values.

    Returns:
    - data: pandas DataFrame containing the original data with an additional 'importance_index' column.
    """

    if color_variable is None:
        df = data[[xvariable, yvariable]]
        
        # Normalize coordinates
        normalized_df = normalize_coordinates(df.copy(), xvariable, yvariable, pixel_width, pixel_height)
    
        # Add a constant to X for the intercept
        X = sm.add_constant(normalized_df['normalized_x'])
        
        # Fit a linear regression model
        model = sm.OLS(normalized_df['normalized_y'], X).fit()
        
        # Calculate leverage scores
        influence = model.get_influence()
        leverage_scores = influence.hat_matrix_diag

        # Calculate density
        density_value = density(normalized_df, 'normalized_x', 'normalized_y')
        
        data.loc[:,'density'] = density_value
        data.loc[:,'distance'] = leverage_scores
        
        # Calculate importance index
        # data.loc[:, 'importance_index'] = balance_density_and_data_inter(density_value, leverage_scores)
        
    else:
        dff = pd.DataFrame()
        normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
        distance_list = []
        density_list = []
        for category in normalized_df[color_variable].unique():
            df = normalized_df[normalized_df[color_variable] == category]
            
            X = sm.add_constant(df['normalized_x'])
            
            model = sm.OLS(df['normalized_y'], X).fit()
            
            influence = model.get_influence()
            leverage_scores = influence.hat_matrix_diag
            
            # Calculate density
            density_value = density(df, 'normalized_x', 'normalized_y')
            
            
            distance_list.append(pd.Series(leverage_scores, index=df.index))
            density_list.append(pd.Series(density_value, index=df.index))
            
        # Combine results
        data['distance'] = pd.concat(distance_list).sort_index()
        data['density'] = pd.concat(density_list).sort_index()

    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)    

    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass
    
    return data




def influence_function(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """
    This function calculates the influence function (Cook's Distance) and then adds an attribute to original 
    data named importance_index
    
    Args:
        color_variable (_categorical_): a categorical variable
        data (_pd_): a pandas data format
        xvariable (_array_): it is an array that contains 1 or more than 1 variables, which is x-axis in the scatterplot
        yvariable (_array_): it is an array that contains 1 or more than 1 variables, which is y-axis in the scatterplot
        mapping_method (str): method to map importance_index values
    """
    
    def cooks_distance(X, y):
        """Calculate Cook's distance for each data point"""
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = (residuals ** 2).sum() / (X.shape[0] - X.shape[1])
        
        # Leverage values
        H = X @ np.linalg.pinv(X.T @ X) @ X.T
        h_ii = np.diag(H)
        
        # Cook's Distance
        cooks_d = (residuals ** 2 / (X.shape[1] * mse)) * (h_ii / (1 - h_ii) ** 2)
        return cooks_d
    
    if color_variable is None:
        df = data[[xvariable, yvariable]]
        
        # Normalize coordinates
        normalized_df = normalize_coordinates(df.copy(), xvariable, yvariable, pixel_width, pixel_height)
        
        X = sm.add_constant(normalized_df['normalized_x']).values
        y = normalized_df['normalized_y'].values
        d = cooks_distance(X, y)
        
        density_value = density(normalized_df, 'normalized_x', 'normalized_y')
        
        data.loc[:,'density'] = density_value
        data.loc[:,'distance'] = d
        
        # data.loc[:, 'importance_index'] = balance_density_and_data_inter(density_value, d)
    else:
        dff = pd.DataFrame()
        normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
        distance_list = []
        density_list = []
        for category in normalized_df[color_variable].unique():
            df = normalized_df[normalized_df[color_variable] == category]
            
            
            X = sm.add_constant(df['normalized_x']).values
            y = df['normalized_y'].values
            data_inter = cooks_distance(X, y)
            
            density_value = density(df, 'normalized_x', 'normalized_y')
            
            # Align with original data indices
            distance_list.append(pd.Series(data_inter, index=df.index))
            density_list.append(pd.Series(density_value, index=df.index))
            
        # Combine results
        data['distance'] = pd.concat(distance_list).sort_index()
        data['density'] = pd.concat(density_list).sort_index()
        
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)

    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass
    
    return data



def centroid_method(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    
    def calculate_centroid(df, xvariable, yvariable):
        """Calculate the centroid of a cluster."""
        centroid_x = df[xvariable].mean()
        centroid_y = df[yvariable].mean()
        return (centroid_x, centroid_y)

    data['importance_index'] = np.nan
    
    normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)

    subsets = [normalized_df] if color_variable is None or color_variable not in normalized_df.columns else [normalized_df[normalized_df[color_variable] == category] for category in normalized_df[color_variable].unique()]

    for subset in subsets:
        centroid = calculate_centroid(subset, 'normalized_x', 'normalized_y')
        distance_importance = subset.apply(lambda row: distance.euclidean((row['normalized_x'], row['normalized_y']), centroid), axis=1)
        density_importance = density(subset, 'normalized_x', 'normalized_y')

        if color_variable and color_variable in data.columns:
            data.loc[subset.index, 'density'] = density_importance
            data.loc[subset.index,'distance'] = distance_importance
            # data.loc[subset.index, 'importance_index'] = balance_density_and_data_inter(density_importance, distance_importance)
        else:
            data.loc[:, 'density'] = density_importance
            data.loc[:, 'distance'] = distance_importance
            # data['importance_index'] = balance_density_and_data_inter(density_importance, distance_importance)
            
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)

    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass

    return data


def single_linkage_method(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    
    def calculate_min_distances(df, xvariable, yvariable):
        """Calculate the minimum distance from each point to its nearest neighbor in the same cluster."""
        if len(df) > 1:  # Ensure there are at least 2 points for distance calculation
            dist_matrix = distance_matrix(df[[xvariable, yvariable]].values, df[[xvariable, yvariable]].values)
            np.fill_diagonal(dist_matrix, np.inf)  # Fill diagonal with inf to ignore distance to self
            min_distances = np.min(dist_matrix, axis=1)  # Get the minimum distance for each point
        else:
            min_distances = [0]  # If only one point in the cluster, distance is 0 (or could be set to None or np.inf based on interpretation)
        return min_distances

    data['importance_index'] = np.nan
    
    normalized_df = normalize_coordinates(data[[xvariable, yvariable]].copy(), xvariable, yvariable, pixel_width, pixel_height)

    subsets = [normalized_df] if color_variable is None or color_variable not in normalized_df.columns else [normalized_df[normalized_df[color_variable] == category] for category in normalized_df[color_variable].unique()]

    for subset in subsets:
        distance_importance = calculate_min_distances(subset, 'normalized_x', 'normalized_y')
        density_importance = density(subset, 'normalized_x', 'normalized_y')

        if color_variable and color_variable in data.columns:
            data.loc[subset.index, 'density'] = density_importance
            data.loc[subset.index, 'distance'] = distance_importance
        else:
            data.loc[:, 'density'] = density_importance
            data.loc[:, 'distance'] = distance_importance
            
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)

    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:    
        pass

    return data


def complete_linkage_method(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    
    def calculate_max_distances(df, xvariable, yvariable):
        """Calculate the maximum distance from each point to its furthest neighbor in the same cluster."""
        if len(df) > 1:  # Ensure there are at least 2 points for distance calculation
            dist_matrix = distance_matrix(df[[xvariable, yvariable]].values, df[[xvariable, yvariable]].values)
            np.fill_diagonal(dist_matrix, 0)  # Fill diagonal with 0 to ignore distance to self
            max_distances = np.max(dist_matrix, axis=1)  # Get the maximum distance for each point
        else:
            max_distances = [0]  # If only one point in the cluster, distance is 0 (or could be set to None or np.inf based on interpretation)
        return max_distances

    data['importance_index'] = np.nan
    
    normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)

    subsets = [normalized_df] if color_variable is None or color_variable not in normalized_df.columns else [normalized_df[normalized_df[color_variable] == category] for category in normalized_df[color_variable].unique()]

    for subset in subsets:
        distance_importance = calculate_max_distances(subset, 'normalized_x', 'normalized_y')
        density_importance = density(subset, 'normalized_x', 'normalized_y')

        if color_variable and color_variable in data.columns:
            data.loc[subset.index, 'density'] = density_importance
            data.loc[subset.index,'distance'] = distance_importance
        else:
            data.loc[:, 'density'] = density_importance
            data.loc[:, 'distance'] = distance_importance
            
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)

    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass

    return data



def average_linkage_method(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    
    def calculate_average_distances(df, xvariable, yvariable):
        """Calculate the average distance from each point to all other points in the same cluster."""
        if len(df) > 1:  # Ensure there are at least 2 points for distance calculation
            dist_matrix = distance_matrix(df[[xvariable, yvariable]].values, df[[xvariable, yvariable]].values)
            np.fill_diagonal(dist_matrix, np.nan)  # Fill diagonal with NaN to ignore distance to self
            average_distances = np.nanmean(dist_matrix, axis=1)  # Get the average distance for each point
        else:
            average_distances = [0]  # If only one point in the cluster, set average distance to 0 (or another value based on interpretation)
        return average_distances

    data['importance_index'] = np.nan
    
    normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)

    subsets = [normalized_df] if color_variable is None or color_variable not in normalized_df.columns else [normalized_df[normalized_df[color_variable] == category] for category in normalized_df[color_variable].unique()]

    for subset in subsets:
        distance_importance = calculate_average_distances(subset, 'normalized_x', 'normalized_y')
        density_importance = density(subset, 'normalized_x', 'normalized_y')

        if color_variable and color_variable in data.columns:
            data.loc[subset.index, 'density'] = density_importance
            data.loc[subset.index, 'distance'] = distance_importance
        else:
            data.loc[:, 'density'] = density_importance
            data.loc[:, 'distance'] = distance_importance
            
    data.loc[:, 'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)

    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:    
        pass

    return data

def vertical_distance_to_lowess_line(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """
    Calculate the vertical distance from each point in `data` to the LOWESS smoothed line.
    
    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - xvariable (str): The name of the column in `data` that serves as the predictor variable.
    - yvariable (str): The name of the column in `data` that serves as the response variable.
    - mapping_method (str): The method used for mapping the importance index.
    
    Returns:
    - pd.DataFrame: The original DataFrame with an additional column 'importance_index' 
                    indicating the vertical distance to the LOWESS line.
    """
    
    def preprocess_lowess_curve(lowess_result):
        # Ensure unique x values by averaging y values where x values duplicate
        unique_x, index = np.unique(lowess_result[:, 0], return_index=True)
        averaged_y = np.array([lowess_result[lowess_result[:, 0] == x, 1].mean() for x in unique_x])
        return unique_x, averaged_y
    
    # Initialize the column to store the importance index
    data['importance_index'] = np.nan

    # Normalize coordinates
    normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
    
    if color_variable is None or color_variable not in normalized_df.columns:
        categories = [None]
    else:
        categories = normalized_df[color_variable].unique()
        
    for category in categories:
        if category is None:
            df = normalized_df
        else:
            df = normalized_df[normalized_df[color_variable] == category]
        
        # Generate LOWESS smoothed line for the category or entire data
        lowess = sm.nonparametric.lowess
        z = lowess(df['normalized_y'], df['normalized_x'], frac=1./5.)
        x_lowess, y_lowess = preprocess_lowess_curve(z)

        # Interpolate the LOWESS line
        lowess_interpolated = interp1d(x_lowess, y_lowess, bounds_error=False, fill_value='extrapolate')

        # Calculate the y-values on the LOWESS line corresponding to the x-values of the data points
        y_lowess = lowess_interpolated(df['normalized_x'])
        
        # Calculate the vertical distances from the data points to the LOWESS line
        vertical_distances = np.abs(df['normalized_y'] - y_lowess)
        density_importance = density(df, 'normalized_x', 'normalized_y')
        
        if category is None:
            data.loc[:, 'density'] = density_importance
            data.loc[:,'distance'] = vertical_distances
        else:
            data.loc[df.index,'density'] = density_importance
            data.loc[df.index, 'distance'] = vertical_distances
            
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)
    
    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass

    return data


def orthogonal_distance_to_lowess_line(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """
    Calculate the orthogonal distance from each point in `data` to the LOWESS smoothed line.
    
    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - xvariable (str): The name of the column in `data` that serves as the predictor variable.
    - yvariable (str): The name of the column in `data` that serves as the response variable.
    - mapping_method (str): The method used for mapping the importance index.
    
    Returns:
    - pd.DataFrame: The original DataFrame with an additional column 'importance_index' 
                    indicating the orthogonal distance to the LOWESS line.
    """
    
    def preprocess_lowess_curve(lowess_result):
        # Ensure unique x values by averaging y values where x values duplicate
        unique_x, index = np.unique(lowess_result[:, 0], return_index=True)
        averaged_y = np.array([lowess_result[lowess_result[:, 0] == x, 1].mean() for x in unique_x])
        return unique_x, averaged_y
    
    def find_orthogonal_distance(point, lowess_curve):
        """
        Find the orthogonal distance from a point to the LOWESS curve.
        """
        x_lowess, y_lowess = preprocess_lowess_curve(lowess_curve)
        lowess_interpolated = interp1d(x_lowess, y_lowess, bounds_error=False, fill_value="extrapolate")

        def distance_to_curve(x):
            y = lowess_interpolated(x)
            return np.sqrt((x - point[0])**2 + (y - point[1])**2)
        
        x_initial = point[0]
        result = minimize(distance_to_curve, x0=x_initial, method='L-BFGS-B', bounds=[(x_lowess.min(), x_lowess.max())])
        
        return result.fun

    # Initialize the column to store the importance index
    data['importance_index'] = np.nan

    # Normalize coordinates
    normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
    
    if color_variable is None or color_variable not in normalized_df.columns:
        categories = [None]
    else:
        categories = normalized_df[color_variable].unique()
        
    for category in categories:
        if category is None:
            df = normalized_df
        else:
            df = normalized_df[normalized_df[color_variable] == category]
        
        # Generate LOWESS smoothed line for the category or entire data
        lowess = sm.nonparametric.lowess
        z = lowess(df['normalized_y'], df['normalized_x'], frac=1./5.)
        
        # Calculate the orthogonal distances from the data points to the LOWESS line
        orthogonal_distances = np.array([find_orthogonal_distance((x, y), z) for x, y in zip(df['normalized_x'], df['normalized_y'])])
        density_importance = density(df, 'normalized_x', 'normalized_y')
        
        if category is None:
            data.loc[:, 'density'] = density_importance
            data.loc[:,'distance'] = orthogonal_distances
        else:
            data.loc[df.index,'density'] = density_importance
            data.loc[df.index, 'distance'] = orthogonal_distances
            
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)
    
    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass

    return data



def horizontal_distance_to_lowess_line(color_variable, data, xvariable, yvariable, pixel_width, pixel_height, weight_density, weight_data_inter, mapping_method=None):
    """
    Calculate the horizontal distance from each point in `data` to the LOWESS smoothed line.
    
    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - xvariable (str): The name of the column in `data` that serves as the predictor variable.
    - yvariable (str): The name of the column in `data` that serves as the response variable.
    - mapping_method (str): The method used for mapping the importance index.
    
    Returns:
    - pd.DataFrame: The original DataFrame with an additional column 'importance_index' 
                    indicating the horizontal distance to the LOWESS line.
    """
    
    def preprocess_lowess_curve(lowess_result):
        # Ensure unique x values by averaging y values where x values duplicate
        unique_x, index = np.unique(lowess_result[:, 0], return_index=True)
        averaged_y = np.array([lowess_result[lowess_result[:, 0] == x, 1].mean() for x in unique_x])
        return unique_x, averaged_y
    
    def find_closest_x_on_curve(y_value, lowess_curve):
        """
        Find the x-coordinate on the LOWESS curve that is closest to the given y_value.
        """
        x_lowess, y_lowess = preprocess_lowess_curve(lowess_curve)
        lowess_interpolated = interp1d(y_lowess, x_lowess, bounds_error=False, fill_value="extrapolate")
        return lowess_interpolated(y_value)

    # Initialize the column to store the importance index
    data['importance_index'] = np.nan

    # Normalize coordinates
    normalized_df = normalize_coordinates(data.copy(), xvariable, yvariable, pixel_width, pixel_height)
    
    if color_variable is None or color_variable not in normalized_df.columns:
        categories = [None]
    else:
        categories = normalized_df[color_variable].unique()
        
    for category in categories:
        if category is None:
            df = normalized_df
        else:
            df = normalized_df[normalized_df[color_variable] == category]
        
        # Generate LOWESS smoothed line for the category or entire data
        lowess = sm.nonparametric.lowess
        z = lowess(df['normalized_y'], df['normalized_x'], frac=1./5.)

        # Calculate the horizontal distances from the data points to the LOWESS line
        horizontal_distances = np.abs(df['normalized_x'] - find_closest_x_on_curve(df['normalized_y'], z))
        density_importance = density(df, 'normalized_x', 'normalized_y')
        
        if category is None:
            data.loc[:, 'density'] = density_importance
            data.loc[:,'distance'] = horizontal_distances
        else:
            data.loc[df.index,'density'] = density_importance
            data.loc[df.index, 'distance'] = horizontal_distances
            
    data.loc[:,'importance_index'] = balance_density_and_data_inter(data['density'], data['distance'], weight_density, weight_data_inter)
    
    # Mapping importance_index values based on the mapping method specified
    if mapping_method == 'linear':
        data['importance_index'] = normalization(data['importance_index'])
    elif mapping_method == 'log':
        data['importance_index'] = log_scale(data['importance_index'])
    elif mapping_method == 'sigmoid':
        data['importance_index'] = sigmoid_scale(data['importance_index'])
    elif mapping_method == 'tanh':
        data['importance_index'] = tanh_scale(data['importance_index'])
    elif mapping_method == 'normalize_and_power_scale':
        data['importance_index'] = normalize_and_power_scale(data['importance_index'], exponent=2)
    elif mapping_method == 'normalize_and_root_square':
        data['importance_index'] = normalize_and_square_root_scale(data['importance_index'])
    else:
        pass

    return data



if __name__ == "__main__":
    
    # =======================================================================
    # We evaluate which methods are better
    # datasets = ['positive_strong', 'negative_strong', 'weak']
    # methods = ['cook_distance', 'isolation_forest', 'mahalanobis_distance', 'leverage_score', 'squared_eucdilean_distance']
    # folder_location = 'datasets/simuliti/case_correlation/' 

    # # Create a subplot grid: rows are datasets, columns are methods
    # fig, axes = plt.subplots(nrows=len(datasets), ncols=len(methods), figsize=(15, 10))  # Adjust the figsize as needed
    
    # # Variable to hold the scatter plots for creating a single color bar
    # scatters = []
    
    # for i, dataset in enumerate(datasets):
    #     for j, method in enumerate(methods):
    #         file_location = folder_location + dataset + '.csv'
    #         data = load_data(file_location)  # Make sure load_data is properly defined
            
    #         # Call the appropriate function based on the method
    #         if method == 'cook_distance':
    #             df_sample_outliers = cook_distance(None, data, ['feature_1'], ['feature_2'], 'linear')
    #         elif method == 'isolation_forest':
    #             df_sample_outliers = isolation_forest(None, data, ['feature_1'], ['feature_2'], 'linear')
    #         elif method == 'mahalanobis_distance':
    #             df_sample_outliers = m_distance(None, data, ['feature_1'], ['feature_2'], 'linear')
    #         elif method == 'leverage_score':
    #             df_sample_outliers = leverage_score(None, data, ['feature_1'], ['feature_2'], 'linear')
    #         elif method == 'squared_eucdilean_distance':
    #             df_sample_outliers = euc_distance(None, data, ['feature_1'], ['feature_2'], 'linear')
            
    #         ax = axes[i, j]  # Get the current Axes instance on the grid
    #         sc = ax.scatter(df_sample_outliers['feature_1'], df_sample_outliers['feature_2'], c=df_sample_outliers['importance_index'], cmap='viridis')
    #         scatters.append(sc)
            
    #         # You may only want to show the colorbar on the last column
    #         if j == len(methods) - 1:
    #             cbar = fig.colorbar(sc, ax=ax)
    #             cbar.set_label('Significant Values', fontsize=15)
                
    #         if i == 0:  # Only set titles on the top row
    #             ax.set_title((method.replace('_', ' ') ).title(), fontsize = 18)
    #         if j == 0:  # Only set row labels on the first column
    #             ax.set_ylabel(dataset.replace('_', ' ').title(), fontsize = 18)
    #         if i < len(datasets) - 1:  # Remove x-axis labels on all but last row
    #             ax.set_xticklabels([])
    #         if j > 0:  # Remove y-axis labels on all but last column
    #             ax.set_yticklabels([])
    #         else:  # Only set x-axis labels on the last row
    #             # ax.set_xlabel('Feature 1')
    #             pass

    #         # Set tick label font size
    #         ax.tick_params(axis='both', labelsize=15)

    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.show()
    # ============================================================================================
    
    
    # ============================================================================================
    # We evaluate which mapping function is better
    # datasets = ['positive_strong', 'weak']
    # # methods = ['cook_distance', 'isolation_forest', 'mahalanobis_distance', 'leverage_score', 'squared_eucdilean_distance']
    # mapping_methods = ['linear', 'normalize_and_power_scale', 'normalize_and_root_square']
    # folder_location = 'datasets/simuliti/case_correlation/' 

    # # Create a subplot grid: rows are datasets, columns are methods
    # fig, axes = plt.subplots(nrows=len(datasets), ncols=len(mapping_methods), figsize=(15, 10))  # Adjust the figsize as needed
    
    # # Variable to hold the scatter plots for creating a single color bar
    # scatters = []
    
    # for i, dataset in enumerate(datasets):
    #     for j, mm in enumerate(mapping_methods):
    #         file_location = folder_location + dataset + '.csv'
    #         data = load_data(file_location)  # Make sure load_data is properly defined

    #         df_sample_outliers = cook_distance(None, data, ['feature_1'], ['feature_2'], mm)

    #         ax = axes[i, j]  # Get the current Axes instance on the grid
    #         sc = ax.scatter(df_sample_outliers['feature_1'], df_sample_outliers['feature_2'], c=df_sample_outliers['importance_index'], cmap='viridis')
    #         scatters.append(sc)
            
    #         # You may only want to show the colorbar on the last column
    #         if j == len(mapping_methods) - 1:
    #             cbar = fig.colorbar(sc, ax=ax)
    #             cbar.set_label('Significant Values', fontsize=15)
                
    #         if i == 0:  # Only set titles on the top row
    #             ax.set_title((mm.replace('_', ' ') ).title(), fontsize = 18)
    #         if j == 0:  # Only set row labels on the first column
    #             ax.set_ylabel(dataset.replace('_', ' ').title(), fontsize = 18)
    #         if i < len(datasets) - 1:  # Remove x-axis labels on all but last row
    #             ax.set_xticklabels([])
    #         if j > 0:  # Remove y-axis labels on all but last column
    #             ax.set_yticklabels([])
    #         else:  # Only set x-axis labels on the last row
    #             # ax.set_xlabel('Feature 1')
    #             pass

    #         # Set tick label font size
    #         ax.tick_params(axis='both', labelsize=15)

    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.show()
    # ===================================================================
    
    
    
    
    # ============================================================================================
    # We test a single method
    dataset = 'weak'
    folder_location = 'datasets/simuliti/case_correlation/' 
    # dataset = 'sin_data'
    # folder_location = 'datasets/simuliti/correlation_1/' 
    
    file_location = folder_location + dataset + '.csv'
    data = load_data(file_location)  # Make sure load_data is properly defined

    # Coorelation-based methods
    # df_sample_outliers = orthogonal_distance_to_lowess_line(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = orthogonal_distance_to_loess_line(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = vertical_distance_to_lowess_line(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = horizontal_distance_to_lowess_line(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = leverage_score(None, data, ['feature_1'], ['feature_2'], 'linear')
    df_sample_outliers = cook_distance(None, data, 'feature_1', 'feature_2', 'linear')
    # df_sample_outliers = influence_function(None, data, ['feature_1'], ['feature_2'], 'linear')
    
    # Cluster-based methods
    # df_sample_outliers = average_linkage_method(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = complete_linkage_method(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = single_linkage_method(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = centroid_method(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = m_distance(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = lof_distance(None, data, ['feature_1'], ['feature_2'], 'linear')
    # df_sample_outliers = isolation_forest(None, data, ['feature_1'], ['feature_2'], 'linear')
    
    # Plot the lowess line
    # Generate LOWESS smoothed line for the original data
    lowess = sm.nonparametric.lowess
    z = lowess(data['feature_2'], data['feature_1'], frac=1./4.)
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Separate the data into two subsets based on NaN values in importance_index
    with_index = df_sample_outliers.dropna(subset=['importance_index'])
    without_index = df_sample_outliers[pd.isnull(df_sample_outliers['importance_index'])]

    # Plot data points with a defined importance_index using a colormap
    plt.scatter(with_index['feature_1'], with_index['feature_2'], 
                c=with_index['importance_index'], cmap='viridis', label='With Index')

    # Plot data points where importance_index is NaN in black
    plt.scatter(without_index['feature_1'], without_index['feature_2'], 
                color='red', label='Without Index (NaN)')

    
    # plt.scatter(df_sample_outliers['feature_1'], df_sample_outliers['feature_2'], c=df_sample_outliers['importance_index'], cmap='viridis')
    plt.plot(z[:, 0], z[:, 1], color='red', label='LOWESS Line', linewidth=2)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()