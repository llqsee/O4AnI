# We put all the related parameters in this file so we can easily use them

importance_methods = ['mahalanobis_distance', 'average_linkage_method', 
                            'complete_linkage_method', 'single_linkage_method', 'centroid_method', 'isolation_forest',
                            'leverage_score', 'cook_distance', 
                            'orthogonal_distance_to_lowess_line', 'vertical_distance_to_lowess_line',
                            'horizontal_distance_to_lowess_line']



mapping_methods = [
    'linear', 
    'log', 
    'sigmoid', 
    'tanh', 
    'normalize_and_power_scale', 
    'normalize_and_root_square'
]


# MNIST dataset
mnist_pred_categories = ['digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4', 'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9']


# adult income dataset
adult_columns = ["Age", "Workclass", "Education-Num", "Marital Status", "Occupation", 
           "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"]

relationship = ['Wife', 'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried']



# wine quality dataset's parameters
wine_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
