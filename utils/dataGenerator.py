# Author: Liqun Liu
# Created: 27.01.2025
# Description: A Python class to generate various types of clusters (data points) with transformations like translation and rotation.

"""
ClusterGenerator class to generate various types of clusters (data points) with transformations like translation and rotation.

Attributes:
    entire_data_number (int): Number of data points to generate.
    main_class_name (str): Name of the main class.
    test_data_numbers (list): Number of data points to change the group, same length as test_class_names.
    class_names (list): Names of the test classes.

Methods:
    apply_transformations(data, move_value, rotate_angle):
        Applies translation and rotation transformations to the data.
    three_order():
        Generates data points following a modified cubic function with added noise, scales the data, and applies transformations.
        Returns a DataFrame with columns 'X coordinate', 'Y coordinate', and 'group'.
    linear():
        Generates data points following a linear function with added noise and applies transformations.
        Returns a DataFrame with columns 'X coordinate', 'Y coordinate', and 'group'.
    cluster(datacenter, scale):
        Generates data points following a normal distribution centered at datacenter with a given scale and applies transformations.
        Returns a DataFrame with columns 'X coordinate', 'Y coordinate', and 'group'.
    lip_cluster(datacenter, axes):
        Generates elliptical cluster data points centered at datacenter with given axes lengths and applies transformations.
        Returns a DataFrame with columns 'X coordinate', 'Y coordinate', and 'group'.
    generate_cluster(cluster_type, **kwargs):
        Generates cluster data based on the specified cluster_type ('three_order', 'linear', 'cluster', 'lip_cluster').
        Returns a DataFrame with columns 'X coordinate', 'Y coordinate', and 'group'.
"""

# Required Libraries
import numpy as np
import pandas as pd

class ClusterGenerator:
    def __init__(self, entire_data_number, main_class_name, test_data_numbers, test_class_names):
        self.entire_data_number = entire_data_number    # Number of data points to generate
        self.main_class_name = main_class_name          # Name of the main class, which is a string
        self.test_data_numbers = test_data_numbers      # Number of data points to change the group, which is an array, same length as test_class_names
        self.class_names = test_class_names             # Name of the classes, which is an array

    def apply_transformations(self, data, move_value, rotate_angle):
        """
        Applies translation and rotation transformations to the dataset.

        Parameters:
        - data (DataFrame): The dataset containing 'X coordinate' and 'Y coordinate'.
        - move_value (list or array): The translation vector [dx, dy].
        - rotate_angle (float): Rotation angle in degrees.

        Returns:
        - Transformed DataFrame.
        """
        # Translate (move)
        data['1'] += move_value[0]
        data['2'] += move_value[1]

        # Convert degrees to radians
        theta = np.radians(rotate_angle)

        # Compute the rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Compute the centroid (mean of X and Y coordinates)
        centroid_x = data['1'].mean()
        centroid_y = data['2'].mean()

        # Shift data to the origin before rotating
        shifted_x = data['1'] - centroid_x
        shifted_y = data['2'] - centroid_y

        # Apply rotation
        rotated_coords = np.dot(rotation_matrix, np.vstack((shifted_x, shifted_y)))

        # Shift data back after rotation
        data['1'] = rotated_coords[0, :] + centroid_x
        data['2'] = rotated_coords[1, :] + centroid_y
        
        # print("Randomly chosen move value:", move_value)
        # print("Randomly chosen rotation angle:", rotate_angle)

        return data


    def three_order(self, cluster_type):
        def modified_cubic_function(x):
            return 0.1 * x**3 + 0.6 * x**2 + 0.4 * x

        def scale_function(x, y, x_min=0, x_max=10, y_min=0, y_max=10):
            x_scaled = x_min + (x - np.min(x)) * (x_max - x_min) / (np.max(x) - np.min(x))
            y_scaled = y_min + (y - np.min(y)) * (y_max - y_min) / (np.max(y) - np.min(y))
            return x_scaled, y_scaled

        x_values = np.random.uniform(0, 10, self.entire_data_number)
        y_values = modified_cubic_function(x_values)
        noise = np.random.normal(-2, 2, y_values.size)
        y_values += noise
        x_values, y_values = scale_function(x_values, y_values)

        data = pd.DataFrame({
            '1': x_values,
            '2': y_values,
            'cluster_label': self.main_class_name,
            'class': self.main_class_name,
            'shape': [cluster_type] * len(x_values)
        })

        for i in range(len(self.class_names)):
            available_population = data[data['class'] == self.main_class_name]
            sample_size = min(self.test_data_numbers[i], len(available_population))
            selected_indices = available_population.sample(n=sample_size, random_state=1).index
            data.loc[selected_indices, 'class'] = self.class_names[i]

        return data.sample(frac=1, random_state=1).reset_index(drop=True)

    def linear(self, cluster_type):
        def linear_function(x, slope=4, intercept=0):
            return slope * x + intercept

        x_values = np.random.uniform(0, 10, self.entire_data_number)
        noise = np.random.normal(0, 1, self.entire_data_number)
        y_values = linear_function(x_values) + noise

        data = pd.DataFrame({
            '1': x_values,
            '2': y_values,
            'cluster_label': self.main_class_name,
            'class': self.main_class_name,
            'shape': [cluster_type] * len(x_values)
        })

        for i in range(len(self.class_names)):
            available_population = data[data['class'] == self.main_class_name]
            sample_size = min(self.test_data_numbers[i], len(available_population))
            selected_indices = available_population.sample(n=sample_size, random_state=1).index
            data.loc[selected_indices, 'class'] = self.class_names[i]

        return data.sample(frac=1, random_state=1).reset_index(drop=True)

    def cluster(self, datacenter, scale, cluster_type):
        cluster_data = np.random.normal(loc=datacenter, scale=scale, size=(self.entire_data_number, 2))
        df = pd.DataFrame(cluster_data, columns=['1', '2'])
        df['cluster_label'] = self.main_class_name
        df['class'] = self.main_class_name
        df['shape'] = [cluster_type] * len(df['cluster_label'])

        for i in range(len(self.class_names)):
            available_population = df[df['class'] == self.main_class_name]
            sample_size = min(self.test_data_numbers[i], len(available_population))
            selected_indices = available_population.sample(n=sample_size, random_state=1).index
            df.loc[selected_indices, 'class'] = self.class_names[i]

        return df.sample(frac=1, random_state=1).reset_index(drop=True)

    def lip_cluster(self, datacenter, axes, cluster_type):
        center_x, center_y = datacenter[0], datacenter[1]
        axis_x, axis_y = axes[0], axes[1]

        theta = np.random.uniform(0, 2 * np.pi, self.entire_data_number)
        r = np.sqrt(np.random.uniform(0, 1, self.entire_data_number))
        x = r * axis_x * np.cos(theta)
        y = r * axis_y * np.sin(theta)

        data = pd.DataFrame({
            '1': x + center_x,
            '2': y + center_y,
            'cluster_label': self.main_class_name,
            'class': self.main_class_name,
            'shape': [cluster_type] * len(x)
        })

        for i in range(len(self.class_names)):
            available_population = data[data['class'] == self.main_class_name]
            sample_size = min(self.test_data_numbers[i], len(available_population))
            selected_indices = available_population.sample(n=sample_size, random_state=1).index
            data.loc[selected_indices, 'class'] = self.class_names[i]

        return data.sample(frac=1, random_state=1).reset_index(drop=True)

    def generate_cluster(self, cluster_type, **kwargs):
        if cluster_type == 'three_order':
            return self.three_order(cluster_type = cluster_type, **kwargs)
        elif cluster_type == 'linear':
            return self.linear(cluster_type = cluster_type, **kwargs)
        elif cluster_type == 'cluster':
            return self.cluster(cluster_type = cluster_type, **kwargs)
        elif cluster_type == 'lip_cluster':
            return self.lip_cluster(cluster_type = cluster_type, **kwargs)
        else:
            raise ValueError(f"Unknown cluster type: {cluster_type}")


if __name__ == "__main__":
    # Example Usage
    generator = ClusterGenerator(
        entire_data_number=500,
        main_class_name="group_0",
        test_data_numbers=[10, 15],
        test_class_names=["group_1", "group_2"]
    )

    # Generate data and apply transformations
    three_order_data = generator.generate_cluster(cluster_type='cluster', datacenter=[5, 5], scale=[0.1, 1])
    three_order_data = generator.apply_transformations(three_order_data, move_value=[2, 3], rotate_angle=110)
    import matplotlib.pyplot as plt

    # Plot the scatterplot of the data
    plt.figure(figsize=(10, 6))
    plt.scatter(three_order_data['1'], three_order_data['1'], c=three_order_data['class'].apply(lambda x: {'group_0': 'blue', 'group_1': 'green', 'group_2': 'red'}[x]), alpha=0.5)
    plt.title('Scatter plot of the transformed data')
    plt.xlabel('1')
    plt.ylabel('2')
    plt.grid(True)
    plt.show()
    
    
