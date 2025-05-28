import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("Current Working Directory:", os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.dataGenerator import ClusterGenerator
import random
from Our_metrics.Scatter_Metrics import Scatter_Metric


# ==================================================================================================
# Define the parameters for generating scatterplots
# The following parameters correspond to each cluster in scatterplot
many_functions = ['cluster', 'cluster', 'cluster', 'cluster', 'cluster', 'cluster', 'three_order', 'three_order','linear', 'lip_cluster', 'lip_cluster', 'lip_cluster']
# the test_data_numbers can be used to randomly generate the number of test data for each cluster
test_data_number_categories = {'small': [0, 0, 1, 2, 3], 'medium': [0, 0, 4, 5, 6], 'large': [0, 0, 7, 8, 9]}

# the following parameters correspond to scatterplot
# cluster_numbers = [2, 3, 5]
cluster_numbers = [3]

# entire_data_numbers = [100, 200, 500]
entire_data_numbers = [500]

# how many times we repeat the generation of scatterplot
repeat_generation_number = 2

# Define the folder for saving the generated scatterplots
csv_folder_name = os.getcwd() + '/datasets/simulated_datasets/csv_files_2'
# Ensure the folder for saving figures exists
os.makedirs(csv_folder_name, exist_ok=True)
# ==================================================================================================






def generate_cluster_centers(num_clusters, min_x=-10, max_x=30, min_y=-10, max_y=30, min_distance=4):
    """
    Generate non-overlapping cluster centers using a minimum separation distance.
    """
    centers = []
    max_attempts = 10000  # Avoid infinite loops

    for _ in range(num_clusters):
        attempts = 0
        while attempts < max_attempts:
            new_center = np.random.uniform([min_x, min_y], [max_x, max_y])
            
            # Ensure minimum separation distance
            if all(np.linalg.norm(new_center - np.array(existing)) >= min_distance for existing in centers):
                centers.append(new_center)
                break
            
            attempts += 1
        else:
            print("Warning: Could not find a non-overlapping center after max attempts")

    return centers



# ==================================================================================================
# Generate and save datasets
for entire_data_number in entire_data_numbers:
    for cluster_number in cluster_numbers:
        for test_number_category, test_data_numbers in test_data_number_categories.items():
            for repeat_number in range(repeat_generation_number):
                name_lists = []
                for _ in range(cluster_number):
                    name_lists.append('group_'+ str(_))
                    
                dataset_lists = []
                
                # generate the datacenters for each cluster
                cluster_centers = generate_cluster_centers(cluster_number, min_x=0, max_x=30, min_y=0, max_y=30, min_distance=3)
                
                # ==================================================================================================
                # Generate a cluster data
                for i in range(cluster_number):
                    test_number = random.choice(test_data_numbers)
                    
                    # Determine `class_number` ensuring it's within limits
                    if test_number == 0:
                        class_number = 0
                    else:
                        max_available_classes = len(name_lists) - 1
                        class_number = min(random.choice(range(1, test_number + 1)), max_available_classes)

                    # Generate `test_data_number` only if `class_number > 0`
                    if class_number > 0:
                        while True:
                            test_data_number = np.random.randint(1, test_number + 1, size=class_number)
                            if test_data_number.sum() == test_number:
                                break
                    else:
                        test_data_number = []

                    main_class_name = name_lists[i]
                    
                    # Select test class names (avoid errors in `random.sample()`)
                    if class_number > 0:
                        test_class_names = random.sample([name for name in name_lists if name != main_class_name], class_number)
                    else:
                        test_class_names = []

                    generator = ClusterGenerator(
                        entire_data_number=entire_data_number,
                        main_class_name=main_class_name,
                        test_data_numbers=test_data_number,
                        test_class_names=test_class_names
                    )
                    cluster_type = random.choice(many_functions)
                    if cluster_type == 'cluster':
                        cluster_data = generator.generate_cluster(
                            cluster_type='cluster',
                            datacenter=cluster_centers[i],
                            scale=[np.random.uniform(1, 2, size=2)]
                        )
                    elif cluster_type == 'three_order':
                        cluster_data = generator.generate_cluster(cluster_type='three_order')
                    elif cluster_type == 'linear':
                        cluster_data = generator.generate_cluster(cluster_type='linear')
                    elif cluster_type == 'lip_cluster':
                        cluster_data = generator.generate_cluster(
                            cluster_type='lip_cluster',
                            datacenter=cluster_centers[i],
                            axes=np.random.uniform(1, 2, size=2)
                        )
                    
                    cluster_data = generator.apply_transformations(
                        cluster_data,
                        move_value=np.random.uniform(-5, 5, size=2),
                        rotate_angle=random.choice([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]),
                    )

                    dataset_lists.append(cluster_data)
                # ==================================================================================================

                # ==================================================================================================
                # Combine datasets
                data = pd.concat(dataset_lists, ignore_index=True)  # Reset index to sequential integers
                # ==================================================================================================

                # ==================================================================================================
                # Save the data as a CSV file
                os.makedirs(csv_folder_name, exist_ok=True)
                csv_path = os.path.join(csv_folder_name, f"clusternumber{cluster_number}_datanumber{entire_data_number}_testnumbercategory{test_number_category}_repeatnumber{repeat_number}.csv")
                data.to_csv(csv_path, index=False)
                # ==================================================================================================

# ==================================================================================================