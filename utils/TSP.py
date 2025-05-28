import pandas as pd
import numpy as np
from itertools import permutations

# https://www.google.com/search?q=open+traveling+salesman+problem&rlz=1C1GCEU_en-gbGB1077GB1077&oq=open+Traveling+Salesman+Problem&gs_lcrp=EgZjaHJvbWUqBwgAEAAYgAQyBwgAEAAYgAQyCAgBEAAYFhgeMggIAhAAGBYYHjINCAMQABiGAxiABBiKBTINCAQQABiGAxiABBiKBTINCAUQABiGAxiABBiKBTINCAYQABiGAxiABBiKBagCALACAA&sourceid=chrome&ie=UTF-8

# If you don't want to fix the starting and ending cities in the route, 
# and if these cities don't need to be the same (i.e., the route does not 
# need to return to the starting city), you're looking at a variation of the 
# Traveling Salesman Problem known as the "Multiple TSP" or "Vehicle Routing 
# Problem." In this scenario:

# The route can start and end at any city.
# The goal is still to minimize the total distance while visiting each city exactly once.


def find_shortest_route_max_start_end_distance(distance_matrix, city_names):
    """
    Finds the shortest route that visits each city exactly once, with the constraint that 
    the first city should have the biggest distance from the last city.

    Args:
    distance_matrix (numpy.ndarray): A 2D array of distances between cities.
    city_names (list): List of city names corresponding to the indices in the distance matrix.

    Returns:
    tuple: A tuple containing the shortest route (as city names) and its total distance.
    """
    def total_distance(route, distance_array):
        total_dist = 0
        for i in range(len(route) - 1):
            total_dist += distance_array[route[i]][route[i+1]]
        return total_dist

    num_cities = len(city_names)
    all_city_indices = list(range(num_cities))
    longest_start_end_distance = 0
    chosen_start = chosen_end = None

    # Find the pair of cities with the maximum start-end distance
    for start in range(num_cities):
        for end in range(num_cities):
            if start != end and distance_matrix[start][end] > longest_start_end_distance:
                longest_start_end_distance = distance_matrix[start][end]
                chosen_start, chosen_end = start, end

    # Generate permutations of the remaining cities
    remaining_cities = [i for i in all_city_indices if i != chosen_start and i != chosen_end]
    shortest_route = None
    shortest_distance = float('inf')

    for route in permutations(remaining_cities):
        full_route = [chosen_start] + list(route) + [chosen_end]
        route_distance = total_distance(full_route, distance_matrix)
        if route_distance < shortest_distance:
            shortest_distance = route_distance
            shortest_route = full_route

    # Convert route indices to city names
    shortest_route_names = [city_names[i] for i in shortest_route]

    return shortest_route_names, shortest_distance


if __name__ == "__main__":
    # File location of the distance matrix CSV
    file_location = 'datasets/distance_matrix.csv'

    # Load the CSV data into a Pandas DataFrame
    distance_matrix_df = pd.read_csv(file_location)

    # Extracting city names from the first row (headers)
    city_names = distance_matrix_df.columns[1:].tolist()

    # Convert all elements in the DataFrame to numeric type and to a NumPy array
    distance_matrix = distance_matrix_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy()


    # Run the function
    shortest_route, shortest_distance = find_shortest_route_max_start_end_distance(distance_matrix, city_names)
    
    # Print the shortest route and its distance
    print("Shortest route:", shortest_route)
    print("Shortest distance:", shortest_distance)
