import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np  # Import NumPy for array manipulation



def generate_grid_with_data(x_min, x_max, y_min, y_max, num_rows, num_cols, data_points, attribute_1, attribute_2):
    # Calculate the width and height of each grid cell
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows
    
    # Initialize a list to store the grid coordinates, data points, and counts
    grid_info = []
    
    # Initialize a counter for grid cell numbering
    grid_counter = 1
    
    # Loop through rows and columns to generate grid coordinates
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the coordinates of the current grid cell
            x1 = x_min + col * grid_width
            x2 = x1 + grid_width
            y1 = y_min + row * grid_height
            y2 = y1 + grid_height
            
            # Create a key for the grid cell (e.g., "grid_1") and store the coordinates as a list
            grid_key = f"grid_{grid_counter}"
            
            # Filter data points that fall within the current grid cell
            # points_in_grid = [point for index,point in data_points.iterrows() if x1 <= float(point[attribute_1]) <= x2 and y1 <= float(point[attribute_2]) <= y2]
            
            points_in_grid = []
            for index, point in data_points.iterrows():
                # Access data in the row using row['column_name']
                # print(index, row['Column1'], row['Column2'])
                if x1 <= float(point[attribute_1]) <= x2 and y1 <= float(point[attribute_2]) <= y2:
                    points_in_grid.append(row)
            
            # Count the number of data points in the grid
            point_count = len(points_in_grid)
            
            # Create a dictionary for the grid cell info and append it to the list
            grid_info.append({
                'grid_key': grid_key,
                'coordinates': [x1, x2, y1, y2],
                'data_points': points_in_grid,
                'point_count': point_count
            })
            
            # Increment the grid counter
            grid_counter += 1
    
    return grid_info













# ----------------------------------------------------------
# Visualize the Sample_Superstore_Order data
current_directory = os.getcwd()
# Load the CSV data into a Pandas DataFrame
df = pd.read_csv(current_directory+'/datasets/Sample_Superstore_Orders.csv')


# Specify the percentage of rows you want to extract (30% in this case)
percentage_to_extract = 0.5

# Use the 'sample' method to randomly extract rows
# Set 'frac' to the desired percentage (0.3 for 30%)
# Set 'random_state' for reproducibility
df = df.sample(frac=percentage_to_extract, random_state=42)


# Parameters
deminsion_1 = 'Sales'
deminsion_2 = 'Profit'
deminsion_3 = 'Cluster'
x_min = -100
x_max = 25000
y_min = -8000
y_max = 8600

# Determine the number of rows and columns in the grid heatmap
num_rows = 6
num_cols = 10

x = df[deminsion_1]
y = df[deminsion_2]
z = df[deminsion_3]

# Create a scatterplot
plt.figure(figsize=(10, 6))




# Set the scale for both x-axis and y-axis
plt.xlim(x_min, x_max)  # Specify the range for the x-axis
plt.ylim(y_min, y_max)  # Specify the range for the y-axis

# Adjust the subplot parameters to set the size of the plot area (excluding labels)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

grid_info = generate_grid_with_data(x_min, x_max, y_min, y_max, num_rows, num_cols, df, deminsion_1, deminsion_2)
# Define the grid cells and their colors based on point_count
grid_coordinates = [grid['coordinates'] for grid in grid_info]
grid_colors = [grid['point_count'] for grid in grid_info]

# Reshape grid_colors into a 2D array
grid_colors_2d = np.array(grid_colors).reshape(num_rows, num_cols)

plt.scatter(x, y, c=z, cmap='viridis', marker='o', alpha=0.6)
# sns.scatterplot(x=x, y=y, hue=z, palette='viridis', marker='o', alpha=0.6, s=100)

# Set the same scale for both x-axis and y-axis in heatmap
ax = plt.gca()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Create a grid heatmap
# sns.heatmap(data=grid_colors_2d, cmap='Blues', cbar=False, annot=True, fmt='d', annot_kws={"size": 12, "color": "black"},
#             linewidths=0.5, linecolor='gray', square=True, xticklabels=False, yticklabels=False, alpha = 0.3)


# Loop through grid cells and add the text annotations (point_count) to the heatmap
# for i, coords in enumerate(grid_coordinates):
#     x_center = (coords[0] + coords[1]) / 2
#     y_center = (coords[2] + coords[3]) / 2
#     plt.text(x_center, y_center, str(grid_colors[i]), ha='center', va='center', fontsize=10, color='black')



# Add labels and a colorbar
plt.xlabel(deminsion_1)
plt.ylabel(deminsion_2)
# plt.colorbar(label='Pixel Height')

# Customize plot appearance as needed
plt.title('Scatterplot of '+ deminsion_1 + ' and ' + deminsion_2)
# plt.grid(True)

# Show the plot
plt.show()




# # ------------------------------------------------------------------------------
# # Use the mobilePhonePrice data
# # Load the CSV data into a Pandas DataFrame

# current_directory = os.getcwd()
# df = pd.read_csv(current_directory+'/datasets/MobilePhonePrice.csv')


# # Specify the percentage of rows you want to extract (30% in this case)
# percentage_to_extract = 0.5

# # Use the 'sample' method to randomly extract rows
# # Set 'frac' to the desired percentage (0.3 for 30%)
# # Set 'random_state' for reproducibility
# df = df.sample(frac=percentage_to_extract, random_state=42)


# # Visualize Scatterplots;

# # Extract three dimensions from the data
# x = df['battery_power']
# y = df['ram']
# z = df['px_height']


# # Create a scatterplot
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, c=z, cmap='viridis', marker='o', alpha=0.6)

# # Add labels and a colorbar
# plt.xlabel('battery_power')
# plt.ylabel('ram')
# plt.colorbar(label='Pixel Height')

# # Customize plot appearance as needed
# plt.title('Scatterplot of Battery Power, RAM, and Pixel Height')
# plt.grid(True)

# # Show the plot
# plt.show()






# Print the grid information including data points and point counts
# for grid in grid_info:
#     print(f"Grid {grid['grid_key']} - Coordinates: {grid['coordinates']} - Point Count: {grid['point_count']}")
#     if grid['point_count'] > 0:
#         print(f"Data Points: {grid['data_points']}")