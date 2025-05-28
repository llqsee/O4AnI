import numpy as np
import math
from math import pi

# references: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html


                        
def is_point_in_triangle(p, p0, p1, p2):
    """
    Check if point p is inside the triangle defined by points p0, p1, and p2.
    """
    # Barycentric coordinates
    A = 1/2 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])
    sign = -1 if A < 0 else 1
    s = (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]) * sign
    t = (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]) * sign

    return s > 0 and t > 0 and (s + t) < 2 * A * sign



def update_pixel_matrix_for_square(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height, color, use_colors, important_value, unique_categories = None):
    
    # calculate the start and end points of the square
    marker_size_pixels = int(np.sqrt(marker_size_pixels) * (100 / 72))  # 72 points per inch
    x_start = int(max(x_pix - marker_size_pixels / 2, 0))
    x_end = int(min(x_pix + marker_size_pixels / 2, pixel_width))
    y_start = int(max(y_pix - marker_size_pixels / 2, 0))
    y_end = int(min(y_pix + marker_size_pixels / 2, pixel_height))

    for px in range(x_start, x_end):
        for py in range(y_start, y_end):
            if 0 <= py < pixel_height and 0 <= px < pixel_width:
                if use_colors == 'yes':
                    # Update color matrix
                    # if color not in self.pixel_color_matrix[py, px]:
                    if important_value is None:
                        data.pixel_color_matrix[py, px].append(1)
                    else:
                        data.pixel_color_matrix[py, px].append({'category': color, 'importance_value': important_value, 'ID': id})
                        # While we don't consider if a data point is important or not
                        # pixel_matrix[py, px] += 1
                        # While we consider if a data point is important or not
                        # pixel_matrix[py,px] = pixel_matrix[py,px] + important_value
                elif use_colors == 'no' or 'continous':
                    if important_value is None:
                        data.pixel_noncolor_matrix[py, px].append(1)
                    else:
                        data.pixel_noncolor_matrix[py, px].append(important_value)
                        
                        

                    
                    
def update_pixel_matrix_for_plus(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height, color, use_colors, important_value, unique_categories = None):
    # Calculate the half size of each small square
    arm_length = int(np.sqrt(marker_size_pixels) * (100 / 72))
    # marker_size_pixels = int(2 * np.sqrt(marker_size_pixels/2) * (100 / 72))  # 72 points per inch
    # small_square = marker_size_pixels / 3
    square_length = arm_length / 3

    # Define the center and four other squares
    plus_squares = [
        (x_pix, y_pix),                 # Center square
        (x_pix - square_length, y_pix),   # Left square
        (x_pix + square_length, y_pix),   # Right square
        (x_pix, y_pix - square_length),   # Top square
        (x_pix, y_pix + square_length)    # Bottom square
    ]


    for (x_square, y_square) in plus_squares:
        # Calculate the boundaries of each square
        x_start = int(max(x_square - square_length/2, 0))
        x_end = int(min(x_square + square_length/2, pixel_width))
        y_start = int(max(y_square - square_length/2, 0))
        y_end = int(min(y_square + square_length/2, pixel_height))

        # Update pixels for the current square
        for px in range(x_start, x_end):
            for py in range(y_start, y_end):
                if 0 <= py < pixel_height and 0 <= px < pixel_width:
                    if use_colors == 'yes':
                        if unique_categories and color in unique_categories:
                            # color_index = unique_categories.index(color)
                            if important_value is None:
                                data.pixel_color_matrix[py, px].append(1)
                            else:
                                data.pixel_color_matrix[py, px].append({'category': color, 'importance_value': important_value, 'ID': id})
                    elif use_colors in ['no', 'continuous']:
                        if important_value is None:
                            data.pixel_noncolor_matrix[py, px].append(1)
                        else:
                            data.pixel_noncolor_matrix[py, px].append(important_value)
                    

                        





def update_pixel_matrix_for_triangle(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height, color, use_colors, important_value, unique_categories = None):
    triangle_height = marker_size_pixels * (np.sqrt(3) / 2)
    vertex1 = (x_pix, y_pix + triangle_height / 2)
    vertex2 = (x_pix - marker_size_pixels / 2, y_pix - triangle_height / 2)
    vertex3 = (x_pix + marker_size_pixels / 2, y_pix - triangle_height / 2)

    y_start = int(max(y_pix - triangle_height // 2, 0))
    y_end = int(min(y_pix + triangle_height // 2, pixel_height))
    x_start = int(max(x_pix - marker_size_pixels // 2, 0))
    x_end = int(min(x_pix + marker_size_pixels // 2, pixel_width))

    for px in range(x_start, x_end):
        for py in range(y_start, y_end):
            if 0 <= py < pixel_height and 0 <= px < pixel_width:
                if is_point_in_triangle((px, py), vertex1, vertex2, vertex3):
                    if use_colors == 'yes':
                        # Update color matrix
                        # if color not in self.pixel_color_matrix[py, px]:
                        #     self.pixel_color_matrix[py, px].add(color)
                        #     pixel_matrix[py, px] += 1
                        if important_value is None:
                            data.pixel_color_matrix[py, px][unique_categories.index(color)].append(1)
                        else:
                            data.pixel_color_matrix[py, px].append({'category': color, 'importance_value': important_value, 'ID': id})
                    elif use_colors == 'no' or 'continous':
                        data.pixel_noncolor_matrix[py, px] += 1






def update_pixel_matrix_for_circle(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height, color, use_colors, important_value, unique_categories = None):
    radius = int(math.sqrt(marker_size_pixels / pi) * (100 / 72))  # 72 points per inch
    for px in range(x_pix - radius, x_pix + radius):
        for py in range(y_pix - radius, y_pix + radius):
            if 0 <= py < pixel_height and 0 <= px < pixel_width:
                if (px - x_pix)**2 + (py - y_pix)**2 <= radius**2:
                    if use_colors == 'yes':
                        if important_value is None:
                            data.pixel_color_matrix[py, px].append(1)
                        else:
                            data.pixel_color_matrix[py, px].append({'category': color, 'importance_value': important_value})
                    elif use_colors in ['no', 'continuous']:
                        if important_value is None:
                            data.pixel_noncolor_matrix[py, px].append(1)
                        else:
                            data.pixel_noncolor_matrix[py, px].append(important_value)






def update_pixel_matrix_for_square_noncolor(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height):
    
    # calculate the start and end points of the square
    marker_size_pixels = int(np.sqrt(marker_size_pixels) * (100 / 72))  # 72 points per inch
    x_start = int(max(x_pix - marker_size_pixels / 2, 0))
    x_end = int(min(x_pix + marker_size_pixels / 2, pixel_width))
    y_start = int(max(y_pix - marker_size_pixels / 2, 0))
    y_end = int(min(y_pix + marker_size_pixels / 2, pixel_height))

    for px in range(x_start, x_end):
        for py in range(y_start, y_end):
            if 0 <= py < pixel_height and 0 <= px < pixel_width:
                data.pixel_matrix[py, px].append(id)



def update_pixel_matrix_for_plus_noncolor(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height):
    # Calculate the half size of each small square
    arm_length = int(np.sqrt(marker_size_pixels) * (100 / 72))
    # marker_size_pixels = int(2 * np.sqrt(marker_size_pixels/2) * (100 / 72))  # 72 points per inch
    # small_square = marker_size_pixels / 3
    square_length = arm_length / 3

    # Define the center and four other squares
    plus_squares = [
        (x_pix, y_pix),                 # Center square
        (x_pix - square_length, y_pix),   # Left square
        (x_pix + square_length, y_pix),   # Right square
        (x_pix, y_pix - square_length),   # Top square
        (x_pix, y_pix + square_length)    # Bottom square
    ]

    area = []
    for (x_square, y_square) in plus_squares:
        # Calculate the boundaries of each square
        x_start = int(max(x_square - square_length/2, 0))
        x_end = int(min(x_square + square_length/2, pixel_width))
        y_start = int(max(y_square - square_length/2, 0))
        y_end = int(min(y_square + square_length/2, pixel_height))

        # Update pixels for the current square
        for px in range(x_start, x_end):
            for py in range(y_start, y_end):
                if 0 <= py < pixel_height and 0 <= px < pixel_width:
                    data.pixel_matrix[py, px].append(id)
    
                    
                    
                    
def update_pixel_matrix_for_triangle_noncolor(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height, color, use_colors, important_value, unique_categories = None):
    triangle_height = marker_size_pixels * (np.sqrt(3) / 2)
    vertex1 = (x_pix, y_pix + triangle_height / 2)
    vertex2 = (x_pix - marker_size_pixels / 2, y_pix - triangle_height / 2)
    vertex3 = (x_pix + marker_size_pixels / 2, y_pix - triangle_height / 2)

    y_start = int(max(y_pix - triangle_height // 2, 0))
    y_end = int(min(y_pix + triangle_height // 2, pixel_height))
    x_start = int(max(x_pix - marker_size_pixels // 2, 0))
    x_end = int(min(x_pix + marker_size_pixels // 2, pixel_width))

    for px in range(x_start, x_end):
        for py in range(y_start, y_end):
            if 0 <= py < pixel_height and 0 <= px < pixel_width:
                if is_point_in_triangle((px, py), vertex1, vertex2, vertex3):
                    data.pixel_matrix[py, px].append(id)
    
    
    
                    
                    
                            
def update_pixel_matrix_for_circle_noncolor(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height, color, use_colors, important_value, unique_categories = None):
    radius = int(math.sqrt(marker_size_pixels / pi) * (100 / 72))  # 72 points per inch
    for px in range(x_pix - radius, x_pix + radius):
        for py in range(y_pix - radius, y_pix + radius):
            if 0 <= py < pixel_height and 0 <= px < pixel_width:
                if (px - x_pix)**2 + (py - y_pix)**2 <= radius**2:
                        data.pixel_matrix[py, px].append(id)
    
    
    
    
    
    

    

# =============================================================================
# density transform for pairewise-based method

def area_triangle(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height, color, use_colors, important_value, unique_categories = None):
    triangle_height = marker_size_pixels * (np.sqrt(3) / 2)
    vertex1 = (x_pix, y_pix + triangle_height / 2)
    vertex2 = (x_pix - marker_size_pixels / 2, y_pix - triangle_height / 2)
    vertex3 = (x_pix + marker_size_pixels / 2, y_pix - triangle_height / 2)

    y_start = int(max(y_pix - triangle_height // 2, 0))
    y_end = int(min(y_pix + triangle_height // 2, pixel_height))
    x_start = int(max(x_pix - marker_size_pixels // 2, 0))
    x_end = int(min(x_pix + marker_size_pixels // 2, pixel_width))
                    
    # calculate the area of the triangle
    area = [[px, py] for px in range(x_start, x_end) for py in range(y_start, y_end) if is_point_in_triangle((px, py), vertex1, vertex2, vertex3)]
    data.data.loc[data.data['ID'] == id, 'covered_pixels'] = data.data.loc[data.data['ID'] == id, 'covered_pixels'].apply(lambda x: area)
    
    
    

def area_plus(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height):
    # Calculate the half size of each small square
    arm_length = int(np.sqrt(marker_size_pixels) * (100 / 72))
    # marker_size_pixels = int(2 * np.sqrt(marker_size_pixels/2) * (100 / 72))  # 72 points per inch
    # small_square = marker_size_pixels / 3
    square_length = arm_length / 3

    # Define the center and four other squares
    plus_squares = [
        (x_pix, y_pix),                 # Center square
        (x_pix - square_length, y_pix),   # Left square
        (x_pix + square_length, y_pix),   # Right square
        (x_pix, y_pix - square_length),   # Top square
        (x_pix, y_pix + square_length)    # Bottom square
    ]

    area = []
    for (x_square, y_square) in plus_squares:
        # Calculate the boundaries of each square
        x_start = int(max(x_square - square_length/2, 0))
        x_end = int(min(x_square + square_length/2, pixel_width))
        y_start = int(max(y_square - square_length/2, 0))
        y_end = int(min(y_square + square_length/2, pixel_height))
                    
        area += [[px, py] for px in range(x_start, x_end) for py in range(y_start, y_end)]
        
    data.data.loc[data.data['ID'] == id, 'covered_pixels'] = data.data.loc[data.data['ID'] == id, 'covered_pixels'].apply(lambda x: area)  
    
    
    
def area_circle(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height):
    radius = int(math.sqrt(marker_size_pixels / pi) * (100 / 72))  # 72 points per inch
    
    # calculate the area of the circle
    area = [[px, py] for px in range(x_pix - radius, x_pix + radius) for py in range(y_pix - radius, y_pix + radius) if (px - x_pix)**2 + (py - y_pix)**2 <= radius**2]
    data.data.loc[data.data['ID'] == id, 'covered_pixels'] = data.data.loc[data.data['ID'] == id, 'covered_pixels'].apply(lambda x: area)
    
    
    
def area_square(data, x_pix, y_pix, id, marker_size_pixels, pixel_width, pixel_height):
    
    # calculate the start and end points of the square
    marker_size_pixels = int(np.sqrt(marker_size_pixels) * (100 / 72))  # 72 points per inch
    x_start = int(max(x_pix - marker_size_pixels / 2, 0))
    x_end = int(min(x_pix + marker_size_pixels / 2, pixel_width))
    y_start = int(max(y_pix - marker_size_pixels / 2, 0))
    y_end = int(min(y_pix + marker_size_pixels / 2, pixel_height))
    
    # Define the area with an array of coordinates
    area = [[px, py] for px in range(x_start, x_end) for py in range(y_start, y_end)]
    
    data.data.loc[data.data['ID'] == id, 'covered_pixels'] = data.data.loc[data.data['ID'] == id, 'covered_pixels'].apply(lambda x: area)




# =============================================================================
# Calculate how many data points are totally covered by other data points (markers)
