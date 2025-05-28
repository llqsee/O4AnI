# =============================================================================
# This function grid_density(grid) is used to calculate the density matrix of the grid.

def grid_density(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Initialize density matrix with same dimensions as grid
    density_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    # Loop through each cell in the grid
    for i in range(rows):
        for j in range(cols):
            # Get the category of the top layer point in the current cell
            current_category = grid[i][j]
            same_category_count = 0
            
            if grid[i][j] is not None:
                # Count how many neighbors have the same top layer category
                for ni, nj in get_neighbors(i, j, rows, cols):  # get_neighbors returns valid neighboring cells
                    if grid[ni][nj] == current_category:
                        same_category_count += 1
            
            # Store the count (density) in the corresponding cell in the density matrix
            density_matrix[i][j] = same_category_count

    return density_matrix

def get_neighbors(i, j, rows, cols):
    # Define the Moore neighborhood (25 surrounding cells)
    # neighbors = [
    #     (i-2, j-2), (i-2, j-1), (i-2, j), (i-2, j+1), (i-2, j+2),
    #     (i-1, j-2), (i-1, j-1), (i-1, j), (i-1, j+1), (i-1, j+2),
    #     (i, j-2),   (i, j-1),             (i, j+1),   (i, j+2),
    #     (i+1, j-2), (i+1, j-1), (i+1, j), (i+1, j+1), (i+1, j+2),
    #     (i+2, j-2), (i+2, j-1), (i+2, j), (i+2, j+1), (i+2, j+2)
    # ]
    neighbors = [
        (i-1, j-1), (i-1, j), (i-1, j+1),
        (i, j-1),             (i, j+1),
        (i+1, j-1), (i+1, j), (i+1, j+1)
    ]
    
    # Filter out invalid neighbors that are out of bounds
    valid_neighbors = [(ni, nj) for ni, nj in neighbors if is_valid(ni, nj, rows, cols)]
    
    return valid_neighbors

def is_valid(i, j, rows, cols):
    # Check if a neighbor is within grid bounds
    return 0 <= i < rows and 0 <= j < cols