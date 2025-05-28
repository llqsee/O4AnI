import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from datasets.generateData import load_data  # Ensure this module and function are correctly defined
import seaborn as sns
from utils.importance_calculation import *
from utils.calApproach import *
from utils.mapping_function import *
from utils.density_transform import *
from utils.sort_pandas import sort_dataframe
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
import time


class Scatter_Metric:
    
    def __init__(self, data = None, 
                 margins = {'left':0.2, 'right': 0.7, 'top':0.8, 'bottom': 0.2}, 
                 marker = 'square', 
                 marker_size = 25, 
                 dpi = 100, 
                 figsize = [8,8], 
                 xvariable = None, 
                 yvariable = None, 
                 zvariable = None, 
                 color_map = 'tab10'):
        # ======================================================================================================
        # Initialize the class and set the parameters for all kinds of metrics
        self.data = data
        self.margins = margins
        self.marker = marker
        self.marker_size = marker_size
        self.dpi = dpi
        self.figsize = figsize
        self.xvariable = xvariable
        self.yvariable = yvariable
        self.zvariable = zvariable
        self.color_map = color_map
        self.color_map_function = None
        self.use_colors = None
        self.pixel_color_width = None
        self.pixel_color_height = None
        
        # -----------------------------------------------------------------------------------------------
        #  Calculate how many pixels in each axis of scatterplot
        plot_width = figsize[0] * (margins['right'] - margins['left'])
        plot_height = figsize[1] * (margins['top'] - margins['bottom'])
        self.pixel_width = int(np.round(plot_width * dpi))
        self.pixel_height = int(np.round(plot_height * dpi))
        
        # -----------------------------------------------------------------------------------------------
        # What type scatterplot it is? we define the use_colors as yes, no or continous
        # Also define unique_categories for the categorical data
        if zvariable is None:
            # Use a default color when zvariable is None
            self.use_colors = 'no'
        elif self.data[zvariable].dtype in ['O', 'int64']:
            # If zvariable is categorical, map it to actual colors
            self.use_colors = 'yes'
            unique_categories = sorted(self.data[zvariable].unique())
            color_map = plt.get_cmap(self.color_map)
            self.color_map_function = {category: color_map(i) for i, category in enumerate(sorted(unique_categories))}
            self.unique_categories = unique_categories
        elif self.data[zvariable].dtype == 'float64':
            self.use_colors = 'continous'
        else:
            self.use_colors = 'no'
            
        # -----------------------------------------------------------------------------------------------
        #  set the marker parameters
        # if marker == 'square':
        #     self.marker = 's'
        # elif marker == 'circle':
        #     self.marker = 'o'
        # elif marker == 'triangle':
        #     self.marker = '^'
        # elif marker == 'plus':
        #     self.marker = 'P'
        
        
        # -----------------------------------------------------------------------------------------------
        #  Other parameters
        self.pixel_matrix = None
        self.pixel_color_matrix = None
        
        
        
        # -----------------------------------------------------------------------------------------------
        # Add an attribute to the data named ID and set it as the index, and covered pixels
        self.data['ID'] = range(len(self.data))
        self.data['covered_pixels'] = [[] for _ in range(len(self.data))]
        # ======================================================================================================
        
    def _cal_importance_index(self, 
                              important_cal_method = None, 
                              mapping_method = None, 
                              weight_density = None, 
                              weight_data_inter = None):
        """Calculate the importance index for each data point."""
        # ======================================================================================================
        # Calculate the importance of each data point
        # Add a column to self.data calculated by specific methods about the important index for each points
        # column is named as importance_index
        if important_cal_method is None:
        # if the importantce_index_method is none, we don't use the important-point method
            if self.zvariable not in self.data.columns:
                raise ValueError("The zvariable must be in existing columns.")
                # data_inter = mahalanobis_importance(self.data[[xvariable[0], yvariable[0]]]).drop(columns=[xvariable[0], yvariable[0]])
                # self.data= pd.concat([self.data, data_inter], axis = 1)
            else:
                # In this situation, we don't add any new attributes
                pass
        else:
            # The cluster-based methods
            if important_cal_method == 'mahalanobis_distance':
            # If the importance_index_method is not none, we use the important-point method
                self.data = m_distance(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'average_linkage_method':
                self.data = average_linkage_method(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'complete_linkage_method':
                self.data = complete_linkage_method(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'single_linkage_method':
                self.data = single_linkage_method(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'centroid_method':
                self.data = centroid_method(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'isolation_forest':
                self.data = isolation_forest(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'lof_distance':
                self.data = lof_distance(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            # THe coorelation based methods
            elif important_cal_method == 'leverage_score':
                self.data = leverage_score(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'cook_distance':
                self.data = cook_distance(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'influence_distance':
                self.data = influence_function(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            # elif importance_index_method == 'euc_distance':
            #     self.data = euc_distance(zvariable, self.data, xvariable, yvariable, mapping_method = mapping_method)
            elif important_cal_method == 'orthogonal_distance_to_lowess_line':
                self.data = orthogonal_distance_to_lowess_line(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'vertical_distance_to_lowess_line':
                self.data = vertical_distance_to_lowess_line(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            elif important_cal_method == 'horizontal_distance_to_lowess_line':
                self.data = horizontal_distance_to_lowess_line(self.zvariable, self.data, self.xvariable, self.yvariable, self.pixel_width, self.pixel_height, weight_density, weight_data_inter, mapping_method = mapping_method)
            else:
                raise ValueError("Please input the correct method for calculating important index")
        # ======================================================================================================
        
    def _sort_data(self, attribute = None, order = None, ascending=True):
        self.data = sort_dataframe(self.data, attribute = attribute, order = order, ascending=ascending)
        
        
    def _setup_figure(self):
        """Initializes the figure and axes for the scatter plot."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self.fig.subplots_adjust(left=self.margins['left'], right=self.margins['right'],
                                 top=self.margins['top'], bottom=self.margins['bottom'])
        x_values = self.data[self.xvariable]
        y_values = self.data[self.yvariable]
        
        if self.marker == 'square':
            mk = 's'
        elif self.marker == 'circle':
            mk = 'o'
        elif self.marker == 'triangle':
            mk = '^'
        elif self.marker == 'plus':
            mk = 'P'

        if self.use_colors == 'yes':
            self.ax.scatter(x_values, y_values, alpha=1, edgecolor='none', marker=mk, s=self.marker_size, c=[self.color_map_function[cat] for cat in self.data[self.zvariable]])

            # ======================================================================================================
            # Add the legend to the plot for the categorical data
            handles = [plt.Line2D([0], [0], marker=mk, color='w', label=category,
                                markerfacecolor=color, markersize=10) for category, color in self.color_map_function.items()]
            # Add the legend to the plot
            plt.legend(title=self.zvariable, handles=handles, bbox_to_anchor=(1, 1), loc='upper left', fontsize=16, title_fontsize=18)
            # ======================================================================================================
        elif self.use_colors == 'no':
            self.ax.scatter(x_values, y_values, alpha=1, edgecolor='none', marker=mk, s=self.marker_size, color='black')
        elif self.use_colors == 'continous':
            self.ax.scatter(x_values, y_values, alpha=1, edgecolor='none', marker=mk, s=self.marker_size, c='black')
            
            # ======================================================================================================
            # Add the colorbar to the plot for the continuous data
            min_val = self.data[self.zvariable].min()
            max_val = self.data[self.zvariable].max()

            # Create a ScalarMappable with the normalization and colormap
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(self.color_map), norm=plt.Normalize(vmin=min_val, vmax=max_val))
            sm.set_array([])  # This line is necessary for older versions of matplotlib, may be optional for newer versions

            # Add the colorbar to the current figure based on ScalarMappable
            cbar = plt.colorbar(sm, ax=self.ax)  # 'ax' should be the current axes object of your plot
            cbar.set_label(self.zvariable, fontdict={'fontsize': 18})  # Label the colorbar with your variable name
            cbar.ax.tick_params(labelsize=16)  # Modify the font size of the colorbar ticks
            # ======================================================================================================
            
        
        # Hide the legend, title, border, and ticks
        self.ax.legend().set_visible(False)
        self.ax.set_title("")
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        x_margin = (self.data[self.xvariable].max() - self.data[self.xvariable].min()) * 0.05
        y_margin = (self.data[self.yvariable].max() - self.data[self.yvariable].min()) * 0.05
        self.ax.set_xlim(self.data[self.xvariable].min() - x_margin, self.data[self.xvariable].max() + x_margin)
        self.ax.set_ylim(self.data[self.yvariable].min() - y_margin, self.data[self.yvariable].max() + y_margin)
        self.xrange = [self.data[self.xvariable].min(), self.data[self.xvariable].max()]
        self.yrange = [self.data[self.yvariable].min(), self.data[self.yvariable].max()]
        self.ax.set_xlabel(self.xvariable, fontsize=20)
        self.ax.set_ylabel(self.yvariable, fontsize=20)
        self.ax.set_title(f'Scatterplot of {self.xvariable} and {self.yvariable}', fontsize=22)
        # Setting the tick label size
        self.ax.tick_params(axis='both', which='major', labelsize=18)
        self.fig.canvas.draw()  # Ensure the plot is fully rendered in the canvas
        


    def _data2pixel_(self, shape, x_pix, y_pix, idx, marker_size, pixel_width, pixel_height, color, use_colors, important_value, unique_categories = None):
        '''Map the data points to the pixel matrix based on the different shapes'''
        if shape == 'square':
            update_pixel_matrix_for_square(self, x_pix, y_pix, idx, marker_size, pixel_width, pixel_height, color, use_colors, important_value, unique_categories)
        elif shape == 'circle':
            update_pixel_matrix_for_circle(self, x_pix, y_pix, idx, marker_size, pixel_width, pixel_height, color, use_colors, important_value, unique_categories)
        elif shape == 'triangle':
            update_pixel_matrix_for_triangle(self, x_pix, y_pix, idx, marker_size, pixel_width, pixel_height, color, use_colors, important_value, unique_categories)
        elif shape == 'plus':
            update_pixel_matrix_for_plus(self, x_pix, y_pix, idx, marker_size, pixel_width, pixel_height, color, use_colors, important_value, unique_categories)


    # @property
    def _final_value(self):
        '''Calculate the final value for the importance-based metrics'''
        total_sum_top_layer = 0
        total_sum_other_layer=0
        for row in self.top_layer_matrix:
            for element in row:
                if not np.isnan(element):
                    total_sum_top_layer += element
                
        for row in self.other_layer_matrix: 
            for element in row:
                if not np.isnan(element):
                    total_sum_other_layer += element
                
        result = 1 - (total_sum_other_layer/(total_sum_top_layer+total_sum_other_layer))
        formatted_result = f'{result:.2f}'  # Format the result to 4 decimal places
        print(f'{self.important_cal_method} quality metric is: {formatted_result}')
        return result

    def importance_metric(self, important_cal_method = None, 
                          mapping_method = None, 
                          weight_diff_class = None, 
                          weight_same_class = None, 
                          weight_density = None, 
                          weight_data_inter = None,
                          order_variable = None,
                          rendering_sequence = None, 
                          asending = None):
        """Creates scatterplot and the matrix for saving the weight values."""
        
        
        # ======================================================================================================
        # Parameters for the importance_metric method and general metrics
        
        # -----------------------------------------------------------------------------------------------
        #  Parameters from the general matrics
        zvariable = self.zvariable
        xvariable = self.xvariable
        yvariable = self.yvariable
        marker = self.marker
        marker_size = self.marker_size
        margins = self.margins
        figsize = self.figsize
        dpi = self.dpi
        pixel_width = self.pixel_width
        pixel_height = self.pixel_height
        use_colors = self.use_colors
        
        # -----------------------------------------------------------------------------------------------
        #  Parameters for the importance_metric method
        self.weight_diff_class = 10 if weight_diff_class is None else weight_diff_class
        self.weight_same_class = 1 if weight_same_class is None else weight_same_class
        self.weight_density = 5 if weight_density is None else weight_density
        self.weight_data_inter = 1 if weight_data_inter is None else weight_data_inter
        
        self.pixel_color_matrix = None
        self.pixel_noncolor_matrix = None
        self.top_layer_matrix = None
        self.other_layer_matrix = None
        self.overall_layer_matrix = None
        self.important_cal_method = important_cal_method
        self.mapping_method = mapping_method
            
        # ======================================================================================================
        
        
        # ======================================================================================================
        # Calculate the importance of each data point
        # Add a column to self.data calculated by specific methods about the important index for each points
        # column is named as importance_index
        start_time = time.time()
        self._cal_importance_index(important_cal_method = important_cal_method, 
                       mapping_method = mapping_method, 
                       weight_density = weight_density, 
                       weight_data_inter = weight_data_inter)
        t1 = time.time() - start_time
        # print(f"Time taken for _cal_importance_index: {end_time - start_time:.2f} seconds")
        # ======================================================================================================
        
        
        # ======================================================================================================
        # we sort the data based on the order_variables and order_method
        if order_variable is not None:
            self._sort_data(attribute = order_variable, order = rendering_sequence, ascending=asending)
        else:
            pass
        # ======================================================================================================
        
        
        # ======================================================================================================
        # Set up the scatterplot and create the scatterplots
        self._setup_figure()
        # ======================================================================================================
        
        
        # ======================================================================================================
        # we create the matrix for saving the weight values, which is the pixel_matrix. In each pixel,
        # we set the format as a list and each element in the list is a dictionary including two keys, 
        # which is as {'category': current_class, 'importance_index': important_value}. 

        start_time_1 = time.time()
        # -----------------------------------------------------------------------------------------------
        # loop all the data and put the calculated importance index into the pixel_matrix
        importance_values = self.data['importance_index'] if important_cal_method is not None else [None] * len(self.data)
        if use_colors == 'yes':

            # -----------------------------------------------------------------------------------------------
            # Initiate the pixel_matrix if the use_colors is yes
            self.pixel_color_matrix = np.full((pixel_height + 1, pixel_width + 1), None)
            for i in range(pixel_height + 1):
                for j in range(pixel_width + 1):
                    # self.pixel_color_matrix[i, j] = set()
                    self.pixel_color_matrix[i, j] = []

            # Map each data point to pixel coordinates and update the matrix
            for x, y, color, idx, important_value in zip(self.data[xvariable], self.data[yvariable], self.data[zvariable], self.data['ID'], importance_values):
                x_pix, y_pix = self.ax.transData.transform((x, y))
                x_pix, y_pix = int(x_pix - figsize[0] * dpi * margins['left']), int(y_pix - figsize[1] * dpi * margins['bottom'])
                
                self._data2pixel_(marker, x_pix, y_pix, idx, self.marker_size, pixel_width, pixel_height, color, use_colors, important_value, self.unique_categories)

                # Check for timeout after each row
                if time.time() - start_time > 15:
                    self.calculation_time = 15
                    # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                    self.result = '-'
                    return

            # Calculate the pixel_matrix if the use_colors is yes
            # max_value = np.nanmax([len(pixel) for row in self.pixel_color_matrix for pixel in row if pixel is not None])
            # print(f"The maximum value in self.pixel_color_matrix is: {max_value}")
            top_layer_matrix, other_layer_matrix, overall_layer_matrix = same_class_average_metric_color(self.pixel_color_matrix, weight_diff_class= self.weight_diff_class, weight_same_class=self.weight_same_class)
            self.top_layer_matrix = top_layer_matrix
            self.other_layer_matrix = other_layer_matrix
            self.overall_layer_matrix = overall_layer_matrix
            # np.save('internal_dataset/pixel_color_matrix.npy', self.pixel_color_matrix)
            
        elif use_colors == 'no' or 'continous':
            # -----------------------------------------------------------------------------------------------
            # Initiate the pixel_matrix if the use_colors is no or continous
            self.pixel_noncolor_matrix = np.full((pixel_height + 1, pixel_width + 1), None)
            for i in range(pixel_height + 1):
                for j in range(pixel_width + 1):
                    # self.pixel_color_matrix[i, j] = set()
                    self.pixel_noncolor_matrix[i, j] = []
            # ======================================================================================================
            # calculate the different matrix
            color = None
            for x, y, idx, important_value in zip(self.data[xvariable[0]], self.data[yvariable[0]], self.data['ID'], importance_values):
                x_pix, y_pix = self.ax.transData.transform((x, y))
                x_pix, y_pix = int(x_pix - figsize[0] * dpi * margins['left']), int(y_pix - figsize[1] * dpi * margins['bottom'])
                
                self._data2pixel_(marker, x_pix, y_pix, idx, self.marker_size, pixel_width, pixel_height, color, use_colors, important_value, self.unique_categories)
                
                # Check for timeout after each row
                if time.time() - start_time > 15:
                    self.calculation_time = 15
                    # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                    self.result = '-'
                    return
            
            top_layer_matrix, other_layer_matrix, overall_layer_matrix = sum_all_weight_pixel_noncolor(self.pixel_noncolor_matrix)
            self.top_layer_matrix = top_layer_matrix
            self.other_layer_matrix = other_layer_matrix
            self.overall_layer_matrix = overall_layer_matrix
            # ======================================================================================================
        # ======================================================================================================
        
        t2 = time.time() - start_time_1
    
        
        t = t1 + t2
        print(f"Time taken for {self.important_cal_method}: {t:.2f} seconds")

        # ======================================================================================================
        # we calculate the final value matrix for the importance index
        self.result = self._final_value()
        self.calculation_time = t
        # ======================================================================================================

    def plot_scatter_with_importance(self, filename=None, cmap='Reds', grid=True, x_grid=10, y_grid=10):
        """
        Plots a scatterplot using the importance index as the color channel.

        Args:
        filename (str): If provided, saves the plot to the specified file.
        """
        if 'importance_index' not in self.data.columns:
            raise ValueError("The 'importance_index' column is missing. Please calculate the importance index first.")

        if self.marker == 'square':
            mk = 's'
        elif self.marker == 'circle':
            mk = 'o'
        elif self.marker == 'triangle':
            mk = '^'
        elif self.marker == 'plus':
            mk = 'P'

        self._setup_figure()
        
        # Extract x, y, and importance values
        x_values = self.data[self.xvariable]
        y_values = self.data[self.yvariable]
        importance_values = self.data['importance_index']

        scatter = plt.scatter(x_values, y_values, c=importance_values, cmap=cmap, s=self.marker_size, edgecolor='face' if mk in ['o', 's'] else 'none', marker=mk)
        plt.colorbar(scatter, label='Importance Index')
        plt.xlabel(self.xvariable, fontsize=18)
        plt.ylabel(self.yvariable, fontsize=18)
        plt.title('Scatterplot with Importance Index as Color', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        if grid:
            # Add grid lines based on specific numbers
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            x_ticks = np.linspace(x_min, x_max, x_grid)
            y_ticks = np.linspace(y_min, y_max, y_grid)
            plt.grid(which='both', linestyle='-', linewidth=1.5, alpha=1.0)
            plt.xticks(np.array(x_ticks).flatten(), fontsize=16)
            plt.yticks(np.array(y_ticks).flatten(), fontsize=16)

        # Save or show the plot
        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Scatterplot saved as {filename}")
        else:
            plt.show()


    def plot_density_heatmap(self, gridsize=100, cmap='Reds', filename=None):
        """
        Plots a heatmap showing the density of the dataset distribution.

        Args:
        gridsize (int): The number of points on the grid for density calculation.
        cmap (str): Colormap for the heatmap.
        """
        # Extract the x and y values from the data
        x_values = self.data[self.xvariable]
        y_values = self.data[self.yvariable]

        # Create a 2D histogram for density calculation
        heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=gridsize)

        # Normalize the heatmap
        heatmap = heatmap.T  # Transpose for correct orientation

        # Plot the heatmap
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.imshow(heatmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   origin='lower', cmap=cmap, aspect='auto')
        plt.colorbar(label='Density').ax.tick_params(labelsize=16)  # Make colorbar labels bigger
        # plt.title('Density Heatmap', fontsize=20)
        plt.xlabel(self.xvariable, fontsize=18)
        plt.ylabel(self.yvariable, fontsize=18)
        plt.xticks(fontsize=16)  # Make x-axis tick labels bigger
        plt.yticks(fontsize=16)  # Make y-axis tick labels bigger
        plt.grid(False)
        if filename:
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Density heatmap saved as {filename}")
        else:   
            plt.show()


    def visilibity_index(self):
        start_time = time.time()
        '''
        Calculate the visibility index
        It is from the paper: Prediction of data visibility in two-dimensional scatterplots
        It calculates The amount of always-visible glyphs is the amount of glyphs g 
        in G such that at least one of its pixels in the visualization is occupied only by itself
        The visibility index is the ratio between the amount of always-visible glyphs and the total amount of glyphs.
        '''
        
        # ======================================================================================================
        # Initiate the pixel_matrix if the use_colors is yes
        self.pixel_matrix = np.full((self.pixel_height + 1, self.pixel_width + 1), None)
        for i in range(self.pixel_height + 1):
            for j in range(self.pixel_width + 1):
                # self.pixel_color_matrix[i, j] = set()
                self.pixel_matrix[i, j] = []
                
            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
        # ======================================================================================================
        
        # ======================================================================================================
        # Set up the scatterplot and create the scatterplots
        self._setup_figure()
        # ======================================================================================================
        
        # ======================================================================================================
        # Calculate the pixel based matrix
        for x, y, idx in zip(self.data[self.xvariable], self.data[self.yvariable], self.data['ID']):
            x_pix, y_pix = self.ax.transData.transform((x, y))
            x_pix, y_pix = int(x_pix - self.figsize[0] * self.dpi * self.margins['left']), int(y_pix - self.figsize[1] * self.dpi * self.margins['bottom'])
            
            if self.marker == 'square':
                update_pixel_matrix_for_square_noncolor(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'circle':
                update_pixel_matrix_for_circle_noncolor(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'triangle':
                update_pixel_matrix_for_triangle_noncolor(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'plus':
                update_pixel_matrix_for_plus_noncolor(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
                
            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
        # ======================================================================================================
        
        # ======================================================================================================
        # Calculate the visibility index based on the pixel matrix
        
        # Step 1: Initialize an empty list to store unique values
        unique_solo_values = []

        # Step 2: Loop through each element in the matrix
        for row in self.pixel_matrix:
            for lst in row:
                if len(lst) == 1:  # Check if the element is a solo list
                    value = lst[0]  # Extract the single value
                    if value not in unique_solo_values:  # If not already in the list
                        unique_solo_values.append(value)  # Add the value to the list
                        
            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
                        
        
        # Step 3: Calculate the visibility index based on the unique solo values
        self.result = len(unique_solo_values) / len(self.data)
        print(f'The visibility index is: {self.result:.2f}')
        # ======================================================================================================
        end_time  = time.time()
        self.calculation_time = end_time - start_time
        print(f"Time taken for visibility_index: {self.calculation_time:.2f} seconds")
        
    def mpix(self):
        start_time = time.time()
        '''
        Calculate the mpix metric
        It is from the paper: On the perceptual influence of shape overlap on data-comparison using scatterplots
        It calculates how many pixels being activated by more than one marker.
        '''
        # ======================================================================================================
        # Initiate the pixel_matrix
        self.pixel_matrix = np.full((self.pixel_height + 1, self.pixel_width + 1), None)
        for i in range(self.pixel_height + 1):
            for j in range(self.pixel_width + 1):
                # self.pixel_color_matrix[i, j] = set()
                self.pixel_matrix[i, j] = []
                
            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
        # ======================================================================================================
        
        # ======================================================================================================
        # Set up the scatterplot and create the scatterplots
        self._setup_figure()
        # ======================================================================================================
        
        # ======================================================================================================
        # Calculate the pixel based matrix
        for x, y, idx in zip(self.data[self.xvariable], self.data[self.yvariable], self.data['ID']):
            x_pix, y_pix = self.ax.transData.transform((x, y))
            x_pix, y_pix = int(x_pix - self.figsize[0] * self.dpi * self.margins['left']), int(y_pix - self.figsize[1] * self.dpi * self.margins['bottom'])
            
            if self.marker == 'square':
                update_pixel_matrix_for_square_noncolor(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'circle':
                update_pixel_matrix_for_circle_noncolor(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'triangle':
                update_pixel_matrix_for_triangle_noncolor(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'plus':
                update_pixel_matrix_for_plus_noncolor(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
                
            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
        # ======================================================================================================
        
        
        # -----------------------------------------------------------------------------------------------
        # Calculate the mpix metric based on the pixel matrix, we calculate how many pixels having more than one data points
        # Step 1: Initialize a counter for the number of pixels with multiple data points
        multiple_data_points = 0
        
        # Step 2: Loop through each element in the matrix
        for row in self.pixel_matrix:
            for lst in row:
                if len(lst) > 1:  # Check if the element has more than one data point
                    multiple_data_points += 1  # Increment the counter
                    
            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
                    
        self.result = multiple_data_points
        print(f'The mpix metric result is: {self.result}')
        # ======================================================================================================
        end_time = time.time()
        self.calculation_time = end_time - start_time
        print(f"Time taken for mpix: {self.calculation_time:.2f} seconds")
        
    def grid_based_density_overlap_degree(self, grid_pixel_ratio = 0.1):
        start_time = time.time()
        '''
        Calculate the grid based density overlap degree metric
        It is from the paper: Generalized scatter plots, 
        It calculates the ratio of the number of data points that they share a single pixel with at least one another data point to the total number of data points
        '''
        # ======================================================================================================
        # Initiate the pixel_matrix
        grid_x = int(self.pixel_width * grid_pixel_ratio)
        grid_y = int(self.pixel_height * grid_pixel_ratio)
        self.pixel_matrix = np.full((grid_y + 1, grid_x + 1), None)
        for i in range(grid_y + 1):
            for j in range(grid_x + 1):
                # self.pixel_color_matrix[i, j] = set()
                self.pixel_matrix[i, j] = []
        # ======================================================================================================
        
        # ======================================================================================================
        # Set up the scatterplot and create the scatterplots
        self._setup_figure()
        # ======================================================================================================
        

        # ======================================================================================================
        # Calculate the pixel based matrix
        for x, y, idx in zip(self.data[self.xvariable], self.data[self.yvariable], self.data['ID']):
            x_pix, y_pix = self.ax.transData.transform((x, y))
            x_pix, y_pix = int(x_pix - self.figsize[0] * self.dpi * self.margins['left']), int(y_pix - self.figsize[1] * self.dpi * self.margins['bottom'])
            
            x_pix = int(x_pix * grid_pixel_ratio)
            y_pix = int(y_pix * grid_pixel_ratio)
            
            self.pixel_matrix[y_pix, x_pix].append(idx)

            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
    
            
        # -----------------------------------------------------------------------------------------------
        # Calculate the overlap degree metric based on the pixel matrix
        # Initialize a list to store elements with more than two numbers
        elements_with_more_than_two = []

        # Loop through each element in the matrix
        for row in self.pixel_matrix:
            for lst in row:
                if len(lst) > 2:
                    elements_with_more_than_two.extend(lst)
                    
            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return

        # Calculate the number of elements in the list
        num_elements = len(elements_with_more_than_two)
  
        # Step 3: Calculate the overlap degree based on the number of pixels with multiple data points
        self.result = num_elements / len(self.data)
        print(f'The overlap degree metric result is: {self.result:.2f}')
        # ======================================================================================================
        end_time = time.time()
        self.calculation_time = end_time - start_time
        print(f"Time taken for grid_based_density_overlap_degree: {self.calculation_time:.2f} seconds")
    
    # def pairwise_bounding_box_based_overlap_degree(self):
    #     start_time = time.time()
    #     '''
    #     calculate the pairwise distance based overlap degree metric
    #     It is from the paper: A Grid-Based Method for Removing Overlaps of Dimensionality Reduction Scatterplot Layouts (Testing the 
    #     degree of overlap for the expected value of random intervals)
        
    #     '''
    #     # ======================================================================================================
    #     # Initiate the pixel_matrix
    #     self.pixel_matrix = np.full((self.pixel_height + 1, self.pixel_width + 1), None)
    #     for i in range(self.pixel_height + 1):
    #         for j in range(self.pixel_width + 1):
    #             # self.pixel_color_matrix[i, j] = set()
    #             self.pixel_matrix[i, j] = []
                
    #             end_time = time.time()
    #             if end_time - start_time > 15:
    #                 self.calculation_time = 15
    #                 print(f"Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds")
    #                 self.result = None
    #                 break
                    
    #     # ======================================================================================================
        
    #     # ======================================================================================================
    #     # Set up the scatterplot and create the scatterplots
    #     self._setup_figure()
    #     # ======================================================================================================
        
    #     # ======================================================================================================
    #     # calculate the covered_pixels for each data point
    #     for i in range(len(self.data)):
    #         x = self.data[self.xvariable][i]
    #         y = self.data[self.yvariable][i]
    #         idx = self.data['ID'][i]
            
    #         x_pix, y_pix = self.ax.transData.transform((x, y))
    #         x_pix, y_pix = int(x_pix - self.figsize[0] * self.dpi * self.margins['left']), int(y_pix - self.figsize[1] * self.dpi * self.margins['bottom'])
            
    #         if self.marker == 'square':
    #             area_square(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
    #         elif self.marker == 'circle':
    #             area_circle(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
    #         elif self.marker == 'triangle':
    #             area_triangle(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
    #         elif self.marker == 'plus':
    #             area_plus(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
                
    #         end_time = time.time()
    #         if end_time - start_time > 15:
    #             self.calculation_time = 15
    #             print(f"Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds")
    #             self.result = None
    #             break
    #     # ======================================================================================================
    


    def pairwise_bounding_box_based_overlap_degree(self):
        start_time = time.time()

        '''
        Calculate the pairwise distance-based overlap degree metric.
        From: "A Grid-Based Method for Removing Overlaps of Dimensionality Reduction Scatterplot Layouts"
        '''

        # ======================================================================================================
        # Initialize the pixel_matrix
        self.pixel_matrix = np.full((self.pixel_height + 1, self.pixel_width + 1), None)
        for i in range(self.pixel_height + 1):
            for j in range(self.pixel_width + 1):
                self.pixel_matrix[i, j] = []

            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
        # ======================================================================================================

        # Set up the scatterplot figure and axis
        self._setup_figure()

        # ======================================================================================================
        # Calculate the covered pixels for each data point
        for i in self.data.index:
            x = self.data[self.xvariable][i]
            y = self.data[self.yvariable][i]
            idx = self.data['ID'][i]

            # Transform to pixel coordinates
            x_pix, y_pix = self.ax.transData.transform((x, y))
            x_pix = int(x_pix - self.figsize[0] * self.dpi * self.margins['left'])
            y_pix = int(y_pix - self.figsize[1] * self.dpi * self.margins['bottom'])

            # Call appropriate marker rendering function
            if self.marker == 'square':
                area_square(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'circle':
                area_circle(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'triangle':
                area_triangle(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'plus':
                area_plus(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)

            # Check for timeout
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during pixel mapping)")
                self.result = '-'
                return
        # ======================================================================================================

        # Continue with overlap degree computation (if any) here...

        
        
        # ======================================================================================================
        # Calculate the pixel based matrix
        pairwise_ratio = 0 # initate the pairewise ratio
        for i in range(len(self.data) - 1):             
            for j in range(i+1, len(self.data)):
                # -----------------------------------------------------------------------------------------------
                # Calculate the overlap degree metric based on the pixel matrix
                set_i = set(map(tuple, self.data.iloc[i]['covered_pixels']))
                set_j = set(map(tuple, self.data.iloc[j]['covered_pixels']))
                intersection = list(set_i & set_j)
                pairwise_ratio += len(intersection)/min(len(self.data.iloc[i]['covered_pixels']),len(self.data.iloc[j]['covered_pixels']))
                
                end_time = time.time()
                if end_time - start_time > 15:
                    self.calculation_time = 15
                    # print(f"Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds")
                    self.result = '-'
                    return
                
        if time.time() - start_time <= 15:
            # -----------------------------------------------------------------------------------------------
            # Calculate the overlap degree metric based on the pixel matrix
            self.result = np.sqrt(pairwise_ratio / (len(self.data) * (len(self.data) - 1)))
            print(f'The pairwise bounding box based overlap degree metric result is: {self.result:.2f}')
            # ======================================================================================================
            end_time = time.time()
            self.calculation_time = end_time - start_time
            print(f"Time taken for pairwise_bounding_box_based_overlap_degree: {self.calculation_time:.2f} seconds")


    def cal_covered_data_points(self):
        '''
        Calculate how many data points are totally covered by the different categoried data points in the scatterplot
        
        args:
        self.pixel_color_matrix: the pixel matrix for the scatterplot
        self.data: the data for the scatterplot (attribute covered_pixels for each data point)
        
        '''
        # ======================================================================================================
        # calculate the covered_pixels for each data point
        start_time = time.time()
        for i in self.data.index:
            # print(f'Calculating the covered pixels for data point {i}')
            # if i == 236:
            #     pass
            x = self.data[self.xvariable][i]
            y = self.data[self.yvariable][i]
            idx = self.data['ID'][i]
            
            x_pix, y_pix = self.ax.transData.transform((x, y))
            x_pix, y_pix = int(x_pix - self.figsize[0] * self.dpi * self.margins['left']), int(y_pix - self.figsize[1] * self.dpi * self.margins['bottom'])
            
            if self.marker == 'square':
                area_square(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'circle':
                area_circle(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'triangle':
                area_triangle(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
            elif self.marker == 'plus':
                area_plus(self, x_pix, y_pix, idx, self.marker_size, self.pixel_width, self.pixel_height)
        # ======================================================================================================
        
        # ======================================================================================================
        #  check occlusion after rendering all markers
        number_of_covered_data_points_different_class = 0
        number_of_covered_data_points = 0
        covered_categories = []
        for idx, datum in self.data.iterrows():
            visible_count_different_class = 0
            visible_count = 0
            
            for i in datum['covered_pixels']:
                if len(self.pixel_color_matrix[i[1], i[0]]) > 1:
                    
                    # Check if a data point in the pixel is on the topmost layer
                    # if idx not in [d['ID'] for d in self.pixel_color_matrix[i[1], i[0]][-1:]]:
                    #     visible_count += 1
                    if idx == self.pixel_color_matrix[i[1], i[0]][-1]['ID']:
                        visible_count += 1

                    # Check if a data point in the pixel is on the topmost layer
                    if self.pixel_color_matrix[i[1], i[0]][-1]['category'] == datum[self.zvariable]:
                        visible_count_different_class += 1
                        
                else:
                    visible_count += 1
                    visible_count_different_class += 1

            if visible_count_different_class == 0:
                covered_categories.append(datum[self.zvariable])
                number_of_covered_data_points_different_class += 1

            # if occluded_count_different_class <= 20:
            #     covered_categories.append(datum[self.zvariable])
            #     number_of_covered_data_points_different_class += 1
                
            if visible_count == 0:
                number_of_covered_data_points += 1
            
            # if occluded_count <= 20:
            #     number_of_covered_data_points += 1
                
                # print(f'The data point {datum["ID"]} is totally covered by the other data points')
                
            # Check for timeout after each row
            if time.time() - start_time > 15:
                self.calculation_time = 15
                # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
                self.result = '-'
                return
            
        # Calculate the number of occurrences for each category in covered_categories
        category_counts = {}
        for category in covered_categories:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

        # Convert the category_counts dictionary to the desired array format
        covered_categories_array = [{'category': category, 'number': count} for category, count in category_counts.items()]
        
        print(f'The covered categories array is: {covered_categories_array}')
        print(f'The number of data points that are totally covered by the other class data points is: {number_of_covered_data_points_different_class}')
        print(f'The number of data points that are totally covered by data points is: {number_of_covered_data_points}')
        
        return number_of_covered_data_points, number_of_covered_data_points_different_class, covered_categories_array
        # ======================================================================================================

    
    def kernel_density_estimation(self, bandwidth=0.1, gridsize=100):
        start_time = time.time()
        """
        Calculate the Kernel Density Estimation (KDE) of the scatterplot.

        Args:
        bandwidth (float): The bandwidth of the kernel.
        gridsize (int): The number of points on the grid for KDE calculation.
        """

        # ======================================================================================================
        # Extract the x and y values from the data and transform them to pixel coordinates
        x_values = []
        y_values = []
        
        start_time = time.time()
        for x, y in zip(self.data[self.xvariable], self.data[self.yvariable]):
            x_pix, y_pix = self.ax.transData.transform((x, y))
            x_values.append(x_pix)
            y_values.append(y_pix)
            
            
            
        # Convert the lists to numpy arrays
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        # ======================================================================================================

        # Create a grid over which to evaluate the KDE
        x_grid = np.linspace(x_values.min(), x_values.max(), gridsize)
        y_grid = np.linspace(y_values.min(), y_values.max(), gridsize)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        grid_coords = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

        # Perform KDE
        kde = gaussian_kde(np.vstack([x_values, y_values]), bw_method=bandwidth)
        kde_values = kde(grid_coords).reshape(gridsize, gridsize)
        

        # Check for timeout after each row
        if time.time() - start_time > 15:
            self.calculation_time = 15
            # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
            self.result = '-'
            return
        
        # calculate the final value for the KDE metric
        self.result = np.sum(kde_values)
        
        print(f'The Kernel Density Estimation (KDE) metric is: {self.result:.2f}')
        end_time = time.time()
        self.calculation_time = end_time - start_time
        print(f"Time taken for kernel_density_estimation: {self.calculation_time:.2f} seconds")
    
    def nearest_neighbor_distance(self):
        start_time = time.time()
        """
        Calculate the Nearest Neighbor Distance (NND) metric for the scatterplot.
        """

        # ======================================================================================================
        # Extract the x and y values from the data and transform them to pixel coordinates
        x_values = []
        y_values = []
        
        start_time = time.time()
        for x, y in zip(self.data[self.xvariable], self.data[self.yvariable]):
            x_pix, y_pix = self.ax.transData.transform((x, y))
            x_values.append(x_pix)
            y_values.append(y_pix)
            
        # Convert the lists to numpy arrays
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        # ======================================================================================================

        # Extract the x and y values from the data
        x_values = x_values.reshape(-1, 1)
        y_values = y_values.reshape(-1, 1)
        points = np.hstack((x_values, y_values))

        # Fit the NearestNeighbors model
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Check for timeout after each row
        if time.time() - start_time > 15:
            self.calculation_time = 15
            # print("Time taken for pairwise_bounding_box_based_overlap_degree more than 15 seconds (during initialization)")
            self.result = '-'
            return

        # The first column of distances contains the distance to the point itself (0), so we take the second column
        nearest_distances = distances[:, 1]

        # Calculate the mean nearest neighbor distance
        self.result = np.mean(nearest_distances)
        print(f'The Nearest Neighbor Distance (NND) metric is: {self.result:.2f}')
        end_time = time.time()
        self.calculation_time = end_time - start_time
        print(f"Time taken for nearest_neighbor_distance: {self.calculation_time:.2f} seconds")
        
        
        
        
        
        
        
                

    def save_figure(self, filename='figure.png'):
        """Saves the figure to a file."""
        import os

        # Extract the real extension
        ext = os.path.splitext(filename)[1]  # Get file extension
        valid_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps', '.tiff', '.webp'}

        if ext not in valid_extensions:
            print(f"Warning: '{filename}' does not have a valid image format. Adding .png by default.")
            filename += '.png'  # Default to PNG
            ext = '.png'

        # Explicitly specify format when saving
        self.fig.savefig(filename, format=ext.lstrip('.'))  # Ensure correct format
        print(f"Figure saved as {filename}")
        
    def show_figure(self):
        """Displays the figure."""
        self.fig.show()

    
    def visualize_heat_map(self, matrix, cmap='Reds'):
        """
        Visualizes the pixel matrix with a heatmap.

        Args:
        pixel_matrix (np.array): A 2D numpy array representing the pixel matrix.
        cmap (str): Colormap for the heatmap.
        """
        
        # Print the maximum value of the matrix
        # print(f"The maximum value in the matrix is: {np.nanmax(matrix)}")
        
        figsize = self.figsize
        margins = self.margins
        dpi = self.dpi

        self.heatmap, self.heatax = plt.subplots(figsize=figsize, dpi=dpi)
        # Adjust the subplots to fit the figure area.
        self.heatmap.subplots_adjust(left=margins['left'], right=margins['right'], top=margins['top'], bottom=margins['bottom'])
        
        # Create the heatmap
        # Replace None values with np.nan for proper handling by seaborn
        matrix = np.where(matrix == None, np.nan, matrix)
        
        # Create a custom colormap that starts with white for NaN values
        cmap = sns.color_palette("Reds", as_cmap=True)
        cmap.set_bad(color='white')
        
        heatmap = sns.heatmap(
            matrix, 
            cmap=cmap, 
            ax=self.heatax, 
            cbar_kws={'shrink': 0.5}, 
            xticklabels=int(matrix.shape[1]/10), 
            yticklabels=int(matrix.shape[0]/10),
            vmin=0,   # Set minimum value for the color scale
            vmax=1    # Set maximum value for the color scale
        )

        self.heatax.invert_yaxis()  # Reverse the y-axis
        # plt.title("Heatmap for Hidden Information")
        plt.xlabel("Pixel X", fontsize=18)
        plt.ylabel("Pixel Y", fontsize=18)
        
        # Set the font size of the tick labels
        self.heatax.tick_params(axis='both', which='major', labelsize=16)
        
        # Reduce the number of ticks on the x and y axes
        # x_ticks_interval = max(1, matrix.shape[1] // 3)  # Adjust the tick interval as needed
        # y_ticks_interval = max(1, matrix.shape[0] // 3)  # Adjust the tick interval as needed
        
        
        # Set exactly 5 ticks on the x and y axes
        x_ticks = np.linspace(0, matrix.shape[1] - 1, 5)  # 5 evenly spaced ticks on x-axis
        y_ticks = np.linspace(0, matrix.shape[0] - 1, 5)  # 5 evenly spaced ticks on y-axis

        # Set the ticks and labels
        self.heatax.set_xticks(x_ticks)
        self.heatax.set_xticklabels([f'{int(i)}' for i in x_ticks])  # Set corresponding labels for x-axis
        self.heatax.set_yticks(y_ticks)
        self.heatax.set_yticklabels([f'{int(i)}' for i in y_ticks])  # Set corresponding labels for y-axis
    
    
        
        # Increase the font size of the color palette
        cbar = heatmap.collections[0].colorbar
        # Reduce the number of ticks on the colorbar
        cbar.locator = plt.MaxNLocator(nbins=5)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=16)
        

        # Add a border (spines) around the entire heatmap
        for _, spine in self.heatax.spines.items():
            spine.set_visible(True)  # Ensure that the spines are visible
            spine.set_color('black')  # Set the color of the spines
            spine.set_linewidth(0.5)  # Set the width of the border lines
        
        # Finalize the rendering
        self.heatmap.canvas.draw()  # Ensure the plot is fully rendered in the canvas
    
    def save_heatmap(self, filename='heatmap.png'):
        """Saves the figure to a file."""
        self.heatmap.savefig(filename)
        print(f"Figure saved as {filename}")
        
    def show_heatmap(self):
        """Displays the figure."""
        self.heatmap.show()


    def visualize_matrix_histogram(self, matrix, bins=50):
        """
        Visualizes the histogram of the pixel matrix values.

        Args:
        matrix (np.array): A 2D numpy array representing the pixel matrix.
        bins (int): Number of bins for the histogram.
        """
        figsize = self.figsize
        margins = self.margins
        dpi = self.dpi
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # Flatten the matrix to get the values as a 1D array
        values = matrix.flatten()
        fig.subplots_adjust(left=margins['left'], right=margins['right'], top=margins['top'], bottom=margins['bottom'])
        # Create the histogram
        plt.hist(values, bins=bins, color='blue', alpha=0.7)
        plt.title("Histogram of Pixel Matrix Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        

    def visualize_importance(self, data):
        figsize = self.figsize
        margins = self.margins
        dpi = self.dpi

        self.importance_fig, self.importance_ax = plt.subplots(figsize=figsize, dpi=dpi)
        # Adjust the margins
        self.importance_fig.subplots_adjust(left=margins['left'], right=margins['right'], top=margins['top'], bottom=margins['bottom'])
        # Create the scatter plot
        scatter = self.importance_ax.scatter(data[self.xvariable], data[self.yvariable], edgecolor='none', marker='o', s=20, c=data['importance_index'], alpha=1)
        plt.colorbar(scatter, ax=self.importance_ax, label='Importance Index')  # Add a color bar to show the importance index
        self.importance_ax.set_title("Scatter Plot of Importance Index")
        self.importance_ax.set_xlabel(self.xvariable)
        self.importance_ax.set_ylabel(self.yvariable)
        self.importance_ax.grid(True)
        # Do not show the plot here
        self.importance_fig.canvas.draw()

    def show_importance_figure(self):
        """Displays the importance figure."""
        self.importance_fig.show()

    def save_importance_figure(self, filename='importance_figure.png'):
        """Saves the importance figure to a file."""
        self.importance_fig.savefig(filename)
        print(f"Importance figure saved as {filename}")
        
    
    def bar_for_importance(self, data):
        figsize = self.figsize
        margins = self.margins
        dpi = self.dpi

        self.importance_bar_distribution, self.importance_bar_distribution_ax = plt.subplots(figsize=figsize, dpi=dpi)
        # Adjust the margins
        self.importance_bar_distribution.subplots_adjust(left=margins['left'], right=margins['right'], top=margins['top'], bottom=margins['bottom'])
        # Create the histogram
        importance_values = data['importance_index']
        self.importance_bar_distribution_ax.hist(importance_values, bins=30, color='skyblue', edgecolor='black')
        self.importance_bar_distribution_ax.set_title("Distribution of Importance Index")
        self.importance_bar_distribution_ax.set_xlabel("Importance Index")
        self.importance_bar_distribution_ax.set_ylabel("Frequency")
        self.importance_bar_distribution_ax.set_yscale('log')  # Set y-axis to log scale
        self.importance_bar_distribution_ax.grid(True)
        # Do not show the plot here
        self.importance_bar_distribution.canvas.draw()

        
    def show_importance_bar(self):
        """Displays the importance figure."""
        self.importance_bar_distribution.show()
        
    def save_importance_bar(self, filename='importance_bar.png'):
        """Saves the importance figure to a file."""
        self.importance_bar_distribution.savefig(filename)
        print(f"Importance figure saved as {filename}")



    def plot_depth_complexity_histogram(self, min_value=None):
        """
        Plots a histogram showing the distribution of depth complexity based on the maximum value in the count matrix.

        Parameters:
        - min_value: Minimum value to start the histogram bins. Defaults to 0 if not provided.
        """
        # Flatten the count_matrix to get the depth complexity values
        depth_values = self.other_layer_matrix.flatten()
        
        # Convert all elements in depth_values to integers and filter out any negative values
        depth_values = depth_values.astype(int)
        depth_values = depth_values[depth_values >= 0]

        # The number of bins is set to the maximum value in the count_matrix
        max_value = int(np.max(depth_values))
        min_value = min_value if min_value is not None else 0

        # Set the bins to span each integer from min_value to the maximum value, inclusive
        # We add 1 to the max_value in arange to include the maximum value as a bin edge
        bins = np.arange(min_value, max_value + 1) - 0.5

        # Plot the histogram with the specified bins
        plt.figure(figsize=(10, 6))
        plt.hist(depth_values, bins=bins, color='skyblue', edgecolor='black', align='mid')

        # Set the title and labels
        plt.title('Depth Complexity Histogram')
        plt.xlabel('Depth Complexity Value')
        plt.ylabel('No. Pixels')

        # Set x-axis ticks to show only integer values
        plt.xticks(np.arange(min_value, max_value + 1))  # Ensure integer ticks

        # Only show non-zero bins in the histogram, and set the limit to max_value + 1 to include the last bin
        plt.xlim(min_value - 0.5, max_value + 0.5)

        # Show the plot
        plt.show()
        




    # def class_seperation(self, array1 = None, array2 = None, feature = None):

    #     # Step 1: Check for non-zero elements in both arrays
    #     non_zero1 = array1 != 0
    #     non_zero2 = array2 != 0

    #     # Step 2: Logical AND to find common non-zero positions
    #     common_non_zero = non_zero1 & non_zero2

    #     # Step 3: Count the number of True values
    #     count = np.sum(common_non_zero)
        
    #     # Count non-zero elements
    #     non_zero_count = np.count_nonzero(array2)
        
    #     print(f"How many overplooted pixels for {feature}? It is {count}/{non_zero_count}")
        
    
    
    # def cate_metric(self,analysis_type, features):
    #     if analysis_type == 'pearson':
    #         similarity_result = pearson_corr(self.data, features)
    #     elif analysis_type == 'ks_test':
    #         similarity_result = ks_test(self.data, features)
    #     else:
    #         raise ValueError("You have to select a correct analysis method.")
    
    #     optimal_order = find_shortest_route_max_start_end_distance(similarity_result, features)[0]
    #     metric = longest_common_subsequence(optimal_order, features)/len(features)
    
    #     return optimal_order, metric 
    

    # def calculate_pixel_coverage(self):
    #     """
    #     This function is to calcualte how many pixels are covered by each class among all instances and how many
    #     pixels are shared among a specific class and other classes

    #     Input: 
    #         - Scatter_metric class, but we only use self.pixel_color_matrix;

    #     Output:
    #         - A dictonary to store the count of pixels and shared pixels for each key
    #             - In each value of key, it is an array where first element is the count of pixels and second
    #             element is the count of pixels shared with others;
    #     """
    #     unique_value_counts = {}  # To store count of pixels for each unique value
    #     covered_once = 0  # To count pixels covered at least once
    #     covered_twice_or_more = 0  # To count pixels covered by at least two different classes

    #     for row in self.pixel_color_matrix:
    #         for pixel_set in row:
    #             if pixel_set is None:
    #                 continue

    #             num_values = len(pixel_set)

    #             # Update counts for pixels covered once or twice or more
    #             if num_values > 0:
    #                 covered_once += 1
    #             if num_values > 1:
    #                 covered_twice_or_more += 1   

    #             # Update counts for each unique value
    #             for value in pixel_set:
    #                 if value not in unique_value_counts:
    #                     unique_value_counts[value] = [0, 0]
    #                 unique_value_counts[value][0] += 1
    #                 if num_values > 1:
    #                     unique_value_counts[value][1] += 1

    #     print(f"Pixels covered at least once: {covered_once}")
    #     print(f"Pixels covered by at least two different classes: {covered_twice_or_more}")
    #     print(f"The quality metrix is: {(covered_once - covered_twice_or_more)/covered_once}")
    #     # print(unique_value_counts)

    #     return unique_value_counts, covered_once, covered_twice_or_more


if __name__ == "__main__":


    file_location = 'collect_scatterplots/collect_simulated_scatterplots/csv_metrics/00000.csv'
    # file_location = 'datasets/insurance.csv'
    data = load_data(file_location)  # Make sure load_data is properly defined
    analysis = Scatter_Metric(data, margins = {'left':0.2, 'right': 0.7, 'top':0.8, 'bottom': 0.2},
                            marker = 'square', 
                            marker_size = 50, 
                            dpi = 100, 
                            figsize= (10, 6),
                            # xvariable = 'bmi', 
                            # yvariable = 'charges',
                            # zvariable='smoker',
                            xvariable = 'X coordinate', 
                            yvariable = 'Y coordinate',
                            zvariable='projected_label',
                            color_map='tab10'
                            )
    analysis.importance_metric(important_cal_method = 'mahalanobis_distance', mapping_method = 'linear')
    # analysis.importance_metric(important_cal_method = 'mahalanobis_distance', mapping_method = 'linear', order_variable = 'smoker', rendering_sequence = ['no', 'yes'])
    analysis.save_figure(filename = 'Our_metrics/output_insurance_scatterplot_1.png')
    
    analysis.visilibity_index()
    
    analysis.mpix()
    
    analysis.grid_based_density_overlap_degree(grid_pixel_ratio = 0.5)
    
    analysis.kernel_density_estimation(bandwidth=0.1, gridsize=100)
    
    analysis.nearest_neighbor_distance()
    
    # analysis.pairwise_bounding_box_based_overlap_degree()
    
    analysis.cal_covered_data_points()
    
    analysis.plot_density_heatmap()



    # print(f'The final value metrix is: {analysis.result}')

    # analysis.visualize_heat_map(analysis.top_layer_matrix, cmap='Reds')
    # # analysis.save_heatmap(filename = 'output_insurance_heatmap_1.png')
    # analysis.show_heatmap()
    
    # analysis.visualize_heat_map(analysis.other_layer_matrix, cmap='Reds')
    # # analysis.save_heatmap(filename = 'output_insurance_heatmap_2.png')
    # analysis.show_heatmap()
    
    # analysis.visualize_heat_map(analysis.overall_layer_matrix, cmap='Reds')
    # # analysis.save_heatmap(filename = 'output_insurance_heatmap_3.png')
    # analysis.show_heatmap()
    
    
    # analysis.bar_for_importance(analysis.data)
    # # analysis.save_importance_bar(filename='importance_bar.png')
    # analysis.show_importance_bar()
    
    
    # analysis.visualize_importance(analysis.data)
    # # analysis.save_importance_figure(filename='importance_figure.png')
    

    # analysis.visualize_matrix_histogram(analysis.top_layer_matrix)    

    # # analysis.plot_depth_complexity_histogram()