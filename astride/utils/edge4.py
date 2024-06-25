import numpy as np
from scipy.optimize import leastsq
from math import atan2, degrees

class EDGE:
    """
    Detect edges (i.e. borders) using the input contours.

    Parameters
    ----------
    contours : array_like
        An array containing contour list.
    min_points : int, optional
        The number of minimum data points in each edge.
    shape_cut : float, optional
        An empirical shape factor cut.
    area_cut : float, optional
        An empirical area cut.
    radius_dev_cut : float, optional
        An empirical radius deviation cut.
    connectivity_angle: float, optional
        A maximum angle to connect each separated edge.
    """
    def __init__(self, contours, min_points=0, shape_cut=0.2,
                 area_cut=10., radius_dev_cut=0.5, connectivity_angle=45.):
        # Set global values.
        self.shape_cut = shape_cut
        self.area_cut = area_cut
        self.radius_dev_cut = radius_dev_cut
        self.connectivity_angle = connectivity_angle

        # Set structure.
        self.edges = []
        for i in range(len(contours)):
            # Remove unclosed contours.
            if contours[i][0][0] != contours[i][-1][0] or \
               contours[i][0][1] != contours[i][-1][1] or \
               len(contours[i]) <= min_points:
                continue

            # All variables are self-explaining except the radius_deviation
            # and the connectivity.
            # The radius_deviation is the ratio of the standard deviation
            # of the distances from the center to the radius.
            # The connectivity indicates an index of an edge likely to be
            # connected with the current edge. -1 indicates no connectivity.
            self.edges.append({
                'index': i + 1,
                # Note that the contours returned from the scikit-image
                # is a list of [row, columns]
                'x': contours[i][::, 1],
                'y': contours[i][::, 0],
                'x_center': 0., 'y_center': 0.,
                'perimeter': 0., 'area': 0.,
                'shape_factor': 0.,
                'radius_deviation': 0.,
                'slope': 0., 'intercept': 0.,
                'connectivity': -1,
                # For plotting a box surrounding the edge.
                'x_min': 0., 'x_max': 0.,
                'y_min': 0., 'y_max': 0.,
                # For plotting. To check if this edge
                # is surrounded by a box already.
                'box_plotted': False,
                'slope_angle': 0.  # Initialize slope_angle
            })

    def quantify(self):
        """Quantify shape of the contours."""
        four_pi = 4. * np.pi
        for edge in self.edges:
            # Positions
            x = edge['x']
            y = edge['y']

            A, perimeter, x_center, y_center, distances = \
                self.get_shape_factor(x, y)

            # Set values.
            edge['area'] = A
            edge['perimeter'] = perimeter
            edge['x_center'] = x_center
            edge['y_center'] = y_center
            # Circle is 1. Rectangle is 0.78. Thread-like is close to zero.
            edge['shape_factor'] = four_pi * edge['area'] / \
                                   edge['perimeter'] ** 2.

            # We assume that the radius of the edge
            # is the median value of the distances from the center.
            radius = np.median(distances)
            edge['radius_deviation'] = np.std(distances - radius) / radius

            edge['x_min'] = np.min(x)
            edge['x_max'] = np.max(x)
            edge['y_min'] = np.min(y)
            edge['y_max'] = np.max(y)

    def get_shape_factor(self, x, y):
        """
        Return values related to the shape based on x and y.

        Parameters
        ----------
        x : array_like
            An array of x coordinates.
        y : array_like
            An array of y coordinates.

        Returns
        -------
        area : float
            Area of the border.
        perimeter : float
            Perimeter of the border.
        x_center : float
            X center coordinate.
        y_center : float
            Y center coordinate.
        distances : numpy.ndarray
            Distances from the center to each border element.
        """

        # Area.
        xyxy = (x[:-1] * y[1:] - x[1:] * y[:-1])
        A = 1. / 2. * np.sum(xyxy)

        # X and Y center.
        one_sixth_a = 1. / (6. * A)
        x_center = one_sixth_a * np.sum((x[:-1] + x[1:]) * xyxy)
        y_center = one_sixth_a * np.sum((y[:-1] + y[1:]) * xyxy)

        # Perimeter.
        perimeter = np.sum(np.sqrt((x[1:] - x[:-1])**2 +
                                   (y[1:] - y[:-1])**2))

        # Distances from the center.
        distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)

        return np.abs(A), perimeter, x_center, y_center, distances

    def get_edges(self):
        return self.edges

    def filter_edges(self):
        """Remove edges unlikely to be streaks."""
        filtered_edges = []
        filtered_cnt = 1
        for edge in self.edges:
            if edge['shape_factor'] <= self.shape_cut and \
               edge['area'] >= self.area_cut and \
               edge['radius_deviation'] >= self.radius_dev_cut:
                # Reset index, incremental from 1.
                edge['index'] = filtered_cnt
                filtered_edges.append(edge)
                filtered_cnt += 1

        # Set filtered edges.
        self.edges = filtered_edges

    def residuals(self, theta, x, y):
        """
        Residual for a straight line.

        Parameters
        ----------
        theta : list of float
            Coefficients of float[2].
        x : array_like
            An array of x values.
        y : array_like
            An array of y values.

        Returns
        -------
        residual : float
            Residuals.
        """

        residu = y - (theta[0] * x + theta[1])

        return residu
    
    def check_center_proximity(self, edge1, edge2, proximity_threshold):
        """
        Check if the centers of two edges are within the proximity threshold.

        Parameters
        ----------
        edge1 : dict
            Dictionary representing the first edge.
        edge2 : dict
            Dictionary representing the second edge.
        proximity_threshold : float
            The maximum allowable distance between the centers of two edges.

        Returns
        -------
        bool
            True if the centers are within the proximity threshold, False otherwise.
        """
        distance = np.sqrt((edge1['x_center'] - edge2['x_center'])**2 +
                           (edge1['y_center'] - edge2['y_center'])**2)
        return distance <= proximity_threshold

    def angle_between(self, edge1, edge2):
        """
        Calculate the angle in degrees between two edges' center points.

        Parameters
        ----------
        edge1 : dict
            Dictionary representing the first edge.
        edge2 : dict
            Dictionary representing the second edge.

        Returns
        -------
        float
            Angle in degrees between the center points of two edges.
        """
        v1 = [edge1['x_center'], edge1['y_center']]
        v2 = [edge2['x_center'], edge2['y_center']]
        angle = degrees(atan2(v2[1] - v1[1], v2[0] - v1[0]))
        return angle

    def merge_edges(self, edge1, edge2):
        merged_edge = {
            'index': edge1['index'],
            'x': np.concatenate((edge1['x'], edge2['x'])),
            'y': np.concatenate((edge1['y'], edge2['y'])),
            'x_center': 0., 'y_center': 0.,
            'perimeter': 0., 'area': 0.,
            'shape_factor': 0.,
            'radius_deviation': 0.,
            'slope': 0., 'intercept': 0.,
            'connectivity': -1,
            'x_min': 0., 'x_max': 0.,
            'y_min': 0., 'y_max': 0.,
            'box_plotted': False,
            'slope_angle': 0.  # Initialize slope_angle
        }
        return merged_edge

    def connect_edges(self, proximity_threshold=5000, angle_range=(0, 45)):
        """Connect detected edges based on their slopes and proximity threshold within an angle range."""
        # Fitting a straight line to each edge.
        p0 = [0., 0.]
        for edge in self.edges:
            x = edge['x']
            y = edge['y']
            result = leastsq(self.residuals, p0, args=(x, y))
            edge['slope'] = result[0][0]
            edge['intercept'] = result[0][1]
            edge['slope_angle'] = degrees(atan2(edge['slope'], 1))  # Calculate slope angle

        # Loop to find and connect edges within angle range and proximity threshold.
        edge_connected = []
        for i in range(len(self.edges)):
            if i in edge_connected:
                continue
            for j in range(i + 1, len(self.edges)):
                if j in edge_connected:
                    continue
                edge_i = self.edges[i]
                edge_j = self.edges[j]
                angle_ij = abs(self.angle_between(edge_i, edge_j))
                # Check if the angle is within the specified range and the centers are within the proximity threshold.
                if self.check_center_proximity(edge_i, edge_j, proximity_threshold) and angle_range[0] <= angle_ij <= angle_range[1]:
                    merged_edge = self.merge_edges(edge_i, edge_j)
                    self.edges[i] = merged_edge
                    edge_connected.append(j)

        # Remove merged edges from the list.
        self.edges = [edge for k, edge in enumerate(self.edges) if k not in edge_connected]

