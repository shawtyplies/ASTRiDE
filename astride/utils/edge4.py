import numpy as np
from scipy.optimize import leastsq

class EDGE:
    def __init__(self, contours, min_points=0, shape_cut=0.2,
                 area_cut=10., radius_dev_cut=0.5, connectivity_angle=45.):
        self.shape_cut = shape_cut
        self.area_cut = area_cut
        self.radius_dev_cut = radius_dev_cut
        self.connectivity_angle = connectivity_angle

        self.edges = []
        for i in range(len(contours)):
            if contours[i][0][0] != contours[i][-1][0] or \
               contours[i][0][1] != contours[i][-1][1] or \
               len(contours[i]) <= min_points:
                continue

            self.edges.append({
                'index': i + 1,
                'x': contours[i][::, 1],
                'y': contours[i][::, 0],
                'x_center': 0., 'y_center': 0.,
                'perimeter': 0., 'area': 0.,
                'shape_factor': 0.,
                'radius_deviation': 0.,
                'slope': 0., 'intercept': 0.,
                'connectivity': -1,
                'x_min': 0., 'x_max': 0.,
                'y_min': 0., 'y_max': 0.,
                'box_plotted': False
            })

    def quantify(self):
        four_pi = 4. * np.pi
        for edge in self.edges:
            x = edge['x']
            y = edge['y']

            A, perimeter, x_center, y_center, distances = self.get_shape_factor(x, y)

            edge['area'] = A
            edge['perimeter'] = perimeter
            edge['x_center'] = x_center
            edge['y_center'] = y_center
            edge['shape_factor'] = four_pi * edge['area'] / edge['perimeter'] ** 2.

            radius = np.median(distances)
            edge['radius_deviation'] = np.std(distances - radius) / radius

            edge['x_min'] = np.min(x)
            edge['x_max'] = np.max(x)
            edge['y_min'] = np.min(y)
            edge['y_max'] = np.max(y)

    def get_shape_factor(self, x, y):
        xyxy = (x[:-1] * y[1:] - x[1:] * y[:-1])
        A = 1. / 2. * np.sum(xyxy)

        one_sixth_a = 1. / (6. * A)
        x_center = one_sixth_a * np.sum((x[:-1] + x[1:]) * xyxy)
        y_center = one_sixth_a * np.sum((y[:-1] + y[1:]) * xyxy)

        perimeter = np.sum(np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2))
        distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)

        return np.abs(A), perimeter, x_center, y_center, distances

    def get_edges(self):
        return self.edges

    def filter_edges(self):
        filtered_edges = []
        filtered_cnt = 1
        for edge in self.edges:
            if edge['shape_factor'] <= self.shape_cut and \
               edge['area'] >= self.area_cut and \
               edge['radius_deviation'] >= self.radius_dev_cut:
                edge['index'] = filtered_cnt
                filtered_edges.append(edge)
                filtered_cnt += 1

        self.edges = filtered_edges

    def residuals(self, theta, x, y):
        residu = y - (theta[0] * x + theta[1])
        return residu
    
    def check_center_proximity(self, edge1, edge2, x_proximity_threshold, y_proximity_threshold):
        x_distance = abs(edge1['x_center'] - edge2['x_center'])
        y_distance = abs(edge1['y_center'] - edge2['y_center'])
        return x_distance <= x_proximity_threshold and y_distance <= y_proximity_threshold

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
            'slope_angle': 0.
        }
        return merged_edge

    def connect_edges(self, x_proximity_threshold=100, y_proximity_threshold=1000):
        p0 = [0., 0.]
        radian2angle = 180. / np.pi
        for edge in self.edges:
            p1, s = leastsq(self.residuals, p0, args=(edge['x'][:-1], edge['y'][:-1]))
            edge['slope'] = p1[0]
            edge['intercept'] = p1[1]
            edge['slope_angle'] = np.arctan(edge['slope']) * radian2angle

        len_edges = len(self.edges)
        i = 0
        while i < len_edges - 1:
            j = i + 1
            while j < len_edges:
                if np.abs(self.edges[i]['slope_angle'] - self.edges[j]['slope_angle']) <= self.connectivity_angle:
                    if self.check_center_proximity(self.edges[i], self.edges[j], x_proximity_threshold, y_proximity_threshold):
                        self.edges[i] = self.merge_edges(self.edges[i], self.edges[j])
                        del self.edges[j]
                        len_edges -= 1
                    else:
                        j += 1
                else:
                    j += 1
            i += 1

if __name__ == '__main__':
    import pylab as pl