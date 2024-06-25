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
                'box_plotted': False,
                'slope_angle': 0.,
                'cov_matrix': None,  # Add for tensor of light
                'principal_dir': None  # Add for principal direction
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
            
            # Compute covariance matrix and principal direction
            edge['cov_matrix'], edge['principal_dir'] = self.compute_cov_matrix_and_principal_dir(x, y)

    def get_shape_factor(self, x, y):
        xyxy = (x[:-1] * y[1:] - x[1:] * y[:-1])
        A = 1. / 2. * np.sum(xyxy)

        one_sixth_a = 1. / (6. * A)
        x_center = one_sixth_a * np.sum((x[:-1] + x[1:]) * xyxy)
        y_center = one_sixth_a * np.sum((y[:-1] + y[1:]) * xyxy)

        perimeter = np.sum(np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2))

        distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)

        return np.abs(A), perimeter, x_center, y_center, distances

    def compute_cov_matrix_and_principal_dir(self, x, y):
        # Center the points
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        
        # Create the covariance matrix
        cov_matrix = np.cov(x_centered, y_centered)
        
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        
        # The principal direction is the eigenvector with the largest eigenvalue
        principal_dir = eigvecs[:, np.argmax(eigvals)]
        
        return cov_matrix, principal_dir

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
    
    def check_center_proximity(self, edge1, edge2, proximity_threshold):
        distance = np.sqrt((edge1['x_center'] - edge2['x_center'])**2 +
                           (edge1['y_center'] - edge2['y_center'])**2)
        return distance <= proximity_threshold

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
            'slope_angle': 0.,
            'cov_matrix': None,
            'principal_dir': None
        }
        return merged_edge

    def connect_edges(self, proximity_threshold=500):
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
                angle_difference = np.abs(self.edges[i]['slope_angle'] - self.edges[j]['slope_angle'])
                if angle_difference <= self.connectivity_angle:
                    if self.check_center_proximity(self.edges[i], self.edges[j], proximity_threshold):
                        principal_dot = np.dot(self.edges[i]['principal_dir'], self.edges[j]['principal_dir'])
                        if np.abs(principal_dot) > 0.5:  # Check if directions are almost parallel
                            self.edges[i] = self.merge_edges(self.edges[i], self.edges[j])
                            del self.edges[j]
                            len_edges -= 1
                        else:
                            j += 1
                    else:
                        j += 1
                else:
                    j += 1
            i += 1

if __name__ == '__main__':
    import pylab as pl

    contours = np.array([[[985.2156529,  385.],
                         [985.21551898,  384.],
                         [985.,  383.78486061],
                         [984.,  383.7864848],
                         [983.21207242,  383.],
                         [983.21186237,  382.],
                         [983.,  381.83091953],
                         [982.,  381.16390331],
                         [981.93404467,  381.],
                         [982.,  380.95885879],
                         [982.96034405,  380.],
                         [982.98009999,  379.],
                         [983.,  378.98009999],
                         [983.,  378.],
                         [982.,  377.],
                         [981.,  377.],
                         [980.,  376.27694305],
                         [979.27696018,  376.],
                         [979.,  375.69899034],
                         [978.85730809,  375.],
                         [978.66888703,  374.],
                         [978.60533867,  373.],
                         [978.57569543,  372.82404834],
                         [978.57569543,  373.],
                         [978.22733739,  373.],
                         [978.,  373.22737824],
                         [977.26820563,  374.],
                         [977.,  374.3434309],
                         [976.28618803,  375.],
                         [976.,  375.59227883],
                         [975.91352882,  376.],
                         [975.9130127,  377.],
                         [975.,  377.91354762],
                         [974.91345822,  378.],
                         [974.91318255,  379.],
                         [975.,  379.15387685],
                         [976.,  380.16790882],
                         [976.08699401,  380.],
                         [977.,  379.08221136],
                         [977.12953692,  379.],
                         [978.,  378.4244201],
                         [978.46654653,  378.],
                         [979.,  377.55453658],
                         [980.,  377.57358196],
                         [980.4267619,  378.],
                         [980.5732381,  378.],
                         [981.,  378.42680567],
                         [981.3685691,  378.],
                         [981.6714309,  378.],
                         [982.,  378.32700688],
                         [983.,  378.12718606],
                         [983.66132027,  379.],
                         [984.15637254,  379.],
                         [985.,  379.76523402],
                         [985.00000548,  380.],
                         [985.00009831,  381.],
                         [985.26987438,  381.],
                         [985.2156529,  385.]]])
    test = EDGE(contours)
    test.quantify()
    test.connect_edges()

    for edge in test.get_edges():
        pl.plot(edge['x'], edge['y'])

    pl.show()

