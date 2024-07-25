import numpy as np
from scipy.optimize import leastsq

class EDGE:
    def __init__(self, contours, min_points=5, shape_cut=0.2,
                 area_cut=10., radius_dev_cut=0.5, connectivity_angle=30.):
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

    def connect_edges(self, x_proximity_threshold=100, y_proximity_threshold=10000):
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
                if (np.abs(self.edges[i]['slope_angle'] - self.edges[j]['slope_angle']) <= self.connectivity_angle and
                    self.check_center_proximity(self.edges[i], self.edges[j], x_proximity_threshold, y_proximity_threshold)):
                    self.edges[i] = self.merge_edges(self.edges[i], self.edges[j])
                    del self.edges[j]
                    len_edges -= 1
                else:
                    j += 1
            i += 1


if __name__ == '__main__':
    import pylab as pl

    # Sample contour with only one edge.
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
                         [983.,  378.96023005],
                         [983.64027253,  378.],
                         [983.91887289,  377.],
                         [983.83748505,  376.],
                         [983.,  375.16429886],
                         [982.8353446,  375.],
                         [982.83521735,  374.],
                         [983.,  373.33646663],
                         [983.08356296,  373.],
                         [983.08346613,  372.],
                         [983.05559534,  371.],
                         [983.,  370.91681365],
                         [982.08104495,  370.],
                         [982.,  369.34829976],
                         [981.8839432,  369.],
                         [981.88365281,  368.],
                         [981.,  367.67102221],
                         [980.67173695,  368.],
                         [980.67158503,  369.],
                         [981.,  369.52649409],
                         [981.78545036,  370.],
                         [982.,  370.0808727],
                         [982.91699005,  371.],
                         [982.,  371.91889354],
                         [981.94108342,  372.],
                         [981.7843259,  373.],
                         [981.,  373.16891089],
                         [980.70386539,  373.],
                         [980.,  372.29777757],
                         [979.,  372.26617816],
                         [978.19968936,  372.],
                         [978.,  371.60280545],
                         [977.,  371.80292069],
                         [976.,  371.80503356],
                         [975.61260341,  372.],
                         [976.,  372.19512255],
                         [976.4018599,  373.],
                         [976.,  373.22993289],
                         [975.,  373.61366516],
                         [974.,  373.61765716],
                         [973.44589678,  374.],
                         [973.11426113,  375.],
                         [973.03984692,  376.],
                         [973.05362787,  377.],
                         [973.,  377.4890333],
                         [972.,  377.49473241],
                         [971.68930353,  377.],
                         [971.,  376.82882956],
                         [970.65813729,  377.],
                         [971.,  377.49982635],
                         [971.68492286,  378.],
                         [972.,  378.1577691],
                         [973.,  378.15951288],
                         [973.38907309,  379.],
                         [973.72426519,  380.],
                         [973.,  380.83857605],
                         [972.83871315,  381.],
                         [973.,  381.65355224],
                         [974.,  381.64301045],
                         [975.,  381.64618349],
                         [975.17591572,  382.],
                         [976.,  382.82588456],
                         [977.,  382.82742928],
                         [977.22970369,  383.],
                         [977.07642179,  384.],
                         [977.,  384.08592729],
                         [976.08630625,  385.],
                         [976.,  385.19764976],
                         [975.29973193,  386.],
                         [976.,  386.7016911],
                         [976.29771644,  387.],
                         [977.,  387.46943243],
                         [977.52989259,  388.],
                         [978.,  388.70553523],
                         [979.,  388.70677412],
                         [979.58307735,  389.],
                         [980.,  389.27882521],
                         [980.54135193,  390.],
                         [981.,  390.45941675],
                         [981.45976149,  390.],
                         [982.,  389.69196352],
                         [982.6064398,  389.],
                         [982.,  388.3940689],
                         [981.2106587,  388.],
                         [982.,  387.21142985],
                         [983.,  387.2128304],
                         [984.,  387.47572913],
                         [984.3566691,  387.],
                         [984.,  386.28734439],
                         [983.42505279,  386.],
                         [984.,  385.42651258],
                         [985.,  385.42957929],
                         [985.2156529,  385.]]])

    edge = EDGE(contours)
    edge.quantify()
    edge.connect_edges()
    edges = edge.get_edges()
    print(edges)

    # Plot the first edge, which is the only edge in the sample data.
    # Since the contour returned from scikit-image is a list of
    # [row, columns], we need to use "1" as x and "0" as y.
    pl.plot(contours[0][::, 1], contours[0][::, 0], 'r+-')
    window_size = 10
    x_added = np.hstack([contours[0][::, 1], contours[0][::, 1][:window_size]])
    y_added = np.hstack([contours[0][::, 0], contours[0][::, 0][:window_size]])
    pl.plot(edges[0]['x_center'], edges[0]['y_center'], 'rs')

    # Plot fitting line.
    x = np.linspace(365, 395, 10)
    y = edges[0]['slope'] * x + edges[0]['intercept']
    pl.plot(x, y, 'b-')
    pl.show()
