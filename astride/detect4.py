import os
import sys

import numpy as np
import pylab as pl

from skimage import measure
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from astropy import coordinates
from astropy import units as u
from astropy.wcs import WCS
from photutils import Background2D, MedianBackground
from photutils.background import MMMBackground

from astride.utils.edge2 import EDGE


def pad_image(image, pad_width=10):
    """Pad the image with a given width."""
    padded_image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)
    return padded_image, pad_width

def crop_image(image, pad_width=10):
    """Crop the image to remove padding."""
    cropped_image = image[pad_width:-pad_width, pad_width:-pad_width]
    return cropped_image

def flag_edges_at_borders(edges, image_shape, border_margin=10):
    """Flag edges that touch the border of the image."""
    for edge in edges:
        if (edge['x'].min() <= border_margin or edge['x'].max() >= image_shape[1] - border_margin or
            edge['y'].min() <= border_margin or edge['y'].max() >= image_shape[0] - border_margin):
            edge['is_border_edge'] = True
        else:
            edge['is_border_edge'] = False
    return edges


class Streak:
    """
    Detect streaks using several morphological values.

    Parameters
    ----------
    filename : str
        Fits filename.
    remove_bkg : {'constant', 'map'}, optional.
        Which method to remove image background. 'constant' uses sigma-clipped
        statistics of the image to calculate the constant background value.
        'map' derives a background map of the image.Default is 'constant'.
        If your image has varing background, use 'map'.
    bkg_box_size : int, optional
        Box size for background estimation.
    contour_threshold : float, optional
        Threshold to search contours (i.e. edges of an input image)
    min_points: int, optional
        The number of minimum data points in each edge.
    shape_cut : float, optional
        An empirical shape factor cut.
    area_cut : float, optional
        An empirical area cut.
    radius_dev_cut : float, optional
        An empirical radius deviation cut.
    connectivity_angle: float, optional
        An maximum angle to connect each separated edge.
    fully_connected: str, optional
        See skimage.measure.find_contours for details.
    output_path: str, optional
        Path to save figures and output files. If None, the input folder name
        and base filename is used as the output folder name.
    """
    def __init__(self, filename, remove_bkg='map', bkg_box_size=50,
                 contour_threshold=3., min_points=10, shape_cut=0.2,
                 area_cut=20., radius_dev_cut=0.5, connectivity_angle=30.,
                 fully_connected='high', output_path=None):
        hdulist = fits.open(filename)
        raw_image = hdulist[0].data.astype(np.float64)

        # check WCS info
        try:
            wcsinfo = hdulist[0].header["CTYPE1"]
            if wcsinfo:
                self.wcsinfo = True
                self.filename = filename
        except:
            self.wcsinfo = False

        hdulist.close()

        # Raw image.
        self.raw_image = raw_image
        # Background structure and background map
        self._bkg = None
        self.background_map = None
        # Background removed image.
        self.image = None
        # Raw edges
        self.raw_borders = None
        # Filtered edges, so streak, by their morphologies and
        # also connected (i.e. linked) by their slope.
        self.streaks = None
        # Statistics for the image data.
        self._med = None
        self._std = None

        # Other variables.
        remove_bkg_options = ('constant', 'map')
        if remove_bkg not in remove_bkg_options:
            raise RuntimeError('"remove_bkg" must be the one among: %s' %
                               ', '.join(remove_bkg_options))
        self.remove_bkg = remove_bkg
        self.bkg_box_size = bkg_box_size
        self.contour_threshold = contour_threshold

        # These variables for the edge detections and linking.
        self.min_points = min_points
        self.shape_cut = shape_cut
        self.area_cut = area_cut
        self.radius_dev_cut = radius_dev_cut
        self.connectivity_angle = connectivity_angle
        self.fully_connected = fully_connected

        # Set output path.
        if output_path is None:
            output_path = '%s' % \
                          (filename[:filename.rfind('.')])
        if output_path[-1] != '/':
            output_path += '/'
        self.output_path = output_path

        # For plotting.
        pl.rcParams['figure.figsize'] = [12, 9]

    def detect(self):
        # Pad the image
        self.raw_image, pad_width = pad_image(self.raw_image)

        # Remove background.
        if self.remove_bkg == 'map':
            self._remove_background()
        elif self.remove_bkg == 'constant':
            _mean, self._med, self._std = sigma_clipped_stats(self.raw_image)
            self.image = self.raw_image - self._med

        # Detect streaks.
        self._detect_streaks()

        # Crop the image back to the original size
        self.image = crop_image(self.image, pad_width)
        self.raw_image = crop_image(self.raw_image, pad_width)

        # Adjust the streaks coordinates
        for streak in self.streaks:
            streak['x'] = streak['x'] - pad_width
            streak['y'] = streak['y'] - pad_width

        # Filter edges at borders
        self.streaks = flag_edges_at_borders(self.streaks, self.image.shape)

    def _remove_background(self):
        # Improved background removal using a combination of estimators
        sigma_clip = SigmaClip(sigma=3., maxiters=10)
        bkg_estimator = MedianBackground()
        # Using Background2D to estimate the background
        bkg = Background2D(self.raw_image, (self.bkg_box_size, self.bkg_box_size),
                           filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        self.background_map = bkg.background
        self.image = self.raw_image - self.background_map

        self._med = bkg.background_median
        self._std = bkg.background_rms_median

    def _detect_streaks(self):
        # Find contours.
        # Returned contours is the list of [row, columns] (i.e. [y, x])
        contours = measure.find_contours(
            self.image, self._std * self.contour_threshold,
            fully_connected=self.fully_connected)

        # Quantify shapes of the contours and save them as 'edges'.
        edge = EDGE(contours, min_points=self.min_points,
                    shape_cut=self.shape_cut, area_cut=self.area_cut,
                    radius_dev_cut=self.radius_dev_cut,
                    connectivity_angle=self.connectivity_angle)
        edge.quantify()
        self.raw_borders = edge.get_edges()

        # Filter the edges, so only streak remains.
        edge.filter_edges()
        edge.connect_edges()

        # Set streaks variable.
        self.streaks = edge.get_edges()

    def _detect_sources(self):
        from photutils import DAOStarFinder

        fwhm = 3.
        detection_threshold = 3.
        daofind = DAOStarFinder(threshold=(self._med + self._std *
                                detection_threshold), fwhm=fwhm)
        sources = daofind.find_stars(self.image)
        pl.plot(sources['xcentroid'], sources['ycentroid'], 'r.')

    def _find_box(self, n, edges, xs, ys):
        """
        Connect edges by their "connectivity" values.

        Recursive function that defines a box surrounding one or more
        edges that are connected to each other.

        Parameters
        ----------
        n : int
            Index of edge currently checking.
        edges: array_like
            An array containing information of all edges.
        xs : array_like
            X min and max coordinates. (N,2) matrix.
        ys : array_like
            Y min and max coordinates. (N,2) matrix.

        Returns
        -------
        x_mins : array_like
            X min and max coordinates.
        y_mins : array_like
            Y min and max coordinates.
        """
        # Add current coordinates.
        current_edge = [edge for edge in edges if edge['index'] == n][0]
        current_edge['box_plotted'] = True
        xs.append([current_edge['x_min'], current_edge['x_max']])
        ys.append([current_edge['y_min'], current_edge['y_max']])

        # If connected with other edge.
        if current_edge['connectivity'] != -1:
            self._find_box(current_edge['connectivity'], edges, xs, ys)
        # Otherwise.
        else:
            return xs, ys
