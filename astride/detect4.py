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

from astride.utils.edge4 import EDGE


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
                 area_cut=20., radius_dev_cut=0.5, connectivity_angle=30.,x_proximity_threshold=100., y_proximity_threshold=3000.,
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
        self.x_proximity_threshold = x_proximity_threshold
        self.y_proximity_threshold = y_proximity_threshold

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
        # Get background map and subtract.
        sigma_clip = SigmaClip(sigma=3., maxiters=10)
        bkg_estimator = MedianBackground()
        # bkg_estimator = MMMBackground()
        self._bkg = Background2D(self.raw_image,
                           (self.bkg_box_size, self.bkg_box_size),
                           filter_size=(3, 3),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        self.background_map = self._bkg.background
        self.image = self.raw_image - self.background_map

        self._med = self._bkg.background_median
        self._std = self._bkg.background_rms_median

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
        edge.connect_edges(self.x_proximity_threshold, self.y_proximity_threshold)

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

     
    def plot_figures(self, cut_threshold=3.):
        """
        Save figures of detected streaks.

        Parameters
        ----------
        cut_threshold: float, optional
            Pixels below cut_threshold * std are removed to visualize streaks
            more clearly.
        """
        # Ensure the output path exists.
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Plot original image.
        pl.imshow(self.raw_image, origin='lower', cmap='Greys_r')
        pl.savefig(self.output_path + 'raw_image.png')
        pl.close()

        # Plot the image.
        plot_data = self.image.copy()

        # Background subtracted image,
        # so the median value should be close to zero.
        med = 0.
        std = self._std

        plot_data[np.where(self.image > med + cut_threshold * std)] = \
            med + cut_threshold * std
        plot_data[np.where(self.image < med - cut_threshold * std)] = \
            med - cut_threshold * std
        pl.clf()
        pl.imshow(plot_data, origin='lower', cmap='gray')

        # Plot background removed image.
        if self.image is not None:
            fig = pl.figure()
            image = self.image.copy()
            image[image < cut_threshold * self._std] = np.nan
            pl.imshow(image, origin='lower', cmap='Greys_r')
            pl.xlim(0, self.raw_image.shape[1])
            pl.ylim(0, self.raw_image.shape[0])
            pl.xlabel("X")
            pl.ylabel("Y")
            pl.colorbar(fraction=0.03).set_label("Pixel Value (e/s)")
            pl.savefig(self.output_path + 'background_removed.png')
            pl.close()

            # If wcs info is available, plot with WCS
            if self.wcsinfo:
                hdu = fits.open(self.filename)[0]
                wcs = WCS(hdu.header)

                fig = pl.figure()
                ax = fig.add_subplot(111, projection=wcs)
                image[image < cut_threshold * self._std] = np.nan
                ax.imshow(image, origin='lower', cmap='Greys_r')
                ax.set_xlim(0, self.raw_image.shape[1])
                ax.set_ylim(0, self.raw_image.shape[0])
                ax.set_xlabel("RA")
                ax.set_ylabel("Dec")
                ax.coords.grid(color='white', ls='solid')
                ax.coords[0].set_major_formatter('hh:mm:ss')
                pl.savefig(self.output_path + 'background_removed_wcs.png')
                pl.close()

        # Plot streaks.
        if self.streaks is not None:
            fig = pl.figure()
            image = self.image.copy()
            image[image < cut_threshold * self._std] = np.nan
            pl.imshow(image, origin='lower', cmap='Greys_r')
            for streak in self.streaks:
                pl.plot(streak['x'], streak['y'], linewidth=2.5)
            pl.xlim(0, self.raw_image.shape[1])
            pl.ylim(0, self.raw_image.shape[0])
            pl.xlabel("X")
            pl.ylabel("Y")
            pl.colorbar(fraction=0.03).set_label("Pixel Value (e/s)")
            pl.savefig(self.output_path + 'streaks.png')
            pl.close()

        edges = self.streaks
        # Plot all contours.
        for n, edge in enumerate(edges):
            pl.plot(edge['x'], edge['y'])
            pl.text(edge['x'][0], edge['y'][1],
                    '%d' % (edge['index']), color='y', fontsize=15,
                    weight='bold')

        # Plot boxes.
        # Box margin in pixel.
        box_margin = 10
        for n, edge in enumerate(edges):
            # plot boxes around the edge.
            if not edge['box_plotted']:
                # Define the box to plot.
                xs = []
                ys = []
                self._find_box(edge['index'], edges, xs, ys)
                x_min = max(np.min(xs) - box_margin, 0)
                x_max = min(np.max(xs) + box_margin, self.image.shape[0])
                y_min = max(np.min(ys) - box_margin, 0)
                y_max = min(np.max(ys) + box_margin, self.image.shape[1])
                box_x = [x_min, x_min, x_max, x_max]
                box_y = [y_min, y_max, y_max, y_min]
                # pl.fill(box_x, box_y, ls='--', fill=False, ec='r', lw=2)
                edge['box_plotted'] = True

        pl.xlabel('X/pixel')
        pl.ylabel('Y/pixel')
        pl.axis([0, self.image.shape[1], 0, self.image.shape[0]])
        pl.savefig('%sall.png' % self.output_path)

        # Visualise the background map
        pl.imshow(self.background_map, cmap='gray')
        pl.colorbar()
        pl.title('Background Map')
        pl.savefig('background_map.png') 

        # Plot all individual edges (connected).
        for n, edge in enumerate(edges):
            # Reset.
            edge['box_plotted'] = False

        for n, edge in enumerate(edges):
            if not edge['box_plotted']:
                # Define the box to plot.
                xs = []
                ys = []
                self._find_box(edge['index'], edges, xs, ys)
                x_min = max(np.min(xs) - box_margin, 0)
                x_max = min(np.max(xs) + box_margin, self.image.shape[0])
                y_min = max(np.min(ys) - box_margin, 0)
                y_max = min(np.max(ys) + box_margin, self.image.shape[1])
                edge['box_plotted'] = True
                pl.axis([x_min, x_max, y_min, y_max])
                pl.savefig('%s%d.png' % (self.output_path, edge['index']))

        # Clear figure.
        pl.clf()

        # Plot connected streaks with boxes.
        if self.streaks is not None:
            fig = pl.figure()
            image = self.image.copy()
            image[image < cut_threshold * self._std] = np.nan
            pl.imshow(image, origin='lower', cmap='Greys_r')

            edges = self.streaks
            for edge in edges:
                edge['box_plotted'] = False

            for edge in edges:
                if edge['box_plotted']:
                    continue

                # Plot edge.
                pl.plot(edge['x'], edge['y'], linewidth=2.5)

                # Plot box.
                xs = []
                ys = []
                self._find_box(edge['index'], edges, xs, ys)
                xs = np.array(xs).flatten()
                ys = np.array(ys).flatten()

                pl.plot([xs.min(), xs.max(), xs.max(), xs.min(), xs.min()],
                        [ys.min(), ys.min(), ys.max(), ys.max(), ys.min()],
                        color='red')

            pl.xlim(0, self.raw_image.shape[1])
            pl.ylim(0, self.raw_image.shape[0])
            pl.xlabel("X")
            pl.ylabel("Y")
            pl.colorbar(fraction=0.03).set_label("Pixel Value (e/s)")
            pl.savefig(self.output_path + 'connected_streaks_with_boxes.png')
            pl.close()

    def xy2sky(self, filename, x, y, sep=':'):
        """
        Converts physical coordinates to WCS coordinates for STDOUT.

        Parameters
        ----------
        filename: str
            FITS image file name with path.
        x: float
            x coordinate of object.
        y: float
            y coordinate of object.
        sep: float
            delimiter for HMSDMS format.

        Returns
        -------
        coord: str
            Converted string of coordinate.
        """

        try:
            header = fits.getheader(filename)
            w = WCS(header)
            astcoords_deg = w.wcs_pix2world([[x, y]], 0)
            c = coordinates.SkyCoord(astcoords_deg * u.deg,
                                             frame='icrs')

            alpha = c.to_string(style='hmsdms', sep=sep, precision=2)[0]
            delta = c.to_string(style='hmsdms', sep=sep, precision=1)[0]

            coord = '{0} {1}'.format(alpha.split(' ')[0],
                                    delta.split(' ')[1])
            return coord
        except Exception as e:
            _ = e
            pass

    def xy2sky2(self, filename, x, y):
        """
        Converts physical coordinates to WCS coordinates for STDOUT.

        Parameters
        ----------
        filename: str
            FITS image file name with path.
        x: float
            x coordinate of object.
        y: float
            y coordinate of object.
        sep: float
            delimiter for HMSDMS format.

        Returns
        -------
        astcoords: list
            a list of coordinates.
        """

        try:
            header = fits.getheader(filename)
            w = WCS(header)
            astcoords_deg = w.wcs_pix2world([[x, y]], 0)

            astcoords = coordinates.SkyCoord(
                astcoords_deg * u.deg, frame='icrs')

            return astcoords[0]

        except Exception as e:
            _ = e
            pass

    def write_outputs(self):
        """Write information of detected streaks."""

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        fp1 = open('%sstreaks.txt' % self.output_path, 'w')
        if not self.wcsinfo:
            fp1.writelines('#ID x_center y_center area perimeter shape_factor ' +
                          'radius_deviation slope_angle intercept ' +
                          'connectivity\n')
            for n, edge in enumerate(self.streaks):
                line_str = '%2d %7.2f %7.2f %6.1f %6.1f %6.3f ' + \
                           '%6.2f %5.2f %7.2f %2d\n'
                line = line_str % \
                       (
                           edge['index'], edge['x_center'], edge['y_center'],
                           edge['area'], edge['perimeter'],
                           edge['shape_factor'],
                           edge['radius_deviation'], edge['slope_angle'],
                           edge['intercept'], edge['connectivity']
                       )
                fp1.writelines(line)
        elif self.wcsinfo:
            fp1.writelines('#ID x_center y_center ra(hms) dec(dms) ra(deg) ' +
                          'dec(deg) area perimeter shape_factor ' +
                          'radius_deviation slope_angle intercept ' +
                          'connectivity\n')
            for n, edge in enumerate(self.streaks):
                line_str = '%2d %7.2f %7.2f %s %s %s %6.1f %6.1f ' + \
                           '%6.3f %6.2f %5.2f %7.2f %2d\n'
                line = line_str % \
                       (
                           edge['index'], edge['x_center'], edge['y_center'],
                           self.xy2sky(self.filename, edge['x_center'],
                                       edge['y_center']),
                           self.xy2sky2(self.filename, edge['x_center'],
                                        edge['y_center']).ra.degree,
                           self.xy2sky2(self.filename, edge['x_center'],
                                        edge['y_center']).dec.degree,
                           edge['area'], edge['perimeter'],
                           edge['shape_factor'],
                           edge['radius_deviation'], edge['slope_angle'],
                           edge['intercept'], edge['connectivity']
                       )
                fp1.writelines(line)
        fp1.close()


if __name__ == '__main__':
    import time

    streak = Streak(sys.argv[1])
    # streak = Streak('/Users/kim/Dropbox/iPythonNotebook/ASTRiDE/mgm035.fts',
    #                shape_cut=0.3, radius_dev_cut=0.4)

    start = time.time()
    streak.detect()
    end = time.time()

    streak.plot_figures()
    streak.write_outputs()

    print('%.2f seconds' % (end - start))
