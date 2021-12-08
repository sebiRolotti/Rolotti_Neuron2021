import numpy as np
import warnings

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class MarkPointsXMLFormatter:
    # Parser_data
    _xml_tree = None
    _file_path = ''
    _root = None

    # Data to get
    _points = None
    _spirals = None
    _stim_duration = None

    def __init__(self, file_path):
        self._file_path = file_path
        self._xml_tree = ET.parse(file_path)
        self._root = self._xml_tree.getroot()

    @property
    def points(self):
        """

        Returns
        -------

        """
        if self._points is None:
            self._get_xy_coordinates()
        return self._points

    @property
    def stim_duration(self):
        """

        Returns
        -------

        """
        if self._stim_duration is None:
            self._get_stimuli_duration()
        return self._stim_duration

    @property
    def spirals(self):
        """

        Returns
        -------

        """
        if self._spirals is None:
            self._get_spiral_sizes()
        return self._spirals

    def _get_xy_coordinates(self):
        """

        Returns
        -------

        """

        coordinates = []

        for element in self._root.iter('Point'):
            coordinates.append([float(element.get('Y')), float(element.get('X'))])

        self._points = np.asarray(coordinates)

    def _get_spiral_sizes(self):
        """

        Returns
        -------

        """
        heights = []
        # Assume spiral heights and widths are always equal
        for element in self._root.iter('Point'):
            heights.append(float(element.get('SpiralHeight')))
        self._spirals = np.asarray(heights)

    def get_xy_coordinates_of(self, point_index):
        """

        Parameters
        ----------
        point_index

        Returns
        -------

        """
        if self._points is None:
            self._get_xy_coordinates()

        return self._points[point_index]

    def get_spiral_sizes_of(self, point_index):
        """

        Parameters
        ----------
        point_index

        Returns
        -------

        """
        if self._spirals is None:
            self._get_spiral_sizes()

        return self._spirals[point_index]

    def _get_stimuli_duration(self):
        """

        Calculates the stimuli duration based on the mark points data
        Assumes all Mark Points have the same structure, given that all PVMarkpoint elements have a single PVGalvoPoint.
        And each PVGalvoPoint has the same number of points.

        Where root is the MarkPointSeriesElements and has:
            > Iterations
            > IterationDelay
        Where the PVMarkpoint has:
            > Repetitions
        Where the PVGalvoPoint has:
            > Initial Delay
            > Duration
            > InterPointDelay
            > numPoints

        Then calculates the duration through:
            Iterations*(IterationDelay)+Repetitions*numPoints*(Initial Delay+Duration+InterPointDelay)/1000
        Returns
        -------

        """
        root = self._root

        # multipliers
        rep_n = float(root[0].get('Repetitions'))

        i = 0
        for el in root[0][0].iter('Point'):
            i += 1

        # add
        duration = float(root[0][0].get('Duration'))
        init_delay = float(root[0][0].get('InitialDelay'))
        intpd = float(root[0][0].get('InterPointDelay'))

        self._stim_duration = rep_n * i * (duration + init_delay + intpd) / 1000
