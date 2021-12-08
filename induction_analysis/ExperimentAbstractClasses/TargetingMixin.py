import abc

from lab.misc.auto_helpers import locate

from PIL import Image
import numpy as np

from Formatters.MarkpointsXMLFormatter import MarkPointsXMLFormatter
from Finders.StimulationLocationFinder import StimulationLocationFinder


class TargetingMixin:
    _xml_formatter = None
    _mark_points_path = None
    _mark_points_xml = None
    _mark_points_reference = None
    _time_average = None
    __metaclass__ = abc.ABCMeta

    @property
    def points(self):
        return self._xml_formatter.points

    @property
    def spiral_sizes(self):
        return self._xml_formatter.spirals

    @property
    def mark_points_location(self):
        return self._mark_points_path

    @property
    def mark_points_xml(self):
        if self._mark_points_xml is None:
            self._find_mark_points_xml()
        return self._mark_points_xml

    @property
    def mark_points_reference_image(self):
        if self._mark_points_reference is None:
            self._find_mark_points_reference_file()
        return self._mark_points_reference

    def _find_mark_points_xml(self):
        self._mark_points_xml = locate("MarkPoints*_MarkPoints.xml",
                                       root=self._mark_points_path).next()

    def _find_mark_points_reference_file(self):
        # Try with 16 bit, but these are sometimes empty for some reason
        # so open 8bit instead otherwise and check to make sure this isnt empty
        try:
            self._mark_points_reference = locate('MarkPoints*16bit-Reference.tif',
                                                 root=self._mark_points_path).next()
        except StopIteration:
            self._mark_points_reference = locate('MarkPoints*Ch2-8bit-Reference.tif',
                                                 root=self._mark_points_path).next()
        else:
            if np.sum(Image.open(self._mark_points_reference)) == 0:
                self._mark_points_reference = locate('MarkPoints*Ch2-8bit-Reference.tif',
                                                     root=self._mark_points_path).next()

        assert(np.sum(Image.open(self._mark_points_reference)) > 0)

    def _initialize_xml_formatter(self):
        self._xml_formatter = MarkPointsXMLFormatter(self.mark_points_xml)

    @property
    def stim_locations(self):
        return self._get_stim_locations()

    def _get_stim_locations(self):
        stimLocator = StimulationLocationFinder(self.points, self.mark_points_reference_image, self._time_average)
        return stimLocator.find()

    @staticmethod
    def find_targeted_cells(self):
        pass
