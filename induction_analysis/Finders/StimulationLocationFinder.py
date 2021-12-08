import cPickle as pkl
import numpy as np
from PIL import Image

from sima.misc.align import align_cross_correlation


class StimulationLocationFinder:
    _coordinates = None
    _ref_path = None
    _img_shape = (0, 0)

    def __init__(self, coordinates, ref_path, time_average):
        self._coordinates = coordinates
        self._ref_path = ref_path
        self._time_average = time_average

    def _time_avg_point_location(self, displacement, coord):
        return [displacement[i] + coord[i] * self._img_shape[i] for i in range(len(coord))]

    def _calculate_displacements(self):
        ref = Image.open(self._ref_path)

        # If 8bit, resize to original frame shape and then
        # Only take B channel so as not to grab yellow circle
        if '8bit' in self._ref_path:
            ref = ref.resize(self._time_average.shape)
            ref = np.array(ref)[..., 2]
        else:
            ref = np.array(ref)

        # Note ref and target not necessarily the same size
        self._img_shape = ref.shape

        _displacements, _ = align_cross_correlation(self._time_average[..., np.newaxis], ref[..., np.newaxis])
        return _displacements

    def _get_points(self, _displacements):
        _points = np.zeros(self._coordinates.shape)
        for i in range(len(self._coordinates)):
            _points[i] = self._time_avg_point_location(_displacements, self._coordinates[i])
        return _points

    def find(self):
        displacements = self._calculate_displacements()
        points = self._get_points(displacements)

        return points