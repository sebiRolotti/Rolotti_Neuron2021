import cPickle as pkl
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass

from sima.motion.frame_align import align_cross_correlation


class TargetedCellsFinder:
    _coordinates = None
    _ref_path = None
    _img_shape = (0, 0)

    def __init__(self, coordinates, ref_path, experiment, label='s2p'):
        self._coordinates = coordinates
        self._ref_path = ref_path
        self._experiment = experiment
        self._label = label

    def _time_avg_point_location(self, displacement, coord):
        return [displacement[i] + coord[i] * self._img_shape[i] for i in range(len(coord))]

    def _calculate_displacements(self):
        ta = self._experiment.imaging_dataset().time_averages[0, ..., -1]
        ref = np.array(Image.open(self._ref_path))

        # Note ref and target not necessarily the same size
        self._img_shape = ref.shape

        _displacements, _ = align_cross_correlation(ta[..., np.newaxis], ref[..., np.newaxis])
        return _displacements

    def _get_points(self, _displacements):
        _points = np.zeros(self._coordinates.shape)
        for i in range(len(self._coordinates)):
            _points[i] = self._time_avg_point_location(_displacements, self._coordinates[i])
        return _points

    @staticmethod
    def _euclidean_distance(p1, p2):
        return np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(p1, p2)]))

    def _closest_cell(self, centroids, point):
        return np.argmin([self._euclidean_distance(c, point) for c in centroids])

    @staticmethod
    def _com(roi):
        return center_of_mass(np.array(roi.mask[0].todense()))

    def _find_targeted_cells(self, _points):
        rois = self._experiment.rois(label=self._label)
        centroids = [self._com(roi) for roi in rois]

        idx = [self._closest_cell(centroids, point) for point in _points]
        return idx

    def find(self):
        displacements = self._calculate_displacements()
        points = self._get_points(displacements)
        return self._find_targeted_cells(points)
