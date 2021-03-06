import numpy as np
from scipy.ndimage import center_of_mass

from ExperimentAbstractClasses import StimulationExperiment, TargetingMixin


class IOExperiment(StimulationExperiment.StimulationExperiment, TargetingMixin.TargetingMixin):

    def __init__(self, trial_id, mark_points_path):
        super(IOExperiment, self).__init__(trial_id)

        self._mark_points_path = mark_points_path
        self._initialize_xml_formatter()

    # ****************************************************************
    # * Targeting Mixin  overwritten methods
    # *****************************************************************

    @property
    def points(self):
        return self._xml_formatter.points[:3]

    @staticmethod
    def _euclidean_distance(p1, p2):
        return np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(p1, p2)]))

    def _closest_cell(self, centroids, point):
        return np.argmin([self._euclidean_distance(c, point) for c in centroids])

    @staticmethod
    def _com(roi):
        return center_of_mass(np.array(roi.mask[0].todense()))

    def find_targeted_cells(self, label='s2p'):
        self._time_average = self.imaging_dataset().time_averages[0, ..., -1]

        rois = self.rois(label=label)
        centroids = [self._com(roi) for roi in rois]

        idx = [self._closest_cell(centroids, point) for point in self.stim_locations]
        return idx

    # ****************************************************************
    # * Stimulation Experiment overwritten methods
    # *****************************************************************

    def _parse_context(self):
        pass

    def _get_stim_frames(self):
        pass

    def _get_stim_times(self):
        behaviour_data = self.behaviorData()
        start_times = behaviour_data['led_context_pin'][:, 0][1:]
        end_times = start_times + self._get_stim_duration()
        stacked = np.vstack((start_times, end_times)).T
        return stacked

    def _get_stim_duration(self):
        return self._xml_formatter.stim_duration
