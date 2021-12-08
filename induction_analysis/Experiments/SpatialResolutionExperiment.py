import numpy as np
from scipy.ndimage import center_of_mass
from scipy.signal import detrend
from lab.signal.filters import maxmin_filter

from lab.misc.auto_helpers import get_element_size_um

from ExperimentAbstractClasses import StimulationExperiment, TargetingMixin

import os


class SpatialResolutionExperiment(StimulationExperiment.StimulationExperiment, TargetingMixin.TargetingMixin):

    def __init__(self, trial_id, mark_points_path):
        super(SpatialResolutionExperiment, self).__init__(trial_id)

        self._mark_points_path = mark_points_path
        self._initialize_xml_formatter()
        self._time_average = self.imaging_dataset().time_averages[0, ..., -1]

    # ****************************************************************
    # * Targeting Mixin  overwritten methods
    # *****************************************************************

    @property
    def points(self):
        return self._xml_formatter.points

    @property
    def center_point(self):
        # Assume most repeated point is over soma
        _, idx, cnt = np.unique(self.stim_locations, return_index=True, return_counts=True, axis=0)
        center_idx = idx[np.argmax(cnt)]
        return self.stim_locations[center_idx, :]

    def dist_bins(self, thresh=5, return_dists=False):
        # Group distances and return list len of point_distances of bin indices
        dists = self.point_distances()
        # Populate these in the dumbest possible way
        bin_idx = [0]
        bin_dists = [[dists[0]]]
        for i in xrange(1, len(dists)):
            binned = False
            for bin_i in xrange(len(bin_dists)):
                if np.abs(dists[i] - np.mean(bin_dists[i])) < thresh:
                    bin_idx.append(bin_i)
                    bin_dists[i].append(dists[i])
                    binned = True
                    break

            if not binned:
                bin_dists.append([dists[i]])
                bin_idx.append(len(bin_dists) - 1)

        if return_dists:
            return bin_idx, bin_dists

        return bin_idx




    def point_distances(self):

        cp = self.center_point
        dists = [self._euclidean_distance(p, cp) * self.px_to_um for p in self.stim_locations]
        return dists

    @property
    def px_to_um(self):

        xml_path = os.path.join(self.get('tSeriesDirectory'),
                                os.path.basename(self.get('tSeriesDirectory')) + '.xml')
        out = get_element_size_um(xml_path)

        # Assume square pixels for now
        return out[0]

    @staticmethod
    def _euclidean_distance(p1, p2):
        return np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(p1, p2)]))

    def _closest_cell(self, centroids, point):
        return np.argmin([self._euclidean_distance(c, point) for c in centroids])

    @staticmethod
    def _com(roi):
        return center_of_mass(np.array(roi.mask[0].todense()))

    # def find_targeted_cell(self, label='s2p'):
    #     self._time_average = self.imaging_dataset().time_averages[0, ..., -1]

    #     rois = self.rois(label=label)
    #     centroids = [self._com(roi) for roi in rois]

    #     idx = self._closest_cell(centroids, self.center_point)
    #     return idx

    # ****************************************************************
    # * Stimulation Experiment overwritten methods
    # *****************************************************************

    def _parse_context(self):
        pass

    def _get_stim_frames(self):
        return (self._get_stim_times() / self.frame_period()).astype(int)

    def _get_stim_times(self):
        behaviour_data = self.behaviorData()
        try:
            start_times = behaviour_data['led_context_pin'][:, 0][1:]
        except KeyError:
            start_times = behaviour_data['sync_pin'][:, 0][1:]
        end_times = start_times + self._get_stim_duration()
        stacked = np.vstack((start_times, end_times)).T
        return stacked

    def _get_stim_duration(self):
        return self._xml_formatter.stim_duration

    @staticmethod
    def remove_outliers(x):

        for row in x:
            row[np.where(row < np.percentile(row, 1))] = np.nanmean(row)
        return x

    def df(self, roi_filter=None):
        sigs = self.imagingData(dFOverF=None, roi_filter=roi_filter)[..., 0]
        if sigs.shape[0] == 0:
            return sigs[..., np.newaxis]
        sigs = detrend(sigs, axis=1, type='linear')
        window = int(30 / self.frame_period())
        baseline = maxmin_filter(self.remove_outliers(sigs), window=window, sigma=5)
        sigs -= baseline

        return sigs[..., np.newaxis]

    def psth(self, pre=3, post=6):

        pre = int(pre / self.frame_period())
        post = int(post / self.frame_period())

        sig = self.df()
        stim_frames = self._get_stim_frames()

        psths = []

        for frame in stim_frames[:, 0]:
            if frame - pre > 0:
                psths.append(sig[:, frame - pre:frame + post])
            else:
                pad_width = pre - frame
                psths.append(np.pad(sig[:, :frame + post], ((0, 0), (pad_width, 0), (0, 0)), mode='constant', constant_values=np.nan))

        return np.vstack(psths)
