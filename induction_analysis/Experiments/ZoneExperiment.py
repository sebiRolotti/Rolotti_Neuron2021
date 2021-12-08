from ExperimentAbstractClasses import TargetingMixin
from Experiments import InductionExperiment

import numpy as np
from skimage import draw


class ZoneExperiment(InductionExperiment.InductionExperiment, TargetingMixin.TargetingMixin):

    def __init__(self, trial_id, mark_points_path):
        super(ZoneExperiment, self).__init__(trial_id)
        self._mark_points_path = mark_points_path
        self._initialize_xml_formatter()

    # ****************************************************************
    # * Targeting Mixin  overwritten methods
    # *****************************************************************

    @staticmethod
    def binarize_mask(roi, thres=0.2):
        mask = np.sum(roi.__array__(), axis=0)
        return mask > (np.max(mask) * thres)

    def find_targeted_cells(self, label='s2p'):
        self._time_average = self.imaging_dataset().time_averages[0, ..., -1]

        stim_mask = np.zeros(self.frame_shape()[1:3])
        for loc, width in zip(self.stim_locations, self.spiral_sizes):
            width = stim_mask.shape[0] * width
            rr, cc = draw.circle(loc[0], loc[1], int(width / 2.), stim_mask.shape)
            stim_mask[rr, cc] += 1
        stim_mask[stim_mask > 0] = 1

        targeted_cells = []

        for ri, roi in enumerate(self.rois(label=label)):
            mask = self.binarize_mask(roi).astype(float)
            roi_nnz = np.sum(mask)

            overlap = mask + stim_mask
            overlap_nnz = len(np.where(overlap == 2)[0])

            if (overlap_nnz / float(roi_nnz)) > 0.5:
                targeted_cells.append(ri)
        # TODO: Decide, do we still need the stim mask?
        return targeted_cells

    # ****************************************************************
    # * Stimulation Experiment overwritten methods
    # *****************************************************************

    def _get_stim_times(self):
        behaviour_data = self.behaviorData()
        start_times = behaviour_data['led_context_pin'][:, 0][1:]
        end_times = start_times + self._get_stim_duration()
        stacked = np.vstack((start_times, end_times)).T
        return stacked

    def _get_stim_duration(self):
        return self._xml_formatter.stim_duration
