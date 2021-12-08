from __future__ import print_function
import numpy as np

from Experiments.InductionExperiment import InductionExperiment


class LEDExperiment(InductionExperiment):

    # ****************************************************************
    # * Stimulation Experiment overwritten methods
    # *****************************************************************

    def _get_stim_times(self, thresh=2):
        behavior_data = self.behaviorData()
        pin_name = 'led_context_pin' if 'led_context_pin' in behavior_data else 'pin_13' # pin_37
        start_times = behavior_data[pin_name][:, 0]
        end_times = behavior_data[pin_name][:, 1]

        bad_idx = []
        for i in xrange(1, len(start_times)):
            if (start_times[i] - start_times[i - 1]) < thresh:
                bad_idx.append(i)
        good_idx = np.array([i for i in xrange(len(start_times))
                             if i not in bad_idx])

        stacked = np.vstack((start_times[good_idx], end_times[good_idx])).T
        return stacked

    def _get_stim_duration(self):
        # TODO Take first? Why aren't these all the exact same
        return np.mean(np.diff(self.stim_times))

