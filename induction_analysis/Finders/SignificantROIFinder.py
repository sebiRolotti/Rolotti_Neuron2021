import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter, detrend
from lab.signal.filters import maxmin_filter

from scipy.ndimage.filters import gaussian_filter1d


try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


def remove_outliers(x):

    for row in x:
        row[np.where(row < np.percentile(row, 1))] = np.nanmean(row)
    return x

def mad_threshold(signal, s=None, n_mad=3):
    if s is not None:
        signal_filtered = signal[s==0]
    else:
        signal_filtered = signal
    thresh = np.nanmean(signal_filtered) + np.nanstd(signal_filtered) * n_mad

    return np.nanmean(signal_filtered), np.nanstd(signal_filtered) * n_mad

class SignificantROIFinder:
    _experiment = None
    _behavior_data = None
    _cue_frames = []
    _p_vals = None
    _shuffle_dict = {'parameters': {}, 'values': {}}
    _args = None

    def __init__(self, experiment, **kwargs):
        self._experiment = experiment
        self._behavior_data = self._experiment.behaviorData(imageSync=True)
        self._args = kwargs

    def save_shuffle_dictionary(self):
        path = '/'.join(self._experiment.sima_path().split('/')[:-1])
        path = path +'/shuffle_dictionary.pkl'
        print ('saving shuffle_dictionary to: {}'.format(path))
        with open(path, 'wb') as fp:
            pickle.dump(self._shuffle_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_cue_onset_frames(self, cue_times):
        """

        Parameters
        ----------
        cue_times

        Returns
        -------

        """
        cue_frames = np.asarray(cue_times) / self._experiment.frame_period()
        cue_frames = cue_frames.astype(int)
        if cue_frames[0, 0] == 0:
            cue_frames = cue_frames[1:, :]

        if cue_frames.shape[0] > 5:
            print '{} Stims in {}'.format(cue_frames.shape[0], self._experiment.get('tSeriesDirectory'))

        return cue_frames[:, 0]

    def _populate_sig_rois_dict(self, rois, values):
        for i, roi in enumerate(rois):
            self._shuffle_dict['values'][roi.label] = values[i]

    def _populate_shuffle_parameter(self, pre_time, post_time, stim_duration, recording_period,  n_shuffles, replace, trials):
        self._shuffle_dict['parameters'] = {
            'pre_time': pre_time, 'post_time': post_time, 'stim_duration': stim_duration,
            'recording_period': recording_period, 'n_shuffles': n_shuffles, 'replace': replace,
            'trials': trials
        }

    @staticmethod
    def _gen_psth(data, cue_onset_frames, pre_frames_on_set_cue, frames_post_stimulation_period, stimuli_duration_in_frames):

        data = [data[:, (frame - pre_frames_on_set_cue):
                        (frame + frames_post_stimulation_period + stimuli_duration_in_frames)]
                for frame in cue_onset_frames]

        return np.stack(data)

    def _get_false_cue_onset_frames(self, frames_pre_to_stimulation,
                                    frames_post_stimulation_period, cue_frames, stimuli_duration_in_frames):
        """

        Parameters
        ----------
        frames_pre_to_stimulation

        frames_post_stimulation_period
        cue_frames

        Returns
        -------

        """
        sig_shape = self._experiment.signal().shape
        frames = np.arange(sig_shape[1]) # get the frames indices through the shape of hte signal

        # exclude the frames that are part of the stimulation period
        del_frames = []
        for cue_frame in cue_frames:
            del_frames.extend(range(cue_frame, cue_frame + stimuli_duration_in_frames))
        del_frames = np.asarray(del_frames)
        frames = np.delete(frames, del_frames)

        # only include frames which  index is greater or equal to the number of frames previous to the stimuli
        frames = frames[np.where(frames >= frames_pre_to_stimulation)]

        # only include frames whcih index is less than the max frame minus the stimuli duration and the post period
        frames = frames[np.where(frames < (sig_shape[1] - frames_post_stimulation_period - stimuli_duration_in_frames))]
        return frames

    @staticmethod
    def _calculate_average_response(psth, frames_pre_to_stimulation, stim_duration,
                                    exclude_stim=False, artifact_filter=False):
        """

        Parameters
        ----------
        psth
        frames_pre_to_stimulation
        stimuli_duration_in_frames
        post_stimulation_period_frames

        Returns
        -------

        """

        # PSTH is n_stims x n_rois x n_frames

        # take the mean across frames
        # pre = np.mean(pre, axis=2)
        psth = np.nanmean(psth, axis=0, keepdims=True)

        # Smooth to deal with artifact
        if artifact_filter:
            buff = int(0.33 * stim_duration)
            filtered_psth = psth[..., frames_pre_to_stimulation - buff:frames_pre_to_stimulation + stim_duration + buff]
            win_size = filtered_psth.shape[2]
            if win_size % 2 == 0:
                win_size -= 1
            filtered_psth = savgol_filter(filtered_psth, win_size, 2)
            psth[..., frames_pre_to_stimulation:frames_pre_to_stimulation + stim_duration] = filtered_psth[..., buff:-1*buff]

        # get the pre and post
        # pre = psth[:, :, :frames_pre_to_stimulation]

        if exclude_stim:
            post = np.nanmean(psth[..., frames_pre_to_stimulation + stim_duration:], axis=2)
        else:
            post = np.nanmean(psth[..., frames_pre_to_stimulation:], axis=2)
        pre = np.nanmean(psth[..., :frames_pre_to_stimulation], axis=2)

        # take the difference
        diff = post - pre

        return diff

    @staticmethod
    def _calculate_response(psth, frames_pre_to_stimulation):

        # Return mean response during after stimulation for each ROI and stimulation
        post = psth[..., frames_pre_to_stimulation:]
        pre = psth[..., :frames_pre_to_stimulation]
        return np.nanmean(post, axis=2) - np.nanmean(pre, axis=2)


    def _build_null_distribution(self, signals, false_frames, num_trials,
                                 frames_post_stimulation_period, frames_pre_to_stimulation,
                                 stimuli_duration_in_frames, iterations=1000, replace=False, exclude_stim=False):
        """

        Parameters
        ----------
        false_frames
        num_trials
        frames_post_stimulation_period
        frames_pre_to_stimulation
        stimuli_duration_in_frames
        iterations

        Returns
        -------

        """
        shuffles = []

        # make iterations of bootstrapping
        for i in range(iterations):

            # do a sample with replacement
            shuffle_frames = np.random.choice(false_frames, size=num_trials, replace=replace)

            # generate teh psth list
            psth_list = [signals[:, (frame - frames_pre_to_stimulation):
                                                       (frame + frames_post_stimulation_period + stimuli_duration_in_frames)]
                         for frame in shuffle_frames]

            # np.stck, later versions don't like to stack through a generator
            shuffle_psth = np.stack(psth_list)

            # Calculate Mean Post - Pre Signals
            trial_ave_shuffle = self._calculate_average_response(shuffle_psth, frames_pre_to_stimulation,
                                                                 stimuli_duration_in_frames, exclude_stim=exclude_stim)

            # keep track of the trial_average
            shuffles.append(trial_ave_shuffle.squeeze())

        # convert to array
        shuffles = np.array(shuffles)

        # get the shuffle transpose, distribution
        return shuffles.transpose()

    @staticmethod
    def _min_responsive_indices(_trial_responses, min_trials=2):
        responsive = _trial_responses > 0
        return np.where(np.sum(responsive, axis=0) >= min_trials)[0]

    @staticmethod
    def _significant_rois_indices(pcts, alpha=0.05):
        """

        Parameters
        ----------
        pcts
        alpha

        Returns
        -------

        """
        significant = pcts >= (1 - alpha)
        return np.where(significant)[1]
        # return np.where(np.sum(significant, axis=0) >= min_trials)[0]

    @staticmethod
    def _percentile(trial_averages, shuffle_distribution):
        """

        Parameters
        ----------
        trial_average
        shuffle_distribution

        Returns
        -------

        """

        pcts = np.zeros(trial_averages.shape)
        for i, trial in enumerate(trial_averages):
            pcts[i, :] = np.asarray(map(stats.percentileofscore,
                                        shuffle_distribution, trial))
        return pcts / 100.

    def get_significant_rois(self, cue_times, pre_time, post_time,
                             exclude_stim=False, iterations=1000,
                             replace=False, artifact_filter=False):
        """

        Parameters
        ----------
        replace
        iterations
        cue_times
        pre_time
        post_time

        Returns
        -------

        """

        # get the period for the experiment
        period = self._experiment.frame_period()

        # number of frames previous to the stimulation
        frames_pre_to_stimulation\
            = int(pre_time / period)

        # number of frames post stimulation
        frames_post_stimulation_period = int(post_time / period)

        # duration of the stimuli in frames
        stimulation_duration_in_frames = int(self._experiment.stim_duration / period)

        # the frames that are are cue onset
        cue_frames = self._get_cue_onset_frames(cue_times)

        # generate the psth for the range of frames:
        #[pre_frames : post_frames]
        signal = self._experiment.signal(signal_type='dff',
                                         label=self._args.get('label', None),
                                         channel=self._args.get('channel', 'Ch2'))[..., 0]

        # Calculate PSTHs
        psth = self._gen_psth(
            signal, cue_frames, frames_pre_to_stimulation,
            frames_post_stimulation_period, stimulation_duration_in_frames)

        # calculate the trial responses for all ROIs
        _trial_average = self._calculate_average_response(
            psth, frames_pre_to_stimulation, stimulation_duration_in_frames,
            exclude_stim=exclude_stim, artifact_filter=artifact_filter)

        # _trial_average = np.nanmean(_trial_responses, axis=0, keepdims=True)

        # find all the frames to include on the shuffle
        false_frames = self._get_false_cue_onset_frames(
            frames_pre_to_stimulation, frames_post_stimulation_period,
            cue_frames, stimulation_duration_in_frames)

        # get the number of trials, based on the psth
        num_trials = psth.shape[0]

        # generate a null distribution
        null_distribution = self._build_null_distribution(signal,
            false_frames, num_trials, frames_post_stimulation_period,
            frames_pre_to_stimulation, stimulation_duration_in_frames,
            iterations=iterations, replace=replace, exclude_stim=exclude_stim)

        # calculate the percentile of each trial response
        # relative to the null distribution of mean responses

        self._p_vals = self._percentile(_trial_average, null_distribution)

        # populate the values for the dictionary
        self._populate_sig_rois_dict(self._experiment.rois(label=self._args.get('label', None)), self._p_vals.T)

        # populate the different parameter for the shuffle
        self._populate_shuffle_parameter(pre_time, post_time, self._experiment.stim_duration,
                                         period, iterations, replace, num_trials)

        # get the significant ROI indexes based on the p_values
        raw_significant_indexes = self._significant_rois_indices(self._p_vals)

        index_dict = {'.05': raw_significant_indexes}
        index_dict['.01'] = self._significant_rois_indices(self._p_vals, alpha=0.01)

        spikes = self._experiment.spikes(label=self._args.get('label', None),
                                             channel=self._args.get('channel', 'Ch2'))

        mean_thresh = []
        consistency_thresh = []
        means = []
        # first_point_thresh = []

        if exclude_stim:
            for cell_signal, cell_spike in zip(signal, spikes):
                _, thresh = mad_threshold(cell_signal, cell_spike, n_mad=1.5)
                mean_thresh.append(thresh)
                mean, thresh = mad_threshold(cell_signal, cell_spike, n_mad=1)
                means.append(mean)
                consistency_thresh.append(thresh)
        else:
            for cell_signal, cell_spike in zip(signal, spikes):
                mean_thresh.append(mad_threshold(cell_signal, cell_spike, n_mad=4))
                consistency_thresh.append(mad_threshold(cell_signal, cell_spike, n_mad=1))

        if exclude_stim:
            relevant_psth = psth[..., frames_pre_to_stimulation + stimulation_duration_in_frames:]

            pre_psth = psth[..., :frames_pre_to_stimulation]
        else:
            relevant_psth = psth[..., frames_pre_to_stimulation:frames_pre_to_stimulation+stimulation_duration_in_frames]

        indexes = [i for i in raw_significant_indexes if
                   np.sum([np.nanmean(stim_response) - np.nanmean(pre_response) > consistency_thresh[i] for
                           stim_response, pre_response in zip(relevant_psth[:, i, :], pre_psth[:, i, :])]) >= 3]

        mean_psths = np.nanmean(relevant_psth, axis=0)

        indexes = [i for i in indexes if np.nanmean(mean_psths[i, :]) > means[i] + mean_thresh[i]]

        path = self._experiment.sima_path() + '/index_dict.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(index_dict, fp)

        return indexes
