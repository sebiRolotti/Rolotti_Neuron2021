from __future__ import print_function
import numpy as np

from Experiments.LEDExperiment import LEDExperiment
from lab.analysis.behavior_analysis import absolutePosition

# place cell stuff
from lab.classes import ExperimentGroup
import cPickle as pickle
import pandas as pd


class LEDConjunctionExperiment(LEDExperiment):

    # ****************************************************************
    # * Stimulation Experiment overwritten methods
    # *****************************************************************

    def _parse_conj_contexts(self):
        contexts = self.behaviorData()['__trial_info']['contexts']
        contexts = {k: contexts[k] for k in contexts
                    if ('tone') in k or ('odor' in k)}

        return contexts

    def _get_toneodor_positions(self, units='normalized'):

        pos = []

        pos.append(self.context_A['locations'][0] - self.context_A['radius'])
        try:
            pos.append(self.context_B['locations'][0] - self.context_B['radius'])
        except IndexError:
            pass

        if units == 'normalized':
            return [(x * 100.) / self.track_length for x in pos]

        return pos

    @property
    def A_key(self):
        return [k for k in self._contexts if k.endswith('A')][0]

    @property
    def B_key(self):
        return [k for k in self._contexts if k.endswith('B')][0]

    @property
    def _contexts(self):
        return self._parse_conj_contexts()

    @property
    def context_A(self):
        return self._contexts[self.A_key]

    @property
    def context_B(self):
        return self._contexts[self.B_key]

    def _get_A_laps(self):

        try:
            laps = self.context_A['decorators'][0]['lap_list']
        except (IndexError, KeyError):
            # All laps
            return range(max(self.laps) + 1)
        else:
            max_lap = np.max(self.laps)
            laps = [x for x in laps if x <= max_lap]

        return laps

    def _get_B_laps(self):

        try:
            laps = self.context_B['decorators'][0]['lap_list']
        except (IndexError, KeyError):
            return []

        max_lap = np.max(self.laps)
        laps = [x for x in laps if x <= max_lap]

        return laps

    def _get_A_times(self):

        return self.behaviorData()[self.A_key]

    def _get_B_times(self):

        return self.behaviorData()[self.B_key]

    # Need to know which frames were post-induction
    def _get_A_frames(self):

        return (self.behaviorData()[self.A_key] / self.frame_period()).astype(int)

    def _get_B_frames(self):

        return (self.behaviorData()[self.B_key] / self.frame_period()).astype(int)

    def _get_A_lap_frames(self):
        A_laps = self._get_A_laps()

        frame_laps = absolutePosition(self.find('trial')).astype(int)

        return np.where(np.isin(frame_laps, A_laps))[0]

    def _get_B_lap_frames(self):

        B_laps = self._get_B_laps()

        frame_laps = absolutePosition(self.find('trial')).astype(int)

        return np.where(np.isin(frame_laps, B_laps))[0]


class conjunctivePCExperimentGroup(ExperimentGroup):

    def __init__(self, experiment_list, nPositionBins=100,
                 channel='Ch2', imaging_label='suite2p', pc_label=None,
                 demixed=False, pf_subset=None, signal=None, **kwargs):

        super(conjunctivePCExperimentGroup, self).__init__(experiment_list, **kwargs)

        # Store all args as a dictionary
        self.args = {}
        self.args['nPositionBins'] = nPositionBins
        self.args['channel'] = channel
        self.args['pc_label'] = pc_label
        self.args['imaging_label'] = imaging_label
        self.args['demixed'] = demixed
        self.args['pf_subset'] = pf_subset
        self.args['signal'] = signal

        self._data, self._data_raw, self._pfs = {}, {}, {}

    def __delitem__(self, i):
        expt = self[i]
        super(conjunctivePCExperimentGroup, self).__delitem__(i)
        self._data.pop(expt, 0)
        self._data_raw.pop(expt, 0)
        self._pfs.pop(expt, 0)
        self._std.pop(expt, 0)

    def data(self, roi_filter=None, dataframe=False):
        # tuning curves by experiment

        indices = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])

            if expt not in self._data:
                try:
                    # check for existence of place_fields.pkl
                    with open(expt.placeFieldsFilePath(
                            channel=self.args['channel'],
                            signal=self.args['signal']), 'rb') as f:
                        place_fields = pickle.load(f)
                except IOError:
                    self._data[expt] = None
                    self._data_raw[expt] = None
                else:
                    demixed_key = 'demixed' if self.args['demixed'] \
                        else 'undemixed'
                    pc_label = self.args['pc_label'] \
                        if self.args['pc_label'] is not None \
                        else expt.most_recent_key(channel=self.args['channel'])
                    if self.args['pf_subset']:
                        try:
                            self._data[expt] = place_fields[
                                pc_label][demixed_key][self.args['pf_subset']][
                                'spatial_tuning_smooth']
                            self._data_raw[expt] = place_fields[
                                pc_label][demixed_key][self.args['pf_subset']][
                                'spatial_tuning']
                        except KeyError:
                            self._data[expt] = None
                            self._data_raw[expt] = None
                    else:
                        try:
                            self._data[expt] = place_fields[
                                pc_label][demixed_key][
                                'spatial_tuning_smooth']
                            self._data_raw[expt] = place_fields[
                                pc_label][demixed_key][
                                'spatial_tuning']
                        except KeyError:
                            self._data[expt] = None
                            self._data_raw[expt] = None

        return_data = [] if dataframe else {}
        if dataframe:
            rois = self.rois(
                roi_filter=roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
        for expt in self:
            try:
                expt_data = self._data[expt][indices[expt], :]
            except (TypeError, IndexError):
                expt_data = None
            if dataframe:
                assert len(rois[expt]) == len(expt_data)
                for roi, dat in zip(rois[expt], expt_data):
                    return_data.append(
                        {'expt': expt, 'roi': roi, 'value': dat})
            else:
                return_data[expt] = expt_data
        if dataframe:
            return pd.DataFrame(return_data)
        return return_data

    def data_raw(self, roi_filter=None, dataframe=False):
        self.data()
        return_data = [] if dataframe else {}
        if dataframe:
            rois = self.rois(
                roi_filter=roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
        for expt in self:
            indices = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            try:
                expt_data = self._data_raw[expt][indices, :]
            except (TypeError, IndexError):
                expt_data = None
            if dataframe:
                assert len(rois[expt]) == len(expt_data)
                for roi, dat in zip(rois[expt], expt_data):
                    return_data.append(
                        {'expt': expt, 'roi': roi, 'value': dat})
            else:
                return_data[expt] = expt_data
        if dataframe:
            return pd.DataFrame(return_data)
        return return_data

    def pfs(self, roi_filter=None):
        indices = {}
        return_data = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            if expt not in self._pfs:
                if self.data()[expt] is None:
                    self._pfs[expt] = None
                else:
                    with open(expt.placeFieldsFilePath(
                            channel=self.args['channel'],
                            signal=self.args['signal']), 'rb') as f:
                        result = pickle.load(f)
                    demixed_key = 'demixed' if self.args['demixed'] \
                        else 'undemixed'
                    pc_label = self.args['pc_label'] \
                        if self.args['pc_label'] is not None \
                        else expt.most_recent_key(channel=self.args['channel'])
                    try:
                        self._pfs[expt] = result[
                            pc_label][demixed_key][self.args['pf_subset']]['pfs']
                    except KeyError:
                        self._pfs[expt] = result[
                            pc_label][demixed_key]['pfs']
            try:
                return_data[expt] = [
                    self._pfs[expt][idx] for idx in indices[expt]]
            except (TypeError, IndexError):
                return_data[expt] = None
        return return_data
