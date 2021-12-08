import abc
import Exceptions

from lab.classes.dbclasses import dbExperiment


class StimulationExperiment(dbExperiment):
    __metaclass__ = abc.ABCMeta
    _signals = None

    def __init__(self, trial_id):

        super(StimulationExperiment, self).__init__(trial_id)



    @property
    def stim_times(self):
        return self._get_stim_times()

    @property
    def parsed_contexts(self):
        if not hasattr(self, 'parsed_contexts'):
            raise Exceptions.ContextNoProcessedException
        else:
            return True

    @property
    def tagged(self):
        if not hasattr(self, 'tagged'):
            raise Exceptions.ExperimentNotTagged
        else:
            return True

    @property
    def stim_frames(self):
        return self._get_stim_frames()

    @property
    def stim_duration(self):
        return self._get_stim_duration()

    @tagged.setter
    def tagged(self, tagged):
        self.tagged = tagged

    # ****************************************************************
    # * Absract Methods
    # *****************************************************************

    @abc.abstractmethod
    def _parse_context(self):
        pass

    @abc.abstractmethod
    def _get_stim_frames(self):
        pass

    @abc.abstractmethod
    def _get_stim_times(self):
        pass

    @abc.abstractmethod
    def _get_stim_duration(self):
        pass

    # ****************************************************************
    # * Stimulation Specific Methods
    # *****************************************************************

    def signal(self, signal_type=None, label='s2p', channel='Ch2'):
        if (self._signals is None) or (signal_type is not None):
            self._resolve_signals(signal_type=signal_type, label=label, channel=channel)
        return self._signals

    def _resolve_signals(self, signal_type='spikes', label=None, channel='Ch2'):
        if signal_type == 'raw':
            self._signals = self.imagingData(dFOverF=None, label=label, channel=channel)[..., 0]
        elif signal_type == 'spikes':
            self._signals = self.spikes(label=label, channel=channel)
        elif signal_type == 'binary_spikes':
            self._signals = self.spikes(label=label, channel=channel, binary=True)
        else:
            self._signals = self.imagingData(dFOverF='from_file', label=label, channel=channel)

    @property
    def stim_session(self):
        return self.session.split('_')[0] + '_induction'
