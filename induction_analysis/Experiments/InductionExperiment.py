import warnings

import ExperimentAbstractClasses.StimulationExperiment
import lab.analysis.behavior_analysis as ba
import numpy as np

from lab.classes.dbclasses import dbExperiment


class InductionExperiment(ExperimentAbstractClasses.StimulationExperiment.StimulationExperiment):
    """
    Induction Experiment abstract class
    """

    # ****************************************************************
    # * Induction Specific Properties
    # *****************************************************************

    @property
    def stim_positions(self):
        return self._get_stim_positions()

    @property
    def laps(self):
        return self._get_laps()

    # ****************************************************************
    # * Stimulation Experiment overwritten methods
    # *****************************************************************

    def _parse_context(self):
        contexts = self.behaviorData()['__trial_info']['contexts']
        contexts = {k: contexts[k] for k in contexts
                    if k.startswith('led')}
        return contexts

    def _get_stim_frames(self):
        behavior_data = self.behaviorData(imageSync=True)
        pin_name = 'led_context_pin' if 'led_context_pin' in behavior_data else 'pin_13'
        return np.where(behavior_data[pin_name])[0]

    def _get_stim_times(self):
        """
        Get's the stimulation time for an induction method.
        Needs to be implemented specifically for each type of induction

        Returns
        -------
        arr: numpy.ndarray 2D array with start and end times
        """
        raise NotImplementedError

    def _get_stim_duration(self):
        """
        get's the duration for the stimulation time
        Needs to be implemented for specific inductions.

        Returns
        -------
        dur: float value of
        """
        raise NotImplementedError

    # ****************************************************************
    # * Induction Specific Methods
    # *****************************************************************

    def get_stimmed_laps(self):
        frames = self._get_stim_frames()
        laps = self._get_laps()
        return np.unique(laps[frames])

    def stim_lap_frames(self):

        # get the laps array by frame
        laps = self._get_laps()

        try:
            # get the lap numbers that were stimulated
            stimmed_laps = self.get_stimmed_laps()
        except KeyError as e:
            warnings.warn('Experiment is not an induction experiment, there is no stim applied.')
            return np.array([])

        # return all the frames that encompasses a lap, if the lap was stimulated
        lap_frames = [np.where(laps == lap)[0] for lap in stimmed_laps]
        lap_frames = np.sort(np.hstack(lap_frames))

        return lap_frames

    def _get_stim_positions(self, units='mm'):
        contexts = self._parse_context()
        positions = [contexts[k]['locations'][0] - contexts[k]['radius']
                     for k in contexts]

        if units == 'mm':
            return positions
        elif units == 'normalized':
            return [(x * 100.) / self.track_length for x in positions]
        else:
            print('Units not recognized')
        return None

    def _get_laps(self):
        abs_pos = ba.absolutePosition(self.find('trial'), imageSync=True)
        laps = abs_pos.astype(int)
        return laps

    def stim_filter(self, session='control_induction', negate=False):

        rois = self._get_session(session).rois()
        if not negate:
            rlabels = [r.label for r in rois if 'stimmed' in r.tags]
        else:
            rlabels = [r.label for r in rois if 'stimmed' not in r.tags]

        return lambda x: x.label in rlabels

    def _get_session(self, session):

        assoc_dict = self.get('assoc_expts', {})
        c, s = session.split('_')
        try:
            return self.__class__(assoc_dict[c][s])
        except KeyError:
            print('{} not in association dictionary {}'.format(
                session, assoc_dict))
            return None

    @property
    def session(self):
        assoc_dict = self.get('assoc_expts')
        for c in assoc_dict.keys():
            for s in assoc_dict[c].keys():
                if assoc_dict[c][s] == self.trial_id:
                    return '_'.join([c, s])