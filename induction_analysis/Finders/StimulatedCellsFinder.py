from SignificantROIFinder import SignificantROIFinder as SigFinder


class StimulatedCellsFinder:
    """

    """
    _experiment = None
    _label = None
    _sig_roi_finder = None
    _pre_time = 1
    # Change back to 1 for LED
    _post_time = 0.5

    def __init__(self, experiment, label='s2p', pre_time=_pre_time,
                 post_time=_post_time, artifact_filter=False, exclude_stim=False):
        self._experiment = experiment
        self._label = label
        self._sig_roi_finder = SigFinder(self._experiment)
        self._pre_time = pre_time
        self._post_time = post_time
        self._artifact_filter = artifact_filter
        self._exclude_stim = exclude_stim

    def _find_stimulated_cells(self):
        """

        Returns
        -------

        """
        stim_times = self._experiment.stim_times
        rois = self._sig_roi_finder.get_significant_rois(
            stim_times, self._pre_time, self._post_time,
            exclude_stim=self._exclude_stim,
            artifact_filter=self._artifact_filter, replace=False)

        self._sig_roi_finder.save_shuffle_dictionary()
        return rois

    def find(self):
        return self._find_stimulated_cells()
