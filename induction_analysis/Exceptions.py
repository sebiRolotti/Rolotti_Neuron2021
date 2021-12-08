class ContextNoProcessedException(Exception):
    _message = "Context has not been properly parsed for this stimulation experiment, please parse before " \
               "attempting any analysis "

    def __str__(self):
        return repr(self._message)


class ExperimentNotTagged(Exception):
    _message = "Induction Experiment has not have its ROI's tagged, please make sure to tagg before attempting any " \
               "analysis "

    def __str__(self):
        return repr(self._message)


class NoMarkPointsAssociated(Exception):
    _message = "The experiment has no markpoint associated with it, please make sure to add the attribute"

    def __str__(self):
        return repr(self._message)
