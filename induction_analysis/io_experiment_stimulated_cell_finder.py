try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

import argparse
import warnings


from lab.classes.dbclasses import dbExperimentSet

from Finders.SignificantROIFinder import SignificantROIFinder
from Finders.TargetedCellsFinder import TargetedCellsFinder
from Formatters.ROITagger import ROITagger
from Experiments.IOExperiment import IOExperiment


def process_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '-d', '--tSeries_dir', action='store', type=str, default=None,
        help='give a directory if you want to process an specific TseriesDirectory '
    )
    argParser.add_argument(
        '-p', '--project_name', action='store', type=str, default=None,
        help='if you gave a Tseries directory '
    )
    argParser.add_argument(
        '-e', '--experiment_id', action='store', type=str, default=None,
        help='experiment id to use',
    )
    argParser.add_argument(
        '-m', '--mark', action='store', type=str, default=None,
        help='MarkPoints folder', required=True
    )

    argParser.add_argument(
        '-l', '--label', action='store', type=str,
        default='s2p', help='change if expecting to use a different ROI label. Default is a string with value \'s2p\''
    )
    argParser.add_argument(
        '-s', '--save', action='store_true',
        help='save changes as tags for rois')

    return argParser.parse_args()


def _get_exp_id_from_directory(directory, project_name):
    try:
        exp = dbExperimentSet.FetchTrials(project_name=project_name, tSeriesDirectory=directory)[0]
    except IndexError as e:
        print(e)
        print('T series directory is not paired or doesn\'t have a .sima folder')
        return None
    return exp.trial_id


def _mark_experiment(experiment, label='s2p', save=False):
    """
    process the experiment and tag it with the respective
    TODO: Make th enumber of cell programmatically
    Parameters
    ----------
    experiment
    label
    save

    Returns
    -------

    """
    roi_tagger = ROITagger(experiment, label=label)

    print ('processing for {} with Tseries: {}'.format(experiment.trial_id, experiment.sima_path()))


    print(experiment.points)

    # targeted cell
    targeted_cell_idx = experiment.find_targeted_cells()

    # cell tags:
    cell_tags = ['Cell_A', 'Cell_B', 'Cell_C']

    # tag the cells
    roi_tagger.tag(targeted_cell_idx, cell_tags )

    # get the stimuli times
    stim_times = experiment.stim_times

    # Cell A, Cell B, Cell C
    cells = (stim_times[2::3], stim_times[::3], stim_times[1::3] )

    ROI_finder = SignificantROIFinder(experiment)


    for i, cell in enumerate(cells):
        # find the stimulated cells]
        stim_cell_idx = ROI_finder.get_significant_rois(cell, pre_time=1, post_time=0.5, exclude_stim=True)

        print('--------For cells stimulated with {} -------------'.format(cell_tags[i]))
        print (stim_cell_idx)
        print ('Percentage: {}'
               .format(len(stim_cell_idx)/float(experiment.signal(label=label, channel='Ch2').shape[0]) * 100))

        # tag rois to stimmed
        roi_tagger.tag(stim_cell_idx, 'stimmed_{}'.format(cell_tags[i]))

    if save:
        roi_tagger.save()
    else:
        warnings.warn(
            "The ROI labeling will not be save, if you want to save the labeling re-run this process with the -s "
            "flag. At that point you won't see the changed printed "
        )
        roi_tagger.print_changes()


def main(directory=None, project_name=None, experiment_id=None, mark=None, label='s2p', save=False):
    # get the mak_points xml

    if directory is not None:
        if project_name is not None:
            experiment_id = _get_exp_id_from_directory(directory, project_name)
        else:
            warnings.warn(
                "Please provide both project name and Tseries directory for efficient search. The process is ending "
                "without processing "
            )
            return

    elif experiment_id is None:
        warnings.warn("Plase provide either a TseriesDirectory + project name or an experiment number")
        return

    experiment = IOExperiment(experiment_id, mark)
    _mark_experiment(experiment, label=label, save=save)
    return


if __name__ == '__main__':
    args = process_args()
    main(directory=args.tSeries_dir, project_name=args.project_name,
         experiment_id=args.experiment_id, mark=args.mark,
         label=args.label, save=args.save)
