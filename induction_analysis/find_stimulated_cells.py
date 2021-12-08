import warnings

from Finders.StimulatedCellsFinder import StimulatedCellsFinder
from Finders.TargetedCellsFinder import TargetedCellsFinder
from Formatters.ROITagger import ROITagger
from Experiments.SingleCellExperiment import SingleCellExperiment
from Experiments.IOExperiment import IOExperiment
from Experiments.LEDExperiment import LEDExperiment
from Experiments.ZoneExperiment import ZoneExperiment

from lab.classes.dbclasses import dbExperimentSet, dbExperiment
import argparse


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
        help='MarkPoints folder'
    )

    argParser.add_argument(
        '-l', '--label', action='store', type=str,
        default='s2p', help='change if expecting to use a different ROI label. Default is a string with value \'s2p\''
    )
    argParser.add_argument(
        '-t', '--exp_type', action='store', type=int,
        default=0, help='change the value if you want to load the experiment as: 0: SingleCellExperiment, '
                        '1:ZoneExperiment, 2:LEDExperiment, 3:IOExperiment '
    )
    argParser.add_argument(
        '-s', '--save', action='store_true',
        help='save changes as tags for rois')

    return argParser.parse_args()


def _set_experiment_type(index, exp_id, mark_points_path=None):
    if index == 0:
        return SingleCellExperiment(exp_id, mark_points_path)
    elif index == 1:
        return ZoneExperiment(exp_id, mark_points_path)
    elif index == 2:
        return LEDExperiment(exp_id)
    elif index == 3:
        return IOExperiment(exp_id, mark_points_path)
    else:
        return dbExperiment(exp_id)


def _mark_experiment(experiment, label='s2p', save=False, artifact_filter=False, exclude_stim=False):
    roi_tagger = ROITagger(experiment, label=label)

    roi_tagger.delete(['targeted', 'stimmed'])

    try:
        targeted_cell_idx = experiment.find_targeted_cells(label=label)

        # tag the cells
        roi_tagger.tag(targeted_cell_idx, 'targeted')
    except Exception as e:
        print('Target marking failed, if not expected check the error bellow:')
        print(e)

    # Initialize the stimulated cell finder
    stim_finder = StimulatedCellsFinder(experiment, label=label, artifact_filter=artifact_filter, exclude_stim=exclude_stim)
    # find the stimulated cells
    stim_cell_idx = stim_finder.find()
    print 'Experiment: {}'.format(experiment.trial_id)
    print('--------Stimulated Cell Indices -----------')
    print(list(stim_cell_idx))
    print('_______Previous Indices-------------')
    old_stim_cell_idx = [i for i, r in enumerate(experiment.rois()) if 'stimmed' in r.tags]
    print(old_stim_cell_idx)
    if len(old_stim_cell_idx):
        print 'Jaccard: {}'.format(len(set(stim_cell_idx).intersection(set(old_stim_cell_idx))) / float(len(set(stim_cell_idx).union(set(old_stim_cell_idx)))))

    # tag rois to stimmed
    roi_tagger.tag(stim_cell_idx, 'stimmed')

    if save:
        roi_tagger.save()
    else:
        warnings.warn(
            "The ROI labeling will not be saved. "
            "If you want to save, re-run this process with the -s flag. "
            "At that point you won't see the changed printed "
        )
        # roi_tagger.print_changes()


def _get_exp_id_from_directory(directory, project_name):
    try:
        exp = dbExperimentSet.FetchTrials(project_name=project_name,
                                          tSeriesDirectory=directory)[0]
    except IndexError as e:
        print(e)
        print('Tseries directory isn\'t paired or doesn\'t have a .sima folder')
        return None
    return exp.trial_id


def main(directory=None, project_name=None, experiment_id=None, etype=0, mark=None, label='s2p', save=False):

    # get the mak_points xml

    if directory is not None:
        if project_name is not None:
            experiment_id = _get_exp_id_from_directory(directory, project_name)
        else:
            warnings.warn(
                "Please provide both project name and Tseries directory for"
                " efficient search. The process is ending without processing "
            )
            return

    elif experiment_id is None:
        warnings.warn("Plase provide either a TseriesDirectory + project name"
                      " or an experiment number")
        return

    experiment = _set_experiment_type(etype, experiment_id, mark)
    if etype in [0, 1, 3]:
        exclude_stim = True
    else:
        exclude_stim = True
    _mark_experiment(experiment, label=label, save=save, artifact_filter=False, exclude_stim=exclude_stim)
    return


if __name__ == '__main__':
    args = process_args()
    main(directory=args.tSeries_dir, project_name=args.project_name,
         experiment_id=args.experiment_id, etype=args.exp_type, mark=args.mark,
         label=args.label, save=args.save)
