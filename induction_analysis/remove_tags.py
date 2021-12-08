import argparse
import warnings

from Formatters.ROITagger import ROITagger
from Helpers import Helpers
from lab.classes.dbclasses import dbExperiment


def process_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-d', '--tSeries_dir', action='store', type=str, default=None,
        help='give a directory if you want to process an specific TseriesDirectory '
    )
    arg_parser.add_argument(
        '-p', '--project_name', action='store', type=str, default=None,
        help='if you gave a Tseries directory '
    )
    arg_parser.add_argument(
        '-e', '--experiment_id', action='store', type=str, default=None,
        help='experiment id to use',
    )

    arg_parser.add_argument(
        '-t', '--tags', type=str, nargs='+', default=['stimmed', 'targeted'],
        help='tags to erase'
    )

    arg_parser.add_argument(
        '-a', '--all_expt',  action='store_true',
        help='Set this fag if you want to erased all the tags for the associated experiments'
    )
    arg_parser.add_argument(
        '-l', '--label', action='store', type=str,
        default='s2p', help='change if expecting to use a different ROI label. Default is a string with value \'s2p\''
    )
    arg_parser.add_argument(
        '-s', '--save', action='store_true',
        help='save changes as tags for rois')

    return arg_parser.parse_args()


def erase_tag(_id, tags, label, save):
    experiment = dbExperiment(_id)
    tagger = ROITagger(experiment, label)
    tagger.delete(tags)
    path = '/'.join( experiment.sima_path().split('/')[:-1])
    print ('For experiment: {}, with tseries: {}'.format(_id, path))
    print ('-----------------------------------------------------------------------------------------------')
    tagger.print_changes()
    print ('-----------------------------------------------------------------------------------------------')

    if save:
        tagger.save()
def get_experiments(expt_id):
    once = False
    assoc =  dbExperiment(expt_id).assoc_expts
    expts = []
    for condition in assoc.keys():
        try :
            expts.extend(list(assoc[condition].values()))
        except AttributeError as e:
            if not once:
                print ('Assuming a non-nested dictionary')
                once = True
            expts.append(assoc[condition])
    print('expts', expts)
    return expts

def main(directory, project_name, experiment_id, tags, label, save, all_expt):
    if directory is not None:
        if project_name is not None:
            experiment_id =Helpers.get_exp_id_from_directory(directory, project_name)
        else:
            warnings.warn(
                "Please provide both project name and Tseries directory for efficient search. The process is ending "
                "without processing "
            )
            return

    elif experiment_id is None:
        warnings.warn("Plase provide either a TseriesDirectory + project name or an experiment number")
        return

    if all_expt:
        ids = get_experiments(experiment_id)
        for id in ids:
            erase_tag(id, tags, label, save)
    else:
        erase_tag(experiment_id, tags, label, save)

    if not save:
        warnings.warn('Printed Changes will not be saved, please run the script again with the -s flag')

if __name__ == '__main__':
    args = process_args()
    main(args.tSeries_dir, args.project_name, args.experiment_id, args.tags, args.label, args.save, args.all_expt)