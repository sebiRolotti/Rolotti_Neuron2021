import argparse
import sys
import warnings

from lab.classes.dbclasses import dbExperiment, dbExperimentSet
import cPickle as pkl
from datetime import datetime


def load_rois(expt, label='s2p'):
    with open(expt.signalsFilePath(), 'rb') as fp:
        rois = pkl.load(fp)[label]['rois']

    return rois


def save_rois(expt, rois, label='s2p'):
    with open(expt.signalsFilePath(), 'rb') as fp:
        sigs = pkl.load(fp)

    sigs[label]['rois'] = rois

    with open(expt.signalsFilePath(), 'wb') as fw:
        pkl.dump(sigs, fw)

def pair_rois(expts, save=False, label='s2p'):
    print('will process {} experiments'.format(len(expts)))
    for ind_expt in expts:
        print('processing: {}'.format(ind_expt.get('tSeriesDirectory')))
        grp = []

        try:
            assoc_expts = ind_expt.assoc_expts
        except KeyError:
            grp.append(ind_expt)
        else:
            for ki in assoc_expts:
                print('Found {} associated {} experiments'.format(len(ind_expt.assoc_expts[ki]), ki))
                for kii in ind_expt.assoc_expts[ki]:
                    grp.append(dbExperiment(ind_expt.assoc_expts[ki][kii]))

        suffix = '-' + str(abs(hash(datetime.now())))
        print("Suffix to assign: {}".format(suffix))
        for expt in grp:

            rois = load_rois(expt, label)

            for roi in rois:

                # assert(len(roi['label']) == 14)
                if len(roi['label']) > 14:
                    roi['label'] = roi['label'][:14]
                roi['label'] = roi['label'] + suffix

            if save:
                print('saving roi labels')
                save_rois(expt, rois, label)
        if not save:
            warnings.warn('If you want to save the rois with a new label, make sure to run the program with the -s flag')

def parse_arguments(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-d", "--tSeriesDirectory", type=str,nargs='+',
        default=[],
        help="tseries directories that you want to process."
    )
    arg_parser.add_argument(
        "-t", "--trial_ids", type=int, nargs='+',
        default=[],
        help ="trial ids that you want to process"
    )
    arg_parser.add_argument(
        "-s", "--save", action="store_true")
    arg_parser.add_argument(
        "-l", "--label", type=str, action="store")
    return arg_parser.parse_args(argv)


def main(argv):
    args = parse_arguments(argv)
    _dirs = args.tSeriesDirectory
    _trials = args.trial_ids
    expts = []

    # Process the lenght/enty of directories
    if len(_dirs) == 0 and len(_trials) ==0:
        warnings.warn('Need to provide trials ids or tSeriesDirectories')
    elif len(_trials) == 0:
        print('Directories given')
        for  _dir in _dirs:
            exp_set = dbExperimentSet.FetchTrials(tSeriesDirectory=_dir)
            assert len(exp_set) == 1
            expts.append(exp_set[0])
    elif len(_dirs) == 0:
        print ('Trials given')
        for trial in _trials:
            expts.append(dbExperiment(trial))
    else:
        warnings.warn('Both trials and TSeries given. WIll not process, please provide one or the other.')
        return 

    pair_rois(expts, args.save, args.label)


if __name__ == '__main__':
    main(sys.argv[1:])
