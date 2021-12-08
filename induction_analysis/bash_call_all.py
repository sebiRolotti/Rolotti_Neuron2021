import argparse
import sys
import os

import remove_tags as rm
import find_stimulated_cells as fs
import io_experiment_stimulated_cell_finder as io_fs

sys.path.insert(0, os.path.expanduser('~/analysis/automaticScripts/'))
import automaticScripts.oasis_spikes as op
import automaticScripts.analyzeSpatialTuning as st

from lab.classes.dbclasses import dbMouse


def process_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-m', '--mouse', action='store', type=str, default=None,
        help='mouse to process'
    )
    arg_parser.add_argument(
        '-l', '--label', action='store', type=str, default='s2p',
        help='change if expecting to use a different ROI label. Default is a string with value \'s2p\''
    )
    arg_parser.add_argument(
        '-P', '--project', action='store', type=str,
        default='Mohsin', help='name of the project to index'
    )
    arg_parser.add_argument(
        '-r', '--remove_tags', action='store_true',
        help='set if you want to run the remove tags script'
    )

    arg_parser.add_argument(
        '-S', '--get_spikes', action='store_true',
        help='set if you want to run the oasis spike script'
    )

    arg_parser.add_argument(
        '-F', '--place_field', action='store_true',
        help='set if you want to run the analyze spatial tuning script'
    )

    arg_parser.add_argument(
        '--stim_file', action='store', default=None,
        help='set this flag with the absolute path of a csv file where each line has the format:'
             'experiment_tseries, experiment_mark_point, experiment_type'
    )

    arg_parser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='set if you want to overwrite data from before or save the values for the removing/adding of tags'
    )

    arg_parser.add_argument(
        '-t', '--threads', action='store', default=3, type=int,
        help='change if you need other processes/threads to run. Defaults to 3. '
    )

    return arg_parser.parse_known_args()[0]


def main():
    args = process_args()
    mouse = args.mouse
    label = args.label
    project = args.project
    overwrite = args.overwrite
    threads = args.threads
    spikes = args.get_spikes
    pf = args.place_field
    remove = args.remove_tags
    stim_file = args.stim_file

    if spikes:
        print('-----------------------Extracting Oasis Spikes-----------------------------------')
        oasis_arguments = [project, '-m', mouse, '-l', label, '-g', '-r', '-mf', '3', '-b', '-1', '30', '-re', '1', '-sd', '-sp']
        if overwrite:
            oasis_arguments.append('-o')

        op.main(oasis_arguments)

    if pf:
        print('-----------------------Calculating Place Fields-----------------------------------')
        p_sql = project + '.sql'
        pf_arguments = [p_sql, '-m', mouse, '-S', '--processes', str(threads), '-l', label]
        if overwrite:
            pf_arguments.append('-o')

        st.main(pf_arguments)

    if remove:
        print('-----------------------Removing tags-----------------------------------')
        db_mouse = dbMouse(mouse)
        expts = db_mouse.imagingExperiments()
        for e in expts:
            tags = ['stimmed', 'targeted']
            if 'inputoutput' in e.get('tSeriesDirectory'):
                tags = ['Cell_A', 'Cell_B', 'Cell_C', 'Stimmed_Cell_A', 'Stimmed_Cell_B', 'Stimmed_Cell_C']

            rm.main(None, None, experiment_id=e.trial_id, tags=tags, label='s2p', save=overwrite, all_expt=False)

    if stim_file is not None:
        print('-----------------------Finding the targeted and Stimulated Cells for the experiments specified in '
              'file-----------------------------------')
        print('FILE: {}'.format(stim_file))
        with open(stim_file, 'rb') as f:
            for line in f:
                # [exp, marks, type]
                print ('processing: {}'.format(line))
                exp, mark, e_type = line.split(',')
                exp = exp.strip(' ')
                mark = mark.strip(' ')
                e_type = int(e_type)
                mark = None if mark == 'None' else mark
                if e_type == 3:
                    io_fs.main(exp, project, mark=mark, label=label, save=overwrite)
                else:
                    fs.main(exp, project, etype=e_type, mark=mark, label=label, save=overwrite)


if __name__ == '__main__':
    main()
