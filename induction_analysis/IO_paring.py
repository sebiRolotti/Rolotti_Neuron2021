# ****************************************************************
# * IMPORTS
# *****************************************************************
import argparse
import warnings

from lab.classes.dbclasses import dbMouse


# ****************************************************************
# * GLOBALS
# *****************************************************************


# ****************************************************************
# * FUNCTIONS
# *****************************************************************

def get_experiments_from_mouse(mouse):
    """
    gets the list of input output experiments for a given mouse
    Parameters
    ----------
    mouse

    Returns
    -------

    """

    print('Getting input output experiments')

    imaging_experiments = mouse.imagingExperiments()
    imaging_experiments = [e for e in imaging_experiments if 'inputoutput' in e.get('tSeriesDirectory')]
    return imaging_experiments


def get_association_dictionary(expts):
    assoc = {}
    for e in expts:
        led_setting = assoc.get(e.LED_setting, None)
        if led_setting is None:
            print (e.LED_setting)
            assoc[int(e.LED_setting)] = e.trial_id
        else:
            warnings.warn('There are two experiments with the same LED setting power level, review experiments and '
                          're-run; keeping both experiments for the time being, second experiment will be under key {}'
                          .format(int(e.LED_setting) + 1))
            assoc[int(e.LED_setting) + 1] = e.trial_id
    return assoc


def associate_io_experiments(expts, assoc, save=False, overwrite=False):
    if not save:
        warnings.warn('Won\'t process saving process, the following association dictionary will not be saved')
        print(assoc)
        return

    for e in expts:
        if hasattr(e, 'assoc_expts') and not overwrite:
            if_save = 'You are running this with the save flag set, you will have diverging associations if other ' \
                      'experiments don\'t have a value '

            warnings.warn('There is already an associated experiments dictionary for this experiment. Re-run with '
                          'overwrite.  {}'
                          .format(if_save if save else ''))
            continue

        print('Saving association')
        e.assoc_expts = assoc
        e.save(store=True)


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m", "--mouse", type=str,
        help="Mouse experiments belong to"
    )
    arg_parser.add_argument(
        "-s", "--save", action="store_true",
        help="Set this flag if you want to save the changes made"
    )
    arg_parser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Set this flag if you want to overwrite any existing association dictionary"
    )
    return arg_parser.parse_args()


def main(mouse_name, save, overwrite):
    mouse = dbMouse(mouse_name)
    expts = get_experiments_from_mouse(mouse)
    assoc = get_association_dictionary(expts)
    associate_io_experiments(expts, assoc, save=save, overwrite=overwrite)


# ****************************************************************
# * MAIN
# *****************************************************************
if __name__ == '__main__':
    args = parse_arguments()
    main(args.mouse, args.save, args.overwrite)
