import sys
import argparse
import warnings
from collections import OrderedDict

from lab.classes.dbclasses import dbMouse

conditions = ['control', 'noblank', 'cno']


def construct_session_type_dictionary():
    # 24hr_pre : 24hr_post
    # List of Possible Experiment Types
    sessions = ['baseline', 'pre', 'induction', 'post', '24h', '48h', '72h']

    # make sure they have a map
    session_kwords = OrderedDict({x: x for x in sessions})

    # update map for some of the differences with Mohsin's
    session_kwords.update({'base': 'baseline', 'induce': 'induction', '24hr_pre': '24hr_post', '24hr_base': '24h'})
    return session_kwords


def parse_arguments(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m", "--mouse", type=str,
        help="Mouse experiments belong to")
    arg_parser.add_argument(
        "-k", "--keywords", action="store", nargs='+', type=str, default=[''],
        help="List of keyword substrings to find in tseries in which to match")
    arg_parser.add_argument(
        "-s", "--save", action="store_true")
    return arg_parser.parse_args(argv)


def get_condition(tdir):
    if 'cno' in tdir:
        condition = 'cno'
    elif 'noblank' in tdir:
        condition = 'noblank'
    else:
        condition = 'control'
    return condition


def initialize_pairing_dictionaries():
    assoc_dict = {}
    print_dict = {}

    for condition in conditions:
        assoc_dict[condition] = {}
        print_dict[condition] = {}
    return assoc_dict, print_dict


def matches_other_session(kywords, t_dr):
    print (kywords)
    print (t_dr)
    matches = [x for x in kywords if x in t_dr]
    if len(matches) == 1:
        return True, matches[0]
    elif len(matches) > 1:
        warnings.warn("the t_series directory {} matches more than one other possible session: {}"
                      .format(t_dr, matches))
    return False, None


def get_keys_by_value(dict_of_elements, value_to_find):
    list_of_items = dict_of_elements.items()
    list_of_keys = [item[0] for item in list_of_items if item[1] == value_to_find]
    return list_of_keys

def _attempt_to_match(expt, tdir, session_kwords, sessions_lists, condition, assoc_dict, print_dict, start):
    # if we reach the end of the session kwords dictionary, return False, since a match was not possible
    if start == len(session_kwords):
        return False

    # other wise, get the session name for the moment that part of the list that we are at.
    session = session_kwords[sessions_lists[start]]

    # if the session name is in the tdir
    if sessions_lists[start] in tdir:
        # get the value
        value = assoc_dict[condition].get(sessions_lists[start], None)

        if value is None:  # if the value is None, then associate the respective values
            assoc_dict[condition][session] = expt.trial_id
            print_dict[condition][session] = expt.get('tSeriesDirectory')

            return True  # return true since a match was possible

    # Otherwise move up on the list to try and get a match.
    return _attempt_to_match(expt, tdir, session_kwords,
                             sessions_lists, condition,
                             assoc_dict, print_dict, start + 1)


def main(argv):
    # parse arguments
    args = parse_arguments(argv)

    # get teh mouse
    mouse = dbMouse(args.mouse)

    # get all its experiments
    all_expts = mouse.findall('experiment')
    # import pudb; pudb.set_trace()
    all_expts = sorted(all_expts, key=lambda x: x.get('tSeriesDirectory'), reverse=True)
    print all_expts
    # Checks that there is a tSeriesDirectory attached
    # and that any of the keywords are on the string
    matched_experiments = [e for e in all_expts if e.get('tSeriesDirectory') and
                           any([x in e.get('tSeriesDirectory') for x in args.keywords])]

    # get the session keywords.
    session_kwords = construct_session_type_dictionary()

    # initialize the associated and print dictionary
    assoc_dict, print_dict = initialize_pairing_dictionaries()

    # holds the experiments that don't have a type
    untyped_expts = []

    # gets the session keywords as a list
    sessions_list = list(session_kwords.keys())

    # loop through the experiments
    for expt in matched_experiments:
        # assume is always true
        typed = True

        # Try to figure out what session type it is
        tdir = expt.get('tSeriesDirectory').lower()

        # get's the corresponding condition
        condition = get_condition(tdir)

        # attempt to match
        if not _attempt_to_match(expt, tdir, session_kwords, sessions_list, condition, assoc_dict, print_dict, 0):
            typed = False
        print('Finished expt {}: with type {}'.format(expt.trial_id, typed))
        if not typed:
            untyped_expts.append(expt)

    # Print Dictionary
    for condition in conditions:
        print '--{}--'.format(condition)
        for session in print_dict[condition]:
            print '  {}: {}'.format(session, print_dict[condition][session])

    # print if untyped dictionaries
    if untyped_expts:
        print '--Unmatched--'
        for expt in untyped_expts:
            print expt.get('tSeriesDirectory')

    # if save, save.
    if args.save:
        for expt in matched_experiments:
            if expt in untyped_expts:
                continue
            expt.assoc_expts = assoc_dict
            expt.save(store=True)


if __name__ == "__main__":
    main(sys.argv[1:])
