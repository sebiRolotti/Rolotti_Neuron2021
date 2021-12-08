import os

import numpy as np

from Metaclasses.SingletonMeta import SingletonMeta
from lab.classes.dbclasses import dbExperimentSet


class Helpers:
    __metaclass__ = SingletonMeta

    @staticmethod
    def get_position_based_start_end_led_context(experiment):
        behavior_data = experiment.BehaviorData()
        center = int(behavior_data['__trial_info']['contexts']['led_context']['locations'][0])
        radius = int(behavior_data['__trial_info']['contexts']['led_context']['radius'])
        start_pos = (((center - radius) / float(1000)) % 1) * 100
        end_pos = (((center + radius) / float(1000)) % 1) * 100
        return np.round([start_pos, end_pos])

    @staticmethod
    def create_mouse_pdf_dir(mouse_name):
        # check if a pdf folder already exists
        if not os.path.isdir(os.path.expanduser('~/pdfs')):
            os.mkdir(os.path.expanduser('~/pdfs'))
        # check if the mouse folder on the pdf folder exists
        elif not os.path.isdir(os.path.expanduser('~/pdfs/{}'.format(mouse_name))):
            os.mkdir(os.path.expanduser('~/pdfs/{}'.format(mouse_name)))

    @staticmethod
    def get_exp_id_from_directory(directory, project_name):
        try:
            exp = dbExperimentSet.FetchTrials(project_name=project_name, tSeriesDirectory=directory)[0]
        except IndexError as e:
            print(e)
            print('T series directory is not paired or doesn\'t have a .sima folder')
            return None
        return exp.trial_id
