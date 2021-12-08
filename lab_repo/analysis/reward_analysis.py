import numpy as np
import pandas as pd
import itertools as it
from collections import defaultdict
import warnings

import lab_repo.analysis.behavior_analysis as ba
from lab_repo.classes.classes import ExperimentGroup
from lab_repo.classes import exceptions as exc

def fractionLicksNearRewardsPerLap(
        expGrp, anticipStartCM=-5, anticipEndCM=-0.1, compareStartCM=-15,
        compareEndCM=-0.1, fractionColName="value", rewardPositions=None,
        exclude_reward=False, exclude_reward_duration=10.0, vel=False, z=False):
    """Fraction of licks in the anticipatory zone vs a compare zone, per lap.

    Parameters
    ----------
    anticipStartCM, anticipEndCM : float
        licks in this spatial window is counted anticipatory.
        Units are in cm. The reward zone start is considered as 0.
        Prereward space is negative, and post reward space is positive.
    copareStartCM, compareEndCM : float
        licks in this window is counted toward total licks.
        Usually, this window should contain the anticipatory window.
    fractionColName : str
        the name to give to the lick fraction column of the returned
        dataframe,  default is "value"
    rewardPositions : {str, None, np.ndarray}
        If a string, assumed to be a condition label, and will use the
        reward positions used for each mouse during the condition.
        If 'None', uses the actual reward positions during the experiment.
        Otherwise pass in normalized reward positions.
    exclude_reward : bool
        exclude licks that occurs after water rewards. This is trying to
        ignore licks that are for drinking water. Default is false.
    exclude_duration : float
        number of seconds after the onset of water reward for which the
        licking should be ignored. Default is 10 seconds.

    Returns
    -------
    pd.DataFrame
        Each row is the licking calculation of a lap, with columns:
            trial - trial instance
            rewardPosition - reward location
            lapNum - the lap number
            anticipLicks - number of licks in the anticipatory zone
            compareLicks - number of licks in the compare zone
            value - anticipLicks/compareLicks.
                the name of this column is set by the kwarg fractionColName,
                when the compare zone contains the anticipatory zone, it is
                similar to fraction of licks near rewards
    """
    result = []

    if(rewardPositions is None):
        rewards_by_exp = {exp: exp.rewardPositions(units='normalized')
                          for exp in expGrp}
    else:
        rewards_by_exp = defaultdict(lambda: np.array(rewardPositions))

    for exp in expGrp:

        belt_length = exp.track_length / 10.
        anticipStart = anticipStartCM / float(belt_length)
        anticipEnd = anticipEndCM / float(belt_length)
        compareStart = compareStartCM / float(belt_length)
        compareEnd = compareEndCM / float(belt_length)

        rewards = rewards_by_exp[exp]

        trial_id = exp.trial_id
        mouse_name = exp.parent.mouse_name

        for trial in exp.findall("trial"):
            position = ba.absolutePosition(
                trial, imageSync=False, sampling_interval="actual")
            bd = trial.behaviorData(
                imageSync=False, sampling_interval="actual")

            if vel:
                velocity = exp.velocity(imageSync=False, sampling_interval='actual', smoothing='flat')[0]
                velocity[velocity < 1] = np.nan
                if z:
                    vmean = np.nanmean(velocity)
                    vstd = np.nanstd(velocity)
                    velocity = (velocity - vmean) / vstd

            lapNum = position.astype("int32")

            for reward in rewards:
                for i in np.r_[0:np.max(lapNum)]:
                    absRewardPos = i + reward
                    anticipS = absRewardPos + anticipStart
                    anticipE = absRewardPos + anticipEnd
                    compareS = absRewardPos + compareStart
                    compareE = absRewardPos + compareEnd

                    anticipBA = (position >= anticipS) & (position < anticipE)
                    compareBA = (position >= compareS) & (position < compareE)

                    # Skip lap if mouse was never inside anticipation zone
                    if np.all(~anticipBA):
                        continue

                    exclusion_dist = 0

                    if exclude_reward:
                        numExcPoints = np.int(np.float(
                            exclude_reward_duration) / bd["samplingInterval"])
                        try:
                            firstWater = np.where(compareBA & bd["water"])[0]
                            if(firstWater.size > 0):
                                firstWater = firstWater[0]
                                compareBA[firstWater:firstWater +
                                          numExcPoints] = False
                                anticipBA[firstWater:firstWater +
                                          numExcPoints] = False

                                last_idx = min(firstWater + numExcPoints, len(position) - 1)
                                exclusion_dist = position[last_idx] - position[firstWater]

                        except KeyError:
                            pass


                    if vel:
                        numAnticipLicks = np.nanmean(velocity[anticipBA])
                        numCompareLicks = np.nanmean(velocity[compareBA])
                    else:
                        numAnticipLicks = np.sum(bd["licking"][anticipBA])
                        numCompareLicks = np.sum(bd["licking"][compareBA])

                    fraction = numAnticipLicks / float(numCompareLicks)

                    result.append({"trial": trial_id,
                                   "mouse": mouse_name,
                                   "condition": exp.condition,
                                   "session": exp.session,
                                   "opsin": exp.opsin,
                                   "belt": exp.belt_name,
                                   "rewardPos": reward,
                                   "lapNum": i,
                                   "anticipLicks": numAnticipLicks,
                                   "compareLicks": numCompareLicks,
                                   fractionColName: fraction,
                                   "exclusion_dist": exclusion_dist,
                                   "chance": (anticipEndCM - anticipStartCM - exclusion_dist) /
                                             float(compareEndCM - compareStartCM - exclusion_dist)})


    return pd.DataFrame(result, columns=[
        'trial', 'mouse', 'condition', 'session', 'belt', 'opsin', 'rewardPos', 'lapNum', 'anticipLicks', 'compareLicks', 'exclusion_dist', 'chance', fractionColName])


def rz2_dict(grp):

    return_dict = {}
    for expt in grp:
        if expt.session == 3:

            mouse_name = expt.parent.mouse_name
            belt = expt.belt_name

            reward = expt.rewardPositions(units='normalized')[0]

            return_dict[(mouse_name, belt)] = reward

    return return_dict


def fractionLicksNearRZ2PerLap(
        expGrp, anticipStartCM=-5, anticipEndCM=-0.1, compareStartCM=-15,
        compareEndCM=-0.1, fractionColName="value",
        exclude_reward=False, exclude_reward_duration=10.0, vel=False, z=False):
    """Fraction of licks in the anticipatory zone vs a compare zone, per lap.

    Parameters
    ----------
    anticipStartCM, anticipEndCM : float
        licks in this spatial window is counted anticipatory.
        Units are in cm. The reward zone start is considered as 0.
        Prereward space is negative, and post reward space is positive.
    copareStartCM, compareEndCM : float
        licks in this window is counted toward total licks.
        Usually, this window should contain the anticipatory window.
    fractionColName : str
        the name to give to the lick fraction column of the returned
        dataframe,  default is "value"
    rewardPositions : {str, None, np.ndarray}
        If a string, assumed to be a condition label, and will use the
        reward positions used for each mouse during the condition.
        If 'None', uses the actual reward positions during the experiment.
        Otherwise pass in normalized reward positions.
    exclude_reward : bool
        exclude licks that occurs after water rewards. This is trying to
        ignore licks that are for drinking water. Default is false.
    exclude_duration : float
        number of seconds after the onset of water reward for which the
        licking should be ignored. Default is 10 seconds.

    Returns
    -------
    pd.DataFrame
        Each row is the licking calculation of a lap, with columns:
            trial - trial instance
            rewardPosition - reward location
            lapNum - the lap number
            anticipLicks - number of licks in the anticipatory zone
            compareLicks - number of licks in the compare zone
            value - anticipLicks/compareLicks.
                the name of this column is set by the kwarg fractionColName,
                when the compare zone contains the anticipatory zone, it is
                similar to fraction of licks near rewards
    """
    result = []

    rewards_dict = rz2_dict(expGrp)

    for ei, exp in enumerate(expGrp):

        belt_length = exp.track_length / 10.
        anticipStart = anticipStartCM / float(belt_length)
        anticipEnd = anticipEndCM / float(belt_length)
        compareStart = compareStartCM / float(belt_length)
        compareEnd = compareEndCM / float(belt_length)

        trial_id = exp.trial_id
        mouse_name = exp.parent.mouse_name
        belt_name = exp.belt_name

        try:
            reward = rewards_dict[(mouse_name, belt_name)]
        except KeyError:
            print '{} {} {}'.format(ei, mouse_name, belt_name)
            continue

        for trial in exp.findall("trial"):
            position = ba.absolutePosition(
                trial, imageSync=False, sampling_interval="actual")
            bd = trial.behaviorData(
                imageSync=False, sampling_interval="actual")

            if vel:
                velocity = exp.velocity(imageSync=False, sampling_interval='actual', smoothing='flat')[0]
                velocity[velocity < 1] = np.nan
                if z:
                    vmean = np.nanmean(velocity)
                    vstd = np.nanstd(velocity)
                    velocity = (velocity - vmean) / vstd

            lapNum = position.astype("int32")

            if exp.session == 2:
                start_idx = 11
            else:
                start_idx = 0

            for i in np.r_[start_idx:np.max(lapNum)]:
                absRewardPos = i + reward
                anticipS = absRewardPos + anticipStart
                anticipE = absRewardPos + anticipEnd
                compareS = absRewardPos + compareStart
                compareE = absRewardPos + compareEnd

                anticipBA = (position >= anticipS) & (position < anticipE)
                compareBA = (position >= compareS) & (position < compareE)

                # Skip lap if mouse was never inside anticipation zone
                if np.all(~anticipBA):
                    continue

                exclusion_dist = 0

                if exclude_reward:
                    numExcPoints = np.int(np.float(
                        exclude_reward_duration) / bd["samplingInterval"])
                    try:
                        firstWater = np.where(compareBA & bd["water"])[0]
                        if(firstWater.size > 0):
                            firstWater = firstWater[0]
                            compareBA[firstWater:firstWater +
                                      numExcPoints] = False
                            anticipBA[firstWater:firstWater +
                                      numExcPoints] = False

                            last_idx = min(firstWater + numExcPoints, len(position) - 1)
                            exclusion_dist = position[last_idx] - position[firstWater]

                    except KeyError:
                        pass


                if vel:
                    numAnticipLicks = np.nanmean(velocity[anticipBA])
                    numCompareLicks = np.nanmean(velocity[compareBA])
                else:
                    numAnticipLicks = np.sum(bd["licking"][anticipBA])
                    numCompareLicks = np.sum(bd["licking"][compareBA])

                fraction = numAnticipLicks / float(numCompareLicks)


                result.append({"trial": trial_id,
                               "mouse": mouse_name,
                               "condition": exp.condition,
                               "session": exp.session,
                               "opsin": exp.opsin,
                               "belt": exp.belt_name,
                               "rewardPos": reward,
                               "lapNum": i,
                               "anticipLicks": numAnticipLicks,
                               "compareLicks": numCompareLicks,
                               fractionColName: fraction,
                               "exclusion_dist": exclusion_dist,
                               "chance": (anticipEndCM - anticipStartCM - exclusion_dist) /
                                         float(compareEndCM - compareStartCM - exclusion_dist)})


    return pd.DataFrame(result, columns=[
        'trial', 'mouse', 'condition', 'session', 'belt', 'opsin', 'rewardPos', 'lapNum', 'anticipLicks', 'compareLicks', 'exclusion_dist', 'chance', fractionColName])