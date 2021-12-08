import numpy as np

import lab.analysis.behavior_analysis as ba

import os
import sys
sys.path.insert(0, '../induction_analysis')
import Analysis.induction_analysis as ia

from lab_repo.classes.place_cell_classes import pcExperimentGroup

import cPickle as pkl


def save_load(func):
    """Decorator to auto save/load roilist filters."""
    def save_load_filter(*args, **kwargs):

        path = '/home/sebi/induction_cache'
        path = os.path.join(path, '{}_{}.pkl'.format(
            args[0]._label, func.__name__))

        try:
            with open(path, 'rb') as fp:
                roilist = pkl.load(fp)
            saved = True
        except IOError:
            saved = False

        if 'overwrite' in kwargs:
            overwrite = kwargs.pop('overwrite')
        else:
            overwrite = False

        if (not saved) or overwrite:
            roilist = func(*args, **kwargs)
            with open(path, 'wb') as fw:
                pkl.dump(roilist, fw)

        return lambda x: x.label in roilist

    return save_load_filter


def dist_threshold(func):
    """Decorator to auto save/load roilist filters."""
    def dist_threshold_filter(*args, **kwargs):

        if 'max_dist' in kwargs:
            dist = kwargs.pop('overwrite')
        else:
            return func(*args, **kwargs)

        grp = args[0]
        dists = ia.dist_to_target(grp, stim_filter=func(*args, **kwargs))
        dists = dists.loc[dists['dist'] <= dist]

        roilist = dists['roi'].tolist()

        return lambda x: x.label in roilist

    return dist_threshold_filter


def zero_lap_spike(spikes, expt, thresh=5):
    # spikes is 1d

    running = expt.runningIntervals(returnBoolList=True)[0]
    spikes = spikes[running]

    pos = ba.absolutePosition(expt.find('trial'))
    pos = pos[running]
    laps = pos.astype(int)
    pos = (pos % 1) * 100

    spike_pos = pos[(laps == 0) & (spikes > 0)]

    stim_pos = expt._get_stim_positions(units='normalized')[0]

    dist_from_stim = [x - stim_pos for x in spike_pos]
    if any([np.abs(x) < thresh for x in dist_from_stim]):
        return True
    else:
        return False


@dist_threshold
@save_load
def grp_stim_filter(grp, label=None, thresh=5, spatial_dist=None, activity_filter=True):
    """Return roi filter of 'stimmed' cells based on list of roi labels."""

    pc_kwargs = {'imaging_label': label,
                 'nPositionBins': 100,
                 'channel': 'Ch2',
                 'demixed': False,
                 'pf_subset': None,
                 'signal': 'spikes'}

    labels = []

    if spatial_dist is not None:
        try:
            dists = ia.dist_to_target(grp)
        except KeyError:
            dists = None
    else:
        dists = None

    for expt in grp:

        # bin_to_cm = expt.track_length / 1000.
        # cm_thresh = thresh * bin_to_cm

        rois = expt.rois(label=label)
        # spikes = expt.spikes(label=label)

        if dists is not None:
            expt_dists = dists.loc[dists['expt_id'] == expt.trial_id]['dist'].values
        else:
            expt_dists = [0] * len(rois)

        stim_idx = [i for i, r in enumerate(rois) if 'stimmed' in r.tags]

        if 'control' in expt.session:
            baseline_session = expt._get_session('control_baseline')
        else:
            baseline_session = expt._get_session('cno_baseline')

        pc_grp = pcExperimentGroup([baseline_session], **pc_kwargs)
        pfs = pc_grp.pfs()[baseline_session]

        stim_pos = expt._get_stim_positions(units='normalized')[0]

        # centroids = ia.activity_centroid_distance(pc_grp, stim_session=expt.session)
        # centroids_overlap = centroids['abs_dist'] <= cm_thresh

        stim_labels = []
        # Exclude any stimulated cells with pf overlapping stim location
        # Or with activity centroid within threshold distance of stim location
        # Or with any spikes within threshold of stim location during running on pre-stim lap
        # Or any cells too far from target location
        for i in stim_idx:
            if activity_filter and ia.pf_overlaps_stim(pfs[i], stim_pos):
                    continue
                # if centroids_overlap[i]:
                    # continue
                # if zero_lap_spike(spikes[i, :], expt):
                #     continue

            if (spatial_dist is None) or (expt_dists[i] <= spatial_dist):
                stim_labels.append(rois[i].label)

        labels.extend(stim_labels)

    labels = set(labels)
    return labels

@dist_threshold
@save_load
def grp_unstimmed_filter(grp, label=None, exclude_baseline_stim=False, thresh=5,
                         spatial_dist=np.inf, activity_filter=True):
    """Return roi filter of 'unstimmed' cells with new pfs near stim zone."""

    if not exclude_baseline_stim:
        labels = []
        for expt in grp:
            labels.extend([r.label for r in expt.rois()
                           if 'stimmed' not in r.tags])
        return set(labels)

    pc_kwargs = {'imaging_label': label,
                 'nPositionBins': 100,
                 'channel': 'Ch2',
                 'demixed': False,
                 'pf_subset': None,
                 'signal': 'spikes'}

    labels = []

    # TODO include this??
    try:
        dists = ia.dist_to_target(grp)
    except KeyError:
        dists = None

    # First find all unstimmed cells
    for expt in grp:

        bin_to_cm = expt.track_length / 1000.
        cm_thresh = thresh * bin_to_cm

        rois = expt.rois(label=label)
        spikes = expt.spikes(label=label)

        if dists is not None:
            expt_dists = dists.loc[dists['expt_id'] == expt.trial_id]['dist'].values
        else:
            expt_dists = [0] * len(rois)

        unstimmed_idx = [i for i, r in enumerate(rois) if 'stimmed' not in r.tags]

        if 'control' in expt.session:
            baseline_session = expt._get_session('control_baseline')
        else:
            baseline_session = expt._get_session('cno_baseline')

        pc_grp = pcExperimentGroup([baseline_session], **pc_kwargs)
        pfs = pc_grp.pfs()[baseline_session]

        stim_pos = expt._get_stim_positions(units='normalized')[0]

        # centroids = ia.activity_centroid_distance(pc_grp, stim_session=expt.session)
        # centroids_overlap = centroids['abs_dist'] <= cm_thresh

        stim_labels = []
        # Exclude any stimulated cells with pf overlapping stim location
        # Or with activity centroid within threshold distance of stim location
        # Or with any spikes within threshold of stim location during running on pre-stim lap
        # Or any cells too far from target location
        for i in unstimmed_idx:
            if activity_filter:
                if ia.pf_overlaps_stim(pfs[i], stim_pos):
                    continue
                # if centroids_overlap[i]:
                    # continue
                if zero_lap_spike(spikes[i, :], expt):
                    continue

            if expt_dists[i] <= spatial_dist:
                stim_labels.append(rois[i].label)

        labels.extend(stim_labels)

    labels = set(labels)

    return labels


@dist_threshold
@save_load
def grp_unstimmed_formation_filter(grp, label=None, stim_filter=None,
                                   thresh=5, non_pf=False):

    if not stim_filter:
        stim_filter = grp_unstimmed_filter(grp, label=label,
                                           exclude_baseline_stim=True,
                                           thresh=thresh)

    # This should not load the precomputed induction filter...
    # Maybe have indutction and unstimmed formation both make 
    # subcall to undecorated filter function
    # grp.label = grp.label + '_unstimmed'
    # return grp_induction_filter(grp, label=label, stim_filter=stim_filter,
    #                             thresh=thresh, non_pf=non_pf)
    pc_kwargs = {'imaging_label': label,
                 'nPositionBins': 100,
                 'channel': 'Ch2',
                 'demixed': False,
                 'pf_subset': None,
                 'signal': 'spikes'}

    # TODO this assumes same track length for all expts
    bin_to_cm = grp[0].track_length / 1000.
    cm_thresh = thresh * bin_to_cm

    if not stim_filter:
        stim_filter = grp_stim_filter(grp, label=label)

    pc_grp = pcExperimentGroup(grp, **pc_kwargs)
    grp_pfs = pc_grp.pfs(roi_filter=stim_filter)

    # If centroid is within thresh dist of stim location
    centroids = ia.centroid_distance(pc_grp, roi_filter=stim_filter,
                                  stim_session=grp[0].session)
    centroids = centroids.loc[centroids['abs_dist'] <= cm_thresh]
    labels = centroids['roi'].values.tolist()

    # If activity centroid within thresh dist
    if non_pf:
        centroids = ia.activity_centroid_distance(pc_grp, roi_filter=stim_filter,
                                               stim_session=grp[0].session)
        centroids = centroids.loc[centroids['abs_dist'] <= cm_thresh]
        labels += centroids['roi'].values.tolist()

    # If PF overlaps stim location
    for expt in grp:
        rois = expt.rois(roi_filter=stim_filter)
        pfs = grp_pfs[expt]
        stim_loc = expt._get_stim_positions(units='normalized')[0]
        for roi, pf in zip(rois, pfs):
            if ia.pf_overlaps_stim(pf, stim_loc):
                labels.append(roi.label)

    labels = set(labels)

    return labels


@dist_threshold
@save_load
def grp_induction_filter(grp, label=None, stim_filter=None, thresh=5, non_pf=False, stim_session=None):

    pc_kwargs = {'imaging_label': label,
                 'nPositionBins': 100,
                 'channel': 'Ch2',
                 'demixed': False,
                 'pf_subset': None,
                 'signal': 'spikes'}

    # TODO this assumes same track length for all expts
    bin_to_cm = grp[0].track_length / 1000.
    cm_thresh = thresh * bin_to_cm

    if not stim_session:
        stim_session = grp[0].session

    if not stim_filter:
        stim_filter = grp_stim_filter(grp, label=label)

    pc_grp = pcExperimentGroup(grp, **pc_kwargs)
    grp_pfs = pc_grp.pfs(roi_filter=stim_filter)

    # If centroid is within thresh dist of stim location
    centroids = ia.centroid_distance(pc_grp, roi_filter=stim_filter,
                                  stim_session=stim_session)
    centroids = centroids.loc[centroids['abs_dist'] <= cm_thresh]
    labels = centroids['roi'].values.tolist()

    # If activity centroid within thresh dist
    if non_pf:
        centroids = ia.activity_centroid_distance(pc_grp, roi_filter=stim_filter,
                                               stim_session=stim_session)
        centroids = centroids.loc[centroids['abs_dist'] <= cm_thresh]
        labels += centroids['roi'].values.tolist()

    # If PF overlaps stim location
    for expt in grp:
        rois = expt.rois(roi_filter=stim_filter)
        pfs = grp_pfs[expt]
        stim_loc = expt._get_session(stim_session)._get_stim_positions(units='normalized')[0]
        for roi, pf in zip(rois, pfs):
            if ia.pf_overlaps_stim(pf, stim_loc):
                labels.append(roi.label)

    labels = set(labels)

    return labels


# def grp_response_filter(grp, label=None, session='control_induction',
#                         pct=0.8):
#     """Return filter for rois that were at least pct percentile responsive."""
#     labels = []
#     for expt in grp:

#         shuffle_dict_path = os.path.join(expt.get('tSeriesDirectory'),
#                                          'shuffle_dictionary.pkl')
#         with open(shuffle_dict_path, 'rb') as fp:
#             shuffle_dict = pkl.load(fp)['values']

#         labels.extend([k for k in shuffle_dict if shuffle_dict[k] > pct])

#     labels = set(labels)

#     return lambda x: x.label in labels


@dist_threshold
@save_load
def pcs_filter(grp, label=None, roi_filter=None):
        pcs = []

        pfs = grp.pfs(roi_filter=roi_filter)

        for expt in grp:
            pc_inds = np.where(pfs[expt])[0]
            rois = expt.rois(label=label,
                             roi_filter=roi_filter)
            pcs.extend([rois[x].label for x in pc_inds])
        pcs = set(pcs)

        return pcs
