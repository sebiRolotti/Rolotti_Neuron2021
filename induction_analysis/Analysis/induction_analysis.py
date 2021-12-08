"""All analyses for induction experiments."""
import matplotlib as mpl
from matplotlib.colors import ListedColormap
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from lab_repo.misc.misc import savefigs

import lab_repo.analysis.place_cell_analysis as pca
import lab_repo.analysis.behavior_analysis as ba
import lab_repo.analysis.identify_place_fields_spikes as idf
from lab_repo.classes.place_cell_classes import pcExperimentGroup
from lab_repo.classes.dbclasses import dbExperiment

from scipy.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import detrend
import numpy as np

import pandas as pd

import os
import cPickle as pkl

from lab_repo.misc.misc import maxmin_filter, get_prairieview_version, get_element_size_um
from sima.ROI import mask2poly
from shapely.geometry import Point

from pycircstat.descriptive import _complex_mean

# Utilities
def pf_overlaps_stim(pfs, stim_loc, nbins=100):

    stim_loc = int(stim_loc)

    pf_on = np.zeros((nbins, ))

    for pf in pfs:
        if pf[0] < pf[1]:
            pf_on[pf[0]:pf[1] + 1] = 1
        else:
            pf_on[pf[0]:] = 1
            pf_on[:pf[1] + 1] = 1

    if pf_on[stim_loc]:
        return True
    else:
        return False


def dist(a, b):
    x = a - b
    if x <= -50:
      x += 100
    elif x > 50:
      x -= 100
    return x

def convert_centroid(x):
    """Convert centroid from complex value to position.

    Take angle and convert from radians.
    """
    return ((np.angle(x) % (2 * np.pi))) / (2 * np.pi) * 100


# Stim zone over-representation
def centroid_distance(grp, roi_filter=None, target=None,
                      stim_session='control_induction'):
    """Find centroid distance for given sessions to induction stim zone."""
    data_list = []

    data = grp.data(roi_filter=roi_filter)
    pfs = grp.pfs(roi_filter=roi_filter)
    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        bin_to_cm = expt.track_length / 1000.

        centroids = pca.calcCentroids(data[expt], pfs[expt], returnAll=True)
        rois = expt.rois(roi_filter=roi_filter)

        if (target is None) or (target == 'stim'):
            pos = expt._get_session(stim_session)._get_stim_positions(units='normalized')[0]

        for roi, centroid in zip(rois, centroids):
            if centroid:
                closest_centroid = np.argmin([dist(x, pos) for x in centroid])

                data_list.append({'mouse_name': mouse_name,
                                  'expt_id': trial_id,
                                  'roi': roi.label,
                                  'centroid': centroid[closest_centroid],
                                  'dist': dist(centroid[closest_centroid], pos) * bin_to_cm,
                                  'abs_dist': np.abs(dist(centroid[closest_centroid], pos)) * bin_to_cm})
            else:
                data_list.append({'mouse_name': mouse_name,
                                  'expt_id': trial_id,
                                  'roi': roi.label,
                                  'centroid': np.nan,
                                  'dist': np.nan,
                                  'abs_dist': np.nan})

    return pd.DataFrame(data_list)


# @save_load
def activity_centroid_distance(grp, roi_filter=None, target=None, stim_session='control_induction'):
    """Find activity centroid distance for sessions to induction stim zone."""
    data_list = []

    centroids = pca.calc_activity_centroids(grp, roi_filter=roi_filter)

    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        bin_to_cm = expt.track_length / 1000.

        rois = expt.rois(roi_filter=roi_filter)

        if (target is None) or (target == 'stim'):
            pos = expt._get_session(stim_session)._get_stim_positions(units='normalized')[0]

        for roi, centroid in zip(rois, centroids[expt]):
            if centroid:

                centroid_pos = convert_centroid(centroid)

                data_list.append({'mouse_name': mouse_name,
                                  'expt_id': trial_id,
                                  'roi': roi.label,
                                  'centroid': centroid_pos,
                                  'dist': dist(centroid_pos, pos) * bin_to_cm,
                                  'abs_dist': abs(dist(centroid_pos, pos)) * bin_to_cm})
            else:
                data_list.append({'mouse_name': mouse_name,
                                  'expt_id': trial_id,
                                  'roi': roi.label,
                                  'centroid': np.nan,
                                  'dist': np.nan,
                                  'abs_dist': np.nan})

    return pd.DataFrame(data_list)


def activity_centroid_distance_shift(grp, roi_filter=None,
                                     session1='control_baseline',
                                     session2='control_induction'):
    """Find activity centroid shift for all rois between two sessions."""
    pc_kwargs = grp.args

    data_list = []

    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        sess1 = expt._get_session(session1)
        sess2 = expt._get_session(session2)

        sess1_id = sess1.trial_id
        sess2_id = sess2.trial_id

        bin_to_cm = expt.track_length / 1000.

        pcgrp = pcExperimentGroup([sess1, sess2], **pc_kwargs)
        centroids = pca.calc_activity_centroids(pcgrp, roi_filter=roi_filter)
        # TODO Hack for now to avoid weird 28875 __eq__ bug
        for ek in centroids.keys():
             centroids[ek.trial_id] = centroids[ek]

        rois = expt.rois(roi_filter=roi_filter, label=grp.args['imaging_label'])

        pos = expt._get_session('control_induction')._get_stim_positions(units='normalized')[0]

        for i, roi in enumerate(rois):

            try:
                centroid1 = convert_centroid(centroids[sess1_id][i])
                centroid2 = convert_centroid(centroids[sess2_id][i])
            except AttributeError:
                continue

            if centroid1 and centroid2:

                value = dist(dist(centroid2, pos), dist(centroid1, pos)) * bin_to_cm

                data_list.append({
                    'expt_id': trial_id,
                    'mouse_name': mouse_name,
                    'roi': roi.label,
                    'value': value,
                    'abs_val': np.abs(value),
                    'sess1_pos': centroid1,
                    'sess2_pos': centroid2,
                    'stim_pos': pos
                })

    return pd.DataFrame(data_list)


def calc_activity_centroids(tcs, n_bins=100):
    """
    Output:
        list = (nROIs,)
    """

    bins = 2 * np.pi * np.arange(0, 1, 1. / n_bins)

    result = []
    for tc in tcs:
        finite_idxs = np.where(np.isfinite(tc))[0]
        p = _complex_mean(bins[finite_idxs], tc[finite_idxs])
        result.append(p)

    return result


def delta_centroid_distance(grp, roi_filter=None):

    session = grp[0].session
    baseline_session = grp[0].session.split('_')[0] + '_baseline'

    data_list = []

    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        bin_to_cm = expt.track_length / 1000.

        rois = expt.rois(roi_filter=roi_filter)

        tcs = tuning_difference(pcExperimentGroup([expt], **grp.args),
                                session1=baseline_session, session2=session,
                                roi_filter=roi_filter, normalize=False)

        centroids = calc_activity_centroids(tcs)

        # Deltas are already centered
        pos = 50

        for roi, centroid in zip(rois, centroids):
            if centroid:

                centroid_pos = convert_centroid(centroid)

                data_list.append({'mouse_name': mouse_name,
                                  'expt_id': trial_id,
                                  'roi': roi.label,
                                  'centroid': centroid_pos,
                                  'dist': dist(centroid_pos, pos) * bin_to_cm,
                                  'abs_dist': abs(dist(centroid_pos, pos)) * bin_to_cm})
            else:
                data_list.append({'mouse_name': mouse_name,
                                  'expt_id': trial_id,
                                  'roi': roi.label,
                                  'centroid': np.nan,
                                  'dist': np.nan,
                                  'abs_dist': np.nan})

    return pd.DataFrame(data_list)



def psth(expt, roi_filter=None, pre=1, post=1):

    sigs = expt.imagingData(dFOverF='from_file', roi_filter=roi_filter)[..., 0]
    spikes = expt.spikes(roi_filter=roi_filter)
    rois = expt.rois(roi_filter=roi_filter)

    stim_frames = np.asarray(expt.stim_times) / expt.frame_period()
    stim_frames = stim_frames.astype(int)[:, 0]
    if stim_frames[0] == 0:
        stim_frames = stim_frames[1:]

    pre = int(pre / expt.frame_period())
    post = int(post / expt.frame_period())

    psths = np.zeros((len(stim_frames), len(rois), pre + post))

    for roi_idx, roi in enumerate(rois):

        rsig = sigs[roi_idx, :]
        signal_filtered = rsig[spikes[roi_idx, :] == 0]
        mean = np.nanmean(signal_filtered)
        std = np.nanstd(signal_filtered)

        for stim_idx, stim_frame in enumerate(stim_frames):
            psths[stim_idx, roi_idx, :] = sigs[roi_idx, stim_frame - pre:stim_frame + post]

        psths[:, roi_idx, :] = (psths[:, roi_idx, :] - mean) / std

    T = np.arange(-1 * pre, post) * expt.frame_period()
    T = np.linspace(-pre, post, psths.shape[-1]) * expt.frame_period()

    return T, psths


# Size of stim response
def burst_size(grp, roi_filter=None, win=(1, 1), exclude_stim=False, uncorrected=False):

    label = grp.args['imaging_label']
    data_list = []

    for j, expt in enumerate(grp):

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        pre = int(win[0] / expt.frame_period())
        post = int(win[1] / expt.frame_period())

        if uncorrected:
            with open(expt.signalsFilePath(), 'rb') as fp:
                sigfile = pkl.load(fp)

            sigs = sigfile[label]['raw'][0] + sigfile[label]['npil'][0]
        else:
            sigs = expt.imagingData(dFOverF='from_file', roi_filter=roi_filter, label=label)[..., 0]
        spikes = expt.spikes(roi_filter=roi_filter, label=label)

        stim_frames = np.asarray(expt.stim_times) / expt.frame_period()
        stim_frames = stim_frames.astype(int)[:, 0]
        if stim_frames[0] == 0:
            stim_frames = stim_frames[1:]

        stim_dur = int(1 / expt.frame_period())

        for roi_idx, roi in enumerate(expt.rois(roi_filter=roi_filter)):

            rsig = sigs[roi_idx, :]
            signal_filtered = rsig[spikes[roi_idx, :] == 0]

            mean = np.nanmean(signal_filtered)
            std = np.nanstd(signal_filtered)

            roi_psth = []
            for stim_frame in stim_frames:
                if exclude_stim:
                    roi_psth.append(np.hstack([sigs[roi_idx, stim_frame - pre:stim_frame],
                                    sigs[roi_idx, stim_frame + stim_dur:stim_frame + stim_dur + post + 1]]))
                else:
                    roi_psth.append(sigs[roi_idx, stim_frame - pre:stim_frame + post + 1])
            psth = np.nanmean(roi_psth, axis=0)

            zpsth = (psth - mean) / std

            data_list.append({'expt_id': trial_id,
                              'mouse_name': mouse_name,
                              'roi': roi.label,
                              'diff': np.nanmean(psth[pre:]) - np.nanmean(psth[:pre]),
                              'mean': np.nanmean(psth[pre:]),
                              'auc': np.nansum(psth[pre:]),
                              'peak': np.nanmax(psth[pre:]),
                              'zdiff': np.nanmean(zpsth[pre:]) - np.nanmean(zpsth[:pre]),
                              'zmean': np.nanmean(zpsth[pre:]),
                              'zauc': np.nansum(zpsth[pre:]),
                              'zpeak': np.nanmax(zpsth[pre:])
                              })

    return pd.DataFrame(data_list)

# Distance From Targeted Stimulation

def get_microns(expt):

    T_xml_dirname = expt.get('tSeriesDirectory')
    T_xml_path = os.path.join(T_xml_dirname, os.path.basename(T_xml_dirname) + '.xml')
    pv_version = get_prairieview_version(T_xml_path)
    [T_y_spacing, T_x_spacing] = get_element_size_um(T_xml_path,
                                                     pv_version)[-2:]

    return T_y_spacing, T_x_spacing


def binarize_mask(roi, thres=0.2):
        mask = np.sum(roi.__array__(), axis=0)
        return mask > (np.max(mask) * thres)


def dist_to_target(grp, stim_filter=None):
    # Distance of stimmed cells to target
    # Everything done in pixel space until final conversion
    dists = []
    for expt in grp:
        rois = expt.rois(roi_filter=stim_filter)
        stim_loc = expt._get_stim_locations()[0, :]
        spiral_height = expt.spiral_sizes[0] * expt.frame_shape()[2]

        spiral_center = Point(stim_loc)
        spiral_footprint = spiral_center.buffer(spiral_height / 2.)

        mouse_name = expt.parent.mouse_name
        eid = expt.trial_id

        for roi in rois:
            # TODO why do we have to transpose here??
            roi_poly = mask2poly(binarize_mask(roi).T)

            dist = roi_poly.distance(spiral_footprint) * get_microns(expt)[0]

            dists.append({'mouse_name': mouse_name,
                          'expt_id': eid,
                          'roi': roi.label,
                          'dist': dist})

    return pd.DataFrame(dists)


# Remapping Functions
def activity_centroid_shift(grp, roi_filter=None, session1='control_baseline',
                            session2='control_induction'):
    """Find activity centroid shift for all rois between two sessions."""

    pc_kwargs = grp.args

    data_list = []

    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        sess1 = expt._get_session(session1)
        sess2 = expt._get_session(session2)

        sess1_id = sess1.trial_id
        sess2_id = sess2.trial_id

        pcgrp = pcExperimentGroup([sess1, sess2], **pc_kwargs)
        centroids = pca.calc_activity_centroids(pcgrp, roi_filter=roi_filter)
        for ek in centroids.keys():
             centroids[ek.trial_id] = centroids[ek]

        rois = expt.rois(roi_filter=roi_filter, label=grp.args['imaging_label'])

        pos = expt._get_session('control_induction')._get_stim_positions(units='normalized')[0]

        for i, roi in enumerate(rois):

            try:
                centroid1 = convert_centroid(centroids[sess1_id][i])
                centroid2 = convert_centroid(centroids[sess2_id][i])
            except AttributeError:
                continue

            if centroid1 and centroid2:

                data_list.append({
                    'expt_id': trial_id,
                    'mouse_name': mouse_name,
                    'roi': roi.label,
                    'value': dist(centroid1, centroid2),
                    'abs_val': np.abs(dist(centroid1, centroid2)),
                    'sess1_pos': centroid1,
                    'sess2_pos': centroid2,
                    'stim_pos': pos
                })

    return pd.DataFrame(data_list)


# @save_load
def centroid_shift(grp, roi_filter=None, session1='control_baseline',
                   session2='control_induction'):
    """Find centroid shift for all rois between two sessions.

    Requires identified place fields on both sessions.
    """
    # TODO this should be defined previously
    pc_kwargs = {'imaging_label': 'suite2p',
                 'nPositionBins': 100,
                 'channel': 'Ch2',
                 'demixed': False,
                 'pf_subset': None,
                 'signal': 'spikes'}

    data_list = []

    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        sess1 = expt._get_session(session1)
        sess2 = expt._get_session(session2)

        pcgrp = pcExperimentGroup([sess1, sess2], **pc_kwargs)
        data = pcgrp.data(roi_filter=roi_filter)
        pfs = pcgrp.pfs(roi_filter=roi_filter)

        centroids1 = pca.calcCentroids(data[sess1], pfs[sess1])
        centroids2 = pca.calcCentroids(data[sess2], pfs[sess2])

        rois = expt.rois(roi_filter=roi_filter)

        pos = expt._get_session('control_induction')._get_stim_positions(units='normalized')[0]

        for i, roi in enumerate(rois):

            try:
                centroid1 = centroids1[i][0]
                centroid2 = centroids2[i][0]
            except IndexError:
                continue

            if centroid1 and centroid2:

                data_list.append({
                    'expt_id': trial_id,
                    'mouse_name': mouse_name,
                    'roi': roi.label,
                    'value': dist(centroid1, centroid2),
                    'abs_val': np.abs(dist(centroid1, centroid2)),
                    'sess1_pos': centroid1,
                    'sess2_pos': centroid2,
                    'stim_pos': pos
                })

    return pd.DataFrame(data_list)


def tuning_correlation(grp, label=None, roi_filter=None, session1='control_baseline',
                       session2='control_induction', pc_only=None):
    """Find tuning curve correlations for all rois between two sessions.

    pc_only governs pf criteria for roi filtering.
    """
    pc_kwargs = {'imaging_label': label,
                 'nPositionBins': 100,
                 'channel': 'Ch2',
                 'demixed': False,
                 'pf_subset': None,
                 'signal': 'spikes'}

    data_list = []

    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        sess1 = expt._get_session(session1)
        sess2 = expt._get_session(session2)

        pcgrp = pcExperimentGroup([sess1, sess2], **pc_kwargs)
        data = pcgrp.data(roi_filter=roi_filter)
        pfs = pcgrp.pfs(roi_filter=roi_filter)

        rois = expt.rois(roi_filter=roi_filter)

        for i, roi in enumerate(rois):

            if pc_only == 'first':
                if not pfs[sess1][i]:
                    continue

            if pc_only == 'second':
                if not pfs[sess2][i]:
                    continue

            if pc_only == 'both':
                if (not pfs[sess1][i]) or (not pfs[sess2][i]):
                    continue

            if pc_only == 'either':
                if (not pfs[sess1][i]) and (not pfs[sess2][i]):
                    continue

            corr = pearsonr(data[sess1][i, :], data[sess2][i, :])[0]

            data_list.append({
                'expt_id': trial_id,
                'mouse_name': mouse_name,
                'roi': roi.label,
                'value': corr,
                'session1': session1,
                'session2': session2,
                'sess1_pf': len(pfs[sess1][i]) > 0,
                'sess2_pf': len(pfs[sess2][i]) > 0
            })

    return pd.DataFrame(data_list)


def tuning_difference(grp, label=None, roi_filter=None, session1='control_baseline',
                       session2='control_induction', pc_only=None, by_mouse=False,
                       normalize=False):

    """Return mean difference of tuning curves.
    """

    all_diffs = []

    for expt in grp:

        sess1 = expt._get_session(session1)
        sess2 = expt._get_session(session2)

        stim_loc = expt._get_stim_positions(units='normalized')[0]
        shift = 50 - int(stim_loc)

        pcgrp = pcExperimentGroup([sess1, sess2], **grp.args)
        data = pcgrp.data(roi_filter=roi_filter)
        pfs = pcgrp.pfs(roi_filter=roi_filter)

        rois = expt.rois(roi_filter=roi_filter)

        idx = []
        for i, roi in enumerate(rois):

            if pc_only == 'first':
                if not pfs[sess1][i]:
                    continue

            if pc_only == 'second':
                if not pfs[sess2][i]:
                    continue

            if pc_only == 'both':
                if (not pfs[sess1][i]) or (not pfs[sess2][i]):
                    continue

            if pc_only == 'either':
                if (not pfs[sess1][i]) and (not pfs[sess2][i]):
                    continue

            idx.append(i)

        if not len(idx):
            continue

        if normalize:
            max1 = np.nanmean(data[sess1][np.array(idx), :], axis=1, keepdims=True)
            max2 = np.nanmean(data[sess2][np.array(idx), :], axis=1, keepdims=True)
            data2 = data[sess2][np.array(idx), :] / max1
            data1 = data[sess1][np.array(idx), :] / max2
            data2[data2 == np.inf] = np.nan
            data1[data1 == np.inf] = np.nan
            diffs = np.nan_to_num(data2) - np.nan_to_num(data1)
        else:
            diffs = data[sess2][np.array(idx), :] - data[sess1][np.array(idx), :]

        diffs = np.roll(diffs, shift, axis=1)

        # Convert from frames to seconds
        diffs = diffs / sess2.frame_period()

        if by_mouse:
            all_diffs.append(np.nanmean(diffs, axis=0))
        else:
            all_diffs.extend(diffs)

    return all_diffs

# Plot lap by lap heatmaps.

def get_heatmap(sig, running, abs_pos, n_bins, fp=1,
                normalized=False):
    n_laps = int(abs_pos[-1]) + 1

    heatmap = np.zeros((n_laps, n_bins))
    counts = np.zeros((n_laps, n_bins))

    for i in xrange(len(sig)):

        if not running[i]:
            continue
        if np.isnan(sig[i]):
            continue

        lap = int(abs_pos[i])
        posbin = int(np.mod(abs_pos[i] * n_bins, n_bins))

        heatmap[lap, posbin] += sig[i]
        counts[lap, posbin] += fp

    heatmap /= counts

    # Normalized
    if normalized and (np.nanmax(heatmap) > 0):
        heatmap = heatmap / np.nanmax(heatmap)

    return heatmap


def grp_heatmap(grp, roi_filter=None, ind_filter=None, signal='spikes', z_score=False, save_path=None, by_expt=False):

    heatmaps = []

    if save_path:
        pp = PdfPages(save_path)

    for expt in grp:

        mouse_name = expt.parent.mouse_name

        all_roi_labels = [r.label for r in expt.rois()]
        rois = expt.rois(roi_filter=roi_filter)

        if signal == 'spikes':
            spikes = expt.spikes(roi_filter=roi_filter, binary=True)
            fp = expt.frame_period()
        else:
            spikes = expt.imagingData(dFOverF='from_file', roi_filter=roi_filter)[..., 0]
            fp = 1

        if z_score:
            spikes = (spikes - np.nanmean(spikes, axis=1, keepdims=True)) / np.nanstd(spikes, axis=1, keepdims=True)

        abs_pos = ba.absolutePosition(expt.find('trial'), imageSync=True)
        running = expt.velocity()[0] > 1

        if 'control' in expt.session:
            stim_loc = expt._get_session('control_induction')._get_stim_positions(units='normalized')[0]
        else:
            stim_loc = expt._get_session('cno_induction')._get_stim_positions(units='normalized')[0]
        shift = 50 - int(stim_loc)

        expt_heatmaps = []

        for i, roi, roi_spikes in zip(xrange(len(rois)), rois, spikes):

            heatmap = get_heatmap(roi_spikes, running, abs_pos, n_bins=100, fp=fp)
            heatmap = np.roll(heatmap, shift, axis=1)

            if save_path:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                sns.heatmap(heatmap, ax=ax, cbar=True, rasterized=True)
                ax.set_xticks([])
                ax.set_yticks([])

                title = 'Mouse: {} ROI #{} label: {}'.format(
                    mouse_name, all_roi_labels.index(roi.label),
                    roi.label)

                if ind_filter:
                    if ind_filter(roi):
                        title += ', Induced'

                ax.set_title(title)

                ax.vlines(50, 1, 6, colors='w', linestyles='dotted')

                savefigs(pp, fig)

            expt_heatmaps.append(heatmap)

        if by_expt:
            heatmaps.append(np.stack(expt_heatmaps))
        else:
            heatmaps.extend(expt_heatmaps)

    if save_path:
        pp.close()

    return heatmaps


def success_rate(grp, ind_filter, stim_filter):

    success_df = []
    for expt in grp:
        n_rois = len(expt.rois())
        n_stimmed = len(expt.rois(roi_filter=stim_filter))
        n_success = len(expt.rois(roi_filter=ind_filter))

        try:
            pct_success = n_success / float(n_stimmed)
        except ZeroDivisionError:
            pct_success = 0.

        success_df.append({'expt': expt.trial_id,
                           'mouse': expt.parent.mouse_name,
                           'pct_stimmed': n_stimmed / float(n_rois),
                           'pct_success': pct_success,
                           'pct_induced': n_success / float(n_rois),
                           'n_rois': n_rois,
                           'n_stimmed': n_stimmed,
                           'n_success': n_success})

    success_df = pd.DataFrame(success_df)

    return success_df


def induction_rate(grp, stim_filter, ind_filter, unstim_filter, ctrl_ind_filter):

    success_df = success_rate(grp, ind_filter, stim_filter)
    control_df = success_rate(grp, ctrl_ind_filter, unstim_filter)

    df = success_df.merge(control_df, on=['expt', 'mouse'], suffixes=['', '_control'])
    df['fold'] = df['pct_success'] / df['pct_success_control']

    return df


def newly_formed_pfs(grp, roi_filter=None):

    data_list = []

    baseline_session = grp[0].session.split('_')[0] + '_baseline'

    baseline_grp = [x._get_session(baseline_session) for x in grp]
    baseline_grp = pcExperimentGroup(baseline_grp, **grp.args)

    data = grp.data(roi_filter=roi_filter)
    pfs = grp.pfs(roi_filter=roi_filter)

    base_pfs = baseline_grp.pfs(roi_filter=roi_filter)

    for expt, baseline_expt in zip(grp, baseline_grp):

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        rois = expt.rois(roi_filter=roi_filter)
        pos = expt._get_stim_positions(units='normalized')[0]

        centroids = pca.calcCentroids(data[expt], pfs[expt], returnAll=True)

        for i, roi in enumerate(rois):

            roi_centroids = centroids[i]
            base_roi_pfs = base_pfs[baseline_expt][i]
            for centroid in roi_centroids:
                # If the centroid is not within a previous PF,
                # we will call it newly formed.
                if not pf_overlaps_stim(base_roi_pfs, centroid):

                    data_list.append({
                        'expt_id': trial_id,
                        'mouse_name': mouse_name,
                        'roi': roi.label,
                        'value': dist(pos, centroid),
                        'true_val': centroid,
                        'stim_pos': pos
                    })

    return pd.DataFrame(data_list)


# McKenzie Remapping
def mckenzie_remapping_cells(grp, baseline_session='control_baseline', roi_filter=None, n_bins=100):

    data_list = []

    baseline_grp = [x._get_session(baseline_session) for x in grp]
    baseline_grp = pcExperimentGroup(baseline_grp, **grp.args)

    data = grp.data(roi_filter=roi_filter)
    base_data = baseline_grp.data(roi_filter=roi_filter)

    # For each cell in each expt, calculate its lap by lap correlation
    for expt, base_expt in zip(grp, baseline_grp):

        rois = expt.rois(roi_filter=roi_filter)

        spikes = expt.spikes(roi_filter=roi_filter)
        abs_pos = ba.absolutePosition(expt.find('trial'), imageSync=True)
        running = expt.runningIntervals(returnBoolList=True)[0]

        base_spikes = base_expt.spikes(roi_filter=roi_filter)
        base_pos = ba.absolutePosition(base_expt.find('trial'), imageSync=True)
        base_running = base_expt.runningIntervals(returnBoolList=True)[0]

        expt_data = data[expt]
        base_expt_data = base_data[base_expt]

        for roi, spike, base_spike, tc, btc in zip(rois, spikes, base_spikes,
                                                   expt_data, base_expt_data):

            # Only consider ten laps on each end
            heatmap = get_heatmap(spike, running, abs_pos, n_bins)[6:16, :]
            base_heatmap = get_heatmap(base_spike, base_running, base_pos, n_bins)[-11:-1, :]

            # flatten, smooth, and reshape
            heatmap = smooth_heatmap(heatmap)
            base_heatmap = smooth_heatmap(base_heatmap)

            corr_df = lap_corrs(expt, roi, heatmap, base_heatmap)

            data_list.extend(corr_df)

    return pd.DataFrame(data_list)


def smooth_heatmap(hm):

    orig_shape = hm.shape
    flat = hm.reshape(-1)

    nan_idx = np.where(np.isnan(flat))
    flat[nan_idx] = 0
    flat = gaussian_filter1d(flat, 3)
    flat[nan_idx] = np.nan

    return flat.reshape(orig_shape)


def lap_corrs(expt, roi, hm, bhm):

    corr_df = []

    tid = expt.trial_id
    rid = roi.label

    # Flip pre to step through backwards
    bhm = np.flip(bhm, axis=0)
    post_template = np.nan_to_num(np.nanmean(hm, axis=0))
    i = -1
    current_row = 0
    for lap in bhm:

        pre_template = np.nan_to_num(np.nanmean(np.delete(bhm, current_row, axis=0), axis=0))
        pre_corr = pearsonr(pre_template, np.nan_to_num(lap))[0]
        post_corr = pearsonr(post_template, np.nan_to_num(lap))[0]
        corr_df.append({'expt': tid,
                        'roi': rid,
                        'lap': i,
                        'pre_corr': pre_corr,
                        'post_corr': post_corr})

        i = i-1
        current_row = current_row + 1

    i = 1
    pre_template = np.nan_to_num(np.nanmean(bhm, axis=0))
    for lap in hm:

        post_template = np.nan_to_num(np.nanmean(np.delete(hm, i-1, axis=0), axis=0))

        pre_corr = pearsonr(pre_template, np.nan_to_num(lap))[0]
        post_corr = pearsonr(post_template, np.nan_to_num(lap))[0]
        corr_df.append({'expt': tid,
                        'roi': rid,
                        'lap': i,
                        'pre_corr': pre_corr,
                        'post_corr': post_corr})

        i = i+1

    return corr_df
