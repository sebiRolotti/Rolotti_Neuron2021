import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_style(style='ticks')

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd

import sys
sys.path.insert(0, '../induction_analysis')

import Analysis.induction_analysis as ia

from lab_repo.classes.place_cell_classes import pcExperimentGroup
from lab_repo.analysis.identify_place_fields import smooth_tuning_curves, _consecutive_integers

from pycircstat.descriptive import _complex_mean

from scipy.stats import ttest_rel, ttest_1samp

import cPickle as pkl


def formatPlot(ax, offset=10):
    for axis in ['left', 'right', 'bottom', 'top']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(labelsize=14)
    sns.despine(ax=ax, offset=offset)
    ax.set_xlabel('')
    ax.set_ylabel('')


def calc_activity_centroid(tuning_curve):

    bins = 2 * np.pi * np.arange(0, 1, 1. / len(tuning_curve))
    finite_idxs = np.where(np.isfinite(tuning_curve))[0]
    p = _complex_mean(bins[finite_idxs], tuning_curve[finite_idxs])

    return ia.convert_centroid(p)


def baseline_activity(grp, roi_filter=None, smooth_length=3):

    # Centered heatmaps, need to shift back
    # to get absolute locations
    heatmaps = ia.grp_heatmap(grp, roi_filter=roi_filter)
    n_bins = heatmaps[0].shape[1]

    data = {}

    count = 0 
    for expt in grp:

        baseline_end = min(expt.get_stimmed_laps())

        rois = expt.rois()

        tcs = np.zeros((len(rois), n_bins))
        for ri, roi in enumerate(rois):

            roi_heatmap = heatmaps[count]
            tc = np.nanmean(roi_heatmap[:baseline_end, :], axis=0)

            tcs[ri, :] = tc
            count += 1

        tcs = smooth_tuning_curves(tcs,
                                   smooth_length=smooth_length,
                                   nan_norm=False, axis=-1)

        data[expt] = tcs

    return data


def baseline_activity_centroid_distance(grp):

    # Centered heatmaps, need to shift back
    # to get absolute locations
    heatmaps = ia.grp_heatmap(grp)

    data_list = []

    count = 0
    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name

        baseline_end = min(expt.get_stimmed_laps())

        stim_loc = expt._get_stim_positions(units='normalized')[0]
        shift_back = int(stim_loc) - 50

        bin_to_cm = expt.track_length / 1000.

        rois = expt.rois()
        for ri, roi in enumerate(rois):

            roi_heatmap = heatmaps[count]
            tc = np.nanmean(roi_heatmap[:baseline_end, :], axis=0)

            centroid_pos = calc_activity_centroid(tc)

            data_list.append({'mouse_name': mouse_name,
                              'expt_id': trial_id,
                              'roi': roi.label,
                              'centroid': centroid_pos,
                              'stim_loc': stim_loc,
                              'dist': ia.dist(centroid_pos, stim_loc) * bin_to_cm,
                              'abs_dist': abs(ia.dist(centroid_pos, stim_loc)) * bin_to_cm,
                              'tc': np.roll(tc, shift_back)})

            count += 1

    return pd.DataFrame(data_list)


def activity_centroid_plots(grp, save_path):

    full_sessions = ['control_induction', 'control_24h', 'control_48h', 'control_72h']
    session_names = ['Pre', 'Post', '24h', '48h', '72h']

    # Plots comparing activity centroid distance to stim zone for
    # a set of sessions - for now has to be all control or all cno

    # First calculate for baseline laps
    # Have to do this manually because these tcs arent saved anywhere
    # only those post-induction are
    df = baseline_activity_centroid_distance(grp)
    df['session'] = session_names[0]

    df = df.drop(columns=['tc', 'stim_loc'])

    stim_session = 'control_induction'
    for session, session_name in zip(full_sessions, session_names[1:]):

        sess_grp = [x._get_session(session) for x in grp]
        sess_grp = pcExperimentGroup(sess_grp, **grp.args)

        sess_df = ia.activity_centroid_distance(sess_grp, stim_session=stim_session)
        sess_df['session'] = session_name

        df = pd.concat([df, sess_df])

    # Plot boxes and swarm by cell
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(x='session', y='abs_dist', data=df, showcaps=False,
                showfliers=False, whis=False, ax=ax)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.swarmplot(x='session', y='abs_dist', data=df,
                  order=session_names, ax=ax, size=7,
                  color='r', edgecolor='k', linewidth=1)

    formatPlot(ax)

    ax.set_ylabel('Activity Centroid Distance to SZ (cm)', fontsize=14)
    # ax.tick_params(axis='x', which='major', labelsize=10)

    ax.axhline(grp[0].track_length / 40., ls='--', color='0.5')

    fig.savefig(save_path + '_activity_centroid_dist_box_by_cell.svg')

    # Stats
    pre_df = df.loc[df['session'] == 'Pre']
    post_df = df.loc[df['session'] == 'Post']

    pval = ttest_rel(pre_df['abs_dist'], post_df['abs_dist'])[1]
    print 'Pre Mean: {}, Post Mean: {}, T-Test p={}'.format(
        pre_df.mean()['abs_dist'], post_df.mean()['abs_dist'], pval)

    # Shift Toward Stim Zone
    shift_df = df.loc[(df['session'] == 'Post')].merge(
        df.loc[(df['session'] == 'Pre')],
        on=['roi', 'mouse_name'],
        suffixes=['_ind', '_base'])

    shift_df['shift'] = shift_df.apply(lambda x: x['abs_dist_base'] - x['abs_dist_ind'], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(y='shift', data=shift_df, showcaps=False,
                showfliers=False, whis=False, ax=ax)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    sns.swarmplot(y='shift', hue='mouse_name', data=shift_df, ax=ax, size=7,
                  color='w', edgecolor='k', linewidth=1)
    med = shift_df.median()['shift']
    ax.plot([-0.25, 0.25], [med, med], color='k')
    ax.plot([-0.5, 0.5], [0, 0], color='0.5', ls='--')

    formatPlot(ax)

    ax.set_ylabel('Shift Toward SZ (cm)', fontsize=18)
    ax.xaxis.set_tick_params(length=0)
    ratio = 4.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    # Stats
    pval = ttest_1samp(shift_df['shift'], 0)[1]
    print 'Shift p={}'.format(pval)

    fig.savefig(save_path + 'activity_centroid_shift_toward_zone_swarm_by_cell.svg')


def average_heatmap(grp, save_path=None, signal='spikes', z_score=False, ax=None, plot_com=False):

    heatmaps = ia.grp_heatmap(grp, signal=signal, z_score=z_score)

    shortest = min([len(x) for x in heatmaps]) - 1
    avg_heatmap = np.zeros((shortest, 100))
    for hm in heatmaps:
        avg_heatmap += np.nan_to_num(hm[:shortest, :])
    avg_heatmap /= float(len(heatmaps))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if signal == 'spikes':
        sns.heatmap(avg_heatmap, ax=ax, rasterized=True, vmax=np.max(avg_heatmap[15:, :]))
    else:
        sns.heatmap(avg_heatmap, ax=ax, rasterized=True, vmin=0,
                    vmax=np.nanpercentile(avg_heatmap, 99))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel('Laps', fontsize=10)
    ax.set_xlabel('Position (centered on stim)', fontsize=10)

    if plot_com:
        post_laps = np.arange(10, avg_heatmap.shape[0])
        coms = []
        for lap in post_laps:
            lap_coms = [calc_activity_centroid(np.nan_to_num(hm[lap, :])) for hm in heatmaps]
            coms.append(np.nanmedian(lap_coms))
        ax.plot(coms, post_laps + 0.5, color='w')

    if save_path:
        fig.savefig(save_path + 'average_heatmap_{}.svg'.format(signal))
    else:
        return ax

# Reused code from id_pf_spikes
def find_pfs(true_tc, confidence_tc,
             n_position_bins, min_run=5, max_nans=1):

    significant = true_tc > confidence_tc

    pfs = []

    for roi, roi_tc in zip(significant, true_tc):

        active_bins = np.where(roi)[0]
        runs = _consecutive_integers(active_bins)

        try:
            # Check for circular consecutive bins
            if (runs[-1][-1] == n_position_bins - 1) and (runs[0][0] == 0):
                runs[0] = runs[-1] + runs[0]
                del runs[-1]
        except IndexError:
            pass

        # PFs are start/stop of consecutive runs
        roi_pfs = [[r[0], r[-1]] for r in runs if len(r) >= min_run]

        pfs.append(roi_pfs)

    return pfs

# Pull out significant portion of pf width
def get_pf_width(expt):

    with open(expt.placeFieldsFilePath(signal='spikes'), 'rb') as fp:
        pfdict = pkl.load(fp)

    tc = pfdict['soma']['undemixed']['spatial_tuning_smooth']
    thresh = pfdict['soma']['undemixed']['confidence_thresh']

    pfs = find_pfs(tc, thresh, 100)

    stim_pos = expt._get_stim_positions(units='normalized')[0]

    closest_pfs = []

    for roi_pfs in pfs:
        min_dist = np.inf
        if not roi_pfs:
            closest_pfs.append(None)
            continue

        for pf in roi_pfs:
            if (stim_pos > pf[0]) and (stim_pos < pf[1]):
                closest_pf = pf
                break
            else:
                dist = min(abs(pf[0] - stim_pos), abs(pf[1] - stim_pos))
                if dist < min_dist:
                    min_dist = dist
                    closest_pf = pf

        closest_pfs.append(closest_pf[1] - closest_pf[0])

    return closest_pfs


def velocity_vs_width(grp, window=(3, 3), all_stims=False):

    data_list = []

    pfs = grp.pfs()
    for expt in grp:

        trial_id = expt.trial_id
        mouse_name = expt.parent.mouse_name
        rois = expt.rois()

        win_frames = int(1 / expt.frame_period())
        vel = expt.velocity(smoothing='flat')[0]
        stim_frames = expt._get_stim_frames()
        # Take mean velocity of all in-stim timepoints per lap
        if all_stims:
            stim_vel = np.mean([np.mean(vel[stim_frame - window[0] * win_frames:stim_frame + window[1] * win_frames + 1])
                                for stim_frame in stim_frames])
        else:
            stim_vel = np.mean(vel[stim_frames[0] - window[0] * win_frames:stim_frames[0] + window[1] * win_frames + 1])

        expt_pfs = pfs[expt]
        pf_widths = get_pf_width(expt)
        stim_loc = expt._get_stim_positions(units='normalized')[0]
        bin_to_cm = expt.track_length / 1000.
        for roi_pfs, roi, width in zip(expt_pfs, rois, pf_widths):
            for roi_pf in roi_pfs:
                if ia.pf_overlaps_stim([roi_pf], stim_loc):

                    data_list.append({'trial_id': trial_id,
                                      'mous_name': mouse_name,
                                      'roi': roi.label,
                                      'pf': roi_pf,
                                      'width': width * bin_to_cm,
                                      'vel': stim_vel,
                                      'window': window})
                    continue

    return pd.DataFrame(data_list)


def plot_velocity_vs_width(grp, save_path, **kwargs):

    df = velocity_vs_width(grp, **kwargs)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    scatter_kws = {'s': 40, 'color': 'w', 'edgecolors': 'k', 'zorder': 1000}
    line_kws = {'color': '0.5'}
    sns.regplot(y='width', x='vel', data=df, ci=0, truncate=True,
                scatter_kws=scatter_kws, line_kws=line_kws)

    formatPlot(ax)
    ax.set_xlabel('Peristim Velocity (cm/s)', fontsize=18)
    ax.set_ylabel('Place Field Width (cm)', fontsize=18)

    ratio = 1
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'vel_vs_width.svg', bbox_inches='tight')


def average_tuning_curve_plot(grp, save_path, offset=True):

    full_sessions = ['control_induction', 'control_24h', 'control_48h', 'control_72h']
    session_names = ['Pre', 'Post', '24h', '48h', '72h']
    tcs = []
    pre_tcs = []
    heatmaps = ia.grp_heatmap(grp)

    roi_count = 0
    for expt in grp:
        stim_loc = expt._get_stim_positions(units='normalized')[0]
        shift_back = 50 - int(stim_loc)

        nrois = len(expt.rois())
        for i in xrange(nrois):
            pre_tc = np.roll(np.nanmean(heatmaps[roi_count][:10, :], axis=0), shift_back)
            pre_tcs.append(smooth_tuning_curves([pre_tc], smooth_length=3, axis=1, nan_norm=False))
            roi_count += 1

    tcs.append(np.nanmean(pre_tcs, axis=0).squeeze())

    for session in full_sessions:
        sess_tcs = []
        sess_grp = [x._get_session(session) for x in grp]
        sess_grp = pcExperimentGroup(sess_grp, **grp.args)

        data = sess_grp.data()
        for expt in sess_grp:
            stim_loc = expt._get_session('control_induction')._get_stim_positions(units='normalized')[0]
            shift_back = 50 - int(stim_loc)
            sess_tcs.extend(np.roll(data[expt], shift_back, axis=1))
        tcs.append(np.nanmean(sess_tcs, axis=0) / expt.frame_period())

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if offset:
        offset_val = 0.07
    else:
        offset_val = 0
    colors = sns.cubehelix_palette(5, start=0.5, rot=-0.75, hue=1)
    for tc, color, session in zip(tcs, colors, session_names):
        ax.plot(tc + offset, color=color, label=session)
        if offset:
            offset_val -= 0.02

    ax.legend()
    sns.despine()
    ax.set_ylim([0, ax.get_ylim()[1]])

    fig.savefig(save_path + 'average_tc_by_day.svg')


def percent_success_by_day(grp, thresh=5):

    sessions = ['control_induction', 'control_24h',
               'control_48h', 'control_72h']
    session_names = ['Post', '24h', '48h', '72h']

    data_list = []
    data_list.append({'num': 0,
                      'pct': 0,
                      'session': 'Pre'})

    bin_to_cm = grp[0].track_length / 1000.
    cm_thresh = thresh * bin_to_cm

    last_subset = []
    for expt in grp:
        last_subset.extend([r.label for r in expt.rois()])

    n_rois = len(last_subset)

    for session, session_name in zip(sessions, session_names):

        sess_grp = [x._get_session(session) for x in grp]
        sess_grp = pcExperimentGroup(sess_grp, **grp.args)

        grp_pfs = sess_grp.pfs()

        # If centroid is within thresh dist of stim location
        # centroids = ia.centroid_distance(sess_grp, stim_session='control_induction')
        # centroids = centroids.loc[centroids['abs_dist'] <= cm_thresh]
        # centroid_thresh = centroids['roi'].values.tolist()

        # If PF overlaps stim location
        for expt in sess_grp:
            rois = expt.rois()
            pfs = grp_pfs[expt]
            stim_loc = expt._get_session('control_induction')._get_stim_positions(units='normalized')[0]
            for roi, pf in zip(rois, pfs):
                if ia.pf_overlaps_stim(pf, stim_loc):
                    labels.append(roi.label)

        labels = set(labels)

        persistent_labels = set([x for x in labels if x in last_subset])
        if last_subset:
            pct_of_previous = len(persistent_labels) / float(len(last_subset))
        else:
            pct_of_previous = np.nan

        data_list.append({'num': len(labels),
                          'pct': len(labels) / float(n_rois),
                          'session': session_name,
                          'pct_of_previous': pct_of_previous,
                          'pct_always': len(persistent_labels) / float(n_rois)})

        last_subset = persistent_labels

    return pd.DataFrame(data_list)


def success_by_day(grp, thresh=5):

    # Just return ROI names and yes or no

    sessions = ['control_induction', 'control_24h',
               'control_48h', 'control_72h']
    session_names = ['Post', '24h', '48h', '72h']

    data_list = []
    bin_to_cm = grp[0].track_length / 1000.
    cm_thresh = thresh * bin_to_cm

    for session, session_name in zip(sessions, session_names):

        sess_grp = [x._get_session(session) for x in grp]
        sess_grp = pcExperimentGroup(sess_grp, **grp.args)

        grp_pfs = sess_grp.pfs()

        # If centroid is within thresh dist of stim location
        # centroids = ia.centroid_distance(sess_grp, stim_session='control_induction')
        # centroids = centroids.loc[centroids['abs_dist'] <= cm_thresh]
        # centroid_thresh = centroids['roi'].values.tolist()

        # If PF overlaps stim location
        for expt in sess_grp:
            rois = expt.rois()
            pfs = grp_pfs[expt]
            stim_loc = expt._get_session('control_induction')._get_stim_positions(units='normalized')[0]
            for roi, pf in zip(rois, pfs):
                if (roi.label in centroid_thresh) or ia.pf_overlaps_stim(pf, stim_loc):

                    data_list.append({'expt': expt.trial_id,
                                      'roi': roi.label,
                                      'session': session_name,
                                      'induced': True})
                else:

                    data_list.append({'expt': expt.trial_id,
                                      'roi': roi.label,
                                      'session': session_name,
                                      'induced': False})

    return pd.DataFrame(data_list)


def plot_percent_success_by_day(grp, prefix):

    sessions = ['Pre', 'Post', '24h', '48h', '72h']

    pct_success = percent_success_by_day(grp)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot([0], [0], mec='k', color='w', marker='o', ms=7)
    vals = pct_success['pct'].values[1:]
    ax.plot(np.arange(1, len(sessions)), vals, marker='o',
            ms=7, mec='k', mfc='w', color='k')

    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions)

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

    formatPlot(ax)

    ax.set_ylabel('Fraction Cells with PF Near Stim', fontsize=18)
    ax.set_xlabel('Session', fontsize=18)

    # plt.tight_layout()

    ratio = 1.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(prefix + 'fraction_success_by_day.svg', bbox_inches='tight')


def left_shift(grp, ind_filter):

    data_list = []

    data = grp.data(roi_filter=ind_filter)
    all_pfs = grp.pfs(roi_filter=ind_filter)
    for expt in grp:

        stim_loc = expt._get_stim_positions(units='normalized')[0]
        bin_to_cm = expt.track_length / 1000.

        rois = expt.rois(roi_filter=ind_filter)
        for tc, pfs, roi in zip(data[expt], all_pfs[expt], rois):

            for pf in pfs:
                if ia.pf_overlaps_stim([pf], stim_loc):
                    if pf[0] < pf[1]:
                        peak_loc = np.argmax(tc[pf[0]:pf[1]]) + pf[0]
                    else:
                        idx1 = np.argmax(tc[pf[1]:]) + pf[1]
                        idx2 = np.argmax(tc[:pf[0]])

                        peak_loc = idx1 if tc[idx1] > tc[idx2] else idx2

            # peak_loc = np.argmax(tc)

            dist = ia.dist(peak_loc, stim_loc) * bin_to_cm

            data_list.append({'mouse': expt.parent.mouse_name,
                              'roi': roi.label,
                              'shift': dist})

    return pd.DataFrame(data_list)


def plot_left_shift(grp, ind_filter, prefix):

    left_shifts = left_shift(grp, ind_filter)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(y='shift', data=left_shifts, showcaps=False,
                showfliers=False, whis=False, ax=ax)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    sns.swarmplot(y='shift', hue='mouse', data=left_shifts, ax=ax, size=7,
                  color='w', edgecolor='k', linewidth=1)
    med = left_shifts.median()['shift']
    ax.plot([-0.25, 0.25], [med, med], color='k')
    ax.plot([-0.5, 0.5], [0, 0], color='0.5', ls='--')

    formatPlot(ax)

    ax.set_ylabel('PF Peak Relative to SZ (cm)', fontsize=18)
    ax.yaxis.set_tick_params(length=1)
    
    ratio = 4.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    # Stats
    pval = ttest_1samp(left_shifts['shift'], 0)[1]
    print 'Shift p={}'.format(pval)

    fig.savefig(prefix + 'peak_shift_swarm_by_cell.svg')


def tuning_difference(grp, roi_filter=None, normalize=False):

    """Return mean difference of tuning curves.
    """

    all_diffs = []

    data = grp.data(roi_filter=roi_filter)
    baseline_data = baseline_activity(grp, roi_filter=roi_filter)

    etcs = []
    btcs = []

    for expt in grp:

        stim_loc = expt._get_stim_positions(units='normalized')[0]
        shift = 50 - int(stim_loc)

        rois = expt.rois(roi_filter=roi_filter)

        # Divide by frame period - this step already happens
        # in the heatmap calc for baseline data
        edata = np.roll(data[expt], shift, axis=1) / expt.frame_period()

        diffs = edata - baseline_data[expt]

        # Convert from frames to seconds
        all_diffs.extend(diffs)
        etcs.append(edata)
        btcs.append(baseline_data[expt])

    return np.vstack(all_diffs), np.vstack(etcs), np.vstack(btcs)
