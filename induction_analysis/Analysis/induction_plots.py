import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_style(style='ticks')

from lab_repo.misc.misc import savefigs
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic, sem, ttest_rel, ttest_ind, pearsonr, ttest_1samp
import statsmodels.api as sm

import pandas as pd

import itertools as it

import sys
sys.path.insert(0, '../induction_analysis')

import Analysis.induction_analysis as ia
from Analysis import filters

from lab_repo.classes.place_cell_classes import pcExperimentGroup
from lab_repo.classes.dbclasses import dbExperiment
import la_repo.analysis.place_cell_analysis as pca
import lab_repo.analysis.behavior_analysis as ba
from lab_repo.misc.misc import preprocess_image, calc_cdf

from sima.ROI import mask2poly
from matplotlib.patches import Polygon, Circle


FILE_ENDING = 'svg'


def formatPlot(ax, offset=10):
    for axis in ['left', 'right', 'bottom', 'top']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(labelsize=14, length=2, width=4)
    sns.despine(ax=ax, offset=offset)
    ax.set_xlabel('')
    ax.set_ylabel('')


def normed(s, sfull=None):
    if sfull is None:
        sfull = s

    smax = np.nanmax(sfull)
    smin = np.nanmin(sfull)
    if smax == smin:
        return np.zeros(s.shape)
    return (s - smin) / (smax - smin)


def get_med(cdf_vals, bins):

    idx = np.where(cdf_vals == 0.5)[0]
    if idx:
        return bins[idx]

    # If there is no clear median, find two values on either side of 0.5
    else:
        high_vals = cdf_vals.copy()
        high_vals[high_vals < 0.5] = 0
        high_idx = np.where(high_vals)[0][0]
        high_val = high_vals[high_idx]
        high_bin = bins[high_idx]

        low_vals = cdf_vals.copy()
        low_vals[low_vals > 0.5] = 0
        try:
            low_idx = np.where(low_vals)[0][-1]
        except IndexError:
            return 0
        low_val = low_vals[low_idx]
        low_bin = bins[low_idx]

        high_diff = high_val - 0.5
        low_diff = 0.5 - low_val
        total_diff = high_diff + low_diff

        # Return average of bins weighted by proximity to 0.5
        # do this weighting by weighing by relative distance of other value
        toreturn = low_bin * (high_diff / total_diff) + high_bin * (low_diff / total_diff)

        return toreturn


def mad_vals(signal, s=None):
    if s is not None:
        signal_filtered = signal[s==0]
    else:
        signal_filtered = signal
    mad = np.nanmedian(np.abs(signal_filtered - np.nanmedian(signal_filtered)))

    return np.nanmedian(signal_filtered), mad


def average_heatmap(grp, save_path, roi_filter=None, signal='dff', z_score=False, vmax=99.9):

    heatmaps = ia.grp_heatmap(grp, roi_filter=roi_filter, signal=signal, z_score=z_score)

    shortest = min([len(x) for x in heatmaps])
    avg_heatmap = np.zeros((shortest, 100))
    for hm in heatmaps:
        avg_heatmap += np.nan_to_num(hm[:shortest, :])
    avg_heatmap /= float(len(heatmaps))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if signal == 'spikes':
        vmax = np.max(avg_heatmap[6:, :])
        sns.heatmap(avg_heatmap, ax=ax, rasterized=True, vmax=vmax)
    else:
        try:
            last_frame = grp[0].stim_lap_frames()[-1]
            abspos = ba.absolutePosition(grp[0].find('trial'))
            vmax_lap = int(abspos[last_frame + 1])
        except IndexError:
            vmax_lap = 0

        vmax = np.nanpercentile(avg_heatmap[vmax_lap:, :], vmax)

        sns.heatmap(avg_heatmap, ax=ax, rasterized=True, vmin=0,
                    vmax=vmax)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel('Laps', fontsize=14)
    ax.set_xlabel('Position (centered on stim)', fontsize=14)

    ax2 = ax.twinx()

    tcs = []
    data = grp.data(roi_filter=roi_filter)
    for expt in grp:
        stim_loc = expt._get_stim_positions(units='normalized')[0]
        shift = 50 - int(stim_loc)
        tcs.extend(np.roll(data[expt], shift, axis=1))
    tc = np.nanmean(tcs, axis=0)
    ax2.plot(tc, color='w')
    ax2.set_ylim([0, np.max(tc) * 2])
    ax2.set_xticks([])

    fig.savefig(save_path + 'average_heatmap.{}'.format(FILE_ENDING), bbox_inches='tight')


def dist_vs_amplitude(grp, save_path, roi_filter=None, savename=None,
                      y='zdiff', win=(1, 0.5), exclude_stim=True):

    amps = ia.burst_size(grp, roi_filter=roi_filter, win=win, exclude_stim=exclude_stim)
    dists = ia.dist_to_target(grp, stim_filter=roi_filter)

    full_df = dists.merge(amps, on=['mouse_name', 'expt_id', 'roi'])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Only look out as far as we have data for all mice
    # Value below is min max across all single cell stim experiments
    # max_dist = df.groupby('mouse_name').max()['dist'].min()
    max_dist = 168.5935987978145
    df = full_df.loc[full_df['dist'] <= max_dist]

    zeros = df.loc[df['dist'] == 0][y]
    rest = df.loc[df['dist'] > 0]
    means, bins, _ = binned_statistic(rest['dist'], rest[y],
                                          statistic=np.mean,
                                          bins=np.linspace(0, max_dist, 21))

    stds, bins, _ = binned_statistic(rest['dist'], rest[y],
                                          statistic=sem,
                                          bins=np.linspace(0, max_dist, 21))

    bin_size = bins[1] - bins[0]

    zero_mean = np.nanmean(zeros)
    zero_std = sem(zeros)

    means = np.concatenate([[zero_mean], means])
    stds = np.concatenate([[zero_std], stds])
    plot_bins = np.concatenate([[0], bins + (bins[1] - bins[0]) / 2.])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    color = sns.xkcd_palette(['pumpkin'])[0]
    ax.errorbar(plot_bins[:-1], means, yerr=stds, fmt='o', mfc=color, mec='k', ecolor='k', ms=3.5, elinewidth=0.5, mew=0.5)
    formatPlot(ax)
    ax.set_xlabel('Distance from Target (um)', fontsize=18)
    ax.set_ylabel('Mean Stim Response ($\sigma$)', fontsize=18)

    ax.set_ylim([-0.5, 5])

    fig.savefig(save_path + 'stim_response_vs_distance.{}'.format(FILE_ENDING), bbox_inches='tight')

    # Stats
    # For now just calculate by cell, compare to cells beyond stim radius
    first_bins = full_df.loc[(full_df['dist'] > 0) & (full_df['dist'] < bin_size)]
    far_bins = full_df.loc[full_df['dist'] > 75]
    pop_mean = full_df[y].mean()
    pval = ttest_ind(first_bins[y], far_bins[y])
    print 'Neighbor: Mean = {}, std = {}, bin = {}, Far: Mean = {}, std = {}, p={}'.format(
        first_bins[y].mean(), first_bins[y].std(), bin_size, far_bins[y].mean(), first_bins[y].std(), pval)

    # Alternatively, we can compare everything to far bins and then correct for multiple comparisons
    zero_bins = full_df.loc[full_df['dist'] == 0]
    pval = ttest_ind(zero_bins[y], far_bins[y])[1]
    print 'Zero: Mean = {}, std = {}, p={}'.format(
        zero_bins[y].mean(), zero_bins[y].std(), pval)

    count = 2
    for i in xrange(1, len(bins) - 1):

        if bins[i + 1] >= 75:
            break

        in_bins = full_df.loc[(full_df['dist'] >= bins[i]) & (full_df['dist'] < bins[i + 1])]
        pval = ttest_ind(in_bins[y], far_bins[y])
        print 'Bin [{}, {}): Mean = {}, std = {}, p={}'.format(
            bins[i], bins[i+1], in_bins[y].mean(), in_bins[y].std(), pval)
        count += 1

    print 'Multiple comparisons: p= 0.05 -> {}, p=0.01 -> {}, p=0.001 -> {}'.format(0.05 / count, 0.01 / count, 0.001 / count)




def burst_size_comparison(grp, save_path, stim_filter, ind_filter, y='zdiff', win=(1, 0.5), exclude_stim=True, cdf=False):

    ind_df = ia.burst_size(grp, roi_filter=ind_filter, win=win, exclude_stim=exclude_stim)
    ind_df['ind'] = 'Induced'

    fail_filter = lambda x: stim_filter(x) and not ind_filter(x)
    fail_df = ia.burst_size(grp, roi_filter=fail_filter, win=win)
    fail_df['ind'] = 'Not Induced'

    df = pd.concat([ind_df, fail_df])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if cdf:

        bins, cdf_vals = calc_cdf(ax, ind_df[y], color='r', label='Induced')
        ax.vlines(get_med(cdf_vals, bins), 0, 0.5, linestyles='-', color='r', alpha=0.5)

        bins, cdf_vals = calc_cdf(ax, fail_df[y], color='r', label='Not Induced', ls='--')
        ax.vlines(get_med(cdf_vals, bins), 0, 0.5, linestyles='--', color='r', alpha=0.5)

    else:

        sns.boxplot(x='ind', y=y, data=df, color='k',
                      order=['Induced', 'Not Induced'], ax=ax,
                      whis=False, showcaps=False, showfliers=False)
        plt.setp(ax.artists, edgecolor='k', facecolor='w')
        plt.setp(ax.lines, color='k')

        sns.swarmplot(x='ind', y=y, data=df, hue='ind', palette=['r', '0.3'],
                      order=['Induced', 'Not Induced'], hue_order=['Induced', 'Not Induced'],
                      ax=ax, size=7, edgecolor='k', linewidth=1)

    formatPlot(ax)
    ratio = 1.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    if cdf:
        ax.set_ylabel('Cumulative Probability', fontsize=18)
        ax.set_xlabel('Mean Stim Response', fontsize=18)
    else:
        ax.set_ylabel('Mean Stim Response', fontsize=18)

    if cdf:
        fig.savefig(save_path + 'burst_size_comparison_cdf.{}'.format(FILE_ENDING), bbox_inches='tight')
    else:
        fig.savefig(save_path + 'burst_size_comparison.{}'.format(FILE_ENDING), bbox_inches='tight')

    # Stats
    pval = ttest_ind(ind_df[y], fail_df[y])[1]

    print 'Burst Size ({}): Induced Mean={}, Not Induced Mean={}, p={}'.format(
        y, ind_df.mean()[y], fail_df.mean()[y], pval)


def tuning_difference_plot(grp, save_path, stim_filter, nostim_filter, **kwargs):

    stim_td = ia.tuning_difference(grp, roi_filter=stim_filter, **kwargs)
    nostim_td = ia.tuning_difference(grp, roi_filter=nostim_filter, **kwargs)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bin_to_cm = grp[0].track_length / 1000.

    nostim_mean = np.nanmean(nostim_td, axis=0)
    nostim_sem = sem(nostim_td, axis=0)

    stim_mean = np.nanmean(stim_td, axis=0)
    stim_sem = sem(stim_td, axis=0)

    x = np.arange(-50, 50) * bin_to_cm

    ax.plot(x, nostim_mean, '0.7')
    ax.fill_between(x, nostim_mean-nostim_sem, nostim_mean+nostim_sem, color='0.7', alpha=0.2)
    ax.plot(x, stim_mean, 'r')
    ax.fill_between(x, stim_mean-stim_sem, stim_mean+stim_sem, color='r', alpha=0.2)

    formatPlot(ax)
    ax.set_xlabel('Position', fontsize=18)
    ax.set_ylabel('Tuning Change (POST - PRE)', fontsize=18)

    ratio = 1
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'tuning_difference.{}'.format(FILE_ENDING), bbox_inches='tight')



def binarize_mask(roi, thres=0.2):
        mask = np.sum(roi.__array__(), axis=0)
        return mask > (np.max(mask) * thres)


def stim_target_example(expt, save_path, thresh=None, n_neighbors=None,
                        roi_filter=None, pre=3, post=6):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ta = expt.imaging_dataset().time_averages[0, ..., -1]

    ax.imshow(preprocess_image(ta), cmap='Greys_r')

    dists = ia.dist_to_target([expt], stim_filter=roi_filter)
    if thresh:
        dists = dists.loc[dists['dist'] <= thresh]
    elif n_neighbors:
        dists = dists.nsmallest(n_neighbors, 'dist')
    else:
        print 'Pass in either Threshold distance or N neighbors to plot'
        return

    rlabels = dists['roi'].tolist()
    print rlabels
    rfilter = lambda x: x.label in rlabels

    rois = expt.rois(roi_filter=rfilter)

    colors = sns.cubehelix_palette(n_neighbors, start=0.5, rot=-0.75, hue=1)

    patches = []
    for i, (roi, color) in enumerate(zip(rois, colors)):
        roi_poly = mask2poly(binarize_mask(roi).T)
        for poly in roi_poly:
            # Swap X and Y
            coords = np.roll(np.array(poly.exterior.coords)[:, :2], 1, axis=1)
            patches.append(Polygon(coords, closed=True, alpha=0.5, color=color))
            ax.add_patch(patches[-1])

    # Demarcate targeted region
    stim_loc = expt._get_stim_locations()[0, :]
    spiral_height = expt.spiral_sizes[0] * expt.frame_shape()[2]
    ec = sns.xkcd_rgb['pumpkin']

    ax.add_patch(Circle([stim_loc[1], stim_loc[0]], radius=spiral_height / 2.,
                 ec=ec, fc='None', lw=1.5))

    ax.set_axis_off()

    fig.savefig(save_path + 'targeted_time_average.png', bbox_inches='tight')

    ax.set_xlim(stim_loc[1] - 3 * spiral_height, stim_loc[1] + 3 * spiral_height)
    ax.set_ylim(stim_loc[0] + 3 * spiral_height, stim_loc[0] - 3 * spiral_height)

    fig.savefig(save_path + 'targeted_ta_zoom.png', bbox_inches='tight')

    # Get PSTHs
    psths = ia.psth(expt, roi_filter=rfilter, pre=pre, post=post)
    pre = int(pre / expt.frame_period())
    post = int(post / expt.frame_period())
    T = np.arange(-1 * pre, post) * expt.frame_period()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    offset = 0
    for roi_idx, (roi, color) in enumerate(zip(rois, colors)):
        # for stim_psth, trial_color in zip(psths[:, roi_idx, :], np.arange(0, 1, 0.15)):
        #     ax.plot(T, stim_psth + offset, color=str(trial_color))
        ax.plot(T, np.nanmean(psths[:, roi_idx, :], axis=0) + offset, color=color)
        offset += np.nanmax(psths[:, roi_idx, :]) + 3

    fig.savefig(save_path + 'targeted_example_psths.{}'.format(FILE_ENDING), bbox_inches='tight')


def activity_centroid_plots(grp, stim_filter, save_path, full_sessions, 
                            nostim_filter=None):

    # Plots comparing activity centroid distance to stim zone for
    # a set of sessions - for now has to be all control or all cno

    session_name_dict = {'induction': 'POST',
                         'baseline': 'PRE',
                         '24h': '24H'}

    if not nostim_filter:
        nostim_filter = lambda x: not stim_filter(x)

    df = pd.DataFrame([])

    for session in full_sessions:

        sess_grp = [x._get_session(session) for x in grp]
        sess_grp = pcExperimentGroup(sess_grp, **grp.args)

        stim_session = session.split('_')[0] + '_induction'

        for stimmed, sfilter in zip(['stimmed', 'unstimmed'], [stim_filter, nostim_filter]):

            sess_df = ia.activity_centroid_distance(sess_grp, roi_filter=sfilter, stim_session=stim_session)
            sess_df['stimmed'] = stimmed
            sess_df['session'] = session_name_dict[session.split('_')[-1]]

            df = pd.concat([df, sess_df])

    sessions = [session_name_dict[x.split('_')[-1]] for x in full_sessions]

    # Plot by Mouse (swarm + box)
    grp_df = df.groupby(['expt_id', 'session', 'stimmed'], as_index=False).mean()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(x='session', y='abs_dist', hue='stimmed', data=grp_df,
                  order=sessions, hue_order=['unstimmed', 'stimmed'], ax=ax,
                  whis=False, showcaps=False, showfliers=False)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.swarmplot(x='session', y='abs_dist', hue='stimmed', data=grp_df, palette=['0.7', 'r'],
                  dodge=True, order=sessions, hue_order=['unstimmed', 'stimmed'], ax=ax, size=7,
                  edgecolor='k', linewidth=1)

    formatPlot(ax)
    ax.set_ylabel('Activity Centroid Distance to SZ (cm)', fontsize=18)
    ax.get_legend().remove()

    fig.savefig(save_path + 'activity_centroid_dist_box_swarm_by_mouse.{}'.format(FILE_ENDING), bbox_inches='tight')

    # Stats
    # For now compare within session
    for session in sessions:
        stim_df = grp_df.loc[(grp_df['stimmed'] == 'stimmed') & (grp_df['session'] == session)]
        nostim_df = grp_df.loc[(grp_df['stimmed'] == 'unstimmed') & (grp_df['session'] == session)]
        pval = ttest_rel(stim_df['abs_dist'], nostim_df['abs_dist'])[1]

        print 'Activity Centroid Dist ({}): Stim Mean={}, No Stim Mean={}, p={}'.format(
            session, stim_df.mean()['abs_dist'], nostim_df.mean()['abs_dist'], pval)

    # Shift Toward Stim Zone
    shift_df = df.loc[(df['session'] == 'POST')].merge(
        df.loc[(df['session'] == 'PRE')],
        on=['roi', 'mouse_name', 'stimmed'],
        suffixes=['_ind', '_base'])

    shift_df['shift'] = shift_df.apply(lambda x: x['abs_dist_base'] - x['abs_dist_ind'], axis=1)

    # Shift by Mouse (box)
    grp_df = shift_df.groupby(['mouse_name', 'stimmed', 'expt_id_ind'], as_index=False).mean()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # box_kwargs = {'boxprops': {'facecolor': 'none'}}

    sns.boxplot(x='stimmed', y='shift', data=grp_df, color='k',
                order=['unstimmed', 'stimmed'], ax=ax,
                whis=False, showcaps=False, showfliers=False)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.swarmplot(x='stimmed', y='shift', hue='stimmed', data=grp_df, palette=['0.7', 'r'],
                  order=['unstimmed', 'stimmed'], hue_order=['unstimmed', 'stimmed'], ax=ax, size=7,
                  edgecolor='k', linewidth=1)

    ax.get_legend().remove()

    formatPlot(ax)
    ax.set_ylabel('Shift Toward SZ (cm)', fontsize=18)
    ax.xaxis.set_tick_params(length=0)
    ax.set_xticklabels(['Unstimmed', 'Stimmed'])

    ratio = 2.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'activity_centroid_shift_toward_zone_by_mouse.{}'.format(FILE_ENDING), bbox_inches='tight')

    # Stats
    stim_df = grp_df.loc[grp_df['stimmed'] == 'stimmed']
    nostim_df = grp_df.loc[grp_df['stimmed'] == 'unstimmed']
    pval = ttest_rel(stim_df['shift'], nostim_df['shift'])[1]

    print 'Activity Centroid Shift: Stim Mean={}, No Stim Mean={}, p={}'.format(
        stim_df.mean()['shift'], nostim_df.mean()['shift'], pval)


def activity_centroid_vs_nstimmed(grp, stim_filter, save_path):

    df = ia.activity_centroid_distance(grp, roi_filter=stim_filter, stim_session=grp[0].stim_session)
    df = df.groupby(['expt_id'], as_index=False).mean()

    def nstim(row):
        return len(dbExperiment(row['expt_id']).rois(roi_filter=stim_filter))

    df['n'] = df.apply(nstim, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.scatterplot(y='abs_dist', x='n', data=df, color='w', edgecolor='k', s=15, ax=ax)
    formatPlot(ax)
    ax.set_xlabel('N stimmed', fontsize=18)
    ax.set_ylabel('Activity Centroid Distance', fontsize=18)

    fig.savefig(save_path + 'activity_centroid_dist_vs_n_stimmed.{}'.format(FILE_ENDING), bbox_inches='tight')


def norm_pos(x, y):
    # return position of x in reference frame
    # where y is placed at 50 (all out of 100)

    dist = ia.dist(x, y)
    return 50 + dist

def single_cell_heatmap(expt, roi_label, save_path, signal='spikes'):

        fp = expt.frame_period()

        roi_filter = lambda x: x.label == roi_label

        if signal == 'spikes':
            spikes = expt.spikes(roi_filter=roi_filter, binary=True)[0, ...]
        else:
            spikes = expt.imagingData(dFOverF='from_file', roi_filter=roi_filter)[0, ..., 0]

        abs_pos = ba.absolutePosition(expt.find('trial'), imageSync=True)
        running = expt.velocity()[0] > 1

        if 'control' in expt.session:
            stim_loc = expt._get_session('control_induction')._get_stim_positions(units='normalized')[0]
        else:
            stim_loc = expt._get_session('cno_induction')._get_stim_positions(units='normalized')[0]
        shift = 50 - int(stim_loc)

        heatmap = ia.get_heatmap(spikes, running, abs_pos, n_bins=100, fp=fp)
        heatmap = np.roll(heatmap, shift, axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.heatmap(heatmap, ax=ax, cbar=True, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.vlines(50, 1, 6, colors='w', linestyles='dotted')

        fig.savefig(save_path + 'single_cell_heatmap.{}'.format(FILE_ENDING), bbox_inches='tight')


def success_rate_plots(grp, stim_filter, ind_filter, nostim_filter, nostim_ind_filter,
                       save_path, n_stimmed=False, n_success=False, box=False):

    ind_rate = ia.induction_rate(grp, stim_filter, ind_filter, nostim_filter, nostim_ind_filter)

    print ind_rate

    ind_rate = ind_rate.loc[ind_rate['n_stimmed'] > 0]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if n_stimmed:
        xcol = 'n_stimmed'
        xcol_label = 'Number Stimmed Cells'
    else:
        xcol = 'pct_stimmed'
        xcol_label = 'Fraction Stimmed'

    if n_success:
        ycol = 'n_success'
        ycol_label = 'Number Induced'
    else:
        ycol = 'pct_success'
        ycol_label = 'Induced Fraction'

    if box:
        sns.boxplot(y='pct_success', data=ind_rate,ax=ax,
                    whis=False, showcaps=False, showfliers=False)
        plt.setp(ax.artists, edgecolor='k', facecolor='w')
        plt.setp(ax.lines, color='k')

    sns.swarmplot(y='pct_success', data=ind_rate, color='r', edgecolor='k', ax=ax, size=7, linewidth=1)

    formatPlot(ax)

    [ymin, ymax] = ax.get_ylim()
    ax.set_ylim([min(0, ymin), max(1, ymax)])
    ax.set_ylabel('Induced Fraction', fontsize=18)
    ax.xaxis.set_tick_params(length=0)

    ratio = 2.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'induction_rate.{}'.format(FILE_ENDING), bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if box:
        sns.boxplot(y='fold', data=ind_rate, ax=ax,
                    whis=False, showcaps=False, showfliers=False)
        plt.setp(ax.artists, edgecolor='k', facecolor='w')
        plt.setp(ax.lines, color='k')

    ax.axhline(1, color='0.5')
    sns.swarmplot(y='fold', data=ind_rate, color='r', edgecolor='k', ax=ax, size=7, linewidth=1)

    formatPlot(ax)
    ax.set_ylabel('Induction Rate (Fold over Control)', fontsize=18)
    ax.xaxis.set_tick_params(length=0)

    ratio = 2.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'induction_rate_fold.{}'.format(FILE_ENDING), bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.scatterplot(x=xcol, y=ycol, data=ind_rate, color='r', edgecolor='k', s=30, ax=ax)
    formatPlot(ax)
    ax.set_ylim([-0.1, ax.get_ylim()[1]])
    ax.set_ylabel(ycol_label, fontsize=18)
    ax.set_xlabel(xcol_label, fontsize=18)

    fig.savefig(save_path + 'induction_rate_vs_{}.{}'.format(xcol, FILE_ENDING), bbox_inches='tight')

    corr = pearsonr(ind_rate[xcol], ind_rate[ycol])
    print '{} V {}: r={}, p={}'.format(ycol, xcol, corr[0], corr[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.scatterplot(x=xcol, y='fold', data=ind_rate, color='r', edgecolor='k', s=30, ax=ax)
    formatPlot(ax)
    ax.set_ylim([-1, ax.get_ylim()[1]])
    ax.set_ylabel('Induction Rate (Fold over Control)', fontsize=18)
    ax.set_xlabel(xcol_label, fontsize=18)

    fig.savefig(save_path + 'induction_fold_vs_{}.{}'.format(xcol, FILE_ENDING), bbox_inches='tight')


def plot_percent_success_by_day(grp, stim_filter, save_path):

    sessions = ['PRE', 'POST', '24H']

    vals = []

    ind_filter = filters.grp_induction_filter(grp, stim_filter=stim_filter, overwrite=False)
    df = ia.success_rate(grp, ind_filter, stim_filter)
    vals.append(df['n_success'].sum() / float(df['n_stimmed'].sum()))

    # 24h group
    next_session = grp[0].session.split('_')[0] + '_24h'
    next_grp = [x._get_session(next_session) for x in grp]
    next_grp = pcExperimentGroup(next_grp, label=grp.label() + '_24h', **grp.args)
    ind_filter = filters.grp_induction_filter(next_grp, stim_filter=stim_filter, overwrite=False, stim_session=grp[0].session)

    df = ia.success_rate(next_grp, ind_filter, stim_filter)
    vals.append(df['n_success'].sum() / float(df['n_stimmed'].sum()))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # By definition no pfs during pre, they have already been excluded
    ax.plot([0], [0], mec='k', color='w', marker='o', ms=7)
    ax.plot(np.arange(1, len(sessions)), vals, marker='o',
            ms=7, mec='k', mfc='w', color='k')

    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions)

    ax.set_yticks([0, 0.2, 0.4, 0.6])

    formatPlot(ax)

    ax.set_ylabel('Fraction Cells with PF Near Stim', fontsize=18)


    ratio = 2
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'fraction_success_by_day.{}'.format(FILE_ENDING), bbox_inches='tight')

