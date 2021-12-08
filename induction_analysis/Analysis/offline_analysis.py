import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.stats import pearsonr

import pandas as pd

import sys
sys.path.insert(0, '/home/sebi/code/analysis/analysis-scripts/induction_analysis')
from Analysis.induction_analysis import dist, df

import lab.analysis.place_cell_analysis as pca


def downsample(arr, factor, agg_func=np.nanmean):

    arr = arr.astype(float)

    extra_pad = factor - (arr.shape[1] % factor)
    arr = np.pad(arr, [(0, 0), (0, extra_pad)], mode='constant', constant_values=np.nan)

    arr = arr.reshape(arr.shape[0], -1, factor)

    arr = agg_func(arr, axis=-1)

    return arr


def downsample_1d(arr, factor, agg_func=np.nanmean):

    arr = arr.astype(float)

    extra_pad = factor - (arr.shape[0] % factor)
    arr = np.pad(arr, (0, extra_pad), mode='constant', constant_values=np.nan)

    arr = arr.reshape(-1, factor)

    arr = agg_func(arr, axis=-1)

    return arr


def prep_data(expt, roi_filter=None, bin_size=0.1, run=False,
              signal='spikes'):

    if signal == 'spikes':
        spikes = expt.spikes(binary=False, roi_filter=roi_filter)
        # Smooth and binarize
        spikes = gaussian_filter(spikes, (0, 1))
        spikes[spikes > 0] = 1
    else:
        spikes = df(expt, roi_filter=roi_filter).squeeze()

    # Bin at ~100 ms
    bin_size = int(bin_size / expt.frame_period())
    if bin_size < 1:
        bin_size = 1

    # Select running intervals and filter nans
    try:
        running = expt.runningIntervals(returnBoolList=True)[0]
    except KeyError:
        running = np.zeros((spikes.shape[1],), dtype=bool)

    spikes = downsample(spikes, bin_size)
    running = downsample_1d(running, bin_size)

    # Any bin that was majority running we'll include for now
    running[np.where(running < 0.5)] = 0
    running[np.where(running > 0)] = 1
    running = running.astype(bool)

    if run:
        spikes = spikes[:, running]
    else:
        spikes = spikes[:, ~running]

    return spikes


def nanz(arr):

    means = np.nanmean(arr, axis=1, keepdims=True)
    stds = np.nanstd(arr, axis=1, keepdims=True)

    return (arr - means) / stds


def offline_population_correlation_change(grp, stim_filter, pc_filter,
                                          ind_filter, bin_size=0.1):
    # Correlation to mean of all other cells

    data_list = []

    for expt in grp:

        pre = expt._get_session('control_pre')
        post = expt._get_session('control_post')

        pre_corrs = []
        post_corrs = []

        for sess_i, sess in enumerate([pre, post]):

            corrs = []

            spikes = prep_data(sess, bin_size=bin_size)
            zspikes = nanz(spikes)

            for i in xrange(spikes.shape[0]):
                corr = pearsonr(zspikes[i, :], np.nanmean(np.delete(zspikes, i, axis=0), axis=0))[0]
                corrs.append(corr)

            if sess_i == 0:
                pre_corrs.extend(corrs)
            else:
                post_corrs.extend(corrs)

        rois = expt.rois()

        mouse_name = expt.parent.mouse_name
        expt_id = expt.trial_id

        for pre_corr, post_corr, roi in zip(pre_corrs, post_corrs, rois):

            data_list.append({'expt_id': expt_id,
                              'mouse': mouse_name,
                              'roi': roi.label,
                              'stimmed': stim_filter(roi),
                              'pc': pc_filter(roi),
                              'ind': ind_filter(roi),
                              'pre_corr': pre_corr,
                              'post_corr': post_corr,
                              'corr_diff': post_corr - pre_corr})

    return pd.DataFrame(data_list)


def nancorr(a, b):

    nans = np.isnan(a) + np.isnan(b)
    return pearsonr(a[~nans], b[~nans])[0]


def pf_dist(c1, c2):

    return np.min([dist(a, b) for a in c1 for b in c2])


def pairwise_correlations(grp, stim_filter, ind_filter, pc_filter, **kwargs):
    # All pairwise correlations

    data_list = []
    for expt in grp:
        print expt.parent.mouse_name

        tid = expt.trial_id
        mouse_name = expt.parent.mouse_name

        spikes = prep_data(expt, **kwargs)
        rois = expt.rois()

        for i in xrange(len(rois)):
            for j in xrange(i):
                corr = nancorr(spikes[i, :], spikes[j, :])

                r1 = rois[i]
                r2 = rois[j]

                if ind_filter(r1) and ind_filter(r2):
                    pair = 'ind-ind'
                elif stim_filter(r1) and stim_filter(r2):
                    pair = 'stim-stim'
                elif (ind_filter(r1) and pc_filter(r2)) or (pc_filter(r1) and ind_filter(r2)):
                    pair = 'ind-pc'
                elif (stim_filter(r1) and pc_filter(r2)) or (pc_filter(r1) and stim_filter(r2)):
                    pair = 'stim-pc'
                elif (ind_filter(r1) and not pc_filter(r2)) or (not pc_filter(r1) and ind_filter(r2)):
                    pair = 'ind-nonpc'
                elif (stim_filter(r1) and not pc_filter(r2)) or (not pc_filter(r1) and stim_filter(r2)):
                    pair = 'stim-nonpc'
                elif pc_filter(r1) and pc_filter(r2):
                    pair = 'pc-pc'
                elif (pc_filter(r1) and not pc_filter(r2)) or (not pc_filter(r1) and pc_filter(r2)):
                    pair = 'pc-nonpc'
                else:
                    pair = 'nonpc-nonpc'

                data_list.append({'mouse': mouse_name,
                                  'expt_id': tid,
                                  'roi_1': r1.label,
                                  'roi_2': r2.label,
                                  'pair': pair,
                                  'corr': corr})

    return pd.DataFrame(data_list)


def tuning_similarity(grp, pc_filter):

    # TODO: move ti ia?

    pfs = grp.pfs()
    data = grp.data()

    data_list = []

    for expt in grp:

        mouse_name = expt.parent.mouse_name
        trial_id = expt.trial_id

        centroids = pca.calcCentroids(data[expt], pfs[expt], returnAll=True)
        rois = expt.rois()

        for i in xrange(len(rois)):
            for j in xrange(i):

                r1 = rois[i]
                r2 = rois[j]

                tuning_corr = nancorr(data[expt][i, :], data[expt][j, :])
                centroid_dist = np.nan

                # TODO: use ind_filter and only look for closest to stim?
                if pc_filter(r1) and pc_filter(r2):
                    centroid_dist = pf_dist(centroids[i], centroids[j])

                data_list.append(
                    {'mouse': mouse_name,
                     'expt_id': trial_id,
                     'roi_1': r1.label,
                     'roi_2': r2.label,
                     'tuning_corr': tuning_corr,
                     'centroid_dist': centroid_dist})

    return pd.DataFrame(data_list)


def offline_pairwise_correlation_change(grp, stim_filter, pc_filter, ind_filter, **kwargs):

    pre = [x._get_session('control_pre') for x in grp]
    post = [x._get_session('control_post') for x in grp]

    pre_df = pairwise_correlations(pre, stim_filter, pc_filter, ind_filter, **kwargs)
    post_df = pairwise_correlations(post, stim_filter, pc_filter, ind_filter, **kwargs)

    # Have to just join row-wise for computation reasons
    # Drop redundant info from post_df so these dont get suffixes
    post_df = post_df.drop(columns=['mouse', 'pair', 'roi_1', 'roi_2'])
    df = pre_df.join(post_df, how='left', lsuffix='_pre', rsuffix='_post')
#
    # df = pre_df.merge(post_df, on=['mouse', 'roi_1', 'roi_2', 'pair'], suffixes=['_pre', '_post'])
    df['diff'] = df['corr_post'] - df['corr_pre']

    return df
