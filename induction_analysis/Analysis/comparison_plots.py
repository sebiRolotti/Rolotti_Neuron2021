import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='ticks')

import sys
sys.path.insert(0, '../induction_analysis')
import Analysis.induction_analysis as ia

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, sem, pearsonr, spearmanr

from lab_repo.classes.dbclasses import dbExperiment
from lab_repo.classes import pcExperimentGroup

FILE_ENDING = 'svg'


def formatPlot(ax, offset=10):
    for axis in ['left', 'right', 'bottom', 'top']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(labelsize=14)
    sns.despine(ax=ax, offset=offset)
    ax.set_xlabel('')
    ax.set_ylabel('')


def activity_centroid_vs_nstimmed(grps, stim_filters, save_path, labels=None, colors=None, n_stimmed=True):

    if labels is None:
        labels = [x.label() for x in grps]

    # def nstim(row):
    #     return len(dbExperiment(row['expt_id']).rois(roi_filter=stim_filter))

    dfs = []
    for grp, stim_filter, label in zip(grps, stim_filters, labels):
        df = ia.activity_centroid_distance(grp, roi_filter=stim_filter, stim_session=grp[0].stim_session)
        grouped_df = df.groupby(['mouse_name', 'expt_id']).mean()
        if n_stimmed:
            count_df = df.groupby(['mouse_name', 'expt_id']).count()
            grouped_df['count'] = count_df['abs_dist']
        else:
            stim_df = ia.success_rate(grp, stim_filter, stim_filter)
            grouped_df = grouped_df.merge(stim_df, left_on=['mouse_name', 'expt_id'],
                                          right_on=['mouse', 'expt'])
            grouped_df['count'] = grouped_df['pct_stimmed'] * 100
        grouped_df['group'] = label
        # df['n'] = df.apply(nstim, axis=1)
        corr = pearsonr(grouped_df['abs_dist'], grouped_df['count'])
        print 'Activity Centroid V N Stimmed: r={}, p={}'.format(corr[0], corr[1])
        corr = spearmanr(grouped_df['abs_dist'], grouped_df['count'])
        print 'Activity Centroid V N Stimmed: spearman r={}, p={}'.format(corr[0], corr[1])
        dfs.append(grouped_df)

    df = pd.concat(dfs)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.scatterplot(y='abs_dist', x='count', hue='group', palette=colors,
                    data=df, linewidth=1, edgecolor='k', s=40, ax=ax)

    formatPlot(ax)
    ratio = 1.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    legend = ax.get_legend()
    if legend:
        legend.remove()

    if n_stimmed:
        ax.set_xlabel('N stimmed', fontsize=18)
        fpath_val = 'n_stimmed'
    else:
        ax.set_xlabel('\% stimmed', fontsize=18)
        fpath_val = 'frac_stimmed'
    ax.set_ylabel('Activity Centroid Distance', fontsize=18)

    # Plot trendlines
    max_count = df['count'].max()

    for i, df in enumerate(dfs):
        pfit = np.polyfit(x=dfs[i]['count'].values[:], y=dfs[i]['abs_dist'].values[:], deg=1)
        x = np.linspace(0, max_count, 3)
        y = x * pfit[0] + pfit[1]
        ax.plot(x, y, color=colors[i], zorder=0, ls='--')

    fig.savefig(save_path + 'activity_centroid_dist_vs_{}.{}'.format(fpath_val, FILE_ENDING), bbox_inches='tight')


def activity_centroid_shift_paired(grps, stim_filters, save_path,
                            sessions=['baseline', 'induction'],
                            labels=None, colors=None):

    df = pd.DataFrame([])

    for grp, stim_filter, label in zip(grps, stim_filters, labels):
        condition = grp[0].session.split('_')[0]
        for session in sessions:

            sess_grp = [x._get_session(condition + '_' + session) for x in grp]
            sess_grp = pcExperimentGroup(sess_grp, **grp.args)

            stim_session = condition + '_induction'

            sess_df = ia.activity_centroid_distance(sess_grp, roi_filter=stim_filter, stim_session=stim_session)
            sess_df['condition'] = label
            sess_df['session'] = session

            df = pd.concat([df, sess_df])

    shift_df = df.loc[(df['session'] == 'induction')].merge(
        df.loc[(df['session'] == 'baseline')],
        on=['roi', 'mouse_name', 'condition'],
        suffixes=['_ind', '_base'])

    shift_df['shift'] = shift_df.apply(lambda x: x['abs_dist_base'] - x['abs_dist_ind'], axis=1)

    grp_df = shift_df.groupby(['mouse_name', 'condition', 'expt_id_ind'], as_index=False).mean()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(x='condition', y='shift', data=grp_df, color='k',
                order=labels, ax=ax,
                whis=False, showcaps=False, showfliers=False)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.swarmplot(x='condition', y='shift', hue='condition', data=grp_df, palette=colors,
                  order=labels, hue_order=labels, ax=ax, size=7,
                  edgecolor='k', linewidth=1)

    ax.get_legend().remove()

    # sns.boxplot(y='shift', x='stimmed_ind', data=grp_df,
    #             order=['unstimmed', 'stimmed'], palette=['0.7', 'r'], ax=ax,
    #             width=0.5, linewidth=1.5)

    formatPlot(ax)
    ax.set_ylabel('Shift Toward Stim Zone (cm)', fontsize=18)
    ax.xaxis.set_tick_params(length=0)
    ax.set_xticklabels(labels)

    ratio = 2.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'activity_centroid_shift_comparison.{}'.format(FILE_ENDING), bbox_inches='tight')

    # Stats
    control_df = grp_df.loc[grp_df['condition'] == labels[0]].dropna()
    cno_df = grp_df.loc[grp_df['condition'] == labels[1]].dropna()
    pval = ttest_ind(control_df['shift'], cno_df['shift'])[1]

    print 'Activity Centroid Shift: Control Mean={}, CNO Mean={}, p={}'.format(
        control_df.mean()['shift'], cno_df.mean()['shift'], pval)


def activity_centroid_plots(grps, stim_filters, nostim_filters, save_path, full_sessions, labels=None, colors=None):

    # Plots comparing activity centroid distance to stim zone for
    # a set of sessions - for now has to be all control or all cno

    session_name_dict = {'induction': 'POST',
                         'baseline': 'PRE',
                         '24h': '24H'}

    df = pd.DataFrame([])

    for grp, stim_filter, nostim_filter, label in zip(grps, stim_filters, nostim_filters, labels):

        for session in full_sessions:

            if session.startswith('_'):
                session = grp[0].session.split('_')[0] + session

            sess_grp = [x._get_session(session) for x in grp]
            sess_grp = pcExperimentGroup(sess_grp, **grp.args)

            stim_session = session.split('_')[0] + '_induction'

            sess_df = ia.activity_centroid_distance(sess_grp, roi_filter=stim_filter, stim_session=stim_session)
            sess_df['stimmed'] = 'stimmed'
            sess_df['condition'] = label
            sess_df['session'] = session_name_dict[session.split('_')[-1]]

            df = pd.concat([df, sess_df])

            sess_df = ia.activity_centroid_distance(sess_grp, roi_filter=nostim_filter, stim_session=stim_session)
            sess_df['stimmed'] = 'unstimmed'
            sess_df['condition'] = label
            sess_df['session'] = session_name_dict[session.split('_')[-1]]

            df = pd.concat([df, sess_df])

    sessions = [session_name_dict[x.split('_')[-1]] for x in full_sessions]

    # Plot by Mouse (swarm + box)
    grp_df = df.groupby(['expt_id', 'session', 'condition', 'stimmed'], as_index=False).mean()
    plot_df = grp_df.loc[grp_df['stimmed'] == 'stimmed']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(x='session', y='abs_dist', hue='condition', data=plot_df,
                  order=sessions, hue_order=labels, ax=ax,
                  whis=False, showcaps=False, showfliers=False)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.swarmplot(x='session', y='abs_dist', hue='condition', data=plot_df, palette=colors,
                  dodge=True, order=sessions, hue_order=labels, ax=ax, size=7,
                  edgecolor='k', linewidth=1)

    formatPlot(ax)
    ax.set_ylabel('Activity Centroid Distance to SZ (cm)', fontsize=18)
    ax.get_legend().remove()

    fig.savefig(save_path + 'activity_centroid_dist_box_swarm_by_mouse.{}'.format(FILE_ENDING), bbox_inches='tight')

    # Stats
    # For now compare within session
    # TODO how to handle cases with no activity during baseline?
    for session in sessions:
        stim_df_1 = grp_df.loc[(grp_df['stimmed'] == 'stimmed') & (grp_df['session'] == session) & (grp_df['condition'] == labels[0])]
        nostim_df_1 = grp_df.loc[(grp_df['stimmed'] == 'unstimmed') & (grp_df['session'] == session) & (grp_df['condition'] == labels[0])]

        stim_df_2 = grp_df.loc[(grp_df['stimmed'] == 'stimmed') & (grp_df['session'] == session) & (grp_df['condition'] == labels[1])]
        nostim_df_2 = grp_df.loc[(grp_df['stimmed'] == 'unstimmed') & (grp_df['session'] == session) & (grp_df['condition'] == labels[1])]

        pval = ttest_rel(stim_df_1['abs_dist'], nostim_df_1['abs_dist'])[1]
        print 'Activity Centroid Dist ({}, {}): Stim Mean={}, No Stim Mean={}, p={}'.format(
            session, labels[0], stim_df_1.mean()['abs_dist'], nostim_df_1.mean()['abs_dist'], pval)

        pval = ttest_rel(stim_df_2['abs_dist'], nostim_df_2['abs_dist'])[1]
        print 'Activity Centroid Dist ({}, {}): Stim Mean={}, No Stim Mean={}, p={}'.format(
            session, labels[1], stim_df_2.mean()['abs_dist'], nostim_df_2.mean()['abs_dist'], pval)

        pval = ttest_ind(stim_df_1['abs_dist'], stim_df_2['abs_dist'])[1]
        print 'Activity Centroid Dist ({}, Stim Only): {} Mean={}, {} Mean={}, p={}'.format(
            session, labels[0], labels[1], stim_df_1.mean()['abs_dist'], stim_df_2.mean()['abs_dist'], pval)


def success_rate_paired(grps, stim_filters, ind_filters, nostim_filters, nostim_ind_filters,
                        save_path, labels=None, colors=None):

    dfs = []
    for grp, label, a, b, c, d in zip(grps, labels, stim_filters, ind_filters, nostim_filters, nostim_ind_filters):

        ind_rate = ia.induction_rate(grp, a, b, c, d)
        ind_rate['condition'] = label

        dfs.append(ind_rate)

    df = pd.concat(dfs)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(x='condition', y='pct_success', data=df, color='k',
                  order=labels, ax=ax,
                  whis=False, showcaps=False, showfliers=False)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.swarmplot(x='condition', y='pct_success', hue='condition', data=df,
                  order=labels, hue_order=labels, palette=colors,
                  edgecolor='k', ax=ax, size=7, linewidth=1)

    formatPlot(ax)

    [ymin, ymax] = ax.get_ylim()
    ax.set_ylim([min(0, ymin), max(1, ymax)])
    ax.set_ylabel('Induced Fraction', fontsize=18)
    ax.xaxis.set_tick_params(length=0)
    ax.get_legend().remove()

    ratio = 2.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'induction_rate_comparison.{}'.format(FILE_ENDING), bbox_inches='tight')

    # Induction Fold
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(x='condition', y='fold', data=df, color='k',
                  order=labels, ax=ax,
                  whis=False, showcaps=False, showfliers=False)
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.swarmplot(x='condition', y='fold', hue='condition', data=df,
                  order=labels, hue_order=labels, palette=colors,
                  edgecolor='k', ax=ax, size=7, linewidth=1)

    formatPlot(ax)

    ax.axhline(1, color='0.5')
    ax.set_ylabel('Induction Rate (Fold Over Control)', fontsize=18)
    ax.xaxis.set_tick_params(length=0)
    ax.get_legend().remove()

    ratio = 2.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    fig.savefig(save_path + 'induction_fold_comparison.{}'.format(FILE_ENDING), bbox_inches='tight')


def tuning_difference(grps, stim_filters, save_path, labels=None, colors=None, **kwargs):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bin_to_cm = grps[0][0].track_length / 1000.
    x = np.arange(-50, 50) * bin_to_cm

    for grp, stim_filter, label, color in zip(grps, stim_filters, labels, colors):
        condition = grp[0].session.split('_')[0]

        td = ia.tuning_difference(grp, roi_filter=stim_filter,
                                  session1=condition + '_baseline',
                                  session2=condition + '_induction',
                                  by_mouse=True, **kwargs)

        td_mean = np.nanmean(td, axis=0)
        td_sem = sem(td, axis=0)

        ax.fill_between(x, td_mean - td_sem, td_mean + td_sem, color=color, alpha=0.2)
        ax.plot(x, td_mean, color=color)

    formatPlot(ax)
    ax.set_xlabel('Position', fontsize=18)
    ax.set_ylabel('Tuning Change (Post - Pre)', fontsize=18)

    fig.savefig(save_path + 'tuning_difference_comparison.{}'.format(FILE_ENDING), bbox_inches='tight')


def delta_centroid_vs_nstimmed(grps, stim_filters, save_path, labels=None, colors=None, n_stimmed=True):

    if labels is None:
        labels = [x.label() for x in grps]

    dfs = []
    for grp, stim_filter, label in zip(grps, stim_filters, labels):
        df = ia.delta_centroid_distance(grp, roi_filter=stim_filter)
        grouped_df = df.groupby(['expt_id', 'mouse_name'], as_index=False).mean()
        if n_stimmed:
            count_df = df.groupby(['mouse_name', 'expt_id']).count()
            grouped_df['count'] = count_df['abs_dist']
        else:
            stim_df = ia.success_rate(grp, stim_filter, stim_filter)
            grouped_df = grouped_df.merge(stim_df, left_on=['mouse_name', 'expt_id'],
                                          right_on=['mouse', 'expt'])
            grouped_df['count'] = grouped_df['pct_stimmed']
        grouped_df['group'] = label
        dfs.append(grouped_df)

    df = pd.concat(dfs)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.scatterplot(y='abs_dist', x='count', hue='group', palette=colors, data=df, linewidth=0, s=25, ax=ax)
    sns.despine()
    if n_stimmed:
        ax.set_xlabel('N stimmed', fontsize=10)
        fpath_val = 'n_stimmed'
    else:
        ax.set_xlabel('Fraction stimmed', fontsize=10)
        fpath_val = 'frac_stimmed'
    ax.set_ylabel('Tuning Change Centroid Distance', fontsize=10)

    fig.savefig(save_path + 'delta_centroid_dist_vs_{}.{}'.format(fpath_val, FILE_ENDING))


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def density_success_comparison(grps, stim_filters, ind_filters,
                               comparison_func=ia.success_rate,
                               labels=None, colors=None,
                               savename=None,
                               n_stimmed=False, n_success=False):

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
        ycol_label = 'Induction Rate'

    full_df = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for grp, stim_filter, ind_filter, label, color in \
            zip(grps, stim_filters, ind_filters, labels, colors):

        df = comparison_func(grp, ind_filter, stim_filter)

        sns.scatterplot(x=xcol, y=ycol,
                        data=df, color=color, s=30, ax=ax,
                        linewidth=1, edgecolor='k', label=label)

        # x_fit = np.hstack([[0], df[xcol].to_numpy()])
        # y_fit = np.hstack([[7/9.], df[ycol].to_numpy()])

        # popt, _ = curve_fit(exp_func, x_fit, y_fit,
        #                     p0=(1, 0.1, 0), bounds=([0, 0, 0], [1, 100, 1]))

        # xn = np.arange(0, 0.20, 0.01)
        # ax.plot(xn, exp_func(xn, *popt), color=color, ls='--')

        full_df.append(df)

    full_df = pd.concat(full_df)
    # full_df.to_pickle('/home/sebi/success_rate_comparison_df.pkl')

    # ax.legend()
    formatPlot(ax)
    ax.set_ylabel(ycol_label, fontsize=18)
    ax.set_xlabel(xcol_label, fontsize=18)

    fig.savefig(savename + '.{}'.format(FILE_ENDING), bbox_inches='tight')


def stim_gain_box(grps, prefix, stim_filters, value='diff', labels=None, colors=None, **kwargs):

    dfs = []
    for grp, stim_filter, label in zip(grps, stim_filters, labels):

        df = ia.stim_gain(grp, roi_filter=stim_filter, **kwargs)

        df = df.groupby(['mouse_name', 'expt_id']).mean()
        df['label'] = label
        dfs.append(df)

    grp_df = pd.concat(dfs)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.boxplot(x='label', y=value, data=grp_df,
                order=labels, ax=ax,
                  whis=False, showcaps=False, showfliers=False, color='k')
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')

    sns.swarmplot(x='label', y=value, hue='label', data=grp_df, palette=colors,
                  order=labels, hue_order=labels, ax=ax, size=7,
                  edgecolor='k', linewidth=1)

    formatPlot(ax)
    ratio = 1.5
    ax.set_aspect(ratio / ax.get_data_ratio())

    legend = ax.get_legend()
    if legend:
        legend.remove()

    save_str = '{}_vs_nostim.{}'
    ylabel = 'Relative Tuning'
    if kwargs.get('normalize', False):
        save_str = 'normalized_' + save_str
        ylabel = 'Relative Normalized Tuning'
    if kwargs.get('diff', False):
        save_str = 'tc_delta_' + save_str
        ylabel = ylabel + ' Change'

    ax.set_ylabel(ylabel, fontsize=18)

    fig.savefig(prefix + save_str.format(value, FILE_ENDING), bbox_inches='tight')

    # Stats
    pval = ttest_ind(dfs[0][value], dfs[1][value])[1]

    print 'Stim Gain {}, {} Mean={} std={}, {} Mean={} std={}, p={}'.format(
        ylabel, labels[0], dfs[0].mean()[value], dfs[0].std()[value],
        labels[1], dfs[1].mean()[value], dfs[1].std()[value], pval)


def stim_gain_comparison_plot(grps, prefix, stim_filters, dist_thresh=None, value='diff', colors=None,
                              n_stimmed=True, **kwargs):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if not colors:
        colors = ['grey', 'coral']
        colors = sns.xkcd_palette(colors)

    for grp, stim_filter, color in zip(grps, stim_filters, colors):

        if not n_stimmed:
            stim_df = ia.success_rate(grp, stim_filter, stim_filter)

        gains = ia.stim_gain(grp, roi_filter=stim_filter, **kwargs)
        amps = ia.burst_size(grp, roi_filter=stim_filter)
        df = gains.merge(amps, on=['expt_id', 'roi', 'mouse_name'], suffixes=['', '_amps'])

        if dist_thresh:
            dists = ia.dist_to_target(grp, stim_filter=stim_filter)
            dists = dists.loc[dists['dist'] < dist_thresh]
            df = df.merge(dists, on=['expt_id', 'roi', 'mouse_name'])

        grouped_df = df.groupby(['mouse_name', 'expt_id']).mean()
        sem_df = df.groupby(['mouse_name', 'expt_id']).sem()
        grouped_df['sem'] = sem_df[value]

        if n_stimmed:
            count_df = df.groupby(['mouse_name', 'expt_id']).count()
            grouped_df['count'] = count_df[value]
        else:
            grouped_df = grouped_df.merge(stim_df, left_on=['mouse_name', 'expt_id'],
                                          right_on=['mouse', 'expt'])
            grouped_df['count'] = grouped_df['pct_stimmed']

        ax.errorbar(grouped_df['count'], grouped_df[value], yerr=grouped_df['sem'],
                    fmt='o', mfc=color, mec='k', ecolor=color, ms=3.5)

    formatPlot(ax)

    if n_stimmed:
        ax.set_xlabel('Number Stimmed Cells', fontsize=18)
    else:
        ax.set_xlabel('Fraction Stimmed', fontsize=18)

    save_str = ''
    ylabel = ''
    if kwargs.get('normalize', False):
        ylabel += 'Normalized '
        save_str += 'mean_normalized_'

    if kwargs.get('diff', False):
        ylabel += 'Relative '
        save_str += 'relative_'

    if value == 'diff':
        ylabel += 'In - Out'
        save_str += 'diff'
    else:
        ylabel += 'Mean Peristim Tuning'
        save_str += 'in_mean'

    if dist_thresh:
        save_str += '{}_distThresh'.format(dist_thresh)

    save_str += '.'

    ax.set_ylabel(ylabel, fontsize=18)
    # ax.set_ylim([-1, 1])

    fig.savefig(prefix + save_str.format(FILE_ENDING), bbox_inches='tight')
