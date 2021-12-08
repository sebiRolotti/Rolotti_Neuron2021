import sys
sys.path.insert(0, '../../induction_analysis')

from Analysis import filters
import Analysis.induction_analysis as ia
import numpy as np

from lab_repo.classes.place_cell_classes import pcExperimentGroup

from Experiments import SingleCellExperiment, LEDExperiment

import lab_repo.analysis.place_cell_analysis as pca
import lab_repo.analysis.behavior_analysis as ba

import pandas as pd


grps = []
rfilters = []

# LED

ind_expt_ids = [22362, 22365, 22140, 22716, 22725, 22727] + [22754, 22755, 23227, 23853, 23882, 23875, 23834, 23866] + [25002, 25020, 25030, 25035]

expts = [LEDExperiment.LEDExperiment(x) for x in ind_expt_ids]
pc_kwargs = {'imaging_label': 'suite2p',
                         'nPositionBins': 100,
                         'channel': 'Ch2',
                         'demixed': False,
                         'pf_subset': None,
                         'signal': 'spikes'}

grp = pcExperimentGroup(expts, label='sparse_led', **pc_kwargs)

# Induced
grps.append(grp)

stim_filter = filters.grp_stim_filter(grp, label='suite2p', overwrite=False, activity_filter=True)
ind_filter = filters.grp_induction_filter(grp, stim_filter=stim_filter, overwrite=False)
pc_filter = filters.pcs_filter(grp, overwrite=False)
nostim_filter = filters.grp_unstimmed_filter(grp)
rfilters.append(ind_filter)

# Other Stimmed
grps.append(grp)
def stim_pc_filter(x):
    return stim_filter(x) and pc_filter(x) and not ind_filter(x)
rfilters.append(stim_pc_filter)

# Unstimulated
grps.append(grp)
def nostim_pc_filter(x):
    return nostim_filter(x) and pc_filter(x)
rfilters.append(nostim_pc_filter)

# GCaMP-only Controls

control_ids = [26235, 26237, 26240, 26252, 26251]

expts = [LEDExperiment.LEDExperiment(x) for x in control_ids]
pc_kwargs = {'imaging_label': 'suite2p',
                         'nPositionBins': 100,
                         'channel': 'Ch2',
                         'demixed': False,
                         'pf_subset': None,
                         'signal': 'spikes'}

grp = pcExperimentGroup(expts, label='led_control', **pc_kwargs)

grps.append(grp)
rfilters.append(filters.pcs_filter(grp, overwrite=False))

labels = ['Induced', 'Stimulated', 'Unstimulated', 'Control']


# # Specificity

# df = []
# for grp, rfilter, label in zip(grps, rfilters, labels):
#     out = pca.specificity(grp, roi_filter=rfilter)
#     out['label'] = label
#     df.append(out)

# df = pd.concat(df)

# df.to_pickle('/home/sebi/induction_cache/pf_compare_specificity.pkl')

# # Sensitivity

# df = []
# for grp, rfilter, label in zip(grps, rfilters, labels):
#     out = pca.sensitivity(grp, roi_filter=rfilter)
#     out['label'] = label
#     df.append(out)

# df = pd.concat(df)

# df.to_pickle('/home/sebi/induction_cache/pf_compare_sensitivity.pkl')

# TC Correlation

# df = []
# for grp, rfilter, label in zip(grps, rfilters, labels):
#     out = ia.tuning_correlation(grp, roi_filter=rfilter, session1='control_induction', session2='control_24h', pc_only='first')
#     out['label'] = label
#     df.append(out)

# df = pd.concat(df)

# df.to_pickle('/home/sebi/induction_cache/pf_compare_tuning_correlation.pkl')


# Event Rates

def event_rate(grp, roi_filter=None, running=True, exclude=True):

    data_list = []
    for expt in grp:

        tid = expt.trial_id

        rois = expt.rois(roi_filter=roi_filter)
        spikes = expt.spikes(roi_filter=roi_filter, binary=True)

        # Exclude stim laps
        if exclude and running:
            laps = ba.absolutePosition(expt.find('trial'), imageSync=True).astype(int)
            interval = np.where((expt.velocity()[0] > 1) & (laps >= 6))[0]
        elif exclude and not running:
            laps = ba.absolutePosition(expt.find('trial'), imageSync=True).astype(int)
            interval = np.where((expt.velocity()[0] <= 1) & (laps >= 6))[0]
        elif running:
            interval = np.where(expt.velocity()[0] > 1)[0]
        else:
            interval = np.where(expt.velocity()[0] <= 1)[0]

        frs = np.nansum(spikes[:, interval], axis=1) / (len(interval) * expt.frame_period())

        for roi, fr in zip(rois, frs):

            data_list.append({'expt_id': tid,
                              'roi': roi.label,
                              'value': fr})

    return pd.DataFrame(data_list)


# df = []
# for grp, rfilter, label in zip(grps, rfilters, labels):
#     out = event_rate(grp, roi_filter=rfilter, running=True)
#     out['label'] = label
#     df.append(out)

# df = pd.concat(df)

# df.to_pickle('/home/sebi/induction_cache/pf_compare_run_event_rate.pkl')

df = []
for grp, rfilter, label in zip(grps, rfilters, labels):
    out = event_rate(grp, roi_filter=rfilter, running=False)
    out['label'] = label
    df.append(out)

df = pd.concat(df)

df.to_pickle('/home/sebi/induction_cache/pf_compare_nonrun_event_rate.pkl')
