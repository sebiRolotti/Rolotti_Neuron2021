import sys
sys.path.insert(0, '/home/sebi/code/analysis/analysis-scripts/induction_analysis')

from Analysis import filters

from lab.classes.place_cell_classes import pcExperimentGroup

from Experiments import SingleCellExperiment, LEDExperiment

import seaborn as sns

# Load all groups

grps = []

# SCE

ind_experiments = [24596, 24761, 24786, 24801, 25079, 25080, 25081]
grp = [LEDExperiment.LEDExperiment(x) for x in ind_experiments]

pc_kwargs = {'imaging_label': 'soma',
               'nPositionBins': 100,
               'channel': 'Ch2',
               'demixed': False,
               'pf_subset': None,
               'signal': 'spikes'}

grp = pcExperimentGroup(grp, label='sce', **pc_kwargs)

grps.append(grp)

# Sparse Targeted

expt_ids = [22745, 22757, 23845, 23890, 23841, 23901] + [24699, 24760, 24702, 24987]
mps = ['/data/Sebi/2p/svr153/102919/MarkPoints-10072019-ref-003',
       '/data/Sebi/2p/svr154/103019/MarkPoints-10302019-1254-001',
       '/data/Sebi/2p/svr159/010820/MarkPoints-01082020-1318-001',
       '/data/Sebi/2p/svr160/010820/MarkPoints-01082020-1318-001',
       '/data/Sebi/2p/svr161/010920/MarkPoints-01092020-1404-003',
       '/data/Sebi/2p/svr164/011020/MarkPoints-01102020-1135-002',
       '/data3/Sebi/2p/svr165/030320/MarkPoints-02282020-induce-002',
       '/data3/Sebi/2p/svr166/030320/MarkPoints-02282020-induce-002',
       '/data3/Sebi/2p/svr169/030320/MarkPoints-02282020-induce-002',
       '/data3/Sebi/2p/svr174/030820/MarkPoints-03082020-induction-001']

expts = [SingleCellExperiment.SingleCellExperiment(x, mp) for x, mp in zip(expt_ids, mps)]

pc_kwargs = {'imaging_label': 'suite2p',
                         'nPositionBins': 100,
                         'channel': 'Ch2',
                         'demixed': False,
                         'pf_subset': None,
                         'signal': 'spikes'}

grp = pcExperimentGroup(expts, label='sparse_targeted', **pc_kwargs)

grps.append(grp)

# Dense Targeted

expt_ids = [22286, 22311, 22417] # 22873]
boris = [24600, 24640, 24659, 24694, 24603, 24677, 24773, 24688, 24781]
expt_ids += boris

mps = ['/data/Mohsin/2p/msa0919_2/10_10_19/MarkPoints-10102019-induce-002',
       '/data/Mohsin/2p/msa0919_2/10_5_19/MarkPoints-10052019-induce-002',
       '/data/Mohsin/2p/msa0919_1/10_18_19/MarkPoints-10182019-induce-001']
       #'/data/Mohsin/2p/msa1119_5/11_18_19/MarkPoints-11182019-induce-001']

boris_mps = ['/data4/Mohsin/2p/msa0220_3/2_24_20/MarkPoints-02242020-induce-003',
             '/data4/Mohsin/2p/msa0220_3/2_28_20/MarkPoints-02282020-induce-003',
             '/data4/Mohsin/2p/msa0220_3/3_4_20/MarkPoints-02282020-induce-003',
             '/data4/Mohsin/2p/msa0220_3/3_8_20/MarkPoints-03082020-induce-003',
             '/data4/Mohsin/2p/msa0220_4/2_24_20/MarkPoints-02242020-induce-003',
             '/data4/Mohsin/2p/msa0320_1/3_6_20/MarkPoints-02282020-induce-003',
             '/data4/Mohsin/2p/msa0320_1/3_10_20/MarkPoints-03102020-induce-003',
             '/data4/Mohsin/2p/msa0320_2/3_9_20/MarkPoints-03092020-induce-003',
             '/data4/Mohsin/2p/msa0320_2/3_11_20/MarkPoints-03082020-induce-003']

mps += boris_mps


expts = [SingleCellExperiment.SingleCellExperiment(x, mp) for x, mp in zip(expt_ids, mps)]
pc_kwargs = {'imaging_label': 's2p',
                         'nPositionBins': 100,
                         'channel': 'Ch2',
                         'demixed': False,
                         'pf_subset': None,
                         'signal': 'spikes'}

grp = pcExperimentGroup(expts, label='dense_all', **pc_kwargs)

grps.append(grp)

# Dense Targeted + CNO

expt_ids = [22326, 22318, 22432]
boris_ids = [24609, 24656, 24676, 24777, 24691, 24792, 24765, 24798]
expt_ids += boris_ids

mps = ['/data/Mohsin/2p/msa0919_2/10_10_19/MarkPoints-10102019-induce-002',
           '/data/Mohsin/2p/msa0919_2/10_7_19/MarkPoints-10072019-CNO_test-006',
           '/data/Mohsin/2p/msa0919_1/10_20_19/MarkPoints-10202019-induce-001']

boris_mps = ['/data4/Mohsin/2p/msa0220_3/2_26_20/MarkPoints-02262020-induce-003',
            '/data4/Mohsin/2p/msa0220_3/3_1_20/MarkPoints-02282020-test-002',
            '/data4/Mohsin/2p/msa0220_3/3_6_20/MarkPoints-02282020-induce-003',
            '/data4/Mohsin/2p/msa0220_3/3_10_20/MarkPoints-03102020-induce-003',
            '/data4/Mohsin/2p/msa0320_1/3_8_20/MarkPoints-03082020-induce-003',
            '/data4/Mohsin/2p/msa0320_1/3_12_20/MarkPoints-03082020-induce-003',
            '/data4/Mohsin/2p/msa0320_2/3_9_20/MarkPoints-03092020-induce-003',
            '/data4/Mohsin/2p/msa0320_2/3_13_20/MarkPoints-03082020-induce-003']

mps += boris_mps


expts = [SingleCellExperiment.SingleCellExperiment(x, mp) for x, mp in zip(expt_ids, mps)]
pc_kwargs = {'imaging_label': 's2p',
                         'nPositionBins': 100,
                         'channel': 'Ch2',
                         'demixed': False,
                         'pf_subset': None,
                         'signal': 'spikes'}


grp = pcExperimentGroup(expts, label='cno_dense_all', **pc_kwargs)
grps.append(grp)

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

grps.append(grp)


labels = ['SCE', 'Sparse Targeted', 'Dense', 'Dense + Dreadd', 'Sparse LED']
colors = ['#85c68f', 'r'] + sns.xkcd_palette(['warm blue', 'golden rod']) + ['k']

print [(i, label) for i, label in enumerate(labels)]

stim_filters = []
ind_filters = []
for grp in grps:

    if grp.label() == 'sce':
        stim_filter = lambda x: True
    else:
        stim_filter = filters.grp_stim_filter(grp, label='suite2p', overwrite=False, activity_filter=True)
    ind_filter = filters.grp_induction_filter(grp, stim_filter=stim_filter, overwrite=False)

    stim_filters.append(stim_filter)
    ind_filters.append(ind_filter)

print 'Vars in Environment: grps, labels, colors, stim_filters, ind_filters'
