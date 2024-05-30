"""
Constant values for importing into magnitude/context remapping project

Sheahan, H.*, Luyckx, F.*, Nelli, S., Taupe, C., & Summerfield, C. (2021). Neural
 state space alignment for magnitude generalisation in humans and recurrent networks.
 Neuron (in press)

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 13/03/2020
"""

import numpy as np

# Save/load directories
DATASET_DIRECTORY = 'datasets/'
MODEL_DIRECTORY = 'models/'
FIGURE_DIRECTORY = 'figures/'
ANIMATION_DIRECTORY = 'animations/'
TRAININGRECORDS_DIRECTORY = 'trainingrecords/'
TB_LOG_DIRECTORY = 'results/runs/'                        # tensorboard records
NETANALYIS_DIRECTORY = 'network_analysis/'
LESIONS_DIRECTORY = 'network_analysis/lesion_tests/'
RDM_DIRECTORY = 'network_analysis/RDMs/'
PARAMETER_DIRECTORY = 'linesmodel_parameters/'
EEG_DIRECTORY = 'datasets/'

# Total maximum numbers for one-hot coding
TOTALMAXNUM = 16    # max numerosity
NCONTEXTS = 3       # max number of contexts for one-hot coding
NTYPEBITS = 1       # just need one input bit to flag whether current trial is 'compare' or 'filler'


# define upper and lower limits for each # range)
FULLR_LLIM = 1      # full # range, lower limit
FULLR_ULIM = 16     # full # range, upper limit
LOWR_LLIM = 1       # low # range, lower limit
LOWR_ULIM = 11      # low # range, upper limit
HIGHR_LLIM = 6      # high # range, lower limit
HIGHR_ULIM = 16     # high # range, upper limit

# the resulting range spans
FULLR_SPAN = FULLR_ULIM - FULLR_LLIM +1
LOWR_SPAN = LOWR_ULIM - LOWR_LLIM +1
HIGHR_SPAN = HIGHR_ULIM - HIGHR_LLIM +1

# trial types
TRIAL_FILLER  = 0
TRIAL_COMPARE = 1

# the same as the spans... but used in lines_model
N_POINTS_LONG = 16
N_POINTS_SHORT = 11

# mean values for each context
CONTEXT_FULL_MEAN = np.mean(range(FULLR_LLIM, FULLR_ULIM+1))  # 8.5
CONTEXT_LOW_MEAN = np.mean(range(LOWR_LLIM, LOWR_ULIM+1))     # 6
CONTEXT_HIGH_MEAN = np.mean(range(HIGHR_LLIM, HIGHR_ULIM+1))  # 11
GLOBAL_MEAN = 8.5 #np.mean([list(range(FULLR_LLIM, FULLR_ULIM+1)), list(range(LOWR_LLIM, LOWR_ULIM+1)), list(range(HIGHR_LLIM, HIGHR_ULIM+1))])

# Figure colours
CONTEXT_COLOURS = [[253/256, 176/256, 4/256], 'dodgerblue', 'orangered', 'black']  # low, high, full
MODEL_COLOURS = ['darkkhaki', 'olivedrab','darkolivegreen']  # change to show both local and global on same plot easily and keep main colours for data

# Single dataset for retraining decoders under blocked, VI conditions
RETRAINING_DATASET = 'dataset_truecontextlabel_numrangeblocked_bpl120_id9999'#'dataset_truecontextlabel_numrangeblocked_bpl120_id9999'
