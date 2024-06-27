"""
Code published in the following paper:
 Sheahan, H.*, Luyckx, F.*, Nelli, S., Taupe, C., & Summerfield, C. (2021). Neural
  state space alignment for magnitude generalisation in humans and recurrent networks.
  Neuron (in press)
* authors contributed equally

 This is a set of simulations for training a simple RNN on a relative magnitude judgement problem.
 The network is trained to answer the question: is input N > input N-t?
 Where t is between 3-5, i.e. the inputs to be compared in each sequence are separated by several 'filler' inputs.

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 04/12/2019
 Notes:
 - requires ffmpeg for 3D animation generation in generate_plots()
 Issues: N/A
"""
# ---------------------------------------------------------------------------- #

import magnitude_network as mnet
import define_dataset as dset
import plotter as mplt
import analysis_helpers as anh
import constants as const
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import theoretical_performance as theory
import numpy as np
import os
import scipy



if __name__ == '__main__':

    # set up dataset and network hyperparams (optionally via command line)
    args, device, multiparams = mnet.define_hyperparams()
    args.all_fullrange = False     # False = blocked; True = interleaved
    args.which_context = 0          # 0 = all contexts; 1 = LOWR (low range context); 2 = HIGHR (high range context)
    args.train_lesion_freq = 0.1    # 0.0 or 0.1  (also 0.2, 0.3, 0.4 for blocked & true context case)
    args.block_int_ttsplit = False  # True: test on a different distribution (block/interleave) than training
    args.retrain_decoder = False
    args.model_id = 4          
    #args.model_id = 9999          # for visualising or analysing a particular trained model

    # Train a network from scratch and save it
    print('Training network...')
    mnet.train_and_save_network(args, device, multiparams)
    print('Training complete and network saved. main')
    
    # Check information about the dataset
    dset.view_dataset_index_info(-50, args)

    # Analyse the trained network (extract and save network activations)
    print('\nAnalysing network...')
    MDS_dict = anh.analyse_network(args)

    # Check the average final performance for trained models matching args
    print('\nChecking average performance...')
    anh.average_perf_across_models(args)

    # Visualise the resultant network activations (RDMs and MDS)
    print('\nGenerating plots...')
    MDS_dict, args = anh.average_activations_across_models(args)
    mplt.generate_plots(MDS_dict, args)  # (Figure 3 + extras)
    
    if args.all_fullrange:
      print('\nTraining Curricula: Interleaved')
    else:
      print('\nTraining Curricula: Blocked')
    print('\nWhich Context: ', args.which_context)
      
    # Plot the lesion test performance
    #print('\nPlotting lesion tests...')
    #mplt.perf_vs_context_distance(args, device)     # Assess performance after a lesion vs context distance (Figure 2 and S1)
    #mplt.compare_lesion_tests(args, device)      # compare the performance across the different lesion frequencies during training (Figure 2)

    # Statistical tests: is network behaviour better fit by an agent using the local-context or global-context policy
    #print('\nStatistical tests...')
    #anh.model_behaviour_vs_theory(args, device)

    # Load representations and check cross-line big/small generalisation
    #print('\nCross-line generalisation...')
    #anh.cross_line_rep_generalisation(args)
    #anh.cross_line_rep_generalisation_human(args)

    # Load a trained network (no VI), freeze the first layer (recurrent) weights and then retrain the decoder with VI and save it
    #print('\nRetraining decoder...')
    #retrain_args, _, _ = mnet.define_hyperparams()
    #retrain_args.train_lesion_freq = 1.0
    #retrain_args.epochs = 30
    #retrain_args.lr_multi = [0.001]
    #retrain_args.retrain_decoder = True
    #anh.retrain_decoder(args, retrain_args, device, multiparams)
    #anh.analyse_retrained_nets()
