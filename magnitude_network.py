"""
This is a selection of functions and classes relating to pytorch network training
on the contextual magnitude mapping project with Fabrice.
A simple RNN or MLP is trained on a relational magnitude problem: is input A > input B?

Sheahan, H.*, Luyckx, F.*, Nelli, S., Taupe, C., & Summerfield, C. (2021). Neural
 state space alignment for magnitude generalisation in humans and recurrent networks.
 Neuron (in press)

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 13/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import define_dataset as dset
import matplotlib.pyplot as plt
import constants as const
import plotter as mplt
import numpy as np
import copy
import sys
import random
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from sklearn.manifold import MDS
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# for training I/O
from datetime import datetime
from itertools import product
import argparse

torch.autograd.set_detect_anomaly(True)

def print_progress(i, numiter):
    """This function prints to the screen the optimisation progress (at each iteration i, out of a total of numiter iterations)."""
    j = i/numiter
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
    sys.stdout.flush()


def batch_to_torch(originalimages):
    """Convert the input batch to a torch tensor"""
    #originalimages = originalimages.unsqueeze(1)   # change dim for the convnet
    originalimages = originalimages.type(torch.FloatTensor)  # convert torch tensor data type
    return originalimages


def answer_correct(output, label):
    """
    This function compares the output to the label on that trial (or batch) and
    determines whether it (or how many) match the target label.
    """
    output = np.squeeze(output, axis=1)
    pred = np.zeros((output.size()))
    for i in range((output.size()[0])):
        if output[i]>0.5:
            pred[i] = 1
        else:
            pred[i] = 0
    tmp = np.squeeze(np.asarray(label))
    return (pred==tmp).sum().item()


def plot_grad_flow(args, layers, ave_grads, max_grads, batch_number):
    """
    This function will take a look at the gradients in all layers of our model during training.
    This will help us to see whether our gradients are flowing well through our RNN
    - For my RNN it really needs to be divided up so that the loop is over recurrent steps rather than an unfolded model.
    - Edited from code on forum: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7 by Roshan Rane (downloaded 20/02/2020)
    - ave_grads are the gradients from our model. Call this function just after each backwards pass.
    """
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow, update #{}".format(batch_number))

    # save figure of gradient flow (this will take ages if you do it every loop)
    #mplt.save_figure('figures/gradients/gradflow_{}_'.format(batch_number), args, 'recurrent', blockTrain, seqTrain, True, givenContext, False, noise_std, retainHiddenState, False, 'compare', True)


def recurrent_train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    """ Train a recurrent neural network on the training set.
    This now trains whilst retaining the hidden state across all trials in the training sequence
    but being evaluated just on pairs of inputs and considering each input pair as a trial for minibatching.
    - we lesion the number input on compare trails with frequency args.train_lesion_freq to enhances the network's use of local context
    - we will include in our total cost function the performance when assessed on a trial that is lesioned
       AND on the compare trial after a lesion
     """
    model.train()
    train_loss = 0
    correct = 0
    trials_counter = 0
    all_trials_counter = 0

    # On the very first trial on training, reset the hidden weights to zeros
    hidden = torch.zeros(args.batch_size, model.recurrent_size)
    latentstate = torch.zeros(args.batch_size, model.recurrent_size)

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients
        inputs, labels, contextsequence, trialtype = batch_to_torch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batch_to_torch(data['contextinput']), batch_to_torch(data['trialtypeinput']).unsqueeze(2)

        # initialise everything for our recurrent model
        recurrentinputs = []
        sequenceLength = inputs.shape[1]
        n_comparetrials = np.nansum(np.nansum(trialtype))
        layers, ave_grads, max_grads = [[] for i in range(3)]
        lesionRecord = np.zeros((sequenceLength,))
        loss = 0
        alternate_lesion_trial = True  # for retraining decoder, lesion alternate trials
        
        for i in range(sequenceLength):
            context = contextsequence[:,i]
            lesionedinput = inputs[:,i]
            if trialtype[0,i]==0:  # remove context indicator on the filler trials
                context_in = torch.full_like(context, 0)
            else:
                context_in = copy.deepcopy(context)
                if args.retrain_decoder:
                    if alternate_lesion_trial: # alternately lesion the number input on compare trials
                        lesionRecord[i] = 1
                        lesionedinput = torch.full_like(inputs[:,i],0)
                else:
                    if (random.random() < args.train_lesion_freq): # occasionally lesion the number input on compare trials
                        lesionRecord[i] = 1
                        lesionedinput = torch.full_like(inputs[:,i],0)

            alternate_lesion_trial = False if alternate_lesion_trial else True
            inputX = torch.cat((lesionedinput, context_in, trialtype[:,i]),1)
            recurrentinputs.append(inputX)

        if not args.retain_hidden_state:
            hidden = torch.zeros(args.batch_size, model.recurrent_size)  # only if you want to reset hidden recurrent weights
        else:
            hidden = latentstate # keep hidden state to reflect recent statistics of previous inputs

        # perform N-steps of recurrence
        for item_idx in range(sequenceLength):
            # inject some noise (Note: no longer in use, set model.hidden_noise to 0.0)
            noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
            #print('hidden1', hidden)
            hidden = hidden.add(noise) # !!! .add_(noise) was there originally
            #print('hidden2', hidden)
            hidden = hidden.to(torch.float32) # turn into a float instead of double
            output, hidden = model(recurrentinputs[item_idx], hidden)
            if item_idx==(sequenceLength-2):                  # extract the hidden state just before the last input in the sequence is presented
                latentstate = hidden.detach()

            # for 'compare' trials only, evaluate performance at every comparison between the current input and previous 'compare' input
            # if item_idx > 0 and (trialtype[0,item_idx]==1):
            # SN double check that trialtype is always 1?
            #     loss = loss + criterion(output, labels[item_idx])   # accumulate the loss (autograd should sort this out for us: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
            #     correct = correct + answer_correct(output, labels[item_idx])
                
            if np.isnan(labels[item_idx]) ==0  and (trialtype[0,item_idx]==1):
                loss = loss + criterion(output, labels[item_idx])   # accumulate the loss (autograd should sort this out for us: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
                correct = correct + answer_correct(output, labels[item_idx])
                trials_counter += 1
            all_trials_counter += 1
        #print("step 1")

        loss.backward()

        #print("step 2")
        # record and visualise our gradients
        if (batch_idx % 100) == 0:
            for n, p in model.named_parameters():
                if(p.requires_grad) and ("bias" not in n):
                    if p.grad is not None:
                        layers.append(n)
                        ave_grads.append(p.grad.abs().mean())
                        max_grads.append(p.grad.abs().max())
                    else:
                        print("Warning: p.grad gradient is type None at batch {} in sequence of length {}".format(batch_idx, sequenceLength) )
            plot_grad_flow(args, layers, ave_grads, max_grads, batch_idx)

        optimizer.step()            # update our weights
        train_loss += loss.item()
        #print("step 3")
        if batch_idx % args.log_interval == 0:
            if printOutput:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader.dataset), loss.item()))
        #print("step 4")
        
    #print("loop ended---------------------------------------------------")
    train_loss /= trials_counter
    print('Trials counter: {}'.format(trials_counter))
    print('All trials counter: {}'.format(all_trials_counter))
    print('train_loader.dataset',train_loader.dataset)
    print('len(train_loader.dataset)',len(train_loader.dataset))
    print('n_comparetrials-1',n_comparetrials-1)
    print('correct',correct)
    accuracy = 100. * correct / trials_counter #(len(train_loader.dataset)*(n_comparetrials-1))
    return train_loss, accuracy


def recurrent_test(args, model, device, test_loader, criterion, printOutput=True):
    """Test a recurrent neural network on the test set (without lesions)."""
    model.eval()
    test_loss = 0
    correct = 0
    trials_counter = 0

    # reset hidden recurrent weights on the very first trial
    hidden = torch.zeros(args.batch_size, model.recurrent_size)
    latentstate = torch.zeros(args.batch_size, model.recurrent_size)

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):
            inputs, labels, contextsequence, trialtype = batch_to_torch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batch_to_torch(data['contextinput']), batch_to_torch(data['trialtypeinput']).unsqueeze(2)

            # reformat the input sequences for our recurrent model
            recurrentinputs = []
            sequenceLength = inputs.shape[1]
            n_comparetrials = np.nansum(np.nansum(trialtype))

            for i in range(sequenceLength):
                context = contextsequence[:,i]
                if trialtype[0,i]==0:  # remove context indicator on the filler trials
                    context_in = torch.full_like(context, 0)
                else:
                    context_in = copy.deepcopy(context)
                inputX = torch.cat((inputs[:, i], context_in, trialtype[:,i]),1)
                recurrentinputs.append(inputX)

            if not args.retain_hidden_state:  # only if you want to reset hidden state between trials
                hidden = torch.zeros(args.batch_size, model.recurrent_size)
            else:
                hidden = latentstate

            # perform a N-step recurrence for the whole sequence of numbers in the input
            for item_idx in range(sequenceLength):

                # inject some noise (Note: no longer in use, set model.hidden_noise to 0.0)
                noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
                hidden.add_(noise)
                output, hidden = model(recurrentinputs[item_idx], hidden)
                if item_idx==(sequenceLength-2):  # extract the hidden state just before the last input in the sequence is presented
                    latentstate = hidden.detach()

                if np.isnan(labels[item_idx]) ==0  and (trialtype[0,item_idx]==1):
                    test_loss += criterion(output, labels[item_idx]).item()
                    correct += answer_correct(output, labels[item_idx])
                    trials_counter += 1

    test_loss /= trials_counter  # there are n_comparetrials-1 instances of feedback per sequence
    accuracy = 100. * correct / trials_counter
    if printOutput:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset)*(n_comparetrials-1), accuracy))
    return test_loss, accuracy


def get_local_model_response(number, context, label):
    # evaluate whether a simulated local context policy would have got this trial correct
    localmedians = [const.CONTEXT_FULL_MEAN, const.CONTEXT_LOW_MEAN, const.CONTEXT_HIGH_MEAN]
    response = 1 if number > localmedians[context-1] else 0
    iscorrect = 1 if response == label else 0
    return iscorrect


def get_global_model_response(number, context, label):
    # evaluate whether a simulated local context policy would have got this trial correct
    globalmedian = const.GLOBAL_MEAN
    response = 1 if number > globalmedian else 0
    iscorrect = 1 if response == label else 0
    return iscorrect


def recurrent_lesion_test(args, model, device, test_loader, criterion, printOutput=True, whichLesion='number', lesionFrequency=1):
    """
    Test a recurrent neural network on the test set, while lesioning occasional inputs.
    Lesioning inputs: select either the context part of the input, or the number input to be lesioned
    - we will assess impact of lesioning a single trial in the dataset, and testing on the primary target (compare trial) immediately following,
      but vary the position in the sequence at which the lesion happens.
    """
    model.eval()

    # reset hidden recurrent weights on the very first trial
    hidden = torch.zeros(args.batch_size, model.recurrent_size)
    latentstate = torch.zeros(args.batch_size, model.recurrent_size)
    n_sequences = 0
    overallcomparisons = 0
    aggregateLesionPerf = 0
    aggregatePerf = 0
    allLesionAssessments = []

    with torch.no_grad():  # dont track the gradients
        # for each sequence
        for batch_idx, data in enumerate(test_loader):
            inputs, labels, contextsequence, contextinputsequence, trialtype = batch_to_torch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batch_to_torch(data['context']), batch_to_torch(data['contextinput']), batch_to_torch(data['trialtypeinput']).unsqueeze(2)
            # setup
            recurrentinputs = []
            sequenceLength = inputs.shape[1]
            sequenceAssessment = []

            # organise the inputs for each trial in our sequence
            for i in range(sequenceLength):
                context = contextinputsequence[:,i]

                if trialtype[0,i]==0:
                    inputcontext = torch.full_like(context, 0)  # all filler trials should have no context input to it
                else:
                    inputcontext = copy.deepcopy(context)
                inputX = torch.cat((inputs[:, i], inputcontext, trialtype[:, i]),dim=1)
                recurrentinputs.append(inputX)

            # consider each number in the sequence
            for assess_idx in range(sequenceLength):
                lesionRecord = np.zeros((sequenceLength,))  # reset out lesion record
                context = dset.turn_one_hot_to_integer(contextsequence[:,assess_idx][0])[0]  # the true underlying context for this input

                # each time we repeat this exercise we need to use the original hidden state from previous sequence
                if not args.retain_hidden_state:  # only if you want to reset hidden state between trials
                    hidden = torch.zeros(args.batch_size, model.recurrent_size)
                else:
                    hidden = latentstate

                # if its a comparison trial, we will use it to assess performance and lesion our sequence up to this point
                if (trialtype[:,assess_idx]==1) and (assess_idx>0):  # don't use the very first trial as an assessment trial
                    # Look backwards from the assessment point, lesion the immediately previous compare trial,
                    # and then every prior compare trial with frequency F
                    isPrevCompareTrial = True
                    for item_idx in range(assess_idx-1,-1,-1):
                         # lesion the compare trial immediately preceeding the assessment trial
                        if (trialtype[:,item_idx]==1) and isPrevCompareTrial:
                            lesionRecord[item_idx] = 1
                            isPrevCompareTrial = False
                        else:
                            # now lesion each other compare trial with frequency F
                            if (trialtype[:,item_idx]==1):
                                if (random.random() < lesionFrequency):
                                    lesionRecord[item_idx] = 1

                    # now that we have performed our lesions up to our assessment trial, pass this sequence through the network and assess performance
                    assess_number = dset.turn_one_hot_to_integer(inputs[:,assess_idx][0])[0]
                    tmpinputs = copy.deepcopy(recurrentinputs)
                    overallperf = 0
                    ncomparetrials = 0

                    for trial in range(assess_idx+1):

                        # if trial designated for lesioning, apply the lesion
                        if lesionRecord[trial]==1:
                            lesion_number = dset.turn_one_hot_to_integer(tmpinputs[trial][0][0:const.TOTALMAXNUM])
                            if whichLesion=='number':
                                tmpinputs[trial][0][0:const.TOTALMAXNUM] = torch.full_like(tmpinputs[trial][0][0:const.TOTALMAXNUM], 0)
                            else:
                                tmpinputs[trial][0][const.TOTALMAXNUM:const.TOTALMAXNUM+const.NCONTEXTS] = torch.full_like(tmpinputs[trial][0][const.TOTALMAXNUM:const.TOTALMAXNUM+const.NCONTEXTS], 0)

                        # inject some noise (Note: no longer in use, set model.hidden_noise to 0.0)
                        noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
                        hidden.add_(noise)
                        output, hidden = model(tmpinputs[trial], hidden)
                        h0activations, h1activations, _ = model.get_activations(tmpinputs[trial], hidden)

                        # assess aggregate performance on whole sequence (including all lesions)
                        if trialtype[:,trial]==1:
                            ncomparetrials += 1
                            overallperf += answer_correct(output, labels[trial])
                            overallcomparisons += 1
                            # once we get to the assessment trial, assess performance
                            if trial==assess_idx:
                                lesionperf = answer_correct(output, labels[trial])
                                post_lesion_activations = h1activations

                    localmodel_perf = get_local_model_response(assess_number, context, labels[trial])   # correct or incorrect
                    globalmodel_perf = get_global_model_response(assess_number, context, labels[trial]) # correct or incorrect

                    mydict = {"assess_number":assess_number, "lesion_number":lesion_number, "lesion_perf":lesionperf, "overall_perf":overallperf,\
                     "desired_lesionF":lesionFrequency, "underlying_context":context,  "assess_idx":assess_idx, "compare_idx":ncomparetrials,\
                     "localmodel_perf":localmodel_perf, "globalmodel_perf":globalmodel_perf, "post_lesion_activations":post_lesion_activations }
                    sequenceAssessment.append(mydict)
                    aggregateLesionPerf += lesionperf
                    aggregatePerf += overallperf
                    n_sequences += 1

                # extract the hidden state just before the last input in the sequence is presented, for passing to next sequence
                # since the network has only processed sequences up to compare trials, we need to pass the whole sequence through again now! inefficient, yes
                if assess_idx==(sequenceLength-2):
                    for i in range(assess_idx+1):
                        noise = torch.from_numpy(np.reshape(np.random.normal(0, model.hidden_noise, hidden.shape[0]*hidden.shape[1]), (hidden.shape)))
                        hidden.add_(noise)
                        output, hidden = model(tmpinputs[i], hidden)  # this should be the sequence of trials that are all lesioned with probability F
                    latentstate = hidden.detach()

            allLesionAssessments.append(sequenceAssessment)

    # summary stats
    allLesionAssessments = np.asarray(allLesionAssessments)
    summarylesionperf = 100. *(aggregateLesionPerf / n_sequences)
    summaryperf = 100. *(aggregatePerf / overallcomparisons)
    print('Mean lesion accuracy: {}/{} ({:.2f}%)'.format(aggregateLesionPerf, n_sequences, summarylesionperf))
    print('Mean overall accuracy: {}/{} ({:.2f}%)'.format(aggregatePerf, overallcomparisons, summaryperf))

    return allLesionAssessments, summarylesionperf, summaryperf


def sort_all_vars_by_x(allvars, sortind):
    """This function sort_all_vars_by_x() will sort all variables input in allvars according to the indices of sortind."""
    sortedvars = []
    for thisvar in allvars:
        thisvar = np.take_along_axis(thisvar, sortind, axis=0)
        sortedvars.append(thisvar)
    return sortedvars


def sort_activations(allvars):
    """This function sort_activations() just sorts all the activation- and label-related variables we care about,
     first into context order and then input number order within each context.
    """
    contexts, activations, MDSlabels, labels_refValues, labels_judgeValues, time_index, counter = allvars

    # sort all variables first by context order
    context_ind = np.argsort(contexts, axis=0)
    contexts, activations, MDSlabels, labels_refValues, labels_judgeValues, time_index, counter = sort_all_vars_by_x(allvars, context_ind)

    # within each context, sort according to numerosity of the judgement value
    for context in range(1,const.NCONTEXTS+1):
        ind = [i for i in range(contexts.shape[0]) if contexts[i]==context]
        numerosity_ind = np.argsort(labels_judgeValues[ind], axis=0) + ind[0]
        allvars = [contexts, activations, MDSlabels, labels_refValues, labels_judgeValues, time_index, counter] # important that this is updated in loop
        contexts[ind], activations[ind], MDSlabels[ind], labels_refValues[ind], labels_judgeValues[ind], time_index[ind], counter[ind] = sort_all_vars_by_x(allvars, numerosity_ind)

    return contexts, activations, MDSlabels, labels_refValues, labels_judgeValues, time_index, counter


def format_input_sequence(TRIAL_TYPE, testset):
    """ *This function is obsolete*
    This function format_input_sequence() is for tidying up get_activations(),
    and will determine the unique inputs in the test set (there will be repeats in the original test set).
    """
    testset_input_n_context, seq_record = [[] for i in range(2)]
    for seq in range(len(testset["input"])):
        inputA, inputB = [None for i in range(2)]
        for item_idx in range(len(testset["input"][seq])):
            trialtype = testset["trialtypeinputs"][seq, item_idx]

            if TRIAL_TYPE==const.TRIAL_COMPARE:
                if trialtype==TRIAL_TYPE:
                    inputA = testset["input"][seq, item_idx]
                    if inputB is not None:
                        if np.all(inputA==inputB):
                            print('Warning: adjacent trial types are same number {}, both of type compare at item {} in sequence {}'.format(dset.turn_one_hot_to_integer(inputA)[:], item_idx,seq))
                        context = testset["context"][seq, item_idx]  # the actual underlying range context, not the label
                        testset_input_n_context.append(np.append(np.append(inputA, inputB), context))
                        seq_record.append([seq, item_idx])

                    inputB = testset["input"][seq, item_idx]  # set the previous state to be the current state

            elif TRIAL_TYPE==const.TRIAL_FILLER:
                if trialtype==TRIAL_TYPE:
                    inputA = testset["input"][seq, item_idx]
                    context = testset["context"][seq, item_idx]  # the actual underlying range context, not the label
                    testset_input_n_context.append(np.append(inputA, context))
                    seq_record.append([seq, item_idx])

    return seq_record, testset_input_n_context


def flatten_lists_to_arrays(testset, sequence_id, seqitem_id, allvarkeys):
    """
    This function flatten_lists_to_arrays() takes a list of dictionary keys and a test set,
     and creates a list of squashed arrays from the elements accessed by those keys at particular
     indices we care about (sequence_id, seqitem_id). For tidying up get_activations()
    """
    arrayvars = []
    for key in allvarkeys:
        thisvar = np.asarray([testset[key][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
        arrayvars.append(thisvar)

    return arrayvars


def get_activations(args, trainset,trained_model, train_loader, whichType='compare'):
    """ This will determine the hidden unit activations for each input pair in the train/test set.

     There are many repeats of each input pair in the train/test set. This will
     lead to slightly different activations for each instance of a particular input pair.
     We therefore take our activation for that unique input pair as the average
     activation over all instances of the pair in the training set.
      - messy but functional.
    """
    # reformat the input sequences for our recurrent model
    recurrentinputs = []
    sequenceLength = trainset["input"].shape[1]
    TRIAL_TYPE = const.TRIAL_COMPARE if whichType=='compare' else const.TRIAL_FILLER
    print("TRIAL_TYPE", TRIAL_TYPE)

    # determine the unique inputs for the training set (there are repeats)
    # consider activations at all instances, then average these activations to get the mean per unique input.
    trainset_input_n_context, seq_record = [[] for i in range(2)]
    for seq in range(len(trainset["input"])):

        inputA, inputB = [None for i in range(2)]
        for item_idx in range(len(trainset["input"][seq])):
            trialtype = trainset["trialtypeinputs"][seq, item_idx]

            if TRIAL_TYPE==const.TRIAL_COMPARE:
                if trialtype==TRIAL_TYPE:
                    inputA = trainset["input"][seq, item_idx]
                    if inputB is not None:
                        if np.all(inputA==inputB):
                            print('Warning: adjacent trial types are same number {}, both of type compare at item {} in sequence {}'.format(dset.turn_one_hot_to_integer(inputA)[:], item_idx,seq))
                        context = trainset["context"][seq, item_idx]  # the actual underlying range context, not the label
                        trainset_input_n_context.append(np.append(np.append(inputA, inputB), context))
                        seq_record.append([seq, item_idx])

                    inputB = trainset["input"][seq, item_idx]  # set the previous state to be the current state

            elif TRIAL_TYPE==const.TRIAL_FILLER:
                if trialtype==TRIAL_TYPE:
                    inputA = trainset["input"][seq, item_idx]
                    context = trainset["context"][seq, item_idx]  # the actual underlying range context, not the label
                    trainset_input_n_context.append(np.append(inputA, context))
                    seq_record.append([seq, item_idx])


    #trainset_input_n_context = [np.append(trainset["input"][i, j],trainset["contextinput"][i]) for i in range(len(trainset["input"]))]  # ignore the context label, but consider the true underlying context
    unique_inputs_n_context, uniqueind = np.unique(trainset_input_n_context, axis=0, return_index=True)
    # print("unique_inputs_n_context", unique_inputs_n_context.shape)
    # print("uniqueind", uniqueind)
    N_unique = (unique_inputs_n_context.shape)[0]
    sequence_id = [seq_record[uniqueind[i]][0] for i in range(len(uniqueind))]
    seqitem_id = [seq_record[uniqueind[i]][1] for i in range(len(uniqueind))]
    num_unique = len(uniqueind)
    trainsize = trainset["label"].shape[0]

    unique_inputs = np.asarray([trainset["input"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    unique_labels = np.asarray([trainset["label"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    unique_context = np.asarray([trainset["context"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    unique_refValue = np.asarray([trainset["refValue"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    unique_judgementValue = np.asarray([trainset["judgementValue"][sequence_id[i]][seqitem_id[i]] for i in range(len(sequence_id))])
    # print("unique_inputs", unique_inputs.shape)
    # print("unique_labels", unique_labels.shape)
    # print("unique_context", unique_context.shape)
    # print("unique_refValue", unique_refValue.shape)
    # print("unique_judgementValue", unique_judgementValue.shape)

    # preallocate some space...
    labels_refValues = np.empty((len(uniqueind),1))
    labels_judgeValues = np.empty((len(uniqueind),1))
    # print("uniqueind", uniqueind)
    contexts = np.empty((len(uniqueind),1))
    time_index = np.empty((len(uniqueind),1))
    MDSlabels = np.empty((len(uniqueind),1))
    hdim = trained_model.hidden_size
    rdim = trained_model.recurrent_size
    activations = np.empty((len(uniqueind), hdim))
    temporal_context = np.zeros((trainsize,sequenceLength))            # for tracking the evolution of context in the training set
    temporal_trialtypes = np.zeros((trainsize,sequenceLength))
    temporal_activation_drift = np.zeros((trainsize, sequenceLength, rdim))

    #  Tally activations for each unique context/input instance, then divide by the count (i.e. take the mean across instances)
    aggregate_activations = np.zeros((len(uniqueind), hdim))  # for adding each instance of activations to
    counter = np.zeros((len(uniqueind),1)) # for counting how many instances of each unique input/context we find

    #  pass each input through the network and see what happens to the hidden layer activations
    if not ((args.network_style=='recurrent') and args.retain_hidden_state):
        print("(args.network_style=='recurrent') and args.retain_hidden_state")
        for sample in range(len(uniqueind)):
            sample_input = batch_to_torch(torch.from_numpy(unique_inputs[sample]))
            sample_label = unique_labels[sample]
            labels_refValues[sample] = dset.turn_one_hot_to_integer(unique_refValue[sample])
            labels_judgeValues[sample] = dset.turn_one_hot_to_integer(unique_judgementValue[sample])
            MDSlabels[sample] = sample_label
            contexts[sample] = dset.turn_one_hot_to_integer(unique_context[sample])
            time_index[sample] = 0  # doesnt mean anything for these not-sequential cases
            counter[sample] = 0     # we dont care how many instances of each unique input for these non-sequential cases

            # get the activations for that input
            if args.network_style=='mlp':
                h1activations,h2activations,_ = trained_model.get_activations(sample_input)
            elif args.network_style=='recurrent':
                if not args.retain_hidden_state:
                    # reformat the paired input so that it works for our recurrent model
                    context = sample_input[contextrange]
                    inputA = (torch.cat((sample_input[Arange], context),0)).unsqueeze(0)
                    inputB = (torch.cat((sample_input[Brange], context),0)).unsqueeze(0)
                    recurrentinputs = [inputA, inputB]
                    h0activations = torch.zeros(1,trained_model.recurrent_size)  # reset hidden recurrent weights

                    # pass inputs through the recurrent network
                    for i in range(2):
                        h0activations,h1activations,_ = trained_model.get_activations(recurrentinputs[i], h0activations)

                    activations[sample] = h1activations.detach()

    else:
        # Do a single pass through the whole training set and look out for ALL instances of each unique input.
        # reset hidden recurrent weights on the very first trial

        h0activations = torch.zeros(1, trained_model.recurrent_size)
        latentstate = torch.zeros(1, trained_model.recurrent_size)

        for batch_idx, data in enumerate(train_loader):
            inputs, labels, contextsequence, contextinputsequence, trialtype = batch_to_torch(data['input']), data['label'].type(torch.FloatTensor)[0].unsqueeze(1).unsqueeze(1), batch_to_torch(data['context']), batch_to_torch(data['contextinput']), batch_to_torch(data['trialtypeinput']).unsqueeze(2)
            recurrentinputs = []
            sequenceLength = inputs.shape[1]
            temporal_trialtypes[batch_idx] = data['trialtypeinput']

            for i in range(sequenceLength):
                temporal_context[batch_idx, i] = dset.turn_one_hot_to_integer(contextinputsequence[:,i][0])
                contextin = contextinputsequence[:,i]
                if trialtype[0,i]==0:  # remove context indicator on the filler trials
                    contextinput = torch.full_like(contextin, 0)
                else:
                    contextinput = copy.deepcopy(contextin)

                inputX = torch.cat((inputs[:, i], contextinput, trialtype[:,i]),1)
                recurrentinputs.append(inputX)

            h0activations = latentstate
            inputA, inputB = [None for i in range(2)]

            # perform N-steps of recurrence
            for item_idx in range(sequenceLength):
                h0activations,h1activations,_ = trained_model.get_activations(recurrentinputs[item_idx], h0activations)
                if item_idx==(sequenceLength-2):  # extract the hidden state just before the last input in the sequence is presented
                    latentstate = h0activations.detach()

                temporal_activation_drift[batch_idx, item_idx,:] = h0activations.detach()   # Note: not currently used
                context = contextsequence[:,item_idx]

                # for 'compare' trials only, evaluate performance at every comparison between the current input and previous 'compare' input
                if trialtype[0,item_idx] == TRIAL_TYPE:
                    if TRIAL_TYPE==const.TRIAL_COMPARE:    # if we are looking at act. for the compare trials only
                        inputA = recurrentinputs[item_idx][0][:const.TOTALMAXNUM]
                        if inputB is not None:
                            input_n_context = np.append(np.append(inputA, inputB), context)  # actual underlying range context
                            for i in range(N_unique):
                                if np.all(unique_inputs_n_context[i,:]==input_n_context):
                                    index = i
                                    break

                            activations[index] = h1activations.detach()
                            labels_refValues[index] = dset.turn_one_hot_to_integer(unique_refValue[index])
                            labels_judgeValues[index] = dset.turn_one_hot_to_integer(unique_judgementValue[index])
                            MDSlabels[index] = unique_labels[index]
                            contexts[index] = dset.turn_one_hot_to_integer(unique_context[index])
                            time_index[index] = batch_idx
                        inputB = recurrentinputs[item_idx][0][:const.TOTALMAXNUM]  # previous state <= current state

                    else:  # for filler trials only, consider just the current number and context
                        inputA = recurrentinputs[item_idx][0][:const.TOTALMAXNUM]
                        input_n_context = np.append(inputA, context)  # actual underlying range context
                        for i in range(N_unique):
                            if np.all(unique_inputs_n_context[i,:]==input_n_context):
                                index = i
                                break
                        activations[index] = h1activations.detach()
                        labels_refValues[index] = dset.turn_one_hot_to_integer(unique_refValue[index])
                        labels_judgeValues[index] = dset.turn_one_hot_to_integer(unique_judgementValue[index])
                        MDSlabels[index] = unique_labels[index]
                        contexts[index] = dset.turn_one_hot_to_integer(unique_context[index])
                        time_index[index] = batch_idx

                    if item_idx > 0:
                        # Aggregate activity associated with each instance of each input
                        aggregate_activations[index] += activations[index]
                        counter[index] += 1    # captures how many instances of each unique input there are in the training set

        # Now turn the aggregate activations into mean activations by dividing by the number of each unique input/context instance
        for i in range(counter.shape[0]):
            if counter[i]==0:
                counter[i]=1  # prevent divide by zero
                print('Warning: index ' + str(i) + ' input had no instances?')

        activations = np.divide(aggregate_activations, counter)

    # Finally, reshape the output activations and labels so that we can easily interpret RSA on the activations
    # print("2labels_judgeValues:", labels_judgeValues)
    # sort all variables first by context order
    context_ind = np.argsort(contexts, axis=0)
    contexts = np.take_along_axis(contexts, context_ind, axis=0)
    activations = np.take_along_axis(activations, context_ind, axis=0)
    MDSlabels = np.take_along_axis(MDSlabels, context_ind, axis=0)
    labels_refValues = np.take_along_axis(labels_refValues, context_ind, axis=0)
    labels_judgeValues = np.take_along_axis(labels_judgeValues, context_ind, axis=0)
    time_index = np.take_along_axis(time_index, context_ind, axis=0)
    counter = np.take_along_axis(counter, context_ind, axis=0)
    # print("1labels_judgeValues:", labels_judgeValues)

    # within each context, sort according to numerosity of the judgement value
    
    for context in range(1,const.NCONTEXTS+1):
        
        # Initialize an empty list to store indices
        ind = []
        # Iterate over the range of the number of rows in the 'contexts' array
        for i in range(contexts.shape[0]):
            # Check if the element at index 'i' in 'contexts' is equal to 'context'
            if contexts[i] == context:
                # If the condition is met, append the index 'i' to the list 'ind'
                ind.append(i)
        # print("context:", context)
        # print("ind:", ind)

        # print("ind[0]:", ind[0])
        # Since the values are exclusive to the context, only have to resort the values within the context
        # so if ind is empty, the following lines will not have to be done.
        # ! might have to chnage that only the ind[0] isnt called.
        if ind != []:
            numerosity_ind = np.argsort(labels_judgeValues[ind], axis=0) + ind[0]
            labels_judgeValues[ind] = np.take_along_axis(labels_judgeValues, numerosity_ind, axis=0)
            labels_refValues[ind] = np.take_along_axis(labels_refValues, numerosity_ind, axis=0)
            contexts[ind] = np.take_along_axis(contexts, numerosity_ind, axis=0)
            MDSlabels[ind] = np.take_along_axis(MDSlabels, numerosity_ind, axis=0)
            activations[ind] = np.take_along_axis(activations, numerosity_ind, axis=0)
            time_index[ind] = np.take_along_axis(time_index, numerosity_ind, axis=0)
            counter[ind] = np.take_along_axis(counter, numerosity_ind, axis=0)

    drift = {"temporal_activation_drift":temporal_activation_drift, "temporal_context":temporal_context}
    return activations, MDSlabels, labels_refValues, labels_judgeValues, contexts, time_index, counter, drift, temporal_trialtypes


class OneStepRNN(nn.Module):

    def __init__(self, D_in, D_out, noise_std, recurrent_size, hidden_size):
        super(OneStepRNN, self).__init__()
        self.recurrent_size = recurrent_size  # was 33 default to match to the parallel MLP; now larger to prevent bottleneck on context rep
        self.hidden_size = hidden_size   # was 60 default
        self.hidden_noise = noise_std
        self.input2hidden = nn.Linear(D_in + self.recurrent_size, self.recurrent_size)
        self.input2fc1 = nn.Linear(D_in + self.recurrent_size, self.hidden_size)  # size input, size output
        self.fc1tooutput = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        self.hidden = F.relu(self.input2hidden(combined))
        self.fc1_activations = F.relu(self.input2fc1(combined))
        self.output = torch.sigmoid(self.fc1tooutput(self.fc1_activations))
        return self.output, self.hidden

    def get_activations(self, x, hidden):
        self.forward(x, hidden)  # update the activations with the particular input
        return self.hidden, self.fc1_activations, self.output

    def get_noise(self):
        return self.hidden_noise


def define_hyperparams():
    """
    This will enable us to take different network training settings/hyperparameters in when we call main.py from the command line.
    e.g. python3 main.py --batch-size=12 --epochs=20 --save-model
    Or if you want it to execute train/test across multiple combinations of hyperparameters, you can define multiple as follows:
    e.g. python3 main.py --batch-size-multi 12 24 48 --lr-multi 0.01 0.001
    If you are running this from a notebook and not the command line, just adjust the params specified in the class argparser()
    """
    args = argsparser()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    command_line = True  # if running from jupyter notebook, set this to false and adjust argsparser() instead
    if command_line:
        parser = argparse.ArgumentParser(description='PyTorch network settings')

        # dataset hyperparameters
        parser.add_argument('--network-style', default="recurrent", help='which network we want, "recurrent" or "mlp" (default: "recurrent")')
        parser.add_argument('--new-dataset', dest='create_new_dataset', action='store_true', help='create a new dataset for this condition? (default: False)')   # re-generate the random train/test dataset each time?
        parser.add_argument('--reuse-dataset', dest='create_new_dataset', action='store_false', help='reuse the existing dataset for this condition? (default: True)')
        parser.add_argument('--remove-fillers', dest='include_fillers', action='store_false', default=False, help='remove fillers from the dataset? (default: False)')     # True: task is like Fabrice's with filler trials; False: solely compare trials
        parser.add_argument('--which_context', type=int, default=0, help='if we want to train on a single context range only: 0=all contexts, 1=low only, 2=high only, (default: 0)')
        parser.add_argument('--interleave', dest='all_fullrange', action='store_true', help='interleave training (default: False)')
        parser.add_argument('--blockrange', dest='all_fullrange', action='store_false', help='block training by contextual number range (default: True)')
        parser.add_argument('--reset-state', dest='retain_hidden_state', action='store_false', help='reset the hidden state between sequences (default: False)')
        parser.add_argument('--retain-state', dest='retain_hidden_state', action='store_true', help='retain the hidden state between sequences (default: True)')
        parser.add_argument('--label-context', default="true", help='label the context explicitly in the input stream? (default: "true", other options: "constant (1)", "random (1-3)")')
        parser.add_argument('--block_int_ttsplit', default="false", help='test on a different blocking/interleaving structure than training? (default: "false", train/test on same e.g. train block, test block")')
        parser.add_argument('--retrain_decoder', default="false", help='whether to retrain the final layer of a trained network, this time using VI. default: "false"')
        parser.add_argument('--train_long', default="false", help='determines if on train_short (no linking pair) or train_long (just linking pair) dataset (default: "false")')
        parser.add_argument('--original_model_name', default="", help='do not adjust manually: to be used for specifying the name of old trained networks to be retrained under new conditions.')

        # network training hyperparameters
        parser.add_argument('--modeltype', default="aggregate", help='input type for selecting which network to train (default: "aggregate", concatenates pixel and location information)')
        parser.add_argument('--train-lesion-freq', default=0.0, type=float, help='frequency of number lesions on compare trials, during training (default=0.0)')
        parser.add_argument('--batch-size-multi', nargs='*', type=int, help='input batch size (or list of batch sizes) for training (default: 48)', default=[1])
        parser.add_argument('--lr-multi', nargs='*', type=float, help='learning rate (or list of learning rates) (default: 0.001)', default=[0.0001])
        parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 48)')
        parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: 48)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--weight_decay', type=int, default=0.0000, metavar='N', help='weight-decay for l2 regularisation (default: 0)')
        parser.add_argument('--save-model', action='store_true', help='For Saving the current Model')
        parser.add_argument('--recurrent-size', type=int, default=200, metavar='N', help='number of nodes in recurrent layer (default: 33)')
        parser.add_argument('--hidden-size', type=int, default=200, metavar='N', help='number of nodes in hidden layer (default: 60)')
        parser.add_argument('--BPTT-len', type=int, default=120, metavar='N', help='length of sequences that we backprop through (default: 120 = whole block length)')
        parser.add_argument('--noise_std', type=float, default=0.0, metavar='N', help='standard deviation of iid noise injected into the recurrent hiden state between numerical inputs (default: 0.0).')
        parser.add_argument('--model-id', type=int, default=0, metavar='N', help='for distinguishing many iterations of training same model (default: 0).')

        parser.set_defaults(create_new_dataset=True, all_fullrange=False, retain_hidden_state=True, retrain_decoder=False, train_long=False)
        args = parser.parse_args()

    if args.which_context>0:
        args.all_fullrange = False         # cant intermingle over context ranges if you only have one context range
    multiparams = [args.batch_size_multi, args.lr_multi]
    return args, device, multiparams


def log_performance(writer, epoch, train_perf, test_perf):
    """ Write out the training and testing performance for this epoch to tensorboard.
          - 'writer' is a SummaryWriter instance
    Note: -  '_standard' means its the typical way people assess training performance vs test, which I think is not a fair comparison,
          because train performance will be average performance across the epoch while network is optimising/changing, vs test which is performance
          on the optimised network over that epoch.
          -  I am logging both this standard train metric and also train performance at the end of the epoch (which is a fairer comparison to test)
    """
    standard_train_loss, standard_train_accuracy, fair_train_loss, fair_train_accuracy = train_perf
    test_loss, test_accuracy = test_perf

    writer.add_scalar('Loss/training_standard', standard_train_loss, epoch)  # inputs: tag, value, iteration
    writer.add_scalar('Loss/training_fair', fair_train_loss, epoch)
    writer.add_scalar('Loss/testing', test_loss, epoch)
    writer.add_scalar('Accuracy/training_standard', standard_train_accuracy, epoch)
    writer.add_scalar('Accuracy/training_fair', fair_train_accuracy, epoch)
    writer.add_scalar('Accuracy/testing', test_accuracy, epoch)


class argsparser():
    """For holding network training arguments, usually entered via command line"""
    def __init__(self):
        self.batch_size = 24    # how many trials that are seen at one time
        self.test_batch_size = 24
        self.epochs = 50
        self.lr = 0.002 # learning rate : bigger means faster learning, but can get stuck
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 1000
        self.weight_decay = 0.00
        self.save_model = False
        self.recurrent_size = 33
        self.hidden_size = 60
        self.BPTT_len = 120
        self.train_lesion_freq = 0.0


def get_dataset_name(args):
    """Return the (unique) name of the dataset, trained model, analysis and training record, based on args.
    """
    # convert the hyperparameter settings into a string ID
    if args.block_int_ttsplit == False:
        ttsplit = ''
    else:
        ttsplit = '_traintestblockintsplit'

    str_args = '_bs'+ str(args.batch_size_multi[0]) + '_lr' + str(args.lr_multi[0]) + '_ep' + str(args.epochs) + '_r' + str(args.recurrent_size) + '_h' + str(args.hidden_size) + '_bpl' + str(args.BPTT_len) + '_trlf' + str(args.train_lesion_freq) + '_id'+ str(args.model_id)
    networkTxt = 'RNN' if args.network_style == 'recurrent' else 'MLP'
    contextlabelledtext = '_'+args.label_context+'contextlabel'
    hiddenstate = '_retainstate' if args.retain_hidden_state else '_resetstate'
    rangetxt = '_numrangeintermingled' if args.all_fullrange else '_numrangeblocked'
    retraindecodertxt = '_retraineddecoderVI' if args.retrain_decoder else ''

    if args.which_context==0:
        whichcontexttext = ''
    elif args.which_context==1:
        # whichcontexttext = '_lowrange_1-4_only'
        whichcontexttext = '_lowrange_{}-{}_only'.format(const.LOWR_LLIM, const.LOWR_ULIM)
    elif args.which_context==2:
        # whichcontexttext = '_highrange_5-8_only'
        whichcontexttext = '_highrange_{}-{}_only'.format(const.HIGHR_LLIM, const.HIGHR_ULIM)
        
    if args.train_long:
        train_state = '_trainlong'
    else:
        train_state = '_trainshort'

    datasetname = 'dataset'+train_state+whichcontexttext+contextlabelledtext+rangetxt + '_bpl' + str(args.BPTT_len) + '_id'+ str(args.model_id)
    analysis_name = const.NETANALYIS_DIRECTORY +'MDSanalysis_'+networkTxt+train_state+whichcontexttext+contextlabelledtext+rangetxt+hiddenstate+'_n'+str(args.noise_std)+str_args + ttsplit + retraindecodertxt
    trainingrecord_name = '_trainingrecord_'+ networkTxt + train_state+whichcontexttext+contextlabelledtext+rangetxt+hiddenstate+'_n'+str(args.noise_std)+str_args+retraindecodertxt
    if args.network_style=='recurrent':
        trained_modelname = const.MODEL_DIRECTORY + networkTxt+'_trainedmodel'+train_state+whichcontexttext+contextlabelledtext+rangetxt+hiddenstate+'_n'+str(args.noise_std)+str_args+retraindecodertxt+'.pth'
    else:
        trained_modelname = const.MODEL_DIRECTORY + networkTxt+'_trainedmodel'+train_state+whichcontexttext+contextlabelledtext+rangetxt+hiddenstate+str_args+retraindecodertxt+'.pth'

    return datasetname, trained_modelname, analysis_name, trainingrecord_name


def train_recurrent_network(args, device, multiparams, trainset, testset):
    """
    This function performs the train/test loop for different parameter settings
     input by the user in multiparams.
     - Train/test performance is logged with a SummaryWriter
     - the trained recurrent model is returned
     - note that the train and test set must be divisible by args.batch_size, do to the shaping of the recurrent input
     """
    _, _, _, trainingrecord_name = get_dataset_name(args)

    # Repeat the train/test model assessment for different sets of hyperparameters
    for batch_size, lr in product(*multiparams):
        args.batch_size = batch_size
        args.test_batch_size = batch_size
        args.lr = lr

        # Define a model for training
        #torch.manual_seed(1)         # if we want the same default weight initialisation every time
        if args.retrain_decoder: # SN: Change this to args.train_long
            print('retraining the decoder layer only')
            model = torch.load(args.original_model_name)
            for name, param in model.named_parameters():
               # if 'fc1tooutput' not in name:
                param.requires_grad = True  # (if retraining model) freeze all weights/biases except for decoder
                    # print('freeze these params: {}, {}'.format(name, param.shape))
               # else:
                 #   print('re-initialise and keep training these params: {}, {}'.format(name, param.shape))
                    # reinitialize these weights
                 #   stdv = 1. / math.sqrt(model.fc1tooutput.weight.size(1))
                 #   param.data.uniform_(-stdv, stdv)


        else:
            model = OneStepRNN(const.TOTALMAXNUM + const.NCONTEXTS + const.NTYPEBITS, 1, args.noise_std, args.recurrent_size, args.hidden_size).to(device)


        criterion = nn.BCELoss() #nn.CrossEntropyLoss()   # binary cross entropy loss
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Define our dataloaders
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

        # Log the model on TensorBoard and label it with the date/time and some other naming string
        now = datetime.now()
        date = now.strftime("_%d-%m-%y_%H-%M-%S")
        comment = "_batch_size-{}_lr-{}_epochs-{}_wdecay-{}".format(args.batch_size, args.lr, args.epochs, args.weight_decay)
        writer = SummaryWriter(log_dir=const.TB_LOG_DIRECTORY + trainingrecord_name + args.modeltype + date + comment)
        print("Open tensorboard in another shell to monitor network training (hannahsheahan$  tensorboard --logdir=runs)")

        # Train/test loop
        n_epochs = args.epochs
        printOutput = False
        trainingPerformance, testPerformance = [[] for i in range(2)]

        print("Training network...")

        # Take baseline performance measures
        optimizer.zero_grad()
        _, base_train_accuracy = recurrent_test(args, model, device, trainloader, criterion, printOutput)
        _, base_test_accuracy = recurrent_test(args, model, device, testloader, criterion, printOutput)
        print('Baseline train: {:.2f}%, Baseline test: {:.2f}%'.format(base_train_accuracy, base_test_accuracy))
        trainingPerformance.append(base_train_accuracy)
        testPerformance.append(base_test_accuracy)
        print_progress(0, n_epochs)

        standard_train_accuracy = 0.0
        epoch = 0
        #SN:  if statement here with args.train_long. 20 trials of train long
        while standard_train_accuracy < 90.0: # trains until the network is performing well on the training set

            # train network
            standard_train_loss, standard_train_accuracy = recurrent_train(args, model, device, trainloader, optimizer, criterion, epoch, printOutput)

            # assess network
            fair_train_loss, fair_train_accuracy = recurrent_test(args, model, device, trainloader, criterion, printOutput)
            test_loss, test_accuracy = recurrent_test(args, model, device, testloader, criterion, printOutput)

            # log performance
            train_perf = [standard_train_loss, standard_train_accuracy, fair_train_loss, fair_train_accuracy]
            test_perf = [test_loss, test_accuracy]
            trainingPerformance.append(standard_train_accuracy)
            testPerformance.append(test_accuracy)
            print('standard_train_accuracy:', standard_train_accuracy)
            print('Train: {:.2f}%, Test: {:.2f}%'.format(standard_train_accuracy, test_accuracy))
            epoch += 1
            log_performance(writer, epoch, train_perf, test_perf)
            print_progress(epoch, n_epochs)
        
        '''  # this is the trainig loop that trained for a set number of epochs
        # for epoch in range(1, n_epochs + 1):  # loop through the whole dataset this many times

        #     # train network
        #     standard_train_loss, standard_train_accuracy = recurrent_train(args, model, device, trainloader, optimizer, criterion, epoch, printOutput)

        #     # assess network
        #     fair_train_loss, fair_train_accuracy = recurrent_test(args, model, device, trainloader, criterion, printOutput)
        #     test_loss, test_accuracy = recurrent_test(args, model, device, testloader, criterion, printOutput)

        #     # log performance
        #     train_perf = [standard_train_loss, standard_train_accuracy, fair_train_loss, fair_train_accuracy]
        #     test_perf = [test_loss, test_accuracy]
        #     trainingPerformance.append(standard_train_accuracy)
        #     testPerformance.append(test_accuracy)
        #     print('Train: {:.2f}%, Test: {:.2f}%'.format(standard_train_accuracy, test_accuracy))
        #     log_performance(writer, epoch, train_perf, test_perf)
        #     print_progress(epoch, n_epochs)
        '''

        print("Training complete.")
        # save this training curve
        record = {"trainingPerformance":trainingPerformance, "testPerformance":testPerformance, "args":vars(args) }
        randnum = str(random.randint(0,10000))
        dat = json.dumps(record)
        f = open(const.TRAININGRECORDS_DIRECTORY+randnum + trainingrecord_name+".json","w")
        f.write(dat)
        
        f.close()

    writer.close()
    return model


def train_and_save_network(args, device, multiparams):
    """This function will:
    - create a new train/test dataset,
    - train a new RNN according to the hyperparameters in args on that dataset,
    - save the model (and training record) with an auto-generated name based on those args.
    """

    # define the network parameters
    datasetname, trained_modelname, analysis_name, _ = get_dataset_name(args)
    #SN: add args.train_long if statement here
    
    if args.create_new_dataset:
        print('Creating new dataset...')
        trainset, testset = dset.create_separate_input_data(datasetname, args)
    else:
        print('Loading existing dataset...')
        trainset, testset, _, _, _, _ = dset.load_input_data(const.DATASET_DIRECTORY, datasetname)
        
    dset.view_dataset_index_info(-50, args)

    # define and train a neural network model, log performance and output trained model
    if args.network_style == 'recurrent':
        print('Training recurrent network...')
        model = train_recurrent_network(args, device, multiparams, trainset, testset)
    else:
        print('Training MLP network...')
        model = trainMLPNetwork(args, device, multiparams, trainset, testset)

    # save the trained weights so we can easily look at them
    print('Saving trained model...')
    print(trained_modelname)
    torch.save(model, trained_modelname)
