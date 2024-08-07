"""
This is a selection of functions for creating a dataset for network training
on the contextual magnitude mapping project with Fabrice. The dataset is structured
to match the experimental conditions in Fabrice's EEG experiment as closely as possible.

Sheahan, H.*, Luyckx, F.*, Nelli, S., Taupe, C., & Summerfield, C. (2021). Neural
 state space alignment for magnitude generalisation in humans and recurrent networks.
 Neuron (in press)

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 13/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import constants as const
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import random
from sklearn.manifold import MDS
from sklearn.utils import shuffle
import copy
import magnitude_network as mnet

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def turn_one_hot(integer, maxSize):
    """This function will take as input an interger and output a one hot representation of that integer up to a max of maxSize."""
    oneHot = np.zeros((maxSize,1))
    oneHot[integer-(const.NCONTEXTS-1)] = 1  # SN old version: oneHot[integer-2] = 1 # CF -1 shifts the indexes to the right place
    return oneHot


def turn_one_hot_to_integer(onehot):
    """This function will take as input a one hot representation and determine the integer interpretation"""
    integer = np.nonzero(onehot)[0]
    return integer+1  # because we are starting counting from 1 not 0


def flatten_all_first_dim_arrays(*allarrays):
    """This function will flatten the first dimension of a series of input numpy arrays"""
    flatarrays = []
    for array in allarrays:
        array = flatten_first_dim(array)
        flatarrays.append(array)
    return  flatarrays


def flatten_first_dim(array):
    """This function with return a numpy array which flattens the first two dimensions together. Only works for 2d-4d np arrays."""
    if len(array.shape) == 2:
        return array.reshape(array.shape[0]*array.shape[1], )
    elif len(array.shape) == 3:
        return array.reshape(array.shape[0]*array.shape[1], array.shape[2])
    elif len(array.shape) == 4:
        return array.reshape(array.shape[0]*array.shape[1], array.shape[2], array.shape[3])
    else:
        print('Error: the array you are trying to partially flatten is not the correct shape.')


class CreateDataset(Dataset):
    """A class to hold a dataset.
    - judgementValue i.e. input2
    - refValue i.e. input1
    - total concatenate input = [input2,input1]
    - label
    """

    def __init__(self, dataset, transform=None):
        """
        Args:
            datafile (string): name of numpy datafile
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # load all original images too - yes memory intensive but useful. Note that this also removes the efficiency point of using dataloaders
        self.index = dataset['index']
        self.label = dataset['label']
        self.refValue = dataset['refValue']
        self.judgementValue = dataset['judgementValue']
        self.input = dataset['input']
        self.context = dataset['context']
        self.contextinput = dataset['contextinputs']
        self.index = (self.index).astype(int)
        self.trialtypeinput = dataset['trialtypeinputs']
        self.data = {'index':self.index, 'label':self.label, 'refValue':self.refValue, 'judgementValue':self.judgementValue, 'input':self.input, 'context':self.context, 'contextinput':self.contextinput, "trialtypeinput":self.trialtypeinput}
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # for retrieving either a single sample of data, or a subset of data
        # lets us retrieve several items at once (HRS: may not be actually used)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'index':self.index[idx], 'label':self.label[idx], 'refValue':self.refValue[idx], 'judgementValue':self.judgementValue[idx], 'input':self.input[idx], 'context':self.context[idx], 'contextinput':self.contextinput[idx], 'trialtypeinput':self.trialtypeinput[idx] }
        return sample


def load_input_data(fileloc,datasetname):
    # load an existing dataset
    print('Loading dataset: ' + datasetname + '.npy')
    data = np.load(fileloc+datasetname+'.npy', allow_pickle=True)
    numpy_trainset = data.item().get("trainset")
    numpy_testset = data.item().get("testset")
    numpy_crossvalset = data.item().get("crossval_testset")

    # turn out datasets into pytorch Datasets
    trainset = CreateDataset(numpy_trainset)
    testset = CreateDataset(numpy_testset)
    crossvalset = CreateDataset(numpy_crossvalset)

    return trainset, testset, crossvalset, numpy_trainset, numpy_testset, numpy_crossvalset


def generate_trial_sequence(include_fillers):
    """
    For generating a sequence of trials combining both the filler task and the compare task, as in Fabrice's experiment
    This will be used in create_separate_input_data()
     - this sequence will always be 120 trials long
     - compare trials are separated by between 2 and 4 filler trials (same as Fabrice's trial scheduling)
     - include_fillers flag determines whether our dataset will contain some filler
    trials like Fabrice used, or whether we have trials solely of the type 'compare'.
     """
    L3_trialtype, L4_trialtype, L5_trialtype= [ [] for i in range(3)]

    # generate 30 sequences, each will different numbers of full range filler trials
    for i in range(10):
        # the type of trials in the sequence
        if include_fillers:
            L3_trialtype.append([ 'compare','filler','filler'])
            L4_trialtype.append([ 'compare','filler','filler','filler'])
            L5_trialtype.append([ 'compare','filler','filler','filler','filler'])
        else:
            L3_trialtype.append([ 'compare','compare','compare'])
            L4_trialtype.append([ 'compare','compare','compare','compare'])
            L5_trialtype.append([ 'compare','compare','compare','compare','compare'])

    # concatenate and permute the ordering of these sequences
    type_sequence = [L3_trialtype, L4_trialtype, L5_trialtype]
    type_sequence = [i for sublist in type_sequence for i in sublist]
    permorder = np.random.permutation(np.asarray(range(len(type_sequence))))
    type_sequence = [type_sequence[permorder[i]] for i in range(len(permorder))]

    # flatten the trial sequences
    type_sequence = [i for sublist in type_sequence for i in sublist]

    return type_sequence


def turn_index_to_context(randind): # ! confused what this function does ; seems wrong for us
    """Get the context from the randomly sampled index for when contexts are intermingled"""
    # if randind < const.FULLR_ULIM:  # randind < 16
    #     context = 1
    # elif randind < const.FULLR_ULIM + const.LOWR_ULIM:  # randind < 27 (full 16+ lower 11)
    #     context = 2
    # else:
    #     context = 3
    
    if randind <= const.LOWR_ULIM:  # <= 4
        context = 1
    else:  # || elif randind > const.HIGHR_LLIM  # >= 5
        context = 2
    return context


def create_separate_input_data(filename, args):
    """This function will create a dataset of inputs for training/testing a network on a relational magnitude task.
    - There are 3 contexts if whichContext==0 (default), or just one range for any other value of whichContext (1-3).
    - the inputs to this function determine the structure in the training and test sets e.g. are they blocked by context.
    - BPTT_len specifies how long to back the sequences we backprop through. So far only works for BPTT_len <= block length
    - messy but functional. To be modularised.
    """
    print('Generating dataset...')
    if args.which_context==0:
        print('- all contexts included')
    elif args.which_context==1:
        print('- context range: {}-{}'.format(const.LOWR_LLIM, const.LOWR_ULIM))   # - context range: 1-4
    elif args.which_context==2:
        print('- context range: {}-{}'.format(const.HIGHR_LLIM, const.HIGHR_ULIM)) # - context range: 5-8
    if args.train_long:
        print('- training on long sequence')
    else:
        print('- training on short sequence')
    if args.label_context=='true':
        print('- network has correct context labelling')
    elif args.label_context=='random':
        print('- network has randomly assigned context labelling')
    elif args.label_context=='constant':
        print('- network has constant (1) context labelling')
    if args.all_fullrange:
        print('- compare numbers are all drawn from the full {}:{} range'.format(const.FULLR_LLIM, const.FULLR_ULIM))
    else:
        print('- compare numbers are drawn from temporally structured ranges')
        print('- training is blocked by context')
        print('- training orders A and B relative to each other in trial sequence (B @ trial t+1 == A @ trial t)')

    # set up the dataset parameters (Train and Test short have different parameters than Train and Test long)
    Mtestsets = const.MTESTSETS             # have multiple test sets for cross-validation of activations
    if args.train_long == False:
        print('- {} test sets generated for cross-validation'.format(Mtestsets))
        Ntrain = const.NTRAIN                          # how many examples we want to use (each of these is a sequence on numbers)
        Ntest = const.NTEST                            # needs to be big enough to almost guarantee that we will get instances of all 460 comparisons (you get 29 comparisons per sequence)
        Mblocks = const.MBLOCKS                     # same as fabrices experiment - there are 24 blocks across 3 different contexts
    else:
        Ntrain = const.NTRAIN_LONG
        Ntest = const.NTEST_LONG
        # if phase == 'train':
        #     Mblocks = const.MBLOCKS_LONG
        # else:
        #     Mblocks = const.MBLOCKS
          
    totalN = Ntrain + Mtestsets * Ntest           # how many sequences across training and test sets
    
    phases = ['train'  if i==0 else 'test' for i in range(Mtestsets+1)]
    testsets = [[] for i in range(Mtestsets)]
    whichtestset = 0                         # a counter

    for phase in phases:   # this method should balance context instances in train and test phases
        # appropriately set Mblocks and N values for the phase and short/long set
        if phase == 'train':
            N = copy.copy(Ntrain)
            if args.train_long == True:
                Mblocks = const.MBLOCKS_LONG
            else:
                Mblocks = const.MBLOCKS
        else:
            N = copy.copy(Ntest)
            Mblocks = const.MBLOCKS

        # perhaps set temporary N to N/24, then generate the data under each context and then shuffle order at the end?
        refValues = np.empty((Mblocks, int(N/Mblocks),args.BPTT_len, const.TOTALMAXNUM))
        judgementValues = np.empty((Mblocks, int(N/Mblocks),args.BPTT_len, const.TOTALMAXNUM))
        input = np.empty((Mblocks, int(N/Mblocks),args.BPTT_len, const.TOTALMAXNUM))
        contextinputs = np.empty((Mblocks, int(N/Mblocks), args.BPTT_len, const.NCONTEXTS ))
        target = np.empty((Mblocks, int(N/Mblocks),args.BPTT_len))
        contexts = np.empty((Mblocks, int(N/Mblocks), args.BPTT_len, const.NCONTEXTS))
        contextdigits = np.empty((Mblocks, int(N/Mblocks),args.BPTT_len))
        blocks = np.empty((Mblocks, int(N/Mblocks),1))
        trialTypes = np.empty((Mblocks, int(N/Mblocks), args.BPTT_len), dtype='str')  # 0='filler, 1='compare'; pytorch doesnt like string numpy arrays
        trialTypeInputs = np.empty((Mblocks, int(N/Mblocks), args.BPTT_len))
        if phase == 'train':
            trainindices = (np.asarray([i for i in range(Ntrain)])).reshape((Mblocks, int(Ntrain/Mblocks),1))
        else:
            testindices = (np.asarray([i for i in range(Ntest)])).reshape((Mblocks, int(Ntest/Mblocks),1))

        fillerRange = [const.FULLR_LLIM,const.FULLR_ULIM]        # the range of numbers spanned by all filler trials

        # print('args.train_long: ', args.train_long, 'phase: ', phase)
        
        for block in range(Mblocks): # Chooses context for each block
            if phase == 'train':
                if args.train_long == False: # train short is within the context
                    if args.which_context==0: 
                    # divide the blocks evenly across the 3 contexts
                        if block < Mblocks/const.NCONTEXTS:             # context A    # now 1-4
                            context = 1
                            minNumerosity = const.LOWR_LLIM
                            maxNumerosity = const.LOWR_ULIM
                            print('minNumerosity: ', minNumerosity, 'maxNumerosity', maxNumerosity)
                        elif block < 2*(Mblocks/const.NCONTEXTS):     # context B    # now 5-8
                            context = 2
                            minNumerosity = const.HIGHR_LLIM
                            maxNumerosity = const.HIGHR_ULIM
                            print('minNumerosity: ', minNumerosity, 'maxNumerosity: ', maxNumerosity)
                    # single context options
                    elif args.which_context==1:     # context A
                        print('\nlow range context')
                        context = 1
                        minNumerosity = const.LOWR_LLIM
                        maxNumerosity = const.LOWR_ULIM
                    elif args.which_context==2:     # context B
                        print('\nhigh range context')
                        context = 2
                        minNumerosity = const.HIGHR_LLIM
                        maxNumerosity = const.HIGHR_ULIM
                else:   # train long is just the linking pair
                    print('train long')
                    minNumerosity = const.LOWR_ULIM
                    maxNumerosity = const.HIGHR_LLIM
            else:
                # sets the numerosity to test the whole set (both contexts) for short and long
                print('args.train_long: ', args.train_long, 'phase: ', phase)
                minNumerosity = const.FULLR_LLIM
                maxNumerosity = const.FULLR_ULIM
    
            # set the range of numerosities for the context
            if args.train_long == True and phase == 'train': # Train long should only be on the linking pair between contexts
                randNumDistribution = Mtestsets = [i for i in range(const.LOWR_ULIM, const.HIGHR_LLIM+1)]
            else: # The whole range (information for linking pair is filtered out later)
                if args.all_fullrange: # args.all_fullrange == True = interleaved 
                    # print('interleaved')
                    tmpDistribution = [[i for i in range(const.FULLR_LLIM, const.FULLR_ULIM+1)],[j for j in range(const.LOWR_LLIM, const.LOWR_ULIM+1)], [k for k in range(const.HIGHR_LLIM, const.HIGHR_ULIM+1)]]
                    randNumDistribution = [i for sublist in tmpDistribution for i in sublist]  # non-uniform distr. over all 3 context ranges together
                else: # args.all_fullrange == False = blocked
                    # print('blocked')
                    randNumDistribution = [i for i in range(minNumerosity, maxNumerosity+1)]  # uniform between min and max
            
            indexDistribution = [i for i in range(len(randNumDistribution))]  # this is going to allow us to know which context a sample which have been drawn from if intermingled
            print('randNumDistribution: ', randNumDistribution)
            print('indexDistribution: ', indexDistribution)
            
            # generate some random numerosity data and label whether the random judgement integers are larger than the refValue
            firstTrialInContext = True              # reset the sequentialAB structure for each new context
            for sample in range(int(N/Mblocks)):    # each sequence
                input_sequence = []
                type_sequence  = generate_trial_sequence(args.include_fillers) # the order of filler trial and compare trials
                trialtypeinput = [0 for i in range(len(type_sequence))]
                contextsequence = []
                contextinputsequence = []
                previousFillerNum = None
                previousTrialtype = None

                # generate adjacent sequences of inputs, where no two adjacent elements within (or between) a sequence are the same
                for item in range(args.BPTT_len):
                    trial_type = type_sequence[item]
                    trialtypeinput[item] = 1 if trial_type=='compare' else 0    # provide a bit-flip input to say whether its a filler or compare trial

                    if trial_type == 'compare':
                        if (firstTrialInContext and (item==0)):
                            randind = random.choice(indexDistribution)
                            # SN print index distribution and randomNumDistribution to screen
                            # print('randind: ', randind)
                            refValue = randNumDistribution[randind]
                            if trial_type == 'filler':
                                print('Warning: sequence starting with a filler trial. This should not happen and will cause a bug in sequence generation.')
                        else:
                            refValue = copy.deepcopy(judgementValue)  # use the previous number and make sure its a copy not a reference to same piece of memory

                        randind = random.choice(indexDistribution)
                        
                        judgementValue = randNumDistribution[randind]
                        
                        #SN print judgement value and randNumDistribution to screen to be sure
                        #print('judgementValue: ', judgementValue)
                        #print('randNumDistribution: ', randNumDistribution)

                        while refValue==judgementValue:    # make sure we dont do inputA==inputB for two adjacent inputs
                            randind = random.choice(indexDistribution)
                            judgementValue = randNumDistribution[randind] 
                        
                        input2 = turn_one_hot(judgementValue, const.TOTALMAXNUM)
                        if args.all_fullrange or args.train_long or phase == 'test':  # if intermingling contexts, then we need to know which context this number was sampled from
                            context = turn_index_to_context(randNumDistribution[randind]) 

                    else:  # filler trial (note fillers are always from uniform 1:15 range)
                        input2 = turn_one_hot(random.randint(*fillerRange), const.TOTALMAXNUM)
                        # make sure (like Fabrice) that after a compare trial the subsequent filler isnt the same as the previous filler
                        if previousFillerNum is not None and previousTrialtype=='compare':
                            while all(input2 == previousFillerNum):
                                input2 = turn_one_hot(random.randint(*fillerRange), const.TOTALMAXNUM) # leave the filler numbers unconstrained just spanning the full range

                        previousFillerNum = copy.copy(input2)
                        # when the trials are interleaved, set filler trials to have random contets
                        # (NOTE this doesnt actually matter because context is later zeroed on fillers)
                        if args.all_fullrange:
                            context = random.randint(1, const.NCONTEXTS) # 1 or 2 

                    previousTrialtype = copy.copy(trial_type)

                    # Define the context input to the network
                    if args.label_context=='true':
                        contextinput = turn_one_hot(context, const.NCONTEXTS)  # there are 2 different contexts
                    elif args.label_context=='random':
                        # Note that NOT changing 'context' means that we should be able to see the correct range label in the RDM
                        contextinput = turn_one_hot(random.randint(1,3), const.NCONTEXTS)  # randomly assign each example to a context, (shuffling examples across context markers in training)
                    elif args.label_context=='constant':
                        # Note that NOT changing 'context' means that we should be able to see the correct range label in the RDM
                        contextinput = turn_one_hot(1, const.NCONTEXTS) # just keep this constant across all contexts, so the input doesnt contain an explicit context indicator

                    # add our new inputs to our sequence
                    input_sequence.append(input2)
                    contextsequence.append(context)
                    contextinputsequence.append(contextinput)

                if firstTrialInContext:
                    judgementValue = turn_one_hot_to_integer(input_sequence[-1])  # and then make sure that the next sequence starts where this one left off (bit of a hack)
                    firstTrialInContext = False

                # determine the correct rel. magnitude judgement for each pair of adjacent numbers in the sequence
                rValue = None
                judgeValue = None
                allJValues = np.zeros((args.BPTT_len, const.TOTALMAXNUM))
                allRValues = np.zeros((args.BPTT_len, const.TOTALMAXNUM))
                
                for i in range(args.BPTT_len):
                    trialtype = trialtypeinput[i]
                    if trialtype==1:  # compare
                        judgeValue = turn_one_hot_to_integer(input_sequence[i])
                        if rValue is not None:
                            if judgeValue==rValue:
                                print('Warning: something gone wrong at index {}.'.format(i))

                            if judgeValue > rValue:
                                target[block, sample, i] = 1
                            else:
                                target[block, sample, i] = 0
                                #SN don't want to do below for test long. could also add args.all_fullrange == True into statement
                            if args.train_long == False:
                                if (judgeValue <= const.LOWR_ULIM and rValue >= const.HIGHR_LLIM) or (judgeValue >= const.HIGHR_LLIM and rValue <= const.LOWR_ULIM):
                                    target[block, sample, i] = np.nan
                        else:
                            target[block, sample, i] = None  # default dont do anything

                    allJValues[i] = np.squeeze(turn_one_hot(turn_one_hot_to_integer(input_sequence[i]), const.TOTALMAXNUM))
                    #print('allJValues[i]: ', allJValues[i])
                    if rValue is None:
                        allRValues[i] = np.zeros((const.TOTALMAXNUM,))
                    else:
                        allRValues[i] = np.squeeze(turn_one_hot(rValue, const.TOTALMAXNUM))

                    if trialtype==1:
                        rValue = turn_one_hot_to_integer(input_sequence[i])  # set the previous state to be the current state

                #print('judgeValue: ', judgeValue)
                if firstTrialInContext:
                    judgementValue = copy.deepcopy(judgeValue)    # and then make sure that the next sequence starts with judgement where this one left off
                contextdigits[block, sample] = contextsequence
                judgementValues[block, sample] = np.squeeze(np.asarray(allJValues))
                refValues[block, sample] = np.squeeze(np.asarray(allRValues))
                contexts[block, sample] = np.squeeze([turn_one_hot(contextsequence[i], const.NCONTEXTS) for i in range(len(contextsequence))])  # still captures context here even if we dont feed context label into network
                contextinputs[block, sample] = np.squeeze(contextinputsequence)
                #input[block, sample] = np.squeeze(np.concatenate((input2,input1,contextinput)))  # for the MLP
                input[block, sample] = np.squeeze(np.asarray(input_sequence))             # for the RNN with BPTT
                blocks[block, sample] = block
                trialTypes[block, sample] = type_sequence
                trialTypeInputs[block, sample] = trialtypeinput
        
        if phase=='train':

            # now shuffle the training block order so that we temporally separate contexts a bit but still blocked
           # input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks, contextinputs, trialTypeInputs = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, trainindices, blocks, contextinputs, trialTypeInputs, random_state=0)

            # now flatten across the first dim of the structure
            input = flatten_first_dim(input)
            refValues = flatten_first_dim(refValues)
            judgementValues = flatten_first_dim(judgementValues)
            #print('judgementValues: ', judgementValues)
            target = flatten_first_dim(target)
            contexts = flatten_first_dim(contexts)
            contextdigits = flatten_first_dim(contextdigits)
            trainindices = flatten_first_dim(trainindices)
            blocks = flatten_first_dim(blocks)
            contextinputs = flatten_first_dim(contextinputs)
            trialTypeInputs  = flatten_first_dim(trialTypeInputs)

            trainset = { 'refValue':refValues, 'judgementValue':judgementValues, 'input':input, 'label':target, 'index':trainindices, 'context':contexts, 'contextdigits':contextdigits, 'contextinputs':contextinputs, "trialtypeinputs":trialTypeInputs }
        else:

            # now shuffle the training block order so that we temporally separate contexts a bit but still blocked
            # input, refValues, judgementValues, target, contexts, contextdigits, testindices, blocks, contextinputs, trialTypeInputs = shuffle(input, refValues, judgementValues, target, contexts, contextdigits, testindices, blocks, contextinputs, trialTypeInputs,  random_state=0)

            # now flatten across the first dim of the structure
            input = flatten_first_dim(input)
            refValues = flatten_first_dim(refValues)
            judgementValues = flatten_first_dim(judgementValues)
            #print('judgementValues: ', judgementValues)
            target = flatten_first_dim(target)
            contexts = flatten_first_dim(contexts)
            contextdigits = flatten_first_dim(contextdigits)
            testindices = flatten_first_dim(testindices)
            blocks = flatten_first_dim(blocks)
            contextinputs = flatten_first_dim(contextinputs)
            trialTypeInputs  = flatten_first_dim(trialTypeInputs)

            testsets[whichtestset] = { 'refValue':refValues, 'judgementValue':judgementValues, 'input':input, 'label':target, 'index':testindices, 'context':contexts, 'contextdigits':contextdigits, 'contextinputs':contextinputs, "trialtypeinputs":trialTypeInputs }
            whichtestset += 1

    # save the dataset so we can use it again
    testset = testsets[0]
    crossvalset = testsets[1]
    dat = {"trainset":trainset, "testset":testset, "crossval_testset":crossvalset}
    print(const.DATASET_DIRECTORY+filename)
    np.save(const.DATASET_DIRECTORY+filename+'.npy', dat)

    # turn out datasets into pytorch Datasets
    trainset = CreateDataset(trainset)
    testset = CreateDataset(testset)

    return trainset, testset

def view_dataset_index_info(array_index, args):
    '''
    This function will print out the information of the dataset at the given index. Used to compare judgementValue, 
    refValue, label, and context to ensure the dataset is correct.
    '''
    datasetname, trained_modelname, analysis_name, _ = mnet.get_dataset_name(args)
    trainset, testset, crossvalset, numpy_trainset, numpy_testset, numpy_crossvalset = load_input_data(const.DATASET_DIRECTORY, datasetname)
    # array_index = -50 # choose an index to check the dataset (check multiple indexes)
    print('array_index:',array_index)
    # print('judgementValue:',numpy_trainset['judgementValue'][array_index])
    # print('refValue:',numpy_trainset['refValue'][array_index])
    # print('label:',numpy_trainset['label'][array_index])
    #print('context:',numpy_trainset['context'][array_index])
    print('Trainset...')
    for i in range(len(numpy_trainset['judgementValue'][array_index])):
        judgementValue = 0
        refValue = 0
        label = 0
        context = -1
        # turn judgementValarray position to value
        for ind in range(len(numpy_trainset['judgementValue'][array_index][i])):
            if numpy_trainset['judgementValue'][array_index][i][ind] == 1:
                judgementValue = ind + 1
        # turn refValue array position to value  
        for ind in range(len(numpy_trainset['refValue'][array_index][i])):
            if numpy_trainset['refValue'][array_index][i][ind] == 1:
                refValue = ind + 1
        # grab label value
        label = numpy_trainset['label'][array_index][i]
        # grab context value
        for ind in range(len(numpy_trainset['context'][array_index][i])):
            if numpy_trainset['context'][array_index][i][ind] == 1:
                context = ind + 1
        # check if the judgementValue and refValue logic is correct
        
        print('judgementValue:',judgementValue, '\trefValue:',refValue, '\tlabel:',label, '\tcontext:',context)
        if (judgementValue < refValue and label == 1) or (judgementValue > refValue and label == 0):
          print('\tWRONG label!')
        elif ((judgementValue <= const.LOWR_ULIM and context == 2) or (judgementValue >= const.HIGHR_LLIM and context == 1)) and args.train_long == False:
          print('\tWRONG context!')
        elif ((judgementValue <= const.LOWR_ULIM and refValue >= const.HIGHR_LLIM) or (judgementValue >= const.HIGHR_LLIM and refValue <= const.LOWR_ULIM)) and (label == 1 or label == 0):
          if args.train_long == False:
            print('\tWRONG label != NaN!')
    print('Testset...')
    for i in range(len(numpy_testset['judgementValue'][array_index])):
        judgementValue = 0
        refValue = 0
        label = 0
        context = -1
        # turn judgementValarray position to value
        for ind in range(len(numpy_testset['judgementValue'][array_index][i])):
            if numpy_testset['judgementValue'][array_index][i][ind] == 1:
                judgementValue = ind + 1
        # turn refValue array position to value  
        for ind in range(len(numpy_testset['refValue'][array_index][i])):
            if numpy_testset['refValue'][array_index][i][ind] == 1:
                refValue = ind + 1
        # grab label value
        label = numpy_testset['label'][array_index][i]
        # grab context value
        for ind in range(len(numpy_testset['context'][array_index][i])):
            if numpy_testset['context'][array_index][i][ind] == 1:
                context = ind + 1
        # check if the judgementValue and refValue logic is correct
        
        print('judgementValue:',judgementValue, '\trefValue:',refValue, '\tlabel:',label, '\tcontext:',context)
        if (judgementValue < refValue and label == 1) or (judgementValue > refValue and label == 0):
          print('\tWRONG label!')
        elif ((judgementValue <= const.LOWR_ULIM and context == 2) or (judgementValue >= const.HIGHR_LLIM and context == 1)) and args.train_long == False:
          print('\tWRONG context!')
          
def create_dataset(args):
    datasetname, trained_modelname, analysis_name, _ = mnet.get_dataset_name(args)
    if args.create_new_dataset:
        print('Creating new dataset...')
        create_separate_input_data(datasetname, args)
        data = np.load(const.DATASET_DIRECTORY+datasetname+'.npy', allow_pickle=True)
        numpy_trainset = data.item().get("trainset")
        #print(numpy_trainset['judgementValue'][4])