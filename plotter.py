"""
This is a selection of functions for plotting figures in the magnitude project.

Sheahan, H.*, Luyckx, F.*, Nelli, S., Taupe, C., & Summerfield, C. (2021). Neural
 state space alignment for magnitude generalisation in humans and recurrent networks.
 Neuron (in press)

Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 14/12/2019
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import define_dataset as dset
import constants as const
import magnitude_network as mnet
import analysis_helpers as anh
import theoretical_performance as theory

from mpl_toolkits import mplot3d
import math
import numpy as np
import scipy as sp
import copy
import sys
import random
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.utils import shuffle


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def rotate_axes(x,y,theta):
    # theta is in degrees
    theta_rad = theta * (math.pi/180)  # convert to radians
    x_new = x * math.cos(theta_rad) + y * math.sin(theta_rad)
    y_new =  -x * math.sin(theta_rad) + y * math.cos(theta_rad)
    return x_new, y_new


def save_figure(basetitle, args, labelNumerosity, plot_diff_code, whichTrialType, saveFig):
    """This function will save the currently open figure with a base title and some details pertaining to how the activations were generated."""

    # conver the hyperparameter settings into a string ID
    if args.block_int_ttsplit == False:
        ttsplit = ''
    else:
        ttsplit = '_traintestblockintsplit'
    str_args = '_bs'+ str(args.batch_size_multi[0]) + '_lr' + str(args.lr_multi[0]) + '_ep' + str(args.epochs) + '_r' + str(args.recurrent_size) + '_h' +\
                str(args.hidden_size) + '_bpl' + str(args.BPTT_len) + '_trlf' + str(args.train_lesion_freq) + '_id' + str(args.model_id) + ttsplit

    # automatic save file title details
    if args.which_context==0:
        whichcontexttext = ''
    elif args.which_context==1:
        whichcontexttext = '_fullrange_1-16_only'
    elif args.which_context==2:
        whichcontexttext = '_lowrange_1-11_only'
    elif args.which_context==3:
        whichcontexttext = '_highrange_6-16_only'
    diffcodetext = '_diffcode' if plot_diff_code else ''
    retainstatetext = '_retainstate' if args.retain_hidden_state else '_resetstate'
    labeltext = '_number' if labelNumerosity else '_outcomes'
    networkTxt = 'RNN' if args.network_style == 'recurrent' else 'MLP'
    trialtypetxt = '_compare' if whichTrialType == 'compare' else '_filler'
    numberrangetxt = '_numrangeblocked' if args.all_fullrange==False else '_numrangeintermingled'
    if args.label_context=='true':
        contextlabelledtext = '_truecontextlabel'
    elif args.label_context=='random':
        contextlabelledtext =  '_randcontextlabel'
    elif args.label_context=='constant':
        contextlabelledtext =  '_constcontextlabel'

    if saveFig:
        plt.savefig(basetitle+networkTxt+whichcontexttext+numberrangetxt+diffcodetext+trialtypetxt+contextlabelledtext+labeltext+retainstatetext+'_n'+str(args.noise_std)+str_args+'.pdf',bbox_inches='tight')

    plt.close()
    return basetitle+networkTxt+whichcontexttext+numberrangetxt+diffcodetext+trialtypetxt+contextlabelledtext+labeltext+retainstatetext+'_n'+str(args.noise_std)+str_args


def shadeplot(ax, x_values, means, sems, colour='black'):
    """Plot mean+-sem shaded"""
    ax.fill_between(x_values, means-sems, means+sems, color=colour, alpha=0.25, linewidth=0.0)


def activation_rdms(MDS_dict, args, plot_diff_code, whichTrialType='compare', saveFig=True):
    """Plot the representational disimilarity structure of the hidden unit activations, sorted by context, and within that magnitude.
    Reorient the context order to match Fabrice's:  i.e. from (1-16, 1-11, 5-16) to (low, high, full)
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """
    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    if plot_diff_code:
        D = pairwise_distances(MDS_dict["diff_sl_activations"], metric='correlation')
        labelticks = ['-15:+15', '-10:+10', '-10:+10']
        ticks = [0,(const.FULLR_SPAN-1)*2, (const.FULLR_SPAN-1)*2 + (const.LOWR_SPAN-1)*2]
        differenceCodeText = 'differencecode_'
    else:
        act = MDS_dict["sl_activations"][:]

        if whichTrialType == 'filler':
            Dfull = act[0:const.FULLR_SPAN]
            Dlow = act[const.FULLR_SPAN:const.FULLR_SPAN+const.FULLR_SPAN]
            Dhigh = act[const.FULLR_SPAN+const.FULLR_SPAN:const.FULLR_SPAN+const.FULLR_SPAN+const.FULLR_SPAN]
            labelticks = ['1-16', '1-16', '1-16']
            ticks = [0,const.FULLR_SPAN,const.FULLR_SPAN*2]
        else:
            Dfull = act[0:const.FULLR_SPAN]
            Dlow = act[const.FULLR_SPAN:const.FULLR_SPAN+const.LOWR_SPAN]
            Dhigh = act[const.FULLR_SPAN+const.LOWR_SPAN:const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN]
            labelticks = ['25-35', '30-40', '25-40']
            ticks = [0, const.LOWR_SPAN, const.HIGHR_SPAN+const.LOWR_SPAN]

        D = np.concatenate((Dlow, Dhigh, Dfull), axis=0)

        #np.save('meanactivations_trlf'+str(args.train_lesion_freq), D)
        D = pairwise_distances(D, metric='correlation')
        differenceCodeText = ''

    im = plt.imshow(D, zorder=2, cmap='viridis', interpolation='nearest')

    #    divider = make_axes_locatable(ax[1])
    #    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im)
    cbar.set_label('disimilarity')
    ax.set_title('Averaged activations')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labelticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labelticks)

    n = save_figure(os.path.join(const.FIGURE_DIRECTORY,'RDM_'+differenceCodeText), args, False, plot_diff_code, whichTrialType, saveFig)


def plot_3mds(MDS_dict, args, labelNumerosity=True, whichTrialType='compare', saveFig=True):
    """This is a function to plot the MDS of activations and label according to numerosity and context"""

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

    # Plot the hidden activations for the 3 MDS dimensions
    colours = plt.cm.get_cmap('viridis')
    diffcolours = plt.cm.get_cmap('viridis')
    outcomecolours = ['red', 'green']

    norm = mplcol.Normalize(vmin=const.FULLR_LLIM, vmax=const.FULLR_ULIM)
    dnorm = mplcol.Normalize(vmin=-const.FULLR_ULIM+1, vmax=const.FULLR_ULIM-1)

    print(MDS_dict.keys())
    if not args.label_context:
        labels_contexts = np.full_like(MDS_dict["labels_contexts"], 1)
    else:
        labels_contexts = MDS_dict["labels_contexts"]
    MDS_act = MDS_dict["MDS_activations"]

    for k in range(5):
        fig,ax = plt.subplots(1,3, figsize=(10,3.3))
        for j in range(3):  # 3 MDS dimensions
            if j==0:
                dimA = 0
                dimB = 1
                ax[j].set_xlabel('dim 1')
                ax[j].set_ylabel('dim 2')
            elif j==1:
                dimA = 0
                dimB = 2
                ax[j].set_xlabel('dim 1')
                ax[j].set_ylabel('dim 3')
            elif j==2:
                dimA = 1
                dimB = 2
                ax[j].set_xlabel('dim 2')
                ax[j].set_ylabel('dim 3')

            for i in range((MDS_act.shape[0])):

                # colour by numerosity
                if k==0:   # difference labels

                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=diffcolours(dnorm(int(MDS_dict["labels_judgeValues"][i]-MDS_dict["labels_refValues"][i]))), s=20)
                    if j==2:
                        if i == (MDS_act.shape[0])-1:
                            cbar = fig.colorbar(im, ticks=[0,1])
                            if labelNumerosity:
                                cbar.ax.set_yticklabels(['-14','14'])
                elif k==1:  # B values
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=colours(norm(int(MDS_dict["labels_refValues"][i])-1)), s=20)
                    if j==2:
                        if i == (MDS_act.shape[0])-1:
                            cbar = fig.colorbar(im, ticks=[0,1])
                            if labelNumerosity:
                                cbar.ax.set_yticklabels(['1','16'])
                elif k==2:  # A values
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=colours(norm(int(MDS_dict["labels_judgeValues"][i])-1)), s=20)
                    if j==2:
                        if i == (MDS_act.shape[0])-1:
                            cbar = fig.colorbar(im, ticks=[0,1])
                            if labelNumerosity:
                                cbar.ax.set_yticklabels(['1','16'])
                elif k==3:  # context labels
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=const.CONTEXT_COLOURS[int(MDS_dict["labels_contexts"][i])-1], s=20)

                elif k==4:
                    im = ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=outcomecolours[int(MDS_dict["MDSlabels"][i])], s=20)


                # some titles
                if k==0:
                    ax[j].set_title('A - B labels')
                    #ax[j].axis('equal')
                    tx = 'AminusBlabel_'
                elif k==1:
                    ax[j].set_title('B labels')
                    tx = 'Blabel_'
                elif k==2:
                    ax[j].set_title('A labels')
                    tx = 'Alabel_'
                elif k==3:
                    ax[j].set_title('context labels')
                    tx = 'contlabel_'
                elif k==4:
                    ax[j].set_title('outcome labels')
                    tx = 'outcomelabel_'

                ax[j].set(xlim=(-1, 1), ylim=(-1, 1))  # set axes equal and the same for comparison

        n = save_figure(os.path.join(const.FIGURE_DIRECTORY,'3MDS60_' + tx), args, labelNumerosity, False, whichTrialType, saveFig)


def plot_3mds_mean(MDS_dict, args, labelNumerosity=True, plot_diff_code=False, whichTrialType='compare', saveFig=True, theta=0, axislimits = (-0.8,0.8), gradedcolour=False):
    """This function is just like plot_3mds and plot_3mdsContexts but for the formatting of the data which has been averaged across one of the two numerosity values.
    Because there are fewer datapoints I also label the numerosity inside each context, like Fabrice does.
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
     - rotate the angle of the data on the 2d component axes by angle theta (degrees). Note that this has no great meaning for MDS so its fine to do.
    """

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]
        Ns = [16,16,16]
    else:
        Ns = [16,11,11]

    fig,ax = plt.subplots(1,3, figsize=(18,5))
    rbg_contextcolours = [mplcol.to_rgba(i) for i in const.CONTEXT_COLOURS]
    white = (1.0, 1.0, 1.0, 1.0)

    diffcolours = get_cmap(20, 'magma')

    if plot_diff_code:
        MDS_act = MDS_dict["MDS_diff_slactivations"]
        contextlabel = MDS_dict["diff_sl_contexts"]
        numberlabel = MDS_dict["sl_diffValues"]
        differenceCodeText = 'differencecode_'
    else:
        MDS_act = MDS_dict["MDS_slactivations"]
        contextlabel = MDS_dict["sl_contexts"]
        numberlabel = MDS_dict["sl_judgeValues"]
        differenceCodeText = ''

    for j in range(3):  # 3 MDS dimensions

        if j==0:
            dimA = 0
            dimB = 1
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 2')
        elif j==1:
            dimA = 0
            dimB = 2
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 3')
        elif j==2:
            dimA = 1
            dimB = 2
            ax[j].set_xlabel('dim 2')
            ax[j].set_ylabel('dim 3')

        ax[j].set_title('context')

        if plot_diff_code:
            contextA = range((const.FULLR_SPAN-1)*2)
            contextB = range((const.FULLR_SPAN-1)*2, (const.FULLR_SPAN-1)*2 + (const.LOWR_SPAN-1)*2)
            contextC = range((const.FULLR_SPAN-1)*2 + (const.LOWR_SPAN-1)*2, (const.FULLR_SPAN-1)*2 + (const.LOWR_SPAN-1)*2 + (const.HIGHR_SPAN-1)*2)
        else:
            if whichTrialType=='filler':
                contextA = range(const.FULLR_SPAN)
                contextB = range(const.FULLR_SPAN,const.FULLR_SPAN*2)
                contextC = range(const.FULLR_SPAN*2, const.FULLR_SPAN*3)
            else:
                contextA = range(const.FULLR_SPAN)
                contextB = range(const.FULLR_SPAN,const.FULLR_SPAN+const.LOWR_SPAN)
                contextC = range(const.FULLR_SPAN+const.LOWR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)

            # Rotate the components on the 2d plot since global orientation doesnt matter (axes are arbitrary)
            rotated_act = copy.deepcopy(MDS_act)

            rotated_act[contextA, dimA], rotated_act[contextA, dimB] = rotate_axes(MDS_act[contextA, dimA], MDS_act[contextA, dimB], theta)
            rotated_act[contextB, dimA], rotated_act[contextB, dimB] = rotate_axes(MDS_act[contextB, dimA], MDS_act[contextB, dimB], theta)
            rotated_act[contextC, dimA], rotated_act[contextC, dimB] = rotate_axes(MDS_act[contextC, dimA], MDS_act[contextC, dimB], theta)

            ax[j].plot(rotated_act[contextA, dimA], rotated_act[contextA, dimB], color=const.CONTEXT_COLOURS[0])
            ax[j].plot(rotated_act[contextB, dimA], rotated_act[contextB, dimB], color=const.CONTEXT_COLOURS[1])
            ax[j].plot(rotated_act[contextC, dimA], rotated_act[contextC, dimB], color=const.CONTEXT_COLOURS[2])

        if gradedcolour:
            markercount=0
            lastc = -1
            for i in range((MDS_act.shape[0])):

                # create colour gradient within each context to signal numerosity
                c = int(contextlabel[i])
                if c!=lastc:
                    markercount=0
                lastc = int(contextlabel[i])
                graded_contextcolours = np.zeros((4, Ns[c]))
                for p in range(4):
                    graded_contextcolours[p] = np.linspace(white[p],rbg_contextcolours[c][p],Ns[c])
                gradedcolour = np.asarray([graded_contextcolours[p][markercount] for p in range(len(graded_contextcolours))])

                # colour by context
                ax[j].scatter(rotated_act[i, dimA], rotated_act[i, dimB], color=gradedcolour, edgecolor=const.CONTEXT_COLOURS[int(contextlabel[i])], s=80, linewidths=2)
                markercount +=1
                # label numerosity in white inside the marker
                firstincontext = [0,15,16,16+10,16+11, 16+21]
                if i in firstincontext:
                    ax[j].text(rotated_act[i, dimA], rotated_act[i, dimB], str(24+int(numberlabel[i])), color=const.CONTEXT_COLOURS[int(contextlabel[i])], size=15, horizontalalignment='center', verticalalignment='center')
        else:
            for i in range((MDS_act.shape[0])):
                ax[j].scatter(rotated_act[i, dimA], rotated_act[i, dimB], color=const.CONTEXT_COLOURS[int(contextlabel[i])], edgecolor=const.CONTEXT_COLOURS[int(contextlabel[i])], s=80, linewidths=2)
                ax[j].text(rotated_act[i, dimA], rotated_act[i, dimB], str(24+int(numberlabel[i])), color='white', size=6.5, horizontalalignment='center', verticalalignment='center')

        ax[j].axis('equal')

        if args.network_style=='mlp':
            ax[j].set(xlim=axislimits, ylim=axislimits)
        else:
            ax[j].set(xlim=axislimits, ylim=axislimits)

    n = save_figure(os.path.join(const.FIGURE_DIRECTORY,'3MDS60_'+differenceCodeText+'meanJudgement_'), args, labelNumerosity, plot_diff_code, whichTrialType, saveFig)


def animate_3d_mds(MDS_dict, args, plot_diff_code=False, whichTrialType='compare', saveFig=True):
    """ This function will plot the numerosity labeled, context-marked MDS projections
     of the hidden unit activations on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
     - use the flag 'plot_diff_code' to plot the difference signal (A-B) rather than the A activations
    """

    if whichTrialType=='filler':
        MDS_dict = MDS_dict["filler_dict"]

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    if plot_diff_code:
        slMDS = MDS_dict["MDS_diff_slactivations"]
        labels = MDS_dict["sl_diffValues"]
        differenceCodeText = 'differencecode_'
        # which MDS points correspond to which contexts
        contextA = range((const.FULLR_SPAN-1)*2)
        contextB = range((const.FULLR_SPAN-1)*2,(const.FULLR_SPAN-1)*2+(const.LOWR_SPAN-1)*2)
        contextC = range((const.FULLR_SPAN-1)*2+(const.LOWR_SPAN-1)*2, (const.FULLR_SPAN-1)*2+(const.LOWR_SPAN-1)*2+(const.HIGHR_SPAN-1)*2)
    else:
        slMDS = MDS_dict["MDS_slactivations"]
        labels = MDS_dict["sl_judgeValues"]
        differenceCodeText = ''
        if whichTrialType=='filler':
            contextA = range(const.FULLR_SPAN)
            contextB = range(const.FULLR_SPAN,const.FULLR_SPAN*2)
            contextC = range(const.FULLR_SPAN*2, const.FULLR_SPAN*3)

        else:
            contextA = range(const.FULLR_SPAN)
            contextB = range(const.FULLR_SPAN,const.FULLR_SPAN+const.LOWR_SPAN)
            contextC = range(const.FULLR_SPAN+const.LOWR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)

    def init():
        points = [contextA, contextB, contextC] #if labelContext else [contextA]

        for i in range(len(points)):
            ax.scatter(slMDS[points[i], 0], slMDS[points[i], 1], slMDS[points[i], 2], color=const.CONTEXT_COLOURS[i])

            if not plot_diff_code:  # the difference code is arranged differently
                ax.plot(slMDS[points[i], 0], slMDS[points[i], 1], slMDS[points[i], 2], color=const.CONTEXT_COLOURS[i])
            for j in range(len(points[i])):
                label = str(24+int(labels[points[i][j]]))
                ax.text(slMDS[points[i][j], 0], slMDS[points[i][j], 1], slMDS[points[i][j], 2], label, color='black', size=8, horizontalalignment='center', verticalalignment='center')
        ax.set_xlabel('MDS dim 1')
        ax.set_ylabel('MDS dim 2')
        ax.set_zlabel('MDS dim 3')
        ax.set(xlim=(-0.65, 0.65), ylim=(-0.65, 0.65), zlim=(-0.65, 0.65))
        #ax.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-3, 3))

        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)

    # save the animation as an mp4.
    if saveFig:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        strng = save_figure(const.ANIMATION_DIRECTORY + 'MDS_3Danimation_'+ differenceCodeText, args, True,  plot_diff_code, whichTrialType, False)
        anim.save(strng+'.mp4', writer=writer)


def instance_counter(MDS_dict, args, whichTrialType='compare'):
    """ Plot a histogram showing the number of times each unique input (reference averaged) and context was in the generated training set."""

    plt.figure()
    rangeA = np.arange(const.FULLR_SPAN)
    rangeB = np.arange(const.FULLR_SPAN,const.FULLR_SPAN+const.LOWR_SPAN)
    rangeC = np.arange(const.FULLR_SPAN+const.LOWR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)
    y = MDS_dict["sl_counter"].flatten()

    plt.bar(rangeA, y[rangeA], color='gold', edgecolor = 'gold')
    plt.bar(rangeB, y[rangeB], color='dodgerblue', edgecolor = 'dodgerblue')
    plt.bar(rangeC, y[rangeC], color='orangered', edgecolor = 'orangered')
    plt.xlabel('Numbers and contexts')
    plt.ylabel('Instances in training set')

    n = save_figure(os.path.join(const.FIGURE_DIRECTORY,'instance_counter_meanJudgement'), args, True, False, whichTrialType, saveFig)


def view_training_sequence(MDS_dict, args, whichTrialType='compare', saveFig=True):
    """Take the data loader and view how the contexts and latent states evolved in time in the training set.
    Also plots the sequence of compare vs filler trials.
    """

    MDS_latentstate = MDS_dict["drift"]["MDS_latentstate"]
    temporal_context = MDS_dict["drift"]["temporal_context"]
    temporal_trialtypes = MDS_dict["temporal_trialtypes"]

    # context in time/trials in training set
    plt.figure()
    plt.plot(temporal_context.flatten())
    plt.xlabel('Trials in training set')
    plt.ylabel('Context (0: 1-16; 1: 1-11; 2: 6-16)')
    n = save_figure(os.path.join(const.FIGURE_DIRECTORY,'temporalcontext_'), args, True, False, whichTrialType, saveFig)

    # trial types changing with time in training set
    plt.figure()
    plt.plot(temporal_trialtypes.flatten())
    plt.xlabel('Trials in training set')
    plt.ylabel('Trial type: 0-filler; 1-compare')
    n = save_figure(os.path.join(const.FIGURE_DIRECTORY,'temporaltrialtype_'), True, False, whichTrialType, saveFig)

    # latent state drift in time/trials in training set
    fig,ax = plt.subplots(1,3, figsize=(18,5))

    for j in range(3):  # 3 MDS dimensions
        if j==0:
            dimA = 0
            dimB = 1
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 2')
        elif j==1:
            dimA = 0
            dimB = 2
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 3')
        elif j==2:
            dimA = 1
            dimB = 2
            ax[j].set_xlabel('dim 2')
            ax[j].set_ylabel('dim 3')

        ax[j].set_title('latent state drift')

        # perhaps draw a coloured line between adjacent numbers
        # ax[j].plot(MDS_latentstate[:, dimA], MDS_latentstate[:, dimB], color='grey')

        #for i in range((MDS_latentstate.shape[0])):
        for i in range(2000,3500): # a subset of trials
            # colour by context
            ax[j].scatter(MDS_latentstate[i, dimA], MDS_latentstate[i, dimB], color=const.CONTEXT_COLOURS[int(temporal_context[i])-1], s=20)
            ax[j].plot([MDS_latentstate[i, dimA], MDS_latentstate[i+1, dimA]], [MDS_latentstate[i, dimB],MDS_latentstate[i+1, dimB]], color=const.CONTEXT_COLOURS[int(temporal_context[i])-1])

        ax[j].axis('equal')
        #ax[j].set(xlim=(-4, 4), ylim=(-4, 4))

    n = save_figure(os.path.join(const.FIGURE_DIRECTORY,'latentstatedrift_'), True, False, whichTrialType, saveFig)


def animate_3d_drift_mds(MDS_dict, args, whichTrialType='compare', saveFig=True):
    """ This function will plot the latent state drift MDS projections
     on a 3D plot, animate/rotate that plot to view it
     from different angles and optionally save it as a mp4 file.
     - currently unused.
    """
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    MDS_latentstate = MDS_dict["drift"]["MDS_latentstate"]
    temporal_context = MDS_dict["drift"]["temporal_context"]

    def init():

        #points = [contextA, contextB, contextC] #if labelContext else [contextA]

        for i in range(2000,3500):
            ax.scatter(MDS_latentstate[i, 0], MDS_latentstate[i, 1], MDS_latentstate[i, 2], color=const.CONTEXT_COLOURS[int(temporal_context[i])-1])
            #ax.plot(slMDS[points[i], 0], slMDS[points[i], 1], slMDS[points[i], 2], color=const.CONTEXT_COLOURS[i])

        ax.set_xlabel('MDS dim 1')
        ax.set_ylabel('MDS dim 2')
        ax.set_zlabel('MDS dim 3')
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)

    # save the animation as an mp4.
    if saveFig:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        strng = save_figure(const.ANIMATION_DIRECTORY + 'latentdrift_MDS_3Danimation_', True, False, whichTrialType, False)
        anim.save(strng+'.mp4', writer=writer)


def plot_optimal_perf(ax):
    """This function plots the performance in each context of theoretical agents
     making decisions using only the current number and knowledge of the local or global context median. """

    # set height of bar
    full_global_perf = 76.67
    full_local_perf = 76.67
    low_global_perf = 71.82
    low_local_perf = 77.27
    high_global_perf = 71.82
    high_local_perf = 77.27
    full_context_bars = [full_global_perf, full_local_perf, full_local_perf, full_local_perf, full_local_perf]
    low_context_bars = [low_global_perf, low_local_perf, low_local_perf, low_local_perf, low_local_perf]
    high_context_bars = [high_global_perf, high_local_perf, high_local_perf, high_local_perf, high_local_perf]

    # Set position of bar on X axis
    barWidth = 0.25
    r1 = np.arange(len(full_context_bars))
    r2 = [x + barWidth+0.02 for x in r1]
    r3 = [x + barWidth+0.02 for x in r2]

    for whichfig in range(len(ax)):
        h1 = ax[whichfig].bar(0,low_context_bars[whichfig], color=const.CONTEXT_COLOURS[1], alpha=0.5 )
        h2 = ax[whichfig].bar(1,high_context_bars[whichfig], color=const.CONTEXT_COLOURS[2], alpha=0.5 )
        h3 = ax[whichfig].bar(2,full_context_bars[whichfig], color=const.CONTEXT_COLOURS[0], alpha=0.5 )
    handles = [h1, h2, h3]

    return handles


def compare_lesion_tests(args, device):
    """
    This function compare_lesion_tests() compares the post-lesion test set performance of networks
     which were trained with different frequencies of lesions in the training set.
     - this will now search for the lesion assessments for all the model instances that match the args
     - this should now plot a dot +- SEM over model instances at each dot to see how variable it is.
     - Note that this uses the test set associated with the training set. So if we want to assess context
       use in context-interleaved networks, we really need to apply a different test set to assess properly.
        That doesnt happen in this function so beware: should only be used for context-blocked networks.
    """
    #frequencylist = [0.0, 0.1, 0.2, 0.3, 0.4]  # training frequencies of different networks to consider
    frequencylist = [0.0, 0.1]  # training frequencies of different networks to consider
    offsets = [0-.05,.2+0.02,.2+.25+0.04]  # for plotting
    overall_lesioned_tests = []

    plt.figure()
    fig, ax = plt.subplots(1,len(frequencylist), figsize=(3.5 * len(frequencylist),3.5))
    handles = plot_optimal_perf(ax)

    # file naming
    blcktxt = '_interleaved' if args.all_fullrange else '_temporalblocked'
    contexttxt = '_contextcued' if args.label_context=='true' else '_nocontextcued'
    range_txt = ''
    if args.which_context==0:
        range_txt = ''
    elif args.which_context==1:
        range_txt = '_fullrangeonly'
    elif args.which_context==2:
        range_txt = '_lowrangeonly'
    elif args.which_context==3:
        range_txt = '_highrangeonly'

    for whichfreq, train_lesion_frequency in enumerate(frequencylist):

        args.train_lesion_freq = train_lesion_frequency
        allmodels = anh.get_model_names(args)
        data = [[] for i in range(len(allmodels))]
        context_tests = np.zeros((const.NCONTEXTS, len(allmodels)))
        perf = np.zeros((const.NCONTEXTS, len(allmodels)))
        counts = np.zeros((const.NCONTEXTS, len(allmodels)))
        unlesioned_test = np.zeros((len(allmodels),))
        lesioned_test = np.zeros((len(allmodels),))

        # find all model ids that fit our requirements
        for ind, m in enumerate(allmodels):
            args.model_id = anh.get_id_from_name(m)
            print('modelid: ' + str(args.model_id))
            testParams = anh.setup_test_parameters(args, device)
            block_ttsplit_text = '_blockttsplit' if args.block_int_ttsplit else ''
            basefilename = const.LESIONS_DIRECTORY + 'lesiontests'+m[:-4] + block_ttsplit_text
            filename = basefilename+'.npy'

            # perform or load the lesion tests
            lesiondata, regulartestdata = anh.perform_lesion_tests(args, testParams, basefilename)
            data[ind] = lesiondata["bigdict_lesionperf"]
            lesioned_test[ind] = lesiondata["lesioned_testaccuracy"]
            unlesioned_test[ind] = regulartestdata["normal_testaccuracy"]

            # evaluate performance on the different contexts
            for seq in range(data[ind].shape[0]):
                for compare_idx in range(data[ind][seq].shape[0]):
                    context = data[ind][seq][compare_idx]["underlying_context"]-1
                    perf[context, ind] += data[ind][seq][compare_idx]["lesion_perf"]
                    counts[context, ind] += 1
            meanperf = 100 * np.divide(perf[:, ind], counts[:, ind])
            for context in range(const.NCONTEXTS):
                print('context {} performance: {}/{} ({:.2f}%)'.format(context+1, perf[context, ind], counts[context, ind], meanperf[context]))
                context_tests[context, ind] = meanperf[context]

        # now determine mean +-sem over models of that lesion frequency
        mean_lesioned_test = np.nanmean(lesioned_test)
        sem_lesioned_test = np.std(lesioned_test)

        mean_unlesioned_test = np.nanmean(unlesioned_test)
        sem_unlesioned_test = np.std(unlesioned_test)

        mean_contextlesion_test = np.nanmean(context_tests,axis=1)
        sem_contextlesion_test = np.std(context_tests,axis=1)


        # plot post-lesion performance divided up by context
        count =0
        for context in range(const.NCONTEXTS):
            colour = context+1 if context<2 else 0
            tmp = ax[whichfreq].errorbar(count, mean_contextlesion_test[colour], sem_contextlesion_test[colour], color=const.CONTEXT_COLOURS[colour], markersize=5, ecolor='black', markeredgecolor='black')
            ax[whichfreq].errorbar(count, mean_contextlesion_test[colour], sem_contextlesion_test[colour], color=const.CONTEXT_COLOURS[colour], markersize=5, marker='o', ecolor='black', markeredgecolor='black')
            count +=1
            if context==0:
                handles.append(tmp)
        print('\n')

    # format plotting
    for i, freq in enumerate(frequencylist):
        ax[i].set_xlabel('context')
        ax[i].set_ylabel(r'p(correct | $\epsilon_{train}$ = '+str(freq)+')')
        ax[i].set_ylim((60,85))
        ax[i].set_xticks([0,1,2])
        ax[i].set_xticklabels(['low','high','full'])
    plt.legend(handles[0:1],['prediction', 'RNN'])
    whichTrialType = 'compare'
    blocking_train_txt = 'interleavedtrain_' if args.all_fullrange else 'blockedtrain_'
    if args.block_int_ttsplit:
        blocking_test_txt = 'blockedtest_' if args.all_fullrange else 'interleavedtest_'
    else:
        blocking_test_txt = 'interleavedtest_' if args.all_fullrange else 'blockedtest_'

    save_figure(os.path.join(const.FIGURE_DIRECTORY,'lesionfreq_trainedlesions_new_'+blocking_train_txt+blocking_test_txt+str(args.train_lesion_freq)), args, True, False, whichTrialType, True)


def get_summarystats(data, axis, method='std'):
    """Calculate mean and std (or sem) along specified axis"""
    mean = np.mean(data, axis=axis)
    if method=='std':
        err = np.std(data, axis=axis)
    elif method=='sem':
        err = np.std(data, axis=axis) / np.sqrt(data.shape[axis])
    return mean, err


def perf_vs_context_distance(args, device):
    """This function plots post-lesion performance as a function of context distance (distance between input and context median).
    - ugly but functional. This ugly list (rather than np matrix) method is because the number of context distance elements in each context is different."""

    frequencylist = [0.0, 0.1]  # training frequencies of different networks to consider
    #frequencylist = [0.0, 0.1, 0.2, 0.3, 0.4]  # training frequencies of different networks to consider
    overall_lesioned_tests = []

    # file naming
    blcktxt = '_interleaved' if args.all_fullrange else '_temporalblocked'
    contexttxt = '_contextcued' if args.label_context=='true' else '_nocontextcued'
    range_txt = ''
    if args.which_context==0:
        range_txt = ''
    elif args.which_context==1:
        range_txt = '_fullrangeonly'
    elif args.which_context==2:
        range_txt = '_lowrangeonly'
    elif args.which_context==3:
        range_txt = '_highrangeonly'

    # generate theoretical predictions under local and global context policies
    numberdiffs, globalnumberdiffs, perf = theory.simulate_theoretical_policies()

    print('Retrieving lesion data for each model meeting criteria...')
    fig, ax = plt.subplots(1,len(frequencylist), figsize=(2.7*len(frequencylist),3.5))
    for j,train_lesion_frequency in enumerate(frequencylist):

        ax[j].set_ylabel(r'p(correct | $\epsilon_{train}$ =' + str(train_lesion_frequency)+')')
        args.train_lesion_freq = train_lesion_frequency
        allmodels = anh.get_model_names(args)

        # allocate some space
        data = [[] for i in range(len(allmodels))]
        global_meanperf = []
        global_uniquediffs = []
        full_context_numberdiffs, low_context_numberdiffs, high_context_numberdiffs = [[] for i in range(3)]
        full_context_perf, low_context_perf, high_context_perf = [[] for i in range(3)]

        # find all model ids that fit our requirements
        for ind, m in enumerate(allmodels):
            args.model_id = anh.get_id_from_name(m)
            testParams = anh.setup_test_parameters(args, device)
            block_ttsplit_text = '_blockttsplit' if args.block_int_ttsplit else ''
            basefilename = const.LESIONS_DIRECTORY + 'lesiontests'+m[:-4] + block_ttsplit_text
            filename = basefilename+'.npy'

            # perform or load the lesion tests
            lesiondata, regulartestdata = anh.perform_lesion_tests(args, testParams, basefilename)
            data[ind] = lesiondata["bigdict_lesionperf"]
            gp, cp, gd, cd = anh.lesion_perf_by_numerosity(data[ind])
            global_meanperf.append(gp)
            global_uniquediffs.append(gd)
            full_context_perf.append(cp[0])
            low_context_perf.append(cp[1])
            high_context_perf.append(cp[2])
            full_context_numberdiffs.append(cd[0])
            low_context_numberdiffs.append(cd[1])
            high_context_numberdiffs.append(cd[2])

        # mean over models
        global_meanperf = np.array(global_meanperf)
        full_context_perf = np.array(full_context_perf)
        low_context_perf = np.array(low_context_perf)
        high_context_perf = np.array(high_context_perf)
        global_uniquediffs = np.array(global_uniquediffs)
        full_context_numberdiffs = np.array(full_context_numberdiffs)
        low_context_numberdiffs = np.array(low_context_numberdiffs)
        high_context_numberdiffs = np.array(high_context_numberdiffs)

        global_meanperf_mean, global_meanperf_sem = get_summarystats(global_meanperf, 0)
        full_context_perf_mean, full_context_perf_sem = get_summarystats(full_context_perf, 0)
        low_context_perf_mean, low_context_perf_sem = get_summarystats(low_context_perf, 0)
        high_context_perf_mean, high_context_perf_sem = get_summarystats(high_context_perf, 0)

        global_uniquediffs = np.mean(global_uniquediffs, axis=0)
        full_context_numberdiffs = np.mean(full_context_numberdiffs, axis=0)
        low_context_numberdiffs = np.mean(low_context_numberdiffs, axis=0)
        high_context_numberdiffs = np.mean(high_context_numberdiffs, axis=0)

        # plot model predictions under local or global predictions
        handles = theory.plot_theoretical_predictions(ax[j], numberdiffs, globalnumberdiffs, perf, j)

        # context-specific performance i.e. how did performance change with dist. to mean in each context
        xnumbers =  [full_context_numberdiffs, low_context_numberdiffs, high_context_numberdiffs]
        means = [full_context_perf_mean, low_context_perf_mean, high_context_perf_mean]
        stds = [full_context_perf_sem, low_context_perf_sem, high_context_perf_sem]

        for i in range(len(xnumbers)):
            shadeplot(ax[j], xnumbers[i], means[i], stds[i], const.CONTEXT_COLOURS[i])
            h = ax[j].errorbar(xnumbers[i], means[i], stds[i], color=const.CONTEXT_COLOURS[i], fmt='o', markersize=5, markeredgecolor='black', ecolor='black')
            handles.append(h)

        ax[j].set_xlabel('context distance')
        ax[j].set_xlim([-0.5, 8])
        ax[j].set_ylim([0.47, 1.03])
        ax[j].set_xticks([0,2,4,6,8])

    ax[j].legend((handles[0], handles[-1]),('prediction','RNN'))
    blocking_train_txt = 'interleavedtrain_' if args.all_fullrange else 'blockedtrain_'
    if args.block_int_ttsplit:
        blocking_test_txt = 'blockedtest_' if args.all_fullrange else 'interleavedtest_'
    else:
        blocking_test_txt = 'interleavedtest_' if args.all_fullrange else 'blockedtest_'
    plt.savefig(os.path.join(const.FIGURE_DIRECTORY, 'perf_v_distToContextMean_postlesion_'+blocking_train_txt + blocking_test_txt + str(args.train_lesion_freq)+'.pdf'), bbox_inches='tight')


def visualise_recurrent_state(MDS_dict):
    """This function does MDS on a subset of test trials to see if the latext recurrent state in the network separates by context.
    - it looks highly structured, but not particularly context separated.
    - the MDS takes ages to compute since over many samples.
    - not currently used for anything.
    """

    states = np.reshape(MDS_dict["drift"]["temporal_activation_drift"], (MDS_dict["drift"]["temporal_activation_drift"].shape[0]*MDS_dict["drift"]["temporal_activation_drift"].shape[1], MDS_dict["drift"]["temporal_activation_drift"].shape[2]))
    context = np.reshape(MDS_dict["drift"]["temporal_context"], (MDS_dict["drift"]["temporal_context"].shape[0]*MDS_dict["drift"]["temporal_context"].shape[1], ))

    plt.figure()
    ax = plt.gca()
    plt.plot(context)
    plt.xlabel('trials (time)')
    plt.ylabel('context')
    ax.set_yticks([1,2,3])
    ax.set_yticklabels(['full','low','high'])

    # select a subset of trials to make MDS compute faster
    mini_context = context[10000:16700]
    mini_states = states[10000:16700, :]

    plt.figure()
    ax = plt.gca()
    plt.plot(mini_context)
    plt.xlabel('trials (time)')
    plt.ylabel('context')
    ax.set_yticks([1,2,3])
    ax.set_yticklabels(['full','low','high'])

    # Do MDS on the latent states
    pairwise_data = pairwise_distances(mini_states, metric='correlation')
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    MDS_act, evals = anh.cmdscale(pairwise_data)

    MDS = MDS_act[:,:3]
    fig,ax = plt.subplots(1,3, figsize=(18,5))
    for j in range(3):  # 3 MDS dimensions
        if j==0:
            dimA = 0
            dimB = 1
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 2')
        elif j==1:
            dimA = 0
            dimB = 2
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 3')
        elif j==2:
            dimA = 1
            dimB = 2
            ax[j].set_xlabel('dim 2')
            ax[j].set_ylabel('dim 3')

        for i in range((MDS.shape[0])):
            ax[j].scatter(MDS[i, dimA], MDS[i, dimB], color=const.CONTEXT_COLOURS[int(mini_context[i])-1], edgecolor=const.CONTEXT_COLOURS[int(mini_context[i])-1], s=8, linewidths=1)
        ax[j].axis('equal')
    plt.savefig('figures/drift_state.pdf',bbox_inches='tight')


def view_postlesion(args, device):
    """View the MDS of activations immediately post lesion for a particular model.
    - model id should be mentioned in args for this functions
    - these look essentially identical to when not lesioning at test."""

    overall_lesioned_tests = []

    # file naming
    blcktxt = '_interleaved' if args.all_fullrange else '_temporalblocked'
    contexttxt = '_contextcued' if args.label_context=='true' else '_nocontextcued'
    range_txt = ''
    if args.which_context==0:
        range_txt = ''
    elif args.which_context==1:
        range_txt = '_fullrangeonly'
    elif args.which_context==2:
        range_txt = '_lowrangeonly'
    elif args.which_context==3:
        range_txt = '_highrangeonly'

    print('Retrieving lesion data for each model meeting criteria...')
    allmodels = anh.get_model_names(args)
    m = [model for model in allmodels if 'id'+str(args.model_id) in model]
    # allocate some space
    data = [[] for i in range(len(allmodels))]
    global_meanperf = []
    global_uniquediffs = []
    full_context_numberdiffs, low_context_numberdiffs, high_context_numberdiffs = [[] for i in range(3)]
    full_context_perf, low_context_perf, high_context_perf = [[] for i in range(3)]

    testParams = anh.setup_test_parameters(args, device)
    basefilename = const.LESIONS_DIRECTORY + 'lesiontests'+m[0][:-4]
    filename = basefilename+'.npy'

    # perform or load the lesion tests
    lesiondata, regulartestdata = anh.perform_lesion_tests(args, testParams, basefilename)
    data = lesiondata["bigdict_lesionperf"]
    count = 0

    sorted_activations = [[] for i in range(const.HIGHR_SPAN + const.LOWR_SPAN + const.FULLR_SPAN)]
    allnumbers = [range(const.FULLR_LLIM, const.FULLR_ULIM+1), range(const.LOWR_LLIM, const.LOWR_ULIM+1), range(const.HIGHR_LLIM, const.HIGHR_ULIM+1)]
    allnumbers = [item for sublist in allnumbers for item in sublist]
    contextlabel = [[1 for i in range(const.FULLR_SPAN)], [2 for i in range(const.LOWR_SPAN)], [3 for i in range(const.HIGHR_SPAN)]]
    contextlabel = [item for sublist in contextlabel for item in sublist]
    keys = [str(contextlabel[i])+'-'+str(allnumbers[i]) for i in range(len(allnumbers)) ]

    # get the post-lesion activations ready for averaging over for each number and context
    for batch_idx in range(data.shape[0]):
        for trial in range(data[batch_idx].shape[0]):
            activations = data[batch_idx][trial]["post_lesion_activations"].numpy()
            number = data[batch_idx][trial]["assess_number"].numpy()
            context = data[batch_idx][trial]["underlying_context"].numpy()

            # find where to save these activations
            key = str(context)+'-'+str(number)
            index = keys.index(key)
            sorted_activations[index].append(activations)

    # take the mean activations over trials
    mean_activations = np.zeros((const.HIGHR_SPAN + const.LOWR_SPAN + const.FULLR_SPAN,200))
    for i in range(len(sorted_activations)):
        tmp = np.asarray((sorted_activations[i]))
        mean_activations[i] = np.mean(tmp, axis=0)

    # Perform MDS on averaged activations for the post-lesion trial data
    pairwise_data = pairwise_distances(mean_activations, metric='correlation') # using correlation distance
    np.fill_diagonal(np.asarray(pairwise_data), 0)
    plt.figure()
    plt.imshow(pairwise_data)
    plt.savefig('figures/RMD_postlesion.pdf', bbox_inches='tight')
    MDS_act, evals = anh.cmdscale(pairwise_data)

    numberlabel = allnumbers
    fig, ax = plt.subplots(1,3,figsize=(15,3.5))

    for j in range(3):  # 3 MDS dimensions
        if j==0:
            dimA = 0
            dimB = 1
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 2')
        elif j==1:
            dimA = 0
            dimB = 2
            ax[j].set_xlabel('dim 1')
            ax[j].set_ylabel('dim 3')
        elif j==2:
            dimA = 1
            dimB = 2
            ax[j].set_xlabel('dim 2')
            ax[j].set_ylabel('dim 3')

        ax[j].set_title('context')
        contextA = range(const.FULLR_SPAN)
        contextB = range(const.FULLR_SPAN,const.FULLR_SPAN+const.LOWR_SPAN)
        contextC = range(const.FULLR_SPAN+const.LOWR_SPAN, const.FULLR_SPAN+const.LOWR_SPAN+const.HIGHR_SPAN)

        ax[j].plot(MDS_act[contextA, dimA], MDS_act[contextA, dimB], color=const.CONTEXT_COLOURS[0])
        ax[j].plot(MDS_act[contextB, dimA], MDS_act[contextB, dimB], color=const.CONTEXT_COLOURS[1])
        ax[j].plot(MDS_act[contextC, dimA], MDS_act[contextC, dimB], color=const.CONTEXT_COLOURS[2])

        for i in range((MDS_act.shape[0])):
            ax[j].scatter(MDS_act[i, dimA], MDS_act[i, dimB], color=const.CONTEXT_COLOURS[int(contextlabel[i])-1], edgecolor=const.CONTEXT_COLOURS[int(contextlabel[i])-1], s=80, linewidths=2)
            ax[j].text(MDS_act[i, dimA], MDS_act[i, dimB], str(24+int(numberlabel[i])), color='white', size=6.5, horizontalalignment='center', verticalalignment='center')

        ax[j].axis('equal')
        axislimits = (-0.8, 0.8)
        ax[j].set(xlim=axislimits, ylim=axislimits)

    n = save_figure(os.path.join(const.FIGURE_DIRECTORY,'MDS_postlesion_'), args, True, False, 'compare', True)


def generate_plots(MDS_dict, args):
    """ This function just plots stuff and saves the generated figures."""
    saveFig = True
    plot_diff_code = False    # do we want to plot the difference code or the average A activations
    labelNumerosity = True    # numerosity vs outcome labels
    trialTypes = ['compare']  # ['compare', 'filler'] if you want to also see the activations for filler numbers

    for whichTrialType in trialTypes:

        # Label activations by mean number A numerosity
        activation_rdms(MDS_dict, args, plot_diff_code, whichTrialType)  # activations RSA
        axislimits = (-0.8, 0.8)
        plot_3mds_mean(MDS_dict, args, labelNumerosity, plot_diff_code, whichTrialType, saveFig, 80, axislimits) # mean MDS of our hidden activations (averaged across number B)

        #plot_3mds(MDS_dict, args, whichTrialType)      # the full MDS cloud, coloured by different labels

        # Label activations by the difference code numerosity
        #plot_diff_code = True
        #activation_rdms(MDS_dict, args, plot_diff_code, whichTrialType)  # activations RSA
        #plot_3mds_mean(MDS_dict, args, labelNumerosity, plot_diff_code, whichTrialType)

        # Plot checks on the training data sequencing
        #n = plt.hist(activations)   # They are quite sparse activations (but we dont really care that much)
        #view_training_sequence(MDS_dict, args)  # Plot the context sequencing in the training set through time
        #instance_counter(MDS_dict, args)  # Check how many samples we have of each unique input (should be context-ordered)

        # MDS with output labels (true/false labels)
        #labelNumerosity = False
        #plot_3mds(MDS_dict, args, labelNumerosity, plot_diff_code)

        # 3D Animations
        #animate_3d_mds(MDS_dict, args, plot_diff_code)  # plot a 3D version of the MDS constructions
