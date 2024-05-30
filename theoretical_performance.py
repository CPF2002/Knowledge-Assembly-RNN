"""
Script for calculating and plotting optimal performance under two different policies post-lesion:
 1. a local policy that uses the local context mean when responding whether number A>B
 2. a global policy that uses the global number mean across all contexts when responding whether A>B

Sheahan, H.*, Luyckx, F.*, Nelli, S., Taupe, C., & Summerfield, C. (2021). Neural
 state space alignment for magnitude generalisation in humans and recurrent networks.
 Neuron (in press)

 Author: Hannah Sheahan, sheahan.hannah@gmail.com
 Date: 07/04/2020
 Notes: N/A
 Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import analysis_helpers as anh
import constants as const
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------- #

def simulate_theoretical_policies():
    """This function calculates the theoretical performance of an agent,
    when making relative magnitude decisions seeing just the current number and using either local or global context info.
    - distribution of numbers in each context are the same as for the human and network relative magnitude task.
    """
    print('Simulating theoretical agent performance...')
    # Define the ranges of primary targets displayed in each context
    localxranges = [[const.FULLR_LLIM,const.FULLR_ULIM], [const.LOWR_LLIM,const.LOWR_ULIM], [const.HIGHR_LLIM,const.HIGHR_ULIM]]
    globalxrange = [i for contextrange in localxranges for i in range(contextrange[0], contextrange[1]+1)]
    globalmean = stats.mean(globalxrange) # should be 8.5

    # record performance as a function of distance between current number and context (or global) mean
    policies = ['global','local']
    numberdiffs = {"global":dict(list(enumerate([[],[],[]]))), "local":dict(list(enumerate([[],[],[]])))}
    globalnumberdiffs = {"global":dict(list(enumerate([[],[],[]]))), "local":dict(list(enumerate([[],[],[]])))}
    perf = {"global":dict(list(enumerate([[],[],[]]))), "local":dict(list(enumerate([[],[],[]])))}

    for whichpolicy, policy in enumerate(policies):
        print('Testing policy: '+ policy)
        Ptotal = 0
        for whichrange in range(len(localxranges)):
            xmin = localxranges[whichrange][0]
            xmax = localxranges[whichrange][1]

            # Possible values for xA and xB
            xvalues = list(range(xmin, xmax+1))
            if policy == 'local':
                xmean = stats.mean(xvalues)
            else:
                xmean = globalmean
            Na = len(xvalues)
            Nb = Na-1         # xA never equals xB
            P_a = 1/Na        # uniform distribution for sampling a from xA

            Pcorrect = 0
            for a in xvalues:
                P_agreaterB = (a-xmin)/Nb
                if (a<=xmean):
                    Pcorrect_a = (1 - P_agreaterB)
                else:
                    Pcorrect_a = P_agreaterB
                Pcorrect += P_a*Pcorrect_a

                # distance of current number a to mean (local or global)
                numberdiffs[policy][whichrange].append(abs(a-stats.mean(xvalues)))
                globalnumberdiffs[policy][whichrange].append(abs(a-globalmean))
                perf[policy][whichrange].append(Pcorrect_a)

            print(('{:.2f}% correct for range {}, under policy '+policy).format(Pcorrect*100, whichrange))
            Ptotal += Pcorrect

        Ptotal /= const.NCONTEXTS
        print('Mean performance across all 3 ranges with ' + policy + ' policy: {:.2f}%'.format(Ptotal*100))
        print('\n')

    return numberdiffs, globalnumberdiffs, perf

# ---------------------------------------------------------------------------- #

def plot_theoretical_predictions(ax, numberdiffs, globalnumberdiffs, perf, whichpolicy):
    """ This function plots performance under each policy as a function of numerical each distance to context median (context distance).
    - plots just the policy specied in 'whichpolicy' i.e. 0=global, 1=local. """

    localxranges = [[const.FULLR_LLIM,const.FULLR_ULIM], [const.LOWR_LLIM,const.LOWR_ULIM], [const.HIGHR_LLIM,const.HIGHR_ULIM]]
    linestyles = ['solid', 'dotted', 'dashed']
    handles = []
    policies = ['global', 'local', 'local', 'local', 'local']  # corresponds to same plots as each lesion frequency: 0.0, 0.1, 0.2, 0.3, 0.4
    policy = policies[whichpolicy]

    for whichrange in range(len(localxranges)):
        context_perf, context_numberdiffs = anh.performance_mean(numberdiffs[policy][whichrange], perf[policy][whichrange])
        h, = ax.plot(context_numberdiffs, context_perf, color=const.CONTEXT_COLOURS[whichrange])
        handles.append(h)
        ax.set_ylim([0.27, 1.03])

    return handles

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    simulate_theoretical_policies()
