#!/usr/bin/env python
# coding: utf-8

# In[3]:


from collections import OrderedDict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy import stats
def lightflash(lam=100, t1=0.8, t2=2.2):
    # submit lam parameter in photons/millisecond;  t1 and t2 parameters in milliseconds
    t = t1 + np.random.exponential(scale=1/lam)
    photon_times = []
    while t < t2:
        photon_times.append(t)
        t += np.random.exponential(scale=1/lam)
    return photon_times

def plotHSPsimulation(lam=100, t1=0.8, t2=2.2, f1=0.8, f2=2.2, s1=1.0, s2=2.0, alpha=0.06):
    photon_times = lightflash(lam=lam, t1=t1, t2=t2)
    photon_times_abbreviated = cut_list(photon_times, f1, f2)
    plt.figure(figsize=(8, 4), dpi=80)
    heads = np.ones(shape=len(photon_times_abbreviated))
    plt.stem(photon_times_abbreviated, heads, linefmt="#ccccff",
             markerfmt=" ", basefmt="#ccccff")
    plt.title(
        "Photons stream $[f_1, f_2]=[{f_1},{f_2}]$".format(f_1=f1, f_2=f2))
    plt.ylabel("Amplitude")
    plt.xlabel("Time(msecs)")
    plt.xlim((t1, t2))
    plt.ylim((0, 2))
    plt.show()

    photon_times_shutter = cut_list(photon_times_abbreviated, s1, s2)
    plt.figure(figsize=(8, 4), dpi=80)
    heads = np.ones(shape=len(photon_times_shutter))
    plt.stem(photon_times_shutter, heads, linefmt="#9999ff",
             markerfmt=" ", basefmt="#9999ff")
    plt.title("Photons through Shutter during $[s_1, s_2]=[{s_1},{s_2}]$".format(
        s_1=s1, s_2=s2))
    plt.ylabel("Amplitude")
    plt.xlabel("Time(msecs)")
    # plt.xlim((s1, s2))
    plt.xlim((t1, t2))
    plt.ylim((0, 2))
    plt.show()

    photon_times_absorbed = get_detected_photon_times(
        photon_times_shutter, alpha=alpha)
    plt.figure(figsize=(8, 4), dpi=80)
    heads = np.ones(shape=len(photon_times_absorbed))
    plt.stem(photon_times_absorbed, heads, linefmt="#5555ff",
             markerfmt=" ", basefmt="#5555ff")
    plt.title("Photons detected by rods")
    plt.ylabel("Amplitude")
    plt.xlabel("Time(msecs)")
    # plt.xlim((s1, s2))
    plt.xlim((t1, t2))
    plt.ylim((0, 2))
    plt.show()
    
def get_detected_photon_times(photon_times, alpha):
    # given an array of photon times and an alpha to describe probability of detection, this should return a subset of those times reflecting alpha
    detected_photon_times = []
    for pt in photon_times:
        if np.random.uniform() <= alpha:
            detected_photon_times.append(pt)
    return detected_photon_times

def cut_list(lis, min, max):
    new_lis = []
    for i in range(len(lis)):
        if lis[i] >= min and lis[i] <= max:
            new_lis.append(lis[i])
    return new_lis

def findfit():
    alphas = np.linspace(0.01, 1.00, 101)
    Ks = np.linspace(1, 12, num=13)
    e_alphas = [0.02, 0.13]
    e_Ks = [2, 12]
    HSP_data = [[24.1, 37.6, 58.6, 91.0, 141.9, 221.3],
                [0.000,  0.040, 0.180, 0.540,  0.940, 1.000]]
    min_total_mse = None
    optimal_alpha = None
    optimal_K = None

    for alpha in alphas:
        for K in Ks:
            # print("on pair " + str(alpha) + "," + str(K))
            HSP_prob = []
            for I in HSP_data[0]:
                HSP_prob.append(probseeing(I, alpha=alpha, K=K))
            total_mse = mse(HSP_prob, HSP_data[1])
            prob = []
            for I in np.linspace(0.01, 100, 100):
                prob.append(probseeing(I, alpha=alpha, K=K))
            for i in range(len(e_alphas)):
                e_prob = []
                for I in np.linspace(0.01, 100, 100):
                    e_prob.append(probseeing(I, alpha=e_alphas[i], K=e_Ks[i]))
                total_mse += mse(prob, e_prob)
            if (min_total_mse == None) or (total_mse < min_total_mse):
                min_total_mse = total_mse
                optimal_alpha = alpha
                optimal_K = K
    return optimal_alpha, optimal_K

def probseeing(I, alpha=0.06, K=6):
    return 1-stats.poisson.cdf(k=(K-1), mu=alpha*I)

def mse(prob, e_prob):
    prob = np.array(prob)
    e_prob = np.array(e_prob)
    return (1/len(prob))*np.sum((prob-e_prob)**2)    

def plotfit(alpha=3, K=3, show_fit=True, xlimit=(0.01, 100), show_ss=False):
    plt.figure(figsize=(8, 5), dpi=80)

    # first plotting our expiremental results
    e_alpha = [0.02, 0.13]
    e_K = [2, 12]
    HSP_data = [[24.1, 37.6, 58.6, 91.0, 141.9, 221.3],
                [0.000,  0.040, 0.180, 0.540,  0.940, 1.000]]

    for i in range(len(e_alpha)):
        e_p = []
        for I in np.linspace(xlimit[0], xlimit[1], 10000):
            e_p.append(probseeing(I, alpha=e_alpha[i], K=e_K[i]))
        plt.plot(np.linspace(xlimit[0], xlimit[1], 10000), e_p,
                 c=purples[6], label="experimental data")

    # plotting the HSP data:
    if show_ss:
        plt.scatter(HSP_data[0], HSP_data[1], c=purples[4], label="HSP SS")

    # then plotting the fitted K values
    if show_fit:
        if type(alpha) != list and type(alpha) != np.ndarray:
            alpha = [alpha]
            K = [K]
        assert len(K) == len(alpha)
        for i in range(len(alpha)):
            p = []
            for I in np.linspace(xlimit[0], xlimit[1], 10000):
                p.append(probseeing(I, alpha=alpha[i], K=K[i]))
            plt.plot(np.linspace(xlimit[0], xlimit[1], 10000),
                     p, c=purples[1], label="fits")

    plt.title("Some Fits: Probability of Detection of a Flash w.r.t. Intensity")
    plt.ylabel("$p$(Detection|Flash)")
    plt.xlabel("Intensity")
    plt.xlim(xlimit[0], xlimit[1])
    plt.xscale('log')
    # Subsequent 3 lines mostly taken from: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

