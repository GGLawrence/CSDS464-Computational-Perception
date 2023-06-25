#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import math
import scipy
import IPython

def read_wav(filepath):
    sr, data = wavfile.read(filepath)
    return sr, data

def harmonic(t, f:int=1, alist:list=[1], phase_list:list=[0]):
    value = 0
    for index in range(len(alist)):
        value += alist[index]*math.cos(f*(index+1)*t+phase_list[index])
    return value

def cosine(t, f:list=[1], alist:list=[1], phase_list:list=[0]):
    value = 0
    for index in range(len(alist)):
        value += alist[index]*math.cos(f[index]*t+phase_list[index])
    return value

def show_harmonics(t, g, f, n=None, alist:list=[1], phase_list:list=[0], title:str="--"):
    y = np.array([g(t=i, f=f, alist=alist,  phase_list=phase_list) for i in t])
    if n is not None and len(y) == len(n):
        y += n
    plt.figure()
    if isinstance(f, int) or isinstance(f, float):
        plt.stem([f*(i+1) for i in range(len(alist))],alist, use_line_collection=True)
    else:
        plt.stem(f, alist, use_line_collection=True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.figure()
    plt.grid()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def autocorr(x, normalize:bool=True):
    pxx = np.zeros(len(x)*2-1)
    norm_sqr = np.linalg.norm(x)**2
    for n in range(-len(x)+1, len(x)):
        for k in range(len(x)):
            if len(x) > k-n >= 0:
                pxx[n+len(x)-1] += x[k-n]*x[k]
        if normalize:
            pxx[n+len(x)-1] /= norm_sqr
    return pxx

