#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import sympy
import librosa
import librosa.display
from copy import deepcopy,copy
from scipy.io.wavfile import read
import scipy.io.wavfile
from IPython.display import display,Latex
from pylab import *
from scipy.io import wavfile
import wave
import random
def gensignal(t,tau,T,g='gammatone',fs=100,f=1.0,phi=0.0,sigma=1
              ,n=4,tscale=1,tunits='secs',plot=True):
    tag = t
    if g=='sin':
        x=np.linspace(0,T,1000,endpoint=True)
        y=np.vectorize(sinewave)(x,f,phi)
        
    if g=='gammatone':
        x=np.linspace(0,T,400,endpoint=True)
        y=np.vectorize(gammatone)(x,fs,f)
        
    if g=='step':
        x,y=step(0,T,1000)
    if g=='delta':
        x,y=delta(0,T,1000,fs)    
    
    x=x+tau
    for i in range(int(tau*1000)):
        y=np.insert(y,0,0)
        x=np.insert(x,0,1/1000*(int(tau*1000)-i-1))
    l=tau+T
    for i in range(int((t-tau-T)*1000)):
        l+=1/1000
        y=np.insert(y,len(y),0)
        x=np.insert(x,len(x),l)
    if plot==True:
        if tunits == 'secs':
            plt.xlabel('Time(secs)')
        else:
            plt.xlabel('Time(msecs)')
            x*=tscale
       
    a=len(x)
    index=np.linspace(0,a,timetoindex(t,fs),endpoint=False)
    x_plot=[]
    y_plot=[]
    for i in index:
        x_plot.append(x[int(i)])
        y_plot.append(y[int(i)])
    if plot==True:
        if tunits == "secs":
            for i in range(0,len(y_plot)):
                plt.scatter(x_plot[i],y_plot[i],color='green',marker='o')
                plt.plot([x_plot[i],x_plot[i]],
                         [0,y_plot[i]],color='r',linestyle='-')
        if tunits == "msecs":           
            for i in range(0,len(y_plot)):
                plt.scatter(x_plot[i]*1000,y_plot[i],color='green',marker='o')
                plt.plot([x_plot[i]*1000,x_plot[i]*1000],
                         [0,y_plot[i]],color='r',linestyle='-')
        plt.grid()
        plt.show()
    else:
        return x_plot,y_plot
        
def step(t1,t2,fs):
    x=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=False)
    y=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=False)
    x,y=unit(x,y)
    return x,y

def unit(x,y):
    for i in range(0,len(x)):
        if x[i]>=0:
            y[i]=1
        else:
            y[i]=0
    return x,y

def delta(t1,t2,fs,f):
    x=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=False)
    y=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=False)
    x,y=ddf(x,y,t)
    return x,y

def ddf(x,y,fs=1):
    for i in range(0,len(x)):
        if x[i]==0:
            y[i]=1
        else:
            y[i]=0
    return x,y

def gammatone(t,fs,f):
    b=calculate_b(f)
    return calculate_gammatone(t,b,f)/calculate_a(fs,b,f)

def calculate_gammatone(t,b,f,n=4,phi=0):
    return pow(t,n-1) * np.exp(-2*np.pi*b*t)*np.cos(2*np.pi*f*t+phi)

def calculate_b(f):
    return 1.019*24.7*(4.37*f/1000+1)

def calculate_a(fs,b,f):
    sum_g=[]
    for index in range(0,fs):
        t=(index)/fs
        sum_g.append(calculate_gammatone(t,b,f))
    return np.linalg.norm(sum_g)

def matched_filter_fig(fs = 2000, t1 = 0.1, f1 = 250, sig = 0.1, tao = 0.05):
    x,y=gensignal(t=0.1,g='gammatone',tau=0,T=0.1,fs=2000,f=250,tscale=1000,tunits='msec',plot=False)
    y=np.array(y)
    x=np.array(x)
    h=y.copy()
    plt.figure(figsize=(10,2))
    plt.plot(x*1000,y * 5)
    plt.grid()
    plt.title('Signal')
    plt.show()

    noise=np.random.uniform(-0.5,0.5,len(y))
    y+=noise
    plt.figure(figsize=(10,2))
    plt.plot(x*1000,y*3)
    plt.grid()
    plt.title('with noise sigma = 0.5')
    plt.show()

    y_convolve=convolve(y,h)
    plt.figure(figsize=(10,2))
    plt.plot(x*1000,y_convolve*6)
    plt.title('matched filter output')
    plt.grid()
    plt.show()
    
def convolve(x,h=[1],h0=1):
    y=[]
    for i in range(len(x)):
        yn=0
        if h0==1:
            for j in range(0,i+1):
                if (i-j)>(len(h)-1):
                    hij=0
                else:
                    hij=h[i-j]
                yn+=x[j]*hij
            y.append(yn)
        else:
            for j in range(0,i+1):
                if abs(i-j)>(len(h)-1):
                    hij=0
                elif i-j<0:
                    hij=h[j-i]
                else:
                    hij=h[i-j]
                yn+=x[j]*hij
            y.append(yn)
    return np.array(y)    

def sinewave(t,f,phi):
    return np.sin(2*np.pi*f*t+phi) 
def timetoindex(t,fs):
    return int(t*fs)

