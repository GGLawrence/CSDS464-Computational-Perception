#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import sympy
import librosa
import librosa.display
from copy import deepcopy,copy
from scipy.io.wavfile import read
from IPython.display import display,Latex
from pylab import *
from scipy.io import wavfile
import wave

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def timetoindex(t,fs):
    return t*fs

def sinewave(t,f,d):
    return np.sin(2*np.pi*f*(t+d))

def pltsinwave(t1,t2,fs,f=1.0,d=0.0):
    i1=int(timetoindex(t1,fs))
    i2=int(timetoindex(t2,fs))
    time=np.linspace(i1,i2,i2-i1+1,endpoint=True)
    time=time/fs
    return np.vectorize(sinewave)(time,f,d)


def gabor(t,sigma,f,phi):#calcuate g without normalizing constant
    return math.exp(-pow(t,2)/(2*pow(sigma,2)))*math.cos(2*math.pi*f*(t)+phi)

def gabore(t,sigma,f):
    return math.exp(-pow(t,2)/(2*pow(sigma,2)))*math.cos(2*math.pi*f*(t))

def gaboro(t,sigma,f):
    return -math.exp(-pow(t,2)/(2*pow(sigma,2)))*math.sin(2*math.pi*f*(t))

def gabor_norm(fs,sigma=1.0,f=1.0,phi=0.0):#calcuate normalizing constant
    sum_g=[]
    for index in range(-fs,fs):
        t=(index)/fs
        sum_g.append(gabor(t,sigma,f,phi))
    return np.linalg.norm(sum_g)

def gabore_norm(fs,sigma=1.0,f=1.0):
    sum_g=[]
    for index in range(-10*fs,10*fs):
        t=(index)/fs
        sum_g.append(gabore(t,sigma,f))
    return np.linalg.norm(sum_g)

def gaboro_norm(fs,sigma=1.0,f=1.0):
    sum_g=[]
    for index in range(-10*fs,10*fs):
        t=(index)/fs
        sum_g.append(gaboro(t,sigma,f))
    return np.linalg.norm(sum_g)


def gabor_a(t,fs,sigma=1.0,f=1.0,phi=0.0):
    return gabor(t,sigma,f,phi)/ gabor_norm(fs,sigma,f,phi)

def gabore_a(t,fs,sigma=1.0,f=1.0):
    return gabore(t,sigma,f)/ gabore_norm(fs,sigma,f)

def gaboro_a(t,fs,sigma=1.0,f=1.0):
    return gaboro(t,sigma,f)/ gaboro_norm(fs,sigma,f)

n=4
phi=0
def calculate_gammatone(t,b,f):
    return pow(t,n-1) * np.exp(-2*np.pi*b*t)*np.cos(2*np.pi*f*t+phi)

def calculate_b(f):
    return 1.019*24.7*(4.37*f/1000+1)

def calculate_a(fs,b,f):
    sum_g=[]
    for index in range(0,fs):
        t=(index)/fs
        sum_g.append(calculate_gammatone(t,b,f))
    return np.linalg.norm(sum_g)

def gammatone(t,fs,f):
    b=calculate_b(f)
    return calculate_gammatone(t,b,f)/calculate_a(fs,b,f)




def localmaxima(s):
    localmax=[]
    index=[]
    for i in range(1,len(s)-1):
        if s[i-1]<s[i] and s[i]>s[i+1]:
            localmax.append(s[i])
            index.append(i)
    return index,localmax


# In[5]:


def crossings(value,threshold, direction):
    index=[]
    negpos_flag=0
    posneg_flag=0
    for i in range(0,len(value)-1):
        if value[i]==threshold:
            index.append(i)
        if value[i]<threshold and value[i+1]>=threshold:
            negpos_flag=1
            if direction == "negpos" and (i not in value):
                index.append(i+1)
        
        if value[i]>=threshold and value[i+1]<threshold:
            posneg_flag=1
            if direction == "posneg" and (i not in value):
                index.append(i+1)
        if direction == "both" and negpos_flag==1 and posneg_flag==1:
            index.append(i+1)
            negpos_flag=0
            posneg_flag=0
    return index




def envelope(y,nblocks):
    if nblocks==0:
        nblocks=len(y)/10
    ylower=[]
    yupper=[]
    blockindices=[]
    block_index=0
    block_length=int(len(y)/nblocks)
    while(block_index<nblocks-1):
        current_index=block_index*block_length
        end_index=current_index+block_length
        current_lower=y[current_index]
        current_upper=y[current_index]
        blockindices.append(current_index)
        while(current_index<end_index):
            if current_lower>y[current_index]:
                current_lower=y[current_index]
            if current_upper<y[current_index]:
                current_upper=y[current_index]
            current_index+=1
        ylower.append(current_lower)
        yupper.append(current_upper)
        block_index+=1
    current_index=block_index*block_length
    blockindices.append(current_index)
    end_index=len(y)
    current_lower=y[current_index]
    current_upper=y[current_index]
    while(current_index<end_index):
        if current_lower>y[current_index]:
            current_lower=y[current_index]
        if current_upper<y[current_index]:
            current_upper=y[current_index]
        current_index+=1
    ylower.append(current_lower)
    yupper.append(current_upper)
    
    return ylower,yupper,blockindices




frame_size=1024
hop_length=512
filename="speech.wav"
speech,sr=librosa.load(filename)
def amplitude_envelope(signal,frame_size,hop_length):
    envelope=[]
    for i in range(0,len(signal),hop_length):
        current_frame_envelope=max(signal[i:i+frame_size])
        envelope.append(current_frame_envelope)
        
    return np.array(envelope)

def fancy_envelope(signal,frame_size,hop_length):
    return np.array([max(signal[i:i+frame_size])for i in range(0,signal.size,hop_length)])

def envelope_min(signal,frame_size,hop_length):
    envelope=[]
    for i in range(0,len(signal),hop_length):
        current_frame_envelope=min(signal[i:i+frame_size])
        envelope.append(current_frame_envelope)
        
    return np.array(envelope)

def fancy_envelope_min(signal,frame_size,hop_length):
    return np.array([min(signal[i:i+frame_size])for i in range(0,signal.size,hop_length)])

fancy_ae_speech=fancy_envelope(speech,frame_size,hop_length)


# In[8]:


ae_speech=amplitude_envelope(speech,frame_size,hop_length)


# In[9]:



# In[ ]:




