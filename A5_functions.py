#!/usr/bin/env python
# coding: utf-8

# In[42]:


#!/usr/bin/env python
# coding: utf-8

# In[63]:


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
import h5py
from matplotlib.patches import Ellipse,Circle
from scipy.fftpack import fft,ifft
import seaborn
from numpy.fft import fft
import time


def timetoindex(t,fs):
    return int(t*fs)

def sinewave(t,f,phi):
    return np.sin(2*np.pi*f*t+phi)

def pltsinwave(t1,t2,fs,f=1.0,phi=0.0):
    time=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=True)
    return np.vectorize(sinewave)(time,f,phi)

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

def gammatone(t,fs,f):
    b=calculate_b(f)
    return calculate_gammatone(t,b,f)/calculate_a(fs,b,f)
def gammatone_new(t,f,fs=100):
    b=calculate_b(f)
    return calculate_gammatone(t,b,f)/calculate_a(fs,b,f)

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

def plot_sampled_function(g='sin',fs=1,tlim=[0,2*np.pi],tscale=1,tunits="secs",f=1,phi=0,sigma=1,n=4):
    if g == 'sin':
        x=np.linspace(tlim[0],tlim[1],1000,endpoint=True)
        y=np.vectorize(sinewave)(x,f,phi)
        plt.title('%d Hz sine wave sampled at %d Hz'%(f,fs),size=10,color='black')
#     if g == 'gabor':
#         x=np.linspace(tlim[0],tlim[1],timetoindex((tlim[1]-tlim[0]),fs),endpoint=True)
#         y=np.vectorize(gabor_a)(y,fs,sigma,f,phi)
    if g == 'gammatone':
        x=np.linspace(tlim[0],tlim[1],400,endpoint=True)
        y=np.vectorize(gammatone)(x,fs,f)
        plt.title('%d Hz gammatone sampled at %d Hz' %(f,fs),size=10,color='black')
        
    a=len(x)
    index=np.linspace(0,a,timetoindex(tlim[1]-tlim[0],fs),endpoint=False)
    x_plot=[]
    y_plot=[]
    for i in index:
        x_plot.append(x[int(i)])
        y_plot.append(y[int(i)])
    if tunits == 'secs':
        plt.xlabel('time(secs)')
    else:
        plt.xlabel('time(msecs)')
        x_plot=np.array(x_plot)
        x_plot*=tscale
        x*=tscale
    plt.grid()
    plt.plot(x,y,color='b')
    for i in range(0,len(y_plot)):
        plt.scatter(x_plot[i],y_plot[i],color='r',marker='o')
        plt.plot([x_plot[i],x_plot[i]],[0,y_plot[i]],color='r',linestyle='-')
        plt.ylabel('amplitude')
    plt.show()

    
# def ddf(x,sig):
#     val = np.zeros_like(x)
#     val[(-(1/(2*sig))<=x) & (x<=(1/(2*sig)))] = 1
#     return val

def ddf(x,y,fs=1):
    for i in range(0,len(x)):
        if x[i]==0:
            y[i]=1
        else:
            y[i]=0
    return x,y

def delta(t1,t2,fs,f):
    x=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=False)
    y=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=False)
    x,y=ddf(x,y,t)
    return x,y
 
def unit(x,y):
    for i in range(0,len(x)):
        if x[i]>=0:
            y[i]=1
        else:
            y[i]=0
    return x,y

def step(t1,t2,fs):
    x=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=False)
    y=np.linspace(t1,t2,timetoindex(t2-t1,fs),endpoint=False)
    x,y=unit(x,y)
    return x,y
    
def gensignal(t,tau,T,g='gammatone',fs=100,f=1.0,phi=0.0,sigma=1,n=4,tscale=1,tunits='secs',plot=True):
    if g=='sin':
        x=np.linspace(0,T,10000,endpoint=True)
        y=np.vectorize(sinewave)(x,f,phi)
        
    if g=='gammatone':
        x=np.linspace(0,T,400,endpoint=True)
        y=np.vectorize(gammatone)(x,fs,f)
    if g=='gabor':
        x=np.linspace(-T,T,timetoindex(2*T,fs),endpoint=False)
        y=np.vectorize(gabor_a)(x,fs,sigma)
    if g=='step':
        x,y=step(0,T,1000)
    if g=='delta':
        x,y=delta(0,T,1000,fs)
    
    
    x=x+tau
    for i in range(int(tau*10000)):
        y=np.insert(y,0,0)
        x=np.insert(x,0,1/10000*(int(tau*10000)-i-1))
    l=tau+T
    for i in range(int((t-tau-T)*10000)):
        l+=1/10000
        y=np.insert(y,len(y),0)
        x=np.insert(x,len(x),l)
    if plot==True:
        if tunits == 'secs':
            plt.xlabel('time(secs)')
        else:
            plt.xlabel('time(msecs)')
            x*=tscale
    
    
    a=len(x)
    index=np.linspace(0,a,timetoindex(t,fs),endpoint=False)
    x_plot=[]
    y_plot=[]
    for i in index:
        x_plot.append(x[int(i)])
        y_plot.append(y[int(i)])
    if plot==True:
        for i in range(0,len(y_plot)):
            plt.scatter(x_plot[i],y_plot[i],color='green',marker='o')
            plt.plot([x_plot[i],x_plot[i]],[0,y_plot[i]],color='skyblue',linestyle='-')
        plt.grid()
        plt.show()
    else:
        return x_plot,y_plot
        

def energy(x):
    sum_e=0
    for i in x:
        sum_e+=(i*i)
    return sum_e

def power(x):
    return (energy(x)/(len(x)+1))

def snr(Ps,Pn):
    return 10*math.log((power(Ps)/power(Pn)),10)

def noisysignal(t,g,tau,T,sigma_noise=0.1,fs=100,f=1,phi=0,sigma=1,n=4,tscale=1,tunits='secs',plot=True):
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
            plt.xlabel('time(secs)')
        else:
            plt.xlabel('time(msecs)')
            x*=tscale
    
    
    a=len(x)
    index=np.linspace(0,a,timetoindex(t,fs),endpoint=False)
    x_plot=[]
    y_plot=[]
    for i in index:
        x_plot.append(x[int(i)])
        y_plot.append(y[int(i)])
    for i in range(0,len(x_plot)):
        y_plot[i]+=random.gauss(0,sigma_noise)
    if plot==True:
        for i in range(0,len(y_plot)):
            plt.scatter(x_plot[i],y_plot[i],color='green',marker='o')
            plt.plot([x_plot[i],x_plot[i]],[0,y_plot[i]],color='skyblue',linestyle='-')
        plt.grid()
        plt.show()
    else:
        return x_plot,y_plot
            
            
def snr2sigma(x,snr=10):
    Ps=np.sum(np.power(x,2))/len(x)
    Pn=Ps/(np.power(10,snr/10))
    noise=np.random.randn(len(x))*np.sqrt(Pn)
    return x+noise

def plot_snr2sigma():
    x=np.linspace(0,0.05,400,endpoint=True)
    y=np.vectorize(gammatone)(x,fs=10,f=100)
    y=snr2sigma(y)
    plt.plot(x,y)

def uniform_f(fmin,fmax):
    f=np.random.uniform(fmin,fmax,1)
    return int(f[0])
def uniform_t(T):
    t=np.random.uniform(0,T,1)
    return t[0]
def signal_to_wav(signal,fname,Fs):
    wav_file=wave.open(fname,'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(Fs)
    wav_file.writeframes(signal)
    wav_file.close()

def estimateSnr(threshold):
    dataset=h5py.File('A3a-testdata.h5','r')
    y=list(dataset[r'/testdata']['y'])
    signal_index=[]
    for i in range(len(y)):
        if y[i]>threshold:
            signal_index.append(i)
    plt.grid()
    plt.plot(y)
    plt.xlabel('Index number')
    plt.plot([0,len(y)],[threshold,threshold],linestyle='--',color='black')
    plt.show()
    s=[]
    n=[]
    for i in range(len(y)):
        if i not in signal_index:
            n.append(y[i])
        else:
            s.append(y[i])
    print('The SNR is:',snr(s,n))
    
def noise(t,tau,T,sigma_noise=0.1,fs=1000,f=100,phi=0,sigma=1,n=4):
    x=np.linspace(0,T,100,endpoint=True)
    y=np.vectorize(gammatone)(x,fs,f)
    x+=tau
    for i in range(int(tau*1000)):
        y=np.insert(y,0,0)
        x=np.insert(x,0,1/1000*(int(tau*1000)-i-1))
    for i in range(len(x)):    
        y[i]+=random.gauss(0,sigma_noise)
    return x,y
    
def grandSynthesis(t,T,fmin,fmax,fs=1000,f=100,phi=0,signma=1,n=4):
    tau_i=np.random.uniform(0,T,100)
    f_i=np.random.uniform(fmin,fmax,100)
    x_plot=[]
    y_plot=[]
    for i in range(0,100):
        tau=tau_i[i]
        f=f_i[i]
        x,y=noise(t,tau,T)
        
        for i in range(len(x)):
            if x[i] not in x_plot:
                x_plot.append(x[i])
                y_plot.append(y[i])
            else:
                index=x_plot.index(x[i])
                y_plot[index]+=y[i]
    return x_plot,y_plot

def movingavg(x,lam=0.5):
    y=[]
    for i in range(len(x)):
        if i == 0:
            yn_1=0
        else:
            yn_1=y[-1]
        yn=lam*yn_1+(1-lam)*x[i]
        y.append(yn)
    return y

def randprocess(N,sigma=1):
    y=[]
    for i in range(0,N):
        if i==0:
            a=0
        else:
            a=np.random.normal(y[i-1],sigma)
        y.append(a)
    return y

def filterIIR(x,a,b):
    y=[]
    x_temp=[]
    y_temp=[]
    for i in range(len(a)):
        y_temp.append(0)
    for i in range(len(b)-1):
        x_temp.append(0)
    for i in range(len(x)):
        x_temp.insert(0,x[i])
        yn=0
        for j in range(len(a)):
            yn-=a[j]*y_temp[j]
        for j in range(len(b)):
            yn+=b[j]*x_temp[j]
        y.append(yn)
        
        y_temp.insert(0,yn)
        y_temp.pop
        x_temp.pop
    return np.array(y)

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


# In[41]:


def plot_circle(k,N):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    cir1=Circle(xy=(0.0,0.0),radius=1,alpha=0.1)
    ax.add_patch(cir1)
    plt.axis('scaled')
    plt.axis('equal')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    for n in range(0,N):
        plt.plot([np.cos(2*np.pi*k/N*n),0],[np.sin(2*np.pi*k/N*n),0],marker = 'o',color = 'black')
        
def w(n,k,N):
    number  = np.cos(2*np.pi*k/N*n)+1j*np.sin(2*np.pi*k/N*n)
    return number

def plotw(k,N):
    for n in range(N):
        number  = w(n,k,N)
        real = number.real
        plt.scatter(n,real,color='r')
        plt.plot([n,n],[0,real],color="skyblue")
    plt.ylabel("real")
    plt.grid()
    plt.show()
    for n in range(N):
        number  = w(n,k,N)
        imag = number.imag
        plt.scatter(n,imag,color='r')
        plt.plot([n,n],[0,imag],color="skyblue")
    plt.ylabel("imaginary")
    plt.grid()
    plt.show()
    
def fourier_matrix(N):
    matrix = np.ones(N*N, dtype=complex).reshape((N, N))
    for n in range(N):
        for k in range(N):
            temp  = complex(w(n,k,N))
            matrix[n][k] = temp
    return matrix

def compare(matrix,compare_value = 1e-10):
    matrix.real[abs(matrix.real) < compare_value] = 0.0
    matrix.imag[abs(matrix.imag) < compare_value] = 0.0
    return matrix

def dft(x):
    N=len(x)
    R = np.zeros(N,dtype=complex)
    for k in range(N):
        for n in range(N):
            temp  = complex(w(-n,k,N))
            R[k]+=x[n]* temp
    return compare(R)


# In[ ]:




