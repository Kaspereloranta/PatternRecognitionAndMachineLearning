# Kasper Eloranta, H274212, kasper.eloranta@tuni.fi
# Pattern Recognition and Machine Learning, DATA.ML.200
# Exercise 3

def plotSize():
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(3)

import pickle
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm

# a
L = 100
zeros1 = np.zeros(500)
zeros2=np.zeros(300)
n = np.arange(L)
sinusoidal = np.cos(2*np.pi*0.1*n)
exponential = np.exp(np.pi/5*1j*n)
noiselessSignal = np.concatenate((zeros1,sinusoidal,zeros2))
n = np.arange(900)

# b
noisySignal = noiselessSignal+np.sqrt(0.5)*np.random.randn(noiselessSignal.size)

# c

plotSize()
plt.plot(n,noiselessSignal)
plt.title("Noiseless Signal")
plt.xlim([0,900])
plt.ylim([-1,1])
plt.show()

plotSize()
plt.plot(n,noisySignal)
plt.title("Noisy signal")
plt.xlim([0,900])
plt.ylim([-3,3])
plt.show()

plotSize()
conv = np.convolve(noisySignal,np.flip(sinusoidal),'same')
plt.plot(conv)
plt.plot(np.abs(conv),'g--')
plt.title("Detection result, when frequency and phase match")
plt.xlim([0,900])
plt.ylim([-40,40])
plt.show()

plotSize()
conv = np.convolve(noisySignal,np.flip(exponential),'same')
plt.plot(np.abs(conv),'g--')
plt.plot(np.real(conv),'r--')
plt.plot(np.imag(conv),'b--')
plt.title("Detection result, when frequency matches")
plt.xlim([0,900])
plt.ylim([-40,40])
plt.legend(['Abs','Re','Im'])
plt.show()