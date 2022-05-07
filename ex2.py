# Kasper Eloranta, H274212, kasper.eloranta@tuni.fi
# Pattern Recognition and Machine Learning, DATA.ML.200
# Exercise 2

import pickle
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm

# Form a sinusoidal signal
N = 160
n = np.arange(N)
f0 = 0.06752728319488948

# Add noise to the signal
sigmaSq = 1.2 # 1c) with sigmaSq=1.2 estimation of frequency seems to work good, if sigmaSq=1.3, it fails sometimes
              # (but not always) clearly, (when A_hat = 1.0*A and phi_hat = 1.0*phi
phi = 0.6090665392794814
A = 0.6669548209299414

x0 = A * np.cos(2 * np.pi * f0 * n+phi)
x = x0 + sigmaSq * np.random.randn(x0.size)

# Estimation parameters

A_hat = A*1.1 # 1d) it seems that adding error to these, A_hat = A*1.1 for instance,
phi_hat = phi*1.1 # the maximum noise level sigmaSq decreases where f0 can be estimated.

fRange = np.linspace(0, 0.5, 201) # 1b) raised amount of samples from 100 to 201 to make the better match
                                  # between true and estimated signals.

SSEs = np.zeros(len(fRange))
TLs = np.zeros(len(fRange))
var = statistics.variance(x)

for index, f in enumerate(fRange):
    SSEs[index] = sum(pow(x-A_hat*np.cos(2*np.pi*f*n+phi_hat),2))
    TLs[index] = pow(1/np.pi*var,len(x))*math.exp((-1/var)*sum(pow(x-A_hat*np.cos(2*np.pi*f*n+phi_hat),2)))

# these seems to be equal, it doesn't matter which we use
fSSEmin = fRange[min(range(len(SSEs)), key=SSEs.__getitem__)]
fTLmax = fRange[max(range(len(TLs)), key=TLs.__getitem__)]
xEstimated = A_hat * np.cos(2 * np.pi * fSSEmin * n+phi_hat)

# 1a)
plt.plot(fRange,SSEs)
plt.title("Squared Error")
plt.show()
# 1a)
plt.plot(fRange,TLs)
plt.title("Likelihood")
plt.show()
# 1a)
plt.plot(n,x0, label = "True signal")
plt.plot(n,x ,'.', label = "Noisy samples")
plt.legend()
plt.show()
# 1a)
plt.plot(n,xEstimated,'-.r', label = "Estimated signal")
plt.plot(n,x0,'b', label = "True signal")
plt.legend()
plt.title("True f0=0.0675 (blue) and estimated f0=0.0657 (red)")
plt.show()
