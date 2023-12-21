# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:52:17 2023

@author: sickl
"""

#v2: Update the definition so that a is the spectral slope.

import numpy as np

# %% Colored noise generation (Code adopted from [1] H. Zhivomirov, Rom. J. Acoust. Vib. 15 (2018) 14–19.)
#White noise is a = 0. Red noise is a = 2 (PSD exponent = 2)
def generate_noise(N,a):
    """

    Parameters
    ----------
    N : int
        number of samples to be returned in row vector
    a : float
        spectral slope. For white noise, use a = 0. For brownian noise, use a = -2. For red noise, use a = 2.

    Returns
    -------
    Colored noise vector of noise samples.

    """
    #generate AWGN signal
    x = np.random.randn(N)
    
    #calculate number of fft pts
    npts = -((N)//-2)
    
    #take FFT of x
    xfft = np.fft.fft(x)
    

    #throw away second half    
    xfft = xfft[:npts]
    n = np.linspace(1,npts,npts)
    
    #modify the signal with the desired power.
    xfft = xfft*(n**(a/2)) #spectral noise exponent is a
    

    #revx = np.array(list(reversed(xfft)))
    revx = np.flip(xfft)
    if N//2 == N/2:
        #odd N, exclude Nyquist point
        #reconstruct full spectrum
        xfft = np.concatenate((xfft,np.conj(revx[:-1])))
    else:
        xfft = np.concatenate((xfft,np.conj(revx[1:-1])))
        
    x = np.real(np.fft.ifft(xfft))
    
    x = x - np.mean(x)
    x = x/np.std(x)
    return x