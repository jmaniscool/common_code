# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:01:20 2023 by Jordan Sickle

Implementation of Jordan's autofilter approach. The approach assumes a few things about a signal:
    1) The signal is contaminated only by an additive noise process which affects the highest frequencies
    2) The signal is dominated by a process with a power-law scaling power spectrum (either colored noise or avalanche)
    3) The spectral properties of the signal are dominated by the filter for frequencies greater than fixfreq

The steps for this are as follows:
    1) Perform a Butterworth filter to obtain the high frequency components of the signal
    2) Use this as a "noise file" in a Wiener filter to obtain a filtered signal
    
Differences between this Wiener filter and Aya's:
    1) The filter is defined in terms of common units (i.e. Hz), instead of in terms of data index.
    2) Fixes an apparent factor-of-two error in the exponent when generating the ideal ("fake") signal.
    3) Uses a small window to find the mean value of the power spectrum near the fixed frequency.
        This corrects for a significant source of variance in the data.

@author: sickl
"""
import scipy
import numpy as np

#test change

#Butterworth filter to obtain the high-frequency components
#specify the cutoff frequency and sampling frequency in the same units (i.e. Hertz), then use second order samples to get the noise.
#cutoff frequency is the frequency at which the filter equals 1/sqrt(2) of its max
def butter_highpass_filter(data, cutoff, fs = 1, order=5):
    nyq = 0.5 * fs #get the nyquist frequency
    normal_cutoff = cutoff / nyq
    sos = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False, output = 'sos')
    datanoise = scipy.signal.sosfiltfilt(sos, data)
    return datanoise

#autofilter by Wiener filtering the data, using Aya's method.
#give cutoff frequency in Hz, sample frequency in Hz. Also, use a window to get a better estimate of the average power at the given fixed frequency.
#please also input numpy arrays, otherwise this will not work!

#exponent value of 2 is default (assuming MFT). But use the exponent you see in the PSD!!
#using window = 200 should give you the correct mean value within like 5-10%, which is totally fine.
def autofilter(data,fixedFrequency,samplingFrequency = 1, butterworthFilterOrder = 5, averageWindowLength = 200, idealExponent = 2,magicFactor = 1):
    
    #use a frequency some "magic factor" larger than the fixed frequency
    #given when doing the Wiener filter so that the ramp up to the actual
    #noise is not included as part of the ideal Wiener signal.
    
    #generally 1 gives a pretty good estimate for statistics
    lengthData = len(data)

    
    #get the high-frequency noise
    autoNoise = butter_highpass_filter(data, fixedFrequency, samplingFrequency, butterworthFilterOrder)
    
    #get the fft of the signal and noise
    #the data length is n//2 + 1 according to the documentation!!
    fftData = np.fft.rfft(data)
    fftNoise = np.fft.rfft(autoNoise)    
    psData = np.absolute(fftData**2)
    psNoise = np.absolute(fftNoise**2)
    
    #get the frequencies using standard FFT stuff, assuming a real signal (mirrored for negative values -N/2 to 0)
    dataStep = 1/lengthData
        
    fftFrequencies = samplingFrequency*np.linspace(dataStep,1/2, len(data)//2 + 1)        
    
    #obtain the "fixed frequency" (used in the standard Wiener filter)
    fixedFrequencyIdx = np.argmax(fftFrequencies > fixedFrequency*magicFactor)
    
    #calculate the ideal ("fake") signal assuming that the ideal signal should vary as a power law with given exponent
    idealAmplitude = np.mean(psData[fixedFrequencyIdx-averageWindowLength:fixedFrequencyIdx+averageWindowLength])    
    #calculate the ideal signal (correcting for apparent factor of 2 error in Aya's definition of exponent!)


    idealSignal = idealAmplitude*((fftFrequencies/fixedFrequency)**-(idealExponent/2))
    #idealSignal = idealAmplitude*((fixedFrequencyIdx+1)**(idealExponent/2))*(np.arange(1,lengthData/2+1)**(-idealExponent/2))
    
    wienerCoefficients =1/(1+psNoise/idealSignal)
    wienerCoefficients[0]=1
    
    fftFiltered = wienerCoefficients*fftData
    
    filtered = np.fft.irfft(fftFiltered)
    
    
    return filtered



#include some stuff on non-uniformly sampled data







    
