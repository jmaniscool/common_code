# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:32:13 2020

Last updated 3-22-2023 by Jordan Sickle.
Updated to account for when noise file is shorter than data.

@author: sickl
"""

import numpy as np
import numpy.fft as fft                         
import math
import scipy.io as io
from matplotlib import pyplot as plt

#find amplitude for the weiner filter (Aya's code called this as just A = power[fixfreq])
#calculate it as the geometric mean of the values within 50 indices of the idx_freq
def find_amplitude(psd,idx_freq,relative_window = 50):
    logpsd = np.log(psd)
    n = len(logpsd)
    lo = int(idx_freq - relative_window//2)
    if lo < 0:
        lo = 0
    hi = int(idx_freq + relative_window//2)
    if hi > n:
        hi = n
    
    logA = np.mean(logpsd[lo:hi])
    
    return np.exp(logA)

#Jordan updated weiner filter Apr 12, 2024.
#Re-writing it to be in terms of frequency units and to get a better estimate of the filter amplitude.
#Previous version had fixfreq as the Nth index in the frequency vector, i.e. at f = f_s*(fixfreq/(2N)) where N is the length of the data.
def weinerfilter_core(noise,unfiltered, fixfreq, exponent, fs = 1,relative_window = 50):
    unfil_fourier=np.fft.rfft(unfiltered)
    noise_fourier=np.fft.rfft(noise)
    
    N = len(unfiltered)
    
    freq = np.fft.rfftfreq(N,d = 1/fs)
    
    psd = np.abs(unfil_fourier)**2
    npsd = np.abs(noise_fourier)**2
    
    idx_freq = np.argmax(freq > fixfreq) #THIS IS WHAT FIXFREQ IS IN AYA'S CODE
    
    A = find_amplitude(psd,idx_freq,relative_window = relative_window)
    
    #note: this is an "ideal" signal which intersects with the data at idx_freq.
    ideal_sig = A*((idx_freq+1)**exponent)*(np.arange(1.0,len(unfil_fourier)+1)**(-exponent))
    
    weiner_coefficents=1/(1+np.abs(noise_fourier)**2/ideal_sig)
    weiner_coefficents[0]=1
    #We now scale the signal by these values, decreasing the higher frequencies
    #based on their inclusion in the noise
    fil_fourier=weiner_coefficents*unfil_fourier
    #Finally we inverse fourier transorm to get a filtered signal. The ifft may
    #sometimes leave small amounts of imaginary components due to rounding error,
    #this is taken out by taking the real portion.
    filtered= np.fft.irfft(fil_fourier)
    
    return filtered

def weinerfilter_wrap(noise, unfiltered, fixfreq, exponent, fs = 1, relative_window = 50):
    filtered = []
    noiselength = len(noise)
    datalength = len(unfiltered)
    nsegs = int(np.ceil(datalength/noiselength))
    seglength = datalength
    if nsegs == 1:
        filtered = weinerfilter_core(noise[:seglength],unfiltered,fixfreq,exponent,fs = fs, relative_window = relative_window)
        return np.array(filtered)
    
    #in this case, the noise is shorter than the data.
    seglength = noiselength
    #if nsegs is greater than 1, do the first (n-1) segments normally before truncating the final one.
    for i in range(nsegs):
        idxst = i*seglength
        idxen = (i+1)*seglength
        #if the final segment, set idxen to be -1
        if i == nsegs - 1:
            idxen = -1
        tmp = weinerfilter_core(noise[idxst:idxen],unfiltered[idxst:idxen],fixfreq,exponent,fs = fs, relative_window = relative_window)
        filtered.append(list(tmp))    
        
    return np.array(filtered)

#when the noise file is shorter than the data, split the data into chunks and
#filter each part individually.
#old version which uses older weinerfilter code
def weinerfilter_wrap_old(noise, unfiltered,fixedfreq,exponent):
    
    filtered = []
    noiseLen = len(noise)
    dataLen = len(unfiltered)
    noiseShorter = (noiseLen <= dataLen)
    if noiseShorter == True:
        numRuns = dataLen//noiseLen
        remainder = dataLen % noiseLen
        minLen = noiseLen
    else:
        numRuns = 1
        minLen = dataLen
        
    #do the first few filters
    for i in range(numRuns):
        idx_start = i*minLen
        idx_end = (i+1)*minLen
        if noiseShorter == True:
            curData = unfiltered[idx_start:idx_end]
            curNoise = noise
        else:
            curNoise = noise[idx_start:idx_end]
            curData = unfiltered
        a = weinerfilter(curNoise,curData,min([fixedfreq,minLen//2 - 1]),exponent) #replace w/ fixedfreq if i get any errors. Current code
        filtered.append(a)
    
    #do the final filter which ties up the loose ends
    #KNOWN BUG: FINAL FILTER FIXFREQ SHOULD BE MODIFIED BASED ON THE LENGTH
    #   OF THE FINAL BIT
    if noiseShorter == True:
        if len(unfiltered[numRuns*minLen:]) > 5:
            curData = unfiltered[numRuns*minLen:]
            curNoise = noise[:len(curData)]
            filtered.append(weinerfilter(curNoise,curData,min([fixedfreq,(len(curNoise))//2-1]),exponent))
        
    tmp = []
    for sublist in filtered:
        for item in sublist:
            tmp.append(item)
            
    filtered = tmp 
    
    return filtered



#Alan Long 6/16/16
#Last edited: Alan Long 5/17/18

#This code takes data and filters it against a noise file through
#a Fourier transform, a method called Weiner filtering, then returning
#the filtered signal. It accepts two arrays of the same length noise and unfiltered
# and two ints exponent and fixedfreq, and returns an array of the same length

#This code is based on Aya's Matlab code

#Note of caution: There is a known issue with the first and last few points of the filtered results. 
#If you plot them over the original data, you may notice some curled up or curled down edges
#Usually, we include some extra points out side our region of interest 
#and just trim of the first and last 200 points after filtering. 

def weinerfilter(noise1, unfiltered1,fixedfreq1,exponent1): #fixedfreq is a value where the amplitude is taken. Recomended 3000. exponent is recomended 2
  noise = noise1.copy()
  unfiltered = unfiltered1.copy()
  fixedfreq = fixedfreq1
  exponent = exponent1
  
  unfiltered=np.array(unfiltered)
  noise=np.array(noise)
  
  


  #We take the fourier transform of both the signal and the noise
  #unfil_fourier=fft.fft(unfiltered)
  #noise_fourier=fft.fft(noise)
  unfil_fourier=fft.rfft(unfiltered)
  noise_fourier=fft.rfft(noise)
   #We now make a theoretical signal based on the given frequency with exponent scaling
  ampl=np.absolute(unfil_fourier[fixedfreq]**2)
  #fake_sig=ampl*((fixedfreq+1)**exponent)*(np.arange(1.0,len(unfil_fourier)+1)**(-exponent))
  fake_sig=ampl*((fixedfreq+1)**exponent)*(np.arange(1.0,len(unfil_fourier)+1)**(-exponent))
   #The weiner coefficents are essentially the inverse of the noise contribution
   #scaled by our theoretical signal
  weiner_coefficents=1/(1+np.absolute(noise_fourier)**2/fake_sig)
  weiner_coefficents[0]=1
   #We now scale the signal by these values, decreasing the higher frequencies
   #based on their inclusion in the noise
  fil_fourier=weiner_coefficents*unfil_fourier
   #Finally we inverse fourier transorm to get a filtered signal. The ifft may
   #sometimes leave small amounts of imaginary components due to rounding error,
    #this is taken out by taking the real portion.
  filtered=(fft.irfft(fil_fourier)).tolist()
  return filtered


#unimplemented
"""
def j_weinerfilter2(noise,unfiltered,fixedfreq,exponent):
    
    filtered = []
    noiseLen = len(noise)
    dataLen = len(unfiltered)
    noiseShorter = (noiseLen <= dataLen)
    if noiseShorter == True:
        numRuns = dataLen // noiseLen
        remainder = dataLen % noiseLen
        minLen = noiseLen
    else:
        numRuns = 1
        minLen = dataLen
        
    #do the first few filters
    for i in range(numRuns):
        idx_start = i*minLen
        idx_end = (i+1)*minLen
        if noiseShorter == True:
            curData = unfiltered[idx_start:idx_end]
            curNoise = noise
        else:
            curNoise = noise[idx_start:idx_end]
            curData = unfiltered
        a = weinerfilter(curNoise,curData,min([fixedfreq,minLen//2 - 1]),exponent) #replace w/ fixedfreq if i get any errors. Current code
        filtered = filtered + a
        
    #print(len(filtered))
    
    #do the final filter. Fix the final window to be the same length as the noise, then append it to the end of the data.
    #Example: If len(data) = 10 and len(noise) = 8, the filter first filters data(0:8) then data()
    r = []
    if noiseShorter == True:
        if remainder > 5:
            filtered = filtered + list(np.ones(remainder))
            curData = unfiltered[-minLen:]
            curNoise = noise
            filtered[-minLen:] = weinerfilter(curNoise,curData,min([fixedfreq,(len(curNoise))//2-1]),exponent)
            
    return filtered

"""
2+2 