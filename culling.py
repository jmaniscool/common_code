# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 19:26:49 2023

@author: sickl

implement the data culling approach to find tau discussed in LeBlanc et al 2016.

"""
import numpy as np
from .get_slips import get_slips_wrap as gs
arr = np.array

def culling(time,velocity,intervals = [1,2,5,10], smin = 0, drops = True):
    nums = np.zeros(len(intervals))
    for i in range(len(intervals)):
        currentinterval = intervals[i]
        tempvelocity = velocity[::currentinterval]
        temptime = time[::currentinterval]
        avs = gs(time = temptime, velocity = tempvelocity, drops = drops)
        s = arr(avs[2])
        sc = s[s > smin]
        nums[i] = len(sc)
        
    
    return intervals, nums