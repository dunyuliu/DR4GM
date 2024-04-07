#!/usr/bin/env python3
##
# @file
# This file is part of SeisSol.
#
# @author Thomas Ulrich  
#
# @section LICENSE
# Copyright (c) 2016, SeisSol Group
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# @section DESCRIPTION
#

#Author: Thomas Ulrich
#Date: 29.09.17
#aim: 
#1 Read time history from free surface output in either hdf5 or posix format
#2 compute ground motion parameter (PGA,PGV,PGD, SA(T))
#3 store output in a hdf5 file readable by paraview

import sys
import os
#import h5py
import numpy as np
#import argparse
#from multiprocessing import Pool,cpu_count,Manager
import time
#import lxml.etree as ET
#import seissolxdmf
#import seissolxdmfwriter as sxw
from scipy import signal
from scipy.integrate import cumtrapz

#sys.path.append("%s/gmpe-smtk/" %(os.path.dirname(sys.argv[0])))
try:
   from smtk.intensity_measures import gmrotipp
except ImportError:
   print('module smtk not found: please follow instruction on the readme (README.md)')
   raise

def chunk(xs, n):
    '''Split the list, xs, into n evenly sized chunks'''
    L = len(xs)
    assert 0 < n <= L
    s, r = divmod(L, n)
    t = s + 1
    return ([xs[p:p+t] for p in range(0, r*t, t)] +
            [xs[p:p+s] for p in range(r*t, L, s)])


def low_pass_filter(waveform, fs, cutoff_freq):
    " applied 2 pass zero-phase order 2 low pass filter on data"
    order=2
    b,a = signal.butter(order, cutoff_freq, 'low', fs=fs)
    return signal.filtfilt(b, a, waveform)

def compute_cav_gmrot(acceleration_x, time_step_x, acceleration_y, time_step_y, angles, percentile):
    """ compute the cumulative velocity using gmrot """
    from smtk.intensity_measures import get_cav, rotate_horizontal
    cav_theta = np.zeros(len(angles), dtype=float)
    for iloc, theta in enumerate(angles):
        if iloc == 0:
            cav_theta[iloc] = np.sqrt(get_cav(acceleration_x, time_step_x) * 
                    get_cav(acceleration_y, time_step_y))
        else:
            rot_x, rot_y = rotate_horizontal(acceleration_x, acceleration_y, theta)
            cav_theta[iloc] = np.sqrt(get_cav(rot_x, time_step_x) * 
                    get_cav(rot_y, time_step_y))
    return np.percentile(cav_theta, percentile)

def gmrotdpp_withPG(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
        percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    modified from gmrotdpp to also return gmrotdpp(PGA, PGV and PGD)
    This is much faster than gmrotdpp_slow
    """
    from smtk.intensity_measures import get_response_spectrum,equalise_series,rotate_horizontal
    if (percentile > 100. + 1E-9) or (percentile < 0.):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    # Get the time-series corresponding to the SDOF
    sax, _, x_a, _, _ = get_response_spectrum(acceleration_x,
                                              time_step_x,
                                              periods, damping,
                                              units, method)
    say, _, y_a, _, _ = get_response_spectrum(acceleration_y,
                                              time_step_y,
                                              periods, damping,
                                              units, method)
    x_a, y_a = equalise_series(x_a, y_a)

    #TU: this is the part I m adding
    #compute vel and disp from acceleration and
    #add to the spectral acceleration time series
    velocity_x = time_step_x * cumtrapz(acceleration_x[0:-1], initial=0.)
    displacement_x = time_step_x * cumtrapz(velocity_x, initial=0.)
    x_a = np.column_stack((acceleration_x[0:-1], velocity_x, displacement_x, x_a))

    velocity_y = time_step_y * cumtrapz(acceleration_y[0:-1], initial=0.)
    displacement_y = time_step_y * cumtrapz(velocity_y, initial=0.)
    y_a = np.column_stack((acceleration_y[0:-1], velocity_y, displacement_y, y_a))

    angles = np.arange(0., 90., 1.)
    max_a_theta = np.zeros([len(angles), len(periods)+3], dtype=float)
    max_a_theta[0, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                np.max(np.fabs(y_a), axis=0))
    for iloc, theta in enumerate(angles):
        if iloc == 0:
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                           np.max(np.fabs(y_a), axis=0))
        else:
            rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(rot_x), axis=0) *
                                           np.max(np.fabs(rot_y), axis=0))

    gmrotd = np.percentile(max_a_theta, percentile, axis=0)

    res =  {"PGA": gmrotd[0],
            "PGV": gmrotd[1],
            "PGD": gmrotd[2],
            "Acceleration": gmrotd[3:]}

    #if args.CAV:
    cav = compute_cav_gmrot(acceleration_x, time_step_x, acceleration_y, time_step_y, angles, percentile)
    res['CAV']=cav

    return res

