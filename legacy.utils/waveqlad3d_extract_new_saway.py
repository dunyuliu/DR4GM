#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:25:16 2020

@author: kwithers
"""

from matplotlib import cm
#import pyrotd
import matplotlib.pyplot as plt
import numpy as np
from ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite import low_pass_filter, gmrotdpp_withPG

def derivative(y,h,order):
#function m = derivative(y,h,order)

    n = y.shape[0]
    m = 0*y;

    i = 0;
    m[i] = (-25*y[i]+48*y[i+1]-36*y[i+2]+16*y[i+3]-3*y[i+4])/(12*h)

    i = 2
    m[i] = (-3*y[i-1]-10*y[i]+18*y[i+1]-6*y[i+2]+y[i+4])/(12*h)

    #I = 3:n-2
    I=np.array(range(2,n-2))
   # I=range(2, n-2)
    m[I] = (y[I-2]-8*y[I-1]+8*y[I+1]-y[I+2])/(12*h)

    i = n-2
    m[i] = (-y[i-3]+6*y[i-2]-18*y[i-1]+10*y[i]+3*y[i+1])/(12*h)

    i = n-1
    m[i] = (3*y[i-4]-16*y[i-3]+36*y[i-2]-48*y[i-1]+25*y[i])/(12*h)

    return m

def main():
    plt.close('all')
    
    data = np.loadtxt('/Users/kwithers/from_frontera/faultdrv1_r05.dat')
    
    
    a=20+data[:,0]
    b=data[:,1]
    c=data[:,2]
    
    fault=np.column_stack([a,c])
    
    #fault=[a(1:627:end),c(1:627:end)];
    
    #left side: 401, right side: 511
    #length: 1121
    lsl=401
    rsl=401
    stl=1601
    lsx=np.zeros(lsl*stl)
    lsy=np.zeros(lsl*stl)
    rsx=np.zeros(rsl*stl)
    rsy=np.zeros(rsl*stl)
    
    templ=np.linspace(0,79.9,stl)
    
    for i in range(stl):
        #lsx=[lsx;np.linspace(0,fault[i,0],lsl)']
        lsx[lsl*i:lsl*(i+1)]= np.linspace(0,fault[i,0],lsl)
        lsy[lsl*i:lsl*(i+1)]=np.ones(lsl)*templ[i]
        
        rsx[rsl*i:rsl*(i+1)]= np.linspace(fault[i,0],40,rsl)
        rsy[rsl*i:rsl*(i+1)]=np.ones(rsl)*templ[i]
    
    
#%drv1c50_f40_h_13
# %drv1c50_f40_h_23
# %drv1c50_f40_h_33#%drv1c50_f40_h_14

    
    fileID4 = '/Users/kwithers/from_frontera/drv1c50_f40_h_a22.Hslice1seisx'
    trup=np.fromfile(fileID4, dtype='float32', count=-1, sep='', offset=0)
    print(trup)
    
    
    fileID4 ='/Users/kwithers/from_frontera/drv1c50_f40_h_a22.Hslice2seisx'
    trup1=np.fromfile(fileID4, dtype='float32', count=-1, sep='', offset=0)
    
    
    fileID4 = '/Users/kwithers/from_frontera/drv1c50_f40_h_a22.Hslice1seisy'
    trupy = np.fromfile(fileID4, dtype='float32', count=-1, sep='', offset=0)
    
    fileID4 = '/Users/kwithers/from_frontera/drv1c50_f40_h_a22.Hslice2seisy'
    trup1y = np.fromfile(fileID4, dtype='float32', count=-1, sep='', offset=0)
    
    padding = 0 
    counter = 0
    GMPE = np.zeros(lsx.shape[0]+lsy.shape[0])
    
    
    period= 0.3
    period= 1
    period= 3
    
    freq=1/period
    
    osc_damping = 0.05
    #osc_freqs = np.logspace(-1, 2, 91)
    osc_freqs=[0.33,1,3.]
    
    
    powerspectra=np.zeros(603)
    slope=np.zeros(401*1601*2)
    
    sa0p3=np.zeros(401*1601*2)
    sa1p0=np.zeros(401*1601*2)
    sa3p3=np.zeros(401*1601*2)
    
    
    counter=0
    for i in range(lsx.shape[0]-1):
        print(i)
        S1=trup[i:-1:lsl*stl]
        S1y=trupy[i:-1:lsl*stl]
        counter=counter+1;                                
        dt=0.314775373058478677E-02*16;
           
        accel_x1 = derivative(S1,dt,4)
        accel_x2 = derivative(S1y,dt,4)
        
  #      rot_osc_resps = pyrotd.calc_rotated_spec_accels(
  #          dt, accel_x1, accel_x2, osc_freqs, osc_damping)
        periods = np.array([0.3,1,3])  
        
#         if (size(powerspectra)==size(tempspec))
#  powerspectra= powerspectra+tempspec;
# end
        #print(np.size(accel_x1))
        #print(np.size(accel_x2))

        if (np.size(accel_x1)==np.size(accel_x2)):
           # print(i)
            result = gmrotdpp_withPG(accel_x1, dt, accel_x2, dt, periods, percentile=50, damping=0.05, units='cm/s/s', method='Nigam-Jennings')
            list_of_dict_values = list(result.values())
        #print(result)
        #pause
            sa0p3[counter]=list_of_dict_values[3][0]
            sa1p0[counter]=list_of_dict_values[3][1]
            sa3p3[counter]=list_of_dict_values[3][2]

            
            
    for i in range(rsx.shape[0]-1):
        print(i)
        S1=trup1[i:-1:rsl*stl]
        S1y=trup1y[i:-1:rsl*stl]
        counter=counter+1;                                
        dt=0.314775373058478677E-02*16;
           
        accel_x1 = derivative(S1,dt,4)
        accel_x2 = derivative(S1y,dt,4)
        
        if (np.size(accel_x1)==np.size(accel_x2)):
            result = gmrotdpp_withPG(accel_x1, dt, accel_x2, dt, periods, percentile=50, damping=0.05, units='cm/s/s', method='Nigam-Jennings')
            list_of_dict_values = list(result.values())
        #print(result)
        #pause
            sa0p3[counter]=list_of_dict_values[3][0]
            sa1p0[counter]=list_of_dict_values[3][1]
            sa3p3[counter]=list_of_dict_values[3][2]
            
    
                
    fig=plt.figure('SA1')
    l1=np.zeros(401*1601*2)             
    l2=np.zeros(401*1601*2)   
    
    l1[0:401*1601]=lsx
    l1[401*1601-1:-1]=rsx
    l2[0:401*1601]=  lsy           
    l2[401*1601-1:-1]=rsy
    plt.scatter(l1,l2,s=1,c=(sa0p3), marker = 'o', cmap = cm.jet )
    plt.colorbar()
    #plt.clim(-2,2)
    #plt.axis(image)
    
    
    plt.plot([20,20],[20,60],'k')
    
    
    
    fig=plt.figure('SA2')
    l1=np.zeros(401*1601*2)             
    l2=np.zeros(401*1601*2)   
    
    l1[0:401*1601]=lsx
    l1[401*1601-1:-1]=rsx
    l2[0:401*1601]=  lsy           
    l2[401*1601-1:-1]=rsy
    plt.scatter(l1,l2,s=1,c=(sa1p0), marker = 'o', cmap = cm.jet )
    plt.colorbar()
    #plt.clim(-2,2)
    #plt.axis(image)
    
    
    plt.plot([20,20],[20,60],'k')
    
    
    fig=plt.figure('SA3')
               
    #scatter([lsx;rsx],[lsy;rsy], pointsize, slope,'filled');
    
    l1=np.zeros(401*1601*2)             
    l2=np.zeros(401*1601*2)   
    
    #l1=np.zeros(401*1601*1)             
    #l2=np.zeros(401*1601*1) 
    l1[0:401*1601]=lsx
    l1[401*1601-1:-1]=rsx
    l2[0:401*1601]=  lsy           
    l2[401*1601-1:-1]=rsy
    plt.scatter(l1,l2,s=1,c=(sa3p3), marker = 'o', cmap = cm.jet )
    plt.colorbar()
    #plt.clim(-2,2)
    #plt.axis(image)
    
    
    plt.plot([20,20],[20,60],'k')
    #np.savetxt('/Users/kwithers/from_frontera/sa0p3_cd.txt', sa0p3)
    np.savetxt('/Users/kwithers/from_frontera/sa0p3_h_a22.txt', sa0p3)  
    np.savetxt('/Users/kwithers/from_frontera/sa1p0_h_a22.txt', sa1p0)  
    np.savetxt('/Users/kwithers/from_frontera/sa3p3_h_a22.txt', sa3p3)  
    
if __name__ == '__main__':
    #freeze_support()
    main()    