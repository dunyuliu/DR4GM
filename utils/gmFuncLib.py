#! /usr/bin/env python3

import os, time
import numpy as np
from scipy.spatial import cKDTree
from user_defined_params import par
import matplotlib.pyplot as plt
import pickle 

class parametersForGM():
    gmSamplingRate = 10
    dt = par.dt*gmSamplingRate
    T = par.term
    numOfTimeStep = round(T/dt)
    g = 9.8
    gmMetricsKeys = ['PGA', 'PGV', 'PGD', 'CAV']
    stInfoKeys = ['Rjb', 'x', 'y']
    periods = np.array([0.100, 0.125, 0.25, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5])
    periodsKeys = [f'RSA_T_{period:.3f}' for period in periods]
    totalGMMetricsKeys = gmMetricsKeys + periodsKeys
    plotTimeseries = False
    faultType = 'strike'
    faultXmin = -20e3
    faultXmax = 20e3
    cmap = 'inferno'
    
def removeDuplicates(stLocIndex):
    uniqueData, uniqueId = np.unique(stLocIndex[:,:3], 
                                axis=0,
                                return_index=True)
    uniqueChunkId = stLocIndex[uniqueId,3]
    uniqueNumOfSt = stLocIndex[uniqueId,4]
    uniqueStLocIndex = np.column_stack((uniqueData, uniqueChunkId, uniqueNumOfSt))
    
    np.savetxt('gmUniqueStLocIndex.txt', 
            np.vstack(uniqueStLocIndex),
            delimiter='\t', fmt='%.6f')
    return uniqueStLocIndex
    
def loadStLocForOneChunk(chunkId):
    fName = 'surface_coor.txt'+str(chunkId)
    with open(fName, 'r') as f:
        data = np.loadtxt(fName)
        dataPlusChunkIdPlusNumOfSt = np.column_stack((data, 
            np.full(data.shape[0],chunkId), 
            np.full(data.shape[0],data.shape[0])))
    return dataPlusChunkIdPlusNumOfSt 
    
def buildStLocIndex(numOfProcessors):
    stLocIndex = []
    for chunkId in range(numOfProcessors):
        fName = 'surface_coor.txt'+str(chunkId)
        if os.path.isfile(fName):
            data = loadStLocForOneChunk(chunkId)
            stLocIndex.append(data)
    
    np.savetxt('gmStLocIndex.txt', 
                np.vstack(stLocIndex),
                delimiter='\t', fmt='%.6f')
    
    uniqueStLocIndex = removeDuplicates(np.vstack(stLocIndex))
    return uniqueStLocIndex

def getNearestStLocAndChunkId(queryPoint, stLocIndex):
    tree = cKDTree(stLocIndex[:,:3])
    stIdGlobal = tree.query(queryPoint)[1]
    
    stInfo = {}
    stInfo['X'] = stLocIndex[stIdGlobal][0]
    stInfo['Y'] = stLocIndex[stIdGlobal][1]
    stInfo['Z'] = stLocIndex[stIdGlobal][2]
    chunkId = int(stLocIndex[stIdGlobal][3])
    stInfo['chunkId'] = chunkId
    stInfo['numOfSt'] = int(stLocIndex[stIdGlobal][4])
    
    stLocIndexOneChunk = loadStLocForOneChunk(chunkId)
    tree = cKDTree(stLocIndexOneChunk[:,:3])
    stIdInChunk = tree.query(queryPoint)[1]
    stInfo['stIdInChunk'] = int(stIdInChunk)

    return stInfo

def extractVel(stInfo):
    dtype = np.float64
    valueSize = np.dtype(dtype).itemsize
    
    chunkId = stInfo['chunkId']
    numOfSt = stInfo['numOfSt']
    stId = stInfo['stIdInChunk']
    
    gmBinaryFileName = 'gm'+str(chunkId)
    
    if os.path.isfile(gmBinaryFileName):
        fileSize = os.path.getsize(gmBinaryFileName)
        numOfDataPoints = int(fileSize/valueSize)
        numOfTimeStep = int(numOfDataPoints/numOfSt/3)
        velAlongStrike = np.empty(numOfTimeStep)
        velFaultNormal = np.empty(numOfTimeStep)
        
        indexForAlongStrike = np.arange(numOfTimeStep)*numOfSt*3+stId*3
        
        with open(gmBinaryFileName, 'rb') as f:
            for i, index in enumerate(indexForAlongStrike):
                f.seek(index*valueSize)
                values = np.fromfile(f, dtype=dtype, count=3) 
                velAlongStrike[i] = values[0]
                velFaultNormal[i] = values[1]
                
        np.savetxt('gmSt1.txt', 
                velAlongStrike,
                delimiter='\t', fmt='%.6f')

    return velAlongStrike, velFaultNormal

def plotAndSaveTimeseries(stLoc, ts1, ts2, var, par):
    dt = par.dt
    
    fig, ax = plt.subplots()
    
    numOfTimeSteps = ts1.shape[0]
    time = np.arange(numOfTimeSteps)*dt 
    ax.plot(time, ts1, label='Along-Strike')
    ax.plot(time, ts2, label='Fault-Normal')

    ax.set_xlabel('Time (s)')
    if var == 'Vel':
        ax.set_ylabel(var+ ' (m/s)')
    elif var == 'Acc':
        ax.set_ylabel(var + ' (m/s/s)')
    ax.legend()

    output_file = 'gm'+var+'ST'+str(stLoc[0])+'FN'+str(stLoc[1])+'.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

def velToAcc(velAlongStrike, velFaultNormal, par):
    #accAlongStrike = np.diff(velAlongStrike)/dt
    #accFaultNormal = np.diff(velFaultNormal)/dt
    
    dt = par.dt
    numOfTimeStep = velAlongStrike.shape[0]
    accAlongStrike = np.zeros(numOfTimeStep)
    accFaultNormal = np.zeros(numOfTimeStep)
    for i in range(1,numOfTimeStep):
        accAlongStrike[i] = (velAlongStrike[i] - velAlongStrike[i-1])/dt
        accFaultNormal[i] = (velFaultNormal[i] - velFaultNormal[i-1])/dt

    return accAlongStrike, accFaultNormal

def calcGMMetricsFromAccForOneSt(accx, accy, par):
    from ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite import low_pass_filter, compute_cav_gmrot, gmrotdpp_withPG
    
    dt = par.dt
    accx = accx*100 # convert to cm/s/s 
    accy = accy*100
    periods = par.periods
    result = gmrotdpp_withPG(accx, dt, accy, dt, periods, percentile=50, damping=0.05, units='cm/s/s', method='Nigam-Jennings')
    
    return result
    
def plotAndSaveGMMetricsContours(xx, yy, gmMetricsValues, par):
    for key in par.totalGMMetricsKeys:
        if 'RSA' in key or 'PGA' in key:
            unit = 'm/s/s'
        elif 'PGV' in key or 'CAV' in key:
            unit = 'm/s'
        elif 'PGD' in key:
            unit = 'm'
        else:
            unit = ''

        fig = plt.figure()
        ax = plt.gca()
        fontsize = 12
        plt.contourf(xx/1e3, yy/1e3, gmMetricsValues[key], levels=20, cmap=par.cmap)
        cb = plt.colorbar(label=str(unit), orientation='horizontal')
        cb.ax.tick_params(labelsize=fontsize)
        cb.ax.yaxis.offsetText.set_fontsize(fontsize)
        ax.set_title(key, fontsize=fontsize, fontweight='bold')
        ax.set_xlabel('Along Strike (km)', fontsize=fontsize, fontweight='bold')
        ax.set_ylabel('Fault Normal (km)', fontsize=fontsize, fontweight='bold')
        ax.set_aspect('equal')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f'gmContour{key}.png', dpi=300)

        with open(f'gmContour{key}.pickle', 'wb') as f:
            pickle.dump(fig, f)

def saveGMMetricsValues(gmMetricsValues, fileName):
    np.savez(fileName, **gmMetricsValues)

def getGMMetricsForOneSt(stLoc, stLocIndex, par):
    #stLoc = np.array([x,y,0])
    gmMetricsValues = {key: np.float64(0.0) for key in par.totalGMMetricsKeys} 
    
    stInfo = getNearestStLocAndChunkId(stLoc, stLocIndex)
    #print('Input stLoc', stLoc, 'returned stInfo', stInfo)
    velAlongStrike, velFaultNormal = extractVel(stInfo)
    accAlongStrike, accFaultNormal = velToAcc(velAlongStrike, velFaultNormal, par)
    
    gmMetricsOneSt = calcGMMetricsFromAccForOneSt(accAlongStrike, accFaultNormal, par)
    for key in par.gmMetricsKeys:
        gmMetricsValues[key] = gmMetricsOneSt.get(key, 0.0) # default to 0.0 if key is not found.
    
    rsaDict = {key: value for key, value in zip(par.periodsKeys, gmMetricsOneSt['Acceleration'])}
    for key in par.periodsKeys:
        gmMetricsValues[key] = rsaDict.get(key, 0.0)
    
    if par.plotTimeseries==True:
        plotAndSaveTimeseries(stLoc, velAlongStrike, velFaultNormal, 'Vel', par)
        plotAndSaveTimeseries(stLoc, accAlongStrike, accFaultNormal, 'Acc', par)
        
    return gmMetricsValues
    
def getGMMetricsFor2DMap(xRange, yRange, gridSize, stLocIndex, par):
    # making contours

    startTime = time.time()
    x_min, x_max = xRange[0], xRange[1] 
    y_min, y_max = yRange[0], yRange[1]
    nx = round((x_max-x_min)/gridSize+1)
    ny = round((y_max-y_min)/gridSize+1)
    xArr = np.linspace(x_min, x_max, nx)
    yArr = np.linspace(y_min, y_max, ny)

    xx, yy = np.meshgrid(xArr, yArr)

    gmMetricsValues = {key: np.zeros_like(xx) for key in par.totalGMMetricsKeys} #np.zeros_like(xx)
    gmStInfoValues = {key: np.zeros_like(xx) for key in par.stInfoKeys}
    
    stTag = 0
    for i in range(nx):
        for j in range(ny):
            if stTag % 100 == 0:
                print(str(stTag)+' stations are processed ...')
            x = xx[j,i]
            y = yy[j,i]
            stLoc = np.array([x,y,0])
            gmMetricsValuesForOneSt = getGMMetricsForOneSt(stLoc, stLocIndex, par)
            
            for key in par.totalGMMetricsKeys:
                gmMetricsValues[key][j,i] = gmMetricsValuesForOneSt.get(key, 0.0)
            
            Rjb = calcRjb(stLoc, par)
            gmStInfoValues['Rjb'][j,i] = Rjb
            gmStInfoValues['x'][j,i] = x
            gmStInfoValues['y'][j,i] = y
            
            stTag = stTag + 1

    plotAndSaveGMMetricsContours(xx, yy, gmMetricsValues, par)
    saveGMMetricsValues(gmMetricsValues, 'gmMetricsValues.npz')
    saveGMMetricsValues(gmStInfoValues, 'gmStInfoValues.npz')
    
    print('Total time used is ', time.time()-startTime, ' for ', stTag, ' stations.')
    
def calcRjb(stLoc, par):
    if par.faultType == 'strike':
        if stLoc[0]<=par.faultXmin:
            Rjb = ((stLoc[0]-par.faultXmin)**2 + stLoc[1]**2)**0.5
        elif stLoc[0]>=par.faultXmax:
            Rjb = ((stLoc[0]-par.faultXmax)**2 + stLoc[1]**2)**0.5
        else:
            Rjb = abs(stLoc[1])
    
    return Rjb 

