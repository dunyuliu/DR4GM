#! /usr/bin/env python3
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class genMapsForEQDYNA:
    def __init__(self, mapType='gm', timeInSec=5, gmComp='strike', cmap='plasma', dim=2):    
        from user_defined_params import par

        self.gmSamplingRate = 10
        self.dt = par.dt*self.gmSamplingRate
        self.T = par.term
        self.dtype = np.float64
        self.valueSize = np.dtype(self.dtype).itemsize
        self.timeStepId = round(timeInSec/self.dt)
        self.timeInSec = timeInSec

        self.mapType = mapType
        self.cmap = cmap
        self.dim = dim

        self.dx = par.dx/1e3
        self.dy = par.dy/1e3
        self.dz = par.dz/1e3

        self.fig = plt.figure(figsize=(8,8))
        self.cbMax = 1 #(m/s)
        self.cbMin = -1 #(m/s)
        self.alpha = 1

        if self.mapType == 'gm':
            self.mapTypeToProcess = ['gm']
            self.dataFileNamePrefix = ['gm']
            self.stLocFileNamePrefix = ['surface_coor.txt']
            self.nValue = [3]
            self.xAxisId = 0 
            self.yAxisId = 1
            self.varLegend = 'Particle velocity (m/s)'
            self.titlePrefix = 'Ground Motion at '
        elif self.mapType == 'src':
            self.mapTypeToProcess = ['src']
            self.dataFileNamePrefix = ['src_evol']
            self.stLocFileNamePrefix = ['frt.txt']
            self.nValue = [1]
            self.xAxisId = 0
            self.yAxisId = 2
            self.varLegend = ['Slip rate (m/s)']
            self.titlePrefix = 'Source at '
        elif self.mapType == 'gm+src':
            if self.dim !=3:
                sys.exit("gm+src map is only available for 3-D map; exiting ...")

            self.mapTypeToProcess = ["gm", "src"]
            self.dataFileNamePrefix = ['gm','src_evol']
            self.stLocFileNamePrefix = ['surface_coor.txt','frt.txt']
            self.nValue = [3,1]
            self.varLegend = 'Particle velocity (m/s) + Slip rate (m/s)'
            self.titlePrefix = ''
        else:
            print("Invalid map type; exiting ...")
            sys.exit(1)

        if gmComp=='strike':
            self.gmCompId = 0
        elif gmComp=='norm':
            self.gmCompId = 1
        elif gmComp=='vert':
            self.gmCompId = 2
        else:
            print("Invalid gm component; exiting ...")
            sys.exit(1)

    def genMap(self):
        nTag = 0
        for iMapType, mapType in enumerate(self.mapTypeToProcess):
            for chunkId in range(1000):
                dataFileName = self.dataFileNamePrefix[iMapType] + str(chunkId)
                stLocFileName = self.stLocFileNamePrefix[iMapType] + str(chunkId)
                if os.path.isfile(dataFileName):
                    stLoc = np.loadtxt(stLocFileName)
                    numOfSt = stLoc.shape[0]
                    map = np.zeros((stLoc.shape[0],4))
                    map[:,:3] = stLoc[:,:3]
                    startIndex = self.timeStepId*numOfSt*self.nValue[iMapType]
                    
                    with open(dataFileName, 'rb') as f:
                        for i in range(numOfSt):
                            index = startIndex + i*self.nValue[iMapType]
                            f.seek(index*self.valueSize)
                            values = np.fromfile(f, dtype=self.dtype, count=self.nValue[iMapType]) 
                            if mapType == 'gm':
                                map[i,3] = values[self.gmCompId]
                            else:
                                map[i,3] = values

                    if nTag==0: 
                        self.fullMap = map
                    else:
                        self.fullMap = np.vstack((self.fullMap, map))
                    nTag += 1
        self.fullMap[:,:3] = self.fullMap[:,:3]/1e3
        return self.fullMap
    
    def saveMap(self):
        np.savetxt(self.mapType+str(self.timeInSec)+'.txt', np.vstack(self.fullMap), delimiter='\t', fmt='%.6f')
    
    def plotMap(self):
        if self.dim == 2: 
            mapXmin, mapXmax = self.fullMap[:,self.xAxisId].min(), self.fullMap[:,self.xAxisId].max()
            mapYmin, mapYmax = self.fullMap[:,self.yAxisId].min(), self.fullMap[:,self.yAxisId].max()
            nx = round((mapXmax - mapXmin)/self.dx)
            ny = round((mapYmax - mapYmin)/self.dy)
            xi = np.linspace(mapXmin, mapXmax, num=nx)
            yi = np.linspace(mapYmin, mapYmax, num=ny)
            xi, yi = np.meshgrid(xi, yi) 
            values = griddata((self.fullMap[:,self.xAxisId], self.fullMap[:,self.yAxisId]), self.fullMap[:,3], (xi, yi), method='linear')
                    
            ax = self.fig.add_subplot()
            plt.pcolormesh(xi, yi, values, cmap=self.cmap)
            plt.colorbar()
            ax.set_xlabel('Along-strike (km)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fault-normal (km)', fontsize=12, fontweight='bold')
            ax.axis('equal')
            ax.set_title(self.titlePrefix+' t='+str(self.timeInSec)+' s', fontsize=12, fontweight='bold')
            plt.show()

        elif self.dim == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.fullMap[:,0], self.fullMap[:,1], self.fullMap[:,2], c=self.fullMap[:,3], cmap=self.cmap, vmin=self.cbMin, vmax=self.cbMax, alpha=self.alpha)
            cbar = plt.colorbar(sc, ax=ax, label=self.varLegend, orientation='horizontal', fraction=0.02, pad=0.1)
            ax.set_xlabel('Along-strike (km)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fault-normal (km)', fontsize=12, fontweight='bold')
            ax.set_zlabel('Up (km)', fontsize=12, fontweight='bold')
            ax.set_title(self.titlePrefix+' t='+str(self.timeInSec)+' s', fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
            ax.grid(True, linewidth=1.5)

            ax.axis('equal')
            ax.view_init(elev=-15, azim=-150)
            plt.show()
        


