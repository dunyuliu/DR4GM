#! /usr/bin/env python3

sourceElements={
    'faultGeometry': ['planar', 'rough', 'dip'],    
    'initialStress': ['onFault', 'inVolume', 'hetero', 'homo'],
    'frictionLaw': ['rsfa', 'rsfs', 'sw', 'tw', 'rsfsSrw'],
    'rheology': ['elastic', 'plastic', 'viscoplastic']
}

pathElements={
    'velocityStructure':['homo', '1D', '3D']
    }
    
siteElements={
    'siteResponse':['yes', 'no']
    }

code = {
    'code': ['eqdyna', 'seisol']
    }
    
codeVersion = {
    'eqdyna': ['v5.3.2']
    }
