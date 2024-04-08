#! /usr/bin/env python3
import os

try:
   from smtk.intensity_measures import gmrotipp
   print('gmrotipp is successfully imported from smtk.intensity_measures')
except ImportError:
   print('module smtk not found: please follow instruction on the readme (README.md)')
   raise
