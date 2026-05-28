import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

class AltarOutput:

    '''
    Class to import and visualise Altar output files.
    '''

    def __init__(self, dir=None):
        self.dir = dir
        self.import_final_from_altar()

    
    def import_final_from_altar(self, filename="step_final.h5"):
        '''
        Imports the final output from an Altar run.
        '''
        self.final = h5py.File(Path(self.dir) / filename, "r")
    

    

