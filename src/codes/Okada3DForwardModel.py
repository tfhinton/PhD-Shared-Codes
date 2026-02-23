import numpy as np
import copy
import matplotlib.pyplot as plt
from . import meade07

class Okada3DForwardModel:
    
    '''
    Class holding model parameters, utils for constructing the model, and functions to run the forward model.
    The model is a 3D slip model in an elastic medium, after equations in Okada (1985).
    
    Properties: patches (list of PatchThreeD): n segments of the fault
    slips (np array, dim n): amount of slip on each of the n patches
    '''

    def __init__(self):
        self.patches = []
        self.slips = np.array([])
    

    ## Helper method to return a copy of the class instance
    def _copy(self):
        return copy.copy(self)
    

    ## Method to compute surface displacement due to slip on a single patch
    def run(self, eval_pts):
        _self = self._copy()
        xs, ys = eval_pts[:,0], eval_pts[:,1]
        sol = np.zeros((eval_pts.shape[0], 3))
        for patch, slip in zip(_self.patches, _self.slips):
            for i, (x, y) in enumerate( zip(xs, ys) ):
                sol[i] += meade07.displacement(x, y, 0., patch.vertices, *slip)