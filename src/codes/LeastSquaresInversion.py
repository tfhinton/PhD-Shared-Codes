import numpy as np
import copy
import scipy.optimize as optimize

class LeastSquaresInversion:

    '''
    Manages the inversion procedure

    Properties:
        forward (ForwardModel): the forward model
            pred_func: prediction function taking evaluation points and n parameters as input
        priors (list of Dist objects): prior distributions for each of the n parameters
        prior_vals (np array dim n): shortcut to access the .val property of each prior distribution
        data
        data_covariance
    '''

    def __init__(self, forward_model, p0, data, verbose=True):
        self.forward = forward_model
        self.p0 = p0
        self.data = data
        self.verbose = verbose
        self.result = None
    
    def _copy(self):
        return copy.copy(self)
    
    ## Helper method to print if verbose is enabled
    def _print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)
    
    def run(self, eps_val=1.e-3):
        _self = self._copy()
        _self._print("Starting inversion...")

        def loss_func(p):
            pred = _self.forward.pred_func(p)
            misfit = np.linalg.norm(pred - _self.data)
            return misfit
        res = optimize.minimize(loss_func, _self.p0, method="Nelder-Mead")
        _self.result = res.x
        return _self


        





        
