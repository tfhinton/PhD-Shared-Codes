import pymc as pm
import arviz as az
import numpy as np
import copy

class HamiltonianInversion:

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

    def __init__(self, forward_model, priors, data, data_covariance, verbose=True):
        self.forward = forward_model
        self.priors = priors
        self.data = data
        self.Cd = data_covariance
        self.verbose = True
        self.result = None
    
    @property
    def prior_vals(self):
        return np.array([p.val for p in self.priors])
    
    def _copy(self):
        return copy.copy(self)
    
    ## Helper method to print if verbose is enabled
    def _print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)
    
    def run(self, eps_val=1.e-3):
        _self = self._copy()
        _self._print("Starting inversion...")


        ####    LINEAR ESTIMATION OF GRADIENT IN GREENS FUNCTION    ####
        p0 = _self.prior_vals
        preds0 = _self.forward.pred_func( p0 )
        eps = np.full_like( p0, eps_val )
        eps = eps * np.abs(p0)
        len_xs = len(_self.forward.xs)
        G = np.zeros((len_xs, p0.size))

        _self._print(f"Linear estimation of gradient in Greens function ({p0.size})")
        for i in range(p0.size):
            _self._print(f"p{i}", end=" ")
            dp = np.zeros_like(p0)
            dp[i] = eps[i]
            plus = _self.forward.pred_func(p0 + dp)
            minus = _self.forward.pred_func(p0 - dp)
            G[:, i] = (plus - minus) / (2.0 * dp[i])
        _self._print("\n... estimation complete:")
        

        ####    CHOLESKY FACTORISATION    ####
        jitter = 1e-10 * np.trace(_self.Cd) / len_xs
        max_tries = 10
        _self._print("Cholesky factorisation...")
        for _ in range(max_tries):
            try:
                L = np.linalg.cholesky(_self.Cd + jitter * np.eye(len_xs))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise RuntimeError("Failed to get Cholesky; increase jitter or check covariance matrix")
        _self._print("... factorisation complete.")

        
        ####    RUN INVERSION    ####
        with pm.Model() as model:

            # Stack priors
            priors = pm.math.stack([p.pm for p in _self.priors])

            # Linearized predictive mean using pm.math.dot
            mu = preds0 + pm.math.dot(G, priors - p0)

            # Multivariate normal likelihood (chol accepts a numpy array)
            pm.MvNormal("obs", mu=mu, chol=L, observed=_self.data)

            # Invert
            trace = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True)

        result = az.summary(trace, var_names=[p.label for p in _self.priors])
        _self.result = result

        return _self
        
