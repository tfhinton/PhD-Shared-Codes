import pymc as pm
import arviz as az
import numpy as np
import copy
import time
from .utils import get_maps_from_arviz

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
        self.ppc = None
    
    @property
    def prior_vals(self):
        return np.array([p.val for p in self.priors])
    
    @property
    def prior_labels(self):
        return np.array([p.label for p in self.priors])
    
    def _copy(self):
        return copy.copy(self)
    
    ## Helper method to print if verbose is enabled
    def _print(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)
    
    def run(self, eps_val=1.e-3, draws=2000, tune=1000, target_accept=0.9, chains=4, cores=None,
            timeout_minutes=30, progressbar=False, init='auto', initvals=None):
        _self = self._copy()
        _self._print("Starting inversion...")


        # Wrap pred_func as a PyTensor/Theano op so PyMC can differentiate through it.
        # For NUTS (gradient-based), we use a black-box op with finite-difference gradients.
        from pytensor.graph.op import Op
        import pytensor.tensor as pt

        class ForwardOp(Op):
            itypes = [pt.dvector]
            otypes = [pt.dvector]

            def perform(_, node, inputs, outputs):
                params = inputs[0]
                outputs[0][0] = _self.forward.pred_func(params).astype(np.float64)

            def grad(_, inputs, output_grads):
                # Finite-difference Jacobian for the black-box forward model
                params = inputs[0]
                g = output_grads[0]
                return [ForwardOpGrad()(params, g)]

        class ForwardOpGrad(Op):
            itypes = [pt.dvector, pt.dvector]
            otypes = [pt.dvector]
            eps = 1e-5

            def perform(_, node, inputs, outputs):
                params, g = inputs
                n = len(params)
                pred0 = _self.forward.pred_func(params)
                jac = np.zeros((len(pred0), n))
                for i in range(n):
                    p_pert = params.copy()
                    p_pert[i] += _.eps
                    jac[:, i] = (_self.forward.pred_func(p_pert) - pred0) / _.eps
                outputs[0][0] = jac.T @ g

        forward_op = ForwardOp()

        _cores = chains if cores is None else cores
        _start_time = time.time()

        def _timeout_callback(*args, **kwargs):
            if timeout_minutes is not None:
                if time.time() - _start_time > timeout_minutes * 60:
                    _self._print(f"\nTimeout of {timeout_minutes} minutes reached — stopping sampling early.")
                    raise KeyboardInterrupt

        with pm.Model() as model:
            # Build priors from UniformDist objects
            param_vars = [p.pm for p in _self.priors]

            # Stack into a single vector for pred_func
            params_vec = pt.stack(param_vars)

            # Forward model prediction
            pred = forward_op(params_vec)

            # Log-likelihood: multivariate Gaussian with full covariance matrix
            pm.MvNormal("obs", mu=pred, cov=_self.Cd, observed=_self.data)

            _self._print(f"Sampling {chains} chains on {_cores} cores "
                         f"(draws={draws}, tune={tune}, timeout={timeout_minutes} min) ...")
            _t0 = time.time()

            # Sample with NUTS (the HMC variant PyMC uses by default)
            result = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,   # increase for more complex posteriors
                chains=chains,
                cores=_cores,
                return_inferencedata=True,
                progressbar=True,
                callback=_timeout_callback,
                init=init,          # 'auto' (jitter+adapt_diag) unless overridden
                initvals=initvals,  # e.g. per-chain dicts for dispersed multimodal starts
            )
            _self._print(f"Sampling done in {(time.time() - _t0) / 60:.1f} min.")
        
        # posterior predictive (linearized)
        with model:
            ppc = pm.sample_posterior_predictive(result, var_names=["obs"])

        _self.result = result
        _self.ppc = ppc

        return _self
            

    def plot_chains(self, var_names=None, title="Distributions and evolutions of Markov chains"):
        if var_names is None:
            var_names = self.prior_labels[:5]
        
        axs = az.plot_trace_dist(self.result, var_names=var_names)
        # fig = np.atleast_2d(axs).flatten()[0].get_figure()
        # fig.suptitle(title)

        # return fig, axs
    

    def plot_posterior(self, var_names=None, title="Posterior distributions of inverted model parameters"):
        axs = az.plot_dist(self.result)
        return axs
        # fig = np.atleast_2d(axs).flatten()[0].get_figure()
        # fig.suptitle(title)

        # return fig, axs
    

    def plot_tradeoffs(self, var_names=None, title="Covariance between inverted model parameters"):
        if var_names is None:
            var_names = self.prior_labels[:5]

        axs = az.plot_pair(self.result, var_names=var_names)
        # fig = np.atleast_2d(axs).flatten()[0].get_figure()
        # fig.suptitle(title)

        # return fig, axs
    

    def plot_ppc(self, title="Compare posterior distribution to observed data distribution"):
        ax = az.plot_ppc_dist(self.ppc)
        # fig = ax.get_figure()
        # fig.suptitle(title)

        # return fig, ax
    

    def diagnostics(self, var_names=None):
        kwarg = {}
        if var_names is not None: kwarg["var_names"] = var_names

        print(az.summary(self.result, var_names=self.prior_labels))
        self.plot_chains(**kwarg)
        self.plot_posterior(**kwarg)
        self.plot_tradeoffs(**kwarg)
        self.plot_ppc()

    def print_summary(self):
        print(az.summary(self.result, var_names=self.prior_labels))
    


        





        
