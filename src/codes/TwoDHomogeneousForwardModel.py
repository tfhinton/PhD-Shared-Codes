import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from .Patch import PatchTwoD

class TwoDHomogeneousForwardModel:

    '''
    Class holding model parameters, utils for constructing the model, and functions to run the forward model.
    The model is a 2D slip model in an elastic medium with a reduced-elastic-modulus damage zone around the fault,
    after equations in Segall (2010).

    Properties:
        patches (list of PatchTwoD): n segments of the fault
        slips (np array, dim n): amount of slip on each of the n patches
        dz_half_widths (np array, dim n): damage zone halfwidth per patch
        modulus_ratios (np array, dim n): Ka (see Segall 2010), reduced modulus ratio in the damage zone
        sol (np array, dim x): stores the solution produced by the forward model
        sol_xs (np array, dim x): x inputs to the forward model
    '''

    def __init__(self, **set_params_kwargs):
        self.patches = [PatchTwoD()]
        self.slips = np.array([1.])
        self.sol = None
        self.sol_xs = None
        self.xs = None
    

    ## Helper method to return a copy of the class instance
    def _copy(self):
        return copy.copy(self)
    

    # Method to compute surface displacement due to slip on a single patch
    def run_for_patch(self, xs, patch, slip):

        sol = (slip / np.pi) * (
            (np.arctan((patch.bottom)/ xs) - np.arctan(0. / xs)) - 
            (np.arctan((patch.top)/ xs) - np.arctan(0. / xs))
        )
        return sol
    

    # Method to run the forward model
    def run(self, xs):
        _self = self._copy()
        sol = np.zeros_like(xs)
        n = len(_self.patches)

        for i in range(n):
            sol += _self.run_for_patch(xs, _self.patches[i], _self.slips[i])
        
        _self.sol = sol
        _self.sol_xs = xs
        return _self
    

    # Method for Inversion classes which runs the model based solely on inversion parameters and returns the solution
    def pred_func(self, p):
        _self = self._copy()
        _self.slips = p
        _self = _self.run(_self.xs)
        return _self.sol
    

    # Helper method to build uniform patches
    def build_uniform_patches(self, n_patches, dd_width, initialise_slip=1.):
        _self = self._copy()

        patch_width = dd_width / n_patches
        zs = patch_width * (np.arange(n_patches) + 0.5)

        _self.patches = [PatchTwoD(0, z, patch_width) for z in zs]
        if initialise_slip != False:
            _self.slips = np.array([initialise_slip] * n_patches)

        return _self
    
    # Helper method to quickly generate a triangular style slip distribution. Assumes that the last patch is the deepest.
    def gen_triangular_slip_distribution(self, surface_slip=0.5, max_slip=2.5, max_slip_fractional_depth=0.3):
        _self = self._copy()

        patch_zs = [p.z for p in _self.patches]
        max_depth = _self.patches[-1].z + _self.patches[-1].dd_width/2
        max_slip_depth = max_depth * max_slip_fractional_depth

        slips = np.zeros(len(patch_zs))

        for i in range(len(patch_zs)):
            z = patch_zs[i]
            if z <= max_slip_depth:
                slips[i] = surface_slip + (max_slip-surface_slip)*(z/max_slip_depth)
            else:
                slips[i] = max_slip * (max_depth - z) / (max_depth - max_slip_depth)
        
        _self.slips = slips
        return _self
    
    # Quickly plot the displacement solution and slip distribution
    def plot(self, title=None, invert_slip_x_axis=False, slip_label="Input slip", true_slips=None, xlim=None, axtitles=None):
        fig, axs = plt.subplots(1,2,layout="constrained",figsize=(10,5), gridspec_kw={'width_ratios': [1,2], "wspace": 0.07})
        if title is not None: fig.suptitle(title)
        if axtitles is not None:
            axs[0].set_title(axtitles[0])
            axs[1].set_title(axtitles[1])

        if true_slips is not None:
            line_xs = []
            line_ys = []
            for i, p in enumerate(self.patches):
                line_xs.extend([true_slips[i]]*2)
                line_ys.extend([p.top, p.bottom])
            axs[0].plot(line_xs, line_ys, label="True slip", color="grey")

        line_xs = []
        line_ys = []
        for i, p in enumerate(self.patches):
            line_xs.extend([self.slips[i]]*2)
            line_ys.extend([p.top, p.bottom])
        axs[0].plot(line_xs, line_ys, label=slip_label)
        axs[0].axvline(x=0., color="lightgray", ls="--")
        axs[0].yaxis.set_inverted(True)
        if invert_slip_x_axis:
            axs[0].xaxis.set_inverted(True)
        if xlim is not None:
            axs[0].set_xlim(xlim)
        axs[0].set_xlabel("Slip (m)")
        axs[0].set_ylabel("Depth (m)")
        axs[0].legend()

        axs[1].plot(self.sol_xs, self.sol, color="crimson", label="Inverted solution")
        axs[1].set_xlabel("Distance from fault (m)")
        axs[1].set_ylabel("Horizontal displacement (m)")

        return fig, axs
    
    def plot_inversion_result(self, result, data, title="Compare inversion result to data", true_slips=None, plot_kwargs={}, savefig=None):
        _self = self._copy()
        plot_kwargs["slip_label"] = "Inferred slip"

        _self.slips = result
        _self = _self.run(_self.xs)

        fig, axs = _self.plot(title=title, true_slips=true_slips, **plot_kwargs)
        axs[1].plot(_self.xs, data, label="Synthetic data", color="grey", zorder=1.5, lw=5., alpha=0.5)
        axs[1].legend()

        if savefig is not None:
            fig.savefig(savefig, dpi=300.)
        return fig, axs
