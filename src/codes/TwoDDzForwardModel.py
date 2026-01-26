import numpy as np
from numba import njit
import copy
import matplotlib.pyplot as plt
from .Patch import PatchTwoD
from .utils import get_maps_from_arviz
import arviz as az

class TwoDDzForwardModel:

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
        self.dz_half_width = 500.
        self.modulus_ratio = 0.5
        self.sol = None
        self.sol_xs = None
        self.xs = None
    

    ## Helper method to return a copy of the class instance
    def _copy(self):
        return copy.copy(self)
    

    # Method to compute surface displacement due to slip on a single patch
    def run_for_patch(self, xs, patch, slip, dz_half_width, modulus_ratio):
        
        # Estimate length of series required for convergence
        tol = 1e-6
        m_max = int(np.ceil(np.log(tol) / np.log(abs(modulus_ratio))))

        # Compute solution down to bottom
        sol = compute_two_d_dz(xs, patch.bottom, slip, dz_half_width, modulus_ratio, m_max)

        # Subtract solution down to top
        if patch.top > 0.:
            sol -= compute_two_d_dz(xs, patch.top, slip, dz_half_width, modulus_ratio, m_max)
        
        return sol
    

    # Method to run the forward model
    def run(self, xs):
        _self = self._copy()
        sol = np.zeros_like(xs)
        n = len(_self.patches)

        for i in range(n):
            sol += _self.run_for_patch(xs, _self.patches[i], _self.slips[i], _self.dz_half_width, _self.modulus_ratio)
        
        _self.sol = sol
        _self.sol_xs = xs
        return _self
    

    # Method for Inversion classes which runs the model based solely on inversion parameters and returns the solution
    def pred_func(self, p):
        _self = self._copy()
        _self.dz_half_width = p[0]
        _self.slips = p[1:]
        _self = _self.run(_self.xs)
        sol = _self.sol
        return sol
    

    def plot_inversion_result(self, result, data, title="Compare inversion result to data", plot_kwargs={}):
        _self = self._copy()
        plot_kwargs["slip_label"] = "Inferred slip"

        maps = get_maps_from_arviz(result)
        _self.dz_half_width = maps[0]
        _self.slips = maps[1:]
        _self = _self.run(_self.xs)

        fig, axs = _self.plot(title=title, **plot_kwargs)
        axs[1].plot(_self.xs, data, label="data", color="grey", zorder=1.5)
        axs[1].legend()

        ax_inset = axs[1].inset_axes([0.12, 0.12, 0.27, .33])
        dzgrid, dzkde = az.kde( result.posterior["dz_halfwidth"].values.flatten() )
        ax_inset.plot(dzgrid, dzkde)
        ax_inset.set_xlabel("DZ halfwidth (m)", size=9)
        ax_inset.set_ylabel("PDF", size=9)
        ax_inset.set_title("Damage zone halfwidth PDF", size=11)
        ax_inset.tick_params(labelsize=8)

        ####    Plot PDF as background colour   ####
        labels = list(result.posterior.keys())[1:]

        for label, patch in zip(labels, _self.patches):
            samples = result.posterior[label].values.flatten()
            grid, kde = az.kde(samples)

            grid_spacing = grid[1]-grid[0]
            grid = grid - grid_spacing/2
            grid = np.append(grid, grid[-1] + grid_spacing)
            grid = np.repeat(grid[:,None], 2, axis=1).T

            zs = np.array([patch.top, patch.bottom])
            zs = np.repeat(zs[:,None], grid.shape[1], axis=1)

            axs[0].pcolormesh(grid, zs, kde[:,None].T, shading="flat", cmap="Blues", alpha=0.5, zorder=1.5)

        return fig, axs

    # Helper method to add Gaussian noise to sol
    def add_gaussian_noise(self, std=0.05):
        _self = self._copy()
        noise_generator = np.random.default_rng()
        noise = noise_generator.normal(loc=0.0, scale=std, size=_self.sol.shape)
        _self.sol += noise
        return _self
    

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
    def plot(self, title=None, invert_slip_x_axis=False, slip_label="Input slip"):
        fig, axs = plt.subplots(1,2,layout="constrained",figsize=(10,5), gridspec_kw={'width_ratios': [1,2]})
        if title is not None: fig.suptitle(title)

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
        axs[0].set_xlabel("Slip (m)")
        axs[0].set_ylabel("Depth (m)")
        axs[0].legend()

        axs[1].plot(self.sol_xs, self.sol, color="crimson", label="Inverted solution")
        axs[1].set_xlabel("Displacement from fault (m)")
        axs[1].set_ylabel("Vertical displacement (m)")

        return fig, axs
            


@njit
def compute_two_d_dz(xs, depth, slip, dz_half_width, modulus_ratio, m_max):

    # Prepare solution array
    n = xs.shape[0]
    sol = np.empty(n, dtype=np.float64)

    # Rename variables
    u = slip
    dz = dz_half_width
    k = modulus_ratio
    d = depth

    # Loop through xs
    for i in range(n):
        x = xs[i]
        total = 0.

        # Solution left of damage zone
        if x < -dz:
            series = 0.
            for m in range(m_max):   # sum convergent series
                series += (k**m) * np.arctan(d / (x- 2.*m*dz))
            total = ((u*(1-k))/np.pi) * series
        
        # Solution inside damage zone
        elif abs(x) <= dz:
            series = 0.
            for m in range(1, m_max):   # sum convergent series
                series += (k ** m) * (
                    np.arctan(d / (x - 2.0 * m * dz)) +
                    np.arctan(d / (x + 2.0 * m * dz))
                )
            total = (u / np.pi) * np.arctan(d / x) + (u / np.pi) * series
        
        # Solution right of damage zone
        else:
            series = 0.
            for m in range(m_max):   # sum convergent series
                series += (k**m) * np.arctan(d / (x + 2.*m*dz))
            total = ((u * (1.-k))/np.pi) * series
        
        sol[i] = total
    return sol