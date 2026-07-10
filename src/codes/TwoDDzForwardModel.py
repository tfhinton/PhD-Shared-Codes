import numpy as np
from numba import njit
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from .Patch import PatchTwoD
from .utils import get_maps_from_arviz, get_medians_from_arviz
import arviz as az

class TwoDDzForwardModel:

    '''
    Class holding model parameters, utils for constructing the model, and functions to run the forward model.
    The model is a 2D slip model in an elastic medium with a reduced-elastic-modulus damage zone around the fault,
    after equations in Segall (2010).

    Dipping faults: the compliant zone stays vertical (|x| < dz_half_width, centred
    on the surface trace) but each patch is an inclined segment between endpoints
    (patch.x_top, patch.top) and (patch.x_bot, patch.bottom). The field of uniform
    slip on a segment is the difference of two screw-dislocation-line fields, and
    each line's field in the zoned half-space is an image series generalising
    Segall's: for a line at horizontal offset x0 the images sit at +-2mh + (-1)^m x0
    (in-zone source) or at the slab-reflection positions (out-of-zone source).
    Vertical patches (x_top == x_bot == 0) reproduce the original expressions exactly.

    Properties:
        patches (list of PatchTwoD): n segments of the fault
        slips (np array, dim n): amount of slip on each of the n patches
        dz_half_widths (np array, dim n): damage zone halfwidth per patch
        modulus_ratios (np array, dim n): Ka (see Segall 2010), reduced modulus ratio in the damage zone
        sol (np array, dim x): stores the solution produced by the forward model
        sol_xs (np array, dim x): x inputs to the forward model
    '''

    def __init__(self, dz_half_width=500., modulus_ratio=0.5, locking_depth=10000., xs=None, **set_params_kwargs):
        self.patches = [PatchTwoD(0, locking_depth/2, locking_depth)]
        self.slips = np.array([1.])
        self.dz_half_width = dz_half_width
        self.modulus_ratio = modulus_ratio
        self.sol = None
        self.sol_xs = None
        self.xs = xs
    

    ## Helper method to return a copy of the class instance
    def _copy(self):
        return copy.copy(self)
    

    # Method to compute surface displacement due to slip on a single patch.
    # The patch is the inclined segment (x_top, top) -> (x_bot, bottom); its field is
    # the difference of the two endpoint dislocation-line fields.
    def run_for_patch(self, xs, patch, slip, dz_half_width, modulus_ratio):

        # Estimate length of series required for convergence
        tol = 1e-6
        k = (1. - modulus_ratio) / (1. + modulus_ratio)
        if abs(k) < 1e-12 or dz_half_width < 1e-3:   # homogeneous limit
            k, m_max = 0., 0
        else:
            m_max = int(np.ceil(np.log(tol) / np.log(abs(k))))

        x_top = getattr(patch, "x_top", patch.x)
        x_bot = getattr(patch, "x_bot", patch.x)

        sol = dislocation_line_dz(xs, x_bot, patch.bottom, dz_half_width, k, m_max)
        sol -= dislocation_line_dz(xs, x_top, patch.top, dz_half_width, k, m_max)
        return slip * sol


    # Helper method to build patches for a dipping fault: interfaces are the depth
    # interfaces (m, len n+1) and x_offsets the horizontal position of the fault at
    # each interface (m, +x side, same length). All zeros = vertical fault.
    def build_dipping_patches(self, interfaces, x_offsets, initialise_slip=0.):
        _self = self._copy()
        patches = []
        for i in range(len(interfaces) - 1):
            p = PatchTwoD(x=(x_offsets[i] + x_offsets[i+1]) / 2.,
                          z=(interfaces[i] + interfaces[i+1]) / 2.,
                          dd_width=interfaces[i+1] - interfaces[i])
            p.x_top, p.x_bot = float(x_offsets[i]), float(x_offsets[i+1])
            patches.append(p)
        _self.patches = patches
        _self.slips = np.full(len(patches), float(initialise_slip))
        return _self
    

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
        _self.modulus_ratio = p[1]
        _self.slips[:(len(p)-2)] = p[2:]
        _self = _self.run(_self.xs)
        sol = _self.sol
        return sol
    

    def plot_inversion_result(self, result, data, title="Compare inversion result to data", true_slips=None, true_dz=None, plot_kwargs={}, savefig=None):
        _self = self._copy()
        plot_kwargs["slip_label"] = "Inferred slip"

        maps = get_medians_from_arviz(result)
        _self.dz_half_width = maps[0]
        _self.modulus_ratio = maps[1]
        _self.slips = maps[2:]
        _self = _self.run(_self.xs)

        fig, axs = _self.plot(title=title, true_slips=true_slips, **plot_kwargs)
        axs[1].plot(_self.xs, data, label="Synthetic data", color="grey", zorder=1.5)
        axs[1].legend()

        ax_inset = axs[1].inset_axes([0.12, 0.6, 0.27, .31])
        if true_dz is not None:
            ax_inset.axvline(true_dz, color="grey", ls="--")
        dzgrid, dzkde = az.kde( result.posterior["dz_halfwidth"].values.flatten() )
        ax_inset.plot(dzgrid, dzkde)
        ax_inset.set_xlabel("DZ halfwidth (m)", size=9)
        ax_inset.set_ylabel("PDF", size=9)
        ax_inset.set_title("Damage zone halfwidth PDF", size=10)
        ax_inset.tick_params(labelsize=8)

        ####    Plot PDF as background colour   ####
        labels = list(result.posterior.keys())[1:]
        cmap = clr.LinearSegmentedColormap.from_list("lightblues", ["white", "lightskyblue"])

        for label, patch in zip(labels, _self.patches):
            samples = result.posterior[label].values.flatten()
            grid, kde = az.kde(samples)

            grid_spacing = grid[1]-grid[0]
            grid = grid - grid_spacing/2
            grid = np.append(grid, grid[-1] + grid_spacing)
            grid = np.repeat(grid[:,None], 2, axis=1).T

            zs = np.array([patch.top, patch.bottom])
            zs = np.repeat(zs[:,None], grid.shape[1], axis=1)

            axs[0].pcolormesh(grid, zs, kde[:,None].T, shading="flat", cmap=cmap, alpha=0.5, zorder=1.5)

        if savefig is not None:
            fig.savefig(savefig, dpi=300.)
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
            


@njit
def compute_two_d_dz(xs, depth, slip, dz_half_width, modulus_ratio, m_max):

    # Prepare solution array
    n = xs.shape[0]
    sol = np.empty(n, dtype=np.float64)

    # Rename variables
    u = slip
    dz = dz_half_width
    k = (1-modulus_ratio) / (1+modulus_ratio)
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

@njit(cache=True)
def dislocation_line_dz(xs, x0, d, h, k, m_max):
    '''Surface displacement (unit slip) of a screw dislocation line at (x0, depth d)
    in a half-space with a vertical compliant zone |x| < h (image strength
    k = (mu1-mu2)/(mu1+mu2)). Image series generalise Segall (2010) / Ragon &
    Simons (2021) eq. A12 to an off-centre (possibly out-of-zone) source; the
    branch cut runs from the line up to the surface at x = x0, so each line
    carries a step of 1 there and the -+1/2 far-field antisymmetry.
    m_max = 0 (or h ~ 0) computes the homogeneous half-space solution.'''
    n = xs.shape[0]
    out = np.empty(n, dtype=np.float64)
    homog = (k == 0.) or (m_max <= 0) or (h < 1e-3)

    for i in range(n):
        x = xs[i]

        if homog:
            val = _atan_dx(d, x - x0) / np.pi

        elif abs(x0) <= h:
            # source inside the zone: images at +-2mh + (-1)^m x0, strength k^m
            if abs(x) <= h:
                s = _atan_dx(d, x - x0)
                km = 1.
                for m in range(1, m_max + 1):
                    km *= k
                    em = x0 if m % 2 == 0 else -x0
                    s += km * (_atan_dx(d, x - (2.*m*h + em))
                               + _atan_dx(d, x - (-2.*m*h + em)))
            else:
                # observer outside: transmitted images march away from it
                sd = 1. if x > 0. else -1.
                s = 0.
                km = 1. - k
                for m in range(0, m_max + 1):
                    em = x0 if m % 2 == 0 else -x0
                    s += km * _atan_dx(d, x - (-sd*2.*m*h + em))
                    km *= k
            val = s / np.pi

        else:
            # source outside the zone; mirror so the source sits at xx0 > h
            sx = 1. if x0 > 0. else -1.
            xx, xx0 = sx * x, sx * x0
            if xx > h:
                # source side: direct + wall reflection + multiples through the slab
                s = _atan_dx(d, xx - xx0) - k * _atan_dx(d, xx - (2.*h - xx0))
                kj = (1. - k*k) * k
                for j in range(1, m_max // 2 + 2):
                    s += kj * _atan_dx(d, xx + ((4.*j - 2.)*h + xx0))
                    kj *= k * k
            elif xx >= -h:
                # inside the zone: transmitted source + internal reflections
                s = 0.
                km = 1. + k
                for m in range(0, m_max + 1):
                    pm = (2.*m*h + xx0) if m % 2 == 0 else -(2.*m*h + xx0)
                    s += km * _atan_dx(d, xx - pm)
                    km *= k
            else:
                # opposite side: only even-order multiples get through
                s = 0.
                kj = 1. - k*k
                for j in range(0, m_max // 2 + 2):
                    s += kj * _atan_dx(d, xx - (xx0 + 4.*j*h))
                    kj *= k * k
            val = sx * s / np.pi

        # branch cut to the surface at x0: continuous there, antisymmetric far field
        if x > x0:
            val -= 0.5
        elif x < x0:
            val += 0.5
        out[i] = val
    return out


@njit(cache=True)
def _atan_dx(d, dx):
    # arctan(d / dx) as in Segall's surface expressions; 0 at the (measure-zero)
    # singular points to keep numba happy
    if dx == 0. or d == 0.:
        return 0.
    return np.arctan(d / dx)
