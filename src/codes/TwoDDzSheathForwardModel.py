import numpy as np
from numba import njit
import copy

from .Patch import PatchTwoD
from .TwoDDzForwardModel import TwoDDzForwardModel


class TwoDDzSheathForwardModel(TwoDDzForwardModel):

    '''
    Boundary-integral (equivalent-density) variant of TwoDDzForwardModel, for a
    damage-zone SHEATH of constant PERPENDICULAR width that dips WITH the fault,
    rather than the fixed vertical column (constant horizontal half-width) used
    by the base class. Drop-in replacement: same public API (build_dipping_patches,
    run, pred_func, plot*, add_gaussian_noise, build_uniform_patches, ...) is
    inherited unmodified from TwoDDzForwardModel -- only the field computation
    (run/run_for_patch) is overridden. dz_half_width now means the sheath's
    perpendicular half-width (not a horizontal one); everything else (patches,
    slips, modulus_ratio, xs) means the same thing as in the base class.

    Why a boundary-integral method (see the walkthrough this class was designed
    in): the base class's k^m image series works because the two zone walls are
    parallel to EACH OTHER and perpendicular to the free surface, so wall<->wall
    reflections compose into a 1D horizontal translation lattice that decouples
    from the free-surface image. A sheath parallel to a dipping fault breaks that
    perpendicularity -- reflections no longer commute with the free-surface
    image, and generically (dip not a fraction of 90 deg) there is no finite
    image set. The general fix is to replace the fixed modulus interface by an
    EQUIVALENT SOURCE DENSITY on its boundary (the two sheath walls, whatever
    their shape), solved so the resulting field satisfies traction continuity
    there. This is the same idea as building an arbitrary slip distribution from
    point dislocation cores (TwoDDzForwardModel's run_for_patch / the classic
    Segall-1985 distributed-dislocation construction), just applied to the
    modulus jump instead of the slip jump, using a different fundamental kernel
    (see below). It reduces to the base class's exact k^m series in the
    vertical-fault limit (validated in test_twod_dz_sheath_forward.py) and is
    cross-checked there against an independent finite-difference PDE solve with
    the modulus mask defined by PERPENDICULAR distance to the (dipping) fault.

    Kernels (positive-depth convention, z=0 at the free surface, domain z>=0):
      H(x,z;x0,d)  = surface-satisfying field of a unit screw dislocation at
                     (x0,d) -- literal transcription of Segall (2010) eq 2.24
                     via plain arctan (branch cut vertical, from the source
                     straight up to the surface at x=x0; this is also why the
                     base class's reciprocal-arctan + explicit sign correction
                     works -- it is the same branch, written differently).
                     Used only for the raw VALUE, and only at z=0 (final output);
                     off-surface it has an unrelated horizontal branch artifact
                     at z=d that is harmless because...
      H's gradient is used instead of H's raw value off-surface (for the
                     boundary-integral right-hand side): it is a smooth,
                     bounded, closed-form rational function everywhere except
                     exactly at the source, so the z=d branch of H itself never
                     enters the computation.
      G(x,z;x0,d)  = -(1/4pi)[ln((x-x0)^2+(d-z)^2) + ln((x-x0)^2+(d+z)^2)], the
                     free-surface (traction-free / Neumann) half-space Green's
                     function for a 2D point SOURCE (not a dislocation): its
                     normal derivative is zero at z=0 by construction (same-sign
                     image, verified in the design walkthrough), and a simple
                     layer of G is continuous in value but its normal derivative
                     jumps by exactly the layer density -- the correct kernel
                     for representing a MODULUS (flux) discontinuity while
                     keeping displacement continuous, as opposed to H (used for
                     the fault itself) whose layers jump in VALUE.

    The sheath boundary is discretised into two piecewise-straight polylines
    (offset from the fault polyline by +-dz_half_width along the local
    perpendicular, with a miter join at interior vertices, extended deep along
    the last patch's dip by `extension_factor` to emulate the base class's
    effectively-infinite column), resampled at higher density near the fault
    (n_near segments down to `split_mult` * locking depth) than in the deep
    tail (n_far segments beyond that) -- see _sheath_walls / _resample_wall.
    Solving self-consistently for the density gives a Fredholm 2nd-kind system
    (see _assemble); flat panels have zero self-influence on their own normal
    derivative at the midpoint by symmetry, so the self term is simply omitted
    rather than needing a separate closed-form self term.
    '''

    def __init__(self, dz_half_width=500., modulus_ratio=0.5, locking_depth=10000.,
                 xs=None, n_near=60, n_far=20, split_mult=3., extension_factor=20.,
                 **set_params_kwargs):
        super().__init__(dz_half_width=dz_half_width, modulus_ratio=modulus_ratio,
                          locking_depth=locking_depth, xs=xs, **set_params_kwargs)
        self.n_near = n_near
        self.n_far = n_far
        self.split_mult = split_mult
        self.extension_factor = extension_factor

    # Build the fault polyline (vertices in (x, depth)) from self.patches.
    def _fault_polyline(self):
        p0 = self.patches[0]
        verts = [(getattr(p0, 'x_top', p0.x), p0.top)]
        for p in self.patches:
            verts.append((getattr(p, 'x_bot', p.x), p.bottom))
        return np.array(verts, dtype=np.float64)

    # Offset the fault polyline by +-dz_half_width along the local perpendicular
    # (miter join at interior vertices), extended deep along the last segment's
    # direction so the sheath behaves like an effectively-infinite column.
    def _sheath_walls(self, dz_half_width):
        verts = self._fault_polyline()
        tang = np.diff(verts, axis=0)
        tang = tang / np.linalg.norm(tang, axis=1, keepdims=True)
        norm = np.column_stack([tang[:, 1], -tang[:, 0]])   # +x side for a vertical fault

        max_depth = verts[-1, 1]
        ext_depth = self.extension_factor * max(max_depth, 1.)
        extra_len = (ext_depth - max_depth) / tang[-1, 1]
        verts_ext = np.vstack([verts, verts[-1] + tang[-1] * extra_len])
        seg_norm = np.vstack([norm, norm[-1:]])

        def offset(sign):
            nseg = len(verts_ext) - 1
            vnorm = np.zeros_like(verts_ext)
            vnorm[0], vnorm[-1] = seg_norm[0], seg_norm[-1]
            for i in range(1, nseg):
                n1, n2 = seg_norm[i - 1], seg_norm[i]
                denom = 1. + n1 @ n2
                vnorm[i] = (n1 + n2) / denom if abs(denom) > 1e-9 else n1
            wall = verts_ext + sign * dz_half_width * vnorm
            # for a dipping fault the perpendicular offset has a nonzero depth
            # component, so naively offsetting the surface vertex leaves the
            # wall not actually touching the surface; slide the first point
            # back along the (unoffset) first segment's own direction until it
            # does -- the wall must start exactly at z=0, its true domain edge.
            t = -wall[0, 1] / tang[0, 1]
            wall[0] = wall[0] + t * tang[0]
            return wall

        return offset(+1.), offset(-1.)

    # Resample a wall polyline: n_near equal-length segments from the surface
    # down to split_mult*locking_depth, then n_far equal-length segments in the
    # (coarser, less influential) deep tail. Returns midpoints, outward-normal
    # unit vectors (pointing away from the sheath interior, i.e. into the host
    # medium on THIS wall's side), and segment lengths.
    def _resample_wall(self, vchain):
        seglen = np.linalg.norm(np.diff(vchain, axis=0), axis=1)
        cum = np.concatenate([[0.], np.cumsum(seglen)])
        total = cum[-1]

        split_depth = self.split_mult * self.patches[-1].bottom
        zs = vchain[:, 1]
        j = np.clip(np.searchsorted(zs, split_depth) - 1, 0, len(seglen) - 1)
        frac = np.clip((split_depth - zs[j]) / (zs[j + 1] - zs[j]), 0., 1.)
        split_s = cum[j] + frac * seglen[j]

        n_near, n_far = self.n_near, self.n_far
        near_t = np.linspace(0., split_s, n_near, endpoint=False) + split_s / n_near / 2.
        far_t = np.linspace(split_s, total, n_far, endpoint=False) + (total - split_s) / n_far / 2.
        targets = np.concatenate([near_t, far_t])
        ds = np.concatenate([np.full(n_near, split_s / n_near), np.full(n_far, (total - split_s) / n_far)])

        mid = np.zeros((len(targets), 2))
        nrm = np.zeros((len(targets), 2))
        for i, t in enumerate(targets):
            k = np.clip(np.searchsorted(cum, t) - 1, 0, len(seglen) - 1)
            f = (t - cum[k]) / seglen[k]
            mid[i] = vchain[k] + f * (vchain[k + 1] - vchain[k])
            d = vchain[k + 1] - vchain[k]
            d = d / np.linalg.norm(d)
            nrm[i] = [d[1], -d[0]]
        return mid, nrm, ds

    # Build sheath collocation geometry (both walls) and factor the transmission
    # matrix M, s.t. M @ sigma = 2*k*b gives the equivalent density for a given
    # source (see class docstring). Independent of slip, so callers can reuse
    # this across multiple sources sharing the same geometry/dz/modulus_ratio.
    def _assemble(self, dz_half_width, modulus_ratio):
        k = (1. - modulus_ratio) / (1. + modulus_ratio)
        wallA, wallB = self._sheath_walls(dz_half_width)
        midA, nrmA, dsA = self._resample_wall(wallA)
        midB, nrmB, dsB = self._resample_wall(wallB)
        mid = np.vstack([midA, midB])
        nrm = np.vstack([nrmA, -nrmB])     # wallB's outward normal is -nrmB
        ds = np.concatenate([dsA, dsB])

        M = len(mid)
        Amat = np.eye(M)
        for i in range(M):
            gx, gz = _G_grad(mid[i, 0], mid[i, 1], mid[:, 0], mid[:, 1])
            contrib = -2. * k * (gx * nrm[i, 0] + gz * nrm[i, 1]) * ds
            contrib[i] = 0.                 # flat-panel self term is exactly 0
            Amat[i, :] += contrib
        return mid, nrm, ds, Amat, k

    # Solve for the equivalent density given point sources (x0, d, weight) and
    # pre-assembled geometry/matrix; returns sigma.
    @staticmethod
    def _solve_sigma(mid, nrm, ds, Amat, k, sources):
        b = np.zeros(len(mid))
        for x0, d, w in sources:
            hx, hz = _H_grad(mid[:, 0], mid[:, 1], x0, d)
            b += w * (hx * nrm[:, 0] + hz * nrm[:, 1])
        return np.linalg.solve(Amat, 2. * k * b)

    @staticmethod
    def _evaluate(xs, mid, ds, sigma, sources):
        u = np.zeros_like(xs)
        for x0, d, w in sources:
            u += w * _H(xs, 0., x0, d)
        for j in range(len(mid)):
            u += sigma[j] * ds[j] * _G(xs, 0., mid[j, 0], mid[j, 1])
        return u

    def _patch_sources(self):
        sources = []
        for p, s in zip(self.patches, self.slips):
            x_bot, x_top = getattr(p, 'x_bot', p.x), getattr(p, 'x_top', p.x)
            sources.append((x_bot, p.bottom, s))
            sources.append((x_top, p.top, -s))
        return sources

    # Method to run the forward model: one geometry build + matrix factorisation
    # shared across all patches (their slip-weighted sources are combined into a
    # single right-hand side), rather than rebuilding per patch.
    def run(self, xs):
        _self = self._copy()
        mid, nrm, ds, Amat, k = _self._assemble(_self.dz_half_width, _self.modulus_ratio)
        sources = _self._patch_sources()
        sigma = _self._solve_sigma(mid, nrm, ds, Amat, k, sources)
        _self.sol = _self._evaluate(xs, mid, ds, sigma, sources)
        _self.sol_xs = xs
        return _self

    # Field due to slip on a single patch (same signature as the base class),
    # with the sheath geometry from ALL of self.patches. Provided for interface
    # parity; run() is more efficient for a full multi-patch model since it
    # shares one matrix factorisation across patches instead of rebuilding it
    # per patch as a naive per-patch loop (e.g. via the inherited run()) would.
    def run_for_patch(self, xs, patch, slip, dz_half_width, modulus_ratio):
        if slip == 0.:
            return np.zeros_like(xs)
        mid, nrm, ds, Amat, k = self._assemble(dz_half_width, modulus_ratio)
        x_bot, x_top = getattr(patch, 'x_bot', patch.x), getattr(patch, 'x_top', patch.x)
        sources = [(x_bot, patch.bottom, 1.), (x_top, patch.top, -1.)]
        sigma = self._solve_sigma(mid, nrm, ds, Amat, k, sources)
        return slip * self._evaluate(xs, mid, ds, sigma, sources)


@njit(cache=True)
def _H(x, zp, x0, d):
    '''Surface-satisfying field of a unit screw dislocation at (x0, d); see
    class docstring. Valid at zp=0 only (final surface evaluation) -- off
    surface it has an unrelated branch at zp=d that only affects the raw value,
    never the smooth gradient used elsewhere (_H_grad).'''
    return -(np.arctan((x - x0) / (d - zp)) - np.arctan((x - x0) / (-d - zp))) / (2. * np.pi)


@njit(cache=True)
def _H_grad(x, zp, x0, d):
    dx = x - x0
    r1 = dx ** 2 + (d - zp) ** 2
    r2 = dx ** 2 + (d + zp) ** 2
    hx = -((d - zp) / r1 + (d + zp) / r2) / (2. * np.pi)
    hz = -(dx / r1 - dx / r2) / (2. * np.pi)
    return hx, hz


@njit(cache=True)
def _G(x, zp, x0, d):
    '''Free-surface (traction-free) Green's function for a 2D point source at
    (x0, d); see class docstring. Used as the equivalent-density kernel for the
    modulus interface.'''
    return -(np.log((x - x0) ** 2 + (d - zp) ** 2) + np.log((x - x0) ** 2 + (d + zp) ** 2)) / (4. * np.pi)


@njit(cache=True)
def _G_grad(x, zp, x0, d):
    dx = x - x0
    r1 = dx ** 2 + (d - zp) ** 2
    r2 = dx ** 2 + (d + zp) ** 2
    gx = -dx * (1. / r1 + 1. / r2) / (2. * np.pi)
    gz = ((d - zp) / r1 - (d + zp) / r2) / (2. * np.pi)
    return gx, gz
