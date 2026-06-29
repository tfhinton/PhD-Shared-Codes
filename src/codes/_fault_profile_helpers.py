"""
Pure numerical helper functions used by OpticalData.evaluate_profiles_fault_aligned.

These are copied verbatim (with only minor signature tweaks where a function
referenced a module-global in the original) from `fault_profile8.py`, which
cannot be imported directly because that file executes a top-level script on
import. Keeping them here lets OpticalData reuse the validated logic while
leaving `fault_profile8.py` untouched.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def smoothed_boxcar(plen, support_width, gaussian_sigma):
    """
    Create a boxcar smoothed by Gaussian convolution.

    plen: center index
    support_width: width of the box before smoothing
    gaussian_sigma: standard deviation of Gaussian (controls taper)
    """
    xx = np.arange(2 * plen + 1)
    box = np.zeros_like(xx, dtype=float)

    center = plen
    half_width = support_width // 2

    box[(xx >= center - half_width) & (xx <= center + half_width)] = 1.0

    # Smooth edges
    smooth_box = gaussian_filter1d(box, sigma=gaussian_sigma)

    # Optional: normalize to maximum of 1
    smooth_box /= np.max(smooth_box)

    return smooth_box


def replace_outliers_robust(vec, window=5, threshold=5.0):
    vec = vec.copy()
    half = window // 2
    n = len(vec)

    for i in range(n):
        # Define window bounds
        start = max(0, i - half)
        end = min(n, i + half + 1)

        # Exclude the center value
        window_vals = np.delete(vec[start:end], i - start)
        # Remove NaNs
        window_vals = window_vals[~np.isnan(window_vals)]

        if len(window_vals) == 0 or np.isnan(vec[i]):
            continue  # can't replace if nothing to average with

        local_mean = np.nanmean(window_vals)
        if abs(vec[i] - local_mean) > threshold:
            vec[i] = local_mean

    return vec


def remove_marginal_outliers(tmp1, plen, buffer=10, error_quantile=0.8):
    """
    Mask outliers in a 2D strain window while protecting a central column band.

    In the original `fault_profile8.py` this referenced a module-global ``plen``;
    here it is passed explicitly. ``buffer`` is the half-width (columns) of the
    protected central band, ``error_quantile`` controls the rejection threshold.
    """
    tmp1_tmp = tmp1 * 1.
    tmp1_tmp[:, plen - buffer:plen + buffer] = np.nan
    tmp1_stack = np.tile(np.nanstd(tmp1_tmp, axis=0), (tmp1.shape[0], 1))
    tmp1_stack_diff_mean = np.nanquantile(np.abs(tmp1 - tmp1_stack), error_quantile)
    tmp1_stack_mask = (tmp1 - tmp1_stack) < tmp1_stack_diff_mean
    tmp1_stack_mask[:, plen - buffer:plen + buffer] = 1.
    tmp1_masked = np.where(tmp1_stack_mask, tmp1, np.nan)
    return tmp1_masked


def bearing_to_xy(displacement, bearing_deg):
    bearing_rad = np.deg2rad(bearing_deg)
    dx = displacement * np.sin(bearing_rad)  # east
    dy = displacement * np.cos(bearing_rad)  # north
    return dx, -dy  # negate dy because positive is down for np arrays


def finite_strain(ew, ns, xres, yres, s=0, k=1, component=('exy', 'dilatation')):
    """
    Finite (Green-Cauchy) strain from EW/NS displacement fields.

    Copied from `fault_profile8.py`. Mutates its ``ew``/``ns`` inputs in place
    (NaN handling), so callers should pass copies.

    Args:
        ew, ns: 2D displacement arrays (east-positive, north-positive).
        xres, yres: pixel size used for the displacement-gradient calculation.
        s: fault strike (bearing from north, deg). If non-zero the strain tensor
           is rotated from EW/NS into fault-parallel / fault-normal axes.
        k: Gaussian smoothing kernel sigma applied before gradients (1 = none).
        component: list of strain components to return, e.g.
           ['exy', 'dilatation', 'maxShear'].

    Returns:
        ndarray of shape (ny, nx, len(component)).
    """
    no_components = len(component)
    xres2 = xres
    yres2 = yres
    step = 1  # sliding window step (if step < k, we are oversampling)

    # ew
    ew[np.isnan(ew)] = 0
    if k != 1 and k != 0:
        ewsm = gaussian_filter(ew, sigma=k, mode='wrap')  # gaussian smoothing
    else:
        ewsm = ew
    ew2 = ewsm[0:-1:step, 0:-1:step]
    ew2[ew2 == 0] = np.nan
    # ns
    ns[np.isnan(ns)] = 0
    if k != 1 and k != 0:
        nssm = gaussian_filter(ns, sigma=k, mode='wrap')  # gaussian smoothing
    else:
        nssm = ns
    ns2 = nssm[0:-1:step, 0:-1:step]
    ns2[ns2 == 0] = np.nan

    # displacement gradient tensor
    dudy, dudx = np.gradient(ew2, yres2, xres2, edge_order=2)
    dvdy, dvdx = np.gradient(ns2, yres2, xres2, edge_order=2)

    F = np.array([[dudx, dudy], [dvdx, dvdy]])  # displacement gradient tensor

    # Finite strain (Green-Cauchy)
    E11 = 0.5 * (2 * dudx + dudx**2 + dvdx**2)
    E12 = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
    E21 = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
    E22 = 0.5 * (2 * dvdy + dudy**2 + dvdy**2)

    E = np.array([[E11, E12],
                  [E21, E22]])

    # remove NaNs
    msk = np.isnan(E)
    E[msk] = 0
    F[msk] = 0

    # rotate basis from ew/ns to fault_parallel-fault_normal
    if s != 0:
        sr = np.deg2rad(90 - s)  # angle from East to fault strike
        R = np.array([[np.cos(sr), -np.sin(sr)],
                      [np.sin(sr),  np.cos(sr)]])  # CCW rotation matrix
        E = np.moveaxis(E, [0, 1], [-2, -1])
        Er = np.einsum('ab,...bc,cd->...ad', R, E, R.T)
        mask = np.isnan(E).any(axis=(0, 1))
        Er[..., mask] = np.nan
        Er = np.moveaxis(Er, [-2, -1], [0, 1])
        E = Er

    strain_full_out = np.zeros((dudy.shape[0], dudy.shape[1], no_components))
    ff = -1
    for f in component:
        ff += 1
        if f == 'dudy':
            strain_out = dudy
        if f == 'dudx':
            strain_out = dudx
        if f == 'dvdy':
            strain_out = dvdy
        if f == 'dvdx':
            strain_out = dvdx
        if f == 'vorticity':
            strain_out = np.rad2deg(0.5 * (F[0, 1] - F[1, 0]))
        if f == 'exx':
            strain_out = E[0, 0]
        if f == 'eyy':
            strain_out = E[1, 1]
        if f == 'exy':
            strain_out = (E[0, 1] + E[1, 0]) / 2
        if f == 'dilatation':
            strain_out = np.trace(E[:, :])
        if f == 'maxShear':
            strain_out = 0.5 * ((E[0, 0] - E[1, 1])**2 + 4 * E[0, 1]**2)**0.5
        if f == 'compression':
            strain_out = (E[0, 0] + E[1, 1]) / 2

        strain_full_out[:, :, ff] = strain_out

    return strain_full_out
