###   Imports
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
import os
import warnings
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import cmcrameri.cm as cmc
import pyproj
import geopandas as gpd
import scipy.optimize as sp_opt
import scipy.spatial.distance as sp_dist
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, linemerge
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from .Profile import Profile
from ._fault_profile_helpers import (
    smoothed_boxcar, replace_outliers_robust, remove_marginal_outliers,
    finite_strain,
)


def _quiet_runtimewarnings(fn):
    '''Suppress the expected NaN-reduction RuntimeWarnings raised by the
    strain/stacking pipeline (empty-slice means, all-NaN max, divide-by-zero).'''
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
#  Module-level workers for parallel fault-aligned profile extraction.
#
#  The per-profile extraction (the expensive step in
#  OpticalData.evaluate_profiles_fault_aligned) is embarrassingly parallel.
#  These live at module scope so they can run in a ProcessPoolExecutor; the
#  large read-only rasters and geometry arrays are stashed in the module global
#  ``_FA_WORKER`` once in the parent and inherited by forked workers (copy-on-
#  write), so they are never pickled per task.
# ---------------------------------------------------------------------------
_FA_WORKER = {}


def _fa_crop_window(prof_pts, arrays, H, W, expwd=5):
    '''Crop identical padded raster windows around a set of profile sample points.

    Args:
        prof_pts (ndarray): (N, 2) array of (col, row) = (x, y) sample coords (px).
        arrays (list): 2D rasters to crop identically.
        H, W (int): raster height/width.
        expwd (int): padding (px) added around the sample bounding box.

    Returns:
        (cropped_list, tmp_prof) where tmp_prof is the (2, N) [row, col]
        window-local sample array, or (None, None) if the window is degenerate.
    '''
    r0 = int(np.min(prof_pts[:, 1])) - expwd
    r1 = int(np.max(prof_pts[:, 1])) + expwd
    c0 = int(np.min(prof_pts[:, 0])) - expwd
    c1 = int(np.max(prof_pts[:, 0])) + expwd
    r0c, r1c = max(0, r0), min(H, r1)
    c0c, c1c = max(0, c0), min(W, c1)
    if (r1c - r0c) < 3 or (c1c - c0c) < 3:
        return None, None
    tmp_prof = np.fliplr(prof_pts).T.copy()   # row 0 = y(row), row 1 = x(col)
    tmp_prof[0, :] -= r0c
    tmp_prof[1, :] -= c0c
    cropped = [a[r0c:r1c, c0c:c1c] for a in arrays]
    return cropped, tmp_prof


def _fa_extract_profile_impl(i, ctx):
    '''Extract one fault-perpendicular profile (displacement over the full
    half-length ``plen``; finite strain only over the central ``strain_plen``
    window around the fault, where it is needed for relocation/FZW).

    Returns (i, cs, ce, strain_arr, disp_arr):
        strain_arr: (3, 2*strain_plen+1)  rows [shear, dilatation, maxShear]
        disp_arr:   (4, 2*plen+1)         rows [EW, NS(S+), normal, parallel]
        cs, ce: inclusive column span (into the full M-length arrays) that
                ``strain_arr`` occupies.
    '''
    ew = ctx['ew']; ns = ctx['ns']
    H = ctx['H']; W = ctx['W']
    M = ctx['M']; plen = ctx['plen']; strain_plen = ctx['strain_plen']
    xres = ctx['xres']; yres = ctx['yres']; expwd = ctx['expwd']
    angle_i = float(ctx['angle'][i])
    p1 = (ctx['x1'][i], ctx['y1'][i])
    p2 = (ctx['x2'][i], ctx['y2'][i])

    prof = np.linspace([p1[0], p1[1]], [p2[0], p2[1]], num=M)   # (M, 2) col, row

    # ----- displacement over the full profile (cheap: no gradients) -----
    disp_arr = np.full((4, M), np.nan)
    cropped, tmp_prof = _fa_crop_window(prof, [ew, ns], H, W, expwd)
    if cropped is not None:
        ew_win, ns_win = cropped
        ew_n = ew_win                  # east-positive
        ns_n = -ns_win                 # north-positive (ns stored south-positive)
        theta = np.radians(90. - angle_i)
        par = ew_n * np.cos(theta) + ns_n * np.sin(theta)
        nrm = -ew_n * np.sin(theta) + ns_n * np.cos(theta)
        # cval=nan: samples outside the loaded raster window must read as gaps,
        # not as exactly 0, or they pollute the far-field bands downstream
        disp_arr[0, :] = map_coordinates(ew_n, tmp_prof, order=1, cval=np.nan)
        disp_arr[1, :] = map_coordinates(ns_n, tmp_prof, order=1, cval=np.nan)
        disp_arr[2, :] = map_coordinates(nrm, tmp_prof, order=1, cval=np.nan)
        disp_arr[3, :] = map_coordinates(par, tmp_prof, order=1, cval=np.nan)

    # ----- finite strain on the central sub-window only -----
    cs = plen - strain_plen
    ce = plen + strain_plen
    Ms = ce - cs + 1
    strain_arr = np.full((3, Ms), np.nan)
    sprof = prof[cs:ce + 1]
    scrop, stmp = _fa_crop_window(sprof, [ew, ns], H, W, expwd)
    if scrop is not None:
        sew, sns = scrop
        try:
            with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
                warnings.simplefilter('ignore', RuntimeWarning)
                strain = finite_strain(sew.copy(), sns.copy(), xres, yres,
                                       angle_i, k=1,
                                       component=['exy', 'dilatation', 'maxShear'])
                s_shear = np.abs(strain[:, :, 0])
                s_dil = strain[:, :, 1]
                s_maxsh = strain[:, :, 2]
                # protect a central band (fault core) from outlier removal. For the
                # full-strain default this is column ``plen`` (original behaviour);
                # for a restricted strain window it is the fault's window-local col.
                buf = int(ctx.get('prot_buffer', 15))
                if strain_plen == plen:
                    prot = plen
                else:
                    wln = sew.shape[1]
                    prot = int(round(stmp[1, strain_plen]))
                    prot = min(max(prot, buf), max(buf, wln - buf))
                s_shear = remove_marginal_outliers(s_shear, prot, buf, 0.5)
                s_dil = remove_marginal_outliers(s_dil, prot, buf, 0.5)
                strain_arr[0, :] = map_coordinates(s_shear, stmp, order=1, cval=np.nan)
                strain_arr[1, :] = map_coordinates(s_dil, stmp, order=1, cval=np.nan)
                strain_arr[2, :] = map_coordinates(s_maxsh, stmp, order=1, cval=np.nan)
        except Exception:
            pass

    return i, cs, ce, strain_arr, disp_arr


def _fa_extract_profile(i):
    '''Pool entry point: reads the forked-in context from the module global.'''
    return _fa_extract_profile_impl(i, _FA_WORKER)


###   OpticalData class
class OpticalData:

    '''
    Container class for optical image correlation data.

    Properties:
        ew (xarray DataArray): East-West optical image correlation data.
        ns (xarray DataArray): North-South optical image correlation data.
        x, y (ndarray or None): Flat UTM pixel coordinates (metres).
            Populated after flatten() or downsample().
        lon, lat (ndarray or None): Flat geographic coordinates (degrees).
        ew_vals, ns_vals (ndarray or None): Flat EW/NS displacement values.
        Cd_ew, Cd_ns (ndarray or None): Data covariance matrices (N x N).
        sigma_ew, lamda_ew (float or None): Fitted covariance params for EW.
        sigma_ns, lamda_ns (float or None): Fitted covariance params for NS.
        verbose (bool): Whether to print status messages.
    '''

    ## Initialise. Optionally import TIF files.
    def __init__(self, verbose=True, **import_tif_kwargs):

        '''
        Args:
            verbose (bool): Whether to print status messages.
            **import_tif_kwargs: Keyword arguments for importing TIF files. See import_tif method.
        '''

        if verbose: print("Init OpticalData with", import_tif_kwargs)

        # Init object attributes
        self.verbose = verbose
        self.ew = None
        self.ns = None

        # Flat array attributes (populated by flatten() or downsample())
        self.x       = None
        self.y       = None
        self.lon     = None
        self.lat     = None
        self.ew_vals = None
        self.ns_vals = None

        # Covariance attributes
        self.Cd_ew    = None
        self.Cd_ns    = None
        self.sigma_ew = None
        self.lamda_ew = None
        self.sigma_ns = None
        self.lamda_ns = None

        # Import TIF files if filepaths provided
        if "ew_filepath" in import_tif_kwargs or "ns_filepath" in import_tif_kwargs:
            self = self.import_tif(**import_tif_kwargs)


    ## Helper method to print if verbose is enabled
    def _print(self, *args):
        if self.verbose: print(*args)


    ## Helper method to return a deep copy of the class instance
    def _copy(self):
        return copy.deepcopy(self)


    ## Convert UTM x/y to lon/lat using the DataArray's embedded CRS
    def _xy2ll(self, x, y):
        '''Convert UTM metres to (lon, lat) degrees using the raster CRS.'''
        da = self.ew if self.ew is not None else self.ns
        transformer = pyproj.Transformer.from_crs(da.rio.crs, "EPSG:4326", always_xy=True)
        return transformer.transform(x, y)


    ## Subset flat arrays to boolean/index mask u
    def _keep_pixels(self, u):
        self.x       = self.x[u]
        self.y       = self.y[u]
        self.lon     = self.lon[u]
        self.lat     = self.lat[u]
        self.ew_vals = self.ew_vals[u]
        self.ns_vals = self.ns_vals[u]
        if self.Cd_ew is not None:
            self.Cd_ew = self.Cd_ew[np.ix_(u, u)]
        if self.Cd_ns is not None:
            self.Cd_ns = self.Cd_ns[np.ix_(u, u)]


    ## Method to import EW and NS TIF files
    def import_tif(self, ew_filepath=None, ns_filepath=None, clear_nan_zero=True):

        '''
        Import EW and NS optical image correlation TIF files to xarray DataArrays. Stores them in
        self.ew and self.ns.

        Kwargs:
            ew_filepath (str): Filepath to East-West TIF file.
            ns_filepath (str): Filepath to North-South TIF file.
        '''
        self._print("Importing TIF files:", ew_filepath, ns_filepath)

        # Import and store raster data as xarray DataArrays
        if ew_filepath is not None:
            self.ew = rxr.open_rasterio(ew_filepath).squeeze("band", drop=True)
            self._print("Imported EW data shape:", self.ew.shape)

        if ns_filepath is not None:
            self.ns = rxr.open_rasterio(ns_filepath).squeeze("band", drop=True)
            self._print("Imported NS data shape:", self.ns.shape)

        self._print("")
        return self


    ## Method to clean out NaN values
    def clear_nan(self, clear_zero=True, ew=True, ns=True):

        '''
        Remove NaN and (optionally) zero values from data.

        Kwargs:
            clear_zero (bool): Remove zero (0.) values from data.
        '''
        _self = self._copy()
        _self._print("Clearing NaNs...")

        if ew:
            _self.ew = _self.ew.where(_self.ew.notnull())
            if clear_zero: _self.ew = _self.ew.where(_self.ew != 0.)
        if ns:
            _self.ns = _self.ns.where(_self.ns.notnull())
            if clear_zero: _self.ns = _self.ns.where(_self.ns != 0.)

        _self._print("... cleared.")
        return _self


    ## Method to quickly downsample (i.e. reduce resolution) of optical data
    def decimate(self, decimate_factor, ew=True, ns=True):

        '''
        Downsample optical data simply by sampling every nth data point (i.e. n=10 is decimation).

        Kwargs:
            decimate_factor (int): Positive integer n, resampling will select every nth point,
                                   reducing the data array size by n^2.
            ew (bool): Decimate and return EW data if true
            ns (bool): Decimate and return NS data if true
        '''
        _self = self._copy()

        # Copy EW DataArray from source and decimate
        if ew:
            _self._print("Decimating EW optical...")
            _self.ew = _self.ew.isel(x=slice(0,None,decimate_factor), y=slice(0,None,decimate_factor))

        # Copy NS DataArray from source and decimate
        if ns:
            _self._print("Decimating NS optical...")
            _self.ns = _self.ns.isel(x=slice(0,None,decimate_factor), y=slice(0,None,decimate_factor))

        _self._print("Decimated")
        _self._print("")
        return _self

    def get_window(self, x0, x1, y0, y1, ew=True, ns=True):
        _self = self._copy()
        if ew:
            _self.ew = _self.ew.sel(x=slice(x0,x1), y=slice(y0,y1))
        if ns:
            _self.ns = _self.ns.sel(x=slice(x0,x1), y=slice(y0,y1))
        return _self


    ## Extract flat pixel arrays from DataArrays
    def flatten(self):

        '''
        Extract flat pixel arrays from the EW and NS DataArrays. Pixels where either
        EW or NS is NaN are excluded. Populates x, y, lon, lat, ew_vals, ns_vals.

        Returns:
            A new OpticalData copy with flat arrays populated.
        '''

        _self = self._copy()
        _self._print("Flattening DataArrays to pixel arrays...")

        XX, YY = np.meshgrid(_self.ew.x.values, _self.ew.y.values)
        ew_flat = _self.ew.values.flatten()
        ns_flat = _self.ns.values.flatten()
        x_flat  = XX.flatten()
        y_flat  = YY.flatten()

        valid = np.isfinite(ew_flat) & np.isfinite(ns_flat)

        _self.x       = x_flat[valid]
        _self.y       = y_flat[valid]
        _self.ew_vals = ew_flat[valid]
        _self.ns_vals = ns_flat[valid]
        _self.lon, _self.lat = _self._xy2ll(_self.x, _self.y)

        _self._print(f"  Flattened: {int(valid.sum())} valid pixels")
        return _self


    ##  Distance-based quadtree downsampling
    def downsample(self, faults, start_size=20000., min_size=2500., char_dist=1500., scaler=1.,
                   expo_dist=0.7, tolerance=0.005, reject_distance=500.):

        '''
        Distance-based quadtree downsampling.

        The image is covered by a regular grid of square blocks (size start_size metres).
        Each block is recursively subdivided into four equal sub-blocks while:
            (distance_to_fault - char_dist) < block_size ** expo_dist
        Subdivision stops when a block reaches min_size. Pixels in each surviving block
        are averaged to yield one downsampled observation for both EW and NS.

        If flat arrays (x, y, ew_vals, ns_vals) are not yet populated, flatten() is
        called internally first.

        Args:
            faults: A Fault3d object or list thereof. Each must expose a .trace
                    GeoDataFrame with line geometry in the same CRS as the optical data.

        Kwargs:
            start_size (float): Initial block size (metres).
            min_size (float): Minimum block size (metres).
            char_dist (float): Characteristic distance for the subdivision criterion (m).
            scaler (float): Scales distance in subdivision criterion.
            expo_dist (float): Exponent applied to block size in the criterion.
            tolerance (float): Minimum fraction of block area that must contain data.
            reject_distance (float): Delete downsampled pixels closer than this
                distance to any fault trace after averaging (m). Set 0 to skip.

        Returns:
            OpticalData: A new downsampled instance with flat arrays populated.
        '''

        if not isinstance(faults, list):
            faults = [faults]

        # Need flat arrays as source
        src = self if self.x is not None else self.flatten()

        # Merge all fault trace geometries into a single shapely geometry
        fault_geom = unary_union([f.trace.geometry.unary_union for f in faults])

        # Estimate pixel spacing from a small probe subset
        n_probe      = min(1000, len(src.x))
        probe        = np.column_stack([src.x[:n_probe], src.y[:n_probe]])
        dmat         = sp_dist.cdist(probe, probe)
        np.fill_diagonal(dmat, np.inf)
        pixel_spacing = float(dmat.min())
        pixel_area    = pixel_spacing**2

        # Initial grid of blocks covering the data extent
        xmin = np.floor(src.x.min())
        xmax = np.floor(src.x.max()) + 1.
        ymin = np.floor(src.y.min())
        ymax = np.floor(src.y.max()) + 1.

        # Block convention: [[x_left, y_top], [x_right, y_top], [x_right, y_bot], [x_left, y_bot]]
        blocks = [
            [[x0,              y0],
             [x0 + start_size, y0],
             [x0 + start_size, y0 - start_size],
             [x0,              y0 - start_size]]
            for x0 in np.arange(xmin - start_size, xmax + start_size, start_size)[:-1]
            for y0 in np.arange(ymin - start_size, ymax + start_size, start_size)[1:]
        ]

        def _size(b):
            return b[1][0] - b[0][0]

        def _cut4(b):
            (x1, y1), (x2, _), (_, y3), _ = b
            xc, yc = (x1 + x2) / 2., (y1 + y3) / 2.
            return [[[x1,y1],[xc,y1],[xc,yc],[x1,yc]],
                    [[xc,y1],[x2,y1],[x2,yc],[xc,yc]],
                    [[x1,yc],[xc,yc],[xc,y3],[x1,y3]],
                    [[xc,yc],[x2,yc],[x2,y3],[xc,y3]]]

        def _dist_to_fault(b):
            return min(Point(c).distance(fault_geom) for c in b)

        # Iterative subdivision
        do_subdivide = True
        it = 0
        while do_subdivide:
            do_subdivide = False
            new_blocks = []
            for b in blocks:
                sz = _size(b)
                if sz > min_size and scaler * (_dist_to_fault(b) - char_dist) < sz**expo_dist:
                    new_blocks.extend(_cut4(b))
                    do_subdivide = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            it += 1
            self._print(f"  Downsample iter {it}: {len(blocks)} blocks")

        # Average pixels within each surviving block
        out = src._copy()
        ew_list  = []; ns_list  = []
        x_list   = []; y_list   = []
        lon_list = []; lat_list = []
        kept_blocks = []

        for b in blocks:
            x_left, y_top  = b[0]
            x_right, y_bot = b[2]
            inside = ((src.x >= x_left) & (src.x < x_right) &
                      (src.y >= y_bot)  & (src.y < y_top))
            n = int(inside.sum())
            if n == 0:
                continue
            if (n * pixel_area) / (_size(b)**2) < tolerance:
                continue

            ew_list.append(float(np.mean(src.ew_vals[inside])))
            ns_list.append(float(np.mean(src.ns_vals[inside])))

            xc = float(np.mean(src.x[inside]))
            yc = float(np.mean(src.y[inside]))
            lonc, latc = src._xy2ll(xc, yc)
            x_list.append(xc);    y_list.append(yc)
            lon_list.append(lonc); lat_list.append(latc)
            kept_blocks.append(b)

        out.ew_vals = np.array(ew_list)
        out.ns_vals = np.array(ns_list)
        out.x       = np.array(x_list)
        out.y       = np.array(y_list)
        out.lon     = np.array(lon_list)
        out.lat     = np.array(lat_list)
        out.Cd_ew   = None
        out.Cd_ns   = None
        out._blocks = kept_blocks

        self._print(f"  Downsampled: {len(src.x)} -> {len(out.x)} pixels")

        if reject_distance > 0.:
            out._reject_near_fault(fault_geom, reject_distance)

        return out


    def _reject_near_fault(self, fault_geom, distance):
        '''Delete pixels within distance metres of the fault trace.'''
        d = np.array([Point(xi, yi).distance(fault_geom)
                      for xi, yi in zip(self.x, self.y)])
        self._keep_pixels(np.flatnonzero(d > distance))
        self._print(f"  After fault rejection ({distance} m): {len(self.x)} pixels")


    ## Method to evaluate displacement along a given profile
    def evaluate_profile(self, profile, swathe_half_width=500.):

        '''
        Evaluate displacement within a swathe around a profile line. All raster pixels
        within swathe_half_width of the line are projected onto it with no interpolation
        or smoothing. Rotates EW and NS into profile-parallel and profile-perpendicular
        components.

        Args:
            profile (Profile): profile with geometry defined (profile.trace is not None)
            swathe_half_width (float): half-width of the swathe in raster CRS units
        '''
        (x0, y0), (x1, y1) = profile.linestring.coords[:2]
        L = np.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L  # unit along-profile vector

        # Build pixel coordinate grid
        XX, YY = np.meshgrid(self.ew.x.values, self.ew.y.values)
        dx, dy = XX - x0, YY - y0

        # Projection of each pixel onto and perpendicular to the profile
        along = dx * ux + dy * uy
        perp  = dx * (-uy) + dy * ux

        # Rotate EW/NS into profile-parallel and profile-perpendicular components
        theta = np.arctan2(y1 - y0, x1 - x0)
        ew_vals = self.ew.values
        ns_vals = self.ns.values
        parallel_vals = ew_vals * np.cos(theta) + ns_vals * np.sin(theta)
        perp_vals     = -ew_vals * np.sin(theta) + ns_vals * np.cos(theta)

        # Swathe mask: within half-width, within profile extent, not NaN
        mask = (
            (np.abs(perp) <= swathe_half_width) &
            (along >= 0) & (along <= L) &
            ~np.isnan(ew_vals) & ~np.isnan(ns_vals)
        )

        along_pts    = along[mask]
        sort_idx     = np.argsort(along_pts)
        profile.xs            = along_pts[sort_idx] - profile.fault_x
        profile.displacements = np.array([parallel_vals[mask][sort_idx],
                                          perp_vals[mask][sort_idx]])

        return profile


    ## Helper method to evaluate multiple profiles in one go
    def evaluate_profiles(self, profiles, **kwargs):
        return [self.evaluate_profile(p, **kwargs) for p in profiles]


    ## Strain-aligned, along-strike-stacked fault-perpendicular profiling
    @_quiet_runtimewarnings
    def evaluate_profiles_fault_aligned(
            self, fault,
            plen=500, stack=300, trace_smooth=15,
            strain_half_width=None, n_jobs=None, expwd=5,
            prof_dtype=np.float64,
            shift_cap=8, shift_cap_final=4, search_half_width=None,
            background_limit=(-750., 750.),
            near_field_limit=(10., 60.),
            far_field_limit=None,
            stdthr=3., attach_to_fault=True, store=True):
        '''
        Build fault-perpendicular displacement profiles that follow the *true*
        fault trace (located from peak shear strain) rather than a hand-drawn
        approximation. This is a serial port of `fault_profile8.py` adapted to
        operate directly on this object's rasters and a `Fault` trace.

        Pipeline (all distances internally in pixels; converted to metres on output):
          1. Resample the fault trace (cubic spline) to ~1 point per pixel and smooth it.
          2. Compute the local strike at every trace point.
          3. Sample a fault-perpendicular profile (half-length ``plen`` px) at each point,
             extracting both finite-strain (shear/dilatation/maxShear) and EW/NS
             displacement rotated into fault-parallel / fault-normal components.
          4. Sub-pixel shift each profile so the peak shear strain lands at x=0.
          5. Median-stack each profile over its +/-``stack`` along-strike neighbours.
          6. A second, finer global re-shift of the stacked profiles.
          7. Per-profile analysis: background-strain detrend, fault-zone width, and
             near/far-field offset estimation by robust regression.

        The NS component is internally flipped to south-positive to reproduce the
        sign convention of the original script (results are converted back to
        north-positive fault-normal on output).

        Args:
            fault (Fault): fault whose ``.trace`` GeoDataFrame (in the same CRS as
                the optical rasters) provides the rupture trace.

        Kwargs:
            plen (int): fault-perpendicular profile half-length, in pixels. The
                displacement profile spans the full +/-plen; making this large
                (e.g. 5000 for 5 km at 1 m/px) only adds cheap displacement
                sampling, not strain cost (see strain_half_width).
            stack (int): along-strike half-window (in profiles/pixels) for stacking.
            trace_smooth (int): boxcar width (px) for smoothing the rasterised trace.
            strain_half_width (float or None): half-width (in the SAME pixel units
                as plen, i.e. metres at 1 m/px) of the central window over which
                finite strain is computed. Strain is only used to relocate the
                profile onto the true fault (peak shear) and to estimate the
                fault-zone width, so it need only span the likely fault-location
                error (~100 m) plus the fault zone. Restricting it makes strain
                cost independent of plen — essential for long profiles. ``None``
                (default) computes strain over the full +/-plen (original
                behaviour, identical results). Clamped to [4, plen] px.
            n_jobs (int or None): number of worker processes for the per-profile
                extraction loop. ``None`` or 1 runs serially (the only safe mode
                on macOS spawn); >1 uses a forked ProcessPoolExecutor (Linux).
                Pass e.g. os.cpu_count().
            expwd (int): padding (px) around each profile's raster window.
            prof_dtype (numpy dtype): storage dtype for the (n_prof, 4, 2*plen+1)
                profile/stack buffers. Defaults to float64 (validated). Pass
                np.float32 to roughly halve memory for very long profiles over a
                long fault (e.g. plen=5000 over the whole trace), at negligible
                precision cost for these displacement/strain magnitudes.
            shift_cap (float): max |shift| (px) allowed in the first relocation.
            shift_cap_final (float): max |shift| (px) allowed in the second relocation.
            search_half_width (float or None): half-width (px) of the window,
                centred on the drawn trace, searched for the peak shear strain
                in the first relocation. ``None`` (default) keeps the original
                hard-coded ~12 px window. The drawn trace can only be corrected
                within this window, so it must exceed the likely trace error;
                values up to ``strain_half_width`` are sensible. Widening it also
                widens the central band protected from strain outlier removal
                (which would otherwise suppress the true fault's strain peak),
                and raises ``shift_cap`` to at least this value. Keep it below
                the distance to the nearest parallel strand, or the relocation
                may lock onto the wrong fault.
            background_limit (tuple): (left, right) px bounds of the far-field region
                used to detrend the background strain.
            near_field_limit (tuple): (inner, outer) px window for the near-field
                offset regression (relative to the fault-zone edge).
            far_field_limit (tuple or None): (inner, outer) px window for the
                far-field offset regression. Defaults to
                (near_field_limit[1], near_field_limit[1] + 50).
            stdthr (float): threshold (in std-devs of detrended background strain)
                for detecting the fault-zone edge.
            attach_to_fault (bool): if True, set ``fault.refined_trace`` to a
                GeoDataFrame of the strain-relocated fault trace.
            store (bool): if True, store the returned profiles on
                ``self.refined_profiles`` and the per-profile metadata table on
                ``self.refined_profile_table``.

        Returns:
            list of Profile: one stacked, fault-aligned Profile per along-strike
            sample, with ``xs`` (m), ``displacements`` ([parallel, normal], m) and
            metadata attributes (fault-zone width, offsets, peak strain, etc.).
        '''

        if far_field_limit is None:
            far_field_limit = (near_field_limit[1], near_field_limit[1] + 50.)
        background_limit = list(background_limit)
        near_field_limit = list(near_field_limit)
        far_field_limit = list(far_field_limit)

        # ----- raster arrays + affine transform -----
        transform = self.ew.rio.transform()
        xres = abs(transform.a)
        yres = xres
        crs = self.ew.rio.crs

        disp_ew_fill = self.ew.values.astype(float)
        disp_ns_fill = -self.ns.values.astype(float)   # south-positive (script convention)
        H, W = disp_ew_fill.shape

        # strain half-length (px): full profile by default, else the requested
        # central window (clamped). Strain cost ~ window area, so this is the key
        # control on speed for long profiles.
        if strain_half_width is None:
            strain_plen = plen
        else:
            strain_plen = int(round(strain_half_width / xres))
        strain_plen = int(np.clip(strain_plen, 4, plen))

        # first-relocation peak-search window (px): original behaviour is a
        # ~12 px hard-coded boxcar; a wider window lets the relocation correct
        # larger drawn-trace errors (it needs the outlier-protected band and the
        # shift cap to cover it, or the peak it should move to is masked/culled)
        if search_half_width is None:
            search_support, prot_buffer = 25, 15
        else:
            search_half_width = int(np.clip(search_half_width, 12, strain_plen))
            search_support = 2 * search_half_width + 1
            prot_buffer = max(15, search_half_width + 5)
            shift_cap = max(shift_cap, search_half_width)

        # worker count for the extraction loop
        n_jobs = 1 if n_jobs is None else max(1, min(int(n_jobs), os.cpu_count() or 1))

        # ----- 1. rasterise + smooth the fault trace into pixel coordinates -----
        x, y, angle, x0, y0 = self._rasterise_and_strike(fault, transform, xres, trace_smooth)
        n_prof = x0.shape[0]
        self._print(f"Fault trace rasterised to {n_prof} profile positions")
        if n_prof <= 2 * stack + 2:
            raise ValueError(
                f"Trace too short ({n_prof} px) for stack half-width {stack}.")

        # ----- profile end points (perpendicular lines) -----
        theta_perp = np.deg2rad(angle + 90.)
        dxp = plen * np.sin(theta_perp)
        dyp = plen * np.cos(theta_perp)
        x1 = x0 + dxp; y1 = y0 - dyp
        x2 = x0 - dxp; y2 = y0 + dyp

        M = 2 * plen + 1
        prof_dist_pxls = np.linspace(-plen, plen, num=M)

        # ----- 2/3. extract strain and displacement profiles -----
        # prof_store_strain rows: 0 shear, 1 dilatation, 2 maxShear, 3 unused
        # prof_store_disp   rows: 0 EW, 1 NS(S+), 2 fault-normal, 3 fault-parallel
        prof_store_strain = np.full((n_prof, 4, M), np.nan, dtype=prof_dtype)
        prof_store_disp = np.full((n_prof, 4, M), np.nan, dtype=prof_dtype)

        # context shared with the (optionally parallel) per-profile workers
        global _FA_WORKER
        _FA_WORKER = dict(
            ew=disp_ew_fill, ns=disp_ns_fill, H=H, W=W, M=M,
            plen=plen, strain_plen=strain_plen, xres=xres, yres=yres,
            expwd=expwd, angle=angle, x1=x1, y1=y1, x2=x2, y2=y2,
            prot_buffer=prot_buffer)

        def _store(res):
            i, cs, ce, strain_arr, disp_arr = res
            prof_store_strain[i, 0:3, cs:ce + 1] = strain_arr
            prof_store_disp[i, :, :] = disp_arr

        self._print(f"Extracting {n_prof} profiles "
                    f"(plen={plen}px, strain_plen={strain_plen}px, n_jobs={n_jobs})")
        if n_jobs == 1:
            for i in range(n_prof):
                if self.verbose and (i % 200 == 0):
                    self._print(f"  extracting profile {i} of {n_prof}")
                _store(_fa_extract_profile_impl(i, _FA_WORKER))
        else:
            # forked workers inherit _FA_WORKER (copy-on-write); nothing is
            # pickled per task except the integer index and the small result.
            mp_ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_ctx) as ex:
                futures = [ex.submit(_fa_extract_profile, i) for i in range(n_prof)]
                for done, fut in enumerate(as_completed(futures)):
                    if self.verbose and (done % 200 == 0):
                        self._print(f"  extracted {done} of {n_prof}")
                    _store(fut.result())

        _FA_WORKER = {}   # release the shared rasters held for the workers

        # ----- 4. first relocation: shift each profile onto peak shear strain -----
        w_shift = smoothed_boxcar(plen, search_support, 1)
        store_shifts = np.zeros(n_prof)
        for i in range(n_prof):
            store_shifts[i] = self._locate_peak_shift(
                prof_store_strain[i, 0, :], w_shift, plen, shift_cap)

        # interpolate failed (zero) shifts from neighbours
        store_shifts = self._interp_failed_shifts(store_shifts)
        # apply shifts: strain rows 0,1,2 and disp rows 0,1,2,3
        tmpx = np.arange(M, dtype=float)
        self._apply_shift(prof_store_strain, store_shifts, tmpx, rows=(0, 1, 2))
        store_shifts_disp = replace_outliers_robust(store_shifts, window=10, threshold=1.0)
        self._apply_shift(prof_store_disp, store_shifts_disp, tmpx, rows=(0, 1, 2, 3))

        # ----- 5. along-strike median stacking -----
        # strain stats rows: 0 med shear,1 std,2 med dil,3 std,4 med maxShear,5 std
        # disp   stats rows: 0 med EW,1 std,2 med NS,3 std,4 med normal,5 std,6 med par,7 std
        strain_stats = self._stack(prof_store_strain, stack, plen, gate_row=1,
                                   med_rows=(0, 1, 2), out_rows=(0, 2, 4),
                                   min_cols=(2 * strain_plen) // 3)
        disp_stats = self._stack(prof_store_disp, stack, plen, gate_row=1,
                                 med_rows=(0, 1, 2, 3), out_rows=(0, 2, 4, 6),
                                 min_cols=plen // 1.5)
        sl = slice(stack, n_prof - stack)
        strain_stats = strain_stats[sl]
        disp_stats = disp_stats[sl]
        n_stack = strain_stats.shape[0]
        self._print(f"Stacked into {n_stack} profiles")

        # ----- 6. second (finer) global re-shift of stacked profiles -----
        store_shifts2 = self._second_reshift(strain_stats, disp_stats, prof_dist_pxls,
                                             plen, stack, shift_cap_final)

        # ----- 7. per-profile analysis (detrend / FZW / offsets) -----
        profiles = []
        table = np.full((n_stack, 42), np.nan)
        # along-fault distance (m) at each stacked profile centre
        seg = np.hypot(np.diff(x0), np.diff(y0)) * xres
        along_fault = np.concatenate([[0.], np.cumsum(seg)])

        total_shift = store_shifts[sl] + store_shifts2   # px, drawn-trace -> true fault
        for ff in range(n_stack):
            j = ff + stack   # index into full-length trace arrays
            result, shifted = self._analyse_profile(
                ff, strain_stats, disp_stats, prof_dist_pxls, plen, xres,
                background_limit, near_field_limit, far_field_limit, stdthr,
                strain_plen)
            table[ff, :len(result)] = result

            # build Profile (geometry in raster CRS, displacements in metres)
            ux0, uy0 = transform * (x0[j], y0[j])
            table[ff, 0:5] = [x0[j], y0[j], angle[j], ux0, uy0]
            ux1, uy1 = transform * (x1[j], y1[j])
            ux2, uy2 = transform * (x2[j], y2[j])
            line = LineString([(ux1, uy1), (ux2, uy2)])
            gdf = gpd.GeoDataFrame({"x_along_fault": [along_fault[j]]},
                                   geometry=[line], crs=crs)
            p = Profile(trace=gdf, fault_x=plen * xres)
            p.xs = prof_dist_pxls * xres
            # parallel = disp_stats row 6, fault-normal flipped back to N-positive sign
            p.displacements = np.array([disp_stats[ff, 6, :], disp_stats[ff, 4, :]])

            # metadata
            p.x_along_fault = along_fault[j]
            p.strike = angle[j]
            p.fault_utm = transform * (x0[j], y0[j])
            p.strain_shear = shifted[0, :]
            p.strain_dilatation = shifted[1, :]
            p.fzw = result[24]
            p.fzw_dilatation = result[28]
            p.x_min = result[25]; p.x_max = result[26]
            p.offset_near = result[5]; p.offset_near_std = result[6]
            p.offset_far = result[11]; p.offset_far_std = result[12]
            p.offset_near_dil = result[17]; p.offset_near_dil_std = result[18]
            p.peak_shear = result[32]
            profiles.append(p)

        # ----- refined trace (drawn trace shifted perpendicular onto peak strain) -----
        refined_col = x0[sl] - total_shift * np.sin(theta_perp[sl])
        refined_row = y0[sl] + total_shift * np.cos(theta_perp[sl])
        rx, ry = transform * (refined_col, refined_row)
        refined_trace = gpd.GeoDataFrame(
            {"id": [0]}, geometry=[LineString(np.column_stack([rx, ry]))], crs=crs)
        if attach_to_fault:
            fault.refined_trace = refined_trace

        if store:
            self.refined_profiles = profiles
            self.refined_profile_table = table
            self.refined_trace = refined_trace

        return profiles


    # ---- helpers for evaluate_profiles_fault_aligned ----

    def _rasterise_and_strike(self, fault, transform, xres, trace_smooth):
        '''Resample fault trace to ~1pt/px in pixel coords, smooth, return
        (x, y, angle, x0, y0) where x/y are smoothed trace pixel coords, angle is
        the local strike (bearing from N, deg) and x0/y0 the strike-defining points.'''
        # gather UTM vertices (single continuous trace expected)
        merged = unary_union(fault.trace.geometry.values)
        if merged.geom_type == 'MultiLineString':
            merged = linemerge(merged)
        if merged.geom_type == 'MultiLineString':
            merged = max(merged.geoms, key=lambda g: g.length)
            self._print("  warning: trace was multi-part; using longest segment")
        verts = np.asarray(merged.coords)

        d = np.concatenate([[0.], np.cumsum(np.hypot(np.diff(verts[:, 0]),
                                                     np.diff(verts[:, 1])))])
        uniq = np.unique(d, return_index=True)[1]
        d, verts = d[uniq], verts[uniq]
        fx = interp1d(d, verts[:, 0], kind='cubic')
        fy = interp1d(d, verts[:, 1], kind='cubic')
        npts = int(d[-1] / xres)
        dd = np.linspace(0., d[-1], npts + 1)
        xfine, yfine = fx(dd), fy(dd)
        col, row = ~transform * (xfine, yfine)   # UTM -> pixel
        x = np.asarray(col, dtype=float)
        y = np.asarray(row, dtype=float)

        # smooth pixel trace (drops endpoints, mode='valid')
        k = np.ones(trace_smooth) / trace_smooth
        x = np.convolve(x, k, mode='valid')
        y = np.convolve(y, k, mode='valid')

        # local strike angle (bearing from N), span=1
        span = 1
        angle = np.zeros(x.shape)
        for i in range(span, x.shape[0] - span):
            x0i = np.mean(x[i - span:i]); x1i = np.mean(x[i:i + span])
            y0i = np.mean(y[i - span:i]); y1i = np.mean(y[i:i + span])
            dx, dy = x1i - x0i, y1i - y0i
            angle[i] = (90. + np.rad2deg(np.arctan2(dy, dx))) % 360.
        angle = angle[span:-span]
        x0 = x[span:-span]
        y0 = y[span:-span]
        return x, y, angle, x0, y0

    def _profile_window(self, p1, p2, M, ew, ns, H, W, expwd=5):
        '''Crop a padded raster window around a profile and return
        ((ew_win, ns_win), tmp_prof) where tmp_prof is the (2, M) [row, col]
        sample coordinate array in window-local pixels. Returns (None, None) if
        the window falls (largely) outside the raster.'''
        prof = np.linspace([p1[0], p1[1]], [p2[0], p2[1]], num=M)   # cols=x, rows=y
        r0 = int(np.min(prof[:, 1])) - expwd
        r1 = int(np.max(prof[:, 1])) + expwd
        c0 = int(np.min(prof[:, 0])) - expwd
        c1 = int(np.max(prof[:, 0])) + expwd
        r0c, r1c = max(0, r0), min(H, r1)
        c0c, c1c = max(0, c0), min(W, c1)
        if (r1c - r0c) < 3 or (c1c - c0c) < 3:
            return None, None
        tmp_prof = np.fliplr(prof).T.copy()   # row 0 = y(row), row 1 = x(col)
        tmp_prof[0, :] -= r0c
        tmp_prof[1, :] -= c0c
        return (ew[r0c:r1c, c0c:c1c], ns[r0c:r1c, c0c:c1c]), tmp_prof

    def _locate_peak_shift(self, shear, w, plen, cap):
        '''Sub-pixel offset (px) of the peak (weighted) shear strain from centre,
        clipped to +/-cap. Returns 0 on failure.'''
        ysp = shear * w
        xsp = np.linspace(0, ysp.shape[0] - 1, ysp.shape[0])
        valid = ~np.isnan(ysp)
        if valid.sum() < 4:
            return 0.
        try:
            spline = UnivariateSpline(xsp[valid], ysp[valid], s=1e-9)
            xno = int(xsp.shape[0] * 1e3)
            x_fine = np.linspace(xsp.min(), xsp.max(), xno)
            y_fine = spline(x_fine)
            x_shift = (np.nanargmax(y_fine) / xno) * xsp.shape[0] - (xsp.shape[0] / 2.)
            if (x_shift >= cap) or (x_shift <= -cap):
                x_shift = 0.
        except Exception:
            x_shift = 0.
        return x_shift

    @staticmethod
    def _interp_failed_shifts(store_shifts):
        valid = store_shifts != 0
        if valid.sum() < 2:
            return store_shifts
        rows = np.arange(store_shifts.shape[0])
        f = interp1d(rows[valid], store_shifts[valid], kind='slinear',
                     bounds_error=False, fill_value=np.nan)
        out = f(rows)
        out[np.isnan(out)] = 0.
        return out

    @staticmethod
    def _apply_shift(prof_store, shifts, tmpx, rows):
        '''Re-interpolate selected channels of prof_store onto a shifted x-axis.'''
        for i in range(prof_store.shape[0]):
            tmpx2 = tmpx - shifts[i]
            for p in rows:
                col = prof_store[i, p, :]
                valid = np.isfinite(col) & np.isfinite(tmpx2)
                if valid.sum() < 2:
                    prof_store[i, p, :] = np.nan
                    continue
                try:
                    f = interp1d(tmpx2[valid], col[valid], kind='slinear',
                                 bounds_error=False, fill_value=np.nan)
                    prof_store[i, p, :] = f(tmpx)
                except Exception:
                    prof_store[i, p, :] = np.nan

    @staticmethod
    def _stack(prof_store, stack, plen, gate_row, med_rows, out_rows, min_cols=None):
        '''Median+std stack each profile over +/-stack along-strike neighbours.
        med_rows are source channels; out_rows are the median destination rows in
        the (n, 8, M) output (std goes to out_row+1). ``min_cols`` is the minimum
        number of well-populated columns required to accept a stack; it defaults
        to plen/1.5 (the full-profile threshold) but is reduced for the strain
        channels when strain is computed over only a central window.'''
        if min_cols is None:
            min_cols = plen // 1.5
        n, _, M = prof_store.shape
        stats = np.full((n, 8, M), np.nan, dtype=prof_store.dtype)
        for i in range(stack, n - stack):
            block_gate = prof_store[i - stack:i + stack, gate_row, :]
            track_nan = np.sum(np.isnan(block_gate), axis=0)
            if np.sum(track_nan < stack // 3) >= min_cols:
                for src, dst in zip(med_rows, out_rows):
                    block = prof_store[i - stack:i + stack, src, :]
                    stats[i, dst, :] = np.nanmedian(block, axis=0)
                    stats[i, dst + 1, :] = np.nanstd(block, axis=0)
        return stats

    def _second_reshift(self, strain_stats, disp_stats, prof_dist_pxls,
                        plen, stack, cap):
        '''Estimate and apply a finer global re-shift of the stacked profiles,
        aligning on the stacked peak shear. Shifts strain rows 0,2 and disp rows
        4,6. Returns the (smoothed) shift array (px).'''
        n = strain_stats.shape[0]
        w = smoothed_boxcar(plen, 8, 1)
        shifts2 = np.zeros(n)
        for ff in range(n):
            ysp = strain_stats[ff, 0, :]
            valid = ~np.isnan(ysp)
            if valid.sum() < 4:
                continue
            try:
                spline = UnivariateSpline(prof_dist_pxls[valid],
                                          (ysp * w)[valid], s=1e-9)
                xno = int(prof_dist_pxls.shape[0] * 1e3)
                x_fine = np.linspace(prof_dist_pxls.min(), prof_dist_pxls.max(), xno)
                y_fine = spline(x_fine)
                xs = (np.nanargmax(y_fine) / xno) * prof_dist_pxls.shape[0] \
                    - (prof_dist_pxls.shape[0] / 2.)
                if (xs >= cap) or (xs <= -cap):
                    xs = 0.
            except Exception:
                xs = 0.
            shifts2[ff] = xs

        valid = shifts2 != 0
        if valid.sum() == 0:
            self._print("  second re-shift: no valid shifts, skipping")
            return np.zeros(n)
        rows = np.arange(n)
        f = interp1d(rows[valid], shifts2[valid], kind='slinear',
                     bounds_error=False, fill_value=np.nan)
        shifts2 = f(rows)
        good = ~np.isnan(shifts2)
        shifts2[~good] = 0.
        shifts2 = gaussian_filter1d(shifts2, sigma=stack / 0.5)

        # apply: strain rows 0,2 ; disp rows 4,6
        for ff in range(n):
            sh = shifts2[ff]
            for arr, row in ((strain_stats, 0), (strain_stats, 2),
                             (disp_stats, 4), (disp_stats, 6)):
                col = arr[ff, row, :]
                valid = np.isfinite(col)
                if valid.sum() < 2:
                    continue
                try:
                    g = interp1d(prof_dist_pxls[valid] - sh, col[valid],
                                 kind='slinear', bounds_error=False, fill_value=np.nan)
                    # samples shifted outside the data range stay NaN (a 0 fill
                    # would fabricate zero displacement at the profile ends)
                    arr[ff, row, :] = g(prof_dist_pxls)
                except Exception:
                    pass
        return shifts2

    def _analyse_profile(self, ff, strain_stats, disp_stats, prof_dist_pxls,
                         plen, xres, background_limit, near_field_limit,
                         far_field_limit, stdthr, strain_plen=None):
        '''Per-profile detrend / fault-zone-width / offset analysis. Returns
        (result_store [len 42], result_shifted [3, M] = shear/dil/parallel).

        ``strain_plen`` is the half-width (px) over which strain exists; when it is
        smaller than plen the background-strain detrend band and the minimum
        valid-point check are pulled inside the strain window so they still have
        data. Defaults to plen (full-profile strain, original behaviour).'''
        if strain_plen is None:
            strain_plen = plen
        M = prof_dist_pxls.shape[0]
        result = np.full(42, np.nan)
        shifted = np.zeros((3, M))

        ysp = strain_stats[ff, 0, :].copy()    # shear
        yspd = strain_stats[ff, 2, :].copy()   # dilatation
        yspss = disp_stats[ff, 6, :].copy()    # fault-parallel displacement
        xsp = prof_dist_pxls

        w = smoothed_boxcar(plen, 50, 1)
        valid = (np.isnan(ysp) + (ysp == 0)) < 1
        valid2 = (np.isnan(yspd) + (yspd == 0)) < 1
        valid3 = (np.isnan(yspss) + (yspss == 0)) < 1

        # require enough valid strain points: half the profile for full strain,
        # or half the (smaller) strain window when strain is restricted.
        min_valid = min(int(ysp.shape[0] * 0.5), strain_plen)
        if valid.sum() < min_valid:
            return result, shifted

        xsp2 = xsp[valid]
        yspd2 = yspd[valid2]
        yspss2 = yspss[valid3]

        threshold = 0.; threshold2 = 0.; r2 = np.nan
        x_min = np.nan; x_max = np.nan; fzw = np.nan
        x_min2 = np.nan; x_max2 = np.nan; fzw2 = np.nan

        # --- detrend background shear strain via RANSAC polynomial ---
        # when strain is restricted to a central window, pull the background band
        # (which lies OUTSIDE +/-background_limit) inward so it still has data.
        if strain_plen < plen:
            bg_lo = max(background_limit[0], -0.6 * strain_plen)
            bg_hi = min(background_limit[1], 0.6 * strain_plen)
        else:
            bg_lo, bg_hi = background_limit[0], background_limit[1]
        try:
            bg = (xsp2 <= bg_lo) | (xsp2 >= bg_hi)
            xx = xsp2[bg].reshape(-1, 1)
            yy = ysp[valid][bg].reshape(-1, 1)
            ransac_in = RANSACRegressor(min_samples=0.6,
                                        residual_threshold=0.2 * np.std(yy),
                                        max_trials=100, random_state=42)
            model = make_pipeline(PolynomialFeatures(3), ransac_in)
            model.fit(xx, yy)
            inlier = model.named_steps['ransacregressor'].inlier_mask_
            y_pred = model.predict(xx)
            r2 = r2_score(yy[inlier], y_pred[inlier])
            ysp_background = model.predict(xsp.reshape(-1, 1))
            ysp_detrend = ysp.reshape(-1, 1) - ysp_background
            std2 = np.nanstd(yy - model.predict(xx))
            threshold = stdthr * std2
            threshold2 = threshold
            ysp = np.squeeze(ysp_detrend)
        except Exception:
            self._print(f"  could not detrend strain, row {ff}")

        # spline fits of detrended shear (and dilatation) for FZW edges
        xno = int(xsp.shape[0] * 1e3)
        x_fine = np.linspace(xsp.min(), xsp.max(), xno)
        ysp2 = ysp[valid]
        try:
            spline = UnivariateSpline(xsp2, ysp2 * (w[valid]), s=1e-9)
            y_fine = spline(x_fine)
        except Exception:
            y_fine = np.zeros_like(x_fine)
        try:
            spline2 = UnivariateSpline(xsp2, yspd2 * (w[valid2]), s=1e-9)
            y2_fine = spline2(x_fine)
        except Exception:
            y2_fine = np.zeros_like(x_fine)

        try:
            x_shift = (np.nanargmax(y_fine) / xno) * xsp.shape[0] - (xsp.shape[0] / 2.)
        except Exception:
            x_shift = 0.

        shifted[0, :] = ysp
        shifted[1, :] = yspd
        shifted[2, :] = yspss

        # fault-zone width from detrended shear (threshold crossings each side)
        x_min, x_max, fzw = self._threshold_width(x_fine, y_fine, threshold)
        x_min2, x_max2, fzw2 = self._threshold_width(x_fine, y2_fine, threshold2)

        # --- offsets by robust regression of fault-parallel displacement ---
        offset, std_o, xl, xr = self._regress_offset(
            xsp2, yspss2, x_min, x_max, near_field_limit, xsp.shape[0])
        offset2, std_o2, xl2, xr2 = self._regress_offset(
            xsp2, yspss2, x_min, x_max, far_field_limit, xsp.shape[0])
        offset3, std_o3, xl3, xr3 = self._regress_offset(
            xsp2, yspss2, x_min2, x_max2, near_field_limit, xsp.shape[0])

        A_fit = mu_fit = sigma_fit = alpha_fit = b_fit = 0.
        r_squared = 0.
        # entries 0-4 (trace coords) are filled by the caller
        result = np.array([
            0., 0., 0., 0., 0.,
            offset, std_o, xl[0], xl[-1], xr[0], xr[-1],
            offset2, std_o2, xl2[0], xl2[-1], xr2[0], xr2[-1],
            offset3, std_o3, xl3[0], xl3[-1], xr3[0], xr3[-1],
            fzw * xres, x_min * xres, x_max * xres, threshold,
            fzw2 * xres, x_min2 * xres, x_max2 * xres, threshold2,
            np.nanmax(ysp), np.nanmax(yspd), x_shift, r2,
            A_fit, mu_fit, sigma_fit, alpha_fit, b_fit, r_squared,
        ], dtype=float)
        return result, shifted

    @staticmethod
    def _threshold_width(x_fine, y_fine, threshold):
        '''Distance from x=0 to the first threshold down-crossing on each side.'''
        try:
            left = np.where(x_fine <= 0)
            xl = (-x_fine[left])[::-1]; yl = (y_fine[left])[::-1]
            below = np.where(yl < threshold)[0]
            x_min = -xl[below[0]]
        except IndexError:
            x_min = np.nan
        try:
            right = np.where(x_fine >= 0)
            xr = (x_fine[right]); yr = (y_fine[right])
            below = np.where(yr < threshold)[0]
            x_max = xr[below[0]]
            fzw = x_max - x_min
        except IndexError:
            x_max = np.nan; fzw = np.nan
        return x_min, x_max, fzw

    @staticmethod
    def _regress_offset(xsp2, yspss2, x_min, x_max, limit, npred):
        '''Robust (RANSAC) linear fits to fault-parallel displacement in windows
        just outside the fault zone on each side; offset = right(0) - left(0).'''
        inner, outer = limit
        # right
        try:
            idx = np.where((xsp2 >= x_max + inner) & (xsp2 <= x_max + outer))
            x_right = xsp2[idx]; y_right = yspss2[idx]
            rt = np.std(y_right) * 3
            ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=rt)
            ransac.fit(x_right.reshape(-1, 1), y_right)
            xr_pred = np.linspace(0, x_right.max(), npred).reshape(-1, 1)
            yr_pred = ransac.predict(xr_pred)
            stdev_r = np.std(y_right - ransac.predict(x_right.reshape(-1, 1)))
        except (ValueError, IndexError):
            yr_pred = np.array([0., 0.]); stdev_r = 0.; x_right = np.array([0.])
        # left
        try:
            idx = np.where((xsp2 <= x_min - inner) & (xsp2 >= x_min - outer))
            x_left = xsp2[idx]; y_left = yspss2[idx]
            rt = np.std(y_left) * 3
            ransac2 = RANSACRegressor(estimator=LinearRegression(), residual_threshold=rt)
            ransac2.fit(x_left.reshape(-1, 1), y_left)
            xl_pred = np.linspace(x_left.min(), 0, npred).reshape(-1, 1)
            yl_pred = ransac2.predict(xl_pred)
            stdev_l = np.std(y_left - ransac2.predict(x_left.reshape(-1, 1)))
        except (ValueError, IndexError):
            yl_pred = np.array([0., 0.]); stdev_l = 0.; x_left = np.array([0.])

        offset = yr_pred[0] - yl_pred[-1]
        combined_std = np.sqrt(stdev_r**2 + stdev_l**2)
        return offset, combined_std, np.atleast_1d(x_left), np.atleast_1d(x_right)


    ##  Estimate spatial covariance for EW and NS components
    def compute_covariance(self, mask_box=None, function='exp', frac=0.01,
                           every=500., distmax=35000., tol=1e-10):

        '''
        Estimate spatial covariance parameters (sigma, lamda) independently for the EW
        and NS components from a random subsample of pixels outside the deforming area,
        then store the fitted parameters.

        Requires flat arrays — call flatten() or downsample() first.

        Algorithm:
          1. Optionally mask pixels inside mask_box (the deforming region).
          2. Randomly subsample frac * N pixels.
          3. Remove a bilinear ramp (ax + by + c + w*x*y) from each component.
          4. Compute empirical covariogram: 0.5 * mean(|d_i * d_j|) in distance bins.
          5. Fit  C(r) = sigma^2 * exp(-r/lamda) + sill  (or Gaussian equivalent).
          6. Store sigma_ew, lamda_ew, sigma_ns, lamda_ns.

        Kwargs:
            mask_box (list): [minlon, maxlon, minlat, maxlat] or list of such lists.
                Pixels in this region are excluded before sampling.
            function (str): 'exp' or 'gauss'.
            frac (float or int): Fraction of pixels to sample (0-1), or an integer count.
            every (float): Covariogram bin width (metres).
            distmax (float): Maximum separation distance for covariogram (metres).
            tol (float): Optimisation tolerance.

        Returns:
            self (modified in-place)
        '''

        assert self.x is not None, \
            "Call flatten() or downsample() before compute_covariance()."

        self._print("Computing spatial covariance...")

        x, y     = self.x.copy(), self.y.copy()
        ew_d     = self.ew_vals.copy()
        ns_d     = self.ns_vals.copy()

        # Exclude deforming area
        if mask_box is not None:
            boxes = [mask_box] if np.ndim(mask_box[0]) == 0 else list(mask_box)
            keep  = np.ones(len(ew_d), dtype=bool)
            for box in boxes:
                minlon, maxlon, minlat, maxlat = box
                in_box = ((self.lon >= minlon) & (self.lon <= maxlon) &
                          (self.lat >= minlat) & (self.lat <= maxlat))
                keep &= ~in_box
            x, y, ew_d, ns_d = x[keep], y[keep], ew_d[keep], ns_d[keep]

        N     = len(ew_d)
        Nsamp = int(min(frac, N)) if isinstance(frac, int) \
                else int(np.floor(frac * N))
        self._print(f"  Sampling {Nsamp} pixels from {N}")

        perm = np.random.permutation(N)[:Nsamp]
        xs, ys = x[perm], y[perm]

        # Detrend: ax + by + c + w*x*y
        G = np.column_stack([xs, ys, np.ones(Nsamp), xs * ys])

        ew_s = ew_d[perm].copy()
        pars, _, _, _ = np.linalg.lstsq(G, ew_s, rcond=None)
        ew_s -= G @ pars

        ns_s = ns_d[perm].copy()
        pars, _, _, _ = np.linalg.lstsq(G, ns_s, rcond=None)
        ns_s -= G @ pars

        # Pairwise distances (shared for both components)
        ii, jj = np.triu_indices(Nsamp, k=1)
        dis    = np.hypot(xs[ii] - xs[jj], ys[ii] - ys[jj])

        bins = np.arange(0., distmax, every)
        inds = np.digitize(dis, bins)

        def _fit_component(d_vals, label):
            dv = np.abs(d_vals[ii] * d_vals[jj])
            distance, covariogram = [], []
            for i in range(len(bins) - 1):
                uu = np.flatnonzero(inds == i)
                if len(uu):
                    distance.append(bins[i] + every / 2.)
                    covariogram.append(0.5 * np.mean(dv[uu]))
            distance    = np.array(distance)
            covariogram = np.array(covariogram)

            sill0 = float(np.mean(covariogram[-4:]))
            lam0  = self._estimate_lam0(distance, covariogram, sill0)
            sig0  = self._estimate_sig0(distance, covariogram, sill0, lam0)

            def _pred(t, sil, sig, lam):
                if function == 'exp':
                    return sig**2 * np.exp(-t / lam) + sil
                else:
                    return sig**2 * np.exp(-t**2 / (2. * lam**2)) + sil

            def _cost(m):
                return np.sum(np.abs(covariogram - _pred(distance, *m)))

            res = sp_opt.minimize(
                _cost, [sill0, sig0, lam0],
                method='SLSQP',
                bounds=[[0., np.inf], [0., np.inf], [0., np.inf]],
                tol=tol,
                options={'maxiter': 200, 'disp': False}
            )
            sill, sigma, lamda = res.x
            self._print(f"  {label}: Sill={sill:.6f}, Sigma={sigma:.6f}, Lambda={lamda:.1f} m")
            return sill, sigma, lamda, distance, covariogram

        sill_ew, sigma_ew, lamda_ew, dist_ew, cov_ew = _fit_component(ew_s, "EW")
        sill_ns, sigma_ns, lamda_ns, dist_ns, cov_ns = _fit_component(ns_s, "NS")

        self.sigma_ew = sigma_ew
        self.lamda_ew = lamda_ew
        self.sigma_ns = sigma_ns
        self.lamda_ns = lamda_ns
        self._cov_distance_ew    = dist_ew
        self._cov_covariogram_ew = cov_ew - sill_ew   # sill-corrected for plotting
        self._cov_distance_ns    = dist_ns
        self._cov_covariogram_ns = cov_ns - sill_ns
        self._cov_function       = function

        return self


    def _estimate_lam0(self, distance, covariogram, sill):
        x = distance[:4]
        y = sill - covariogram[:4]
        if len(x) < 2:
            return 5000.
        m   = np.polyfit(x, y, 1)
        lam = -m[1] / m[0] if m[0] != 0. else 5000.
        return max(float(lam), 100.)

    def _estimate_sig0(self, distance, covariogram, sill, lam):
        y   = np.clip(sill - covariogram, 0., None)
        exp = np.where(np.exp(-distance / lam) < 1e-30, 1e-30,
                       np.exp(-distance / lam))
        return float(np.sqrt(np.mean(y / exp)))


    ##  Build data covariance matrices for EW and NS
    def build_Cd(self, sigma_ew=None, lamda_ew=None,
                 sigma_ns=None, lamda_ns=None, function=None):

        '''
        Build N x N data covariance matrices for EW and NS and store in self.Cd_ew, self.Cd_ns.

          exp:   Cd[i,j] = sigma^2 * exp(-d_ij / lamda)
          gauss: Cd[i,j] = sigma^2 * exp(-d_ij^2 / (2 * lamda^2))

        Kwargs:
            sigma_ew, sigma_ns (float): Amplitudes. Default to self.sigma_ew/ns.
            lamda_ew, lamda_ns (float): Length scales (m). Default to self.lamda_ew/ns.
            function (str): 'exp' or 'gauss'. Defaults to self._cov_function if set.

        Returns:
            self (modified in-place)
        '''

        if sigma_ew is None: sigma_ew = self.sigma_ew
        if lamda_ew is None: lamda_ew = self.lamda_ew
        if sigma_ns is None: sigma_ns = self.sigma_ns
        if lamda_ns is None: lamda_ns = self.lamda_ns
        if function  is None: function = getattr(self, '_cov_function', 'exp')

        assert sigma_ew is not None and lamda_ew is not None, \
            "sigma_ew and lamda_ew must be provided or computed via compute_covariance()"
        assert sigma_ns is not None and lamda_ns is not None, \
            "sigma_ns and lamda_ns must be provided or computed via compute_covariance()"

        N = len(self.x)
        self._print(f"Building Cd ({N}x{N}) for EW and NS")

        dist = np.sqrt((self.x[:, None] - self.x[None, :])**2 +
                       (self.y[:, None] - self.y[None, :])**2)

        if function == 'exp':
            self.Cd_ew = sigma_ew**2 * np.exp(-dist / lamda_ew)
            self.Cd_ns = sigma_ns**2 * np.exp(-dist / lamda_ns)
        else:
            self.Cd_ew = sigma_ew**2 * np.exp(-dist**2 / (2. * lamda_ew**2))
            self.Cd_ns = sigma_ns**2 * np.exp(-dist**2 / (2. * lamda_ns**2))

        return self


    ##  Covariogram diagnostic plot
    def plot_covariance(self, axes=None):

        '''
        Plot the empirical covariogram and fitted model for EW and NS components.

        Kwargs:
            axes (array of Axes or None): Two Axes to plot on; new figure created if None.

        Returns:
            (fig, (ax_ew, ax_ns)) if a new figure was created, otherwise (ax_ew, ax_ns).
        '''

        assert hasattr(self, '_cov_distance_ew'), \
            "Run compute_covariance() before plotting the covariogram."

        created_fig = axes is None
        if created_fig:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')

        ax_ew, ax_ns = axes

        for ax, dist, cov, sigma, lamda, label in [
            (ax_ew, self._cov_distance_ew, self._cov_covariogram_ew,
             self.sigma_ew, self.lamda_ew, "EW"),
            (ax_ns, self._cov_distance_ns, self._cov_covariogram_ns,
             self.sigma_ns, self.lamda_ns, "NS"),
        ]:
            ax.plot(dist, cov, '.k', ms=8, label='Empirical')
            t = np.linspace(0., dist.max(), 300)
            fit = (sigma**2 * np.exp(-t / lamda) if self._cov_function == 'exp'
                   else sigma**2 * np.exp(-t**2 / (2. * lamda**2)))
            ax.plot(t, fit, '-r',
                    label=fr'Fit  $\sigma$={sigma:.4f}, $\lambda$={lamda:.0f} m')
            ax.axhline(0., color='gray', lw=0.7, ls='--')
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Covariance')
            ax.set_title(f'{label} covariance')
            ax.legend(fontsize=9)

        if created_fig:
            return fig, axes
        return axes


    ##  Compute Green's functions for EW and NS
    def compute_greens_functions(self, fault):

        '''
        Compute Green's functions for EW and NS surface displacement at each pixel location.

        For each fault patch, the east and north surface displacements are extracted
        directly from the 3-component GF array — no LOS projection is needed.

        Requires flat arrays — call flatten() or downsample() first.

        Args:
            fault: A Fault3d object with a compute_greens_functions() method that returns
                   an object with .gfs of shape (n_patches, 3, n_pts), where the three
                   rows correspond to [east, north, up] displacements.

        Returns:
            A copy of self with gfs_ew and gfs_ns set, each of shape (n_patches, n_pts).
        '''

        assert self.x is not None, \
            "Call flatten() or downsample() before compute_greens_functions()."

        _self = self._copy()
        pts = np.vstack((_self.x, _self.y))
        gfs = fault.compute_greens_functions(pts).gfs  # (2, n_patches, 3, n_pts)

        _self.gfs_ew = gfs[:, :, 0, :]   # east component:  (2, n_patches, n_pts)
        _self.gfs_ns = gfs[:, :, 1, :]   # north component: (2, n_patches, n_pts)

        return _self


    ## Helper method to easily plot the optical data using Matplotlib. Provide axes or they will be generated.
    def plot(self, ew=True, ns=True, title=None, fault=None, profiles=None,
             profile_swathe_width=None, cmap=cmc.vik, xlim=None, ylim=None):

        '''
        Plots a nice looking map of optical data. Will plot on provided axes if given, else will produce and return a figure.

        Kwargs:
            ew (bool or Mpl axes): does not plot EW displacement if false, creates axes and plots if True, plots on provided axes if given.
            ns (bool or Mpl axes): does not plot NS displacement if false, creates axes and plots if True, plots on provided axes if given.
            title (str): Figure title
            profile_swathe_width (float or None): if given, each profile is drawn as a thin
                centre line plus a translucent swathe of this full width in metres (buffered
                in the profile's UTM CRS before reprojection to EPSG:4326).
        '''
        from matplotlib.patches import FancyBboxPatch

        _self = self._copy()

        ####    SET UP FIG, AXES   ####
        _self._print("Plotting...")
        n_axs = sum((bool(ew), bool(ns)))   # count number of axes to plot

        if ew==True or ns==True:   # create figure if not supplied with axes
            fig = plt.figure(figsize=(1.5+5.*n_axs, 5), layout="constrained")
            if title is not None: fig.suptitle(title)

        if isinstance(ew, mpl.axes.Axes):    # if supplied EW axis to plot, save reference
            ax_ew = ew
        elif ew:   # else if plotting EW axis, create axis and add to figure
            ax_ew = fig.add_subplot(101+10*n_axs)

        if isinstance(ns, mpl.axes.Axes):   # if supplied NS axis to plot, save reference
            ax_ns = ns
        elif ns:   # else if plotting NS axis, create axis and add to figure
            ax_ns = fig.add_subplot(100+11*n_axs)


        ####    PLOT THE DATA    ####
        axs = []
        _cb_data = []  # (label, image_artist, axis)

        # EW data
        if ew:
            ew_latlon = _self.ew.rio.reproject("EPSG:4326")
            if xlim is not None and ylim is not None:
                ew_latlon = ew_latlon.sel(x=slice(xlim[0], xlim[1]), y=slice(ylim[1], ylim[0]))
            im_ew = ew_latlon.plot(ax=ax_ew, cmap=cmap, add_colorbar=False)
            ax_ew.set_title("EW displacement" if ns else "")
            axs.append(ax_ew)
            _cb_data.append(("East-west displacement (m)", im_ew, ax_ew))

        # NS data
        if ns:
            ns_latlon = _self.ns.rio.reproject("EPSG:4326")
            if xlim is not None and ylim is not None:
                ns_latlon = ns_latlon.sel(x=slice(xlim[0], xlim[1]), y=slice(ylim[1], ylim[0]))
            im_ns = ns_latlon.plot(ax=ax_ns, cmap=cmap, add_colorbar=False)
            ax_ns.set_title("NS displacement" if ew else "")
            axs.append(ax_ns)
            _cb_data.append(("North-south displacement (m)", im_ns, ax_ns))


        ####    CONFIGURE PLOTS    ####
        for ax in axs:
            ax.set_xlabel("Latitude (˚)")
            ax.set_ylabel("Longitude (˚)")

            if fault is not None:
                trace = fault.trace.to_crs("EPSG:4326")
                trace.plot(ax=ax, color="black", linewidth=2.)
            if profiles is not None:
                for p in profiles:
                    if profile_swathe_width is not None:
                        buf = p.trace.copy()
                        buf["geometry"] = buf.geometry.buffer(profile_swathe_width / 2.)
                        buf.to_crs("EPSG:4326").plot(ax=ax, color="deeppink", alpha=0.25, linewidth=0)
                    p.trace.to_crs("EPSG:4326").plot(ax=ax, color="deeppink", linewidth=1.)

            # Scale bar (lower left), auto-sized to ~20% of plot width
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            lat_c = (y_min + y_max) / 2
            km_per_deg = np.cos(np.radians(lat_c)) * 111.32
            bar_km_target = 0.2 * (x_max - x_min) * km_per_deg
            nice_km = [1, 2, 5, 10, 20, 50, 100, 200, 500]
            bar_km = min(nice_km, key=lambda v: abs(v - bar_km_target))
            bar_deg = bar_km / km_per_deg
            x_range, y_range = x_max - x_min, y_max - y_min
            bar_x0 = x_min + 0.05 * x_range
            bar_y0 = y_min + 0.05 * y_range
            bar_tick_h = 0.01 * y_range
            ax.plot([bar_x0, bar_x0 + bar_deg], [bar_y0, bar_y0],
                    color="black", linewidth=2.5, zorder=5, solid_capstyle="butt")
            ax.plot([bar_x0, bar_x0], [bar_y0, bar_y0 + bar_tick_h], color="black", linewidth=2.5, zorder=5)
            ax.plot([bar_x0 + bar_deg, bar_x0 + bar_deg], [bar_y0, bar_y0 + bar_tick_h], color="black", linewidth=2.5, zorder=5)
            ax.text(bar_x0 + bar_deg / 2, bar_y0 + 0.012 * y_range,
                    f"{bar_km:.0f} km", ha="center", va="bottom", fontsize=10., zorder=5)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

        # Inset colorbars (upper right) with white background
        for cb_label, im, ax in _cb_data:
            bg = FancyBboxPatch((0.6, 0.81), 0.36, 0.16,
                                boxstyle="square,pad=0.01",
                                transform=ax.transAxes,
                                facecolor="white", edgecolor="lightgray", linewidth=0.5,
                                zorder=4, clip_on=False)
            ax.add_patch(bg)
            cax = ax.inset_axes([0.62, 0.91, 0.32, 0.04])
            cax.set_zorder(5)
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(cb_label, fontsize=9., labelpad=2)
            cax.tick_params(labelsize=7)


        ####    RETURN FIG, AXS    ####
        res = []
        if fig: res.append(fig)
        if ew and ns: res.append((ax_ew, ax_ns))
        elif ew: res.append(ax_ew)
        elif ns: res.append(ax_ns)

        _self._print("... plotted.")
        return res

    ## Helper methods to plot only EW or NS
    def plot_ew(self, **kwargs):
        return self.plot(ns=False, **kwargs)
    def plot_ns(self, **kwargs):
        return self.plot(ew=False, **kwargs)
