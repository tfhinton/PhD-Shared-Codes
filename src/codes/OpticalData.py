###   Imports
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
import cmcrameri.cm as cmc
import pyproj
import scipy.optimize as sp_opt
import scipy.spatial.distance as sp_dist
from shapely.geometry import Point
from shapely.ops import unary_union


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
                fault = fault.trace_to_crs("EPSG:4326")
                fault.trace.plot(ax=ax, color="black", linewidth=2.)
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
