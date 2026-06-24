###   Imports
import numpy as np
import pyproj
import scipy.optimize as sp_opt
import scipy.spatial.distance as sp_dist
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import copy
from netCDF4 import Dataset as _NetCDF
from shapely.geometry import Point, LineString
from shapely.ops import unary_union


###   InSAR class
class InSAR:

    '''
    Container and processing class for unwrapped InSAR LOS displacement data.

    Pixel positions are stored in two coordinate systems:
      - Geographic: lon (degrees), lat (degrees)
      - Projected:  x, y (metres, UTM)

    Properties:
        lon, lat (ndarray): Pixel coordinates in degrees.
        x, y (ndarray): Pixel coordinates in metres (UTM).
        vel (ndarray): LOS displacement (native GRD units).
        err (ndarray): Per-pixel uncertainty.
        los (ndarray): LOS unit vectors, shape (N, 3), columns [east, north, up].
        Cd (ndarray or None): Data covariance matrix (N x N).
        sigma, lamda (float or None): Fitted covariance amplitude and length scale (m).
        verbose (bool)
    '''

    ##  Initialise
    def __init__(self, verbose=True, **import_grd_kwargs):

        '''
        Kwargs:
            verbose (bool): Print progress messages.
            **import_grd_kwargs: Passed to import_grd() if provided.
        '''

        self.verbose = verbose

        self.lon   = None
        self.lat   = None
        self.x     = None
        self.y     = None
        self.vel   = None
        self.err   = None
        self.los   = None
        self.Cd    = None
        self.sigma = None
        self.lamda = None

        if import_grd_kwargs:
            self.import_grd(**import_grd_kwargs)


    ##  Helpers
    def _print(self, *args):
        if self.verbose: print(*args)

    def _copy(self):
        return copy.deepcopy(self)


    ##  Coordinate transforms (internal)
    def _ll2xy(self, lon, lat):
        '''Convert lon/lat (degrees, 0-360 or -180-180) to UTM metres.'''
        lon_std = np.where(np.asarray(lon) > 180., lon - 360., lon)
        return self._proj(lon_std, np.asarray(lat, dtype=float))

    def _xy2ll(self, x, y):
        '''Convert UTM metres to lon/lat degrees.'''
        return self._proj(x, y, inverse=True)


    ##  Internal pixel selection
    def _keep_pixels(self, u):
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x   = self.x[u]
        self.y   = self.y[u]
        self.vel = self.vel[u]
        if self.err is not None:
            self.err = self.err[u]
        if self.los is not None:
            self.los = self.los[u]
        if self.Cd is not None:
            self.Cd = self.Cd[np.ix_(u, u)]


    ##  Read a GMT .grd (NetCDF) file
    @staticmethod
    def _read_grd(filepath):
        '''Return (z_array, lon_1d, lat_1d) from a .grd file.'''
        with _NetCDF(filepath, 'r') as nc:
            z = np.array(nc.variables['z'][:])

            if 'x' in nc.variables:
                lons = np.array(nc.variables['x'][:])
                lats = np.array(nc.variables['y'][:])
            elif 'lon' in nc.variables:
                lons = np.array(nc.variables['lon'][:])
                lats = np.array(nc.variables['lat'][:])
            else:
                nlon = int(nc.variables['dimension'][0])
                nlat = int(nc.variables['dimension'][1])
                lons = np.linspace(float(nc.variables['x_range'][0]),
                                   float(nc.variables['x_range'][1]), nlon)
                lats = np.linspace(float(nc.variables['y_range'][1]),
                                   float(nc.variables['y_range'][0]), nlat)

        return z, lons, lats


    ##  Import unwrapped GRD file
    def import_grd(self, unw_filepath, los_filepaths=None, utm_zone=None, bbox=None):

        '''
        Read an unwrapped InSAR displacement GRD file and (optionally) per-pixel
        LOS unit-vector GRD files.

        Args:
            unw_filepath (str): Path to unwrapped displacement GRD.

        Kwargs:
            los_filepaths (list of str): [east_grd, north_grd, up_grd] paths.
                If None, defaults to [1, 0, 0] (east-only LOS).
            utm_zone (int or str or None): UTM zone number.  If None, auto-detected
                from the data's mean longitude.
            bbox (list): [minlon, maxlon, minlat, maxlat] geographic subset applied
                immediately after loading.

        Returns:
            self
        '''

        self._print(f"Importing GRD: {unw_filepath}")

        z, lons_1d, lats_1d = self._read_grd(unw_filepath)
        Lon, Lat = np.meshgrid(lons_1d, lats_1d)

        vel_flat = z.flatten() if z.ndim == 2 else np.array(z)
        lon_flat = Lon.flatten()
        lat_flat = Lat.flatten()

        finite = np.isfinite(vel_flat)

        if los_filepaths is not None:
            los_components = []
            for fp in los_filepaths:
                comp, _, _ = self._read_grd(fp)
                los_components.append(comp.flatten()[finite])
            los_arr = np.column_stack(los_components)
        else:
            los_arr = np.zeros((int(finite.sum()), 3))
            los_arr[:, 0] = 1.

        self.vel = vel_flat[finite]
        self.lon = lon_flat[finite]
        self.lat = lat_flat[finite]
        self.err = np.zeros(self.vel.shape)
        self.los = los_arr

        # Set up UTM projection
        if utm_zone is None:
            mean_lon = float(np.mean(self.lon))
            mean_lon_std = mean_lon - 360. if mean_lon > 180. else mean_lon
            utm_zone = int((mean_lon_std + 180.) / 6.) + 1
            self._print(f"  Auto-detected UTM zone: {utm_zone}")
        south = float(np.mean(self.lat)) < 0.
        self._proj = pyproj.Proj(proj='utm', zone=int(utm_zone),
                                 south=south, ellps='WGS84')

        self.x, self.y = self._ll2xy(self.lon, self.lat)

        self._print(f"  Loaded {len(self.vel)} pixels")

        if bbox is not None:
            self.select_pixels(*bbox)

        return self


    ##  Geographic bounding box selection
    def select_pixels(self, minlon, maxlon, minlat, maxlat):

        '''
        Keep only pixels within the specified geographic bounding box.

        Args:
            minlon, maxlon (float): Longitude bounds (same convention as loaded data).
            minlat, maxlat (float): Latitude bounds (degrees).

        Returns:
            self
        '''

        u = np.flatnonzero(
            (self.lat > minlat) & (self.lat < maxlat) &
            (self.lon > minlon) & (self.lon < maxlon)
        )
        self._keep_pixels(u)
        self._print(f"  After bbox selection: {len(self.vel)} pixels")
        return self


    ##  Remove NaN pixels
    def check_nans(self):

        '''
        Remove any pixel that has a NaN in vel, err, or any LOS component.

        Returns:
            self
        '''

        bad = np.isnan(self.vel)
        if self.err is not None:
            bad |= np.isnan(self.err)
        if self.los is not None:
            bad |= np.any(np.isnan(self.los), axis=1)
        self._keep_pixels(np.flatnonzero(~bad))
        self._print(f"  After NaN check: {len(self.vel)} pixels")
        return self


    ##  Estimate spatial covariance
    def compute_covariance(self, mask_box=None, function='exp', frac=0.01,
                           every=500., distmax=35000., tol=1e-10):

        '''
        Estimate spatial covariance parameters (sigma, lamda) from a random
        subsample of pixels outside the deforming area, then build Cd.

        Algorithm:
          1. Optionally mask pixels inside mask_box (the deforming region).
          2. Randomly subsample frac * N pixels.
          3. Remove a bilinear ramp (ax + by + c + w*x*y) to detrend.
          4. Compute the empirical covariogram: 0.5 * mean(|d_i * d_j|) in distance bins.
          5. Fit  C(r) = sigma^2 * exp(-r/lamda) + sill  (or Gaussian equivalent).
          6. Call build_Cd() with the fitted parameters.

        Kwargs:
            mask_box (list): [minlon, maxlon, minlat, maxlat] or list of such lists.
                Pixels in this region are excluded before sampling.
            function (str): 'exp' or 'gauss'.
            frac (float or int): Fraction of pixels to sample (0-1), or an integer count.
            every (float): Covariogram bin width (metres).
            distmax (float): Maximum separation distance for covariogram (metres).
            tol (float): Optimisation tolerance.

        Returns:
            (sigma, lamda) tuple
        '''

        self._print("Computing spatial covariance...")

        x, y, d = self.x.copy(), self.y.copy(), self.vel.copy()

        # Exclude deforming area
        if mask_box is not None:
            boxes = [mask_box] if np.ndim(mask_box[0]) == 0 else list(mask_box)
            keep  = np.ones(len(d), dtype=bool)
            for box in boxes:
                minlon, maxlon, minlat, maxlat = box
                in_box = ((self.lon >= minlon) & (self.lon <= maxlon) &
                          (self.lat >= minlat) & (self.lat <= maxlat))
                keep &= ~in_box
            x, y, d = x[keep], y[keep], d[keep]

        N     = len(d)
        Nsamp = int(min(frac, N)) if isinstance(frac, int) \
                else int(np.floor(frac * N))
        self._print(f"  Sampling {Nsamp} pixels from {N}")

        perm       = np.random.permutation(N)[:Nsamp]
        xs, ys, ds = x[perm], y[perm], d[perm]

        # Detrend: ax + by + c + w*x*y
        G = np.column_stack([xs, ys, np.ones(Nsamp), xs * ys])
        pars, _, _, _ = np.linalg.lstsq(G, ds, rcond=None)
        ds = ds - G @ pars

        # Pairwise amplitude products for upper triangle
        ii, jj = np.triu_indices(Nsamp, k=1)
        dis = np.hypot(xs[ii] - xs[jj], ys[ii] - ys[jj])
        dv  = np.abs(ds[ii] * ds[jj])

        # Bin into covariogram
        bins = np.arange(0., distmax, every)
        inds = np.digitize(dis, bins)
        distance, covariogram = [], []
        for i in range(len(bins) - 1):
            uu = np.flatnonzero(inds == i)
            if len(uu):
                distance.append(bins[i] + every / 2.)
                covariogram.append(0.5 * np.mean(dv[uu]))

        distance    = np.array(distance)
        covariogram = np.array(covariogram)

        # Starting estimates
        sill0 = float(np.mean(covariogram[-4:]))
        lam0  = self._estimate_lam0(distance, covariogram, sill0)
        sig0  = self._estimate_sig0(distance, covariogram, sill0, lam0)

        # Fit:  covariogram ~ sig^2 * exp(-r/lam) + sill
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

        self.sigma = sigma
        self.lamda = lamda
        self._cov_distance    = distance
        self._cov_covariogram = covariogram - sill   # sill-corrected for plotting
        self._cov_function    = function
        self._cov_sill        = sill

        self._print(f"  Sill={sill:.6f}, Sigma={sigma:.6f}, Lambda={lamda:.1f} m")
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


    ##  Build data covariance matrix
    def build_Cd(self, sigma=None, lamda=None, function=None):

        '''
        Build the N x N data covariance matrix and store in self.Cd.

          exp:   Cd[i,j] = sigma^2 * exp(-d_ij / lamda)
          gauss: Cd[i,j] = sigma^2 * exp(-d_ij^2 / (2 * lamda^2))

        Kwargs:
            sigma (float): Amplitude. Defaults to self.sigma.
            lamda (float): Length scale (m). Defaults to self.lamda.
            function (str): 'exp' or 'gauss'. Defaults to self._cov_function if set.

        Returns:
            self
        '''

        if sigma    is None: sigma    = self.sigma
        if lamda    is None: lamda    = self.lamda
        if function is None: function = getattr(self, '_cov_function', 'exp')

        assert sigma is not None and lamda is not None, \
            "sigma and lamda must be provided or computed via compute_covariance()"

        self._print(f"Building Cd ({len(self.vel)}x{len(self.vel)}): "
                    f"sigma={sigma:.6f}, lambda={lamda:.1f} m")

        dist = np.sqrt((self.x[:, None] - self.x[None, :])**2 +
                       (self.y[:, None] - self.y[None, :])**2)

        if function == 'exp':
            self.Cd = sigma**2 * np.exp(-dist / lamda)
        else:
            self.Cd = sigma**2 * np.exp(-dist**2 / (2. * lamda**2))

        return self


    ##  Distance-based quadtree downsampling
    def downsample(self, faults, start_size=20000., min_size=2500., char_dist=1500., scaler=1.,
                   expo_dist=0.7, tolerance=0.005, reject_distance=500.):

        '''
        Distance-based quadtree downsampling.

        The image is covered by a regular grid of square blocks (size start_size metres).
        Each block is recursively subdivided into four equal sub-blocks while:
            (distance_to_fault - char_dist) < block_size ** expo_dist
        Subdivision stops when a block reaches min_size.  Pixels in each surviving
        block are averaged to yield one downsampled observation.

        Args:
            faults: A Fault3d object or list thereof.  Each must expose a .trace
                    GeoDataFrame with line geometry in the same CRS as the InSAR data.

        Kwargs:
            start_size (float): Initial block size (metres).
            min_size (float): Minimum block size (metres).
            char_dist (float): Characteristic distance for the subdivision criterion (m).
            expo_dist (float): Exponent applied to block size in the criterion.
            tolerance (float): Minimum fraction of block area that must contain data.
            reject_distance (float): Delete downsampled pixels closer than this
                distance to any fault trace after averaging (m).  Set 0 to skip.

        Returns:
            InSAR: A new downsampled InSAR instance (self is unchanged).
        '''

        if not isinstance(faults, list):
            faults = [faults]

        # Merge all fault trace geometries into a single shapely geometry
        fault_geom = unary_union([f.trace.geometry.unary_union for f in faults])

        # Estimate pixel spacing from a small probe subset
        n_probe      = min(1000, len(self.x))
        probe        = np.column_stack([self.x[:n_probe], self.y[:n_probe]])
        dmat         = sp_dist.cdist(probe, probe)
        np.fill_diagonal(dmat, np.inf)
        pixel_spacing = float(dmat.min())
        pixel_area    = pixel_spacing**2

        # Initial grid of blocks covering the data extent
        xmin = np.floor(self.x.min())
        xmax = np.floor(self.x.max()) + 1.
        ymin = np.floor(self.y.min())
        ymax = np.floor(self.y.max()) + 1.

        # Block convention:  [[x_left, y_top], [x_right, y_top],  [x_right, y_bot], [x_left,  y_bot]]
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

        dists = []
        def _dist_to_fault(b):
            res = min(Point(c).distance(fault_geom) for c in b)
            dists.append(res)
            return res

        # Iterative subdivision
        do_subdivide = True
        it = 0
        while do_subdivide:
            do_subdivide = False
            new_blocks = []
            for b in blocks:
                sz = _size(b)
                if sz > min_size and scaler*(_dist_to_fault(b) - char_dist) < sz**expo_dist:
                    new_blocks.extend(_cut4(b))
                    do_subdivide = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            it += 1
            self._print(f"  Downsample iter {it}: {len(blocks)} blocks")

        # Average pixels within each surviving block
        out = self._copy()
        vel_list = []; err_list = []; los_list = []
        x_list   = []; y_list   = []; lon_list = []; lat_list = []
        kept_blocks = []

        for b in blocks:
            x_left, y_top = b[0]
            x_right, y_bot = b[2]
            inside = ((self.x >= x_left) & (self.x < x_right) &
                      (self.y >= y_bot)  & (self.y < y_top))
            n = int(inside.sum())
            if n == 0:
                continue
            if (n * pixel_area) / (_size(b)**2) < tolerance:
                continue

            vel_list.append(float(np.mean(self.vel[inside])))

            if self.err is not None and np.any(self.err[inside] != 0.):
                err_list.append(float(np.sqrt(np.sum(self.err[inside]**2)) / n))
            else:
                err_list.append(float(np.std(self.vel[inside])))

            los_mean = np.mean(self.los[inside], axis=0)
            los_mean /= np.linalg.norm(los_mean)
            los_list.append(los_mean)

            xc = float(np.mean(self.x[inside]))
            yc = float(np.mean(self.y[inside]))
            lonc, latc = self._xy2ll(xc, yc)
            x_list.append(xc);   y_list.append(yc)
            lon_list.append(lonc); lat_list.append(latc)
            kept_blocks.append(b)

        out.vel = np.array(vel_list)
        out.err = np.array(err_list)
        out.los = np.array(los_list)
        out.x   = np.array(x_list)
        out.y   = np.array(y_list)
        out.lon = np.array(lon_list)
        out.lat = np.array(lat_list)
        out.Cd  = None
        out._blocks = kept_blocks

        self._print(f"  Downsampled: {len(self.vel)} -> {len(out.vel)} pixels")

        if reject_distance > 0.:
            out._reject_near_fault(fault_geom, reject_distance)

        return out


    def _reject_near_fault(self, fault_geom, distance):
        '''Delete pixels within distance metres of the fault trace.'''
        d = np.array([Point(xi, yi).distance(fault_geom)
                      for xi, yi in zip(self.x, self.y)])
        self._keep_pixels(np.flatnonzero(d > distance))
        self._print(f"  After fault rejection ({distance} m): {len(self.vel)} pixels")


    ##  Map plot
    def plot(self, ax=None, cmap=cmc.vik, title=None, fault=None, vlim=None,
             arrows=False, arrow_stride=1, arrow_scale=None, arrow_color='k'):

        '''
        Scatter-plot LOS displacement in geographic coordinates.

        Kwargs:
            ax (Axes or None): Axes to plot on; new figure created if None.
            cmap: Matplotlib colormap.
            title (str): Axes title.
            fault: Fault3d object with a .trace GeoDataFrame (UTM CRS), or None.
            vlim (tuple): (vmin, vmax) colour limits; auto-scaled if None.
            arrows (bool): Overlay quiver arrows showing the horizontal projection
                of the LOS displacement vector (vel * [east, north]).  Default False.
            arrow_stride (int): Plot one arrow every arrow_stride pixels (useful for
                dense full-resolution data).  Default 1 (every pixel).
            arrow_scale (float or None): Passed to ax.quiver() as ``scale``; larger
                values make arrows shorter.  None lets matplotlib auto-scale.
            arrow_color (str): Arrow colour.  Default 'k'.

        Returns:
            (fig, ax) if a new figure was created, otherwise ax.
        '''

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')

        vmin, vmax = (None, None) if vlim is None else vlim
        sc = ax.scatter(self.lon, self.lat, c=self.vel, cmap=cmap,
                        s=1, vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(sc, ax=ax, label='LOS displacement (m)', shrink=0.8)

        if fault is not None and hasattr(fault, 'trace'):
            for geom in fault.trace.geometry:
                lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)
                for line in lines:
                    coords = np.array(line.coords)
                    lons, lats = self._proj(coords[:, 0], coords[:, 1], inverse=True)
                    ax.plot(lons, lats, 'k-', lw=1.5)

        if arrows and self.los is not None:
            idx = np.arange(0, len(self.vel), max(1, int(arrow_stride)))
            lat_rad = np.radians(self.lat[idx])
            # Convert physical displacement (m) to approximate degree offsets
            u_deg = (self.vel[idx] * self.los[idx, 0]) / (np.cos(lat_rad) * 111320.)
            v_deg = (self.vel[idx] * self.los[idx, 1]) / 111320.
            ax.quiver(self.lon[idx], self.lat[idx], u_deg, v_deg,
                      scale=arrow_scale, color=arrow_color, angles='xy')

        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        if title is not None:
            ax.set_title(title)

        # Scalebar
        x_span_km = (self.x.max() - self.x.min()) / 1000.
        approx_km = x_span_km * 0.2
        mag = 10. ** np.floor(np.log10(max(approx_km, 1e-9)))
        bar_km = next(f * mag for f in [1, 2, 5, 10] if f * mag >= approx_km * 0.5)
        mean_lat_rad = np.radians(np.mean(self.lat))
        bar_deg = bar_km / (np.cos(mean_lat_rad) * 111.32)
        lon_range = self.lon.max() - self.lon.min()
        lat_range = self.lat.max() - self.lat.min()
        sb_x = self.lon.min() + 0.05 * lon_range
        sb_y = self.lat.min() + 0.04 * lat_range
        ax.plot([sb_x, sb_x + bar_deg], [sb_y, sb_y], 'k-', lw=3,
                solid_capstyle='butt', transform=ax.transData)
        ax.text(sb_x + bar_deg / 2., sb_y + 0.01 * lat_range,
                f'{bar_km:.0f} km', ha='center', va='bottom', fontsize=8)

        if created_fig:
            return fig, ax
        return ax


    ##  Covariogram diagnostic plot
    def plot_covariance(self, ax=None):

        '''
        Plot the  covariogram and fitted model.

        Kwargs:
            ax (Axes or None): Axes to plot on; new figure created if None.

        Returns:
            (fig, ax)
        '''

        assert hasattr(self, '_cov_distance'), \
            "Run compute_covariance() before plotting the covariogram."

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')

        ax.plot(self._cov_distance, self._cov_covariogram,
                '.k', ms=8, label='Empirical')

        t = np.linspace(0., self._cov_distance.max(), 300)
        fit = (self.sigma**2 * np.exp(-t / self.lamda) if self._cov_function == 'exp'
               else self.sigma**2 * np.exp(-t**2 / (2. * self.lamda**2)))
        ax.plot(t, fit, '-r',
                label=fr'Fit  $\sigma$={self.sigma:.4f}, $\lambda$={self.lamda:.0f} m')

        ax.axhline(0., color='gray', lw=0.7, ls='--')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Covariance')
        ax.legend(fontsize=9)

        if created_fig:
            return fig, ax
        return ax
    
    def compute_greens_functions(self, fault):
        _self = self._copy()
        pts = np.vstack((_self.x, _self.y))
        gfs = fault.compute_greens_functions(pts).gfs  # (2, n_patches, 3, n_pts)

        # Project each slip component (SS, DS) to LOS: (2, n_patches, n_pts)
        _self.gfs = (gfs[:, :, 0, :] * _self.los[:, 0] +
                     gfs[:, :, 1, :] * _self.los[:, 1] +
                     gfs[:, :, 2, :] * _self.los[:, 2])
        return _self


