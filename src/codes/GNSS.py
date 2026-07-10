###   Imports
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import copy
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


###   GNSS class
class GNSS:

    '''
    Container and processing class for co-seismic GNSS offset data.

    Station positions are stored in two coordinate systems:
      - Geographic: lon (degrees), lat (degrees)
      - Projected:  x, y (metres, UTM)

    Horizontal displacement components (east, north) and their 1-sigma
    uncertainties are stored per station.  Vertical is loaded but not
    used in covariance or Green's functions.

    The data vector is interleaved:  d = [de_0, dn_0, de_1, dn_1, ...]
    so that Cd and gfs rows/columns align with this ordering.

    Properties:
        sta  (ndarray): Station name strings.
        lon, lat (ndarray): Station coordinates in degrees.
        x, y (ndarray): Station coordinates in metres (UTM).
        de, dn (ndarray): East and north displacement (m).
        sde, sdn (ndarray): 1-sigma uncertainties on de, dn (m).
        Cd (ndarray or None): (2N x 2N) diagonal data covariance matrix.
        gfs (ndarray or None): Green's functions, shape (n_patches, 2N).
        verbose (bool)
    '''

    ##  Initialise
    def __init__(self, verbose=True, **import_kwargs):

        '''
        Kwargs:
            verbose (bool): Print progress messages.
            **import_kwargs: Passed to import_data() if provided.
        '''

        self.verbose = verbose

        self.sta  = None
        self.lon  = None
        self.lat  = None
        self.x    = None
        self.y    = None
        self.de   = None
        self.dn   = None
        self.sde  = None
        self.sdn  = None
        self.Cd   = None
        self.gfs  = None

        if import_kwargs:
            self.import_data(**import_kwargs)


    ##  Helpers
    def _print(self, *args):
        if self.verbose: print(*args)

    def _copy(self):
        return copy.deepcopy(self)


    ##  Internal station selection
    def _keep_stations(self, u):
        self.sta = self.sta[u]
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x   = self.x[u]
        self.y   = self.y[u]
        self.de  = self.de[u]
        self.dn  = self.dn[u]
        self.sde = self.sde[u]
        self.sdn = self.sdn[u]
        if self.Cd is not None:
            self.Cd = self.Cd[np.ix_(u, u)]


    ##  Coordinate transforms
    def _ll2xy(self, lon, lat):
        '''Convert lon/lat (degrees) to UTM metres.'''
        lon_std = np.where(np.asarray(lon) > 180., lon - 360., lon)
        return self._proj(lon_std, np.asarray(lat, dtype=float))

    def _xy2ll(self, x, y):
        '''Convert UTM metres to lon/lat degrees.'''
        return self._proj(x, y, inverse=True)


    ##  Import data file
    def import_data(self, filepath, utm_zone=None, fault=None, max_dist=None):

        '''
        Read a space-delimited GNSS offset file.

        Expected columns (after a two-line header):
            Sta  Lon  Lat  de(m)  dn(m)  du(m)  sde(m)  sdn(m)  sdu(m)

        Args:
            filepath (str): Path to the GNSS offsets file.

        Kwargs:
            utm_zone (int or None): UTM zone number.  Auto-detected if None.
            fault: GeoDataFrame (or object with a .geometry column) whose geometry
                is in the same UTM CRS as the station coordinates.  Used together
                with max_dist to reject distant stations.
            max_dist (float or None): If given, discard any station more than
                max_dist metres from the nearest fault geometry.

        Returns:
            self
        '''

        self._print(f"Importing GNSS: {filepath}")

        sta, lon, lat = [], [], []
        de, dn = [], []
        sde, sdn = [], []

        with open(filepath, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('Sta') or line.startswith('='):
                    continue
                parts = line.split()
                sta.append(parts[0])
                lon.append(float(parts[1]))
                lat.append(float(parts[2]))
                de.append(float(parts[3]))
                dn.append(float(parts[4]))
                sde.append(float(parts[6]))
                sdn.append(float(parts[7]))

        self.sta = np.array(sta)
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.de  = np.array(de)
        self.dn  = np.array(dn)
        self.sde = np.array(sde)
        self.sdn = np.array(sdn)

        if utm_zone is None:
            mean_lon = float(np.mean(self.lon))
            mean_lon_std = mean_lon - 360. if mean_lon > 180. else mean_lon
            utm_zone = int((mean_lon_std + 180.) / 6.) + 1
            self._print(f"  Auto-detected UTM zone: {utm_zone}")
        south = float(np.mean(self.lat)) < 0.
        self._proj = pyproj.Proj(proj='utm', zone=int(utm_zone),
                                 south=south, ellps='WGS84')

        self.x, self.y = self._ll2xy(self.lon, self.lat)

        self._print(f"  Loaded {len(self.sta)} stations")

        if fault is not None and max_dist is not None:
            fault_geom = fault.get_surface_trace_xy().geometry.unary_union
            dists = np.array([Point(xi, yi).distance(fault_geom)
                              for xi, yi in zip(self.x, self.y)])
            u = np.flatnonzero(dists <= max_dist)
            self._keep_stations(u)
            self._print(f"  After distance filter ({max_dist/1000:.1f} km): "
                        f"{len(self.sta)} stations")
        return self


    ##  Build diagonal covariance matrix
    def compute_covariance(self, sde_scale_factor=1.):

        '''
        Build the (2N x 2N) diagonal data covariance matrix from per-station
        horizontal uncertainties.

        Data vector ordering: [de_0, dn_0, de_1, dn_1, ...]
        Diagonal entries: [sde_0^2, sdn_0^2, sde_1^2, sdn_1^2, ...]

        Returns:
            self
        '''

        N = len(self.de)
        sigmas = np.empty(2 * N)
        sigmas[0::2] = self.sde
        sigmas[1::2] = self.sdn
        self.Cd = np.diag((sigmas*sde_scale_factor)**2)
        self._print(f"  Built diagonal Cd ({2*N} x {2*N})")
        return self


    ##  Compute Green's functions
    def compute_greens_functions(self, fault):

        '''
        Compute Okada Green's functions for horizontal GNSS displacements.

        Calls fault.compute_greens_functions() at each station location and
        extracts the east and north components, assembling them into a matrix
        aligned with the interleaved data vector [de_0, dn_0, de_1, dn_1, ...].

        Args:
            fault: Fault3d object with built patches.

        Returns:
            A new GNSS instance with self.gfs set to shape (n_patches, 2N).
        '''

        _self = self._copy()
        pts = np.vstack((_self.x, _self.y))
        raw = fault.compute_greens_functions(pts).gfs   # (2, n_patches, 3, N)

        n_patches = raw.shape[1]
        N = len(_self.x)

        gfs = np.zeros((2, n_patches, 2 * N))
        gfs[:, :, 0::2] = raw[:, :, 0, :]   # east component
        gfs[:, :, 1::2] = raw[:, :, 1, :]   # north component

        _self.gfs = gfs  # (2, n_patches, 2*N)
        self._print(f"  Green's functions shape: {gfs.shape}")
        return _self


    ##  Map plot
    def plot(self, ax=None, fault=None, scale=None, title=None):

        '''
        Plot horizontal GNSS displacements as arrows on a geographic map.
        Vertical component is not shown.

        Kwargs:
            ax (Axes or None): Axes to plot on; new figure created if None.
            fault: Fault3d object with a .get_surface_trace_xy() GeoDataFrame (UTM CRS), or None.
            scale (float): Quiver scale (data units per plot-coordinate unit).
                Larger values make arrows shorter.  Auto-computed if None.
            title (str): Axes title.

        Returns:
            (fig, ax) if a new figure was created, otherwise ax.
        '''

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')

        lon_range = self.lon.max() - self.lon.min()
        lat_range = self.lat.max() - self.lat.min()

        if scale is None:
            max_disp = float(np.max(np.hypot(self.de, self.dn)))
            if max_disp > 0.:
                scale = max_disp / (0.08 * lon_range)
            else:
                scale = 1.

        ax.quiver(self.lon, self.lat, self.de, self.dn,
                  angles='xy', scale_units='xy', scale=scale,
                  color='black', width=0.003)

        # Reference arrow: round down to nearest power of 10 in mm
        max_disp = float(np.max(np.abs(self.de)))
        ref_m = 10. ** np.floor(np.log10(max(max_disp, 1e-9)))
        ref_len_deg = ref_m / scale
        ref_x = self.lon.min() + 0.05 * lon_range
        ref_y = self.lat.min() + 0.05 * lat_range
        ax.quiver(ref_x, ref_y, ref_m, 0.,
                  angles='xy', scale_units='xy', scale=scale,
                  color='black', width=0.003)
        ax.text(ref_x + ref_len_deg / 2., ref_y - 0.02 * lat_range,
                f'{ref_m * 1000:.0f} mm', ha='center', va='top', fontsize=8)

        if fault is not None and hasattr(fault, 'trace'):
            for geom in fault.get_surface_trace_xy().geometry:
                lines = ([geom] if isinstance(geom, LineString)
                         else list(geom.geoms))
                for line in lines:
                    coords = np.array(line.coords)
                    lons_t, lats_t = self._proj(
                        coords[:, 0], coords[:, 1], inverse=True)
                    ax.plot(lons_t, lats_t, 'r-', lw=1.5)

        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        if title is not None:
            ax.set_title(title)

        if created_fig:
            return fig, ax
        return ax
