###   Imports
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
import cmcrameri.cm as cmc

###   OpticalData class
class OpticalData:

    '''
    Container class for optical image correlation data.

    Properties:
        ew (xarray DataArray): East-West optical image correlation data in form of xarray DataArray
        ns (xarray DataArray): North-South optical image correlation data in form of xarray DataArray
        verbose (bool): Whether to print status messages
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
        # self.strain = None

        # Import TIF files if filepaths provided
        if "ew_filepath" in import_tif_kwargs or "ns_filepath" in import_tif_kwargs:
            self = self.import_tif(**import_tif_kwargs)


    ## Helper method to print if verbose is enabled
    def _print(self, *args):
        if self.verbose: print(*args)


    ## Helper method to return a copy of the class instance
    def _copy(self):
        return copy.copy(self)


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


