###   Imports
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy

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
    

    ## Method to evaluate displacement along a given profile
    def evaluate_profile(self, profile, n_eval_pts=200):

        '''
        Evaluate displacement along a profile given a profile geometry. Rotates EW and NS into profile-parallel
        and profile-perpendicular components.

        Args:
            profile (Profile): profile class with geometry defined (i.e. profile.trace is not None)
            n_eval_pts (int): the number of evenly-spaced points to evaluate along the profile
        '''
        (x0, y0), (x1, y1) = profile.linestring.coords[:2]

        xs = np.linspace(x0, x1, n_eval_pts)
        ys = np.linspace(y0, y1, n_eval_pts)

        xs_along_profile = np.hypot(xs - x0, ys - y0)

        theta = np.arctan2(y1-y0, x1-x0)

        parallel = self.ew * np.cos(theta) + self.ns * np.sin(theta)
        perp = -self.ew * np.sin(theta) + self.ns * np.cos(theta)

        # Interpolate along profile
        parallel_vals = parallel.interp(
            x=("points", xs),
            y=("points", ys)
        ).values
        perp_vals = perp.interp(
            x=("points", xs),
            y=("points", ys)
        ).values

        # Pack up into np array
        displacements = np.array([parallel_vals, perp_vals])

        # Alter profile object
        profile.displacements = displacements
        profile.xs = xs_along_profile

        return profile
    

    ## Helper method to evaluate multiple profiles in one go
    def evaluate_profiles(self, profiles):
        return [self.evaluate_profile(p) for p in profiles]


    ## Helper method to easily plot the optical data using Matplotlib. Provide axes or they will be generated.
    def plot(self, ew=True, ns=True, title=None, fault=None, profiles=None):

        '''
        Plots a nice looking map of optical data. Will plot on provided axes if given, else will produce and return a figure.

        Kwargs:
            ew (bool or Mpl axes): does not plot EW displacement if false, creates axes and plots if True, plots on provided axes if given.
            ns (bool or Mpl axes): does not plot NS displacement if false, creates axes and plots if True, plots on provided axes if given.
            title (str): Figure title
        '''
        _self = self._copy()

        ####    SET UP FIG, AXES   ####
        _self._print("Plotting...")
        n_axs = sum((bool(ew), bool(ns)))   # count number of axes to plot

        if ew==True or ns==True:   # create figure if not supplied with axes
            fig = plt.figure(figsize=(2+5*n_axs, 6), layout="constrained")
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

        # EW data
        if ew:
            ew_latlon = _self.ew.rio.reproject("EPSG:4326")
            ew_latlon.plot(ax=ax_ew, cmap="turbo", add_colorbar=True)
            ax_ew.set_title("EW displacement" if ns else "")
            axs.append(ax_ew)
        
        # NS data
        if ns:
            ns_latlon = _self.ew.rio.reproject("EPSG:4326")
            ns_latlon.plot(ax=ax_ns, cmap="turbo", add_colorbar=True)
            ax_ns.set_title("NS displacement" if ew else "")
            axs.append(ax_ns)
        

        ####    CONFIGURE PLOTS    ####
        for ax in axs:
            ax.set_xlabel("Latitude (˚)")
            ax.set_ylabel("Longitude (˚)")

            if fault is not None:
                fault = fault.trace_to_crs("EPSG:4326")
                fault.trace.plot(ax=ax, color="black", linewidth=2.)
            if profiles is not None:
                for p in profiles:
                    p = p.trace_to_crs("EPSG:4326")
                    p.trace.plot(ax=ax, color="deeppink", linewidth=1.5)



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


