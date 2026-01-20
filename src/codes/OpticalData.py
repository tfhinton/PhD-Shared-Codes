###   Imports
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib as mpl

###   OpticalData class
class OpticalData:

    '''
    Container class for optical image correlation data.

    Properties:
        ew (xarray DataArray): East-West optical image correlation data in form of xarray DataArray
        ew_source (xarray DataArray): original copy of EW optical DataArray (i.e. before downsampling, etc)
        ns (xarray DataArray): North-South optical image correlation data in form of xarray DataArray
        ns_source (xarray DataArray): original copy of NS optical DataArray (i.e. before downsampling, etc)
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
        self.ew_source = None
        self.ns = None
        self.ns_source = None
        self.strain = None

        # Import TIF files if filepaths provided
        if "ew_filepath" in import_tif_kwargs or "ns_filepath" in import_tif_kwargs:
            self.import_tif(**import_tif_kwargs)


    ## Helper method to print if verbose is enabled
    def _print(self, *args):
        if self.verbose: print(*args)


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
            self.ew_source = rxr.open_rasterio(ew_filepath).squeeze("band", drop=True)
            self.ew = self.ew_source
            self._print("Imported EW data shape:", self.ew.shape)
        
        if ns_filepath is not None:
            self.ns_source = rxr.open_rasterio(ns_filepath).squeeze("band", drop=True)
            self.ns = self.ns_source
            self._print("Imported NS data shape:", self.ns.shape)

        self._print("")
        return
    

    ## Method to clean out NaN values
    def clear_nan(self, clear_zero=True, ew=True, ns=True):

        '''
        Remove NaN and (optionally) zero values from data.

        Kwargs:
            clear_zero (bool): Remove zero (0.) values from data.
        '''

        self._print("Clearing NaNs...")

        if ew:
            self.ew = self.ew.where(self.ew.notnull())
            if clear_zero: self.ew = self.ew.where(self.ew != 0.)
        if ns:
            self.ns = self.ns.where(self.ns.notnull())
            if clear_zero: self.ns = self.ns.where(self.ns != 0.)
        
        self._print("... cleared.")
        return


    ## Method to quickly downsample (i.e. reduce resolution) of optical data
    def decimate(self, decimate_factor, ew=True, ns=True):

        '''
        Downsample optical data simply by sampling every nth data point (i.e. n=10 is decimation).
        Saves downsampled data to self.ew but leaves self.ew_source untouched.
        
        Kwargs:
            decimate_factor (int): Positive integer n, resampling will select every nth point,
                                   reducing the data array size by n^2.
            ew (bool): Decimate and return EW data if true
            ns (bool): Decimate and return NS data if true
        '''

        # Copy EW DataArray from source and decimate
        if ew:
            self._print("Decimating EW optical...")
            self.ew = self.ew_source.copy()
            self.ew = self.ew.isel(x=slice(0,None,decimate_factor), y=slice(0,None,decimate_factor))
        
        # Copy NS DataArray from source and decimate
        if ns:
            self._print("Decimating NS optical...")
            self.ns = self.ns_source.copy()
            self.ns = self.ns.isel(x=slice(0,None,decimate_factor), y=slice(0,None,decimate_factor))
        
        self._print("Decimated")
        self._print("")
    

    ## Method to reproject into new CRS
    def reproject(self, target_crs, ew=True, ns=True, save=True):

        '''
        Reproject data into new coordinate system

        Kwargs:
            target_crs (str): target CRS code, passed to rio.reproject()
            ew (bool): whether to reproject EW data
            ns (bool): whether to reproject NS data
            save (bool): whether to overwrite self.es and/or self.ns. If not, just returns reprojected data.
        
        Returns:
            ew_reproj (xarray DataArray) (if ew=True): reprojected East-West data array
            ns_reproj (xarray DataArray) (if ns=True): reprojected North-South data array
        '''

        self._print("Reprojecting into new coordinate system:", target_crs)

        if ew:
            ew_reproj = self.ew.rio.reproject(target_crs)
            if save: self.ew = ew_reproj
            if not ns:
                self._print("... reprojected.")
                return ew_reproj
        if ns:
            ns_reproj = self.ns.rio.reproject(target_crs)
            if save: self.ns = ns_reproj
            if not ew:
                self._print("... reprojected.")
                return ns_reproj
        
        self._print("... reprojected.")
        return ew_reproj, ns_reproj


    ## Helper method to easily plot the optical data using Matplotlib. Provide axes or they will be generated.
    def plot(self, ew=True, ns=True, title=None):

        '''
        Plots a nice looking map of optical data. Will plot on provided axes if given, else will produce and return a figure.

        Kwargs:
            ew (bool or Mpl axes): does not plot EW displacement if false, creates axes and plots if True, plots on provided axes if given.
            ns (bool or Mpl axes): does not plot NS displacement if false, creates axes and plots if True, plots on provided axes if given.
            title (str): Figure title
        '''

        ####    SET UP FIG, AXES   ####
        self._print("Plotting...")
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
            ew_latlon = self.reproject("EPSG:4326", ns=False, save=False)
            ew_latlon.plot(ax=ax_ew, cmap="turbo", add_colorbar=True)
            ax_ew.set_title("EW displacement" if ns else "")
            # if ns:   # label axis to differentiate from NS if necessary
            #     ax_ew.annotate("EW displacement", (0.2, 0.15), xycoords="axes fraction")
            axs.append(ax_ew)
        
        # NS data
        if ns:
            ns_latlon = self.reproject("EPSG:4326", ew=False, save=False)
            ns_latlon.plot(ax=ax_ns, cmap="turbo", add_colorbar=True)
            ax_ns.set_title("NS displacement" if ew else "")
            # if ew:   # label axis to differentiate from NS if necessary
            #     # ax_ns.annotate("NS displacement", (0.2, 0.15), xycoords="axes fraction")
            axs.append(ax_ns)
        

        ####    CONFIGURE PLOTS    ####
        for ax in axs:
            ax.set_xlabel("Latitude (˚)")
            ax.set_ylabel("Longitude (˚)")


        ####    RETURN FIG, AXS    ####
        res = []
        if fig: res.append(fig)
        if ew and ns: res.append((ax_ew, ax_ns))
        elif ew: res.append(ax_ew)
        elif ns: res.append(ax_ns)

        self._print("... plotted.")
        return res


