import matplotlib.pyplot as plt
import copy

class Profile:

    '''
    Class to handle displacement profile data (e.g. a 2D profile across an optical image correlation image).
    Usually generated using Fault.gen_profiles() and OpticalData.evaluate_profiles().

    Properties:
        trace (geopandas GeoDataFrame): a data frame containing a single LineString defining the surface trace of the profile.
        linestring (shapely LineString): a helper property to quickly get the LineString from the self.trace GeoDataFrame
        xs (1d numpy array, dim n): along-trace sampling points for the displacement profiles
        displacements (2d numpy array, dim 2xn): u_parallel and u_perpendicular displacements at sampling points defined by
            the trace and along-trace xs.
    '''

    def __init__(self, trace=None, fault_x=None):
        self.trace = trace
        self.fault_x = fault_x
        self.xs = None
        self.displacements = None
    

    ## Helper method to get the trace linestring
    @property
    def linestring(self):
        return self.trace.geometry.values[0]
    

    ## Helper method to return a copy of the class instance
    def _copy(self):
        return copy.copy(self)
    

    ## Method to change CRS of fault trace
    def trace_to_crs(self, target_crs):
        '''
        Change CRS of profile trace

        Args:
            target_crs (str): target CRS passed onto geopandas.GeoDataFrame.to_crs
        '''
        _self = self._copy()
        _self.trace = _self.trace.to_crs(target_crs)
        return _self
    

    ## Helper method to easily plot the optical data using Matplotlib. Provide axes or they will be generated.
    def plot(self, ax=None, parallel=True, perpendicular=True, title=None):

        '''
        Plots the profile. Will plot on provided axes if given, else will produce and return a figure.

        Kwargs:
            ax (None or Mpl.axes.Axes): axis to plot onto
            parallel (bool): plot parallel displacement if true
            perpendicular (bool): plot perpendicular displacement if true
            title (str): Figure title
        '''

        # Set up fig, axs
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(6,4.5), layout="constrained")
            if title is not None:
                fig.suptitle(title)
            

        # Plot data
        if parallel:
            ax.plot(self.xs, self.displacements[0], label="parallel")
        if perpendicular:
            ax.plot(self.xs, self.displacements[1], label="perpendicular")
        

        # Configure plots
        ax.set_xlabel("Distance along profile (m)")
        ax.set_ylabel("Displacement (m)")
        if parallel and perpendicular:
            ax.legend()


        # Return fig, ax
        if fig: return (fig, ax)
        return ax