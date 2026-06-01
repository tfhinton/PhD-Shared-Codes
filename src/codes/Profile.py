import matplotlib.pyplot as plt
import numpy as np
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
    

    ## Method to bin-average profile data along the x-axis
    def bin_average(self, n_bins=50, n_near_fault_bins=None, near_fault_dist=None):
        '''
        Split profile data into bins along the x-axis and return a copy with each bin
        replaced by its mean. Empty bins are NaN.

        If n_near_fault_bins and near_fault_dist are both given, the profile is divided
        into three zones:
            left far-field  : [x_min, -near_fault_dist]
            near-fault      : [-near_fault_dist, +near_fault_dist]  →  n_near_fault_bins bins
            right far-field : [+near_fault_dist,  x_max]
        The remaining (n_bins - n_near_fault_bins) far-field bins are distributed
        proportionally between left and right based on their data range.

        Kwargs:
            n_bins (int): total number of bins
            n_near_fault_bins (int): number of bins in the near-fault zone (requires near_fault_dist)
            near_fault_dist (float): half-width of the near-fault zone in the same units as xs

        Returns:
            Profile: copy with xs set to bin centres and displacements set to bin means
        '''
        _self = self._copy()
        xs = self.xs
        x_min, x_max = xs.min(), xs.max()

        if n_near_fault_bins is not None and near_fault_dist is not None:
            n_far = n_bins - n_near_fault_bins
            left_range  = max(-near_fault_dist - x_min, 0.)
            right_range = max(x_max - near_fault_dist, 0.)
            total_far   = left_range + right_range

            n_left  = round(n_far * left_range / total_far) if total_far > 0 else n_far // 2
            n_right = n_far - n_left

            parts = []
            if n_left > 0:
                parts.append(np.linspace(x_min, -near_fault_dist, n_left + 1))
            parts.append(np.linspace(-near_fault_dist, near_fault_dist, n_near_fault_bins + 1))
            if n_right > 0:
                parts.append(np.linspace(near_fault_dist, x_max, n_right + 1))

            edges = parts[0]
            for part in parts[1:]:
                edges = np.concatenate([edges, part[1:]])
        else:
            edges = np.linspace(x_min, x_max, n_bins + 1)

        n_total = len(edges) - 1
        centres = (edges[:-1] + edges[1:]) / 2.
        bin_idx = np.clip(np.digitize(xs, edges, right=False) - 1, 0, n_total - 1)

        binned = np.full((self.displacements.shape[0], n_total), np.nan)
        for i in range(n_total):
            mask = bin_idx == i
            if mask.any():
                binned[:, i] = np.nanmean(self.displacements[:, mask], axis=1)

        _self.xs = centres
        _self.displacements = binned
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