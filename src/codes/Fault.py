import geopandas as gpd
import copy
import numpy as np
from shapely.geometry import LineString, Point
from .Profile import Profile

class Fault:

    '''
    Class to import and visualise faults, and generate profile lines.

    Properties:
        trace (geopandas GeoDataFrame): GeoDataFrame holding the fault trace
    '''

    def __init__(self, **import_trace_args):
        self.trace = None
        if "filepath" in import_trace_args:
            self.import_trace(**import_trace_args)
    

    ## Import fault trace from .shp shapefile
    def import_trace(self, filepath=None, crs=None):

        '''
        Read a .shp shapefile to a GeoPandas data frame and save to self.trace

        Args:
            filepath (str): filepath of .shp fault trace file
            crs (str): passed on to self.trace_to_crs
        '''

        if filepath is None: return
        self.trace = gpd.read_file(filepath)
        if crs is not None:
            self.trace = self.trace_to_crs(crs).trace
        return self
    

    ## Import patches from .mesh.triangles file
    def import_patches(self, filepath=None):

        '''
        Read 
        '''

    ## Helper method to return a copy of the class instance
    def _copy(self):
        return copy.copy(self)
    
    
    ## Method to change CRS of fault trace
    def trace_to_crs(self, target_crs):
        '''
        Change CRS of fault trace

        Args:
            target_crs (str): target CRS passed onto geopandas.GeoDataFrame.to_crs
        '''
        _self = self._copy()
        _self.trace = _self.trace.to_crs(target_crs)
        return _self
    

    ## Method to generate programatically profile lines perpendicular to the fault trace
    def gen_profiles(self, n_profiles=10, half_length=5000.):
        '''
        Generate profile lines perpendicular to a fault trace, spaced evenly along the fault (includes the start and end of fault).

        Kwargs:
            n_profiles (int): the number of profiles to generate
            half_length (float): the length of the profile on either side of the fault in metres

        Returns:
            list of Profile: one Profile per sample position along the fault trace
        '''
        x_len = self.trace.length[0]
        profiles = []

        for x in np.linspace(0, x_len, n_profiles):
            line = profile_geom(self.trace, x, half_length)
            geolines = gpd.GeoDataFrame({"x_along_fault_trace": [x]}, geometry=[line], crs=self.trace.crs)
            profiles.append(Profile(trace=geolines, fault_x=half_length))

        return profiles


def profile_geom(trace_gdf, x, half_length):
    point = trace_gdf.interpolate(x)

    # Offset slightly to estimate tangent
    eps = 1.
    p1 = trace_gdf.interpolate(max(x-eps, 0))
    p2 = trace_gdf.interpolate(min(x+eps, trace_gdf.length[0]))

    dx = p2.x - p1.x
    dy = p2.y - p1.y

    # Calculate normalised perpendicular direction
    len = np.hypot(dx, dy)
    nx = -dy/len
    ny = dx/len

    # Create profile line object
    start = Point(
        point.x - nx * half_length,
        point.y - ny * half_length
    )
    end = Point(
        point.x + nx * half_length,
        point.y + ny * half_length
    )

    line = LineString([start, end])
    return line