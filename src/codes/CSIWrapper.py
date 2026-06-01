import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from .Profile import Profile


class CSIWrapper:

    def __init__(self, multi=None, faults=None, datasets=None, trans=None):
        self.multi = multi
        self.faults = faults
        self.datasets = datasets
        self.trans = trans

    def _surface_trace_arclength(self, fault_idx=0):
        """
        Build the arc-length parameterisation of a CSI fault's surface trace.

        Args:
            fault_idx (int): Index into self.faults.

        Returns:
            xf      (ndarray): UTM easting of trace vertices (km)
            yf      (ndarray): UTM northing of trace vertices (km)
            arc     (ndarray): cumulative arc-length at each vertex (km), arc[0] == 0
            seg_len (ndarray): length of each segment between consecutive vertices (km)
            dx      (ndarray): easting difference for each segment
            dy      (ndarray): northing difference for each segment
        """
        fault = self.faults[fault_idx]
        xf = np.asarray(fault.xf)
        yf = np.asarray(fault.yf)
        dx = np.diff(xf)
        dy = np.diff(yf)
        seg_len = np.hypot(dx, dy)
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        return xf, yf, arc, seg_len, dx, dy

    def vertical_profile(self, along_strike_dist, fault_idx=0):
        """
        Intersect the fault mesh with a vertical plane at a given along-strike position.

        The cutting plane passes through the point on the surface trace at arc-length
        `along_strike_dist` and is oriented perpendicular to the local strike direction
        (i.e. its horizontal normal is the local along-strike tangent). For each
        triangular patch, the plane-triangle intersection is computed by linearly
        interpolating along the two edges that straddle the plane. This gives an exact
        intersection line segment whose depth range is the true depth interval of the
        profile at that patch.

        Args:
            along_strike_dist (float): Arc-length along the fault surface trace (km),
                measured from the first point of fault.xf / fault.yf.
            fault_idx (int): Index into self.faults. Default 0.

        Returns:
            list of dict, sorted by depth_top (shallow first), each containing:
                'patch_idx'   : int      – index into fault.patch
                'patch'       : (3,3)    – the original patch vertices [x, y, z] in UTM km
                'intersection': (2,3)    – the two endpoints of the intersection segment
                'depth_top'   : float    – shallower endpoint depth (km)
                'depth_bot'   : float    – deeper endpoint depth (km)
        """
        fault = self.faults[fault_idx]
        xf, yf, arc, seg_len, dx, dy = self._surface_trace_arclength(fault_idx)

        # --- Find the point on the trace and local tangent at along_strike_dist ---
        idx = int(np.clip(np.searchsorted(arc, along_strike_dist, side='right') - 1,
                          0, len(seg_len) - 1))
        t = (along_strike_dist - arc[idx]) / seg_len[idx]
        plane_origin = np.array([xf[idx] + t * dx[idx],
                                 yf[idx] + t * dy[idx],
                                 0.0])
        # Unit tangent along strike — this is the normal to the cutting plane
        plane_normal = np.array([dx[idx] / seg_len[idx],
                                 dy[idx] / seg_len[idx],
                                 0.0])

        # --- Intersect each triangular patch with the cutting plane ---
        results = []
        for i, patch in enumerate(fault.patch):
            # Signed distance of each vertex from the cutting plane
            # (only x and y matter since normal[2] == 0)
            d = (patch[:, 0] - plane_origin[0]) * plane_normal[0] + \
                (patch[:, 1] - plane_origin[1]) * plane_normal[1]

            # Collect intersection points on edges that straddle the plane
            pts = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                da, db = d[a], d[b]
                if da * db < 0.0:
                    # Edge crosses the plane: interpolate
                    t_cross = da / (da - db)
                    pts.append(patch[a] + t_cross * (patch[b] - patch[a]))
                elif da == 0.0 and db != 0.0:
                    # Vertex a is exactly on the plane
                    pts.append(patch[a].copy())

            # Deduplicate (can happen when a vertex is exactly on the plane)
            unique_pts = []
            for pt in pts:
                if not any(np.allclose(pt, up, atol=1e-10) for up in unique_pts):
                    unique_pts.append(pt)

            if len(unique_pts) < 2:
                continue  # no intersection or just a tangent touch

            p0, p1 = unique_pts[0], unique_pts[1]
            depth_top = min(p0[2], p1[2])
            depth_bot = max(p0[2], p1[2])

            intersection = np.array([p0, p1])
            if intersection[0][2] > intersection[1][2]:
                intersection[[0, 1]] = intersection[[1, 0]]  # swap to ensure top is first
            # intersection = np.sort(intersection, axis=2)  # sort by depth (z)

            results.append({
                'patch_idx': i,
                'patch': patch,
                'intersection': intersection,
                'intersection_top': intersection[0][2],
                'intersection_bot': intersection[1][2],
                'depth_top': depth_top,
                'depth_bot': depth_bot,
            })

        results.sort(key=lambda r: r['depth_top'])
        return results

    def gen_profiles(self, n_profiles=10, half_length=5., fault_idx=0):
        """
        Generate profile lines perpendicular to the fault surface trace, spaced evenly
        along it. Equivalent to Fault.gen_profiles() but works with CSI fault geometry
        (xf/yf in km UTM) rather than a GeoDataFrame.

        Kwargs:
            n_profiles (int): number of profiles to generate (includes trace endpoints)
            half_length (float): half-length of each profile line in km
            fault_idx (int): index into self.faults

        Returns:
            profiles          (list of Profile): one Profile per sample position;
                geometry coordinates are in metres (UTM), CRS taken from the CSI fault.
            vertical_profiles (list of list):    vertical_profile() result per sample position
        """
        fault = self.faults[fault_idx]
        xf, yf, arc, seg_len, dx_segs, dy_segs = self._surface_trace_arclength(fault_idx)
        total_length = arc[-1]
        half_length_m = half_length * 1000.

        profiles = []
        vertical_profiles = []
        for s in np.linspace(0, total_length, n_profiles):
            line_km = _csi_profile_geom(s, xf, yf, arc, seg_len, dx_segs, dy_segs, half_length)
            # CSI coordinates are in km; scale to metres to match the raster CRS
            line_m = LineString([(x * 1000., y * 1000.) for x, y in line_km.coords])
            geolines = gpd.GeoDataFrame({"x_along_fault_trace": [s]}, geometry=[line_m], crs=fault.utm)
            profiles.append(Profile(trace=geolines, fault_x=half_length_m))
            vertical_profiles.append(self.vertical_profile(s, fault_idx=fault_idx))

        return profiles, vertical_profiles


def _csi_profile_geom(s, xf, yf, arc, seg_len, dx_segs, dy_segs, half_length):
    """Return a LineString perpendicular to the fault trace at arc-length s (km)."""
    idx = int(np.clip(np.searchsorted(arc, s, side='right') - 1, 0, len(seg_len) - 1))
    t = (s - arc[idx]) / seg_len[idx]

    px = xf[idx] + t * dx_segs[idx]
    py = yf[idx] + t * dy_segs[idx]

    # Unit along-strike tangent, then rotate 90° for the perpendicular
    tx = dx_segs[idx] / seg_len[idx]
    ty = dy_segs[idx] / seg_len[idx]
    nx, ny = -ty, tx

    return LineString([
        Point(px - nx * half_length, py - ny * half_length),
        Point(px + nx * half_length, py + ny * half_length),
    ])
