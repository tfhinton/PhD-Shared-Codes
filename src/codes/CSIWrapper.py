import numpy as np


class CSIWrapper:

    def __init__(self, multi=None, faults=None, datasets=None, trans=None):
        self.multi = multi
        self.faults = faults
        self.datasets = datasets
        self.trans = trans

    def vertical_profile(self, along_strike_dist, fault_idx=0, tolerance=1.0):
        """
        Extract patches along a vertical cross-section at a given along-strike position.

        For each triangular patch, the centroid is projected onto the fault surface
        trace (parameterised as cumulative arc-length from its start). Patches whose
        projected position falls within `tolerance` km of `along_strike_dist` are
        returned, sorted from shallow to deep.

        Args:
            along_strike_dist (float): Target distance along the surface trace (km),
                measured from the first point of fault.xf / fault.yf.
            fault_idx (int): Index into self.faults. Default 0.
            tolerance (float): Along-strike half-width of the search window (km).
                Default 1.0.

        Returns:
            list of dict, sorted by depth_top (shallow first), each containing:
                'patch_idx'    : int   – index into fault.patch
                'patch'        : ndarray (3, 3) – vertex coordinates [x, y, z] in UTM km
                'depth_top'    : float – shallowest vertex depth of the patch (km)
                'depth_bot'    : float – deepest vertex depth of the patch (km)
                'along_strike' : float – projected along-strike distance of the centroid (km)
        """
        fault = self.faults[fault_idx]
        xf, yf = np.asarray(fault.xf), np.asarray(fault.yf)

        # Cumulative arc-length along the surface trace
        dx = np.diff(xf)
        dy = np.diff(yf)
        seg_len = np.hypot(dx, dy)
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])

        def _project(px, py):
            """Arc-length of the nearest point on the polyline to (px, py)."""
            best_s, best_d2 = 0.0, np.inf
            for i in range(len(seg_len)):
                if seg_len[i] == 0.0:
                    continue
                t = np.clip(
                    ((px - xf[i]) * dx[i] + (py - yf[i]) * dy[i]) / seg_len[i] ** 2,
                    0.0, 1.0,
                )
                nearest_x = xf[i] + t * dx[i]
                nearest_y = yf[i] + t * dy[i]
                d2 = (px - nearest_x) ** 2 + (py - nearest_y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_s = arc[i] + t * seg_len[i]
            return best_s

        results = []
        for i, patch in enumerate(fault.patch):
            centroid_x = patch[:, 0].mean()
            centroid_y = patch[:, 1].mean()
            s = _project(centroid_x, centroid_y)
            if abs(s - along_strike_dist) <= tolerance:
                results.append({
                    'patch_idx': i,
                    'patch': patch,
                    'depth_top': patch[:, 2].min(),
                    'depth_bot': patch[:, 2].max(),
                    'along_strike': s,
                })

        results.sort(key=lambda r: r['depth_top'])
        return results
