import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring


class FaultRects:

    def __init__(self, trace_shp_filepath=None, vertical_discretisation=None,
                 patch_length=None, geom=None):
        if trace_shp_filepath is not None:
            self.import_trace(trace_shp_filepath)
            if (vertical_discretisation is not None and patch_length is not None
                    and geom is not None):
                self.build_patches(vertical_discretisation, patch_length, geom)

    def import_trace(self, shp_filepath):
        gdf = gpd.read_file(shp_filepath)
        gdf = gdf.set_index("id")
        self.trace = gdf

    def build_patches(self, vertical_discretisation, patch_length, geom,
                      last_segment="merge"):
        '''
        Build dipping, even-width rectangular patches grouped into inversion cells.

        Geometry model (two-layer listric):
            Each trace feature (one along-strike domain) dips at ``dip_sh`` from the
            surface down to ``kink_z`` then steepens to ``dip_dp`` below it.  Patch
            rows deeper than the first are offset horizontally so the panel stays
            connected.  The down-dip side is chosen so Okada's natural down-dip
            direction matches the feature's ``dip_az`` (down-dip azimuth); if that
            requires reversing the strike sense, the strike-slip sign is negated so
            the physical slip sense is preserved.

        Even-width segmentation:
            Each feature's LineString is cut into even ``patch_length`` pieces with
            ``shapely.ops.substring`` (intermediate trace vertices are retained, so a
            piece subdivides into fine sub-segments).  One inversion *cell* = one
            even piece x one vertical row; its Green's function is the sum over all
            fine sub-patches inside it (Okada is linear).

        Args:
            vertical_discretisation (sequence): depth nodes (m, positive down),
                shallow -> deep.
            patch_length (float): target along-strike cell width (m).
            geom (dict): per-feature-``id`` modelling params, each a dict with
                ``dip_sh``, ``dip_dp`` (deg), ``kink_z`` (m), ``slip_sign`` (+/-1).
            last_segment (str): "merge" folds the trailing remainder into the last
                full piece; "keep" leaves it as a short piece.
        '''

        vd = np.asarray(vertical_discretisation, dtype=float)
        self.vertical_discretisation = vd
        n_rows = len(vd) - 1

        cells = {}
        seg_global = 0
        fault_x_run = {}   # cumulative along-strike position per fault label

        has_fault_col = "fault" in self.trace.columns
        has_dipaz_col = "dip_az" in self.trace.columns

        for line_id, row in self.trace.iterrows():
            if line_id not in geom:
                continue
            params = geom[line_id]
            dip_sh = np.deg2rad(float(params["dip_sh"]))
            dip_dp = np.deg2rad(float(params["dip_dp"]))
            kink_z = float(params["kink_z"])
            slip_sign = int(params.get("slip_sign", 1))
            fault_label = row["fault"] if has_fault_col else str(line_id)
            dip_az = float(row["dip_az"]) if has_dipaz_col else 0.0

            line = row.geometry
            coords = [c[:2] for c in line.coords]   # drop any Z component

            # Choose strike orientation so Okada's +ys down-dip side faces dip_az.
            first = np.asarray(coords[0], dtype=float)
            last = np.asarray(coords[-1], dtype=float)
            strike_vec = last - first
            # +ys (down-dip) for this orientation = strike rotated +90 CCW: (-Sy, Sx)
            ydir = np.array([-strike_vec[1], strike_vec[0]])
            az_rad = np.deg2rad(dip_az)
            az_vec = np.array([np.sin(az_rad), np.cos(az_rad)])  # (east, north)
            if np.dot(ydir, az_vec) < 0:
                line = LineString(coords[::-1])
                eff_sign = -slip_sign
            else:
                line = LineString(coords)
                eff_sign = slip_sign

            # Even along-strike split.
            L = line.length
            if patch_length is None or patch_length <= 0 or L <= patch_length:
                boundaries = [0.0, L]
            else:
                n_full = int(np.floor(L / patch_length))
                boundaries = [k * patch_length for k in range(n_full + 1)]
                rem = L - n_full * patch_length
                if rem > 1e-6:
                    if last_segment == "merge":
                        boundaries[-1] = L
                    else:
                        boundaries.append(L)

            fault_x_run.setdefault(fault_label, 0.0)

            for b in range(len(boundaries) - 1):
                d0, d1 = boundaries[b], boundaries[b + 1]
                piece = substring(line, d0, d1)
                piece_coords = [c[:2] for c in piece.coords]   # drop any Z component
                piece_len = d1 - d0
                x_along = fault_x_run[fault_label]

                # One empty cell per vertical row for this even piece.
                piece_cells = {
                    v: Cell(fault=fault_label, slip_sign=eff_sign, x_along=x_along,
                            width=piece_len, z_top=vd[v], z_bot=vd[v + 1])
                    for v in range(n_rows)
                }

                # Walk each fine sub-segment down-dip, accumulating offset.
                for j in range(len(piece_coords) - 1):
                    ax_, ay_ = piece_coords[j]
                    bx_, by_ = piece_coords[j + 1]
                    dx, dy = bx_ - ax_, by_ - ay_
                    seglen = np.hypot(dx, dy)
                    if seglen == 0:
                        continue
                    sx, sy = dx / seglen, dy / seglen
                    nx, ny = -sy, sx          # +ys down-dip horizontal unit

                    cax, cay, cbx, cby = ax_, ay_, bx_, by_  # upper-edge endpoints
                    for v in range(n_rows):
                        z_top_row, z_bot_row = vd[v], vd[v + 1]
                        # Split the row at the kink if it straddles it.
                        if z_bot_row <= kink_z:
                            subrows = [(z_top_row, z_bot_row, dip_sh)]
                        elif z_top_row >= kink_z:
                            subrows = [(z_top_row, z_bot_row, dip_dp)]
                        else:
                            subrows = [(z_top_row, kink_z, dip_sh),
                                       (kink_z, z_bot_row, dip_dp)]

                        for (zt, zb, dip) in subrows:
                            patch = Patch(cax, cay, zb, cbx, cby, zt, dip=dip,
                                          dip_az=dip_az, fault=fault_label,
                                          slip_sign=eff_sign)
                            piece_cells[v].patches.append(patch)
                            run = (zb - zt) * np.cos(dip) / np.sin(dip)
                            cax += nx * run
                            cay += ny * run
                            cbx += nx * run
                            cby += ny * run

                for v in range(n_rows):
                    cells[(seg_global, v)] = piece_cells[v]
                seg_global += 1
                fault_x_run[fault_label] += piece_len

        self.cells = cells
        self.slips = np.zeros(len(cells))
        return self

    def plot_slip(self, axes=None, cmap='plasma', vmin=None, vmax=None,
                  sigma_grey=0.5, edgecolor='k', linewidth=0.5):
        '''2D slip plot (along-strike vs depth), one axes per fault label.'''
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        from matplotlib.collections import PatchCollection
        from matplotlib.cm import ScalarMappable

        slips = np.asarray(self.slips)
        has_sigmas = hasattr(self, 'sigmas') and self.sigmas is not None
        cells = list(self.cells.values())

        # Preserve fault order of first appearance.
        faults = list(dict.fromkeys(c.fault for c in cells))

        if axes is None:
            fig, axes = plt.subplots(1, len(faults),
                                     figsize=(6 * len(faults), 4), squeeze=False)
            axes = list(axes[0])
        else:
            axes = list(np.atleast_1d(axes))
            fig = axes[0].get_figure()

        if vmin is None:
            vmin = slips.min()
        if vmax is None:
            vmax = slips.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)

        for ax, fault_label in zip(axes, faults):
            idx = [i for i, c in enumerate(cells) if c.fault == fault_label]
            rects = [mpatches.Rectangle((cells[i].x_along, cells[i].z_top),
                                        cells[i].width,
                                        cells[i].z_bot - cells[i].z_top)
                     for i in idx]
            vals = slips[idx]
            face = cmap_obj(norm(vals))
            if has_sigmas:
                sig = np.asarray(self.sigmas)[idx]
                rng = sig.max() - sig.min()
                sn = (sig - sig.min()) / (rng if rng > 0 else 1.0)
                grey = np.array([sigma_grey, sigma_grey, sigma_grey, 1.0])
                face = (1.0 - sn[:, None]) * face + sn[:, None] * grey
            pc = PatchCollection(rects, facecolor=face, edgecolor=edgecolor,
                                 linewidth=linewidth)
            ax.add_collection(pc)
            ax.autoscale_view()
            ax.invert_yaxis()
            ax.set_xlabel('Along-strike distance (m)')
            ax.set_ylabel('Depth (m)')
            ax.set_title(fault_label)

        sm = ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=list(axes), label='Slip')
        return fig, axes

    def plot_fault3d(self, ax=None, color_by='slip', cmap='plasma', vmin=None,
                     vmax=None, elev=25, azim=-60, edgecolor='k', linewidth=0.3):
        '''3D rendering of the fine patch surfaces for visual confirmation.'''
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        cells = list(self.cells.values())
        slips = np.asarray(self.slips)

        quads, vals, fault_labels = [], [], []
        for i, c in enumerate(cells):
            cell_slip = slips[i] if i < len(slips) else 0.0
            for p in c.patches:
                quads.append(p.corners3d())
                vals.append(cell_slip)
                fault_labels.append(c.fault)

        cmap_obj = plt.get_cmap(cmap)
        if color_by == 'slip':
            v = np.asarray(vals)
            if vmin is None:
                vmin = v.min()
            if vmax is None:
                vmax = v.max()
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            face = cmap_obj(norm(v))
        else:  # color by fault
            uniq = list(dict.fromkeys(fault_labels))
            tab = plt.get_cmap('tab10')
            cmap_map = {f: tab(k % 10) for k, f in enumerate(uniq)}
            face = [cmap_map[f] for f in fault_labels]

        coll = Poly3DCollection(quads, facecolors=face, edgecolors=edgecolor,
                                linewidths=linewidth)
        ax.add_collection3d(coll)

        allpts = np.array([pt for q in quads for pt in q])
        ax.set_xlim(allpts[:, 0].min(), allpts[:, 0].max())
        ax.set_ylim(allpts[:, 1].min(), allpts[:, 1].max())
        ax.set_zlim(allpts[:, 2].max(), allpts[:, 2].min())  # depth increases downward
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_zlabel('Depth (m)')
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1]),
                               np.ptp(allpts[:, 2])))
        except Exception:
            pass

        if color_by == 'slip':
            sm = ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='Slip', shrink=0.6)
        return fig, ax

    def compute_greens_functions(self, pts):
        '''
        Okada surface-displacement Green's functions per inversion cell.

        Each cell's GF is the sum over its fine sub-patches (Okada is linear).
        Returns self with ``self.gfs`` of shape (2, n_cells, 3, n_pts), axis 0 =
        [strike-slip, dip-slip], axis 2 = [east, north, up].
        '''
        n_cells = len(self.cells)
        n_pts = pts.shape[1]
        gfs_ss = np.zeros((n_cells, 3, n_pts))
        gfs_ds = np.zeros((n_cells, 3, n_pts))

        for i, cell in enumerate(self.cells.values()):
            for patch in cell.patches:

                ##  Rotate eval pts into the patch reference frame  ##
                _xs = pts[0] - patch.x0
                _ys = pts[1] - patch.y0

                patch_strike = np.arctan2(patch.y1 - patch.y0, patch.x1 - patch.x0)
                cos_s = np.cos(patch_strike)
                sin_s = np.sin(patch_strike)

                xs = cos_s * _xs + sin_s * _ys
                ys = -sin_s * _xs + cos_s * _ys

                ##  Strike-slip  ##
                u1, u2, u3 = compute_okada(patch, np.array([patch.slip_sign, 0.]), xs, ys)
                ux = cos_s * u1 - sin_s * u2
                uy = sin_s * u1 + cos_s * u2
                gfs_ss[i] += np.vstack((ux, uy, u3))

                ##  Dip-slip  ##
                u1, u2, u3 = compute_okada(patch, np.array([0., 1.]), xs, ys)
                ux = cos_s * u1 - sin_s * u2
                uy = sin_s * u1 + cos_s * u2
                gfs_ds[i] += np.vstack((ux, uy, u3))

        self.gfs = np.stack([gfs_ss, gfs_ds], axis=0)  # (2, n_cells, 3, n_pts)
        return self


def compute_okada(patch, slip, xs, ys, eps=1e-10):

    ##  COMPUTE OKADA  ##
    s1 = slip[0]
    s2 = slip[1]

    def _R(xi, eta, q):
        return np.sqrt(xi**2 + eta**2 + q**2)
    def _d(eta, delta, q):
        return (eta*np.sin(delta) - q*np.cos(delta))
    def _I1(xi, delta, R, d_tild, I5, q, mu=3e10, lam=3e10):
        if abs(np.cos(delta)) < eps:
            return -(mu / (2*(lam+mu))) * ((xi*q)/((R+d_tild)**2))
        else:
            return (mu / (lam+mu)) * (-xi / (np.cos(delta)*(R+d_tild))) - I5 * (np.sin(delta) / np.cos(delta))
    def _X(xi, q):
        return np.sqrt(xi**2 + q**2)
    def _I5(delta, eta, X, q, R, xi, d_tild, mu=3e10, lam=3e10):
        if abs(np.cos(delta)) < eps:
            return -(mu/(lam+mu))*((xi*np.sin(delta))/(R+d_tild))
        # Okada rule (ii): set I5 = 0 when ξ = 0
        numerator = eta*(X + q*np.cos(delta)) + X*(R + X)*np.sin(delta)
        denom     = xi * (R + X) * np.cos(delta)
        safe_denom = np.where(np.abs(xi) < eps, 1.0, denom)
        raw = (mu/(lam+mu)) * (2/np.cos(delta)) * np.arctan(numerator / safe_denom)
        return np.where(np.abs(xi) < eps, 0.0, raw)
    def _y(eta, delta, q):
        return (eta*np.cos(delta) + q*np.sin(delta))
    def _I2(R, eta, I3, mu=3e10, lam=3e10):
        return ((mu / (lam+mu)) * (-np.log(R+eta)) - I3)
    def _I3(y_tild, delta, R, d_tild, eta, I4, q, mu=3e10, lam=3e10):
        if abs(np.cos(delta)) < eps:
            return (( mu/(2*(lam+mu)) ) * ( (eta/(R+d_tild) + ((y_tild*q)/((R+d_tild)**2)) - np.log(R+eta)) ))
        else:
            return (( mu/(lam+mu) ) * ( y_tild/(np.cos(delta)*(R+d_tild)) - np.log(R+eta) ) + (np.sin(delta)/np.cos(delta))*I4)
    def _I4(delta, R, d_tild, eta, q, mu=3e10, lam=3e10):
        if abs(np.cos(delta)) < eps:
            return -(mu/(lam+mu))*(q/(R+d_tild))
        else:
            return ( mu/(lam+mu) )*(1/np.cos(delta))*(np.log(R+d_tild) - np.sin(delta)*np.log(R+eta) )
    def _u1_arctan_safe(xi, eta, q, R):
        safe_denom = np.where(np.abs(q) < eps, 1.0, q * R)
        return np.where(np.abs(q) < eps, 0.0, np.arctan((xi * eta) / safe_denom))

    def _u1(xi, eta, q, delta):
        R = _R(xi, eta, q)
        d_tild = _d(eta, delta, q)
        X = _X(xi, q)
        I5 = _I5(delta, eta, X, q, R, xi, d_tild)
        I1 = _I1(xi, delta, R, d_tild, I5, q)
        return ( -s1 / (2*np.pi) ) * ( (xi * q)/(R*(R+eta)) + _u1_arctan_safe(xi, eta, q, R) + I1*np.sin(delta))
    def _u2(xi, eta, q, delta):
        y_tild = _y(eta, delta, q)
        R = _R(xi, eta, q)
        d_tild = _d(eta, delta, q)
        I4 = _I4(delta, R, d_tild, eta, q)
        I3 = _I3(y_tild, delta, R, d_tild, eta, I4, q)
        I2 = _I2(R, eta, I3)
        return ( -s1 / (2*np.pi) ) * ( (y_tild*q) / (R*(R+eta)) + (q*np.cos(delta))/(R+eta) + I2*np.sin(delta) )
    def _u3(xi, eta, q, delta):
        d_tild = _d(eta, delta, q)
        R = _R(xi, eta, q)
        I4 = _I4(delta, R, d_tild, eta, q)
        return (-s1 / (2*np.pi)) * ( (d_tild * q) / (R * (R + eta)) + (q * np.sin(delta)) / (R + eta) + I4 * np.sin(delta) )

    ##  Dip-slip Okada formulas (Okada 1985, Table 2, U2 component)
    def _u1_dip(xi, eta, q, delta):
        R = _R(xi, eta, q)
        y_tild = _y(eta, delta, q)
        d_tild = _d(eta, delta, q)
        I4 = _I4(delta, R, d_tild, eta, q)
        I3 = _I3(y_tild, delta, R, d_tild, eta, I4, q)
        return (-s2 / (2*np.pi)) * (q / R - I3*np.sin(delta)*np.cos(delta))
    def _u2_dip(xi, eta, q, delta):
        R = _R(xi, eta, q)
        d_tild = _d(eta, delta, q)
        y_tild = _y(eta, delta, q)
        X = _X(xi, q)
        I5 = _I5(delta, eta, X, q, R, xi, d_tild)
        I1 = _I1(xi, delta, R, d_tild, I5, q)
        Rxi = R * (R + xi)
        yq = np.where(np.abs(Rxi) < eps, 0., y_tild * q / np.where(np.abs(Rxi) < eps, 1., Rxi))
        return (-s2 / (2*np.pi)) * (yq + np.cos(delta)*_u1_arctan_safe(xi, eta, q, R) - I1*np.sin(delta)*np.cos(delta))
    def _u3_dip(xi, eta, q, delta):
        R = _R(xi, eta, q)
        d_tild = _d(eta, delta, q)
        X = _X(xi, q)
        I5 = _I5(delta, eta, X, q, R, xi, d_tild)
        Rxi = R * (R + xi)
        dq = np.where(np.abs(Rxi) < eps, 0., d_tild * q / np.where(np.abs(Rxi) < eps, 1., Rxi))
        return (-s2 / (2*np.pi)) * (dq + np.sin(delta)*_u1_arctan_safe(xi, eta, q, R) - I5*np.sin(delta)*np.cos(delta))

    # Geometry mapping into Okada's (1985) natural frame (per Segall 3.6.4 / Okada
    # 1985, eqs 25-30).  Observation (xs, ys) arrive in the patch frame with origin
    # at the upper-edge corner (xs along strike, ys toward the +ys down-dip side).
    # Okada references the *deep* edge at depth d, with the up-dip coordinate
    # positive: the deep-edge corner sits at perpendicular +W*cos(delta) and the
    # surface trace is up-dip of it.  So shift the origin to the deep-edge corner
    # and flip the perpendicular axis to up-dip-positive.
    delta = patch.get_dip()
    W = patch.get_dd_width()
    L = patch.get_along_strike_length()
    d = patch.z0                                # deep-edge depth
    yo = W * np.cos(delta) - ys                 # up-dip coordinate from deep-edge corner
    ps = yo * np.cos(delta) + d * np.sin(delta)
    qs = yo * np.sin(delta) - d * np.cos(delta)

    def _chinnery(f, xi, ps, qs, delta, W, L):
        return f(xi, ps, qs, delta) - f(xi, ps-W, qs, delta) - f(xi-L, ps, qs, delta) + f(xi-L, ps-W, qs, delta)

    u1 = _chinnery(_u1, xs, ps, qs, delta, W, L) + _chinnery(_u1_dip, xs, ps, qs, delta, W, L)
    u2 = _chinnery(_u2, xs, ps, qs, delta, W, L) + _chinnery(_u2_dip, xs, ps, qs, delta, W, L)
    u3 = _chinnery(_u3, xs, ps, qs, delta, W, L) + _chinnery(_u3_dip, xs, ps, qs, delta, W, L)

    # u2 is along Okada +Y (up-dip); the caller's frame uses +ys (down-dip), so flip.
    return u1, -u2, u3


class Cell:
    '''An even-width inversion unit: a group of fine Okada sub-patches that share
    one slip value.  Carries 2D plotting metadata (along-strike position, width,
    depth extent) and the fault-group label.'''

    def __init__(self, fault, slip_sign, x_along, width, z_top, z_bot):
        self.fault = fault
        self.slip_sign = slip_sign
        self.x_along = x_along
        self.width = width
        self.z_top = z_top
        self.z_bot = z_bot
        self.patches = []


class Patch:
    '''A single dipping rectangular Okada patch.

    (x0,y0) and (x1,y1) are the along-strike endpoints of the *upper* edge (already
    offset down-dip for deeper rows); z1 is the upper-edge depth, z0 the lower-edge
    depth (both positive down); ``dip`` is in radians.  The patch extends down-dip
    toward the +ys side (strike rotated +90 CCW), chosen at build time to match
    ``dip_az``.'''

    def __init__(self, x0, y0, z0, x1, y1, z1, dip=np.pi/2, dip_az=None,
                 fault=None, slip_sign=1):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.dip = dip
        self.dip_az = dip_az
        self.fault = fault
        self.slip_sign = slip_sign

    def get_dip(self):
        return self.dip

    def get_dd_width(self):
        return (self.z0 - self.z1) / np.sin(self.dip)

    def get_along_strike_length(self):
        return np.sqrt((self.x1-self.x0)**2 + (self.y1-self.y0)**2)

    def get_area(self):
        return self.get_dd_width() * self.get_along_strike_length()

    def corners3d(self):
        '''Return the 4 corner (x, y, z) tuples: upper-A, upper-B, lower-B, lower-A.'''
        L = self.get_along_strike_length()
        if L == 0:
            sx = sy = 0.0
        else:
            sx = (self.x1 - self.x0) / L
            sy = (self.y1 - self.y0) / L
        nx, ny = -sy, sx                       # +ys down-dip horizontal unit
        run = (self.z0 - self.z1) * np.cos(self.dip) / np.sin(self.dip)
        A = (self.x0, self.y0, self.z1)
        B = (self.x1, self.y1, self.z1)
        Bl = (self.x1 + nx*run, self.y1 + ny*run, self.z0)
        Al = (self.x0 + nx*run, self.y0 + ny*run, self.z0)
        return [A, B, Bl, Al]
