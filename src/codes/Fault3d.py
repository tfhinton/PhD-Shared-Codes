import geopandas as gpd
import numpy as np

class Fault3d:

    def __init__(self, trace_shp_filepath=None, vertical_discretisation=None):
        if trace_shp_filepath is not None:
            self.import_trace(trace_shp_filepath)
            if vertical_discretisation is not None:
                self.build_patches(vertical_discretisation)

    def import_trace(self, shp_filepath):
        gdf = gpd.read_file(shp_filepath)
        gdf = gdf.set_index("id")
        self.trace = gdf
    
    def build_patches(self, vertical_discretisation):
        n_patches_vertical = len(vertical_discretisation) - 1
        patches = {}

        for i in range(n_patches_vertical):
            z1 = vertical_discretisation[i]
            z0 = vertical_discretisation[i+1]

            k = 0
            for line_id, ls in self.trace["geometry"].items():
                n_patches_horizontal = len(ls.coords) - 1
                for j in range(n_patches_horizontal):
                    x0, y0 = ls.coords[j]
                    x1, y1 = ls.coords[j+1]
                    
                    patch = Patch(x0, y0, z0, x1, y1, z1)
                    patch.line_id = line_id
                    patch.slip_sign = int(self.trace.loc[line_id, "slip_sign"]) if "slip_sign" in self.trace.columns else 1

                    patches[(j+k, i)] = patch
                k += n_patches_horizontal
        
        self.patches = patches
        self.slips = np.zeros(len(patches))
    
    def plot_slip(self, ax=None, cmap='plasma', vmin=None, vmax=None,
                 sigma_grey=0.5, edgecolor='k', linewidth=0.5, patch_indices=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        from matplotlib.collections import PatchCollection
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        has_sigmas = hasattr(self, 'sigmas') and self.sigmas is not None

        # Compute cumulative along-strike x position for each horizontal patch index
        h_indices = sorted(set(k[0] for k in self.patches.keys()))
        v_indices = sorted(set(k[1] for k in self.patches.keys()))

        along_strike_x = {}
        x_pos = 0.0
        for h in h_indices:
            along_strike_x[h] = x_pos
            x_pos += self.patches[(h, v_indices[0])].get_along_strike_length()

        # Build boolean selection mask (default: all patches)
        n = len(self.patches)
        if patch_indices is None:
            mask = np.ones(n, dtype=bool)
        else:
            idx = np.asarray(patch_indices)
            if idx.dtype == bool:
                mask = idx
            else:
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True

        # Build rectangles in the same iteration order as self.slips
        rects = []
        for (h, _), patch in self.patches.items():
            x = along_strike_x[h]
            w = patch.get_along_strike_length()
            z_top = min(patch.z0, patch.z1)
            z_bot = max(patch.z0, patch.z1)
            rects.append(mpatches.Rectangle((x, z_top), w, z_bot - z_top))

        rects = [r for r, m in zip(rects, mask) if m]
        slip_vals = np.asarray(self.slips)[mask]
        if vmin is None:
            vmin = slip_vals.min()
        if vmax is None:
            vmax = slip_vals.max()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        face_colors = cmap_obj(norm(slip_vals))  # (N, 4) RGBA

        if has_sigmas:
            sigmas = np.asarray(self.sigmas)[mask]
            s_range = sigmas.max() - sigmas.min()
            sigma_norm = (sigmas - sigmas.min()) / (s_range if s_range > 0 else 1.0)
            grey_rgba = np.array([sigma_grey, sigma_grey, sigma_grey, 1.0])
            face_colors = (1.0 - sigma_norm[:, None]) * face_colors + sigma_norm[:, None] * grey_rgba

        pc = PatchCollection(rects, facecolor=face_colors, edgecolor=edgecolor, linewidth=linewidth)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.invert_yaxis()

        sm = ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Slip')

        ax.set_xlabel('Along-strike distance')
        ax.set_ylabel('Depth')

        return ax

    def compute_greens_functions(self, pts):

        n_patches = len(self.patches)
        n_pts = pts.shape[1]
        gfs_ss = np.zeros((n_patches, 3, n_pts))
        gfs_ds = np.zeros((n_patches, 3, n_pts))

        for i, (_, patch) in enumerate(self.patches.items()):

            ##  Rotate eval pts into reference frame  ##
            _xs = pts[0] - patch.x0
            _ys = pts[1] - patch.y0

            patch_strike = np.arctan2(patch.y1-patch.y0, patch.x1-patch.x0)
            cos_s = np.cos(patch_strike)
            sin_s = np.sin(patch_strike)

            xs = cos_s * _xs + sin_s * _ys
            ys = -sin_s * _xs + cos_s * _ys

            ##  Strike-slip  ##
            u1, u2, u3 = compute_okada(patch, np.array([patch.slip_sign, 0.]), xs, ys)
            ux = cos_s * u1 - sin_s * u2
            uy = sin_s * u1 + cos_s * u2
            gfs_ss[i] = np.vstack((ux, uy, u3))

            ##  Dip-slip  ##
            u1, u2, u3 = compute_okada(patch, np.array([0., 1.]), xs, ys)
            ux = cos_s * u1 - sin_s * u2
            uy = sin_s * u1 + cos_s * u2
            gfs_ds[i] = np.vstack((ux, uy, u3))

        self.gfs = np.stack([gfs_ss, gfs_ds], axis=0)  # (2, n_patches, 3, n_pts)
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

    delta = patch.get_dip()
    d = patch.z1
    ps = ys*np.cos(delta) + d*np.sin(delta)
    qs = ys*np.sin(delta) - d * np.cos(delta)
    W = patch.get_dd_width()
    L = patch.get_along_strike_length()

    def _chinnery(f, xi, ps, qs, delta, W, L):
        return f(xi, ps, qs, delta) - f(xi, ps-W, qs, delta) - f(xi-L, ps, qs, delta) + f(xi-L, ps-W, qs, delta)

    u1 = _chinnery(_u1, xs, ps, qs, delta, W, L) + _chinnery(_u1_dip, xs, ps, qs, delta, W, L)
    u2 = _chinnery(_u2, xs, ps, qs, delta, W, L) + _chinnery(_u2_dip, xs, ps, qs, delta, W, L)
    u3 = _chinnery(_u3, xs, ps, qs, delta, W, L) + _chinnery(_u3_dip, xs, ps, qs, delta, W, L)

    return u1, u2, u3



class Patch:
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.slip_sign = 1
    
    def get_dip(self):
        return np.pi/2
    
    def get_dd_width(self):
        dip = self.get_dip()
        return (self.z0-self.z1) / np.sin(dip)
    
    def get_along_strike_length(self):
        return np.sqrt((self.x1-self.x0)**2 + (self.y1-self.y0)**2)
    
    def get_area(self):
        return self.get_dd_width() * self.get_along_strike_length()