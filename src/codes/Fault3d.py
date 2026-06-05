import geopandas as gpd
import numpy as np
import copy

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
                    patches[(j+k, i)] = patch
                k += n_patches_horizontal
        
        self.patches = patches
    
    def compute_greens_functions(self, pts):

        n_patches = len(self.patches)
        n_pts = pts.shape[1]
        gfs = np.zeros((n_patches, 3, n_pts))

        for i, (id, patch) in enumerate(self.patches.items()):

            ##  Rotate eval pts into reference frame  ##
            _pts = copy.deepcopy(pts)
            _xs = pts[0] - patch.x0
            _ys = pts[1] - patch.y0

            patch_strike = np.arctan2(patch.y1-patch.y0, patch.x1-patch.x0)
            cos_s = np.cos(patch_strike)
            sin_s = np.sin(patch_strike)

            xs = cos_s * _xs + sin_s * _ys
            ys = -sin_s * _xs + cos_s * _ys


            ##  Compute okada  ##
            u1, u2, u3 = compute_okada(patch, np.array([1., 0.]), xs, ys)


            ##  Rotate okada displacements back to NS/EW  ##
            ux = cos_s * u1 - sin_s * u2
            uy = sin_s * u1 + cos_s * u2

            u = np.vstack((ux, uy, u3))
            gfs[i] = u
        
        self.gfs = gfs

            


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
    
    delta = patch.get_dip()
    d = patch.z0
    ps = ys*np.cos(delta) + d*np.sin(delta)
    qs = ys*np.sin(delta) - d * np.cos(delta)
    W = patch.get_dd_width()
    L = patch.get_along_strike_length()

    u1 = _u1(xs, ps, qs, delta) - _u1(xs, ps-W, qs, delta) - _u1(xs-L, ps, qs, delta) + _u1(xs-L, ps-W, qs, delta)
    u2 = _u2(xs, ps, qs, delta) - _u2(xs, ps-W, qs, delta) - _u2(xs-L, ps, qs, delta) + _u2(xs-L, ps-W, qs, delta)
    u3 = _u3(xs, ps, qs, delta) - _u3(xs, ps-W, qs, delta) - _u3(xs-L, ps, qs, delta) + _u3(xs-L, ps-W, qs, delta)

    return u1, u2, u3



class Patch:
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
    
    def get_dip(self):
        return np.pi/2
    
    def get_dd_width(self):
        dip = self.get_dip()
        return (self.z1-self.z0) / np.sin(dip)
    
    def get_along_strike_length(self):
        return np.sqrt((self.x1-self.x0)**2 + (self.y1-self.y0)**2)
    
    def get_area(self):
        return self.get_dd_width() * self.get_along_strike_length()