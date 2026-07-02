###   Imports
import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import h5py
import os
import copy


###   Ramp class
class Ramp:

    '''
    Builds a 3-parameter linear ramp basis for a dataset.

    The three basis functions are [1, x_norm, y_norm], where x and y are
    centred and scaled to [-0.5, 0.5] over the dataset extent. Normalising
    prevents ill-conditioning when coordinates are in UTM metres.

    Usage:
        R = Ramp.build_gfs(x, y)   # (N, 3)
    '''

    @staticmethod
    def build_gfs(x, y):
        '''
        Build the (N, 3) ramp design matrix [1, x_norm, y_norm].

        Args:
            x, y (ndarray): Pixel coordinates, length N.

        Returns:
            ndarray: Shape (N, 3).
        '''
        x_range = max(float(x.max() - x.min()), 1.)
        y_range = max(float(y.max() - y.min()), 1.)
        x_n = (x - float(x.mean())) / x_range
        y_n = (y - float(y.mean())) / y_range
        return np.column_stack([np.ones(len(x)), x_n, y_n])


###   InversionManager class
class InversionManager:

    '''
    Assembles multiple geodetic datasets for joint weighted least-squares inversion.

    Supports InSAR, GNSS, and OpticalData objects. Each dataset must have its
    Green's functions and covariance matrix computed before being registered.

    The joint system assembled across all datasets:
        d  = [d_1; d_2; ...]               data vector       (N_total,)
        G  = [G_fault | R_1 | R_2 | ...]   design matrix     (N_total, n_params)
        Cd = block_diag(Cd_1, Cd_2, ...)   data covariance   (N_total, N_total)

    where ramp columns R_i are zero except in the row block for dataset i.

    The weighted least-squares solution is found by Cholesky-whitening:
        L^{-1} G m = L^{-1} d   (Cd = L L^T)
    then solved via scipy.linalg.lstsq (unconstrained) or
    scipy.optimize.lsq_linear (bounded, e.g. non-negative slip).

    Properties:
        datasets (dict):        Labelled dataset objects.
        d  (ndarray):           Assembled data vector  (N_total,).
        G  (ndarray):           Design matrix          (N_total, n_params).
        Cd (ndarray):           Covariance matrix      (N_total, N_total).
        m  (ndarray):           Estimated model        (n_params,).
        Cm (ndarray):           Model covariance       (n_params, n_params).
        m_std (ndarray):        Per-parameter 1-sigma  (n_params,).
        residuals (dict):       Per-dataset residual vectors.
        _dataset_slices (dict): label -> row slice in d / G.
        _model_slices (dict):   param_name -> column slice in G / m.
        verbose (bool)
    '''

    ##  Initialise
    def __init__(self, verbose=True):
        self.verbose  = verbose
        self.datasets = {}
        self.d        = None
        self.G        = None
        self.Cd       = None
        self.m        = None
        self.Cm       = None
        self.m_std    = None
        self.residuals       = {}
        self._dataset_slices = {}
        self._dataset_types  = {}
        self._model_slices   = {}


    ##  Helpers
    def _print(self, *args):
        if self.verbose: print(*args)

    def _copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def _dataset_type_name(dataset):
        if hasattr(dataset, 'gfs_ew'): return 'OpticalData'
        if hasattr(dataset, 'de'):     return 'GNSS'
        return 'InSAR'


    ##  Extract (d, G, Cd) from a single dataset using duck typing
    def _extract_dataset(self, label, dataset):

        if hasattr(dataset, 'gfs_ew'):
            # OpticalData: block ordering d = [ew_vals, ns_vals]
            # gfs_ew / gfs_ns each (2, n_patches, N): axis-0 = [SS, DS]
            assert dataset.gfs_ew is not None, \
                f"'{label}': call compute_greens_functions() first"
            assert dataset.Cd_ew is not None, \
                f"'{label}': call build_Cd() first"
            d    = np.concatenate([dataset.ew_vals, dataset.ns_vals])
            G_ss = np.concatenate([dataset.gfs_ew[0], dataset.gfs_ns[0]], axis=1).T  # (2N, P)
            G_ds = np.concatenate([dataset.gfs_ew[1], dataset.gfs_ns[1]], axis=1).T  # (2N, P)
            G    = np.hstack([G_ss, G_ds])                                            # (2N, 2P)
            Cd   = block_diag(dataset.Cd_ew, dataset.Cd_ns)

        elif hasattr(dataset, 'de'):
            # GNSS: interleaved d = [de_0, dn_0, de_1, dn_1, ...]
            # gfs (2, n_patches, 2*N): axis-0 = [SS, DS]
            assert dataset.gfs is not None, \
                f"'{label}': call compute_greens_functions() first"
            assert dataset.Cd is not None, \
                f"'{label}': call compute_covariance() first"
            N  = len(dataset.de)
            d  = np.empty(2 * N)
            d[0::2] = dataset.de
            d[1::2] = dataset.dn
            G  = np.hstack([dataset.gfs[0].T, dataset.gfs[1].T])  # (2N, 2P)
            Cd = dataset.Cd

        else:
            # InSAR
            # gfs (2, n_patches, N): axis-0 = [SS, DS]
            assert dataset.gfs is not None, \
                f"'{label}': call compute_greens_functions() first"
            assert dataset.Cd is not None, \
                f"'{label}': call build_Cd() first"
            d  = dataset.vel
            G  = np.hstack([dataset.gfs[0].T, dataset.gfs[1].T])  # (N, 2P)
            Cd = dataset.Cd

        return d, G, Cd


    ##  Get (x, y) arrays sized to match the dataset data vector
    def _get_ramp_xy(self, label, dataset):
        '''Return x, y coordinates with length matching the data vector.'''

        if hasattr(dataset, 'gfs_ew'):
            # OpticalData block ordering [ew, ns]: tile coords twice
            return (np.concatenate([dataset.x, dataset.x]),
                    np.concatenate([dataset.y, dataset.y]))

        elif hasattr(dataset, 'de'):
            # GNSS interleaved [de, dn]: repeat each station coordinate
            x_rep = np.empty(2 * len(dataset.x))
            y_rep = np.empty(2 * len(dataset.y))
            x_rep[0::2] = x_rep[1::2] = dataset.x
            y_rep[0::2] = y_rep[1::2] = dataset.y
            return x_rep, y_rep

        else:
            # InSAR: direct coordinates
            return dataset.x, dataset.y


    ##  Register datasets and assemble the joint system
    def set_datasets(self, datasets, solve_for_ramp=False):

        '''
        Register datasets and assemble the joint d, G, and Cd.

        Args:
            datasets (dict): Mapping of label (str) -> dataset object (InSAR,
                GNSS, or OpticalData). Each must have .gfs and .Cd (or
                .gfs_ew/.gfs_ns and .Cd_ew/.Cd_ns) already computed.

        Kwargs:
            solve_for_ramp (bool or list): If True, append a 3-parameter linear
                ramp [1, x_norm, y_norm] for every dataset. Pass a list of
                dataset labels to apply ramps selectively. Each ramp adds 3
                independent columns to G (zeroed outside its own dataset rows),
                contributing 3 extra model parameters per ramp dataset.

        Returns:
            self
        '''

        self.datasets        = datasets
        self._dataset_slices = {}
        self._dataset_types  = {}
        self._model_slices   = {}

        # Resolve which datasets get a ramp
        if solve_for_ramp is True:
            ramp_labels = set(datasets.keys())
        elif solve_for_ramp:
            ramp_labels = set(solve_for_ramp)
        else:
            ramp_labels = set()

        d_parts, G_parts, Cd_parts = [], [], []
        offset = 0

        for label, ds in datasets.items():
            d, G, Cd = self._extract_dataset(label, ds)
            self._dataset_types[label]  = self._dataset_type_name(ds)
            self._dataset_slices[label] = slice(offset, offset + len(d))
            offset += len(d)
            self._print(f"  '{label}' ({self._dataset_types[label]}): "
                        f"{len(d)} obs, G {G.shape}, Cd {Cd.shape}")
            d_parts.append(d)
            G_parts.append(G)
            Cd_parts.append(Cd)

        N_total        = offset
        G_fault        = np.vstack(G_parts)          # (N_total, 2*n_patches)
        n_fault_params = G_fault.shape[1]
        n_patches      = n_fault_params // 2

        self._model_slices['fault_ss'] = slice(0, n_patches)
        self._model_slices['fault_ds'] = slice(n_patches, n_fault_params)

        # Append ramp columns (block-diagonal across datasets)
        ramp_col_parts = []
        ramp_offset    = n_fault_params

        for label, ds in datasets.items():
            if label not in ramp_labels:
                continue
            x_r, y_r = self._get_ramp_xy(label, ds)
            R_i  = Ramp.build_gfs(x_r, y_r)              # (N_i, 3)
            R_col = np.zeros((N_total, 3))
            R_col[self._dataset_slices[label]] = R_i
            ramp_col_parts.append(R_col)
            ramp_key = f'ramp:{label}'
            self._model_slices[ramp_key] = slice(ramp_offset, ramp_offset + 3)
            ramp_offset += 3
            self._print(f"  '{label}': ramp added (cols {self._model_slices[ramp_key]})")

        if ramp_col_parts:
            self.G = np.hstack([G_fault] + ramp_col_parts)
        else:
            self.G = G_fault

        self.d  = np.concatenate(d_parts)
        self.Cd = block_diag(*Cd_parts)

        self._print(f"Assembled: {len(self.d)} obs, "
                    f"{self.G.shape[1]} model params "
                    f"({n_patches} SS + {n_patches} DS patches + {ramp_offset - n_fault_params} ramp)")
        return self


    ##  Print a human-readable summary of the assembled system
    def summary(self):

        '''Print a summary of datasets, observation ranges, and model parameters.'''

        if self.d is None:
            print("No datasets assembled — call set_datasets() first.")
            return

        print("=" * 60)
        print("InversionManager summary")
        print("=" * 60)
        print(f"  Total observations : {len(self.d)}")
        print(f"  Total model params : {self.G.shape[1]}")
        print()

        # Dataset table
        col_w = max(len(k) for k in self._dataset_slices) + 2
        header = f"  {'Dataset':<{col_w}}  {'Type':<12}  {'N obs':>6}  {'Row slice'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for label, sl in self._dataset_slices.items():
            dtype = self._dataset_types.get(label, '?')
            n_obs = sl.stop - sl.start
            print(f"  {label:<{col_w}}  {dtype:<12}  {n_obs:>6}  {sl.start}:{sl.stop}")

        print()

        # Model parameter table
        print(f"  {'Parameter':<30}  {'Col slice'}")
        print("  " + "-" * 45)
        for name, sl in self._model_slices.items():
            n_params = sl.stop - sl.start
            print(f"  {name:<30}  {sl.start}:{sl.stop}  ({n_params} params)")

        if self.m is not None:
            print()
            print("  Inversion result:")
            for name, sl in self._model_slices.items():
                vals = self.m[sl]
                std  = self.m_std[sl] if self.m_std is not None else None
                print(f"    {name}: "
                      f"min={vals.min():.4f}, max={vals.max():.4f}, "
                      f"mean={vals.mean():.4f}"
                      + (f",  std min/max={std.min():.4f}/{std.max():.4f}"
                         if std is not None else ""))
            print()
            for label, res in self.residuals.items():
                rms = float(np.sqrt(np.mean(res**2)))
                print(f"    '{label}' residual RMS: {rms:.6f}")

        print("=" * 60)


    ##  Weighted least-squares inversion
    def run(self, bounds=None, nugget=0.):

        '''
        Run a weighted least-squares inversion.

        Cholesky-whitens the system (Cd = L L^T) and solves L^{-1} G m = L^{-1} d
        via scipy.linalg.lstsq (unconstrained) or scipy.optimize.lsq_linear
        (when bounds are supplied).

        Kwargs:
            bounds (tuple or None): (lower, upper) bounds on model parameters,
                passed directly to scipy.optimize.lsq_linear. Use scalars for
                uniform bounds (e.g. bounds=(0, np.inf) for non-negative slip)
                or arrays of length n_params for per-parameter bounds.
            nugget (float): Small value added to the Cd diagonal before
                factorisation to improve conditioning (default 0).

        Returns:
            self
        '''

        assert self.d is not None, "Call set_datasets() first."

        self._print("Running inversion...")
        self._print(f"  System: {self.G.shape[0]} obs x {self.G.shape[1]} params")

        Cd = self.Cd
        if nugget > 0.:
            Cd = Cd + nugget * np.eye(len(Cd))

        L   = scipy.linalg.cholesky(Cd, lower=True)
        G_w = scipy.linalg.solve_triangular(L, self.G, lower=True)
        d_w = scipy.linalg.solve_triangular(L, self.d, lower=True)

        if bounds is None:
            m, _, rank, _ = scipy.linalg.lstsq(G_w, d_w)
            self._print(f"  Matrix rank: {rank}")
        else:
            res = scipy.optimize.lsq_linear(G_w, d_w, bounds=bounds,
                                            method='bvls', verbose=0)
            m = res.x
            self._print(f"  lsq_linear cost: {res.cost:.6e}, "
                        f"status: {res.message}")

        GtG        = G_w.T @ G_w
        self.Cm    = scipy.linalg.inv(GtG)
        self.m     = m
        self.m_std = np.sqrt(np.diag(self.Cm))

        predicted = self.G @ m
        for label, sl in self._dataset_slices.items():
            res_vec = self.d[sl] - predicted[sl]
            self.residuals[label] = res_vec
            rms = float(np.sqrt(np.mean(res_vec**2)))
            self._print(f"  '{label}' residual RMS: {rms:.6f}")

        return self


    ##  Save assembled system to separate HDF5 files
    def save_to_hdf5(self, directory,
                     data_filename='data.h5',
                     covariance_filename='covariance.h5',
                     gf_filename='gf.h5'):

        '''
        Save the assembled d, Cd, and G to three separate HDF5 files.

        Files written (each contains a single dataset at the root key):
            data_filename       — /data        (N_total,)
            covariance_filename — /covariance  (N_total, N_total)
            gf_filename         — /gf          (N_total, n_params)

        Dataset-slice and model-slice metadata is stored in the data file
        under /datasets/<label> and /model_slices/<name>.

        Args:
            directory (str): Output directory.

        Kwargs:
            data_filename (str):       Filename for the data vector     (default 'data.h5').
            covariance_filename (str): Filename for the covariance matrix (default 'covariance.h5').
            gf_filename (str):         Filename for the design matrix    (default 'gf.h5').

        Returns:
            self
        '''

        assert self.d is not None, "Call set_datasets() first."

        data_path = os.path.join(directory, data_filename)
        cov_path  = os.path.join(directory, covariance_filename)
        gf_path   = os.path.join(directory, gf_filename)

        self._print(f"Saving data       -> {data_path}")
        with h5py.File(data_path, 'w') as fh:
            fh.create_dataset('data', data=self.d, compression='gzip')
            grp_ds = fh.create_group('datasets')
            for label, sl in self._dataset_slices.items():
                grp_ds.create_dataset(label, data=[sl.start, sl.stop])
            grp_ms = fh.create_group('model_slices')
            for name, sl in self._model_slices.items():
                grp_ms.create_dataset(name, data=[sl.start, sl.stop])

        self._print(f"Saving covariance -> {cov_path}")
        with h5py.File(cov_path, 'w') as fh:
            fh.create_dataset('covariance', data=self.Cd, compression='gzip')

        self._print(f"Saving gf         -> {gf_path}")
        with h5py.File(gf_path, 'w') as fh:
            fh.create_dataset('gf', data=self.G, compression='gzip')

        self._print(f"  d {self.d.shape}, Cd {self.Cd.shape}, G {self.G.shape}")
        return self


    ##  Internal helpers for map plotting

    def _get_proj(self):
        '''Return a pyproj.Proj from the first dataset that has one.'''
        for ds in self.datasets.values():
            if hasattr(ds, '_proj'):
                return ds._proj
        return None

    @staticmethod
    def _add_scalebar(ax):
        '''Add an auto-sized map scalebar (lower-left) to a lon/lat axes.'''
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lat_c      = (y0 + y1) / 2.
        km_per_deg = np.cos(np.radians(lat_c)) * 111.32
        target_km  = 0.2 * (x1 - x0) * km_per_deg
        nice_km    = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        bar_km     = min(nice_km, key=lambda v: abs(v - target_km))
        bar_deg    = bar_km / km_per_deg
        xr, yr     = x1 - x0, y1 - y0
        bx         = x0 + 0.05 * xr
        by         = y0 + 0.05 * yr
        tick_h     = 0.012 * yr
        kw = dict(color='k', lw=2.5, zorder=5, solid_capstyle='butt')
        ax.plot([bx, bx + bar_deg], [by, by], **kw)
        ax.plot([bx, bx],           [by, by + tick_h], **kw)
        ax.plot([bx + bar_deg, bx + bar_deg], [by, by + tick_h], **kw)
        ax.text(bx + bar_deg / 2., by + 0.015 * yr, f'{bar_km:.0f} km',
                ha='center', va='bottom', fontsize=8, zorder=5)

    @staticmethod
    def _add_inset_colorbar(ax, sc, label):
        '''Add an inset colorbar (upper-right) with a white background.'''
        from matplotlib.patches import FancyBboxPatch
        bg = FancyBboxPatch((0.60, 0.81), 0.36, 0.16,
                            boxstyle='square,pad=0.01',
                            transform=ax.transAxes,
                            facecolor='white', edgecolor='lightgray',
                            linewidth=0.5, zorder=4, clip_on=False)
        ax.add_patch(bg)
        cax = ax.inset_axes([0.62, 0.91, 0.32, 0.04])
        cax.set_zorder(5)
        cb = plt.colorbar(sc, cax=cax, orientation='horizontal')
        cb.set_label(label, fontsize=9, labelpad=2)
        cax.tick_params(labelsize=7)

    def _plot_fault_trace(self, ax, fault):
        '''Plot fault trace(s) (UTM) reprojected to lon/lat on ax.'''
        if fault is None:
            return
        faults = fault if isinstance(fault, list) else [fault]
        proj   = self._get_proj()
        from shapely.geometry import LineString as _LS
        for f in faults:
            if not hasattr(f, 'trace'):
                continue
            for geom in f.trace.geometry:
                lines = [geom] if isinstance(geom, _LS) else list(geom.geoms)
                for line in lines:
                    coords = np.array(line.coords)
                    if proj is not None:
                        lons_t, lats_t = proj(coords[:, 0], coords[:, 1], inverse=True)
                    else:
                        lons_t, lats_t = coords[:, 0], coords[:, 1]
                    ax.plot(lons_t, lats_t, 'k-', lw=1.5, zorder=4)

    def _plot_geodetic_map(self, ax, values, raster_label, optical_component,
                           gnss_labels, fault, cmap, vlim, gnss_scale,
                           colorbar_label, title, ref_m=None):
        '''
        Core map-plotting implementation shared by plot_map() and plot_residuals().

        values : dict  label -> 1-D ndarray in dataset data-vector ordering:
                   InSAR       (N,)   LOS values
                   OpticalData (2N,)  concat([ew_vals, ns_vals])
                   GNSS        (2N,)  interleaved [de_0, dn_0, de_1, dn_1, ...]
        '''
        import cmcrameri.cm as cmc

        if cmap is None:
            cmap = cmc.vik

        raster_types   = {'InSAR', 'OpticalData'}
        raster_labels  = [l for l, t in self._dataset_types.items() if t in raster_types]
        gnss_ds_labels = [l for l, t in self._dataset_types.items() if t == 'GNSS']

        if raster_label is None and raster_labels:
            raster_label = raster_labels[0]
        if gnss_labels is None:
            gnss_labels = gnss_ds_labels

        sc       = None
        cb_label = colorbar_label

        # ---- Raster / scatter background ----
        if raster_label is not None and raster_label in values:
            ds    = self.datasets[raster_label]
            dtype = self._dataset_types[raster_label]
            vals  = values[raster_label]

            if dtype == 'OpticalData':
                N    = len(ds.x)
                vals = vals[:N] if optical_component == 'ew' else vals[N:]
                if cb_label is None:
                    comp     = 'EW' if optical_component == 'ew' else 'NS'
                    cb_label = f'{comp} displacement (m)'
            else:
                if cb_label is None:
                    cb_label = 'LOS displacement (m)'

            if vlim is None:
                p = float(np.nanpercentile(np.abs(vals), 98))
                vmin, vmax = -p, p
            else:
                vmin, vmax = vlim

            sc = ax.scatter(ds.lon, ds.lat, c=vals, cmap=cmap,
                            s=2, vmin=vmin, vmax=vmax,
                            rasterized=True, zorder=2)

        # ---- GNSS arrows ----
        # Unified scale across all GNSS datasets
        ref_lon_range = 1.
        if raster_label is not None and raster_label in self.datasets:
            rds = self.datasets[raster_label]
            ref_lon_range = float(rds.lon.max() - rds.lon.min()) or 1.
        elif gnss_labels:
            all_lons = np.concatenate([self.datasets[l].lon for l in gnss_labels
                                       if l in self.datasets])
            ref_lon_range = float(all_lons.max() - all_lons.min()) or 1.

        if gnss_scale is None:
            all_max = max(
                (float(np.max(np.hypot(values[l][0::2], values[l][1::2])))
                 for l in gnss_labels if l in values),
                default=0.
            )
            _scale = all_max / (0.06 * ref_lon_range) if all_max > 0. else 1.
        else:
            _scale = gnss_scale

        gnss_ref_done = False
        for label in gnss_labels:
            if label not in values:
                continue
            ds      = self.datasets[label]
            res     = values[label]
            de_vals = res[0::2]
            dn_vals = res[1::2]

            ax.quiver(ds.lon, ds.lat, de_vals, dn_vals,
                      angles='xy', scale_units='xy', scale=_scale,
                      color='black', width=0.003, zorder=3)

            if not gnss_ref_done:
                # Reference arrow — placed at lower-right of the full data extent
                max_de = float(np.max(np.abs(de_vals)))
                if ref_m is None:
                    ref_m  = 10. ** np.floor(np.log10(max(max_de, 1e-9)))
                print("ref m", ref_m, max_de)
                all_lons_list = [self.datasets[l].lon for l in gnss_labels
                                 if l in self.datasets]
                all_lats_list = [self.datasets[l].lat for l in gnss_labels
                                 if l in self.datasets]
                if raster_label is not None and raster_label in self.datasets:
                    all_lons_list.append(self.datasets[raster_label].lon)
                    all_lats_list.append(self.datasets[raster_label].lat)
                all_lons_arr = np.concatenate(all_lons_list)
                all_lats_arr = np.concatenate(all_lats_list)
                lon0, lon1   = all_lons_arr.min(), all_lons_arr.max()
                lat0, lat1   = all_lats_arr.min(), all_lats_arr.max()
                lonr, latr   = lon1 - lon0, lat1 - lat0
                ref_len_deg  = ref_m / _scale
                rx = lon1 - 0.08 * lonr - ref_len_deg
                ry = lat0 + 0.05 * latr
                ax.quiver(rx, ry, ref_m, 0.,
                          angles='xy', scale_units='xy', scale=_scale,
                          color='black', width=0.003, zorder=5)
                ax.text(rx + ref_len_deg / 2., ry - 0.015 * latr,
                        f'{ref_m * 1000:.0f} mm',
                        ha='center', va='top', fontsize=8, zorder=5)
                gnss_ref_done = True

        # ---- Fault trace ----
        self._plot_fault_trace(ax, fault)

        # ---- Colorbar ----
        if sc is not None:
            self._add_inset_colorbar(ax, sc, cb_label)

        # ---- Labels & scalebar ----
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        if title is not None:
            ax.set_title(title)
        self._add_scalebar(ax)


    ##  Plot data map
    def plot_map(self, ax=None, raster_label=None, optical_component='ew',
                 gnss_labels=None, fault=None,
                 cmap=None, vlim=None, gnss_scale=None,
                 colorbar_label=None, title=None, gnss_arrow_length_m=None):

        '''
        Plot a geographic overview map of the registered datasets.

        One pixel-based dataset (InSAR or OpticalData) is shown as a scatter
        background.  All GNSS datasets (or a specified subset) are overlaid as
        displacement arrows.  An optional fault trace and map scalebar are drawn.

        Kwargs:
            ax (Axes or None): Axes to plot on; new figure created if None.
            raster_label (str or None): Label of the InSAR / OpticalData dataset
                to use as the scatter background.  Defaults to the first
                registered raster-type dataset.
            optical_component (str): 'ew' or 'ns' — component to display when
                the raster dataset is OpticalData (default 'ew').
            gnss_labels (list or None): Labels of GNSS datasets to overlay.
                Defaults to all registered GNSS datasets.
            fault: Fault3d object (or list thereof) for the fault trace.
            cmap: Matplotlib colormap (default cmc.vik).
            vlim (tuple or None): (vmin, vmax) colour limits; auto ±98th
                percentile if None.
            gnss_scale (float or None): Quiver scale (data units per axis unit).
                Auto-computed if None.
            colorbar_label (str or None): Inset colorbar label.
            title (str or None): Axes title.

        Returns:
            (fig, ax) if a new figure was created, otherwise ax.
        '''

        assert self.datasets, "No datasets registered — call set_datasets() first."

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')

        values = {}
        for label, ds in self.datasets.items():
            dtype = self._dataset_types.get(label, self._dataset_type_name(ds))
            if dtype == 'InSAR':
                values[label] = ds.vel
            elif dtype == 'OpticalData':
                values[label] = np.concatenate([ds.ew_vals, ds.ns_vals])
            elif dtype == 'GNSS':
                N = len(ds.de)
                v = np.empty(2 * N)
                v[0::2] = ds.de
                v[1::2] = ds.dn
                values[label] = v

        self._plot_geodetic_map(ax, values, raster_label, optical_component,
                                gnss_labels, fault, cmap, vlim, gnss_scale,
                                colorbar_label, title, ref_m=gnss_arrow_length_m)
        if created_fig:
            return fig, ax
        return ax


    ##  Plot residuals map
    def plot_residuals(self, ax=None, raster_label=None, optical_component='ew',
                       gnss_labels=None, fault=None,
                       cmap=None, vlim=None, gnss_scale=None,
                       colorbar_label=None, title=None, gnss_arrow_length_m=None):

        '''
        Plot a geographic map of post-inversion residuals (data − predicted).

        Same signature as plot_map(); residuals from self.residuals are used in
        place of raw data values.  Colorbar label defaults to 'Residual (m)'.

        Returns:
            (fig, ax) if a new figure was created, otherwise ax.
        '''

        assert self.residuals, "No residuals — call run() first."

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')

        if colorbar_label is None:
            colorbar_label = 'Residual (m)'

        self._plot_geodetic_map(ax, self.residuals, raster_label, optical_component,
                                gnss_labels, fault, cmap, vlim, gnss_scale,
                                colorbar_label, title, ref_m=gnss_arrow_length_m)
        if created_fig:
            return fig, ax
        return ax


    ##  Load assembled system from separate HDF5 files
    @classmethod
    def load_from_hdf5(cls, directory,
                       data_filename='data.h5',
                       covariance_filename='covariance.h5',
                       gf_filename='gf.h5',
                       verbose=True):

        '''
        Reconstruct an InversionManager from files written by save_to_hdf5().

        Args:
            directory (str): Directory containing the three HDF5 files.

        Kwargs:
            data_filename (str):       Filename for the data vector.
            covariance_filename (str): Filename for the covariance matrix.
            gf_filename (str):         Filename for the design matrix.
            verbose (bool)

        Returns:
            InversionManager with d, G, Cd, _dataset_slices, and
            _model_slices populated.
        '''

        obj = cls(verbose=verbose)

        data_path = os.path.join(directory, data_filename)
        cov_path  = os.path.join(directory, covariance_filename)
        gf_path   = os.path.join(directory, gf_filename)

        obj._print(f"Loading data       <- {data_path}")
        with h5py.File(data_path, 'r') as fh:
            obj.d = fh['data'][:]
            if 'datasets' in fh:
                for label, ds in fh['datasets'].items():
                    obj._dataset_slices[label] = slice(int(ds[0]), int(ds[1]))
            if 'model_slices' in fh:
                for name, ds in fh['model_slices'].items():
                    obj._model_slices[name] = slice(int(ds[0]), int(ds[1]))

        obj._print(f"Loading covariance <- {cov_path}")
        with h5py.File(cov_path, 'r') as fh:
            obj.Cd = fh['covariance'][:]

        obj._print(f"Loading gf         <- {gf_path}")
        with h5py.File(gf_path, 'r') as fh:
            obj.G = fh['gf'][:]

        obj._print(f"  d {obj.d.shape}, Cd {obj.Cd.shape}, G {obj.G.shape}")
        return obj
