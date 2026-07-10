import numpy as np
import copy
import pickle
from pathlib import Path

from . import meade07


class FaultTriangles:
    """A triangular-mesh fault with Meade (2007) half-space Green's functions

    Properties
    ----------
    vertices : (V, 3) float
        z is up, free surface at z=0
    triangles : (T, 3) int
        indices of ``vertices``.
    layers : (T,) int or None
        If layered mesh, the layer index of each triangle; otherwise ``None``.
    slips : (T,) float
        Slip value per triangle
    gfs : (2, T, 3, n_pts) float
        Green's functions populated by `compute_greens_functions`.
    """

    def __init__(self, vertices, triangles, layers=None, nu=0.25, name="Fault"):
        self.vertices = np.asarray(vertices, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.layers = None if layers is None else np.asarray(layers, dtype=int)
        self.nu = float(nu)  # Poisson's ratio
        self.slips = np.zeros(len(self.triangles))
        self.gfs = None
        self.dip_reference = None  # Optional per-subfault reference normals fixing a consistent dip-slip
        self.fault_ids = np.zeros(len(self.triangles), dtype=int)  # Per-triangle subfault index
        self.name = name
        self.subfault_names = {0: name}

    ## Helper method to return a deep copy of the class instance
    def _copy(self):
        return copy.copy(self)

    ####    Merge several FaultTriangles into one    ####
    @classmethod
    def merge(cls, faults):
        faults = list(faults)
        V_parts, T_parts, L_parts, F_parts, n_parts = [], [], [], [], []
        voff = 0
        for k, f in enumerate(faults):
            V_parts.append(f.vertices)
            T_parts.append(f.triangles + voff)
            voff += len(f.vertices)
            L_parts.append(f.layers if f.layers is not None
                           else np.zeros(f.n_patches, dtype=int))
            F_parts.append(np.full(f.n_patches, k, dtype=int))
            n_parts.append(f.name)
        obj = cls(np.vstack(V_parts), np.vstack(T_parts),
                  layers=np.concatenate(L_parts),nu=faults[0].nu)
        obj.fault_ids = np.concatenate(F_parts)
        obj.subfault_names = {k: name for k, name in enumerate(n_parts)}
        return obj


    ####    GOCAD TSurf I/O    ####
    @classmethod
    def from_gocad(cls, path, name=None):
        """Read a GOCAD TSurf .txt (z-up elevation). Vertices are kept as-is."""
        verts, tris = {}, []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] in ("VRTX", "PVRTX"):
                    verts[int(parts[1])] = [float(p) for p in parts[2:5]]
                elif parts[0] == "TRGL":
                    tris.append([int(p) for p in parts[1:4]])
        ids = sorted(verts)                     # GOCAD ids may be 1-based/sparse
        index = {vid: i for i, vid in enumerate(ids)}
        vertices = np.array([verts[vid] for vid in ids])
        triangles = np.array([[index[i] for i in t] for t in tris])
        return cls(vertices, triangles, name=name or Path(path).stem)

    def write_gocad(self, path):
        """Write the mesh as a GOCAD TSurf .txt (ZPOSITIVE Elevation)."""
        lines = [
            "GOCAD TSurf 1",
            "HEADER {",
            f"name:{self.name}",
            "*visible:true",
            "}",
            "GOCAD_ORIGINAL_COORDINATE_SYSTEM",
            "NAME Default",
            'AXIS_NAME "X" "Y" "Z"',
            'AXIS_UNIT "m" "m" "m"',
            "ZPOSITIVE Elevation",
            "END_ORIGINAL_COORDINATE_SYSTEM",
            "TFACE",
        ]
        for i, (x, y, z) in enumerate(self.vertices, start=1):
            lines.append(f"VRTX {i} {x:.6f} {y:.6f} {z:.6f}")
        for a, b, c in self.triangles + 1:      # GOCAD is 1-based
            lines.append(f"TRGL {a} {b} {c}")
        lines.append("END")
        Path(path).write_text("\n".join(lines) + "\n")
        return self

    def save(self, path):
        """Save as <path>.pickle (full object) and <path>.txt (GOCAD, geometry only)."""
        path = Path(path)
        with open(path.with_suffix(".pickle"), "wb") as fh:
            pickle.dump(self, fh)
        self.write_gocad(path.with_suffix(".txt"))
        return self


    ####   Basic triangle geometry    ####
    @property
    def n_patches(self):
        return len(self.triangles)
    
    @property
    def n_subfaults(self):
        return len(np.unique(self.fault_ids))
    
    def triangle_xyz(self, i):
        """(3, 3) vertices of triangle ``i``, z-up frame (rows are x, y, z)."""
        return self.vertices[self.triangles[i]]

    @property
    def centroids(self):
        """(T, 3) triangle centroids, z-up frame."""
        return self.vertices[self.triangles].mean(axis=1)

    @property
    def areas(self):
        """(T,) triangle areas (m^2) = 0.5 |(v1-v0) x (v2-v0)|."""
        v = self.vertices[self.triangles]
        a = v[:, 1] - v[:, 0]
        b = v[:, 2] - v[:, 0]
        return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)

    @property
    def depths(self):
        """(T,) mean depth, positive-down (metres below datum)."""
        return -self.vertices[self.triangles][:, :, 2].mean(axis=1)


    #### Surface trace GeoDataFrame (the .trace interface InSAR/InversionManager expect)
    @property
    def trace(self):
        return self.get_surface_trace_xy()

    def get_surface_trace_xy(self, tol=1.0):
        import geopandas as gpd
        from shapely.geometry import LineString
        geoms = []
        for i in range(self.n_subfaults):
            vertices = self.vertices[self.triangles[self.fault_ids == i]].reshape(-1, 3)
            z = vertices[:, 2]
            top = z > (z.max() - tol)
            pts = vertices[top][:, :2]
            c = pts - pts.mean(axis=0)
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            order = np.argsort(c @ vt[0])
            linestring = LineString(pts[order])
            geoms.append(linestring)
        gdf = gpd.GeoDataFrame({"id": list(range(len(geoms)))}, geometry=geoms)
        return gdf


    ####    Save patch areas for AlTar inversion    ####
    def save_patch_areas(self, path):
        import h5py
        with h5py.File(path, "w") as fh:
            fh.create_dataset("patch_areas", data=self.areas)
        return self

    ####    Meade07 normals and dip-slip reference   ####
    def forced_up_normals(self):
        """Return the normal to each triangle, forced to point +z (depth positive-down) per Meade 2007.
        """
        v = self.vertices[self.triangles].astype(float).copy()
        v[:, :, 2] *= -1.0                          # z-up -> depth positive-down
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        n[n[:, 2] < 0] *= -1.0                      # meade07 forces normal +z
        return n

    def _resolve_dip_reference(self, dip_normals):
        """Expand ``dip_normals`` to an (T, 3) array of per-patch reference
        normals in meade07's (depth positive-down) frame, or ``None``.

        Accepted forms (reference vectors are given in the natural z-up frame):
          * ``None``      -> no reference (legacy per-element behaviour).
          * ``"auto"``    -> each strand's mean forced-up normal (most robust
                             way to just make every strand internally uniform).
          * ``(3,)``      -> one reference for all strands.
          * ``(n_strands, 3)`` -> one row per strand, indexed by ``fault_ids``.
          * ``dict``      -> ``{strand_name or strand_id: (3,)}``; strands absent
                             from the dict are left on the legacy behaviour.
        """
        if dip_normals is None:
            return None
        if isinstance(dip_normals, str):
            if dip_normals != "auto":
                raise ValueError(f"unknown dip_normals option {dip_normals!r}")
            fup = self.forced_up_normals()
            ref = np.zeros((self.n_patches, 3))
            for k in np.unique(self.fault_ids):
                m = self.fault_ids == k
                r = fup[m].mean(axis=0)
                nr = np.linalg.norm(r)
                ref[m] = r / nr if nr else r
            return ref

        def to_meade(vec):                          # z-up -> depth positive-down
            vec = np.asarray(vec, dtype=float).copy()
            vec[2] *= -1.0
            return vec

        ref = np.zeros((self.n_patches, 3))
        if isinstance(dip_normals, dict):
            name_to_id = {nm: k for k, nm in enumerate(self.subfault_names)}
            for key, vec in dip_normals.items():
                k = name_to_id.get(key, key)        # accept strand name or int id
                ref[self.fault_ids == k] = to_meade(vec)
            return ref
        arr = np.asarray(dip_normals, dtype=float)
        if arr.ndim == 1:                           # single normal for all strands
            return np.tile(to_meade(arr), (self.n_patches, 1))
        return np.array([to_meade(arr[k]) for k in self.fault_ids])  # (n_strands,3)

    # ------------------------------------------------------------------ #
    #  Green's functions
    # ------------------------------------------------------------------ #
    def compute_greens_functions(self, pts, verbose=False, dip_normals="auto"):
        """Meade (2007) surface-displacement Green's functions per triangle.

        Args
        ----------
        pts : array-like, shape (2, n_pts) or (3, n_pts)
            Observation coordinates; row 0 = easting, row 1 = northing (metres,
            same CRS as the mesh).  An optional row 2 is the observation depth
            (positive-down); omitted -> surface (z = 0), as for InSAR/GNSS.
        verbose : bool
            Print a progress line every ~50 triangles.
        dip_normals :
            Enforces a positive dip-slip sense. See _resolve_dip_reference. 

        Returns
        -------
        self, with ``self.gfs`` of shape ``(2, n_patches, 3, n_pts)``:
        axis 0 = [strike-slip, dip-slip], axis 2 = [east, north, up].
        """
        _self = self._copy()  # avoid overwriting the original if we need to re-run
        pts = np.asarray(pts, dtype=float)
        sx = pts[0]
        sy = pts[1]
        # meade07 wants observation depth positive-down; data sit at the surface.
        sz = pts[2] if pts.shape[0] > 2 else np.zeros_like(sx)

        n_pts = sx.shape[0]
        n_tri = _self.n_patches
        gfs_ss = np.zeros((n_tri, 3, n_pts))
        gfs_ds = np.zeros((n_tri, 3, n_pts))

        for i in range(n_tri):
            v = _self.triangle_xyz(i)
            # z-up (z<=0) -> depth positive-down for meade07.
            verts = [np.array([p[0], p[1], -p[2]]) for p in v]

            # Strike-slip: meade07's ss is -Okada U1, so negate to match
            # the Fault3d/Okada convention.  displacement() returns (E, N, Up).
            ux, uy, uz = meade07.displacement(sx, sy, sz, [p.copy() for p in verts],
                                              1.0, 0.0, 0.0, nu=_self.nu)
            gfs_ss[i] = -np.vstack((ux, uy, uz))

            # Dip-slip: meade07's ds already matches Okada U2.
            ux, uy, uz = meade07.displacement(sx, sy, sz, [p.copy() for p in verts],
                                              0.0, 1.0, 0.0, nu=_self.nu)
            gfs_ds[i] = np.vstack((ux, uy, uz))

            if verbose and (i % 50 == 0):
                print(f"[FaultTriangles] GF {i + 1}/{n_tri}")

        # Optional consistent dip-slip sense: negate the DS GF (only) on patches
        # whose meade07 forced-up normal points to the opposite side of the
        # supplied reference.  Strike-slip is left completely untouched.
        ref = _self._resolve_dip_reference(dip_normals)
        if ref is not None:
            fup = _self.forced_up_normals()
            dots = np.einsum("ij,ij->i", fup, ref)
            rn = np.linalg.norm(ref, axis=1)          # ref rows may be non-unit
            valid = rn > 0                            # zero row = strand left as-is
            cos = np.zeros_like(dots)
            cos[valid] = dots[valid] / rn[valid]      # fup is unit
            ill = valid & (np.abs(cos) < 0.17)        # within ~10 deg of perp
            if np.any(ill):
                print(f"[FaultTriangles] WARNING: {int(ill.sum())} patch(es) have "
                      f"a normal within ~10 deg of perpendicular to the dip "
                      f"reference; their dip-slip sign is ill-conditioned. Choose "
                      f"a reference closer to the fault normal for those strands.")
            signs = np.ones(n_tri)
            signs[valid] = np.where(dots[valid] >= 0.0, 1.0, -1.0)
            gfs_ds *= signs[:, None, None]

        _self.gfs = np.stack([gfs_ss, gfs_ds], axis=0)  # (2, n_tri, 3, n_pts)
        return _self

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #
    def plot_fault3d(self, ax=None, color_by="slip", cmap="plasma", vmin=None,
                     vmax=None, elev=25, azim=-60, edgecolor="k", linewidth=0.3):
        """3D rendering of the triangular mesh (coloured by slip or layer)."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.cm import ScalarMappable

        if ax is None:
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()

        # Plot in depth-positive-down z so the fault reads as depth.
        tris3d = [[(p[0], p[1], -p[2]) for p in self.triangle_xyz(i)]
                  for i in range(self.n_patches)]

        cmap_obj = plt.get_cmap(cmap)
        if color_by == "slip":
            v = np.asarray(self.slips, dtype=float)
            if vmin is None:
                vmin = float(v.min()) if v.size else 0.0
            if vmax is None:
                vmax = float(v.max()) if v.size else 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            face = cmap_obj(norm(v))
        elif color_by == "layer" and self.layers is not None:
            tab = plt.get_cmap("tab10")
            face = [tab(int(l) % 10) for l in self.layers]
        else:
            face = "0.7"

        coll = Poly3DCollection(tris3d, facecolors=face, edgecolors=edgecolor,
                                linewidths=linewidth)
        ax.add_collection3d(coll)

        allpts = self.vertices
        ax.set_xlim(allpts[:, 0].min(), allpts[:, 0].max())
        ax.set_ylim(allpts[:, 1].min(), allpts[:, 1].max())
        ax.set_zlim(-allpts[:, 2].min(), -allpts[:, 2].max())  # depth downward
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_zlabel("Depth (m)")
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect((np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1]),
                               np.ptp(allpts[:, 2])))
        except Exception:
            pass

        if color_by == "slip":
            sm = ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Slip", shrink=0.6)
        return fig, ax

    def plot_slip_2d(self, slip=None, cmap="plasma", vmin=None, vmax=None,
                     colorbar_label="Slip (m)", units="km", edgecolor="0.5",
                     linewidth=0.15, figsize=None, equal_aspect=True):
        """2D along-strike vs depth view of the fault, coloured by slip.

        Each strand (``fault_ids`` group) is drawn in its own labelled subplot on
        a shared figure with one shared colorbar.  Every triangle is projected
        onto that strand's vertical section: the along-strike coordinate is the
        distance along the straight line from the *start* to the *end* of the
        strand's surface trace, and the vertical coordinate is depth (positive
        down).  This flattens dip out of the plot, so overlapping dip layers are
        drawn back-to-front (deepest first) for readable stacking.

        Parameters
        ----------
        slip : (n_patches,) array_like, optional
            Value to colour by, in merged-patch order; defaults to ``self.slips``.
        cmap, vmin, vmax : matplotlib colour controls.  ``vmin``/``vmax`` default
            to the min/max of ``slip`` (shared across all strands).
        colorbar_label : str
        units : {"km", "m"}
            Axis units for both along-strike distance and depth.
        edgecolor, linewidth : triangle outline styling.
        figsize : (w, h), optional.
        equal_aspect : bool
            Keep depth and along-strike at 1:1 (an honest cross-section).  The
            subplots share a common depth range, so their column widths are made
            proportional to each strand's length -- giving equal-height panels
            whose widths reflect the true along-strike extents.  Set False to let
            every strand fill an equal-width column instead.

        Returns
        -------
        (fig, axes) : the figure and a 1-D array of per-strand Axes.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.collections import PolyCollection
        from matplotlib.cm import ScalarMappable

        slip = np.asarray(self.slips if slip is None else slip, dtype=float)
        scale = 1.0e3 if units == "km" else 1.0

        if vmin is None:
            vmin = float(slip.min()) if slip.size else 0.0
        if vmax is None:
            vmax = float(slip.max()) if slip.size else 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)

        # lines = self._component_lines()
        trace = self.get_surface_trace_xy()
        lines = [np.asarray(geom.coords) for geom in trace.geometry]
        n = len(self.subfault_names)
        tri_xyz = self.vertices[self.triangles]          # (T, 3, 3), z-up

        # Project each strand onto its trace-defined vertical section first, so
        # we know the along-strike extents before laying out the subplots.
        sections, widths = [], []
        for k in range(n):
            mask = self.fault_ids == k

            # Strike direction = start -> end of this strand's surface trace.
            line = np.asarray(lines[k], dtype=float)
            if len(line) >= 2:
                origin = line[0]
                strike = line[-1] - line[0]
            else:                                        # degenerate trace: use PCA
                xy = tri_xyz[mask][:, :, :2].reshape(-1, 2)
                origin = xy.mean(axis=0)
                _, _, vt = np.linalg.svd(xy - origin, full_matrices=False)
                strike = vt[0]
            strike = strike / np.linalg.norm(strike)

            v = tri_xyz[mask]                            # (n_k, 3, 3)
            s = (v[:, :, :2] - origin) @ strike          # (n_k, 3) along-strike, m
            depth = -v[:, :, 2]                          # (n_k, 3) positive-down, m
            polys = np.stack([s, depth], axis=-1) / scale
            sections.append((polys, slip[mask], depth))
            widths.append(max(float(s.max() - s.min()) / scale, 1e-9))

        # Equal aspect + shared depth range => column widths proportional to
        # along-strike length give equal-height panels.
        gridspec_kw = {"width_ratios": widths} if equal_aspect else None
        if figsize is None:
            figsize = (6.5 * n, 5.0)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False,
                                 sharey=True, layout="constrained",
                                 gridspec_kw=gridspec_kw)
        axes = axes[0]

        for k, ax in enumerate(axes):
            polys, vals, depth = sections[k]
            # Draw deepest patches first so shallow slip sits on top when dip
            # projection overlaps layers.
            order = np.argsort(-depth.mean(axis=1))
            coll = PolyCollection(list(polys[order]),
                                  array=vals[order], cmap=cmap_obj,
                                  norm=norm, edgecolors=edgecolor,
                                  linewidths=linewidth)
            ax.add_collection(coll)
            ax.autoscale_view()
            if equal_aspect:
                ax.set_aspect("equal")
            if not ax.yaxis_inverted():
                ax.invert_yaxis()                        # surface at top
            ax.set_xlabel(f"Along strike ({units})")
            ax.set_title(str(self.subfault_names[k]))
        axes[0].set_ylabel(f"Depth ({units})")

        sm = ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, label=colorbar_label)

        # constrained_layout sizes the colorbar to the grid cell, so with the
        # aspect-shrunk panels it ends up taller than them.  Freeze the resolved
        # layout, then match the colorbar's vertical extent to the panels (which
        # share y0/height via sharey + equal aspect).
        try:
            fig.draw_without_rendering()
        except AttributeError:                           # older matplotlib
            fig.canvas.draw()
        fig.set_layout_engine("none")
        panel = axes[0].get_position()
        cbpos = cb.ax.get_position()
        cb.ax.set_position([cbpos.x0, panel.y0, cbpos.width, panel.height])
        return fig, axes

    def __repr__(self):
        return (f"FaultTriangles(name={self.name!r}, {self.n_patches} triangles, "
                f"{len(self.vertices)} vertices)")
