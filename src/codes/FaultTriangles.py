"""Triangular-dislocation-element (TDE) fault for half-space Green's functions.

The rectangular :class:`~codes.Fault3d.Fault3d` workflow uses Okada (1985)
patches; this module is its triangular counterpart, built on the Meade (2007)
TDE displacements (``codes.meade07``, the CSI translation of Brendan Meade's
MATLAB code).  It consumes the depth-layered triangular mesh produced by
``scripts/remeshFault.py`` (a ``RemeshedFault`` / its ``.npz``) and returns
surface-displacement Green's functions in the *same* convention and array layout
as ``Fault3d.compute_greens_functions`` so the rest of the inversion pipeline is
unchanged.

Conventions (validated to machine precision against the trusted Okada reference
``scripts/_okada_check.py`` by splitting one rectangle into two TDEs, per Meade
2007 Fig. 6 -- agreement ~1e-14):

* **Inversion unit = one triangle.**  ``self.gfs`` has shape
  ``(2, n_patches, 3, n_pts)``: axis 0 = ``[strike-slip, dip-slip]``,
  axis 2 = ``[east, north, up]`` -- identical to ``Fault3d.gfs``.
* **Vertices** are stored z-up / datum-shifted exactly as the mesh gives them
  (z = 0 free surface, fault at z <= 0).  ``meade07`` internally wants depth
  positive-down, so we feed it ``-z``.  (The remesh handover's suggestion to feed
  z-up straight in applies to Nikkhoo-style TDE codes; CSI's ``meade07`` -- which
  stores ``self.depth = max(vertex_z)`` -- wants positive-down, so we convert.)
* **Sign convention matches Okada/Fault3d.**  ``meade07``'s strike-slip is the
  negative of Okada's ``U1`` (its dip-slip already matches ``U2``), so the
  strike-slip Green's function is negated here.  Positive strike-slip and
  positive dip-slip therefore mean the same thing they do for the rectangular
  patches.  ``meade07`` forces each element normal to point +z (clockwise-from-
  above winding) before resolving slip, so the strike/dip sense is fixed by the
  triangle geometry alone and is independent of the mesh's stored winding.
"""

import numpy as np
import copy

from . import meade07


class FaultTriangles:
    """A triangular-mesh fault with Meade (2007) half-space Green's functions.

    Each triangle is one inversion unit (one slip value), unlike the rectangular
    ``Fault3d`` where the unit is an even-width ``Cell`` grouping many sub-patches.

    Attributes
    ----------
    vertices : (V, 3) float
        Shared vertex array, z-up / datum-shifted (metres; z = 0 at surface,
        fault at z <= 0), matching ``RemeshedFault.vertices``.
    triangles : (T, 3) int
        0-based connectivity into ``vertices``.
    layers : (T,) int or None
        Depth-band index per triangle (0 = shallowest), used to keep the mesh
        separable at depth contours.
    slips : (T,) float
        Slip value per triangle (aligned with ``triangles``); zero until set.
    gfs : (2, T, 3, n_pts) float
        Green's functions after :meth:`compute_greens_functions`.
    """

    def __init__(self, vertices, triangles, layers=None, datum=0.0,
                 name="fault_triangles", nu=0.25, winding_ref=None):
        self.vertices = np.asarray(vertices, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        self.layers = None if layers is None else np.asarray(layers, dtype=int)
        self.datum = float(datum)
        self.name = str(name)
        self.nu = float(nu)
        self.winding_ref = (None if winding_ref is None
                            else np.asarray(winding_ref, dtype=float))
        self.slips = np.zeros(len(self.triangles))
        self.gfs = None
        # Optional per-strand reference normals fixing a consistent dip-slip
        # sense (see set_dip_convention / compute_greens_functions). None ->
        # meade07's per-element forced-up normal decides the DS sense.
        self.dip_reference = None
        # Per-triangle strand index (0 for a single mesh); populated by merge().
        self.fault_ids = np.zeros(len(self.triangles), dtype=int)
        self.component_names = [self.name]
        # Cached surface-trace polylines (one per strand); built lazily by .trace.
        self._trace_lines = None

    # ------------------------------------------------------------------ #
    #  Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_remesh(cls, rf, nu=0.25):
        """Build from a ``scripts/remeshFault.RemeshedFault`` object."""
        return cls(rf.vertices, rf.triangles, layers=rf.layers, datum=rf.datum,
                   name=rf.name, nu=nu, winding_ref=rf.winding_ref)

    @classmethod
    def merge(cls, faults, name="merged_fault"):
        """Concatenate several meshes into one fault (patches stay per-triangle).

        Vertices are stacked and triangle indices offset, so the merged object
        behaves as a single fault for InSAR/GNSS/InversionManager (one combined
        ``compute_greens_functions`` and one multi-line ``.trace``), while
        ``fault_ids`` records which strand each triangle came from -- used by
        :meth:`patch_areas_by_fault` / :meth:`save_patch_areas`.  The merged
        patch order is strand 0's triangles, then strand 1's, ... so it matches
        the SS/DS parameter ordering the inversion produces.
        """
        faults = list(faults)
        V_parts, T_parts, L_parts, F_parts, lines = [], [], [], [], []
        voff = 0
        for k, f in enumerate(faults):
            V_parts.append(f.vertices)
            T_parts.append(f.triangles + voff)
            voff += len(f.vertices)
            L_parts.append(f.layers if f.layers is not None
                           else np.zeros(f.n_patches, dtype=int))
            F_parts.append(np.full(f.n_patches, k, dtype=int))
            lines.extend(f._component_lines())
        obj = cls(np.vstack(V_parts), np.vstack(T_parts),
                  layers=np.concatenate(L_parts), datum=faults[0].datum,
                  name=name, nu=faults[0].nu)
        obj.fault_ids = np.concatenate(F_parts)
        obj.component_names = [f.name for f in faults]
        obj._trace_lines = lines
        return obj

    @classmethod
    def from_npz(cls, path, nu=0.25):
        """Build directly from a ``remeshFault`` ``.npz`` (no remeshFault import).

        Keys: ``vertices`` (V,3), ``triangles`` (T,3 int, 0-based, wound),
        ``layers`` (T,), ``depths``, ``datum``, ``name``, ``winding_ref``.
        """
        d = np.load(path, allow_pickle=True)
        wref = d["winding_ref"] if "winding_ref" in d.files else None
        if wref is not None and np.isnan(np.asarray(wref, float)).any():
            wref = None
        return cls(d["vertices"], d["triangles"],
                   layers=d["layers"] if "layers" in d.files else None,
                   datum=float(d["datum"]) if "datum" in d.files else 0.0,
                   name=str(d["name"]) if "name" in d.files else "fault_triangles",
                   nu=nu, winding_ref=wref)

    # ------------------------------------------------------------------ #
    #  Geometry accessors (vectorised over triangles)
    # ------------------------------------------------------------------ #
    @property
    def n_patches(self):
        return len(self.triangles)

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

    def initializeslip(self, value=0.0):
        """Reset ``self.slips`` (mirrors the CSI/Fault3d entry point)."""
        self.slips = np.full(len(self.triangles), float(value))
        return self

    # ------------------------------------------------------------------ #
    #  Surface trace (for downsampling / station filtering / plots)
    # ------------------------------------------------------------------ #
    def surface_trace_xy(self, tol=1.0):
        """(M, 2) top-contour vertices (UTM), ordered along strike.

        The remesh cuts horizontal contours, so the shallowest contour is the
        set of vertices at the maximum z (a single value); we select them and
        order them by their projection onto the trace's principal (strike) axis.
        """
        z = self.vertices[:, 2]
        top = z > (z.max() - tol)
        pts = self.vertices[top][:, :2]
        if len(pts) < 2:
            return pts
        c = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        order = np.argsort(c @ vt[0])
        return pts[order]

    def _component_lines(self):
        """List of per-strand trace polylines (one entry for a single mesh)."""
        if self._trace_lines is not None:
            return list(self._trace_lines)
        return [self.surface_trace_xy()]

    @property
    def trace(self):
        """GeoDataFrame of the surface trace(s) in UTM (EPSG:32611).

        One row per strand; geometry is a ``LineString`` of the top-contour
        vertices.  Matches the ``.trace`` duck-typing that ``InSAR``/``GNSS``/
        ``InversionManager`` expect (``.geometry`` iterable / ``.unary_union``).
        """
        import geopandas as gpd
        from shapely.geometry import LineString
        geoms = [LineString(l) for l in self._component_lines() if len(l) >= 2]
        return gpd.GeoDataFrame({"id": list(range(len(geoms)))},
                                geometry=geoms, crs="EPSG:32611")

    # ------------------------------------------------------------------ #
    #  Per-fault patch areas (for AlTar moment / smoothing priors)
    # ------------------------------------------------------------------ #
    def patch_areas_by_fault(self):
        """Dict ``{strand_name: (n_k,) areas}`` split by ``fault_ids``."""
        a = self.areas
        return {name: a[self.fault_ids == k]
                for k, name in enumerate(self.component_names)}

    def save_patch_areas(self, path):
        """Write patch areas (m^2) to an HDF5 file for the AlTar inversion.

        ``/patch_areas`` holds all areas in GF/parameter order; ``/by_fault/<name>``
        holds each strand's block (same order as the merged patches).
        """
        import h5py
        with h5py.File(path, "w") as fh:
            fh.create_dataset("patch_areas", data=self.areas)
            grp = fh.create_group("by_fault")
            for name, ar in self.patch_areas_by_fault().items():
                grp.create_dataset(str(name), data=ar)
        return self

    # ------------------------------------------------------------------ #
    #  Dip-slip sign convention
    # ------------------------------------------------------------------ #
    def forced_up_normals(self):
        """(T, 3) unit element normals in meade07's frame (depth positive-down),
        each forced to point +z exactly as ``meade07`` does internally.

        This is the vector whose horizontal azimuth sets each element's
        strike/dip decomposition.  For a near-vertical fault it is ~horizontal
        and points to one side of the fault; where the dip azimuth reverses
        along strike (or the element is within a degree or two of vertical, so
        the ``normVec[2] < 0`` test is numerical noise) it flips side, which is
        what makes the raw dip-slip sense inconsistent patch-to-patch.
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
            name_to_id = {nm: k for k, nm in enumerate(self.component_names)}
            for key, vec in dip_normals.items():
                k = name_to_id.get(key, key)        # accept strand name or int id
                ref[self.fault_ids == k] = to_meade(vec)
            return ref
        arr = np.asarray(dip_normals, dtype=float)
        if arr.ndim == 1:                           # single normal for all strands
            return np.tile(to_meade(arr), (self.n_patches, 1))
        return np.array([to_meade(arr[k]) for k in self.fault_ids])  # (n_strands,3)

    def set_dip_convention(self, dip_normals):
        """Store per-strand reference normals so subsequent
        :meth:`compute_greens_functions` calls (including the ones made inside
        ``InSAR``/``GNSS.compute_greens_functions``) fix a consistent dip-slip
        sense.  See :meth:`compute_greens_functions` for the accepted forms.
        Returns ``self``.
        """
        self.dip_reference = dip_normals
        return self

    # ------------------------------------------------------------------ #
    #  Green's functions
    # ------------------------------------------------------------------ #
    def compute_greens_functions(self, pts, verbose=False, dip_normals=None):
        """Meade (2007) surface-displacement Green's functions per triangle.

        Parameters
        ----------
        pts : array-like, shape (2, n_pts) or (3, n_pts)
            Observation coordinates; row 0 = easting, row 1 = northing (metres,
            same CRS as the mesh).  An optional row 2 is the observation depth
            (positive-down); omitted -> surface (z = 0), as for InSAR/GNSS.
        verbose : bool
            Print a progress line every ~50 triangles.
        dip_normals : optional
            Per-strand reference normal(s) that fix a *consistent* positive
            dip-slip sense.  Without this (and without a stored
            :meth:`set_dip_convention`), meade07's per-element forced-up normal
            sets the DS sense, which flips where a strand's dip azimuth reverses
            along strike or where elements are near-vertical -- so a single "+DS"
            parameter does not mean one physical thing across the mesh.  Given a
            reference, each patch's DS Green's function is negated where its
            forced-up normal points to the opposite side (``forced_up_normal .
            ref < 0``), so "+DS" is uniform per strand.  **Strike-slip is never
            touched.**  Accepted forms (z-up frame):  ``"auto"`` (each strand's
            mean forced-up normal), a single ``(3,)`` vector, an
            ``(n_strands, 3)`` array by ``fault_ids``, or a ``dict``
            ``{strand_name or id: (3,)}``.  Falls back to
            :attr:`dip_reference` (set via :meth:`set_dip_convention`) when this
            argument is ``None``; pass a value here to override it.

        Returns
        -------
        self, with ``self.gfs`` of shape ``(2, n_patches, 3, n_pts)``:
        axis 0 = [strike-slip, dip-slip], axis 2 = [east, north, up].
        """
        pts = np.asarray(pts, dtype=float)
        sx = pts[0]
        sy = pts[1]
        # meade07 wants observation depth positive-down; data sit at the surface.
        sz = pts[2] if pts.shape[0] > 2 else np.zeros_like(sx)

        n_pts = sx.shape[0]
        n_tri = self.n_patches
        gfs_ss = np.zeros((n_tri, 3, n_pts))
        gfs_ds = np.zeros((n_tri, 3, n_pts))

        for i in range(n_tri):
            v = self.triangle_xyz(i)
            # z-up (z<=0) -> depth positive-down for meade07.
            verts = [np.array([p[0], p[1], -p[2]]) for p in v]

            # Strike-slip: meade07's ss is -Okada U1, so negate to match
            # the Fault3d/Okada convention.  displacement() returns (E, N, Up).
            ux, uy, uz = meade07.displacement(sx, sy, sz, [p.copy() for p in verts],
                                              1.0, 0.0, 0.0, nu=self.nu)
            gfs_ss[i] = -np.vstack((ux, uy, uz))

            # Dip-slip: meade07's ds already matches Okada U2.
            ux, uy, uz = meade07.displacement(sx, sy, sz, [p.copy() for p in verts],
                                              0.0, 1.0, 0.0, nu=self.nu)
            gfs_ds[i] = np.vstack((ux, uy, uz))

            if verbose and (i % 50 == 0):
                print(f"[FaultTriangles] GF {i + 1}/{n_tri}")

        # Optional consistent dip-slip sense: negate the DS GF (only) on patches
        # whose meade07 forced-up normal points to the opposite side of the
        # supplied reference.  Linearity means -GF(+1 DS) == GF(-1 DS), i.e. slip
        # down the reversed dip -- a valid physical field, just relabelled so
        # "+DS" is uniform.  Strike-slip is left completely untouched.
        ref = self._resolve_dip_reference(
            dip_normals if dip_normals is not None else self.dip_reference)
        if ref is not None:
            fup = self.forced_up_normals()
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

        self.gfs = np.stack([gfs_ss, gfs_ds], axis=0)  # (2, n_tri, 3, n_pts)
        return self

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

        lines = self._component_lines()
        n = len(self.component_names)
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
            ax.set_title(str(self.component_names[k]))
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
                f"{len(self.vertices)} vertices, datum={self.datum:.1f} m)")
