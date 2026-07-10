"""Remesh a triangular fault surface into a depth-layered mesh.

Two entry points:

  * :func:`fault_from_cfm` reads a GOCAD TSurf community fault model (CFM)
    into a FaultTriangles and shifts it so the datum -- the lowest elevation of
    the topographic top trace -- sits at z = 0 (Okada/Meade assume a flat free
    surface; the topographic wedge above the datum is discarded by the remesh).
  * :func:`remesh_fault` takes any FaultTriangles in that frame and rebuilds it
    layer by layer between horizontal depth contours: each contour is sliced
    from the mesh, resampled evenly by arc length (point counts interpolate
    from ``n_top`` to ``n_bottom``, so the mesh coarsens with depth), and
    consecutive contours are stitched into a triangle strip. Adjacent layers
    share the exact contour vertices, so the mesh splits cleanly at any contour
    (each triangle carries its ``layer`` index).

Contours deeper than ``project_below`` are not sliced from the geometry; the
deepest sliced contour is projected vertically down instead. This extends a
fault below an uneven bottom edge (e.g. the Ridgecrest CFM, only fully defined
along strike to ~10 km) while keeping the mesh watertight.
"""
from pathlib import Path
from collections import defaultdict

import numpy as np

from ..FaultTriangles import FaultTriangles


####    Building a FaultTriangles from a community fault model    ####
def fault_from_cfm(path, name=None):
    """Read a CFM GOCAD TSurf and shift z so the top-trace datum sits at 0.

    Depths passed to :func:`remesh_fault` are measured down from this datum.
    """
    fault = FaultTriangles.from_gocad(path, name=name)
    datum = _top_datum(fault.vertices, fault.triangles)
    fault.vertices[:, 2] -= datum
    print(f"[remesh] {fault.name}: datum (z=0) at elevation {datum:.1f} m")
    return fault


####    Main remeshing routine    ####
def remesh_fault(fault, depths, n_top, n_bottom, tol=1.0, project_below=None,
                 clip_to_fault=True, taper_warn_frac=0.85, check_dip=True,
                 save=None):
    """Remesh a FaultTriangles into a depth-layered triangular mesh.

    Parameters
    ----------
    fault : FaultTriangles
        z-up, with the top-trace datum at z = 0 (see :func:`fault_from_cfm`).
    depths : sequence of positive-down contour depths (m) from the datum;
        0 is prepended if missing.
    n_top, n_bottom : point counts for the shallowest and deepest contours
        (forced even); intermediate contours interpolate linearly in layer
        index, so the triangle count drops ~linearly per layer.
    tol : endpoint snapping tolerance (m) for stitching slice segments.
    project_below : depth (m) beyond which contours are projected vertically
        from the deepest sliced contour instead of sliced from the geometry
        (None = always slice). Use where the source bottom edge is uneven.
    clip_to_fault : drop requested depths below the fault's deepest extent
        (keeps one call working across strands of differing depth).
    taper_warn_frac : warn if a contour's strike span falls below this fraction
        of the top contour's (normalized stitching stretches such contours).
    check_dip : run :func:`check_dip_sense` on the result to confirm the "auto"
        dip-slip convention resolves to a single sense across the mesh.
    save : optional path; writes ``<save>.pickle`` (FaultTriangles) and
        ``<save>.txt`` (GOCAD TSurf).

    Returns a new FaultTriangles with per-triangle ``layers``. Triangle winding
    is whatever the strip stitch emits (mixed): meade07 forces each element
    normal +z anyway, and the dip-slip sense is set by ``dip_normals`` in
    ``compute_greens_functions``, so winding carries no meaning here.
    """
    V, T = fault.vertices, fault.triangles
    depths = np.asarray(sorted(set(float(d) for d in depths)))
    if depths[0] != 0.0:
        depths = np.concatenate([[0.0], depths])

    eps = max(1.0, tol)
    cut = np.inf if project_below is None else project_below + eps
    geom_depths = depths[depths <= cut]
    proj_depths = depths[depths > cut]

    if clip_to_fault:
        fault_max_depth = -V[:, 2].min()
        keep = geom_depths < fault_max_depth - eps
        if not keep.all():
            print(f"[remesh] clipping depths below fault extent "
                  f"({fault_max_depth:.0f} m): dropped "
                  f"{geom_depths[~keep].astype(int)}")
            geom_depths = geom_depths[keep]
        if len(geom_depths) < 2:
            raise RuntimeError(f"Fewer than 2 geometry contours remain "
                               f"(fault only {fault_max_depth:.0f} m deep).")

    depths = np.concatenate([geom_depths, proj_depths])
    n_geom = len(geom_depths)

    # strike axis from a slice just below the datum (avoids tangency at z=0)
    segs = _plane_segments(V, T, -eps)
    axis = _strike_axis(max(_stitch_segments(segs, tol=tol),
                            key=_polyline_length))

    # point count per contour, linear in layer index, forced even
    K = len(depths)
    fr = np.arange(K) / max(K - 1, 1)
    counts = [_even(n_top + f * (n_bottom - n_top)) for f in fr]

    contours, top_span, deepest_raw = [], None, None
    for i, (d, n) in enumerate(zip(depths, counts)):
        if i >= n_geom:                  # projected: deepest slice dropped down
            raw = deepest_raw.copy()
            raw[:, 2] = -d
        else:
            z = -eps if d == 0.0 else -d  # nudge the datum slice off tangency
            raw = _cut_contour(V, T, z, axis, tol=tol)
            if raw is None:
                raise RuntimeError(f"Mesh does not cross depth {d:.0f} m; "
                                   f"trim your depth list.")
            deepest_raw = raw
        span = float(np.ptp(raw[:, :2] @ axis))
        if top_span is None:
            top_span = span
        elif i < n_geom and span < taper_warn_frac * top_span:
            print(f"[remesh] WARNING: depth {d:.0f} m contour spans "
                  f"{span / 1000:.1f} km ({span / top_span:.0%} of top); "
                  f"normalized stitching will stretch it laterally.")
        contours.append(_resample_polyline(raw, n))

    # assemble global vertices and stitch consecutive contours into strips
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in contours])])
    vertices = np.vstack(contours)
    triangles, layers = [], []
    for k in range(len(contours) - 1):
        strip = _stitch_two(offsets[k], len(contours[k]),
                            offsets[k + 1], len(contours[k + 1]))
        triangles.extend(strip)
        layers.extend([k] * len(strip))

    out = FaultTriangles(vertices, np.asarray(triangles, dtype=int),
                         layers=np.asarray(layers, dtype=int),
                         nu=fault.nu, name=fault.name)
    print(f"[remesh] {fault.name}: {len(vertices)} vertices, "
          f"{len(triangles)} triangles, {len(contours) - 1} layers")

    if check_dip:
        check_dip_sense(out)

    if save is not None:
        out.save(save)
        print(f"[remesh] saved -> {Path(save).with_suffix('.pickle')} / .txt")
    return out


####    Dip-slip sense check    ####
def check_dip_sense(fault, offset=2500.):
    """Check the "auto" dip-slip convention gives a single vertical sense.

    Impose +1 unit dip-slip on every patch and read the vertical (Up) surface
    displacement just off the fault. A consistent convention pushes every patch
    the same way (all up or all down); a mix means the forced-up normal flips
    along the strand, so "auto" is ambiguous and an explicit ``dip_normals``
    should be passed to ``compute_greens_functions``.
    """
    trace = fault.get_surface_trace_xy()
    cen = fault.centroids[:, :2]
    perp = np.zeros((fault.n_patches, 2))
    for k in range(fault.n_subfaults):
        ln = np.asarray(trace.geometry[k].coords, dtype=float)
        s = ln[-1] - ln[0]
        s = s / np.linalg.norm(s)
        perp[fault.fault_ids == k] = [-s[1], s[0]]
    pts = np.vstack([cen + offset * perp, cen - offset * perp]).T   # (2, 2N)
    gds = fault.compute_greens_functions(pts).gfs[1]                # (N, 3, 2N)
    up = np.sign([gds[i, 2, i] for i in range(fault.n_patches)])    # Up at +perp
    for k in range(fault.n_subfaults):
        m = fault.fault_ids == k
        nup, ndn = int((up[m] > 0).sum()), int((up[m] < 0).sum())
        flag = "" if nup == 0 or ndn == 0 else "  <-- MIXED: set dip_normals"
        print(f"[dip-sense] +1 DS, strand {k} ({fault.subfault_names[k]}): "
              f"up {nup} / down {ndn}{flag}")


####    Geometry helpers    ####
def _even(n):
    """Round to the nearest positive even integer (>= 2)."""
    n = int(round(n))
    return max(2, n + n % 2)


def _polyline_length(line):
    p = np.asarray(line, dtype=float)
    return float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum())


def _strike_axis(line):
    """Unit XY strike direction from the principal axis (PCA) of a contour."""
    p = np.asarray(line, dtype=float)[:, :2]
    p = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    return vt[0]


def _top_datum(vertices, triangles, depth_frac=0.5, dip_max=45.0):
    """Datum elevation = lowest vertex of the topographic top edge.

    Boundary edges (used by one triangle) trace the mesh perimeter; the top
    edge is the sub-horizontal part (dip < dip_max, rejects the steep lateral
    tip edges) in the upper part of the depth range (rejects the equally
    shallow-dipping bottom edge). Follows topography, no DEM needed.
    """
    count = defaultdict(int)
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            count[(min(u, v), max(u, v))] += 1
    z = vertices[:, 2]
    z_cut = z.min() + depth_frac * (z.max() - z.min())
    top_z = []
    for (u, v), n in count.items():
        if n != 1:
            continue                            # interior edge
        a, b = vertices[u], vertices[v]
        if (a[2] + b[2]) / 2 <= z_cut:
            continue                            # bottom edge
        dl = np.hypot(a[0] - b[0], a[1] - b[1])
        if np.degrees(np.arctan2(abs(a[2] - b[2]), dl)) > dip_max:
            continue                            # steep lateral (tip) edge
        top_z += [a[2], b[2]]
    if not top_z:
        raise RuntimeError("Could not extract a top edge to define the datum.")
    return min(top_z)


def _plane_segments(vertices, triangles, z_level):
    """Intersect each triangle with the plane z = z_level; list of 2D segments."""
    segments = []
    for tri in triangles:
        p = vertices[tri]
        d = p[:, 2] - z_level
        crossings = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (d[a] > 0) == (d[b] > 0) or d[a] == d[b]:
                continue                        # edge does not cross the plane
            t = d[a] / (d[a] - d[b])
            crossings.append(tuple(p[a, :2] + t * (p[b, :2] - p[a, :2])))
        if len(crossings) == 2:                 # clean crossing only
            segments.append((crossings[0], crossings[1]))
    return segments


def _stitch_segments(segments, tol=1.0):
    """Join segments sharing endpoints (within ``tol`` m) into polylines."""
    def key(pt):
        return (round(pt[0] / tol), round(pt[1] / tol))

    adj = defaultdict(list)
    coords = {}
    for a, b in segments:
        ka, kb = key(a), key(b)
        coords[ka], coords[kb] = a, b
        adj[ka].append(kb)
        adj[kb].append(ka)

    visited = set()

    def edge_id(u, v):
        return (u, v) if u <= v else (v, u)

    lines = []
    # trace from endpoints (degree 1) first, then any remaining loops
    starts = [n for n in adj if len(adj[n]) == 1] + list(adj.keys())
    for start in starts:
        for nxt in adj[start]:
            if edge_id(start, nxt) in visited:
                continue
            line = [coords[start]]
            u, v = start, nxt
            while True:
                visited.add(edge_id(u, v))
                line.append(coords[v])
                nxts = [w for w in adj[v] if edge_id(v, w) not in visited]
                if not nxts:
                    break
                u, v = v, nxts[0]
            if len(line) >= 2:
                lines.append(line)
    return lines


def _cut_contour(vertices, triangles, z_level, axis, tol=1.0):
    """Longest stitched polyline at z = z_level, as an (N, 3) array oriented
    to run in the +``axis`` direction (consistent across contours)."""
    segs = _plane_segments(vertices, triangles, z_level)
    if not segs:
        return None
    lines = _stitch_segments(segs, tol=tol)
    if not lines:
        return None
    line = max(lines, key=_polyline_length)     # drop degenerate touches
    pts = np.column_stack([np.asarray(line, dtype=float),
                           np.full(len(line), z_level)])
    proj = pts[:, :2] @ axis
    if proj[0] > proj[-1]:
        pts = pts[::-1]
    return pts


def _resample_polyline(pts, n):
    """Resample an (M, 3) polyline to ``n`` points evenly by arc length."""
    pts = np.asarray(pts, dtype=float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(target, s, pts[:, k])
    return out


def _stitch_two(i_top0, m, i_bot0, n):
    """Two-pointer triangle strip between a top row (m verts, global base
    ``i_top0``) and a bottom row (n verts, base ``i_bot0``), matched by
    normalized along-strike position. Returns (a, b, c) global-index triples.
    """
    tp = np.linspace(0.0, 1.0, m)
    bp = np.linspace(0.0, 1.0, n)
    tris = []
    i = j = 0
    while i < m - 1 or j < n - 1:
        if j >= n - 1:                       # bottom exhausted: fan from top
            tris.append((i_top0 + i, i_top0 + i + 1, i_bot0 + j))
            i += 1
        elif i >= m - 1:                     # top exhausted: fan from bottom
            tris.append((i_top0 + i, i_bot0 + j, i_bot0 + j + 1))
            j += 1
        elif tp[i + 1] <= bp[j + 1]:         # advance whichever's next is nearer
            tris.append((i_top0 + i, i_top0 + i + 1, i_bot0 + j))
            i += 1
        else:
            tris.append((i_top0 + i, i_bot0 + j, i_bot0 + j + 1))
            j += 1
    return tris
