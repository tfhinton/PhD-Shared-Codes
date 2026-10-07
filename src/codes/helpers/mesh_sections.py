'''Cross-sections of a triangular mesh by a vertical plane.

Given a triangulated fault surface and a vertical plane (specified by a point
and a horizontal in-plane direction -- e.g. a fault-perpendicular profile), the
functions here find which triangles the plane cuts, where it cuts them, and how
much of each cut falls inside a given depth interval.

Coordinates follow the ``FaultTriangles`` convention: vertices are (x, y, z)
with z UP and the free surface at z = 0, so depth = -z.  Section coordinates
are (a, z_depth), where ``a`` is the signed in-plane horizontal distance from
the plane's origin along ``direction`` and ``z_depth`` is positive-down.
'''
import numpy as np


def vertical_plane_crossings(vertices, triangles, origin, direction,
                             subset=None):
    '''Triangles cut by the vertical plane through ``origin`` spanning
    ``direction`` (a horizontal 2-vector) and the vertical.

    Parameters
    ----------
    vertices : (V, 3) array, z-up.
    triangles : (T, 3) int array of vertex indices.
    origin : (2,) horizontal point the plane passes through.
    direction : (2,) horizontal in-plane direction; need not be normalised.
        Sets the sign and scale of the returned ``a`` coordinate.
    subset : (T,) bool array or index array, optional
        Restrict the search to these triangles (e.g. one strand, or the layers
        shallower than some depth).  Returned indices are still into the full
        ``triangles`` array.

    Returns
    -------
    idx : (n,) int array
        Indices of the triangles the plane cuts, ordered shallow -> deep by the
        top of their crossing segment.
    segs : (n, 4) float array
        ``[a_top, depth_top, a_bottom, depth_bottom]`` -- the endpoints of each
        triangle's intersection with the plane, shallower endpoint first.

    Notes
    -----
    A triangle is 'cut' when its vertices do not all lie strictly on one side.
    Triangles that merely touch the plane at a single vertex, or lie in it,
    produce fewer than two distinct intersection points and are dropped.
    '''
    v = np.asarray(vertices, dtype=float)[np.asarray(triangles, dtype=int)]
    origin = np.asarray(origin, dtype=float)
    u_hat = np.asarray(direction, dtype=float)
    u_hat = u_hat / np.hypot(*u_hat)
    n_hat = np.array([-u_hat[1], u_hat[0]])      # horizontal plane normal

    f = (v[:, :, :2] - origin) @ n_hat           # (T, 3) signed distance
    cut = (f.min(axis=1) <= 0.) & (f.max(axis=1) >= 0.)
    if subset is not None:
        keep = np.zeros(len(v), dtype=bool)
        keep[subset] = True
        cut &= keep

    idx, segs = [], []
    for t in np.flatnonzero(cut):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            fa, fb = f[t, i], f[t, j]
            if fa == fb:                          # edge parallel to the plane
                continue
            s = fa / (fa - fb)
            if -1e-12 <= s <= 1. + 1e-12:
                pts.append(v[t, i] + s * (v[t, j] - v[t, i]))
        if len(pts) < 2:
            continue
        pts = np.asarray(pts)
        a = (pts[:, :2] - origin) @ u_hat
        z = -pts[:, 2]
        o = np.argsort(z)                         # shallowest / deepest points
        if z[o[-1]] - z[o[0]] <= 0. and np.hypot(a[o[-1]] - a[o[0]],
                                                 z[o[-1]] - z[o[0]]) == 0.:
            continue                              # degenerate (single point)
        idx.append(t)
        segs.append((a[o[0]], z[o[0]], a[o[-1]], z[o[-1]]))

    if not idx:
        return np.zeros(0, dtype=int), np.zeros((0, 4))
    idx = np.array(idx, dtype=int)
    segs = np.array(segs, dtype=float)
    order = np.argsort(segs[:, 1])
    return idx[order], segs[order]


def depth_band_weights(segs, interfaces):
    '''How much of each crossing segment lies inside each depth band.

    Parameters
    ----------
    segs : (n, 4) array from :func:`vertical_plane_crossings`.
    interfaces : (k+1,) increasing depths (positive-down) bounding k bands.

    Returns
    -------
    (k, n) array of arc lengths, in the (a, depth) section plane, of the part
    of each segment inside each band.

    Using arc length rather than depth overlap means the weights partition each
    band's slice of the fault surface exactly: a shallowly-dipping triangle
    spanning a band contributes proportionally more than a steep one covering
    the same depth range, which is the sense in which it makes up more of that
    band's fault area.
    '''
    segs = np.atleast_2d(np.asarray(segs, dtype=float))
    interfaces = np.asarray(interfaces, dtype=float)
    if segs.size == 0:
        return np.zeros((len(interfaces) - 1, 0))

    a0, z0, a1, z1 = segs.T
    length = np.hypot(a1 - a0, z1 - z0)
    dz = z1 - z0                                  # >= 0 by construction

    W = np.zeros((len(interfaces) - 1, len(segs)))
    for k, (top, bot) in enumerate(zip(interfaces[:-1], interfaces[1:])):
        overlap = np.clip(np.minimum(z1, bot) - np.maximum(z0, top), 0., None)
        # a horizontal segment has no depth extent to share out: it belongs
        # entirely to whichever band contains it
        frac = np.where(dz > 1e-9, overlap / np.where(dz > 1e-9, dz, 1.),
                        ((z0 >= top) & (z0 <= bot)).astype(float))
        W[k] = frac * length
    return W
