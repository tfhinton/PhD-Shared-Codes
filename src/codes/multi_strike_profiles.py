'''Multi-strike swathe profiling on a refined fault trace.

How a profile looks depends on the strike it is drawn perpendicular to, and
the strike of a fault depends on the length scale it is measured over. Here
that ambiguity is turned into an uncertainty. For each picked profile:

  1. centre: the point of the refined trace nearest to the pick.
  2. strikes: a straight line is fitted to the refined trace over
     +/-strike_windows m (along the trace) around the centre, giving one
     strike per length scale, from very local to long-term.
  3. for each strike, one straight swathe through the centre, perpendicular to
     that strike (+/-swathe_half_width m along it, +/-plen m across it):
       - displacements are resolved parallel / perpendicular to THAT strike;
       - every pixel is placed by its signed shortest distance to the whole
         refined strand (+ on the left of strike);
       - the pixels are median-stacked in distance bins (inversion_bin_edges).
  4. each swathe is shifted by a constant so that all swathes have the same
     mean. Turning the strike by an angle delta mixes the mean fault-normal
     displacement into the parallel component (~ normal * sin(delta)), which
     moves the whole profile up or down; the 2D inversion solves for a
     constant offset, so this shift carries no information.
  5. the final profile is the per-bin MEDIAN across the shifted swathes, and
     its uncertainty the per-bin STANDARD DEVIATION across them:
         Cd = diag(std^2)
     This is the strike-ambiguity term only. It holds no pixel noise and no
     correlation between bins, and it tends to zero near the fault, where all
     swathes share the same pixels -- the noise term is added downstream.

Output conventions follow swathe_profiles (xs = 0 on the fault, + on the left
of strike; displacements = [parallel (+strike), normal (+xs)]; linestring from
xs = -plen to +plen; original pick attached as .picked). Everything is in
metres, UTM.
'''
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .Profile import Profile
from .swathe_profiles import (_match_strand, _swathe_pixels, _bin_stack,
                              _quiet_runtimewarnings)


@_quiet_runtimewarnings
def evaluate_picked_profiles_multi_strike(
        opt, trace, picked, plen=4000., swathe_half_width=50.,
        strike_windows=(25., 50., 100., 200., 400., 800., 1600., 3200.),
        bin_kw=None, min_count=20, verbose=True):
    '''Multi-strike swathe evaluation of picked profiles (see module docstring).

    opt: OpticalData; trace: GeoDataFrame of refined strands (one LineString
    each); picked: profiles from pick_profiles.py (fault_utm, strike, xs).

    Kwargs (m):
      plen: profile half-length across the fault.
      swathe_half_width: half-width of each swathe along its strike.
      strike_windows: half-lengths of refined trace fitted for the strikes.
      bin_kw: kwargs for inversion_bin_edges.
      min_count: minimum number of pixels for a bin of one swathe to count.
    '''
    edges = inversion_bin_edges(plen, **(bin_kw or {}))
    out = []
    for k, p in enumerate(picked):
        ep = _evaluate_at(opt, trace, p, plen, swathe_half_width,
                          np.asarray(strike_windows, dtype=float), edges, min_count)
        out.append(ep)
        if verbose:
            print(f'[{k + 1}/{len(picked)}] strand {ep.strand_id} @ '
                  f'{p.x_along_fault / 1000.:.1f} km: strikes '
                  f'{ep.strikes.min():.1f}-{ep.strikes.max():.1f} deg, '
                  f'median parallel sigma {np.median(ep.displacements_std[0]):.3f} m')
    return out


def _evaluate_at(opt, trace, p, plen, swathe_half_width, strike_windows, edges,
                 min_count):
    pick_xy = np.asarray(p.fault_utm, dtype=float)[:2]

    # ----- 1. strand under the pick (oriented like the pick) and the centre -----
    strand_id, strand, s0 = _match_strand(trace, pick_xy, p.strike)
    fault_xy = np.asarray(strand.coords)[:, :2]
    centre = np.asarray(strand.interpolate(s0).coords[0])[:2]

    # ----- 2. one strike per length scale -----
    tangents = np.array([fit_strike(fault_xy, s0, w) for w in strike_windows])
    strikes = np.degrees(np.arctan2(tangents[:, 0], tangents[:, 1])) % 360.

    # ----- 3. one swathe per strike -----
    x, par, nrm, scatter, count = [], [], [], [], []
    for t_hat in tangents:
        u_hat = np.array([-t_hat[1], t_hat[0]])            # left of strike = +xs
        sw = _swathe_pixels(opt, centre, t_hat, u_hat, plen, swathe_half_width)
        d = signed_distance(sw['xy'], fault_xy)
        disp = np.vstack([sw['ew'] * t_hat[0] + sw['nn'] * t_hat[1],    # parallel
                          sw['ew'] * u_hat[0] + sw['nn'] * u_hat[1]])   # normal
        st = _bin_stack(d, disp, edges, min_count)
        x.append(st['x'])
        par.append(st['median'][0])
        nrm.append(st['median'][1])
        scatter.append(st['robust_std'])
        count.append(st['count'])
    xs = np.median(x, axis=0)
    swathes = np.stack([par, nrm], axis=1)        # (n_swathes, 2, n_bins)

    # profile frame: the median strike (relative to the first, to avoid the
    # 0/360 wrap)
    rel = (strikes - strikes[0] + 180.) % 360. - 180.
    strike = (strikes[0] + np.median(rel)) % 360.
    t_med = np.array([np.sin(np.radians(strike)), np.cos(np.radians(strike))])
    u_med = np.array([-t_med[1], t_med[0]])
    total_shift = float((centre - pick_xy) @ u_med)    # pick -> refined trace, m

    # keep bins that every swathe filled and that lie in the pick's trusted span
    keep = (np.isfinite(swathes).all(axis=(0, 1))
            & (xs >= p.xs.min() - total_shift) & (xs <= p.xs.max() - total_shift))
    xs, swathes = xs[keep], swathes[:, :, keep]

    # ----- 4. shift every swathe onto the common mean level -----
    offsets = swathes.mean(axis=2) - swathes.mean(axis=(0, 2))   # (n_swathes, 2)
    swathes = swathes - offsets[:, :, None]

    # ----- 5. median and spread across the swathes -----
    median = np.median(swathes, axis=0)           # (2, n_bins)
    std = np.std(swathes, axis=0, ddof=1)

    # ----- output Profile -----
    line = LineString([centre - plen * u_med, centre + plen * u_med])
    gdf = gpd.GeoDataFrame({'x_along_fault': [p.x_along_fault]},
                           geometry=[line], crs=p.trace.crs)
    ep = Profile(trace=gdf, fault_x=plen)
    ep.method = 'multi_strike'
    ep.xs = xs
    ep.displacements = median
    ep.displacements_std = std                     # spread across the swathes
    ep.Cd = np.diag(std[0] ** 2)                   # fault-parallel
    ep.Cd_normal = np.diag(std[1] ** 2)
    ep.bin_edges = edges

    # the individual swathes, for inspection
    ep.strike_windows = strike_windows
    ep.strikes = strikes
    ep.swathe_displacements = swathes              # after the constant shift
    ep.swathe_offsets = offsets                    # the shift removed from each
    # scatter of the pixels within each bin (1.4826 MAD), median over swathes;
    # NOT included in Cd
    ep.stack_std = np.median(scatter, axis=0)[:, keep]
    ep.counts = np.median(count, axis=0)[keep]

    ep.fault_id = p.fault_id
    ep.strand_id = strand_id
    ep.x_along_fault = p.x_along_fault
    ep.strike = strike
    ep.fault_utm = tuple(pick_xy)
    ep.fault_utm_refined = tuple(centre)           # xs = 0
    ep.total_shift = total_shift
    ep.swathe_half_width = swathe_half_width
    ep.keep_extent = getattr(p, 'keep_extent', None)
    ep.picked = p
    return ep


def inversion_bin_edges(plen, near_dist=500., near_width=10., growth=1.1):
    '''Bin edges symmetric about the fault (an edge at d = 0): near_width wide
    out to near_dist, then each bin growth x the previous one, ending at plen.
    The defaults are the bins remove_deep_slip_contribution.py passes to the 2D
    inversion.'''
    e = list(np.arange(0., near_dist + 0.5 * near_width, near_width))
    w = near_width
    while e[-1] < plen:
        w *= growth
        e.append(min(e[-1] + w, plen))
    e = np.asarray(e)
    return np.concatenate([-e[:0:-1], e])


def fit_strike(line_xy, s0, half_window):
    '''Unit vector (E, N) along the best-fitting straight line through the
    vertices of a polyline within +/-half_window m (arc length) of position s0,
    pointing in the direction of the polyline.'''
    s = np.concatenate([[0.], np.cumsum(np.hypot(*np.diff(line_xy, axis=0).T))])
    pts = line_xy[np.abs(s - s0) <= half_window]
    # principal direction of the points = total-least-squares line fit
    t_hat = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)[2][0]
    return t_hat if t_hat @ (pts[-1] - pts[0]) > 0 else -t_hat


def signed_distance(pts, line_xy):
    '''Shortest distance of pts (M, 2) to a polyline (N, 2), positive on the
    LEFT of the polyline direction. The nearest vertex is found with a KD-tree
    and the two segments meeting there are tested exactly, which is the true
    shortest distance for a finely, evenly sampled polyline.'''
    nearest = cKDTree(line_xy).query(pts)[1]
    dist = np.full(pts.shape[0], np.inf)
    side = np.ones(pts.shape[0])
    for first in (nearest - 1, nearest):               # segment first -> first + 1
        first = np.clip(first, 0, line_xy.shape[0] - 2)
        a = line_xy[first]
        ab = line_xy[first + 1] - a
        ap = pts - a
        # closest point of the segment: a + f * ab, 0 <= f <= 1
        f = np.clip((ap * ab).sum(axis=1) / (ab * ab).sum(axis=1), 0., 1.)
        d = np.hypot(*(ap - f[:, None] * ab).T)
        cross = ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]
        closer = d < dist
        dist[closer] = d[closer]
        side[closer] = np.where(cross[closer] >= 0., 1., -1.)
    return side * dist
