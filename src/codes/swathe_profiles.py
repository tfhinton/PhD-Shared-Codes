'''Swathe profiling by distance to a strain-refined fault trace.

Alternative to codes.fault_aligned_profiles for the picked profiles. The
fault-aligned scheme stacks many profiles drawn perpendicular to the LOCAL
strike, which fan out in the far field (sampling pixels well off the profile
line) and resample the same pixels more than once. Here each profile is a
single straight swathe perpendicular to the AVERAGE local strike, and every
pixel in it is placed by its shortest distance to a carefully refined trace:

  1. coarse location: the drawn trace is rigidly shifted onto the stacked
     fault-parallel displacement step (fault_aligned_profiles._step_refine_shift,
     robust to drawn-trace errors of a few hundred metres).
  2. trace refinement: the shear strain eps_tn (rotated into the local strike
     frame) is resampled onto a "straightened" grid along the trace, lightly
     smoothed along strike, and its peak is picked every metre along strike.
     Picks with no clear peak or far from a running median are rejected, the
     rest interpolated + smoothed, and the trace moved onto them. Repeated
     with progressively narrower searches and less along-strike smoothing.
  3. swathe: every pixel within +/-swathe_half_width (along strike) of the
     profile line and +/-plen across it, the line being perpendicular to the
     mean strike of the refined trace over +/-strike_half_window.
  4. signed shortest distance of each pixel to the refined trace (extended
     straight beyond its ends), positive on the +xs side (left of strike).
  5. displacements resolved onto the MEAN strike (parallel) and the profile
     direction (normal) -- no per-pixel rotation -- then median-stacked in
     distance bins that are 1 m wide near the fault and widen outward
     (decaying_bin_edges).

Output conventions follow fault_aligned_profiles.evaluate_picked_profiles
(xs = 0 on the refined fault, displacements = [parallel (+strike), normal
(+xs)], linestring index 0 at xs = -plen, fault_utm_refined at xs = 0), so the
result is a drop-in input to remove_deep_slip_contribution.py. Everything is
in metres; geometry is handled in UTM using the raster's pixel-CENTRE
coordinates (the affine transform refers to pixel corners).
'''
import warnings
import functools

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring
from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d

from .Profile import Profile
from .fault_aligned_profiles import _rasterise_and_strike, _step_refine_shift


def _quiet_runtimewarnings(fn):
    '''Suppress the expected NaN-reduction RuntimeWarnings (all-NaN medians,
    empty-slice means) raised on gappy data.'''
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------
@_quiet_runtimewarnings
def evaluate_picked_profiles_swathe(
        opt, fault, picked, plen=4000., swathe_half_width=50.,
        strike_half_window=100., refine_half_length=400.,
        step_search_half_width=350., step_stack=150, trace_smooth=15,
        stages=((100., 10., 0.5), (20., 5., 0.5)),
        strain_sigma=1.5, peak_quality=2., prior_weight=0.5, offset_sigma=4.,
        bin_kw=None, min_count=5, near_pixels_within=100.,
        min_support=0.5, min_quality=3., verbose=True):
    '''Swathe/distance-to-fault re-evaluation of interactively picked profiles
    (pick_profiles.py conventions: fault_utm on the drawn trace, strike).

    Each pick is matched to the nearest strand of ``fault.trace`` by its UTM
    location (not fault_id, which goes stale if the trace pickle is rebuilt),
    and the strand is oriented to agree with the pick's strike.

    Key kwargs (m):
      plen: profile half-length across the fault.
      swathe_half_width: half-width of the swathe along strike.
      strike_half_window: half-length of refined trace whose mean strike sets
        the profile direction (and the displacement projection).
      refine_half_length: half-length of trace refined on strain. Pixels whose
        nearest fault point lies beyond it see the trace extended straight.
      step_search_half_width / step_stack: coarse displacement-step location
        (None skips it).
      stages: (search half-width, along-strike smoothing sigma, max lateral
        slope) per strain refinement pass, coarse -> fine (see _refine_trace).
      strain_sigma: Gaussian pre-smoothing (px) of EW/NS before differentiating.
      peak_quality: a ridge offset is kept only where the strain there exceeds
        this multiple of the background (median across the widest search).
      prior_weight: strength of the pull towards the previous stage's trace
        (score units per row at the edge of the search window).
      offset_sigma: smoothing (m) of the accepted ridge offsets along strike.
      bin_kw: kwargs for decaying_bin_edges; min_count: minimum pixels per bin.
      near_pixels_within: keep the raw pixel cloud within this distance of the
        fault on the profile (.near_pixels) for plotting; None/0 to skip.
      min_support / min_quality: the profile is flagged .localized when, within
        +/-swathe_half_width of the pick in the final refinement stage, at least
        this fraction of ridge picks were accepted AND their median strain is
        at least this multiple of background. Otherwise the deformation is
        distributed (no sharp surface rupture) and x=0 is essentially the
        displacement-step location -- treat it with caution.
    '''
    bin_kw = {} if bin_kw is None else dict(bin_kw)
    kw = dict(plen=plen, swathe_half_width=swathe_half_width,
              strike_half_window=strike_half_window,
              refine_half_length=refine_half_length,
              step_search=step_search_half_width, step_stack=step_stack,
              trace_smooth=trace_smooth, stages=stages,
              strain_sigma=strain_sigma, peak_quality=peak_quality,
              prior_weight=prior_weight, offset_sigma=offset_sigma,
              edges=decaying_bin_edges(plen, **bin_kw), min_count=min_count,
              near_pixels_within=near_pixels_within, min_support=min_support,
              min_quality=min_quality)
    out = []
    for k, p in enumerate(picked):
        if verbose:
            print(f'[{k + 1}/{len(picked)}] fault {p.fault_id} @ '
                  f'{p.x_along_fault / 1000.:.1f} km along strike')
        out.append(_evaluate_swathe_at(opt, fault.trace, p, **kw))
    return out


def _evaluate_swathe_at(opt, trace, p, plen, swathe_half_width,
                        strike_half_window, refine_half_length, step_search,
                        step_stack, trace_smooth, stages, strain_sigma,
                        peak_quality, prior_weight, offset_sigma, edges,
                        min_count, near_pixels_within, min_support, min_quality):
    pick_xy = np.asarray(p.fault_utm, dtype=float)[:2]
    strand_id, strand, s0 = _match_strand(trace, pick_xy, p.strike)

    # ----- drawn-trace segment around the pick -----
    seg_half = max(refine_half_length, step_stack) + trace_smooth + 50.
    lo = np.clip(s0 - seg_half, 0., max(strand.length - 2. * seg_half, 0.))
    seg = substring(strand, lo, lo + 2. * seg_half).segmentize(25.)
    seg_gdf = gpd.GeoDataFrame(geometry=[LineString(seg.coords)], crs=p.trace.crs)

    # ----- near-fault raster window (step stack + strain refinement) -----
    margin = (step_search or 0.) + 200. + stages[0][0]
    minx, miny, maxx, maxy = seg.bounds
    win = _read_window(opt, minx - margin, maxx + margin,
                       miny - margin, maxy + margin)

    # smoothed drawn trace at 1 px spacing (pixel coords of the window)
    _, _, angle, x0, y0 = _rasterise_and_strike(seg_gdf, win['transform'],
                                                win['res'], trace_smooth)
    theta_perp = np.deg2rad(angle + 90.)
    cx, cy = win['transform'] * (x0, y0)
    j = int(np.argmin(np.hypot(cx - pick_xy[0], cy - pick_xy[1])))

    # ----- 1. coarse location on the displacement step -----
    step_shift = 0.
    if step_search is not None:
        jj = int(np.clip(j, step_stack, x0.size - step_stack - 1))
        H, W = win['ew'].shape
        # _step_refine_shift expects nodata = NaN and NS south-positive
        step_shift = _step_refine_shift(
            win['ew'], -win['nn'], H, W, x0, y0, angle, theta_perp, jj,
            step_stack, step_search / win['res'], 5) * win['res']

    # coarse trace in UTM, shifted along its local left normal onto the step
    # (the _step_refine_shift sign convention: + = towards +xs / left of strike)
    coarse = np.column_stack([cx, cy]) + step_shift * _left_normal(angle)

    # ----- 2. strain refinement on the straightened grid -----
    i0 = max(0, j - int(refine_half_length))
    i1 = min(coarse.shape[0], j + int(refine_half_length) + 1)
    strain = _strain_maps(win, strain_sigma)
    refined, diag = _refine_trace(coarse[i0:i1], strain, stages, peak_quality,
                                  prior_weight, offset_sigma, j - i0)
    jc = j - i0   # pick index into the refined trace

    # ----- 3. profile frame: mean strike of the refined trace -----
    t_hat = _mean_direction(refined, jc, strike_half_window)
    u_hat = np.array([-t_hat[1], t_hat[0]])    # left of strike = +xs
    centre = refined[jc]
    strike = np.degrees(np.arctan2(t_hat[0], t_hat[1])) % 360.

    # ----- 4. swathe pixels + signed distance to the refined trace -----
    sw = _swathe_pixels(opt, centre, t_hat, u_hat, plen, swathe_half_width)
    fault_line = _extend_line(refined, 1.5 * plen)
    d = _signed_distance(sw['xy'], fault_line,
                         near=np.abs((sw['xy'] - centre) @ u_hat) <= 300.)

    # ----- 5. resolve onto the mean strike and stack by distance -----
    par = sw['ew'] * t_hat[0] + sw['nn'] * t_hat[1]
    nrm = sw['ew'] * u_hat[0] + sw['nn'] * u_hat[1]
    e_tn = _sample_strain_tn(strain, sw['xy'], t_hat, u_hat)
    st = _bin_stack(d, np.vstack([par, nrm, e_tn]), edges, min_count)

    # ----- output Profile -----
    line = LineString([centre - plen * u_hat, centre + plen * u_hat])
    gdf = gpd.GeoDataFrame({'x_along_fault': [p.x_along_fault]},
                           geometry=[line], crs=p.trace.crs)
    ep = Profile(trace=gdf, fault_x=plen)
    ep.method = 'swathe'
    ep.xs = st['x']
    ep.displacements = st['median'][0:2]
    ep.displacements_std = st['robust_std'][0:2]   # 1.4826 * MAD of the bin
    ep.counts = st['count']
    ep.bin_edges = edges
    ep.strain_tn = st['median'][2]                 # eps_tn (mean-strike frame)

    ep.fault_id = p.fault_id
    ep.strand_id = strand_id
    ep.x_along_fault = p.x_along_fault
    ep.strike = strike
    ep.fault_utm = tuple(coarse[j])              # step-refined drawn trace
    ep.fault_utm_refined = tuple(centre)         # xs = 0
    ep.step_shift = step_shift
    ep.total_shift = float((centre - pick_xy) @ u_hat)  # drawn -> refined, m
    ep.refined_trace = LineString(refined)
    ep.swathe_half_width = swathe_half_width
    ep.keep_extent = getattr(p, 'keep_extent', None)
    ep.refine_diag = diag
    near = np.abs(diag['s']) <= swathe_half_width
    last = diag['stages'][-1]
    ep.refine_support = float(np.isfinite(last['accepted'][near]).mean())
    ep.refine_quality = float(np.nanmedian(last['quality'][near]))
    ep.localized = bool(ep.refine_support >= min_support
                        and ep.refine_quality >= min_quality)
    ep.picked = p

    if near_pixels_within:
        u = np.abs(d) <= near_pixels_within
        ep.near_pixels = dict(d=d[u].astype(np.float32),
                              par=par[u].astype(np.float32),
                              nrm=nrm[u].astype(np.float32),
                              along=((sw['xy'][u] - centre) @ t_hat).astype(np.float32))

    # crop to the pick's trusted data span (drawn-trace frame -> refined frame)
    xlo = p.xs.min() - ep.total_shift
    xhi = p.xs.max() - ep.total_shift
    u = (ep.xs >= xlo) & (ep.xs <= xhi)
    ep.xs = ep.xs[u]
    ep.displacements = ep.displacements[:, u]
    ep.displacements_std = ep.displacements_std[:, u]
    ep.counts = ep.counts[u]
    ep.strain_tn = ep.strain_tn[u]
    return ep


# ---------------------------------------------------------------------------
#  Distance binning
# ---------------------------------------------------------------------------
def decaying_bin_edges(dmax, full_res_bin=1., full_res_dist=30., mid_dist=200.,
                       mid_growth=1.05, far_growth=1.12, max_bin=100.):
    '''Bin edges symmetric about (and with an edge at) the fault, d = 0:
    full_res_bin wide out to full_res_dist (native 1 m resolution), then each
    bin mid_growth x the previous out to mid_dist, then far_growth x, capped
    at max_bin. Defaults give ~1 m to 30 m, ~6 m at 200 m, 100 m beyond ~1 km.'''
    e = [0.]
    w = full_res_bin
    while e[-1] < dmax:
        if e[-1] >= mid_dist:
            w *= far_growth
        elif e[-1] >= full_res_dist:
            w *= mid_growth
        w = min(w, max_bin)
        e.append(e[-1] + w)
    e = np.asarray(e)
    return np.concatenate([-e[:0:-1], e])


def _bin_stack(d, values, edges, min_count):
    '''Median, robust std (1.4826 MAD) and count of each row of ``values`` in
    bins of ``d``; x is the median d of the bin's members. Bins with fewer than
    min_count finite pixels are NaN.'''
    nb = edges.size - 1
    k = np.digitize(d, edges) - 1
    inside = (k >= 0) & (k < nb)
    order = np.argsort(k[inside], kind='stable')
    ks = k[inside][order]
    ds = d[inside][order]
    vs = values[:, inside][:, order]
    splits = np.searchsorted(ks, np.arange(1, nb))

    nv = values.shape[0]
    out = dict(x=np.full(nb, np.nan), count=np.zeros(nb, int),
               median=np.full((nv, nb), np.nan),
               robust_std=np.full((nv, nb), np.nan))
    for b, (db, vb) in enumerate(zip(np.split(ds, splits),
                                     np.split(vs, splits, axis=1))):
        good = np.isfinite(vb[0])
        out['count'][b] = good.sum()
        if good.sum() < min_count:
            continue
        out['x'][b] = np.median(db[good])
        med = np.nanmedian(vb[:, good], axis=1)
        out['median'][:, b] = med
        out['robust_std'][:, b] = 1.4826 * np.nanmedian(
            np.abs(vb[:, good] - med[:, None]), axis=1)
    keep = np.isfinite(out['x'])
    out['x'][~keep] = 0.5 * (edges[:-1] + edges[1:])[~keep]
    return out


def step_width(x, y, half=60., inner=30.):
    '''Diagnostic: (10-90 % width, amplitude) of the displacement step at x=0.
    A line + step (a + b x + c sign(x)) is fitted to inner <= |x| <= half;
    after removing the line, the profile within +/-half is normalised to the
    step (0 -> 1), made monotonic, and the 10 % / 90 % crossings measured.
    Returns (nan, nan) if there are too few points.'''
    u = np.isfinite(y) & (np.abs(x) <= half)
    x, y = x[u], y[u]
    side = np.abs(x) >= inner
    if side.sum() < 6 or (x[side] < 0).sum() < 2 or (x[side] > 0).sum() < 2:
        return np.nan, np.nan
    A = np.column_stack([np.ones(side.sum()), x[side], np.sign(x[side])])
    c = np.linalg.lstsq(A, y[side], rcond=None)[0]
    z = (y - c[0] - c[1] * x) / (2. * c[2]) + 0.5      # -c..+c -> 0..1
    o = np.argsort(x)
    xo, zo = x[o], np.maximum.accumulate(z[o])
    return np.interp(0.9, zo, xo) - np.interp(0.1, zo, xo), 2. * c[2]


# ---------------------------------------------------------------------------
#  Trace refinement on shear strain
# ---------------------------------------------------------------------------
def _refine_trace(trace_xy, strain, stages, peak_quality, prior_weight,
                  offset_sigma, jc, dn=0.5, diag_half=150):
    '''Iteratively move a ~1 m-spaced trace (N, 2; UTM) onto the ridge of
    |eps_tn|. Returns (refined (N, 2), diag).

    Each stage (search half-width, along-strike sigma, max lateral slope):
      - smooths the current trace with sigma = search / 2 into a BASE curve,
        whose normals cannot cross within +/-search (no folding);
      - samples |eps_tn| on the straightened grid base + n * normal,
        n in [-search, search], and smooths it along strike;
      - tracks the ridge as the best continuous path (_track_ridge): a
        Viterbi path through log(strain / background) with a bounded lateral
        slope and a weak quadratic pull towards the previous trace;
      - keeps path offsets only where the ridge stands peak_quality x above
        the background (row median of the first, widest stage; others are
        interpolated along strike), refines them to
        sub-sample precision, smooths by offset_sigma and moves the trace.

    diag['stages'] holds per stage: the grid n, the straightened image (rows
    within diag_half of the pick; final stage only), and per-row path / accepted / final
    offsets (relative to that stage's base curve), the prior and quality.'''
    xy = trace_xy.copy()
    N = xy.shape[0]
    s = np.concatenate([[0.], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    s = s - s[jc]
    rows = slice(max(0, jc - diag_half), min(N, jc + diag_half + 1))
    diag = dict(s=s, s_rows=s[rows], stages=[])

    bg = None
    for search, along_sigma, max_slope in stages:
        base = gaussian_filter1d(xy, search / 2., axis=0, mode='nearest')
        t, u = _tangents(base)
        prior = np.einsum('ij,ij->i', xy - base, u)   # previous trace, base frame
        n = np.arange(-search, search + dn / 2., dn)
        P = base[:, None, :] + n[None, :, None] * u[:, None, :]
        A = np.abs(_sample_strain_tn(strain, P.reshape(-1, 2),
                                     np.repeat(t, n.size, axis=0),
                                     np.repeat(u, n.size, axis=0))).reshape(N, n.size)
        A = _nan_gaussian1d(A, along_sigma, axis=0)

        # strain relative to the off-fault background, taken from the first
        # (widest) stage: in narrow windows the row median IS the ridge
        if bg is None:
            bg = np.nanmedian(A, axis=1)
            bg = np.where(np.isfinite(bg), bg, np.nanmedian(bg))
        rel = A / bg[:, None]
        # log-compressed: damps isolated hot pixels without flattening the
        # ridge top (a clipped score saturates on strong ridges, leaving the
        # path free to drift towards the prior within the ridge)
        score = np.nan_to_num(np.log(np.maximum(rel, 1.)), nan=0.)
        score -= prior_weight * ((n[None, :] - prior[:, None]) / search) ** 2
        k = _track_ridge(score, max(1, int(round(max_slope / dn))))

        # sub-sample refinement (parabola through the path sample + neighbours)
        r = np.arange(N)
        kk = np.clip(k, 1, n.size - 2)
        a0, a1, a2 = A[r, kk - 1], A[r, kk], A[r, kk + 1]
        den = a0 - 2. * a1 + a2
        frac = np.where((np.abs(den) > 0) & (k == kk), 0.5 * (a0 - a2) / den, 0.)
        path = n[k] + np.clip(np.nan_to_num(frac), -0.5, 0.5) * dn
        quality = rel[r, k]
        acc = np.where(quality >= peak_quality, path, np.nan)

        valid = np.isfinite(acc)
        skipped = valid.sum() < 0.2 * N   # too little signal: keep previous trace
        if skipped:
            off = prior
        else:
            off = np.interp(s, s[valid], acc[valid])
            off = gaussian_filter1d(off, offset_sigma, mode='nearest')
        xy = base + off[:, None] * u
        diag['stages'].append(dict(search=search, along_sigma=along_sigma,
                                   max_slope=max_slope, n=n, image=A[rows],
                                   path=path,
                                   accepted=acc, offset=off, prior=prior,
                                   quality=quality, skipped=skipped))
    # keep the straightened image of the final stage only (pickle size)
    for st in diag['stages'][:-1]:
        st['image'] = None
    diag['stages'][-1]['image'] = diag['stages'][-1]['image'].astype(np.float32)
    return xy, diag


def _track_ridge(score, m, step_penalty=0.05):
    '''Viterbi path k[i] maximising sum(score[i, k[i]]) subject to
    |k[i] - k[i-1]| <= m, with a small penalty per lateral sample moved.'''
    N, K = score.shape
    V = score[0].copy()
    back = np.zeros((N, K), dtype=np.int8)
    shifts = np.arange(-m, m + 1)
    for i in range(1, N):
        # cand[s, k] = V[k + shift_s] - penalty, out-of-range = -inf
        cand = np.full((shifts.size, K), -np.inf)
        for si, sh in enumerate(shifts):
            if sh >= 0:
                cand[si, :K - sh] = V[sh:] - step_penalty * abs(sh)
            else:
                cand[si, -sh:] = V[:sh] - step_penalty * abs(sh)
        best = np.argmax(cand, axis=0)
        V = cand[best, np.arange(K)] + score[i]
        back[i] = shifts[best]
    k = np.empty(N, dtype=int)
    k[-1] = int(np.argmax(V))
    for i in range(N - 1, 0, -1):
        k[i - 1] = k[i] + back[i, k[i]]
    return k


def _strain_maps(win, sigma):
    '''Infinitesimal strain components (E-N frame) of the window from
    NaN-aware Gaussian-smoothed EW / NS (north-positive) displacement.'''
    ew = _nan_gaussian(win['ew'], sigma)
    nn = _nan_gaussian(win['nn'], sigma)
    xc, yc = win['xc'], win['yc']
    dEdy, dEdx = np.gradient(ew, yc, xc)
    dNdy, dNdx = np.gradient(nn, yc, xc)
    return dict(exx=dEdx, eyy=dNdy, exy=0.5 * (dEdy + dNdx), xc=xc, yc=yc)


def _sample_strain_tn(strain, xy, t, u):
    '''eps_tn = t . eps . u at UTM points xy (M, 2) (bilinear, NaN outside),
    with t/u either a single unit vector (2,) or one per point (M, 2).'''
    xc, yc = strain['xc'], strain['yc']
    idx = np.vstack([(xy[:, 1] - yc[0]) / (yc[1] - yc[0]),
                     (xy[:, 0] - xc[0]) / (xc[1] - xc[0])])
    exx, eyy, exy = (map_coordinates(strain[c], idx, order=1, cval=np.nan)
                     for c in ('exx', 'eyy', 'exy'))
    t = np.broadcast_to(t, xy.shape)
    u = np.broadcast_to(u, xy.shape)
    return (t[:, 0] * u[:, 0] * exx + t[:, 1] * u[:, 1] * eyy
            + (t[:, 0] * u[:, 1] + t[:, 1] * u[:, 0]) * exy)


# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------
def _match_strand(trace, xy, strike):
    '''Nearest strand to xy, oriented so its direction agrees with ``strike``
    (bearing, deg). Returns (strand index, oriented LineString, arc position).'''
    geoms = list(trace.geometry)
    pt = Point(xy)
    k = int(np.argmin([g.distance(pt) for g in geoms]))
    g = geoms[k]
    s0 = g.project(pt)
    a = np.asarray(g.interpolate(max(s0 - 50., 0.)).coords[0])[:2]
    b = np.asarray(g.interpolate(min(s0 + 50., g.length)).coords[0])[:2]
    t = np.array([np.sin(np.radians(strike)), np.cos(np.radians(strike))])
    if (b - a) @ t < 0:
        g = shapely.reverse(g)
        s0 = g.length - s0
    return k, g, s0


def _left_normal(angle):
    '''Unit vector(s) (E, N) 90 deg anticlockwise of strike bearing(s) angle.'''
    a = np.radians(np.atleast_1d(angle))
    return np.column_stack([-np.cos(a), np.sin(a)])


def _tangents(xy, h=5):
    '''Unit tangents / left normals of a ~1 m-spaced polyline from central
    differences over +/-h samples (one-sided at the ends).'''
    N = xy.shape[0]
    i = np.arange(N)
    d = xy[np.minimum(i + h, N - 1)] - xy[np.maximum(i - h, 0)]
    t = d / np.hypot(d[:, 0], d[:, 1])[:, None]
    return t, np.column_stack([-t[:, 1], t[:, 0]])


def _mean_direction(xy, jc, half):
    '''Principal direction of trace points within +/-half m (arc length) of
    index jc, oriented along the trace.'''
    s = np.concatenate([[0.], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    sel = np.abs(s - s[jc]) <= half
    pts = xy[sel] - xy[sel].mean(axis=0)
    v = np.linalg.svd(pts, full_matrices=False)[2][0]
    return v if v @ (xy[sel][-1] - xy[sel][0]) > 0 else -v


def _extend_line(xy, length, n_end=30):
    '''Polyline extended straight by ``length`` at both ends along the mean
    direction of its last n_end samples.'''
    t0 = xy[0] - xy[min(n_end, xy.shape[0] - 1)]
    t1 = xy[-1] - xy[-1 - min(n_end, xy.shape[0] - 1)]
    t0 /= np.hypot(*t0); t1 /= np.hypot(*t1)
    return np.vstack([xy[0] + length * t0, xy, xy[-1] + length * t1])


def _signed_distance(pts, line_xy, near, tol_near=0.02, tol_far=0.5, chunk=4000):
    '''Exact shortest distance of pts (M, 2) to a polyline, signed positive on
    the LEFT of the line direction. Brute force over segments of the line
    simplified to tol_near (points flagged ``near``) or tol_far (the rest,
    where a decimetre error is irrelevant), in chunks of points.'''
    out = np.empty(pts.shape[0])
    for mask, tol in ((near, tol_near), (~near, tol_far)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        v = np.asarray(LineString(line_xy).simplify(tol).coords)
        a, ab = v[:-1], np.diff(v, axis=0)
        ab2 = np.einsum('ij,ij->i', ab, ab)
        for c in range(0, idx.size, chunk):
            ic = idx[c:c + chunk]
            ap = pts[ic, None, :] - a[None]                       # (c, S, 2)
            f = np.clip(np.einsum('csk,sk->cs', ap, ab) / ab2, 0., 1.)
            r = ap - f[..., None] * ab[None]
            d2 = np.einsum('csk,csk->cs', r, r)
            kmin = np.argmin(d2, axis=1)
            rows = np.arange(ic.size)
            cross = (ab[kmin, 0] * ap[rows, kmin, 1]
                     - ab[kmin, 1] * ap[rows, kmin, 0])
            out[ic] = np.where(cross >= 0., 1., -1.) * np.sqrt(d2[rows, kmin])
    return out


# ---------------------------------------------------------------------------
#  Raster access
# ---------------------------------------------------------------------------
def _read_window(opt, x0, x1, y0, y1):
    '''EW / NS (north-positive) window with nodata zeros -> NaN, plus pixel-
    centre coordinates, the window's affine transform and pixel size.'''
    sel = dict(x=slice(x0, x1), y=slice(y1, y0))
    ew_da = opt.ew.sel(**sel)
    ew = ew_da.values.astype(float)
    nn = opt.ns.sel(**sel).values.astype(float)
    ew[ew == 0.] = np.nan
    nn[nn == 0.] = np.nan
    return dict(ew=ew, nn=nn, xc=ew_da.x.values.astype(float),
                yc=ew_da.y.values.astype(float),
                transform=ew_da.rio.transform(),
                res=abs(ew_da.rio.transform().a))


def _swathe_pixels(opt, centre, t_hat, u_hat, plen, half_width, chunk=250.):
    '''Pixel centres (UTM) and EW / NS (north-positive) values of every valid
    pixel in the rectangle |along| <= half_width, |across| <= plen, read in
    chunks along the profile so the bounding boxes stay small.'''
    xy, ew, nn = [], [], []
    for b0 in np.arange(-plen, plen, chunk):
        b1 = min(b0 + chunk, plen)
        corners = np.array([centre + a * t_hat + b * u_hat
                            for a in (-half_width, half_width) for b in (b0, b1)])
        (x0, y0), (x1, y1) = corners.min(axis=0) - 2., corners.max(axis=0) + 2.
        w = _read_window(opt, x0, x1, y0, y1)
        if w['ew'].size == 0:
            continue
        X, Y = np.meshgrid(w['xc'], w['yc'])
        rel = np.column_stack([X.ravel() - centre[0], Y.ravel() - centre[1]])
        a = rel @ t_hat
        b = rel @ u_hat
        e = w['ew'].ravel(); n = w['nn'].ravel()
        # half-open in b so pixels on chunk boundaries are taken once
        m = ((np.abs(a) <= half_width) & (b >= b0) & ((b < b1) | (b1 == plen))
             & (b <= plen) & np.isfinite(e) & np.isfinite(n))
        xy.append(rel[m] + centre); ew.append(e[m]); nn.append(n[m])
    return dict(xy=np.vstack(xy), ew=np.concatenate(ew), nn=np.concatenate(nn))


# ---------------------------------------------------------------------------
#  NaN-aware smoothing
# ---------------------------------------------------------------------------
def _nan_gaussian(a, sigma, min_weight=0.25):
    '''2D Gaussian smoothing that ignores NaNs (normalised convolution).'''
    valid = np.isfinite(a)
    num = gaussian_filter(np.where(valid, a, 0.), sigma, mode='nearest')
    den = gaussian_filter(valid.astype(float), sigma, mode='nearest')
    out = num / den
    out[(den < min_weight) | ~valid] = np.nan
    return out


def _nan_gaussian1d(a, sigma, axis, min_weight=0.25):
    '''1D Gaussian smoothing along ``axis`` that ignores NaNs.'''
    valid = np.isfinite(a)
    num = gaussian_filter1d(np.where(valid, a, 0.), sigma, axis=axis, mode='nearest')
    den = gaussian_filter1d(valid.astype(float), sigma, axis=axis, mode='nearest')
    out = num / den
    out[den < min_weight] = np.nan
    return out
