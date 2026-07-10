'''Fault-aligned profiling: strain-relocated, along-strike-stacked
fault-perpendicular profiles. Serial port of `fault_profile8.py` (which cannot
be imported: it runs a script at import time), extracted from OpticalData.

Two drivers, sharing the same pipeline machinery:
  evaluate_profiles_fault_aligned(opt, fault, ...)   profile everywhere along the trace
  evaluate_picked_profiles(opt, fault, picked, ...)  accurate re-evaluation at picked
                                                     locations only (one stack per pick)

Pipeline: resample+smooth the drawn trace -> local strike -> per-point
perpendicular profiles (displacement + finite strain) -> sub-pixel relocation
onto peak shear strain -> along-strike median stacking -> second finer re-shift
-> per-profile detrend / fault-zone-width / offset analysis. All distances
internally in pixels (1 px = 1 m for the Ridgecrest tifs); NS is flipped to
south-positive internally to match the original script's sign convention.
'''
import os
import warnings
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge, substring
from scipy.ndimage import map_coordinates, gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from .Profile import Profile


def _quiet_runtimewarnings(fn):
    '''Suppress the expected NaN-reduction RuntimeWarnings raised by the
    strain/stacking pipeline (empty-slice means, all-NaN max, divide-by-zero).'''
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
#  Per-profile extraction workers.
#
#  The per-profile extraction is embarrassingly parallel. The large read-only
#  rasters and geometry arrays are stashed in the module global ``_FA_WORKER``
#  once in the parent and inherited by forked workers (copy-on-write), so they
#  are never pickled per task. Fork only works on Linux; serial on macOS.
# ---------------------------------------------------------------------------
_FA_WORKER = {}


def _fa_crop_window(prof_pts, arrays, H, W, expwd=5):
    '''Crop identical padded raster windows around (N, 2) (col, row) sample
    points. Returns (cropped_list, tmp_prof) where tmp_prof is the (2, N)
    window-local [row, col] sample array, or (None, None) if degenerate.'''
    r0 = int(np.min(prof_pts[:, 1])) - expwd
    r1 = int(np.max(prof_pts[:, 1])) + expwd
    c0 = int(np.min(prof_pts[:, 0])) - expwd
    c1 = int(np.max(prof_pts[:, 0])) + expwd
    r0c, r1c = max(0, r0), min(H, r1)
    c0c, c1c = max(0, c0), min(W, c1)
    if (r1c - r0c) < 3 or (c1c - c0c) < 3:
        return None, None
    tmp_prof = np.fliplr(prof_pts).T.copy()   # row 0 = y(row), row 1 = x(col)
    tmp_prof[0, :] -= r0c
    tmp_prof[1, :] -= c0c
    cropped = [a[r0c:r1c, c0c:c1c] for a in arrays]
    return cropped, tmp_prof


def _fa_extract_profile_impl(i, ctx):
    '''Extract one fault-perpendicular profile: displacement over the full
    half-length ``plen``, finite strain only over the central ``strain_plen``
    window (where it is needed for relocation/FZW).

    Returns (i, cs, ce, strain_arr, disp_arr):
        strain_arr: (3, 2*strain_plen+1)  rows [shear, dilatation, maxShear]
        disp_arr:   (4, 2*plen+1)         rows [EW, NS(S+), normal, parallel]
        cs, ce: inclusive column span (into the full M-length arrays) that
                ``strain_arr`` occupies.
    '''
    ew = ctx['ew']; ns = ctx['ns']
    H = ctx['H']; W = ctx['W']
    M = ctx['M']; plen = ctx['plen']; strain_plen = ctx['strain_plen']
    xres = ctx['xres']; yres = ctx['yres']; expwd = ctx['expwd']
    angle_i = float(ctx['angle'][i])
    p1 = (ctx['x1'][i], ctx['y1'][i])
    p2 = (ctx['x2'][i], ctx['y2'][i])

    prof = np.linspace([p1[0], p1[1]], [p2[0], p2[1]], num=M)   # (M, 2) col, row

    # ----- displacement over the full profile (cheap: no gradients) -----
    # sample EW/NS along the line FIRST, then rotate the 1-D samples into
    # parallel/normal: bilinear sampling and the pointwise rotation commute,
    # and this avoids window-sized rotation arrays (the bbox of a diagonal
    # profile can be ~plen x plen).
    disp_arr = np.full((4, M), np.nan)
    cropped, tmp_prof = _fa_crop_window(prof, [ew, ns], H, W, expwd)
    if cropped is not None:
        ew_win, ns_win = cropped
        # cval=nan: samples outside the loaded raster window must read as gaps,
        # not as exactly 0, or they pollute the far-field bands downstream
        ew_s = map_coordinates(ew_win, tmp_prof, order=1, cval=np.nan)   # east-positive
        ns_s = -map_coordinates(ns_win, tmp_prof, order=1, cval=np.nan)  # north-positive
        theta = np.radians(90. - angle_i)
        disp_arr[0, :] = ew_s
        disp_arr[1, :] = ns_s
        disp_arr[2, :] = -ew_s * np.sin(theta) + ns_s * np.cos(theta)   # normal
        disp_arr[3, :] = ew_s * np.cos(theta) + ns_s * np.sin(theta)    # parallel

    # ----- finite strain on the central sub-window only -----
    cs = plen - strain_plen
    ce = plen + strain_plen
    Ms = ce - cs + 1
    strain_arr = np.full((3, Ms), np.nan)
    sprof = prof[cs:ce + 1]
    scrop, stmp = _fa_crop_window(sprof, [ew, ns], H, W, expwd)
    if scrop is not None:
        sew, sns = scrop
        try:
            with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
                warnings.simplefilter('ignore', RuntimeWarning)
                strain = finite_strain(sew.copy(), sns.copy(), xres, yres,
                                       angle_i, k=1,
                                       component=['exy', 'dilatation', 'maxShear'])
                s_shear = np.abs(strain[:, :, 0])
                s_dil = strain[:, :, 1]
                s_maxsh = strain[:, :, 2]
                # protect a central band (fault core) from outlier removal. For the
                # full-strain default this is column ``plen`` (original behaviour);
                # for a restricted strain window it is the fault's window-local col.
                buf = int(ctx.get('prot_buffer', 15))
                if strain_plen == plen:
                    prot = plen
                else:
                    wln = sew.shape[1]
                    prot = int(round(stmp[1, strain_plen]))
                    prot = min(max(prot, buf), max(buf, wln - buf))
                s_shear = remove_marginal_outliers(s_shear, prot, buf, 0.5)
                s_dil = remove_marginal_outliers(s_dil, prot, buf, 0.5)
                strain_arr[0, :] = map_coordinates(s_shear, stmp, order=1, cval=np.nan)
                strain_arr[1, :] = map_coordinates(s_dil, stmp, order=1, cval=np.nan)
                strain_arr[2, :] = map_coordinates(s_maxsh, stmp, order=1, cval=np.nan)
        except Exception:
            pass

    return i, cs, ce, strain_arr, disp_arr


def _fa_extract_profile(i):
    '''Pool entry point: reads the forked-in context from the module global.'''
    return _fa_extract_profile_impl(i, _FA_WORKER)


# ---------------------------------------------------------------------------
#  Full-trace driver
# ---------------------------------------------------------------------------
@_quiet_runtimewarnings
def evaluate_profiles_fault_aligned(
        opt, fault,
        plen=500, stack=300, trace_smooth=15,
        strain_half_width=None, n_jobs=None, expwd=5,
        prof_dtype=np.float64,
        shift_cap=8, shift_cap_final=4, search_half_width=None,
        background_limit=(-750., 750.),
        near_field_limit=(10., 60.),
        far_field_limit=None,
        stdthr=3., attach_to_fault=True, store=True):
    '''Stacked fault-aligned profiles at every point along ``fault.trace``.

    ``opt`` is an OpticalData with (windowed, nodata-cleared) ew/ns rasters.
    Returns a list of Profile with xs (m, 0 on the relocated fault),
    displacements ([parallel, normal], m) and analysis metadata; also attaches
    ``fault.refined_trace`` and ``opt.refined_profiles`` (see kwargs).

    Key kwargs (px = m at 1 m/px):
      plen: profile half-length. Displacement spans the full +/-plen cheaply.
      stack: along-strike half-window for the median stack.
      strain_half_width: central window (same units as plen) for finite strain
        (relocation + FZW only) -- keeps strain cost independent of plen.
        None = full profile (original behaviour).
      search_half_width: half-width of the peak-shear search window around the
        drawn trace in the first relocation. Must exceed the drawn-trace error
        but stay below the distance to the nearest parallel strand. None keeps
        the original ~12 px window. Also widens the outlier-protected band and
        raises shift_cap to match.
      n_jobs: >1 forks a ProcessPoolExecutor (Linux only); None/1 = serial.
      prof_dtype: np.float32 halves the profile-buffer memory for long profiles.
      background_limit / near_field_limit / far_field_limit / stdthr: detrend
        band, offset-regression windows (px, relative to FZW edge) and the FZW
        edge threshold (std devs of detrended background strain).
    '''
    if far_field_limit is None:
        far_field_limit = (near_field_limit[1], near_field_limit[1] + 50.)
    background_limit = list(background_limit)
    near_field_limit = list(near_field_limit)
    far_field_limit = list(far_field_limit)

    # ----- raster arrays + affine transform -----
    transform = opt.ew.rio.transform()
    xres = abs(transform.a)
    yres = xres
    crs = opt.ew.rio.crs

    disp_ew_fill = opt.ew.values.astype(float)
    disp_ns_fill = -opt.ns.values.astype(float)   # south-positive (script convention)
    H, W = disp_ew_fill.shape

    strain_plen, search_support, prot_buffer, shift_cap = _resolve_windows(
        plen, strain_half_width, search_half_width, shift_cap, xres)

    n_jobs = 1 if n_jobs is None else max(1, min(int(n_jobs), os.cpu_count() or 1))

    # ----- 1. rasterise + smooth the fault trace into pixel coordinates -----
    x, y, angle, x0, y0 = _rasterise_and_strike(fault.trace, transform, xres,
                                                trace_smooth)
    n_prof = x0.shape[0]
    opt._print(f"Fault trace rasterised to {n_prof} profile positions")
    if n_prof <= 2 * stack + 2:
        raise ValueError(
            f"Trace too short ({n_prof} px) for stack half-width {stack}.")

    # ----- profile end points (perpendicular lines) -----
    theta_perp = np.deg2rad(angle + 90.)
    dxp = plen * np.sin(theta_perp)
    dyp = plen * np.cos(theta_perp)
    x1 = x0 + dxp; y1 = y0 - dyp
    x2 = x0 - dxp; y2 = y0 + dyp

    M = 2 * plen + 1
    prof_dist_pxls = np.linspace(-plen, plen, num=M)

    # ----- 2/3. extract strain and displacement profiles -----
    # prof_store_strain rows: 0 shear, 1 dilatation, 2 maxShear, 3 unused
    # prof_store_disp   rows: 0 EW, 1 NS(S+), 2 fault-normal, 3 fault-parallel
    prof_store_strain = np.full((n_prof, 4, M), np.nan, dtype=prof_dtype)
    prof_store_disp = np.full((n_prof, 4, M), np.nan, dtype=prof_dtype)

    # context shared with the (optionally parallel) per-profile workers
    global _FA_WORKER
    _FA_WORKER = dict(
        ew=disp_ew_fill, ns=disp_ns_fill, H=H, W=W, M=M,
        plen=plen, strain_plen=strain_plen, xres=xres, yres=yres,
        expwd=expwd, angle=angle, x1=x1, y1=y1, x2=x2, y2=y2,
        prot_buffer=prot_buffer)

    def _store(res):
        i, cs, ce, strain_arr, disp_arr = res
        prof_store_strain[i, 0:3, cs:ce + 1] = strain_arr
        prof_store_disp[i, :, :] = disp_arr

    opt._print(f"Extracting {n_prof} profiles "
               f"(plen={plen}px, strain_plen={strain_plen}px, n_jobs={n_jobs})")
    if n_jobs == 1:
        for i in range(n_prof):
            if opt.verbose and (i % 200 == 0):
                opt._print(f"  extracting profile {i} of {n_prof}")
            _store(_fa_extract_profile_impl(i, _FA_WORKER))
    else:
        # forked workers inherit _FA_WORKER (copy-on-write); nothing is
        # pickled per task except the integer index and the small result.
        mp_ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_ctx) as ex:
            futures = [ex.submit(_fa_extract_profile, i) for i in range(n_prof)]
            for done, fut in enumerate(as_completed(futures)):
                if opt.verbose and (done % 200 == 0):
                    opt._print(f"  extracted {done} of {n_prof}")
                _store(fut.result())

    _FA_WORKER = {}   # release the shared rasters held for the workers

    # ----- 4. first relocation: shift each profile onto peak shear strain -----
    w_shift = smoothed_boxcar(plen, search_support, 1)
    store_shifts = np.zeros(n_prof)
    for i in range(n_prof):
        store_shifts[i] = _locate_peak_shift(
            prof_store_strain[i, 0, :], w_shift, plen, shift_cap)

    # interpolate failed (zero) shifts from neighbours
    store_shifts = _interp_failed_shifts(store_shifts)
    # apply shifts: strain rows 0,1,2 and disp rows 0,1,2,3
    tmpx = np.arange(M, dtype=float)
    _apply_shift(prof_store_strain, store_shifts, tmpx, rows=(0, 1, 2))
    store_shifts_disp = replace_outliers_robust(store_shifts, window=10, threshold=1.0)
    _apply_shift(prof_store_disp, store_shifts_disp, tmpx, rows=(0, 1, 2, 3))

    # ----- 5. along-strike median stacking -----
    # strain stats rows: 0 med shear,1 std,2 med dil,3 std,4 med maxShear,5 std
    # disp   stats rows: 0 med EW,1 std,2 med NS,3 std,4 med normal,5 std,6 med par,7 std
    strain_stats = _stack(prof_store_strain, stack, plen, gate_row=1,
                          med_rows=(0, 1, 2), out_rows=(0, 2, 4),
                          min_cols=(2 * strain_plen) // 3)
    disp_stats = _stack(prof_store_disp, stack, plen, gate_row=1,
                        med_rows=(0, 1, 2, 3), out_rows=(0, 2, 4, 6),
                        min_cols=plen // 1.5)
    sl = slice(stack, n_prof - stack)
    strain_stats = strain_stats[sl]
    disp_stats = disp_stats[sl]
    n_stack = strain_stats.shape[0]
    opt._print(f"Stacked into {n_stack} profiles")

    # ----- 6. second (finer) global re-shift of stacked profiles -----
    store_shifts2 = _second_reshift(strain_stats, disp_stats, prof_dist_pxls,
                                    plen, stack, shift_cap_final)

    # ----- 7. per-profile analysis (detrend / FZW / offsets) -----
    profiles = []
    table = np.full((n_stack, 42), np.nan)
    # along-fault distance (m) at each stacked profile centre
    seg = np.hypot(np.diff(x0), np.diff(y0)) * xres
    along_fault = np.concatenate([[0.], np.cumsum(seg)])

    total_shift = store_shifts[sl] + store_shifts2   # px, drawn-trace -> true fault
    for ff in range(n_stack):
        j = ff + stack   # index into full-length trace arrays
        result, shifted = _analyse_profile(
            ff, strain_stats, disp_stats, prof_dist_pxls, plen, xres,
            background_limit, near_field_limit, far_field_limit, stdthr,
            strain_plen)
        table[ff, :len(result)] = result

        # build Profile (geometry in raster CRS, displacements in metres)
        table[ff, 0:5] = [x0[j], y0[j], angle[j],
                          *(transform * (x0[j], y0[j]))]
        ux1, uy1 = transform * (x1[j], y1[j])
        ux2, uy2 = transform * (x2[j], y2[j])
        line = LineString([(ux1, uy1), (ux2, uy2)])
        gdf = gpd.GeoDataFrame({"x_along_fault": [along_fault[j]]},
                               geometry=[line], crs=crs)
        p = Profile(trace=gdf, fault_x=plen * xres)
        p.xs = prof_dist_pxls * xres
        # parallel = disp_stats row 6, fault-normal flipped back to N-positive sign
        p.displacements = np.array([disp_stats[ff, 6, :], disp_stats[ff, 4, :]])

        p.x_along_fault = along_fault[j]
        p.strike = angle[j]
        p.fault_utm = transform * (x0[j], y0[j])
        p.strain_shear = shifted[0, :]
        p.strain_dilatation = shifted[1, :]
        _attach_analysis(p, result)
        profiles.append(p)

    # ----- refined trace (drawn trace shifted perpendicular onto peak strain) -----
    refined_col = x0[sl] - total_shift * np.sin(theta_perp[sl])
    refined_row = y0[sl] + total_shift * np.cos(theta_perp[sl])
    rx, ry = transform * (refined_col, refined_row)
    refined_trace = gpd.GeoDataFrame(
        {"id": [0]}, geometry=[LineString(np.column_stack([rx, ry]))], crs=crs)
    if attach_to_fault:
        fault.refined_trace = refined_trace

    if store:
        opt.refined_profiles = profiles
        opt.refined_profile_table = table
        opt.refined_trace = refined_trace

    return profiles


# ---------------------------------------------------------------------------
#  Picked-profile driver
# ---------------------------------------------------------------------------
@_quiet_runtimewarnings
def evaluate_picked_profiles(opt, fault, picked, plen=4000, stack=150,
                             trace_smooth=15, strain_half_width=200.,
                             search_half_width=100., expwd=5,
                             shift_cap=8, shift_cap_final=4,
                             background_limit=(-750., 750.),
                             near_field_limit=(10., 60.),
                             far_field_limit=None, stdthr=3.,
                             seg_pad=100., step_search_half_width=None,
                             verbose=True):
    '''Accurately re-evaluate interactively picked profiles (pick_profiles.py
    conventions: fault_id, x_along_fault, fault_utm on the drawn trace).

    For each pick, the fault-aligned pipeline runs on a short segment of the
    drawn trace centred on the pick (half-length stack + seg_pad + trace_smooth
    metres) and only the stacked profile at the pick is kept. Output Profiles
    use the fault-aligned conventions (xs=0 on the strain-relocated fault,
    displacements=[parallel, normal]); the pick's trusted data span crops the
    output, and the original pick is attached as ``.picked``.

    step_search_half_width (m or None): if set, a coarse fault-position
    refinement runs FIRST, on the displacement step rather than strain: the
    stacked fault-parallel component is demeaned and its steepest zero
    crossing within this half-width of the drawn trace becomes the expected
    fault position (stored as .step_shift; the trace segment is rigidly
    shifted onto it). The strain relocation then only needs a narrow
    search_half_width around it — more robust where the drawn trace is off by
    more than the distance to the nearest off-fault strain feature.

    Sign note: the fault-parallel component here is the projection onto the
    local strike direction — OPPOSITE in sign to the picked profiles' quick
    parallel row (projection onto -strike).
    '''
    if far_field_limit is None:
        far_field_limit = (near_field_limit[1], near_field_limit[1] + 50.)
    kw = dict(plen=plen, stack=stack, trace_smooth=trace_smooth,
              strain_half_width=strain_half_width,
              search_half_width=search_half_width, expwd=expwd,
              shift_cap=shift_cap, shift_cap_final=shift_cap_final,
              background_limit=list(background_limit),
              near_field_limit=list(near_field_limit),
              far_field_limit=list(far_field_limit), stdthr=stdthr,
              seg_pad=seg_pad, step_search=step_search_half_width)
    out = []
    for k, p in enumerate(picked):
        if verbose:
            print(f'[{k + 1}/{len(picked)}] fault {p.fault_id} @ '
                  f'{p.x_along_fault / 1000.:.1f} km along strike')
        out.append(_evaluate_at(opt, fault.trace.geometry.iloc[p.fault_id], p, **kw))
    return out


def _evaluate_at(opt, strand, p, plen, stack, trace_smooth, strain_half_width,
                 search_half_width, expwd, shift_cap, shift_cap_final,
                 background_limit, near_field_limit, far_field_limit, stdthr,
                 seg_pad, step_search=None):
    # short, densified trace segment centred on the pick (px = m)
    seg_half = stack + seg_pad + trace_smooth
    lo = np.clip(p.x_along_fault - seg_half, 0.,
                 max(strand.length - 2. * seg_half, 0.))
    seg = substring(strand, lo, lo + 2. * seg_half).segmentize(25.)
    trace = gpd.GeoDataFrame(geometry=[LineString(seg.coords)], crs=p.trace.crs)

    # raster window covering the segment corridor; nodata zeros -> NaN
    margin = plen + 200. + (step_search or 0.)
    minx, miny, maxx, maxy = seg.bounds
    ew_da = opt.ew.sel(x=slice(minx - margin, maxx + margin),
                       y=slice(maxy + margin, miny - margin))
    ns_da = opt.ns.sel(x=slice(minx - margin, maxx + margin),
                       y=slice(maxy + margin, miny - margin))
    transform = ew_da.rio.transform()
    xres = abs(transform.a)
    yres = xres
    ew = ew_da.values.astype(float)
    ns = ns_da.values.astype(float)
    ew[ew == 0.] = np.nan
    ns[ns == 0.] = np.nan
    ns = -ns   # south-positive (script convention)
    H, W = ew.shape

    strain_plen, search_support, prot_buffer, shift_cap = _resolve_windows(
        plen, strain_half_width, search_half_width, shift_cap, xres)

    x, y, angle, x0, y0 = _rasterise_and_strike(trace, transform, xres, trace_smooth)
    n_prof = x0.shape[0]
    if n_prof <= 2 * stack + 2:
        raise ValueError(f"Segment too short ({n_prof} px) for stack {stack}.")

    theta_perp = np.deg2rad(angle + 90.)

    # index of the trace point nearest the picked fault location
    pc, pr = ~transform * p.fault_utm
    j = int(np.argmin(np.hypot(x0 - pc, y0 - pr)))
    j = int(np.clip(j, stack, n_prof - stack - 1))

    # optional stage 0: coarse refinement on the displacement STEP (robust to
    # off-fault strain), then the strain search only needs its narrow window.
    # The whole segment is rigidly shifted perpendicular onto the step.
    step_shift = 0.
    if step_search is not None:
        step_shift = _step_refine_shift(ew, ns, H, W, x0, y0, angle, theta_perp,
                                        j, stack, step_search / xres, expwd)
        x0 = x0 - step_shift * np.sin(theta_perp)
        y0 = y0 + step_shift * np.cos(theta_perp)

    x1 = x0 + plen * np.sin(theta_perp); y1 = y0 - plen * np.cos(theta_perp)
    x2 = x0 - plen * np.sin(theta_perp); y2 = y0 + plen * np.cos(theta_perp)
    M = 2 * plen + 1
    prof_dist_pxls = np.linspace(-plen, plen, num=M)

    ctx = dict(ew=ew, ns=ns, H=H, W=W, M=M, plen=plen, strain_plen=strain_plen,
               xres=xres, yres=yres, expwd=expwd, angle=angle,
               x1=x1, y1=y1, x2=x2, y2=y2, prot_buffer=prot_buffer)
    prof_strain = np.full((n_prof, 4, M), np.nan)
    prof_disp = np.full((n_prof, 4, M), np.nan)
    for i in range(n_prof):
        _, cs, ce, strain_arr, disp_arr = _fa_extract_profile_impl(i, ctx)
        prof_strain[i, 0:3, cs:ce + 1] = strain_arr
        prof_disp[i, :, :] = disp_arr

    # first relocation onto peak shear strain
    w_shift = smoothed_boxcar(plen, search_support, 1)
    shifts = np.array([_locate_peak_shift(prof_strain[i, 0, :], w_shift, plen,
                                          shift_cap) for i in range(n_prof)])
    shifts = _interp_failed_shifts(shifts)
    tmpx = np.arange(M, dtype=float)
    _apply_shift(prof_strain, shifts, tmpx, rows=(0, 1, 2))
    shifts_disp = replace_outliers_robust(shifts, window=10, threshold=1.0)
    _apply_shift(prof_disp, shifts_disp, tmpx, rows=(0, 1, 2, 3))

    # stack ONCE, at the pick
    strain_stats = _stack_one(prof_strain, j, stack, gate_row=1,
                              med_rows=(0, 1, 2), out_rows=(0, 2, 4),
                              min_cols=(2 * strain_plen) // 3)[None]
    disp_stats = _stack_one(prof_disp, j, stack, gate_row=1,
                            med_rows=(0, 1, 2, 3), out_rows=(0, 2, 4, 6),
                            min_cols=plen // 1.5)[None]

    # second, finer re-shift of the single stacked profile
    w2 = smoothed_boxcar(plen, 8, 1)
    sh2 = _locate_peak_shift(strain_stats[0, 0, :], w2, plen, shift_cap_final)
    for arr, row in ((strain_stats, 0), (strain_stats, 2),
                     (disp_stats, 4), (disp_stats, 6)):
        col = arr[0, row, :]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            g = interp1d(prof_dist_pxls[valid] - sh2, col[valid], kind='slinear',
                         bounds_error=False, fill_value=np.nan)
            arr[0, row, :] = g(prof_dist_pxls)

    result, shifted = _analyse_profile(0, strain_stats, disp_stats,
                                       prof_dist_pxls, plen, xres,
                                       background_limit, near_field_limit,
                                       far_field_limit, stdthr, strain_plen)

    # ----- output Profile (fault-aligned conventions + pick metadata) -----
    ux1, uy1 = transform * (x1[j], y1[j])
    ux2, uy2 = transform * (x2[j], y2[j])
    gdf = gpd.GeoDataFrame({'x_along_fault': [p.x_along_fault]},
                           geometry=[LineString([(ux1, uy1), (ux2, uy2)])],
                           crs=p.trace.crs)
    ep = Profile(trace=gdf, fault_x=plen * xres)
    ep.xs = prof_dist_pxls * xres
    ep.displacements = np.array([disp_stats[0, 6, :], disp_stats[0, 4, :]])
    ep.displacements_std = np.array([disp_stats[0, 7, :], disp_stats[0, 5, :]])
    ep.strain_shear = shifted[0, :]
    ep.strain_dilatation = shifted[1, :]

    ep.fault_id = p.fault_id
    ep.x_along_fault = p.x_along_fault
    ep.strike = angle[j]
    # x0/y0 already carry the step refinement, so fault_utm here is the
    # step-refined point and the strain shifts are relative to it
    ep.fault_utm = tuple(transform * (x0[j], y0[j]))
    ep.step_shift = step_shift * xres    # m, drawn trace -> displacement step
    total_shift = step_shift + shifts[j] + sh2
    ep.total_shift = total_shift * xres  # m, drawn trace -> relocated fault
    rc = x0[j] - (shifts[j] + sh2) * np.sin(theta_perp[j])
    rr = y0[j] + (shifts[j] + sh2) * np.cos(theta_perp[j])
    ep.fault_utm_refined = tuple(transform * (rc, rr))
    ep.keep_extent = getattr(p, 'keep_extent', None)
    _attach_analysis(ep, result)
    ep.picked = p

    # crop to the pick's trusted data span (drawn-trace frame -> relocated frame)
    xlo = p.xs.min() - ep.total_shift
    xhi = p.xs.max() - ep.total_shift
    u = (ep.xs >= xlo) & (ep.xs <= xhi)
    ep.xs = ep.xs[u]
    ep.displacements = ep.displacements[:, u]
    ep.displacements_std = ep.displacements_std[:, u]
    ep.strain_shear = ep.strain_shear[u]
    ep.strain_dilatation = ep.strain_dilatation[u]
    return ep


# ---------------------------------------------------------------------------
#  Pipeline steps
# ---------------------------------------------------------------------------
def _resolve_windows(plen, strain_half_width, search_half_width, shift_cap, xres):
    '''Resolve the strain window, first-relocation search support, protected
    band and shift cap (all px) from the metre-valued kwargs.'''
    # strain half-length: full profile by default, else the requested central
    # window (clamped). Strain cost ~ window area, so this is the key control
    # on speed for long profiles.
    if strain_half_width is None:
        strain_plen = plen
    else:
        strain_plen = int(round(strain_half_width / xres))
    strain_plen = int(np.clip(strain_plen, 4, plen))

    # first-relocation peak-search window: original behaviour is a ~12 px
    # hard-coded boxcar; a wider window lets the relocation correct larger
    # drawn-trace errors (it needs the outlier-protected band and the shift
    # cap to cover it, or the peak it should move to is masked/culled)
    if search_half_width is None:
        search_support, prot_buffer = 25, 15
    else:
        search_half_width = int(np.clip(search_half_width, 12, strain_plen))
        search_support = 2 * search_half_width + 1
        prot_buffer = max(15, search_half_width + 5)
        shift_cap = max(shift_cap, search_half_width)
    return strain_plen, search_support, prot_buffer, shift_cap


def _step_refine_shift(ew, ns, H, W, x0, y0, angle, theta_perp, j, stack,
                       step_search, expwd, smooth=10.):
    '''Coarse fault-position refinement from the displacement step: median-stack
    the fault-parallel component over the pick's stacking bundle, remove the
    mean, and take the zero crossing with the steepest slope within
    +/-step_search px of the drawn trace. Returns the shift (px, same sign
    convention as the strain shifts) or 0 if no crossing is found.'''
    half = step_search + 150.
    M = int(2 * half) + 1
    sxs = np.linspace(-half, half, M)
    bundle = np.full((2 * stack + 1, M), np.nan)
    for k, i in enumerate(range(j - stack, j + stack + 1)):
        p1 = (x0[i] + half * np.sin(theta_perp[i]),
              y0[i] - half * np.cos(theta_perp[i]))
        p2 = (x0[i] - half * np.sin(theta_perp[i]),
              y0[i] + half * np.cos(theta_perp[i]))
        prof = np.linspace([p1[0], p1[1]], [p2[0], p2[1]], num=M)
        cropped, tmp = _fa_crop_window(prof, [ew, ns], H, W, expwd)
        if cropped is None:
            continue
        ew_s = map_coordinates(cropped[0], tmp, order=1, cval=np.nan)
        ns_s = -map_coordinates(cropped[1], tmp, order=1, cval=np.nan)  # N-positive
        th = np.radians(90. - float(angle[i]))
        bundle[k] = ew_s * np.cos(th) + ns_s * np.sin(th)

    s = np.nanmedian(bundle, axis=0)
    good = np.isfinite(s)
    if good.sum() < M // 2:
        return 0.
    s = np.interp(sxs, sxs[good], s[good])
    s = gaussian_filter1d(s, smooth)

    win = np.abs(sxs) <= step_search
    d = s - s[win].mean()
    dw, xw = d[win], sxs[win]
    cross = np.flatnonzero(np.sign(dw[:-1]) != np.sign(dw[1:]))
    if cross.size == 0:
        return 0.
    slope = np.abs(np.gradient(s, sxs))[win]
    c = cross[np.argmax(slope[cross])]
    # linear interpolation of the crossing position
    return float(xw[c] - dw[c] * (xw[c + 1] - xw[c]) / (dw[c + 1] - dw[c]))


def _rasterise_and_strike(trace, transform, xres, trace_smooth):
    '''Resample a trace GeoDataFrame to ~1pt/px in pixel coords, smooth, return
    (x, y, angle, x0, y0) where x/y are smoothed trace pixel coords, angle is
    the local strike (bearing from N, deg) and x0/y0 the strike-defining points.'''
    merged = unary_union(trace.geometry.values)
    if merged.geom_type == 'MultiLineString':
        merged = linemerge(merged)
    if merged.geom_type == 'MultiLineString':
        merged = max(merged.geoms, key=lambda g: g.length)
        print("  warning: trace was multi-part; using longest segment")
    verts = np.asarray(merged.coords)

    d = np.concatenate([[0.], np.cumsum(np.hypot(np.diff(verts[:, 0]),
                                                 np.diff(verts[:, 1])))])
    uniq = np.unique(d, return_index=True)[1]
    d, verts = d[uniq], verts[uniq]
    fx = interp1d(d, verts[:, 0], kind='cubic')
    fy = interp1d(d, verts[:, 1], kind='cubic')
    npts = int(d[-1] / xres)
    dd = np.linspace(0., d[-1], npts + 1)
    xfine, yfine = fx(dd), fy(dd)
    col, row = ~transform * (xfine, yfine)   # UTM -> pixel
    x = np.asarray(col, dtype=float)
    y = np.asarray(row, dtype=float)

    # smooth pixel trace (drops endpoints, mode='valid')
    k = np.ones(trace_smooth) / trace_smooth
    x = np.convolve(x, k, mode='valid')
    y = np.convolve(y, k, mode='valid')

    # local strike angle (bearing from N), span=1
    span = 1
    angle = np.zeros(x.shape)
    for i in range(span, x.shape[0] - span):
        x0i = np.mean(x[i - span:i]); x1i = np.mean(x[i:i + span])
        y0i = np.mean(y[i - span:i]); y1i = np.mean(y[i:i + span])
        dx, dy = x1i - x0i, y1i - y0i
        angle[i] = (90. + np.rad2deg(np.arctan2(dy, dx))) % 360.
    angle = angle[span:-span]
    x0 = x[span:-span]
    y0 = y[span:-span]
    return x, y, angle, x0, y0


def _locate_peak_shift(shear, w, plen, cap):
    '''Sub-pixel offset (px) of the peak (weighted) shear strain from centre,
    clipped to +/-cap. Returns 0 on failure.'''
    ysp = shear * w
    xsp = np.linspace(0, ysp.shape[0] - 1, ysp.shape[0])
    valid = ~np.isnan(ysp)
    if valid.sum() < 4:
        return 0.
    try:
        spline = UnivariateSpline(xsp[valid], ysp[valid], s=1e-9)
        # evaluate only over the valid span (strain may exist only in a small
        # central window; the peak cannot lie in the extrapolated fringe)
        xlo, xhi = xsp[valid].min(), xsp[valid].max()
        x_fine = np.linspace(xlo, xhi, int((xhi - xlo) * 1e3) + 1)
        y_fine = spline(x_fine)
        x_shift = x_fine[np.nanargmax(y_fine)] - (xsp.shape[0] / 2.)
        if (x_shift >= cap) or (x_shift <= -cap):
            x_shift = 0.
    except Exception:
        x_shift = 0.
    return x_shift


def _interp_failed_shifts(store_shifts):
    valid = store_shifts != 0
    if valid.sum() < 2:
        return store_shifts
    rows = np.arange(store_shifts.shape[0])
    f = interp1d(rows[valid], store_shifts[valid], kind='slinear',
                 bounds_error=False, fill_value=np.nan)
    out = f(rows)
    out[np.isnan(out)] = 0.
    return out


def _apply_shift(prof_store, shifts, tmpx, rows):
    '''Re-interpolate selected channels of prof_store onto a shifted x-axis.'''
    for i in range(prof_store.shape[0]):
        tmpx2 = tmpx - shifts[i]
        for p in rows:
            col = prof_store[i, p, :]
            valid = np.isfinite(col) & np.isfinite(tmpx2)
            if valid.sum() < 2:
                prof_store[i, p, :] = np.nan
                continue
            try:
                f = interp1d(tmpx2[valid], col[valid], kind='slinear',
                             bounds_error=False, fill_value=np.nan)
                prof_store[i, p, :] = f(tmpx)
            except Exception:
                prof_store[i, p, :] = np.nan


def _stack_one(prof_store, i, stack, gate_row, med_rows, out_rows, min_cols):
    '''Median+std stack of profile i over its +/-stack along-strike neighbours.
    Returns an (8, M) stats row (all-NaN if too few well-populated columns).'''
    M = prof_store.shape[2]
    out = np.full((8, M), np.nan, dtype=prof_store.dtype)
    block_gate = prof_store[i - stack:i + stack, gate_row, :]
    track_nan = np.sum(np.isnan(block_gate), axis=0)
    if np.sum(track_nan < stack // 3) >= min_cols:
        for src, dst in zip(med_rows, out_rows):
            block = prof_store[i - stack:i + stack, src, :]
            out[dst, :] = np.nanmedian(block, axis=0)
            out[dst + 1, :] = np.nanstd(block, axis=0)
    return out


def _stack(prof_store, stack, plen, gate_row, med_rows, out_rows, min_cols=None):
    '''Stack every profile. med_rows are source channels; out_rows the median
    destination rows in the (n, 8, M) output (std goes to out_row+1).'''
    if min_cols is None:
        min_cols = plen // 1.5
    n = prof_store.shape[0]
    M = prof_store.shape[2]
    stats = np.full((n, 8, M), np.nan, dtype=prof_store.dtype)
    for i in range(stack, n - stack):
        stats[i] = _stack_one(prof_store, i, stack, gate_row, med_rows,
                              out_rows, min_cols)
    return stats


def _second_reshift(strain_stats, disp_stats, prof_dist_pxls, plen, stack, cap):
    '''Estimate and apply a finer global re-shift of the stacked profiles,
    aligning on the stacked peak shear. Shifts strain rows 0,2 and disp rows
    4,6. Returns the (smoothed) shift array (px).'''
    n = strain_stats.shape[0]
    w = smoothed_boxcar(plen, 8, 1)
    shifts2 = np.zeros(n)
    for ff in range(n):
        ysp = strain_stats[ff, 0, :]
        valid = ~np.isnan(ysp)
        if valid.sum() < 4:
            continue
        try:
            spline = UnivariateSpline(prof_dist_pxls[valid],
                                      (ysp * w)[valid], s=1e-9)
            xlo, xhi = prof_dist_pxls[valid].min(), prof_dist_pxls[valid].max()
            x_fine = np.linspace(xlo, xhi, int((xhi - xlo) * 1e3) + 1)
            y_fine = spline(x_fine)
            xs = x_fine[np.nanargmax(y_fine)]
            if (xs >= cap) or (xs <= -cap):
                xs = 0.
        except Exception:
            xs = 0.
        shifts2[ff] = xs

    valid = shifts2 != 0
    if valid.sum() == 0:
        print("  second re-shift: no valid shifts, skipping")
        return np.zeros(n)
    rows = np.arange(n)
    f = interp1d(rows[valid], shifts2[valid], kind='slinear',
                 bounds_error=False, fill_value=np.nan)
    shifts2 = f(rows)
    good = ~np.isnan(shifts2)
    shifts2[~good] = 0.
    shifts2 = gaussian_filter1d(shifts2, sigma=stack / 0.5)

    # apply: strain rows 0,2 ; disp rows 4,6
    for ff in range(n):
        sh = shifts2[ff]
        for arr, row in ((strain_stats, 0), (strain_stats, 2),
                         (disp_stats, 4), (disp_stats, 6)):
            col = arr[ff, row, :]
            valid = np.isfinite(col)
            if valid.sum() < 2:
                continue
            try:
                g = interp1d(prof_dist_pxls[valid] - sh, col[valid],
                             kind='slinear', bounds_error=False, fill_value=np.nan)
                # samples shifted outside the data range stay NaN (a 0 fill
                # would fabricate zero displacement at the profile ends)
                arr[ff, row, :] = g(prof_dist_pxls)
            except Exception:
                pass
    return shifts2


def _analyse_profile(ff, strain_stats, disp_stats, prof_dist_pxls,
                     plen, xres, background_limit, near_field_limit,
                     far_field_limit, stdthr, strain_plen=None):
    '''Per-profile detrend / fault-zone-width / offset analysis. Returns
    (result_store [len 42], result_shifted [3, M] = shear/dil/parallel).

    ``strain_plen`` is the half-width (px) over which strain exists; when it is
    smaller than plen the background-strain detrend band and the minimum
    valid-point check are pulled inside the strain window so they still have
    data. Defaults to plen (full-profile strain, original behaviour).'''
    if strain_plen is None:
        strain_plen = plen
    M = prof_dist_pxls.shape[0]
    result = np.full(42, np.nan)
    shifted = np.zeros((3, M))

    ysp = strain_stats[ff, 0, :].copy()    # shear
    yspd = strain_stats[ff, 2, :].copy()   # dilatation
    yspss = disp_stats[ff, 6, :].copy()    # fault-parallel displacement
    xsp = prof_dist_pxls

    w = smoothed_boxcar(plen, 50, 1)
    valid = (np.isnan(ysp) + (ysp == 0)) < 1
    valid2 = (np.isnan(yspd) + (yspd == 0)) < 1
    valid3 = (np.isnan(yspss) + (yspss == 0)) < 1

    # require enough valid strain points: half the profile for full strain,
    # or half the (smaller) strain window when strain is restricted.
    min_valid = min(int(ysp.shape[0] * 0.5), strain_plen)
    if valid.sum() < min_valid:
        return result, shifted

    xsp2 = xsp[valid]
    yspd2 = yspd[valid2]
    yspss2 = yspss[valid3]

    threshold = 0.; threshold2 = 0.; r2 = np.nan

    # --- detrend background shear strain via RANSAC polynomial ---
    # when strain is restricted to a central window, pull the background band
    # (which lies OUTSIDE +/-background_limit) inward so it still has data.
    if strain_plen < plen:
        bg_lo = max(background_limit[0], -0.6 * strain_plen)
        bg_hi = min(background_limit[1], 0.6 * strain_plen)
    else:
        bg_lo, bg_hi = background_limit[0], background_limit[1]
    try:
        bg = (xsp2 <= bg_lo) | (xsp2 >= bg_hi)
        xx = xsp2[bg].reshape(-1, 1)
        yy = ysp[valid][bg].reshape(-1, 1)
        ransac_in = RANSACRegressor(min_samples=0.6,
                                    residual_threshold=0.2 * np.std(yy),
                                    max_trials=100, random_state=42)
        model = make_pipeline(PolynomialFeatures(3), ransac_in)
        model.fit(xx, yy)
        inlier = model.named_steps['ransacregressor'].inlier_mask_
        y_pred = model.predict(xx)
        r2 = r2_score(yy[inlier], y_pred[inlier])
        ysp_background = model.predict(xsp.reshape(-1, 1))
        ysp_detrend = ysp.reshape(-1, 1) - ysp_background
        std2 = np.nanstd(yy - model.predict(xx))
        threshold = stdthr * std2
        threshold2 = threshold
        ysp = np.squeeze(ysp_detrend)
    except Exception:
        print(f"  could not detrend strain, row {ff}")

    # spline fits of detrended shear (and dilatation) for FZW edges,
    # evaluated over the valid (strain-window) span only
    xlo_f, xhi_f = xsp2.min(), xsp2.max()
    x_fine = np.linspace(xlo_f, xhi_f, int((xhi_f - xlo_f) * 1e3) + 1)
    ysp2 = ysp[valid]
    try:
        spline = UnivariateSpline(xsp2, ysp2 * (w[valid]), s=1e-9)
        y_fine = spline(x_fine)
    except Exception:
        y_fine = np.zeros_like(x_fine)
    try:
        spline2 = UnivariateSpline(xsp2, yspd2 * (w[valid2]), s=1e-9)
        y2_fine = spline2(x_fine)
    except Exception:
        y2_fine = np.zeros_like(x_fine)

    try:
        x_shift = x_fine[np.nanargmax(y_fine)]
    except Exception:
        x_shift = 0.

    shifted[0, :] = ysp
    shifted[1, :] = yspd
    shifted[2, :] = yspss

    # fault-zone width from detrended shear (threshold crossings each side)
    x_min, x_max, fzw = _threshold_width(x_fine, y_fine, threshold)
    x_min2, x_max2, fzw2 = _threshold_width(x_fine, y2_fine, threshold2)

    # --- offsets by robust regression of fault-parallel displacement ---
    offset, std_o, xl, xr = _regress_offset(
        xsp2, yspss2, x_min, x_max, near_field_limit, xsp.shape[0])
    offset2, std_o2, xl2, xr2 = _regress_offset(
        xsp2, yspss2, x_min, x_max, far_field_limit, xsp.shape[0])
    offset3, std_o3, xl3, xr3 = _regress_offset(
        xsp2, yspss2, x_min2, x_max2, near_field_limit, xsp.shape[0])

    A_fit = mu_fit = sigma_fit = alpha_fit = b_fit = 0.
    r_squared = 0.
    # entries 0-4 (trace coords) are filled by the caller
    result = np.array([
        0., 0., 0., 0., 0.,
        offset, std_o, xl[0], xl[-1], xr[0], xr[-1],
        offset2, std_o2, xl2[0], xl2[-1], xr2[0], xr2[-1],
        offset3, std_o3, xl3[0], xl3[-1], xr3[0], xr3[-1],
        fzw * xres, x_min * xres, x_max * xres, threshold,
        fzw2 * xres, x_min2 * xres, x_max2 * xres, threshold2,
        np.nanmax(ysp), np.nanmax(yspd), x_shift, r2,
        A_fit, mu_fit, sigma_fit, alpha_fit, b_fit, r_squared,
    ], dtype=float)
    return result, shifted


def _attach_analysis(p, result):
    '''Copy the per-profile analysis results onto a Profile as attributes.
    (Indices follow the fault_profile8 result_store layout: 5 trace-coord
    slots, then 3x6 offset entries, then the FZW/threshold/peak block.)'''
    p.offset_near = result[5]; p.offset_near_std = result[6]
    p.offset_far = result[11]; p.offset_far_std = result[12]
    p.offset_near_dil = result[17]; p.offset_near_dil_std = result[18]
    p.fzw = result[23]
    p.x_min = result[24]; p.x_max = result[25]
    p.strain_threshold = result[26]   # FZW edge threshold (stdthr * background std)
    p.fzw_dilatation = result[27]
    p.peak_shear = result[31]
    p.peak_dilatation = result[32]


def _threshold_width(x_fine, y_fine, threshold):
    '''Distance from x=0 to the first threshold down-crossing on each side.'''
    try:
        left = np.where(x_fine <= 0)
        xl = (-x_fine[left])[::-1]; yl = (y_fine[left])[::-1]
        below = np.where(yl < threshold)[0]
        x_min = -xl[below[0]]
    except IndexError:
        x_min = np.nan
    try:
        right = np.where(x_fine >= 0)
        xr = (x_fine[right]); yr = (y_fine[right])
        below = np.where(yr < threshold)[0]
        x_max = xr[below[0]]
        fzw = x_max - x_min
    except IndexError:
        x_max = np.nan; fzw = np.nan
    return x_min, x_max, fzw


def _regress_offset(xsp2, yspss2, x_min, x_max, limit, npred):
    '''Robust (RANSAC) linear fits to fault-parallel displacement in windows
    just outside the fault zone on each side; offset = right(0) - left(0).'''
    inner, outer = limit
    # right
    try:
        idx = np.where((xsp2 >= x_max + inner) & (xsp2 <= x_max + outer))
        x_right = xsp2[idx]; y_right = yspss2[idx]
        rt = np.std(y_right) * 3
        ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=rt)
        ransac.fit(x_right.reshape(-1, 1), y_right)
        xr_pred = np.linspace(0, x_right.max(), npred).reshape(-1, 1)
        yr_pred = ransac.predict(xr_pred)
        stdev_r = np.std(y_right - ransac.predict(x_right.reshape(-1, 1)))
    except (ValueError, IndexError):
        yr_pred = np.array([0., 0.]); stdev_r = 0.; x_right = np.array([0.])
    # left
    try:
        idx = np.where((xsp2 <= x_min - inner) & (xsp2 >= x_min - outer))
        x_left = xsp2[idx]; y_left = yspss2[idx]
        rt = np.std(y_left) * 3
        ransac2 = RANSACRegressor(estimator=LinearRegression(), residual_threshold=rt)
        ransac2.fit(x_left.reshape(-1, 1), y_left)
        xl_pred = np.linspace(x_left.min(), 0, npred).reshape(-1, 1)
        yl_pred = ransac2.predict(xl_pred)
        stdev_l = np.std(y_left - ransac2.predict(x_left.reshape(-1, 1)))
    except (ValueError, IndexError):
        yl_pred = np.array([0., 0.]); stdev_l = 0.; x_left = np.array([0.])

    offset = yr_pred[0] - yl_pred[-1]
    combined_std = np.sqrt(stdev_r**2 + stdev_l**2)
    return offset, combined_std, np.atleast_1d(x_left), np.atleast_1d(x_right)


# ---------------------------------------------------------------------------
#  Numerical helpers copied verbatim from `fault_profile8.py` (with only minor
#  signature tweaks where a function referenced a module-global there).
# ---------------------------------------------------------------------------
def smoothed_boxcar(plen, support_width, gaussian_sigma):
    '''Boxcar of width support_width centred at index plen, Gaussian-tapered.'''
    xx = np.arange(2 * plen + 1)
    box = np.zeros_like(xx, dtype=float)
    half_width = support_width // 2
    box[(xx >= plen - half_width) & (xx <= plen + half_width)] = 1.0
    smooth_box = gaussian_filter1d(box, sigma=gaussian_sigma)
    smooth_box /= np.max(smooth_box)
    return smooth_box


def replace_outliers_robust(vec, window=5, threshold=5.0):
    vec = vec.copy()
    half = window // 2
    n = len(vec)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_vals = np.delete(vec[start:end], i - start)   # exclude centre
        window_vals = window_vals[~np.isnan(window_vals)]
        if len(window_vals) == 0 or np.isnan(vec[i]):
            continue
        local_mean = np.nanmean(window_vals)
        if abs(vec[i] - local_mean) > threshold:
            vec[i] = local_mean
    return vec


def remove_marginal_outliers(tmp1, plen, buffer=10, error_quantile=0.8):
    '''Mask outliers in a 2D strain window, protecting the central column band
    (half-width ``buffer`` around column ``plen``).'''
    tmp1_tmp = tmp1 * 1.
    tmp1_tmp[:, plen - buffer:plen + buffer] = np.nan
    tmp1_stack = np.tile(np.nanstd(tmp1_tmp, axis=0), (tmp1.shape[0], 1))
    tmp1_stack_diff_mean = np.nanquantile(np.abs(tmp1 - tmp1_stack), error_quantile)
    tmp1_stack_mask = (tmp1 - tmp1_stack) < tmp1_stack_diff_mean
    tmp1_stack_mask[:, plen - buffer:plen + buffer] = 1.
    tmp1_masked = np.where(tmp1_stack_mask, tmp1, np.nan)
    return tmp1_masked


def bearing_to_xy(displacement, bearing_deg):
    bearing_rad = np.deg2rad(bearing_deg)
    dx = displacement * np.sin(bearing_rad)  # east
    dy = displacement * np.cos(bearing_rad)  # north
    return dx, -dy  # negate dy because positive is down for np arrays


def finite_strain(ew, ns, xres, yres, s=0, k=1, component=('exy', 'dilatation')):
    '''Finite (Green-Cauchy) strain from EW/NS displacement fields
    (east-positive, north-positive). Mutates its inputs (NaN handling), so pass
    copies. ``s`` (strike bearing, deg) rotates the tensor into fault-parallel /
    fault-normal axes; ``k`` is a Gaussian pre-smoothing sigma (1 = none).
    Returns (ny, nx, len(component)).'''
    no_components = len(component)
    xres2 = xres
    yres2 = yres
    step = 1  # sliding window step (if step < k, we are oversampling)

    # ew
    ew[np.isnan(ew)] = 0
    if k != 1 and k != 0:
        ewsm = gaussian_filter(ew, sigma=k, mode='wrap')
    else:
        ewsm = ew
    ew2 = ewsm[0:-1:step, 0:-1:step]
    ew2[ew2 == 0] = np.nan
    # ns
    ns[np.isnan(ns)] = 0
    if k != 1 and k != 0:
        nssm = gaussian_filter(ns, sigma=k, mode='wrap')
    else:
        nssm = ns
    ns2 = nssm[0:-1:step, 0:-1:step]
    ns2[ns2 == 0] = np.nan

    # displacement gradient tensor
    dudy, dudx = np.gradient(ew2, yres2, xres2, edge_order=2)
    dvdy, dvdx = np.gradient(ns2, yres2, xres2, edge_order=2)

    F = np.array([[dudx, dudy], [dvdx, dvdy]])

    # Finite strain (Green-Cauchy)
    E11 = 0.5 * (2 * dudx + dudx**2 + dvdx**2)
    E12 = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
    E21 = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
    E22 = 0.5 * (2 * dvdy + dudy**2 + dvdy**2)

    E = np.array([[E11, E12],
                  [E21, E22]])

    # remove NaNs
    msk = np.isnan(E)
    E[msk] = 0
    F[msk] = 0

    # rotate basis from ew/ns to fault_parallel-fault_normal
    if s != 0:
        sr = np.deg2rad(90 - s)  # angle from East to fault strike
        R = np.array([[np.cos(sr), -np.sin(sr)],
                      [np.sin(sr),  np.cos(sr)]])  # CCW rotation matrix
        E = np.moveaxis(E, [0, 1], [-2, -1])
        Er = np.einsum('ab,...bc,cd->...ad', R, E, R.T)
        mask = np.isnan(E).any(axis=(0, 1))
        Er[..., mask] = np.nan
        Er = np.moveaxis(Er, [-2, -1], [0, 1])
        E = Er

    strain_full_out = np.zeros((dudy.shape[0], dudy.shape[1], no_components))
    ff = -1
    for f in component:
        ff += 1
        if f == 'dudy':
            strain_out = dudy
        if f == 'dudx':
            strain_out = dudx
        if f == 'dvdy':
            strain_out = dvdy
        if f == 'dvdx':
            strain_out = dvdx
        if f == 'vorticity':
            strain_out = np.rad2deg(0.5 * (F[0, 1] - F[1, 0]))
        if f == 'exx':
            strain_out = E[0, 0]
        if f == 'eyy':
            strain_out = E[1, 1]
        if f == 'exy':
            strain_out = (E[0, 1] + E[1, 0]) / 2
        if f == 'dilatation':
            strain_out = np.trace(E[:, :])
        if f == 'maxShear':
            strain_out = 0.5 * ((E[0, 0] - E[1, 1])**2 + 4 * E[0, 1]**2)**0.5
        if f == 'compression':
            strain_out = (E[0, 0] + E[1, 1]) / 2

        strain_full_out[:, :, ff] = strain_out

    return strain_full_out
