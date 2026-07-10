'''Fault-perpendicular profile generation, quick swathe evaluation, and an
interactive keep/reject picker.

Meant for a first pass over noisy near-field data: generate many quick profiles
along a fault trace, look at each one by eye, and keep only those clean enough
to invert. The kept profiles are typically re-evaluated later with the accurate
(fault-aligned) profiling.
'''
import pickle
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
from shapely.geometry import LineString

from .Profile import Profile


def profiles_along_trace(trace, n_profiles=300, half_length=1500.,
                         strike_span=250., end_margin=200.):
    '''Fault-perpendicular Profiles at points evenly spaced along each linestring
    of a fault-trace GeoDataFrame. n_profiles is either a total (int, allocated
    by strand length) or a per-strand list. Local strike is taken from the chord
    s +/- strike_span/2, which smooths over small-scale trace wiggles.

    Each Profile gets metadata: fault_id, x_along_fault (m), strike (bearing
    from N, deg), fault_utm (easting, northing of the trace point).
    '''
    lengths = np.array([g.length for g in trace.geometry])
    if np.ndim(n_profiles) == 0:
        counts = np.maximum(1, np.round(n_profiles * lengths / lengths.sum())).astype(int)
    else:
        counts = np.asarray(n_profiles, dtype=int)

    profiles = []
    for fid, (ls, n) in enumerate(zip(trace.geometry, counts)):
        for s in np.linspace(end_margin, ls.length - end_margin, n):
            c = np.asarray(ls.interpolate(s).coords[0])[:2]
            a = np.asarray(ls.interpolate(max(s - strike_span / 2., 0.)).coords[0])[:2]
            b = np.asarray(ls.interpolate(min(s + strike_span / 2., ls.length)).coords[0])[:2]
            u = (b - a) / np.linalg.norm(b - a)
            nrm = np.array([-u[1], u[0]])
            line = LineString([c - half_length * nrm, c + half_length * nrm])
            gdf = gpd.GeoDataFrame({'x_along_fault': [s]}, geometry=[line], crs=trace.crs)
            p = Profile(trace=gdf, fault_x=half_length)
            p.fault_id = fid
            p.x_along_fault = s
            p.strike = np.degrees(np.arctan2(u[0], u[1])) % 360.
            p.fault_utm = tuple(c)
            profiles.append(p)
    return profiles


def evaluate_profiles_quick(opt, profiles, swathe_half_width=100., n_bins=300,
                            min_valid=0.25, min_side=100., verbose=True):
    '''Swathe-evaluate each profile on an OpticalData scene and bin-average it.

    Displacements are reordered to [fault-parallel, fault-normal] (the
    fault-aligned convention): for a fault-perpendicular profile,
    evaluate_profile's along-profile row is the fault-NORMAL component.
    Where the profile runs off the data (nodata zeros), it is cropped to the
    well-populated span. Profiles are dropped if they have fewer than min_valid
    finite bins, or lack min_side metres of data on either side of the fault.
    '''
    out = []
    for i, p in enumerate(profiles):
        if verbose and i % 20 == 0:
            print(f'  evaluating profile {i}/{len(profiles)}')
        p = opt.evaluate_profile(p, swathe_half_width=swathe_half_width,
                                 clear_zero=True)
        if p.xs.size < n_bins:
            continue
        _crop_to_data(p)
        if p.xs.size < n_bins or p.xs.min() > -min_side or p.xs.max() < min_side:
            continue
        p = p.bin_average(n_bins=n_bins)
        p.displacements = p.displacements[::-1]
        if np.isfinite(p.displacements[0]).mean() < min_valid:
            continue
        out.append(p)
    if verbose:
        print(f'  kept {len(out)}/{len(profiles)} profiles with enough data')
    return out


def plot_picked_profiles(profiles, background=None, bg_extent=None, trace=None,
                         path=None):
    '''Map of picked profiles over the optical scene. Each profile is drawn over
    its actual data span (xs range), so extent trims are visible.'''
    fig, ax = plt.subplots(figsize=(9, 9), layout='constrained')
    if background is not None:
        vmax = np.nanpercentile(np.abs(background), 98)
        ax.imshow(background, extent=bg_extent, origin='upper', cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax, interpolation='nearest')
    if trace is not None:
        for g in trace.geometry:
            xy = np.asarray(g.coords)
            ax.plot(xy[:, 0], xy[:, 1], 'k-', lw=0.6)
    for p in profiles:
        c = np.asarray(p.linestring.coords)
        u = (c[-1] - c[0]) / np.hypot(*(c[-1] - c[0]))
        a = c[0] + (p.xs.min() + p.fault_x) * u
        b = c[0] + (p.xs.max() + p.fault_x) * u
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='limegreen', lw=1.)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{len(profiles)} picked profiles')
    if path is not None:
        fig.savefig(path, dpi=250)
    return fig, ax


def _crop_to_data(p, n_check=50, min_density=0.1):
    '''Trim a profile's ends to the well-populated span: a few stray valid
    pixels in the nodata fringe would otherwise stretch the extent.'''
    counts, edges = np.histogram(p.xs, bins=n_check)
    good = np.flatnonzero(counts > min_density * np.median(counts[counts > 0]))
    if good.size == 0:
        p.xs = np.array([])
        p.displacements = np.zeros((2, 0))
        return
    lo, hi = edges[good[0]], edges[good[-1] + 1]
    u = (p.xs >= lo) & (p.xs <= hi)
    p.xs = p.xs[u]
    p.displacements = p.displacements[:, u]


class ProfilePicker:
    '''Interactive keep/reject picking of evaluated profiles.

    Shows each profile in turn (with a minimap locating it on the optical
    scene); keep or reject with the buttons or the y/n keys, step back with
    "back"/left-arrow. Drag on the profile axes to keep only that x-extent
    (stored as p.keep_extent; the kept profile is trimmed to it on save).
    "finish & save" pickles the kept profiles to save_path and closes.

    Args:
        profiles: list of evaluated Profile instances
        save_path: output .pickle for the kept profiles
        background, bg_extent: minimap image (2D array) + [x0, x1, y0, y1]
        trace: fault-trace GeoDataFrame drawn on the minimap
    '''

    def __init__(self, profiles, save_path, background=None, bg_extent=None,
                 trace=None):
        self.profiles = profiles
        self.save_path = save_path
        self.keep = [None] * len(profiles)
        self.extents = [None] * len(profiles)
        self.i = 0
        self.picked = []

        self.fig = plt.figure(figsize=(13, 6.5))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2.4, 1],
                                   left=0.07, right=0.98, top=0.9, bottom=0.2)
        self.ax = self.fig.add_subplot(gs[0])
        self.axmap = self.fig.add_subplot(gs[1])

        if background is not None:
            vmax = np.nanpercentile(np.abs(background), 98)
            self.axmap.imshow(background, extent=bg_extent, origin='upper',
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              interpolation='nearest')
        if trace is not None:
            for g in trace.geometry:
                xy = np.asarray(g.coords)
                self.axmap.plot(xy[:, 0], xy[:, 1], 'k-', lw=0.6)
        self.axmap.set_aspect('equal')
        self.axmap.set_xticks([]); self.axmap.set_yticks([])
        self.profline, = self.axmap.plot([], [], 'r-', lw=2)
        self.profdot, = self.axmap.plot([], [], 'ro', ms=4)

        def _button(x, w, label, cb):
            bax = self.fig.add_axes([x, 0.04, w, 0.06])
            b = Button(bax, label)
            b.on_clicked(cb)
            return b
        self._buttons = [
            _button(0.07, 0.10, 'keep (y)', self._on_keep),
            _button(0.19, 0.10, 'reject (n)', self._on_reject),
            _button(0.31, 0.10, 'back', self._on_back),
            _button(0.43, 0.12, 'clear extent', self._on_clear_extent),
            _button(0.60, 0.14, 'finish & save', self._on_finish),
        ]
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._show()

    def run(self):
        plt.show()
        return self.picked

    # ---- display ----
    def _show(self):
        p = self.profiles[self.i]
        self.ax.clear()
        self.ax.plot(p.xs, p.displacements[0], '.-', ms=3, lw=0.7,
                     color='crimson', label='fault-parallel')
        self.ax.plot(p.xs, p.displacements[1], '.-', ms=3, lw=0.7,
                     color='steelblue', alpha=0.7, label='fault-normal')
        self.ax.axvline(0., color='lightgray', ls='--', zorder=0)
        if self.extents[self.i] is not None:
            self.ax.axvspan(*self.extents[self.i], color='tab:green', alpha=0.15)
        self.ax.set_xlabel('Distance from fault (m)')
        self.ax.set_ylabel('Displacement (m)')
        self.ax.legend(loc='upper right', fontsize=8)

        status = {None: 'undecided', True: 'KEEP', False: 'reject'}[self.keep[self.i]]
        n_kept = sum(k is True for k in self.keep)
        n_done = sum(k is not None for k in self.keep)
        self.ax.set_title(
            f'{self.i + 1}/{len(self.profiles)}   fault {p.fault_id}   '
            f'along-strike {p.x_along_fault / 1000.:.1f} km   [{status}]   '
            f'(decided {n_done}, kept {n_kept})',
            color={'KEEP': 'green', 'reject': 'firebrick'}.get(status, 'black'))

        xy = np.asarray(p.linestring.coords)
        self.profline.set_data(xy[:, 0], xy[:, 1])
        self.profdot.set_data([p.fault_utm[0]], [p.fault_utm[1]])

        # SpanSelector must be rebuilt after ax.clear()
        self.span = SpanSelector(self.ax, self._on_select, 'horizontal',
                                 useblit=True, interactive=True,
                                 props=dict(alpha=0.25, facecolor='tab:green'))
        self.fig.canvas.draw_idle()

    # ---- callbacks ----
    def _on_select(self, xmin, xmax):
        self.extents[self.i] = None if (xmax - xmin) < 1. else (xmin, xmax)

    def _on_clear_extent(self, *_):
        self.extents[self.i] = None
        self._show()

    def _decide(self, keep):
        self.keep[self.i] = keep
        if self.i < len(self.profiles) - 1:
            self.i += 1
            self._show()
        else:
            self._show()
            self.ax.set_title('Last profile — finish & save when ready',
                              color='navy')
            self.fig.canvas.draw_idle()

    def _on_keep(self, *_):
        self._decide(True)

    def _on_reject(self, *_):
        self._decide(False)

    def _on_back(self, *_):
        self.i = max(0, self.i - 1)
        self._show()

    def _on_key(self, event):
        if event.key == 'y':
            self._on_keep()
        elif event.key == 'n':
            self._on_reject()
        elif event.key == 'left':
            self._on_back()

    def _on_finish(self, *_):
        self.picked = []
        for p, keep, ext in zip(self.profiles, self.keep, self.extents):
            if keep is not True:
                continue
            p.keep_extent = ext
            if ext is not None:
                u = (p.xs >= ext[0]) & (p.xs <= ext[1])
                p.xs = p.xs[u]
                p.displacements = p.displacements[:, u]
            self.picked.append(p)
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.picked, f)
        print(f'Saved {len(self.picked)} picked profiles to {self.save_path}')
        plt.close(self.fig)
