"""Load and visualise AlTar Bayesian slip-inversion output.

An :class:`AltarOutput` wraps one AlTar run directory (the one holding the
``step_*.h5`` / ``step_final.h5`` files) together with the ``FaultTriangles``
mesh it was run on and, optionally, the cached ``InversionManager`` that built
the AlTar inputs (which carries the dataset coordinates, ``G`` and ``d``).

The AlTar model vector is the ParameterSets concatenated in the same order as
the ``G`` columns written by ``InversionManager.save_to_hdf5`` -- i.e.
``[*ss_keys, *ds_keys, ramp_key]``.  For the Ridgecrest doublet that is
``[strikeslipmain, strikeslipsecond, dipslip, ramp]``, matching
``[fault_ss (merged patch order), fault_ds, ramp]``.

Everything here is fault-agnostic; case-specific paths and choices live in the
calling script.
"""
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
import cmcrameri.cm as cmc


MU = 30e9          # shear modulus for moment (Pa)
KDE_MAX_SAMPLES = 4000
KDE_GRID = 256

# White -> red -> dark slip palette (the group's usual coseismic-slip colormap).
_COLORSCO = [(250, 250, 250), (255, 247, 236), (254, 232, 200), (253, 212, 158),
             (253, 187, 132), (252, 141, 89), (239, 101, 72), (215, 48, 31),
             (179, 0, 0), (127, 0, 0)]
SLIP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cptslip", [(r / 255., g / 255., b / 255.) for r, g, b in _COLORSCO], N=256)

# Bivariate slip/uncertainty palette, base->top, left->right (from utils.py).
_BIVCOLORS = [(208, 208, 208),
              (232, 232, 232), (164, 128, 128),
              (250, 250, 250), (237, 204, 187), (214, 137, 127), (149, 75, 75),
              (254, 248, 241), (254, 227, 190), (253, 173, 119), (248, 130, 84),
              (233, 87, 61), (210, 41, 27), (173, 0, 0), (127, 0, 0)]
BIVCOLORS_RGBA = [(r / 255., g / 255., b / 255.) for r, g, b in _BIVCOLORS]

# Low-sigma (confident) row of the palette == the 1-D slip ramp used by the
# *programmatic* bivariate schemes (near-white cream -> red -> dark red).
_BIV_SLIP = [(254, 248, 241), (254, 227, 190), (253, 173, 119), (248, 130, 84),
             (233, 87, 61), (210, 41, 27), (173, 0, 0), (127, 0, 0)]
BIV_SLIP_RGB = [(r / 255., g / 255., b / 255.) for r, g, b in _BIV_SLIP]

# Default slip ramp for the bivariate scheme (near-white cream -> dark red).
BIV_SLIP_CMAP = mcolors.LinearSegmentedColormap.from_list("bivslip", BIV_SLIP_RGB,
                                                          N=256)

# Defaults for the programmatic bivariate scheme.  ``grey`` is the colour a
# maximally-uncertain patch fades to; ``cmap`` is the zero-sigma slip ramp;
# ``uncertainty`` selects the secondary axis (std vs coefficient of variation,
# sigma/slip); ``sigma_floor`` keeps patches saturated until the uncertainty
# passes that fraction; ``wedge`` picks a discrete or continuous colour key.
DEFAULT_BIV_STYLE = dict(scheme="continuous", grey=(0.35, 0.35, 0.35),
                         edgecolor="0.55", match_spines=True, uncertainty="sigma",
                         sigma_floor=0.0, cmap=None, wedge="discrete")


def _resolve_cmap(cmap):
    """None -> the house bivariate slip ramp; a name -> that registered cmap;
    otherwise assume it is already a Colormap."""
    if cmap is None:
        return BIV_SLIP_CMAP
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def bivariate_rgb(slip, sigma, valmax, sigmamax, grey=(0.35, 0.35, 0.35),
                  cmap=None, floor=0.0):
    """Continuous bivariate colour: the slip ramp ``cmap`` (default cream ->
    dark red) faded toward ``grey`` in proportion to the (normalised) uncertainty
    ``sigma``, so sigma=0 is the fully saturated ``cmap`` and sigma=sigmamax is
    fully grey.  ``floor`` holds the colour saturated until sigma passes that
    fraction of sigmamax, then ramps to grey.  Returns an (N, 3) array."""
    smap = _resolve_cmap(cmap)
    x = np.clip(np.asarray(slip, float) / valmax, 0., 1.)
    y = np.clip(np.asarray(sigma, float) / sigmamax, 0., 1.)   # starts at zero
    if floor > 0.:
        y = np.clip((y - floor) / (1. - floor), 0., 1.)
    base = smap(x)[..., :3]
    grey = np.asarray(grey, float)
    return (1. - y)[..., None] * base + y[..., None] * grey


def _round_up_nice(v):
    e = 10. ** np.floor(np.log10(max(v, 1e-9)))
    return float(np.ceil(v / e) * e)


def _kde_mode(samples, seed=0):
    """Per-column posterior mode from a Gaussian KDE on each 1-D marginal.

    ``samples`` is (n_samples, n_params).  The KDE is fit on a random subsample
    (the mode location converges long before KDE_MAX_SAMPLES) and evaluated on a
    KDE_GRID-point grid spanning the sample range.
    """
    rng = np.random.default_rng(seed)
    n, p = samples.shape
    idx = rng.choice(n, size=min(n, KDE_MAX_SAMPLES), replace=False)
    out = np.empty(p)
    for j in range(p):
        x = samples[idx, j]
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            out[j] = lo
            continue
        grid = np.linspace(lo, hi, KDE_GRID)
        out[j] = grid[np.argmax(gaussian_kde(x)(grid))]
    return out


def _kde_curve(x):
    """(grid, density) for a 1-D sample, or None if degenerate."""
    x = np.asarray(x)
    if x.max() - x.min() < 1e-12:
        return None
    grid = np.linspace(x.min(), x.max(), KDE_GRID)
    return grid, gaussian_kde(x[:KDE_MAX_SAMPLES])(grid)


class AltarOutput:

    def __init__(self, output_dir, fault, inv=None,
                 G=None, d=None, coords=None, dataset_slices=None,
                 ss_keys=("strikeslipmain", "strikeslipsecond"),
                 ds_keys=("dipslip",), ramp_key="ramp",
                 figs_dir=None, seed=0):
        self.output_dir = str(output_dir)
        self.fault = fault
        self.inv = inv
        # The assembled system + dataset coords come either from a cached
        # InversionManager (fresh runs) or explicit archived artifacts.
        if inv is not None:
            self.G, self.d = inv.G, inv.d
            self._dslices = inv._dataset_slices
        else:
            self.G, self.d = G, d
            self._dslices = dataset_slices
        self._coords_arg = coords
        self.ss_keys = list(ss_keys)
        self.ds_keys = list(ds_keys)
        self.ramp_key = ramp_key
        self.seed = seed
        self.figs_dir = str(figs_dir) if figs_dir else os.path.join(self.output_dir, "figs")
        os.makedirs(self.figs_dir, exist_ok=True)

        self.ps, self.beta = self.load_step(os.path.join(self.output_dir, "step_final.h5"))
        self.samples = self.model_matrix(self.ps)
        self._sections = self._build_sections()
        self._summ = None
        self._m_mode = None
        print(f"[altar] {self.samples.shape[0]} samples x {self.samples.shape[1]} "
              f"params (final beta = {self.beta:.4g})")

    # ---- posterior loading ------------------------------------------------ #
    @staticmethod
    def load_step(path):
        """Read one AlTar step file -> (ParameterSets dict, beta)."""
        with h5py.File(path, "r") as fh:
            ps = {k: fh["ParameterSets"][k][:] for k in fh["ParameterSets"]}
            beta = float(fh["Annealer"]["beta"][()]) if "Annealer" in fh else np.nan
        return ps, beta

    def model_matrix(self, ps):
        """(n_samples, n_params) ordered like the G columns."""
        return np.hstack([ps[k] for k in self.ss_keys + self.ds_keys + [self.ramp_key]])

    @property
    def summ(self):
        """Posterior summaries per patch: modes and stds of SS, DS, |total|."""
        if self._summ is None:
            print("[altar] computing per-patch posterior modes (KDE)...")
            ss = np.hstack([self.ps[k] for k in self.ss_keys])
            ds = np.hstack([self.ps[k] for k in self.ds_keys])
            tot = np.hypot(ss, ds)
            assert ss.shape[1] == self.fault.n_patches, (ss.shape, self.fault.n_patches)
            self._summ = dict(
                ss_mode=_kde_mode(ss, self.seed), ds_mode=_kde_mode(ds, self.seed),
                tot_mode=_kde_mode(tot, self.seed),
                ss_std=ss.std(axis=0), ds_std=ds.std(axis=0), tot_std=tot.std(axis=0),
                ss=ss, ds=ds, tot=tot)
        return self._summ

    @property
    def m_mode(self):
        """Full posterior-mode model vector, in G-column order."""
        if self._m_mode is None:
            self._m_mode = np.concatenate(
                [self.summ["ss_mode"], self.summ["ds_mode"],
                 _kde_mode(self.ps[self.ramp_key], self.seed)])
        return self._m_mode

    def _path(self, path, default):
        return path if path else os.path.join(self.figs_dir, default)

    def _save(self, fig, path):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {os.path.basename(path)}")

    def check_ordering(self):
        """Verify the ParameterSet ordering against G by checking the
        posterior-mean prediction beats the data RMS."""
        assert self.G is not None, "need G and d to check ordering"
        d, G = self.d, self.G
        rms_d = float(np.sqrt(np.mean(d ** 2)))
        rms_r = float(np.sqrt(np.mean((d - G @ self.samples.mean(axis=0)) ** 2)))
        print(f"[altar] posterior-mean residual RMS {rms_r:.4f} m "
              f"(data RMS {rms_d:.4f} m)")
        if rms_r > 0.8 * rms_d:
            raise RuntimeError("posterior-mean prediction barely beats the data -- "
                               "parameter ordering vs G is probably wrong")

    # --------------------------------------------------------------------- #
    #  Fault geometry -> per-strand 2D sections
    # --------------------------------------------------------------------- #
    def _build_sections(self):
        """Per-strand projection onto the trace-defined vertical section (km)."""
        trace = self.fault.get_surface_trace_xy()
        lines = [np.asarray(geom.coords, dtype=float) for geom in trace.geometry]
        tri_xyz = self.fault.vertices[self.fault.triangles]
        sections = []
        for k in range(self.fault.n_subfaults):
            mask = self.fault.fault_ids == k
            line = lines[k]
            origin, strike = line[0], line[-1] - line[0]
            strike = strike / np.linalg.norm(strike)
            v = tri_xyz[mask]
            s = (v[:, :, :2] - origin) @ strike
            depth = -v[:, :, 2]
            sections.append(dict(ids=np.flatnonzero(mask),
                                 polys=np.stack([s, depth], axis=-1) / 1e3,
                                 depth=depth.mean(axis=1) / 1e3,
                                 width=float(s.max() - s.min()) / 1e3,
                                 name=str(self.fault.subfault_names[k])))
        return sections

    def _draw_sections(self, axes, facecolors=None, values=None, cmap=None,
                       norm=None, edgecolor="0.6", linewidth=0.15):
        """Draw each strand's triangles into its axes; returns the PolyCollections."""
        colls = []
        for ax, sec in zip(axes, self._sections):
            order = np.argsort(-sec["depth"])
            polys = list(sec["polys"][order])
            gids = sec["ids"][order]
            if facecolors is not None:
                coll = PolyCollection(polys, facecolors=[facecolors[i] for i in gids],
                                      edgecolors=edgecolor, linewidths=linewidth)
            else:
                coll = PolyCollection(polys, array=np.asarray(values)[gids],
                                      cmap=cmap, norm=norm, edgecolors=edgecolor,
                                      linewidths=linewidth)
            ax.add_collection(coll)
            ax.autoscale_view()
            ax.set_aspect("equal")
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            ax.set_xlabel("Along strike (km)")
            ax.set_title(sec["name"], fontsize=10)
            colls.append(coll)
        axes[0].set_ylabel("Depth (km)")
        return colls

    def _section_axes(self, fig, subplot_spec=None):
        """A row of per-strand axes with widths proportional to strand length."""
        widths = [sec["width"] for sec in self._sections]
        n = len(self._sections)
        if subplot_spec is None:
            gs = fig.add_gridspec(1, n, width_ratios=widths)
        else:
            gs = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=subplot_spec,
                                                  width_ratios=widths)
        axes = [fig.add_subplot(gs[0, 0])]
        for i in range(1, n):
            axes.append(fig.add_subplot(gs[0, i], sharey=axes[0]))
            axes[-1].tick_params(labelleft=False)
        return axes

    # --------------------------------------------------------------------- #
    #  2D / 3D slip coloured by posterior mode
    # --------------------------------------------------------------------- #
    def plot_slip_2d(self, component="total", path=None, cmap=None,
                     vmin=None, vmax=None):
        if component == "total":
            vals, label = self.summ["tot_mode"], "Total slip (m)"
            cmap = cmap or SLIP_CMAP
            vmin = 0. if vmin is None else vmin
            vmax = float(vals.max()) if vmax is None else vmax
            default = "slip2d_total.png"
        elif component == "strikeslip":
            vals, label = np.abs(self.summ["ss_mode"]), "|Strike-slip| (m)"
            cmap = cmap or SLIP_CMAP
            vmin = 0. if vmin is None else vmin
            default = "slip2d_strikeslip.png"
        elif component == "dipslip":
            vals, label = self.summ["ds_mode"], "Dip-slip (m)"
            cmap = cmap or cmc.vik
            if vmin is None and vmax is None:
                lim = float(np.abs(vals).max()) or 1e-3
                vmin, vmax = -lim, lim
            default = "slip2d_dipslip.png"
        else:
            raise ValueError(component)
        fig, _ = self.fault.plot_slip_2d(slip=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                                         colorbar_label=label)
        fig.suptitle("Posterior mode (per-patch KDE maximum)", fontsize=11)
        self._save(fig, self._path(path, default))

    def plot_slip_3d(self, path=None):
        vals = self.summ["tot_mode"]
        self.fault.slips = vals
        fig, ax = self.fault.plot_fault3d(color_by="slip", cmap=SLIP_CMAP, vmin=0.,
                                          vmax=float(vals.max()), edgecolor="0.4",
                                          linewidth=0.15)
        ax.set_title("Total slip, posterior mode")
        self.fault.slips = np.zeros(self.fault.n_patches)
        self._save(fig, self._path(path, "slip3d_total.png"))

    # --------------------------------------------------------------------- #
    #  Bivariate slip / uncertainty with the wedge colorbar
    # --------------------------------------------------------------------- #
    @staticmethod
    def _bivariate_colval(slip, sigma, valmax, sigmamax):
        """Map (slip, sigma) to the 15 bivariate categories (utils.ColValSup)."""
        sigma1, sigma2 = sigmamax / 2., 3. * sigmamax / 4.
        colval = np.zeros(np.shape(slip))
        colval[sigma >= sigmamax] = 0
        colval[(slip < valmax / 2.) & (sigma >= sigma2) & (sigma < sigmamax)] = 1
        colval[(slip >= valmax / 2.) & (sigma >= sigma2) & (sigma < sigmamax)] = 2
        cat = 2
        for s in np.arange(0, valmax, valmax / 4.):
            cat += 1
            colval[(sigma < sigma2) & (sigma >= sigma1)
                   & (slip >= s) & (slip < s + valmax / 4.)] = cat
        colval[(sigma < sigma2) & (sigma >= sigma1) & (slip >= valmax)] = 6
        cat = 6
        for s in np.arange(0, valmax, valmax / 8.):
            cat += 1
            colval[(sigma < sigma1) & (slip >= s) & (slip < s + valmax / 8.)] = cat
        colval[(sigma < sigma1) & (slip >= valmax)] = cat
        return colval

    @staticmethod
    def _draw_wedge_legend(ax, valmax, sigmamax, slip_label="Slip (m)",
                           sigma_label="Standard deviation (m)", palette=None):
        """The quarter 'colour wheel' bivariate legend (0..1 data coords, equal
        aspect).  Radius = slip (finer bins at low sigma), rings = sigma.  Stays
        discrete even when the fault faces are coloured continuously."""
        palette = palette or BIVCOLORS_RGBA
        fmt = lambda v: f"{v:.2g}"
        ax.set_axis_off()
        ax.set_aspect("equal")
        center, L = (0.5, 0.02), 0.70
        wdgs = []
        kw = dict(ec="white", lw=0.3)
        wdgs.append(mpatches.Wedge(center, L / 4, 60, 120, width=None,
                                   fc=palette[0], **kw))
        wdgs.append(mpatches.Wedge(center, 2 * L / 4, 90, 120, width=L / 4,
                                   fc=palette[1], **kw))
        wdgs.append(mpatches.Wedge(center, 2 * L / 4, 60, 90, width=L / 4,
                                   fc=palette[2], **kw))
        for i, (a0, a1) in enumerate([(60, 75), (75, 90), (90, 105), (105, 120)]):
            wdgs.append(mpatches.Wedge(center, 3 * L / 4, a0, a1, width=L / 4,
                                       fc=palette[6 - i], **kw))
        angles = np.linspace(60, 120, 9)
        for i in range(8):
            wdgs.append(mpatches.Wedge(center, L, angles[i], angles[i + 1],
                                       width=L / 4, fc=palette[14 - i], **kw))
        for w in wdgs:
            ax.add_patch(w)
        coords = [wdgs[i].get_path().vertices[6] for i in [14, 12, 10, 8]]
        coords += [wdgs[i].get_path().vertices[0] for i in [7, 3, 2, 0]]
        labels = ([fmt(v) for v in np.arange(0, valmax, valmax / 4.)]
                  + [fmt(valmax), fmt(sigmamax / 2.), fmt(3 * sigmamax / 4.),
                     fmt(sigmamax)])
        rots = [30, 15, 0, -15, -30, 60, 60, 60]
        vas = ["center"] * 5 + ["top"] * 3
        offs = [[0, 0.05 * L]] * 5 + [[0.05 * L, 0]] * 3
        for (x, y), lab, rot, va, off in zip(coords, labels, rots, vas, offs):
            ax.text(x + off[0], y + off[1], lab, rotation=rot,
                    rotation_mode="anchor", ha="center", va=va, fontsize=8)
        ax.text(center[0], center[1] + 1.18 * L, slip_label, ha="center",
                va="center", fontsize=9)
        ax.text(center[0] + 0.42 * L, center[1] + 0.30 * L, sigma_label,
                rotation=60, ha="center", va="center", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    @staticmethod
    def _draw_wedge_legend_continuous(ax, valmax, sigmamax, style,
                                      slip_label="Slip (m)",
                                      sigma_label="Standard deviation (m)"):
        """Continuous version of the wedge key: a filled quarter sector coloured
        by the same bivariate map (angle = slip, radius = sigma).  Uses the
        discrete legend's radii/labels so the two are interchangeable."""
        fmt = lambda v: f"{v:.2g}"
        ax.set_axis_off()
        ax.set_aspect("equal")
        center, L = (0.5, 0.02), 0.70

        def sigma_of_r(r):
            # Inverse of the discrete ring radii: sigma=0 at r=L, sigmamax/2 at
            # 3L/4, sigmamax at L/4, saturating through the centre cap.
            r = np.asarray(r, float)
            s = np.where(r >= 3 * L / 4.,
                         sigmamax / 2. * (L - r) / (L / 4.),
                         sigmamax / 2. + sigmamax / 2. * (3 * L / 4. - r) / (L / 2.))
            return np.clip(s, 0., sigmamax)

        nr, nt = 80, 120
        r = np.linspace(0., L, nr)
        thd = np.linspace(120., 60., nt)                    # slip 0 -> valmax
        th = np.deg2rad(thd)
        R, TH = np.meshgrid(r, th, indexing="ij")
        X = center[0] + R * np.cos(TH)
        Y = center[1] + R * np.sin(TH)
        rc, tc = 0.5 * (r[:-1] + r[1:]), 0.5 * (thd[:-1] + thd[1:])
        RC, TC = np.meshgrid(rc, tc, indexing="ij")
        slip = valmax * (120. - TC) / 60.
        rgb = bivariate_rgb(slip.ravel(), sigma_of_r(RC).ravel(), valmax, sigmamax,
                            grey=style["grey"], cmap=style["cmap"],
                            floor=style["sigma_floor"])
        mesh = ax.pcolormesh(X, Y, np.zeros((nr - 1, nt - 1)), shading="flat")
        mesh.set_array(None)
        mesh.set_facecolor(rgb)
        mesh.set_edgecolor("face")
        ax.plot(center[0] + L * np.cos(th), center[1] + L * np.sin(th),
                color="0.5", lw=0.5)

        for f in (0., .25, .5, .75, 1.):                    # slip ticks (arc)
            a = np.deg2rad(120. - 60. * f)
            ax.text(center[0] + (L + 0.03) * np.cos(a),
                    center[1] + (L + 0.03) * np.sin(a), fmt(f * valmax),
                    rotation=np.degrees(a) - 90., rotation_mode="anchor",
                    ha="center", va="bottom", fontsize=8)
        for sig, rr in [(sigmamax / 2., 3 * L / 4.), (3 * sigmamax / 4., L / 2.),
                        (sigmamax, L / 4.)]:                # sigma ticks (radius)
            a = np.deg2rad(60.)
            ax.text(center[0] + rr * np.cos(a) + 0.03,
                    center[1] + rr * np.sin(a) - 0.015, fmt(sig), rotation=-30.,
                    rotation_mode="anchor", ha="left", va="top", fontsize=8)
        ax.text(center[0], center[1] + 1.18 * L, slip_label, ha="center",
                va="center", fontsize=9)
        ax.text(center[0] + 0.42 * L, center[1] + 0.30 * L, sigma_label,
                rotation=60, ha="center", va="center", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    @staticmethod
    def _biv_bin_centers(valmax, sigmamax):
        """Representative (slip, sigma) for each of the 15 ``_bivariate_colval``
        categories -- used to sample a discrete palette from the continuous map."""
        slip, sig = np.empty(15), np.empty(15)
        slip[0], sig[0] = 0., sigmamax                 # most uncertain -> grey
        slip[1], sig[1] = 0.25 * valmax, 0.875 * sigmamax
        slip[2], sig[2] = 0.75 * valmax, 0.875 * sigmamax
        for c, f in zip(range(3, 7), (0.125, 0.375, 0.625, 0.875)):
            slip[c], sig[c] = f * valmax, 0.625 * sigmamax
        for c, f in zip(range(7, 15), (np.arange(8) + 0.5) / 8.):
            slip[c], sig[c] = f * valmax, 0.25 * sigmamax
        return slip, sig

    def _biv_palette(self, valmax, sigmamax, style):
        """15-colour discrete palette (legend + discrete faces).  ``original``
        is the ported house palette; otherwise it is sampled from the continuous
        map at each category's representative (slip, sigma)."""
        if style["scheme"] == "original":
            return BIVCOLORS_RGBA
        slip, sig = self._biv_bin_centers(valmax, sigmamax)
        return list(bivariate_rgb(slip, sig, valmax, sigmamax, grey=style["grey"],
                                  cmap=style["cmap"], floor=style["sigma_floor"]))

    @staticmethod
    def _uncertainty_metric(slip, sigma, mode):
        """Secondary-axis value per patch: raw std, or coefficient of variation
        (sigma / slip) when ``mode == 'cv'`` -- low-slip patches then read as
        relatively uncertain regardless of their absolute std."""
        if mode == "cv":
            return np.abs(np.asarray(sigma, float)) / np.maximum(
                np.abs(np.asarray(slip, float)), 1e-3)
        return np.asarray(sigma, float)

    def _bivariate_scaffold(self, valmax, sigmamax, style):
        """Figure with per-strand section axes + wedge legend; returns collections."""
        widths = [sec["width"] for sec in self._sections]
        dmax = max(float(sec["polys"][:, :, 1].max()) for sec in self._sections)
        dmin = min(float(sec["polys"][:, :, 1].min()) for sec in self._sections)
        in_per_km, wedge_w = 0.15, 2.6
        fig = plt.figure(figsize=(in_per_km * sum(widths) + wedge_w + 1.0,
                                  in_per_km * dmax + 1.7), layout="constrained")
        gs = fig.add_gridspec(1, len(self._sections) + 1,
                              width_ratios=[in_per_km * w for w in widths] + [wedge_w])
        axes = [fig.add_subplot(gs[0, 0])]
        for i in range(1, len(self._sections)):
            axes.append(fig.add_subplot(gs[0, i], sharey=axes[0]))
            axes[-1].tick_params(labelleft=False)
        palette = self._biv_palette(valmax, sigmamax, style)
        base = [palette[0]] * self.fault.n_patches
        colls = self._draw_sections(axes, facecolors=base,
                                    edgecolor=style["edgecolor"], linewidth=0.25)
        # Tight data limits (no autoscale margin); shared y spans all strands.
        for ax, sec in zip(axes, self._sections):
            s = sec["polys"][:, :, 0]
            ax.set_xlim(float(s.min()), float(s.max()))
        axes[0].set_ylim(dmax, dmin)              # inverted axis -> (bottom, top)
        if style["match_spines"]:
            for ax in axes:
                for sp in ax.spines.values():
                    sp.set_edgecolor(style["edgecolor"])
        sigma_label = ("Coeff. of variation (sigma / slip)"
                       if style["uncertainty"] == "cv" else "Standard deviation (m)")
        legend_ax = fig.add_subplot(gs[0, -1])
        if style["wedge"] == "continuous":
            self._draw_wedge_legend_continuous(legend_ax, valmax, sigmamax, style,
                                               sigma_label=sigma_label)
        else:
            self._draw_wedge_legend(legend_ax, valmax, sigmamax,
                                    sigma_label=sigma_label, palette=palette)
        return fig, axes, colls

    def _set_bivariate_colors(self, colls, slip, sigma, valmax, sigmamax, style):
        slip = np.asarray(slip, float)
        u = self._uncertainty_metric(slip, sigma, style["uncertainty"])
        if style["scheme"] == "continuous":
            rgb = bivariate_rgb(slip, u, valmax, sigmamax, grey=style["grey"],
                                cmap=style["cmap"], floor=style["sigma_floor"])
        else:
            cmap = mcolors.ListedColormap(self._biv_palette(valmax, sigmamax, style))
            norm = mcolors.Normalize(0, 15)
            colval = self._bivariate_colval(slip, u, valmax, sigmamax)
            rgb = cmap(norm(colval))[..., :3]
        for coll, sec in zip(colls, self._sections):
            order = np.argsort(-sec["depth"])
            coll.set_facecolor(rgb[sec["ids"][order]])

    def _bivariate_limits(self, valmax, sigmamax):
        valmax = valmax or _round_up_nice(float(self.summ["tot_mode"].max()))
        sigmamax = sigmamax or _round_up_nice(float(self.summ["tot_std"].max()))
        return valmax, sigmamax

    @staticmethod
    def _biv_style(scheme, grey, edgecolor, match_spines, uncertainty,
                   sigma_floor, cmap, wedge):
        return dict(scheme=scheme, grey=grey, edgecolor=edgecolor,
                    match_spines=match_spines, uncertainty=uncertainty,
                    sigma_floor=sigma_floor, cmap=cmap, wedge=wedge)

    def plot_slip_bivariate(self, valmax=None, sigmamax=None, path=None,
                            title="Total slip and posterior uncertainty",
                            scheme="continuous", grey=(0.35, 0.35, 0.35),
                            edgecolor="0.55", match_spines=True, uncertainty="sigma",
                            sigma_floor=0.0, cmap=None, wedge="discrete"):
        """``scheme``: 'continuous' (interpolated cmap->grey), 'discrete'
        (programmatic 15-bin palette with the same knobs), or 'original' (the
        ported house palette).  ``grey`` is the maximally-uncertain colour;
        ``cmap`` is the zero-sigma slip ramp (name or Colormap; None = house
        ramp); ``uncertainty`` picks the secondary axis ('sigma' or 'cv' =
        sigma/slip, for which sigmamax is ~1); ``sigma_floor`` keeps patches
        saturated until the uncertainty passes that fraction of sigmamax;
        ``wedge`` is 'discrete' or 'continuous' for the colour key."""
        style = self._biv_style(scheme, grey, edgecolor, match_spines,
                                uncertainty, sigma_floor, cmap, wedge)
        valmax, sigmamax = self._bivariate_limits(valmax, sigmamax)
        print(f"[biv] valmax = {valmax}, sigmamax = {sigmamax}, scheme = {scheme}, "
              f"uncertainty = {uncertainty}")
        fig, axes, colls = self._bivariate_scaffold(valmax, sigmamax, style)
        self._set_bivariate_colors(colls, self.summ["tot_mode"], self.summ["tot_std"],
                                   valmax, sigmamax, style)
        fig.suptitle(title, fontsize=11)
        self._save(fig, self._path(path, "slip_bivariate.png"))

    def plot_convergence_video(self, valmax=None, sigmamax=None, path=None,
                               fps=2, scheme="continuous", grey=(0.35, 0.35, 0.35),
                               edgecolor="0.55", match_spines=True, uncertainty="sigma",
                               sigma_floor=0.0, cmap=None, wedge="discrete"):
        """One bivariate frame per AlTar step file (posterior mode + std of |slip|)."""
        style = self._biv_style(scheme, grey, edgecolor, match_spines,
                                uncertainty, sigma_floor, cmap, wedge)
        valmax, sigmamax = self._bivariate_limits(valmax, sigmamax)
        step_files = sorted(os.path.join(self.output_dir, f)
                            for f in os.listdir(self.output_dir)
                            if f.startswith("step_") and f.endswith(".h5"))
        frames = []
        for f in step_files:
            ps, beta = self.load_step(f)
            tot = np.hypot(np.hstack([ps[k] for k in self.ss_keys]),
                           np.hstack([ps[k] for k in self.ds_keys]))
            frames.append((os.path.splitext(os.path.basename(f))[0],
                           beta, _kde_mode(tot, self.seed), tot.std(axis=0)))
            print(f"  [video] processed {frames[-1][0]} (beta={beta:.4g})")

        fig, axes, colls = self._bivariate_scaffold(valmax, sigmamax, style)
        label = fig.text(0.02, 0.97, "", ha="left", va="top", fontsize=11)

        def update(i):
            name, beta, slip, sigma = frames[i]
            self._set_bivariate_colors(colls, slip, sigma, valmax, sigmamax, style)
            label.set_text(f"{name}   " + (f"beta = {beta:.3g}"
                                           if np.isfinite(beta) else ""))
            return colls

        ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
        out = self._path(path, "convergence_bivariate.mp4")
        ani.save(out, writer="ffmpeg", fps=fps, dpi=180)
        plt.close(fig)
        print(f"[fig] {os.path.basename(out)} ({len(frames)} frames)")

    # --------------------------------------------------------------------- #
    #  Per-patch posterior PDFs
    # --------------------------------------------------------------------- #
    def random_patches(self, n=8):
        rng = np.random.default_rng(self.seed)
        return rng.choice(self.fault.n_patches, n, replace=False)

    def top_slip_patches(self, n=8):
        return np.argsort(self.summ["tot_mode"])[-n:][::-1]

    def depth_transect(self, n=8):
        """Down-dip column of patches through the along-strike slip maximum."""
        peak = int(np.argmax(self.summ["tot_mode"]))
        strand = int(self.fault.fault_ids[peak])
        sec = self._sections[strand]
        local_peak = int(np.flatnonzero(sec["ids"] == peak)[0])
        s_cen = sec["polys"][:, :, 0].mean(axis=1)
        near = np.argsort(np.abs(s_cen - s_cen[local_peak]))
        chosen, depths_taken = [], []
        for j in near:
            d = sec["depth"][j]
            if all(abs(d - t) > 0.8 for t in depths_taken):
                chosen.append(j)
                depths_taken.append(d)
            if len(chosen) == n:
                break
        chosen = sorted(chosen, key=lambda j: sec["depth"][j])
        return [int(sec["ids"][j]) for j in chosen]

    def plot_patch_pdfs(self, patch_ids, subtitle="", path=None):
        """Top: fault sections (total-slip mode, grey) with the selected patches
        filled in qualitative colours.  Bottom: their |strike-slip| posterior
        KDEs (peak-normalised), in matching colours."""
        patch_ids = list(patch_ids)
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:len(patch_ids)]

        fig = plt.figure(figsize=(11, 7))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.32)
        top_axes = self._section_axes(fig, subplot_spec=outer[0])

        norm = mcolors.Normalize(0., max(float(self.summ["tot_mode"].max()), 1e-6))
        self._draw_sections(top_axes, values=self.summ["tot_mode"],
                            cmap=plt.get_cmap("Greys"), norm=norm, edgecolor="0.75",
                            linewidth=0.15)
        for ax, sec in zip(top_axes, self._sections):
            lookup = {g: i for i, g in enumerate(sec["ids"])}
            for c, pid in zip(colors, patch_ids):
                if pid in lookup:
                    poly = sec["polys"][lookup[pid]]
                    ax.fill(poly[:, 0], poly[:, 1], facecolor=c, edgecolor="k",
                            linewidth=0.8, zorder=5)

        ax_pdf = fig.add_subplot(outer[1])
        for c, pid in zip(colors, patch_ids):
            curve = _kde_curve(np.abs(self.summ["ss"][:, pid]))
            if curve is None:
                continue
            grid, dens = curve
            dens = dens / dens.max()
            ax_pdf.plot(grid, dens, color=c, lw=1.6, label=f"patch {pid}")
            ax_pdf.fill_between(grid, dens, color=c, alpha=0.06)
        ax_pdf.set_ylim(0., 1.1)
        ax_pdf.set_xlabel("|Strike-slip| (m)")
        ax_pdf.set_ylabel("Normalised posterior density")
        ax_pdf.legend(fontsize=8, ncol=8, frameon=False, loc="lower center",
                      bbox_to_anchor=(0.5, 1.01))
        for spine in ("top", "right"):
            ax_pdf.spines[spine].set_visible(False)
        fig.suptitle(f"Per-patch posterior PDFs -- {subtitle}", fontsize=11)
        self._save(fig, self._path(path, "patch_pdfs.png"))

    def plot_ramp_pdfs(self, path=None):
        """Posterior KDEs of the ramp parameters, one subplot each."""
        ramp = self.ps[self.ramp_key]
        names = ["offset (m)", "x gradient (m/norm)", "y gradient (m/norm)"]
        n = ramp.shape[1]
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2), layout="constrained")
        for j, ax in enumerate(np.atleast_1d(axes)):
            x = ramp[:, j]
            grid = np.linspace(x.min(), x.max(), KDE_GRID)
            dens = gaussian_kde(x[:KDE_MAX_SAMPLES])(grid)
            ax.plot(grid, dens, lw=1.6, color="#3266ad")
            ax.fill_between(grid, dens, color="#3266ad", alpha=0.08)
            ax.axvline(np.median(x), color="crimson", lw=1., ls="--",
                       label=f"median {np.median(x):.3g}")
            ax.set_xlabel(names[j % 3] if n % 3 == 0 else f"ramp[{j}]")
            ax.legend(frameon=False, fontsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        np.atleast_1d(axes)[0].set_ylabel("Posterior density")
        fig.suptitle("Ramp parameter posteriors", fontsize=11)
        self._save(fig, self._path(path, "ramp_pdfs.png"))

    # --------------------------------------------------------------------- #
    #  Data / model / residual maps (needs the InversionManager)
    # --------------------------------------------------------------------- #
    def _trace_lonlat(self, proj):
        trace = []
        for geom in self.fault.get_surface_trace_xy().geometry:
            c = np.asarray(geom.coords)
            lo, la = proj(c[:, 0], c[:, 1], inverse=True)
            trace.append(np.column_stack([lo, la]))
        return trace

    def _fit_coords(self):
        """{label -> {type, lon, lat, trace}} from the manager or archived cache."""
        if self.inv is not None:
            trace = self._trace_lonlat(self.inv._get_proj())
            return {label: dict(type=self.inv._dataset_types[label],
                                lon=ds.lon, lat=ds.lat, trace=trace)
                    for label, ds in self.inv.datasets.items()}
        return self._coords_arg

    def plot_fit_maps(self):
        """Data / model / residual maps for every dataset, model = G @ (mode)."""
        assert self.G is not None and self._dslices is not None, \
            "need G, d and dataset slices for fit maps"
        d = self.d
        pred = self.G @ self.m_mode
        coords = self._fit_coords()
        for label, sl in self._dslices.items():
            c = coords[label]
            dtype = str(c.get("type", "insar")).lower()
            path = os.path.join(self.figs_dir, f"fit_{label}.png")
            if "gnss" in dtype:
                self._fit_gnss(label, c, d[sl], pred[sl], path)
            elif "optical" in dtype:
                self._fit_optical(label, c, d[sl], pred[sl], path)
            else:
                self._fit_insar(label, c, d[sl], pred[sl], path)

    def _fit_insar(self, label, c, d_obs, d_mod, path):
        resid = d_obs - d_mod
        vlim = float(np.nanpercentile(np.abs(d_obs), 98))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True,
                                 layout="constrained")
        for ax, vals, title in zip(axes, (d_obs, d_mod, resid),
                                   ("Data", "Model (posterior mode)", "Residual")):
            sc = ax.scatter(c["lon"], c["lat"], c=vals, cmap=cmc.vik, s=8,
                            vmin=-vlim, vmax=vlim, zorder=2)
            for t in c["trace"]:
                ax.plot(t[:, 0], t[:, 1], "k-", lw=1.2, zorder=3)
            rms = float(np.sqrt(np.mean(vals ** 2))) if title == "Residual" else None
            ax.set_title(title + (f"  (RMS {rms*100:.1f} cm)" if rms is not None
                                  else ""))
            ax.set_xlabel("Longitude (°)")
            ax.set_aspect(1. / np.cos(np.radians(float(np.mean(c["lat"])))))
        axes[0].set_ylabel("Latitude (°)")
        fig.colorbar(sc, ax=axes, label="LOS displacement (m)", shrink=0.75)
        fig.suptitle(f"{label}: data vs. posterior-mode prediction", fontsize=12)
        self._save(fig, path)

    def _fit_gnss(self, label, c, d_obs, d_mod, path):
        """Left: data (black) vs model (red) arrows.  Right: residual arrows."""
        de_o, dn_o = d_obs[0::2], d_obs[1::2]
        de_m, dn_m = d_mod[0::2], d_mod[1::2]
        de_r, dn_r = de_o - de_m, dn_o - dn_m

        lon, lat = c["lon"], c["lat"]
        lon_range = float(lon.max() - lon.min()) or 1.
        max_disp = float(np.max(np.hypot(de_o, dn_o)))
        scale = max_disp / (0.15 * lon_range)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True,
                                 layout="constrained")
        for ax in axes:
            for t in c["trace"]:
                ax.plot(t[:, 0], t[:, 1], "r-", lw=1.2, zorder=2)
            ax.set_xlabel("Longitude (°)")
            ax.set_aspect(1. / np.cos(np.radians(float(np.mean(lat)))))
            ax.locator_params(axis="x", nbins=6)
        axes[0].set_ylabel("Latitude (°)")

        q1 = axes[0].quiver(lon, lat, de_o, dn_o, angles="xy", scale_units="xy",
                            scale=scale, color="k", width=0.004, zorder=3)
        axes[0].quiver(lon, lat, de_m, dn_m, angles="xy", scale_units="xy",
                       scale=scale, color="crimson", width=0.0025, zorder=4)
        ref = 10. ** np.floor(np.log10(max(max_disp, 1e-9)))
        axes[0].quiverkey(q1, 0.85, 0.06, ref, f"{ref*1000:.0f} mm", labelpos="S",
                          coordinates="axes", fontproperties=dict(size=8))
        axes[0].set_title("Data (black) vs model (red)")

        q2 = axes[1].quiver(lon, lat, de_r, dn_r, angles="xy", scale_units="xy",
                            scale=scale, color="k", width=0.004, zorder=3)
        axes[1].quiverkey(q2, 0.85, 0.06, ref, f"{ref*1000:.0f} mm", labelpos="S",
                          coordinates="axes", fontproperties=dict(size=8))
        rms = float(np.sqrt(np.mean(np.r_[de_r, dn_r] ** 2)))
        axes[1].set_title(f"Residuals, same scale  (RMS {rms*1000:.1f} mm)")
        fig.suptitle(f"{label}: horizontal offsets vs. posterior-mode prediction",
                     fontsize=12)
        self._save(fig, path)

    def _fit_optical(self, label, c, d_obs, d_mod, path):
        """Optical EW/NS: 2x3 (component x data/model/residual) scatter maps."""
        N = len(c["lon"])
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True,
                                 layout="constrained")
        for row, comp in enumerate(("EW", "NS")):
            obs = d_obs[row * N:(row + 1) * N]
            mod = d_mod[row * N:(row + 1) * N]
            vlim = float(np.nanpercentile(np.abs(obs), 98))
            for ax, vals, title in zip(axes[row], (obs, mod, obs - mod),
                                       ("Data", "Model", "Residual")):
                sc = ax.scatter(c["lon"], c["lat"], c=vals, cmap=cmc.vik, s=8,
                                vmin=-vlim, vmax=vlim, zorder=2)
                for t in c["trace"]:
                    ax.plot(t[:, 0], t[:, 1], "k-", lw=1.2, zorder=3)
                if row == 0:
                    ax.set_title(title)
            fig.colorbar(sc, ax=axes[row], label=f"{comp} displacement (m)",
                         shrink=0.8)
        fig.suptitle(f"{label}: data vs. posterior-mode prediction", fontsize=12)
        self._save(fig, path)

    # --------------------------------------------------------------------- #
    #  Trade-offs, moment, resolution, annealing
    # --------------------------------------------------------------------- #
    def _param_blocks(self):
        """(name, start, stop) in model-vector order for each ParameterSet."""
        blocks, off = [], 0
        for key in self.ss_keys + self.ds_keys + [self.ramp_key]:
            w = self.ps[key].shape[1]
            blocks.append((key, off, off + w))
            off += w
        return blocks

    def plot_correlation_matrix(self, path=None):
        corr = np.corrcoef(self.samples.T)
        blocks = self._param_blocks()
        fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
        im = ax.imshow(corr, cmap=cmc.vik, vmin=-1, vmax=1, interpolation="nearest")
        for _, start, stop in blocks:
            for edge in (start, stop):
                ax.axhline(edge - 0.5, color="k", lw=0.6)
                ax.axvline(edge - 0.5, color="k", lw=0.6)
        centers = [(start + stop) / 2 for _, start, stop in blocks]
        names = [name for name, _, _ in blocks]
        ax.set_xticks(centers, names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(centers, names, fontsize=9)
        fig.colorbar(im, ax=ax, label="Posterior correlation", shrink=0.8)
        ax.set_title("Parameter trade-offs: posterior correlation matrix")
        self._save(fig, self._path(path, "posterior_correlation_matrix.png"))

    def plot_correlation_map(self, ref_patch=None, ref_comp="SS", path=None):
        """Correlation of one patch's slip with every patch's SS and DS, painted
        on the fault."""
        if ref_patch is None:
            ref_patch = int(np.argmax(self.summ["tot_mode"]))
        ref = (self.summ["ss"] if ref_comp == "SS" else self.summ["ds"])[:, ref_patch]
        ref = (ref - ref.mean()) / (ref.std() or 1.)

        def corr_with(arr):
            a = (arr - arr.mean(axis=0)) / np.where(arr.std(axis=0) > 0,
                                                    arr.std(axis=0), 1.)
            return (a * ref[:, None]).mean(axis=0)

        corr_ss = corr_with(self.summ["ss"])
        corr_ds = corr_with(self.summ["ds"])
        off_ref = np.r_[np.delete(corr_ss, ref_patch), corr_ds]
        vmax = float(np.ceil(np.abs(off_ref).max() * 10.) / 10.)

        fig = plt.figure(figsize=(11, 6))
        outer = fig.add_gridspec(2, 1, hspace=0.35)
        norm = mcolors.Normalize(-vmax, vmax)
        for row, (comp, vals) in enumerate((("strike-slip", corr_ss),
                                            ("dip-slip", corr_ds))):
            axes = self._section_axes(fig, subplot_spec=outer[row])
            self._draw_sections(axes, values=np.clip(vals, -vmax, vmax),
                                cmap=cmc.vik, norm=norm, edgecolor="0.7",
                                linewidth=0.1)
            axes[0].set_ylabel(f"Depth (km)\n[vs {comp}]")
            for ax, sec in zip(axes, self._sections):
                hit = np.flatnonzero(sec["ids"] == ref_patch)
                if hit.size:
                    poly = sec["polys"][hit[0]]
                    ax.fill(poly[:, 0], poly[:, 1], facecolor="none",
                            edgecolor="limegreen", linewidth=1.6, zorder=6)
        sm = ScalarMappable(cmap=cmc.vik, norm=norm)
        fig.colorbar(sm, ax=fig.axes, label="Posterior correlation", shrink=0.8)
        fig.suptitle(f"Trade-offs with patch {ref_patch} {ref_comp} (green outline)",
                     fontsize=11)
        self._save(fig, self._path(path, f"correlation_map_patch{ref_patch}.png"))

    def plot_moment_magnitude(self, path=None, mu=MU):
        """Posterior of moment magnitude, total and per strand."""
        areas = self.fault.areas
        m0_strand = {}
        for k in range(self.fault.n_subfaults):
            m = self.fault.fault_ids == k
            name = str(self.fault.subfault_names[k])
            m0_strand[name] = mu * (self.summ["tot"][:, m] * areas[m]).sum(axis=1)
        m0_tot = sum(m0_strand.values())

        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for name, m0 in {**m0_strand, "combined": m0_tot}.items():
            mw = (np.log10(m0) - 9.1) * 2. / 3.
            grid = np.linspace(mw.min(), mw.max(), KDE_GRID)
            dens = gaussian_kde(mw[:KDE_MAX_SAMPLES])(grid)
            lw = 2.2 if name == "combined" else 1.4
            ax.plot(grid, dens, lw=lw, label=f"{name} (Mw {np.median(mw):.2f})")
            ax.fill_between(grid, dens, alpha=0.08)
        ax.set_xlabel("Moment magnitude Mw")
        ax.set_ylabel("Posterior density")
        ax.legend(frameon=False, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title(f"Posterior moment magnitude (mu = {mu/1e9:.0f} GPa)")
        self._save(fig, self._path(path, "moment_magnitude.png"))

    def plot_uncertainty_vs_depth(self, path=None):
        fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
        depth = self.fault.depths / 1e3
        for k in range(self.fault.n_subfaults):
            m = self.fault.fault_ids == k
            ax.scatter(depth[m], self.summ["tot_std"][m], s=14, alpha=0.7,
                       label=str(self.fault.subfault_names[k]))
        ax.set_xlabel("Patch depth (km)")
        ax.set_ylabel("Posterior std of |slip| (m)")
        ax.legend(frameon=False, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_title("Loss of resolution with depth")
        self._save(fig, self._path(path, "uncertainty_vs_depth.png"))

    def plot_beta_annealing(self, stats_path=None, path=None):
        stats_path = stats_path or os.path.join(self.output_dir, "BetaStatistics.txt")
        rows = np.genfromtxt(stats_path, delimiter=",", skip_header=1,
                             usecols=(0, 1, 2))
        acc = []
        with open(stats_path) as fh:
            next(fh)
            for line in fh:
                a, i, r = (float(v) for v in
                           line.split("(")[1].rstrip(")\n").split(","))
                tot = a + i + r
                acc.append(a / tot if tot else np.nan)
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        ax.semilogy(rows[:, 0], np.maximum(rows[:, 1], 1e-12), "o-", ms=3,
                    label="beta")
        ax.set_xlabel("Annealing iteration")
        ax.set_ylabel("beta (log)")
        ax2 = ax.twinx()
        ax2.plot(rows[:, 0], acc, "s-", ms=3, color="crimson",
                 label="acceptance rate")
        ax2.set_ylabel("Acceptance rate", color="crimson")
        ax2.tick_params(axis="y", colors="crimson")
        ax.set_title("AlTar annealing schedule")
        self._save(fig, self._path(path, "beta_annealing.png"))
