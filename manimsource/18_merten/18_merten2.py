from manim import *
import numpy as np
import math
import sys

sys.path.append('../../')
import manimhelper as mh

"""
mertens_plot_large.py
=====================
Plots M(x) from x=1 to x=x_max (default 1e12) using the Hyperbola DP method.

The sample points used are:
  • x = 1 .. u  (sieve range):  n_small evenly-spaced integers (default 300)
  • x = u .. x_max (DP range):  ALL floor-quotients of x_max computed by the DP
                                 (~10 000 points for x_max=1e12, ~2 000 for 1e10)

These are plotted on a LOG x-axis (which is natural: the floor-quotients x//k
for k=1,2,... are harmonically spaced, appearing evenly spread on a log scale).

Runtime:  ~100 s for x_max = 1e12  (sieve + DP)
           ~3 s  for x_max = 1e10
           ~0.5 s for x_max = 1e9
"""

import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── re-use the computation core from mertens_large.py ─────────────────────────
try:
    from mertens_large import _build_M_small
except ImportError:
    # Inline fallback (identical to mertens_large._build_M_small)
    def _build_M_small(u: int) -> np.ndarray:
        sq = int(math.isqrt(u)) + 1
        is_p = np.ones(sq + 1, dtype=bool)
        is_p[0] = is_p[1] = False
        for p in range(2, int(math.isqrt(sq)) + 1):
            if is_p[p]:
                is_p[p * p :: p] = False
        small_primes = np.where(is_p)[0].tolist()
        mu = np.ones(u + 1, dtype=np.int8); mu[0] = 0
        for p in small_primes:
            mu[p::p] *= -1
            p2 = p * p
            if p2 <= u: mu[p2::p2] = 0
        cofactor = np.arange(u + 1, dtype=np.int32)
        for p in small_primes:
            cofactor[p::p] //= p
        mu[(cofactor != 1) & (mu != 0)] *= -1
        M = np.empty(u + 1, dtype=np.int32); M[0] = 0
        M[1:] = np.cumsum(mu[1:])
        return M


# ── core computation ───────────────────────────────────────────────────────────

def compute_plot_data(x_max: int, n_small: int = 300, verbose: bool = True):
    """
    Returns (xs, Ms) — parallel arrays of x values and M(x) values.

    xs includes:
      • n_small evenly-spaced integers in [1, u]
      • all floor-quotients of x_max in (u, x_max]   (~10 000 for x_max=1e12)

    No extra computation beyond what MertensDP(x_max) already does.
    """
    u = max(100, int(math.ceil(x_max ** (2 / 3))) + 1)

    # ── Phase 1: sieve ─────────────────────────────────────────────────────────
    if verbose:
        print(f"[1/2] Sieve up to u = {u:,}  (~{u*5/1e6:.0f} MB) …", flush=True)
    t0 = time.perf_counter()
    M_small = _build_M_small(u)
    if verbose:
        print(f"      done in {time.perf_counter() - t0:.1f}s")

    # ── Phase 2: DP over floor-quotients of x_max ──────────────────────────────
    k_max  = x_max // (u + 1)
    large  = [x_max // k for k in range(k_max, 0, -1)]   # ascending

    if verbose:
        print(f"[2/2] DP over {len(large):,} floor-quotients …", flush=True)
    t1 = time.perf_counter()

    cache: dict[int, int] = {}

    def M(v: int) -> int:
        return int(M_small[v]) if v <= u else cache[v]

    for idx, v in enumerate(large):
        sqv = int(math.isqrt(v))
        result = 1

        k = 2; k_end = v // (sqv + 1)
        while k <= k_end:
            q = v // k; k_next = v // q + 1
            count = min(k_next, k_end + 1) - k
            result -= count * M(q)
            k = k_next

        if sqv > 0:
            j    = np.arange(1, sqv + 1, dtype=np.int64)
            result -= int(np.dot(v // j - v // (j + 1),
                                 M_small[j].astype(np.int64)))

        cache[v] = result

        if verbose and (idx + 1) % 1000 == 0:
            done = idx + 1
            el   = time.perf_counter() - t1
            eta  = el / done * (len(large) - done)
            print(f"      {done}/{len(large)}  elapsed={el:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    if verbose:
        print(f"      DP done in {time.perf_counter() - t1:.1f}s")

    # ── Assemble plot data ─────────────────────────────────────────────────────
    # Small range: n_small evenly spaced in [1, u]
    small_xs = np.linspace(1, u, n_small, dtype=np.int64)
    small_Ms = M_small[small_xs].astype(np.int64)

    # Large range: all floor-quotients (the DP cache)
    large_xs = np.array(sorted(cache.keys()), dtype=np.int64)
    large_Ms = np.array([cache[v] for v in large_xs], dtype=np.int64)

    xs = np.concatenate([small_xs, large_xs])
    Ms = np.concatenate([small_Ms, large_Ms])
    order = np.argsort(xs)
    return xs[order], Ms[order]


class MertensPlot2(Scene):
    def construct(self):
        x_max = 10 ** 11
        xmax = 1.
        ymax = 1.

        x_exp = int(round(math.log10(x_max)))

        print(f"Mertens plot: x = 1 .. {x_max:.0e}\n")
        t0 = time.perf_counter()
        xs, Ms = compute_plot_data(x_max, n_small=300, verbose=True)
        print(f"\nTotal compute: {time.perf_counter() - t0:.1f}s")

        axes = VGroup()

        for mask, color, title_sfx in [
            (xs <= 10**8,   YELLOW, "sieve range: linear x-axis"),
            (xs > 10**8,    BLUE,   f"DP range: log x-axis  [{10**8:.0e}, 10^{x_exp}]"),
        ]:
            xi = xs[mask]
            Mi = Ms[mask]

            xmax = xi[-1]
            ymax = max(max(Mi), -min(Mi)) * 1.05

            print(xmax.shape)

            ax = Axes(x_range=[0., xmax*1.05], x_length=12, y_length=3.5, y_range=[-ymax, ymax],
                 axis_config={'color': WHITE, 'stroke_width': 4,
                              'include_ticks': False, 'include_tip': True})
            print(xi[-1])
            print('1')
            plt = ax.plot_line_graph(xi, Mi, add_vertex_dots=False, line_color=color, stroke_width=6)
            print('2')
            axes += VGroup(ax, plt)
            print('3')

        axes.arrange(direction=DOWN)

        self.add(axes)

# ── plotting ───────────────────────────────────────────────────────────────────

def plot(xs, Ms, x_max: int, out_path: str = "mertens_large_plot.png"):
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10),
        facecolor="#0d0d0d",
        gridspec_kw=dict(hspace=0.45, top=0.90, bottom=0.08, left=0.09, right=0.97)
    )
    fig.suptitle(
        r"Mertens Function   $M(x) = \sum_{n=1}^{x} \mu(n)$   up to   $x = 10^{" +
        f"{int(round(math.log10(x_max)))}" + r"}$",
        color="white", fontsize=16, fontweight="bold"
    )

    x_exp = int(round(math.log10(x_max)))
    YELLOW = "#FFD700"
    CYAN   = "#00E5FF"
    GREY   = "#555555"

    for ax, (mask, color, title_sfx) in zip(
        axes,
        [
            (xs <= 10**8,   YELLOW, "sieve range: linear x-axis"),
            (xs > 10**8,    CYAN,   f"DP range: log x-axis  [{10**8:.0e}, 10^{x_exp}]"),
        ]
    ):
        ax.set_facecolor("#111111")
        for sp in ax.spines.values(): sp.set_color("#333333")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")

        xi = xs[mask]
        Mi = Ms[mask]
        if len(xi) == 0:
            continue

        ax.axhline(0, color=GREY, lw=0.8, ls="--", zorder=1)
        ax.fill_between(xi, Mi, 0, where=(Mi >= 0),
                        color=color, alpha=0.10, zorder=2)
        ax.fill_between(xi, Mi, 0, where=(Mi < 0),
                        color=color, alpha=0.07, zorder=2)
        ax.plot(xi, Mi, color=color, lw=0.8, zorder=3)

        # ± sqrt(x) Mertens-conjecture bounds
        sq = np.sqrt(xi.astype(np.float64))
        ax.plot(xi,  sq, color="white", lw=0.5, ls=":", alpha=0.3, label=r"$\pm\sqrt{x}$")
        ax.plot(xi, -sq, color="white", lw=0.5, ls=":", alpha=0.3)

        # annotate global min/max
        amin, amax = Mi.argmin(), Mi.argmax()
        for ai, yv in [(amin, Mi[amin]), (amax, Mi[amax])]:
            xv = xi[ai]
            dy = 8 if yv < 0 else -14
            ax.annotate(
                f"M={yv:,}\nx={xv:.3g}",
                xy=(xv, yv), xytext=(-10, dy), textcoords="offset points",
                fontsize=6.5, color="white",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.5)
            )

        # legend
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)

        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("M(x)", fontsize=9)

    # bottom panel: log x-axis
    ax_log = axes[1]
    ax_log.set_xscale("log")
    ax_log.set_title(f"M(x), {axes[1].get_title() if axes[1].get_title() else 'DP range (log scale)'}",
                     color="white", fontsize=10, pad=5)
    axes[0].set_title("M(x), sieve range (linear scale)", color="white", fontsize=10, pad=5)

    # x-axis formatter for top panel
    axes[0].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else
                                          f"{v/1e3:.0f}K" if v >= 1e3 else f"{v:.0f}")
    )

    fig.text(
        0.5, 0.01,
        f"Sample points: {int((xs <= 10**8).sum()):,} (sieve, linear) + "
        f"{int((xs > 10**8).sum()):,} (floor-quotients of 10^{x_exp}, log)   •   "
        "Dotted: Mertens conjecture bound ±√x (disproved for sufficiently large x)",
        ha="center", va="bottom", fontsize=7, color="#777777"
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out_path}")
    return out_path


# ── entry point ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print(f"Plot points:   {len(xs):,}  (x range {xs[0]:,} .. {xs[-1]:,})")
#     print(f"M({x_max:.0e}) = {Ms[-1]:,}")
#
#     out = f"mertens_large_plot_{int(math.log10(x_max))}.png"
#     plot(xs, Ms, x_max, out_path=out)