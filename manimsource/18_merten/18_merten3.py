"""
mertens_plot_large.py  —  Plot M(x) from 1 to x_max (default 1e12)

Architecture
============
The hyperbola DP naturally computes M at floor-quotients of x_max,
which cluster at small x (k=1..x//u gives x//k ~ u..x_max but harmonically spaced).
For a LINEAR-scale plot, the upper range needs extra sample points.

Cost of an extra point v: O(v^(2/3)) sieve + O(v^(1/3)) DP values each needing O(v^(1/6)) numpy work.
Sharing M_small means only the DP phase is extra. At v~1e10 this is ~0.1s; at v~1e11 ~15s; at v~1e12 ~85s.

Strategy (3 panels):
  Panel A – linear, 1 → u                : sieve lookups, 300 pts, free
  Panel B – linear, u → x_max/10         : main DP floor-quotients, very dense
  Panel C – linear, x_max/10 → x_max     : main DP pts (only 10) + batch extra DPs
                                            feasible for x_max ≤ ~1e11; for 1e12 still shows 10 pts

Usage:
    python mertens_plot_large.py               # x_max=1e12, no extra pts in top decade (~100s)
    python mertens_plot_large.py 1e12 --extra 9    # +9 pts in top decade (~10 min)
    python mertens_plot_large.py 1e11 --extra 200  # dense linear plot up to 1e11 (~5 min)
    python mertens_plot_large.py 1e10 --extra 200  # dense linear plot up to 1e10 (~30s)
    python mertens_plot_large.py 1e9              # fast: u covers 99%, trivial extra pts
"""

import math, sys, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from mertens_large import _build_M_small
except ImportError:
    def _build_M_small(u):
        sq = int(math.isqrt(u)) + 1
        isp = np.ones(sq+1, dtype=bool); isp[0]=isp[1]=False
        for p in range(2, int(math.isqrt(sq))+1):
            if isp[p]: isp[p*p::p] = False
        sp = np.where(isp)[0].tolist()
        mu = np.ones(u+1, dtype=np.int8); mu[0]=0
        for p in sp:
            mu[p::p] *= -1
            p2=p*p
            if p2<=u: mu[p2::p2]=0
        cof = np.arange(u+1, dtype=np.int32)
        for p in sp: cof[p::p] //= p
        mu[(cof!=1)&(mu!=0)] *= -1
        M = np.empty(u+1, dtype=np.int32); M[0]=0; M[1:]=np.cumsum(mu[1:])
        return M


# ── DP for one target, reusing a pre-built M_small ────────────────────────────

def _dp_single(v: int, M_small: np.ndarray, u: int,
               tag: str = "", verbose: bool = True) -> dict:
    """Return {fq: M(fq)} for all large floor-quotients fq of v."""
    k_max = v // (u + 1)
    if k_max == 0:
        return {}
    large = [v // k for k in range(k_max, 0, -1)]   # ascending
    cache: dict[int, int] = {}

    def M(w):
        return int(M_small[w]) if w <= u else cache[w]

    t0 = time.perf_counter()
    for idx, w in enumerate(large):
        sqw = int(math.isqrt(w))
        res = 1
        k = 2; k_end = w // (sqw + 1)
        while k <= k_end:
            q = w // k; k_next = w // q + 1
            res -= (min(k_next, k_end+1) - k) * M(q)
            k = k_next
        if sqw > 0:
            j = np.arange(1, sqw+1, dtype=np.int64)
            res -= int(np.dot(w//j - w//(j+1), M_small[j].astype(np.int64)))
        cache[w] = res

        if verbose and len(large) > 500 and (idx+1) % 1000 == 0:
            el  = time.perf_counter()-t0
            eta = el/(idx+1)*(len(large)-idx-1)
            print(f"      {tag} {idx+1}/{len(large)} "
                  f"elapsed={el:.0f}s ETA={eta:.0f}s", flush=True)

    if verbose and len(large) <= 500:
        print(f"      {tag}: {len(large)} values  {time.perf_counter()-t0:.2f}s",
              flush=True)
    elif verbose:
        print(f"      {tag}: {len(large)} values  {time.perf_counter()-t0:.1f}s",
              flush=True)
    return cache


# ── main computation ───────────────────────────────────────────────────────────

def compute_plot_data(x_max: int, n_small: int = 300,
                      n_extra: int = 0, verbose: bool = True):
    """
    Returns (xs, Ms) covering 1..x_max.

    n_extra extra DP targets are placed linearly in [x_max//10, x_max].
    Each costs a fraction of the main DP (they share M_small).
    """
    u = max(100, int(math.ceil(x_max ** (2/3))) + 1)

    # ── sieve ──────────────────────────────────────────────────────────────────
    if verbose:
        print(f"[1/3] Sieve  u={u:,}  (~{u*5//1_000_000} MB)", flush=True)
    t0 = time.perf_counter()
    M_small = _build_M_small(u)
    if verbose:
        print(f"      done in {time.perf_counter()-t0:.1f}s")

    # ── main DP ────────────────────────────────────────────────────────────────
    if verbose:
        print(f"[2/3] Main DP  x_max={x_max:.3e}", flush=True)
    main_cache = _dp_single(x_max, M_small, u, tag="main", verbose=verbose)

    # ── extra DPs ──────────────────────────────────────────────────────────────
    # Place extra targets in [x_max//10, x_max], evenly spaced.
    # Their floor-quotients fill in the linear gaps.
    all_extra: dict[int, int] = {}
    if n_extra > 0:
        lo = x_max // 10
        extras = np.linspace(lo, x_max, n_extra + 2, dtype=np.int64)[1:-1]
        extras = [int(e) for e in np.unique(extras)
                  if e not in main_cache and e > u]
        if verbose:
            print(f"[3/3] Extra DPs: {len(extras)} targets in "
                  f"[{lo:.2e}, {x_max:.2e}]", flush=True)
        for i, v in enumerate(extras):
            if verbose:
                print(f"  [{i+1}/{len(extras)}] v={v:.4e}", flush=True)
            partial = _dp_single(v, M_small, u, tag=f"v={v:.3e}", verbose=verbose)
            all_extra.update(partial)
    elif verbose:
        print("[3/3] No extra DPs  (pass --extra N to add N extra targets)")

    # ── assemble ───────────────────────────────────────────────────────────────
    small_xs = np.linspace(1, u, n_small, dtype=np.int64)
    small_Ms = M_small[small_xs].astype(np.int64)

    combined: dict[int, int] = {}
    combined.update(all_extra)
    combined.update(main_cache)     # main takes priority
    if x_max in main_cache:
        combined[x_max] = main_cache[x_max]

    large_xs = np.array(sorted(combined), dtype=np.int64)
    large_Ms = np.array([combined[v] for v in large_xs], dtype=np.int64)

    xs = np.concatenate([small_xs, large_xs])
    Ms = np.concatenate([small_Ms, large_Ms])
    order = np.argsort(xs)
    return xs[order], Ms[order], u


# ── plotting ───────────────────────────────────────────────────────────────────

def plot(xs, Ms, u, x_max, out_path):
    x_exp  = int(round(math.log10(x_max)))
    thresh = x_max // 10           # boundary between panels B and C
    u_exp  = int(math.log10(u))

    GOLD = "#FFD700"
    CYAN = "#00E5FF"
    MINT = "#00FF99"

    fig = plt.figure(figsize=(15, 13), facecolor="#0d0d0d")
    fig.suptitle(
        r"Mertens Function   $M(x)=\sum_{n=1}^{x}\mu(n)$"
        f"   up to   $x=10^{{{x_exp}}}$",
        color="white", fontsize=16, fontweight="bold", y=0.97
    )

    # 3 rows: sieve | lower DP | upper DP
    gs = fig.add_gridspec(3, 1, hspace=0.50,
                          top=0.93, bottom=0.06, left=0.09, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    def fmt_x(v, _):
        if v >= 1e12: return f"{v/1e12:.2g}T"
        if v >= 1e9:  return f"{v/1e9:.2g}G"
        if v >= 1e6:  return f"{v/1e6:.2g}M"
        if v >= 1e3:  return f"{v/1e3:.0f}K"
        return f"{v:.0f}"

    def draw(ax, mask, color, title):
        ax.set_facecolor("#111111")
        for sp in ax.spines.values(): sp.set_color("#333333")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.set_title(title, color="white", fontsize=9.5, pad=5)
        ax.set_xlabel("x", fontsize=8.5)
        ax.set_ylabel("M(x)", fontsize=8.5)

        xi, Mi = xs[mask], Ms[mask]
        if len(xi) == 0:
            ax.text(0.5, 0.5, "no data in this range",
                    transform=ax.transAxes, color="#888", ha="center")
            return

        ax.axhline(0, color="#555", lw=0.8, ls="--", zorder=1)
        ax.fill_between(xi, Mi, 0, where=(Mi>=0), color=color, alpha=0.10, zorder=2)
        ax.fill_between(xi, Mi, 0, where=(Mi< 0), color=color, alpha=0.07, zorder=2)
        ax.plot(xi, Mi, color=color, lw=0.75, zorder=3)

        sq = np.sqrt(xi.astype(np.float64))
        ax.plot(xi,  sq, color="white", lw=0.5, ls=":", alpha=0.25, label=r"$\pm\sqrt{x}$")
        ax.plot(xi, -sq, color="white", lw=0.5, ls=":", alpha=0.25)
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", framealpha=0.6)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_x))

        # annotate min/max
        for ai, yv in [(Mi.argmin(), Mi.min()), (Mi.argmax(), Mi.max())]:
            xv = xi[ai]
            dy = 8 if yv < 0 else -14
            ax.annotate(f"M={yv:,}\nx={fmt_x(xv,'')}", xy=(xv, yv),
                        xytext=(-10, dy), textcoords="offset points",
                        fontsize=6.5, color="white",
                        arrowprops=dict(arrowstyle="-", color="#888", lw=0.5))

        n = len(xi)
        ax.text(0.99, 0.03, f"{n:,} pts", transform=ax.transAxes,
                color="#777", fontsize=7, ha="right", va="bottom")

    draw(axes[0], xs <= u,
         GOLD,
         f"Panel A — sieve range  [1, {u:.2e}]  (linear x-axis)")

    draw(axes[1], (xs > u) & (xs <= thresh),
         CYAN,
         f"Panel B — DP range     [{u:.1e}, {thresh:.1e}]  (linear x-axis)")

    n_top = int(((xs > thresh) & (xs <= x_max)).sum())
    label_extra = f"  —  {n_top} pts" + (
        "  (add --extra N for more)" if n_top < 20 else "")
    draw(axes[2], (xs > thresh) & (xs <= x_max),
         MINT,
         f"Panel C — top decade   [{thresh:.1e}, 10^{x_exp}]  (linear x-axis){label_extra}")

    n_s = int((xs<=u).sum())
    n_b = int(((xs>u)&(xs<=thresh)).sum())
    n_c = n_top
    fig.text(0.5, 0.01,
             f"Points: {n_s:,} (sieve) + {n_b:,} (DP, panel B) + {n_c:,} (DP, panel C)   •   "
             "Dotted: ±√x Mertens conjecture bound",
             ha="center", va="bottom", fontsize=7, color="#777777")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot M(x) up to x_max")
    p.add_argument("x_max",  nargs="?", type=float, default=1e12,
                   help="upper limit (default 1e12)")
    p.add_argument("--extra", type=int, default=0,
                   help="extra DP targets in top decade (each adds ~0.1s per 1e10 scale)")
    p.add_argument("--small", type=int, default=300,
                   help="sieve sample points (default 300)")
    args = p.parse_args()

    x_max = int(args.x_max)
    x_exp = int(round(math.log10(x_max)))

    print(f"Mertens plot: 1 → {x_max:.0e}  (extra={args.extra})\n")
    t_start = time.perf_counter()
    xs, Ms, u = compute_plot_data(x_max, n_small=args.small,
                                  n_extra=args.extra, verbose=True)
    elapsed = time.perf_counter() - t_start
    print(f"\nTotal compute: {elapsed:.1f}s")
    print(f"Plot points:   {len(xs):,}")
    print(f"M({x_max:.0e}) = {Ms[xs == x_max][0] if x_max in xs else '(not in xs)'}")

    out = f"mertens_linear_1e{x_exp}.png"
    plot(xs, Ms, u, x_max, out)