from manim import *
import numpy as np
import math
import sys

sys.path.append('../../')
import manimhelper as mh
# from common.wigner import *

"""
Mertens function M(x) = sum of mu(n) for n = 1 to x
where mu(n) is the Möbius function:
  mu(1) = 1
  mu(n) = 0     if n has a squared prime factor
  mu(n) = (-1)^k if n is a product of k distinct primes
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# Core sieve-based computation
# ---------------------------------------------------------------------------

def compute_mobius_sieve(limit: int) -> np.ndarray:
    """Return mu[1..limit] using a linear sieve (O(n) time)."""
    mu = np.zeros(limit + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(limit + 1, dtype=bool)
    primes = []

    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1                        # i is prime → mu = -1

        for p in primes:
            if i * p > limit:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0                 # p² divides i·p
                break
            else:
                mu[i * p] = -mu[i]            # multiply by one more prime

    return mu


def mertens_cumsum(mu: np.ndarray) -> np.ndarray:
    """Cumulative sum of mu[1..] to give M(1), M(2), …"""
    return np.cumsum(mu[1:]).astype(np.int32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

RANGES = [100, 1_000, 1_000_000]
COLORS = ["#2196F3", "#E91E63", "#4CAF50"]
ALPHA_FILL = 0.12


def format_label(n: int) -> str:
    if n >= 1_000_000:
        return f"x up to {n // 1_000_000:,}M"
    if n >= 1_000:
        return f"x up to {n // 1_000:,}K"
    return f"x up to {n:,}"


class MertensPlot1(Scene):
    def construct(self):
        xmax = 1000000
        t0 = time.perf_counter()
        mu = compute_mobius_sieve(xmax)
        print(mu[:100])
        M_full = mertens_cumsum(mu)
        print(len(mu))
        print(mu.shape)
        n = 500
        xstep = round(xmax / n)
        print(f"done in {time.perf_counter() - t0:.2f}s")

        ymax = math.sqrt(xmax)
        ax = Axes(x_range=[1., xmax * 1.1], y_range=[-ymax, ymax], x_length=12, y_length=7,
                  axis_config={'color': WHITE, 'stroke_width': 4,
                               'include_ticks': False, 'include_tip': True})

        yvals = M_full[::xstep]
        n1 = len(yvals)
        xvals = np.linspace(0., xmax, n1)

        plt1 = ax.plot_line_graph(xvals, yvals, add_vertex_dots=False, line_color=YELLOW, stroke_width=6)

        self.add(ax, plt1)
