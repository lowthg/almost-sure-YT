import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi
import primecountpy as primecount


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

A = 10**3
B = 10**10

# Number of logarithmically stratified sample points
N = 200_000

# Histogram bucket width
bin_width = 0.02

# Expected range of
# (pi(x) - Li(x)) / Li(sqrt(x))
bins = np.arange(-1.1, 0.1 + bin_width, bin_width)

seed = 12345


# ------------------------------------------------------------
# Stratified sampling in logarithmic measure
# ------------------------------------------------------------

rng = np.random.default_rng(seed)

logA = np.log(A)
logB = np.log(B)

du = (logB - logA) / N

# Take one random point from each equal-width interval in log(x).
#
# Since dx/x = d(log x), every point represents exactly the same
# amount of logarithmic measure.
u = logA + du * (np.arange(N) + rng.random(N))

x = np.exp(u).astype(np.int64)


# ------------------------------------------------------------
# Compute pi(x)
# ------------------------------------------------------------

print("Computing pi(x)...")

pi_x = np.fromiter(
    (primecount.prime_pi(int(xx)) for xx in x),
    dtype=np.int64,
    count=N
)

print("Finished computing pi(x)")


# ------------------------------------------------------------
# Compute Li(x) and Li(sqrt(x))
#
# Li(x) = Ei(log x)
# ------------------------------------------------------------

logx = np.log(x.astype(np.float64))

Li_x = expi(logx)
Li_sqrt_x = expi(0.5 * logx)


# ------------------------------------------------------------
# Statistic
# ------------------------------------------------------------

z = (pi_x - Li_x) / Li_sqrt_x


# ------------------------------------------------------------
# Histogram
# ------------------------------------------------------------

counts, edges = np.histogram(z, bins=bins)

# Each sample represents this amount of logarithmic mass:
mass_per_sample = (logB - logA) / N

# So this approximates
#
#     sum 1/n
#
# over n whose statistic lies in each bucket
weighted_mass = counts * mass_per_sample


# ------------------------------------------------------------
# Display histogram
# ------------------------------------------------------------

centres = 0.5 * (edges[:-1] + edges[1:])
widths = np.diff(edges)

plt.figure(figsize=(10, 6))

plt.bar(
    centres,
    weighted_mass,
    width=0.92 * widths,
    align="center"
)

plt.xlabel(
    r"$(\pi(x)-\mathrm{Li}(x))/\mathrm{Li}(\sqrt{x})$",
    fontsize=13
)

plt.ylabel(
    r"$1/n$-weighted mass",
    fontsize=13
)

plt.title(
    rf"$1/n$-weighted histogram, ${A:.0e} \leq x \leq {B:.0e}$"
)

plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

# This is the line that actually opens/displays the plot:
plt.show()


# ------------------------------------------------------------
# Some summary information
# ------------------------------------------------------------

print()
print("Number of samples:", N)
print("Minimum sampled value:", z.min())
print("Maximum sampled value:", z.max())
print("Mean:", z.mean())
print("Standard deviation:", z.std())