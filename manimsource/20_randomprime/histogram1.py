import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi
from scipy.stats import norm
import mpmath as mp
import primecountpy as primecount


# ============================================================
# PARAMETERS
# ============================================================

# Choose:
#
#     "uniform" : log X uniform
#     "normal"  : log X normal
#
SAMPLING = "uniform"


# ------------------------------------------------------------
# Parameters for uniform log sampling
#
# Used only when SAMPLING == "uniform"
# ------------------------------------------------------------

X_MIN = 1e8
X_MAX = 1e10


# ------------------------------------------------------------
# Parameters for Gaussian log sampling
#
# Used only when SAMPLING == "normal"
#
# This distribution is NOT truncated.
# ------------------------------------------------------------

MU = np.log(1e9)
SIGMA = 1.0


# ------------------------------------------------------------
# Numerical parameters
# ------------------------------------------------------------

N_EMPIRICAL = 200_000

# The theoretical simulation is cheap compared with prime_pi,
# so this can be much larger.
N_THEORY = 1_000_000

# Number of zeta zeros treated explicitly.
# Remaining zeros are approximated by Gaussian noise with
# exactly the correct remaining variance.
N_ZEROS = 200

BIN_WIDTH = 0.02

SEED_EMPIRICAL = 72345
SEED_THEORY_X = 54321
SEED_THEORY_PHASES = 98765


# ============================================================
# Li(x)
#
# Li(x) = Ei(log x)
# ============================================================

def Li_from_logx(logx):
    return expi(logx)


# ============================================================
# SAMPLE log(X)
#
# IMPORTANT:
#
# This is stratified random sampling.
#
# Rather than taking iid uniform random numbers q in [0,1],
# divide [0,1] into N deterministic equal-probability strata,
# and choose one random point within each.
#
# This usually gives substantially lower Monte Carlo noise.
# ============================================================

def sample_logx(N, seed):

    rng = np.random.default_rng(seed)

    # One random point in each deterministic probability bucket
    q = (
        np.arange(N, dtype=float)
        + rng.random(N)
    ) / N

    if SAMPLING == "uniform":

        y0 = np.log(X_MIN)
        y1 = np.log(X_MAX)

        # Inverse CDF of Uniform[y0,y1]
        y = y0 + (y1 - y0) * q


    elif SAMPLING == "normal":

        # Inverse CDF of N(MU, SIGMA^2)
        #
        # No truncation.
        y = MU + SIGMA * norm.ppf(q)


    else:

        raise ValueError(
            "SAMPLING must be 'uniform' or 'normal'"
        )

    return y


# ============================================================
# FINITE-x THEORETICAL CENTRE
#
# c(x) =
#
#   [ -Li(x^(1/2))/2 - Li(x^(1/3))/3 ]
#   -----------------------------------
#              Li(x^(1/2))
#
# =
#
#   -1/2 - Li(x^(1/3))/(3 Li(x^(1/2)))
#
# We work directly with y = log x:
#
# log(x^(1/2)) = y/2
# log(x^(1/3)) = y/3
# ============================================================

def theoretical_centre_from_logx(y):

    Li_sqrt = Li_from_logx(0.5 * y)
    Li_cubert = Li_from_logx(y / 3.0)

    return (
        -0.5
        - Li_cubert / (3.0 * Li_sqrt)
    )


# ============================================================
# EMPIRICAL DISTRIBUTION
# ============================================================

print("Generating empirical sample...")

y_emp = sample_logx(
    N_EMPIRICAL,
    SEED_EMPIRICAL
)

x_emp = np.exp(y_emp).astype(np.int64)


# This should never matter for the Gaussian parameters used
# above, but pi(x) is only meaningful here for positive x.
if np.any(x_emp < 3):

    raise ValueError(
        "The chosen Gaussian is so broad that some sampled "
        "x values are below 3. Move MU upward or reduce SIGMA."
    )


print()
print("Empirical x range:")
print("minimum =", x_emp.min())
print("median  =", int(np.median(x_emp)))
print("maximum =", x_emp.max())
print()


# ------------------------------------------------------------
# Compute pi(x)
# ------------------------------------------------------------

print(
    "Computing pi(x) at",
    N_EMPIRICAL,
    "sample points..."
)

pi_emp = np.fromiter(
    (
        primecount.prime_pi(int(x))
        for x in x_emp
    ),
    dtype=np.int64,
    count=N_EMPIRICAL
)

print("Finished computing pi(x).")
print()


# ------------------------------------------------------------
# Statistic
#
# Z(x) =
#
#      pi(x) - Li(x)
#     ----------------
#        Li(sqrt x)
# ------------------------------------------------------------

Li_x_emp = Li_from_logx(y_emp)
Li_sqrt_emp = Li_from_logx(0.5 * y_emp)

z_emp = (
    pi_emp.astype(float)
    - Li_x_emp
) / Li_sqrt_emp


# ============================================================
# ZETA ZEROS
# ============================================================

print(
    "Computing first",
    N_ZEROS,
    "zeta zeros..."
)

mp.mp.dps = 30

gammas = np.array([
    float(mp.im(mp.zetazero(k)))
    for k in range(1, N_ZEROS + 1)
])

print("First zero =", gammas[0])
print("Last zero  =", gammas[-1])
print()


# ============================================================
# LIMITING OSCILLATORY PART
#
# Sum over positive zeros:
#
#     a_gamma cos(theta_gamma)
#
# with
#
#     a_gamma = 1/sqrt(1/4 + gamma^2).
#
# The sign is immaterial for the distribution.
# ============================================================

coeff = 1.0 / np.sqrt(
    0.25 + gammas**2
)


# ============================================================
# VARIANCE OF ALL ZEROS
#
# Identity:
#
# sum_{gamma > 0} 1/(1/4 + gamma^2)
#
#     = (2 + EulerGamma - log(4*pi))/2
#
# Since Var(cos theta) = 1/2:
#
# total variance =
#
#     (2 + EulerGamma - log(4*pi))/4
# ============================================================

variance_total = (
    2.0
    + np.euler_gamma
    - np.log(4.0 * np.pi)
) / 4.0

variance_explicit = (
    0.5 * np.sum(coeff**2)
)

variance_tail = (
    variance_total
    - variance_explicit
)


print(
    "Total oscillatory SD =",
    np.sqrt(variance_total)
)

print(
    "SD represented explicitly =",
    np.sqrt(variance_explicit)
)

print(
    "SD of omitted-zero tail =",
    np.sqrt(variance_tail)
)

print()


# ============================================================
# SIMULATE OSCILLATORY PART OF THEORETICAL DISTRIBUTION
# ============================================================

rng = np.random.default_rng(
    SEED_THEORY_PHASES
)

oscillation = np.zeros(
    N_THEORY,
    dtype=float
)


# Treat low zeros explicitly
for a in coeff:

    theta = rng.uniform(
        0.0,
        2.0 * np.pi,
        N_THEORY
    )

    oscillation += (
        a * np.cos(theta)
    )


# Approximate all omitted high zeros by a Gaussian having
# exactly their remaining variance.
if variance_tail > 0:

    oscillation += rng.normal(
        0.0,
        np.sqrt(variance_tail),
        N_THEORY
    )


# ============================================================
# MIX FINITE-x CENTRING INTO THE THEORY
#
# Sample x from EXACTLY the same weighting distribution as
# used for the empirical data.
#
# Each theoretical realization therefore has centre c(x)
# corresponding to a random x drawn from that weighting.
# ============================================================

y_theory = sample_logx(
    N_THEORY,
    SEED_THEORY_X
)

centre_theory = (
    theoretical_centre_from_logx(
        y_theory
    )
)

z_theory = (
    centre_theory
    + oscillation
)


# ============================================================
# SUMMARY
# ============================================================

print("EMPIRICAL")
print("mean =", z_emp.mean())
print("SD   =", z_emp.std())
print()

print("THEORY")
print("mean =", z_theory.mean())
print("SD   =", z_theory.std())
print()

print(
    "Mean theoretical centre =",
    centre_theory.mean()
)

print(
    "SD of theoretical centre =",
    centre_theory.std()
)


# ============================================================
# COMMON HISTOGRAM BINS
# ============================================================

z_min = min(
    z_emp.min(),
    z_theory.min()
)

z_max = max(
    z_emp.max(),
    z_theory.max()
)

left = (
    np.floor(z_min / BIN_WIDTH)
    * BIN_WIDTH
)

right = (
    np.ceil(z_max / BIN_WIDTH)
    * BIN_WIDTH
)

edges = np.arange(
    left,
    right + 1.5 * BIN_WIDTH,
    BIN_WIDTH
)

centres = (
    0.5
    * (edges[:-1] + edges[1:])
)


# ============================================================
# EMPIRICAL DENSITY
# ============================================================

hist_emp, _ = np.histogram(
    z_emp,
    bins=edges
)

density_emp = (
    hist_emp
    / (N_EMPIRICAL * BIN_WIDTH)
)


# ============================================================
# THEORETICAL DENSITY
# ============================================================

hist_theory, _ = np.histogram(
    z_theory,
    bins=edges
)

density_theory = (
    hist_theory
    / (N_THEORY * BIN_WIDTH)
)


# ============================================================
# PLOT: EMPIRICAL ALONE
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    centres,
    density_emp,
    width=0.92 * BIN_WIDTH
)

plt.xlabel(
    r"$(\pi(x)-\mathrm{Li}(x))/\mathrm{Li}(\sqrt{x})$",
    fontsize=13
)

plt.ylabel(
    "probability density",
    fontsize=13
)


if SAMPLING == "uniform":

    title = (
        "Empirical distribution: "
        r"$\log x$ uniform"
        "\n"
        + rf"${X_MIN:.1e}\leq x\leq{X_MAX:.1e}$"
    )

else:

    title = (
        "Empirical distribution: Gaussian log-window"
        "\n"
        + rf"$\mu=\log(10^9),\ \sigma={SIGMA}$"
    )


plt.title(title)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()
plt.show()


# ============================================================
# PLOT: EMPIRICAL VERSUS THEORY
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    centres,
    density_emp,
    width=0.92 * BIN_WIDTH,
    alpha=0.55,
    label="Empirical"
)

plt.plot(
    centres,
    density_theory,
    linewidth=2.2,
    label="Theory with finite-$x$ centring"
)

plt.xlabel(
    r"$(\pi(x)-\mathrm{Li}(x))/\mathrm{Li}(\sqrt{x})$",
    fontsize=13
)

plt.ylabel(
    "probability density",
    fontsize=13
)


if SAMPLING == "uniform":

    title = (
        "Empirical versus theoretical distribution"
        "\n"
        + rf"$\log x$ uniform, "
          rf"${X_MIN:.1e}\leq x\leq{X_MAX:.1e}$"
    )

else:

    title = (
        "Empirical versus theoretical distribution"
        "\n"
        + rf"Gaussian log-window: "
          rf"$\mu=\log(10^9),\ \sigma={SIGMA}$"
    )


plt.title(title)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

plt.tight_layout()
plt.show()