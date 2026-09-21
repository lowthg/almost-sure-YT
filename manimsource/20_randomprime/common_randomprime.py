import numpy as np
from manim import always_redraw, VGroup, ValueTracker, ReplacementTransform, linear, ORIGIN

def build_prime_count(limit=1_000_000):
    # is_prime[n] is 1 if n is prime
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    is_prime[4::2] = False

    for p in range(3, int(np.sqrt(limit)) + 1, 2):
        if is_prime[p]:
            is_prime[p * p:: 2 * p] = False

    count = np.cumsum(is_prime, dtype=np.int64)

    return count

def prime_counting_step_vectors(pi):
    pi = np.asarray(pi)

    # n is prime exactly when pi(n) - pi(n-1) = 1
    primes = np.flatnonzero(np.diff(pi) == 1) + 1

    x = np.empty(2 * len(primes) + 1, dtype=np.int64)
    y = np.empty(2 * len(primes) + 1, dtype=np.int64)

    x[0] = 0
    y[0] = 0

    # Include each prime twice
    x[1:] = np.repeat(primes, 2)

    # For each prime n, include pi(n-1), then pi(n)
    y[1::2] = pi[primes - 1]
    y[2::2] = pi[primes]

    return x, y

def prime_counting_vectors(pi, max_n):
    pi_n = np.asarray(pi[:max_n + 1])
    integers = np.arange(max_n + 1)

    # Treat pi(-1) as zero
    pi_previous = np.concatenate(([0], pi_n[:-1]))

    # Primes need two points; other integers need one
    is_prime = pi_n != pi_previous
    point_counts = 1 + is_prime.astype(int)

    x = np.repeat(integers, point_counts)
    y = np.empty(point_counts.sum(), dtype=pi_n.dtype)

    # Index of the first point for each integer
    starts = np.cumsum(point_counts) - point_counts

    # First point: (n, pi(n-1))
    y[starts] = pi_previous

    # Second point for primes: (n, pi(n))
    y[starts[is_prime] + 1] = pi_n[is_prime]

    return x, y

def animation_scale_redraw(x_scale=1., y_scale=1., obj_scale=None, obj1x=None, obj2x=None, obj1y=None, obj2y=None,
                           origin=ORIGIN, y_scale_func=None, x_scale_func=None, obj2_scale=None):
    anim_tracker = ValueTracker(0.)
    if obj1x is not None:
        animationx = ReplacementTransform(obj1x, obj2x, rate_func=linear)
        animationx.begin()
    if obj1y is not None:
        animationy = ReplacementTransform(obj1y, obj2y, rate_func=linear)
        animationy.begin()
    if obj2_scale is not None:
        animation = ReplacementTransform(obj_scale, obj2_scale, rate_func=linear)
        animation.begin()

    def obj_func1():
        res = VGroup()
        u = anim_tracker.get_value()
        scalex = np.exp(np.log(x_scale) * u) if x_scale_func is None else x_scale_func(u)
        scaley = np.exp(np.log(y_scale) * u) if y_scale_func is None else y_scale_func(u)
        scale = np.array([scalex, scaley, 1])
        if obj2_scale is not None:
            animation.interpolate(u)
        res.add(obj_scale.copy().apply_points_function_about_point(lambda p: p * scale, origin))
        if obj1x is not None:
            tx = (1 - scalex) / (1 - x_scale)
            animationx.interpolate(tx)
            res.add(obj1x.copy())
        if obj1y is not None:
            ty = (1 - scaley) / (1 - y_scale) if abs(y_scale - 1.) > 1e-6 else u
            animationy.interpolate(ty)
            res.add(obj1y.copy())
        return res

    return anim_tracker, always_redraw(obj_func1)
