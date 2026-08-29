from __future__ import annotations

import os

import numpy as np
from scipy.special import expi
from manim import *

def rate_func_log(x1, x2):
    """
    x1 * exp(at), x1 * exp(a) = x2
    """
    a = np.log(x2/x1)
    def rate_func(t):
        print(x1 * np.exp(a*t))
        return (np.exp(a*t) - 1) / (x2/x1-1)

    return rate_func

def li(x: np.ndarray) -> np.ndarray:
    """Principal-value logarithmic integral for x > 1."""
    return expi(np.log(x))


def normalized_error(counting_function: np.ndarray, n_min, n_max_end, bias) -> np.ndarray:
    x = np.arange(n_min, n_max_end + 1, dtype=float)
    li_x = li(x)
    if bias:
        return (
            counting_function[n_min:]
            - li_x
            + li(np.sqrt(x)) / 2
            + li(np.cbrt(x)) / 3
        ) / np.sqrt(li_x)
    else:
        return (
            counting_function[n_min:]
            - li_x
        ) / np.sqrt(li_x)

def empirical_prime_counting(n) -> np.ndarray:
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    is_prime[4::2] = False

    for p in range(3, int(np.sqrt(n)) + 1, 2):
        if is_prime[p]:
            is_prime[p * p :: 2 * p] = False

    return np.cumsum(is_prime, dtype=np.int64)


def cramer_prime_counting(seed, n_min, n_max) -> np.ndarray:
    """One Cramér path, with no residue-class or Chebyshev weighting."""
    rng = np.random.default_rng(seed)
    n = np.arange(n_max + 1)

    probabilities = np.zeros(n_max + 1, dtype=float)
    probabilities[2] = 1.0
    probabilities[3:] = 1.0 / np.log(n[3:])

    selected = rng.random(n_max + 1) < probabilities
    count = np.cumsum(selected)[n_min:]
    mean = np.cumsum(probabilities)[n_min:]
    return (count - mean) / np.sqrt(mean)

def normal_density(z: float) -> float:
    return np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)


def x_sampling_weights(log_weighting, n_min, n_max) -> np.ndarray:
    x = np.arange(n_min, n_max + 1, dtype=float)
    if log_weighting:
        # Uniform measure in log(x) has density proportional to 1/x.
        return 1.0 / x
    return np.ones_like(x)


def bin_indices(values: np.ndarray, bin_min, bin_width, bin_count) -> np.ndarray:
    indices = np.floor((values - bin_min) / bin_width).astype(int)
    if indices.min() < 0 or indices.max() >= bin_count:
        raise ValueError(
            "A value falls outside the histogram range. "
            "Increase BIN_MIN or BIN_MAX."
        )
    return indices


def maximum_prefix_density(indices: np.ndarray, weights: np.ndarray, n_min, n_max_start, bin_width, bin_count) -> float:
    """Fixed y-scale covering every prefix shown in the animation."""
    counts = np.zeros(bin_count, dtype=float)
    total_weight = 0.0
    maximum = 0.0

    for offset, (index, weight) in enumerate(zip(indices, weights)):
        counts[index] += weight
        total_weight += weight
        n_max = n_min + offset
        if n_max >= n_max_start:
            maximum = max(
                maximum,
                counts.max() / (total_weight * bin_width),
            )

    return np.ceil(maximum * 2) / 2

class Histogram:
    def __init__(self, bin_min, bin_max, bin_count, y_scale=1., x_scale=1.):
        bin_width = (bin_max - bin_min) / bin_count
        bar_width = bin_width * x_scale * 0.94
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_width = bin_width
        self.bin_edges = np.linspace(bin_min, bin_max, bin_count+1)
        self.bin_count = bin_count
        self.bars = VGroup(
            *[
                Rectangle(width=bar_width, height=1e-4, stroke_width=0,
                    fill_color=BLUE_C, fill_opacity=0.58,
                )
                for _ in range(self.bin_count)
            ]
        )
        self.y_scale=y_scale
        for i in range(len(self.bars)):
            x_center = (self.bin_edges[i] + self.bin_edges[i + 1]) / 2
            self.bars[i].move_to(x_center * x_scale * RIGHT)
        self.bars.move_to(ORIGIN)


    def update_bars(self, samples, weights):
        counts = np.bincount(
            samples,
            weights=weights,
            minlength=self.bin_count,
        )
        densities = counts / (weights.sum() * self.bin_width)

        for i, (bar, density) in enumerate(zip(self.bars, densities)):
            scene_height = abs(self.y_scale * density)
            bar.stretch_to_fit_height(max(scene_height, 1e-4), about_edge=DOWN)


class HistogramAnimation(Scene):
    log_weighting = True
    n_min = 10
    n_max = 100
    n_max_end = 10_000
    animation_seconds = 14
    bin_min = -3.5
    bin_max = 3.5
    bin_count = 31
    y_max = 1.5
    bias = True

    title = ""

    def normalized_errors(self) -> np.ndarray:
        raise NotImplementedError

    def construct(self) -> None:
        xlen = 11.2
        ylen = 5.5
        x_scale = xlen / (self.bin_max - self.bin_min)
        hist = Histogram(self.bin_min, self.bin_max, self.bin_count, x_scale=x_scale,
                         y_scale=ylen / self.y_max)

        y_tick = 0.5

        axes = Axes(
            x_range=[self.bin_min, self.bin_max, 1],
            y_range=[0, self.y_max, y_tick],
            x_length=xlen,
            y_length=ylen,
            tips=False,
            axis_config={"include_numbers": True, "font_size": 22},
        ).shift(DOWN * 0.45)

        hist.bars.next_to(axes.c2p(self.bin_min, 0), UR, buff=0)

        x_label = axes.get_x_axis_label(MathTex(r"\text{normalized error}"))
        y_label = axes.get_y_axis_label(MathTex(r"\text{density}"))

        tracker = ValueTracker(self.n_max)
        counter_label = MathTex(r"n_{\max}=").scale(0.8)
        counter_value = Integer(self.n_max, group_with_commas=True).scale(0.8)
        counter = VGroup(counter_label, counter_value).arrange(RIGHT, buff=0.12)
        counter.next_to(axes, UP, buff=0.12).align_to(axes, LEFT)

        def update_counter(number: Integer) -> None:
            number.set_value(int(round(tracker.get_value())))
            number.next_to(counter_label, RIGHT, buff=0.12)

        counter_value.add_updater(update_counter)

        errors = self.normalized_errors()
        indices = bin_indices(errors, self.bin_min, hist.bin_width, hist.bin_count)
        weights = x_sampling_weights(self.log_weighting, self.n_min, self.n_max_end)

        def update_bars(group: VGroup) -> None:
            n_max = int(round(tracker.get_value()))
            prefix_length = n_max - self.n_min + 1
            hist.update_bars(indices[:prefix_length], weights[:prefix_length])

        update_bars(hist.bars)
        hist.bars.add_updater(update_bars)

        normal_curve = axes.plot(
            normal_density,
            x_range=[self.bin_min, self.bin_max, 0.02],
            color=ORANGE,
            stroke_width=4,
        )

        normal_key = VGroup(
            Line(LEFT * 0.17, RIGHT * 0.17, color=ORANGE, stroke_width=4),
            MathTex(r"N(0,1)", font_size=26),
        ).arrange(RIGHT, buff=0.12)
        legend = VGroup(normal_key).arrange(RIGHT, buff=0.35)
        legend.next_to(axes, UP, buff=0.12).align_to(axes, RIGHT)

        self.add(
            axes,
            x_label,
            y_label,
            hist.bars,
            normal_curve,
            counter,
            legend,
        )
        self.wait(0.5)
        self.play(
            tracker.animate.set_value(self.n_max_end),
            run_time=self.animation_seconds,
            rate_func=rate_func_log(self.n_max, self.n_max_end),
        )
        self.wait(1)


class EmpiricalPrimesHistogramLog(HistogramAnimation):
    log_weighting = True
    bin_width = 0.05
    n_min = 1000
    n_max = 10_000
    n_max_end = 1_000_000
    bias = True

    def normalized_errors(self) -> np.ndarray:
        count = empirical_prime_counting(self.n_max_end)
        x = np.arange(self.n_min, self.n_max_end + 1, dtype=float)
        li_x = li(x)
        errors = (
                count[self.n_min:]
                - li_x
                + li(np.sqrt(x)) / 2
                + li(np.cbrt(x)) / 3
        ) / np.sqrt(li(x))

        return errors


class CramerPrimesHistogramLog(HistogramAnimation):
    log_weighting = True
    bin_width = 0.2
    n_min = 1000
    n_max = 10_000
    n_max_end = 1_000_000
    bias = False

    def normalized_errors(self) -> np.ndarray:
        errors = cramer_prime_counting(20_260_823, self.n_min, self.n_max_end)
        # errors = normalized_error(count, self.n_min, self.n_max_end, self.bias)
        return errors
