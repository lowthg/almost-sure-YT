from __future__ import annotations

import os

import numpy as np
from scipy.special import expi
from manim import *

import sys
sys.path.append('../../')
import manimhelper as mh
from common.wigner import *

col_txt = ManimColor( r'#FFAC2B')

def rate_func_log(x1, x2):
    """
    x1 * exp(at), x1 * exp(a) = x2
    """
    a = np.log(x2/x1)
    b = x2/x1 - 1
    return lambda t: (np.exp(a*t) - 1) / b

def li(x: np.ndarray) -> np.ndarray:
    """Principal-value logarithmic integral for x > 1."""
    return expi(np.log(x))


def normalized_error(counting_function: np.ndarray, n_min, n_max_end, bias) -> np.ndarray:
    x = np.arange(n_min, n_max_end + 1, dtype=float)
    li_x = expi(np.log(x))
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
    indices = np.floor((values - bin_min) / bin_width).astype(int).clip(-1, bin_count)
    # if indices.min() < 0 or indices.max() >= bin_count:
    #     raise ValueError(
    #         "A value falls outside the histogram range. "
    #         "Increase BIN_MIN or BIN_MAX."
    #     )
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
    def __init__(self, bin_min, bin_max, bin_count, y_scale=1., x_scale=1., use_depth=False, rel_width=0.9):
        bin_width = (bin_max - bin_min) / bin_count
        bar_width = bin_width * x_scale * rel_width
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_width = bin_width
        self.bin_edges = np.linspace(bin_min, bin_max, bin_count+1)
        self.bin_count = bin_count
        self.bars = VGroup(
            *[
                Rectangle(width=bar_width, height=1e-4, stroke_width=2, stroke_color=BLUE, stroke_opacity=1,
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
        self.bar_weights = np.zeros(self.bin_count+2)
        self.use_depth = use_depth

    def update_bars(self):
        densities = self.bar_weights[1:-1] / (self.bar_weights.sum() * self.bin_width)

        for i, (bar, density) in enumerate(zip(self.bars, densities)):
            scene_height = abs(self.y_scale * density)
            if self.use_depth:
                bar.stretch_to_fit_depth(max(scene_height, 1e-4), about_edge=IN)
            else:
                bar.stretch_to_fit_height(max(scene_height, 1e-4), about_edge=DOWN)

    def set_data(self, samples, weights=None):
        indices = bin_indices(samples, self.bin_min, self.bin_width, self.bin_count)
        self.bar_weights[:] = np.bincount(indices+1, weights=weights, minlength=self.bin_count+2)
        self.update_bars()

    def add_data(self, samples, weights=None):
        indices = bin_indices(samples, self.bin_min, self.bin_width, self.bin_count)
        self.bar_weights += np.bincount(indices+1, weights=weights, minlength=self.bin_count+2)
        self.update_bars()



class CramerCount:
    def __init__(self, seed=1, n_max=1000, n_min = 1, nstep=100, uniform=False, store=False):
        """One Cramér path, with no residue-class or Chebyshev weighting."""
        self.rng = rng = np.random.default_rng(seed)
        n = np.arange(1, n_max+1)

        probabilities = np.zeros(n_max, dtype=float)
        probabilities[1] = 1.0
        probabilities[2:] = 1.0 / np.log(n[2:])

        selected = rng.random(n_max) < probabilities
        count = np.cumsum(selected)
        mean = np.cumsum(probabilities)
        self.count = (count - mean) / np.sqrt(mean.clip(1))
        self.weights = x_sampling_weights(not uniform, 1, n_max)
        self.x0 = float(n_min)
        self.n_max = n_max
        self.count0 = count[-1] - mean[-1]
        self.mean0 = mean[-1]
        self.nstep = nstep
        self.uniform = uniform
        self.values = self.xvals = None
        if store:
            self.values = self.count[::10]
            self.xvals = n[::10]
            assert len(self.values) == len(self.xvals)

    def new_samples(self, x):
        count = []
        weights = []
        if self.x0 < self.n_max:
            n_max = int(round(x)) - 1
            n0 = int(round(self.x0)) - 1
            count.append(self.count[n0:n_max])
            weights.append(self.weights[n0:n_max])
        if self.n_max < x:
            x0 = max(self.x0, self.n_max)
            xvec = np.exp(np.linspace(np.log(x0), np.log(x), self.nstep+1))
            livec = li(xvec)
            li_diff = livec[1:] - livec[:-1]
            count_diffs = self.rng.normal(loc=0, scale=np.sqrt(li_diff))
            count_vec = np.cumsum(count_diffs) + self.count0
            means = livec[1:] - livec[0] + self.mean0
            weight_vec = xvec if self.uniform else np.log(xvec)
            self.count0 = count_vec[-1]
            self.mean0 = means[-1]
            count_norm = count_vec / np.sqrt(means)
            count.append(count_norm)
            weights.append(weight_vec[1:] - weight_vec[:-1])
            if self.xvals is not None:
                self.xvals = np.concatenate((self.xvals, xvec[1:][5::10]))
                self.values = np.concatenate((self.values, count_norm[5::10]))
                assert len(self.values) == len(self.xvals)

        self.x0 = x
        return np.concatenate(count), np.concatenate(weights)


class CramerPrimesHistogramLog(Scene):
    animation_seconds = 14
    y_max = 0.7
    log_weighting = True
    bin_count = 15
    n_min = 1000
    n_max = 10_000
    n_max_end = 10_000_000_000_000
    bias = False
    bin_min = -3.
    bin_max = 3.
    counter = CramerCount(seed=1, n_min=1000, n_max=1_000_000, nstep=1000)

    def normalized_errors(self) -> np.ndarray:
        raise NotImplementedError

    def construct(self) -> None:
        xlen = 11.2
        ylen = 5.5
        x_scale = xlen / (self.bin_max - self.bin_min)
        hist = Histogram(self.bin_min, self.bin_max, self.bin_count, x_scale=x_scale,
                         y_scale=ylen / self.y_max)

        y_tick = 0.5

        axes = Axes(x_range=[self.bin_min, self.bin_max, 1], y_range=[0, self.y_max, y_tick],
            x_length=xlen, y_length=ylen, tips=False,
            axis_config={"include_numbers": True, "font_size": 35},
        ).shift(DOWN * 0.45)

        hist.bars.next_to(axes.c2p(self.bin_min, 0), UR, buff=0)

        x_label = axes.get_x_axis_label(MathTex(r"\text{normalized error}"))
        y_label = axes.get_y_axis_label(MathTex(r"\text{density}"))

        tracker = ValueTracker(self.n_max)
        counter_label = MathTex(r"n_{\max}=").scale(0.8)
        counter_value = Integer(self.n_max, group_with_commas=True).scale(0.8)
        counter_obj = VGroup(counter_label, counter_value).arrange(RIGHT, buff=0.12)
        counter_obj.next_to(axes, UP, buff=0.12).align_to(axes, LEFT)

        def update_counter(number: Integer) -> None:
            number.set_value(int(round(tracker.get_value())))
            number.next_to(counter_label, RIGHT, buff=0.12)

        counter_value.add_updater(update_counter)

        def update_bars(group: VGroup) -> None:
            errors, weights = self.counter.new_samples(tracker.get_value())
            hist.add_data(errors, weights)

        hist.bars.add_updater(update_bars)

        normal_curve = axes.plot(normal_density,
            x_range=[self.bin_min, self.bin_max, 0.02],
            color=ORANGE, stroke_width=4)

        normal_key = VGroup(
            Line(LEFT * 0.17, RIGHT * 0.17, color=ORANGE, stroke_width=4),
            MathTex(r"N(0,1)", font_size=26),
        ).arrange(RIGHT, buff=0.12)
        legend = VGroup(normal_key).arrange(RIGHT, buff=0.35)
        legend.next_to(axes, UP, buff=0.12).align_to(axes, RIGHT)

        self.add(axes.x_axis, x_label, y_label, hist.bars, normal_curve, counter_obj,legend)

        self.play(
            tracker.animate.set_value(self.n_max_end),
            run_time=self.animation_seconds,
            rate_func=rate_func_log(self.n_max, self.n_max_end),
        )
        print(hist.bar_weights/hist.bar_weights.sum())
        self.wait(1)

class NormalUniform(Scene):
    bgcol = GREY
    trcol = BLACK

    def __init__(self, *args, **kwargs):
        config.background_color = self.trcol if config.transparent else self.bgcol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        xlen = 5.
        ylen = 2.
        bin_max = 3.
        bin_min = -3.
        x_scale = xlen / (bin_max - bin_min)
        y_max = 0.9
        rng = np.random.default_rng(2)

        hist = Histogram(bin_min, bin_max, bin_count=11, x_scale=x_scale,
                         y_scale=ylen / y_max * 5.5, rel_width=0.8)
        axes = Axes(x_range=[bin_min, bin_max], y_range=[0, y_max],
            x_length=xlen, y_length=ylen, tips=False,
            axis_config={"include_ticks": False},
        ).set_z_index(2)
        axes.y_axis.set_opacity(0)
        box = SurroundingRectangle(axes, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                   buff=0.2, corner_radius=0.15)
        hist.bars.next_to(axes.c2p(bin_min, 0), UR, buff=0).set_z_index(1)
        hist.set_data((hist.bin_edges[1:] + hist.bin_edges[:-1])/2, np.ones(hist.bin_count))

        normal_curve = axes.plot(normal_density,
            x_range=[bin_min, bin_max, 0.02],
            color=ORANGE, stroke_width=4).set_z_index(5)
        area = axes.get_area(normal_curve, (bin_min, bin_max), color=ORANGE, opacity=0.2).set_z_index(4)

        self.add(axes, box)
        bars = hist.bars.copy()
        self.play(FadeIn(bars))
        self.wait(0.1)
        hist.y_scale /= 5.5
        hist.set_data(rng.normal(loc=0, scale=1, size=10))
        self.play(mh.transform(bars, hist.bars.copy()))
        self.wait(0.1)
        for _ in range(6):
            hist.add_data(rng.normal(loc=0, scale=1, size=1))
            self.play(mh.transform(bars, hist.bars.copy(), run_time=0.5))

        self.play(Create(normal_curve, rate_func=linear),
                  Succession(Wait(0.5), FadeIn(area)))

        sample_tracker = ValueTracker(0)
        n0 = [0.]

        self.remove(bars)
        self.add(hist.bars)

        def update_bars(obj):
            n1 = sample_tracker.get_value()
            n = int(n1) - int(n0[0])
            n0[0] = n1
            hist.add_data(rng.normal(loc=0, scale=1, size=n))

        hist.bars.add_updater(update_bars)

        self.play(sample_tracker.animate.set_value(1500), run_time=15, rate_func=linear)



class CramerHistogramUniform(ThreeDScene):
    counter = CramerCount(seed=1, n_min=2, n_max=1_000_000, nstep=1000, uniform=True, store=True)

    def construct(self):
        xlen = 11.2
        ylen = 5.5
        bin_max = 2.5
        bin_min = -2.5
        x_scale = xlen / (bin_max - bin_min)
        y_max = 1.5
        n_max_end=1_000_000_000
        hist = Histogram(bin_min, bin_max, bin_count=21, x_scale=x_scale,
                         y_scale=ylen / y_max, use_depth=True)
        axes = Axes(x_range=[bin_min, bin_max], y_range=[0, y_max],
            x_length=xlen, y_length=ylen, tips=False,
            axis_config={"include_numbers": False, "include_ticks": False},
        )
        ax2 = Axes(x_range=[0,1], y_range=[bin_min,bin_max], x_length=10, y_length=xlen).rotate(PI/2).set_opacity(0)
        rect0 = SurroundingRectangle(ax2, buff=0, stroke_width=0, stroke_opacity=0, fill_color=GREY,
                                     fill_opacity=0.)
        rect_txt = (Tex(r'\sf normalized error', color=col_txt, stroke_width=2, font_size=70)
                    .move_to(ax2.c2p(0.5,bin_max*0.9)).set_opacity(0).rotate(90*DEGREES).shift(OUT*0.05))
        rect1 = VGroup(rect0, rect_txt.set_z_index(5))
        # print('included', axes.x_axis.numbers_to_include)
        hist.bars.next_to(axes.c2p(bin_min, 0), UR, buff=0)
        axes.y_axis.set_opacity(0)

        n_max = 1000
        tracker = ValueTracker(n_max)
        bar_shift_val = ValueTracker(0.)
        counter_label = MathTex(r"n=", font_size=50, stroke_width=2).rotate(90*DEGREES, RIGHT)
        counter_label[0][0].set_color(col_var)
        counter_label.move_to(axes.c2p(-1.4, 0.2))
        counter_value = always_redraw(
            lambda: Integer(round(tracker.get_value()), color=col_num, group_with_commas=True)
            .rotate(90 * DEGREES, axis=RIGHT).next_to(counter_label, RIGHT, buff=0.12)
        )

        VGroup(axes, hist.bars).rotate(90*DEGREES, RIGHT)
        VGroup(ax2, rect1).next_to(axes.x_axis, DOWN, buff=0)
        self.camera.set_phi(90*DEGREES)

        normal_curve = axes.plot(normal_density,
            x_range=[bin_min, bin_max, 0.02],
            color=ORANGE, stroke_width=4)
        normal_curve.set_fill(color=ORANGE, opacity=0.2)

        bar_shift = 5*UP + 2*LEFT
        def update_bars():
            x = tracker.get_value()
            errors, weights = self.counter.new_samples(x)
            hist.add_data(errors, weights)
            res = hist.bars.copy().shift(bar_shift_val.get_value()*bar_shift)
            xvals = np.linspace(10, x, 400)
            yvals = np.interp(xvals, self.counter.xvals, self.counter.values)
            op = bar_shift_val.get_value()
            plt = ax2.plot_line_graph(xvals/xvals[-1], -yvals, add_vertex_dots=False, stroke_width=6,
                                      stroke_color=BLUE, stroke_opacity=op).set_z_index(10)
            return VGroup(res, plt['line_graph'])

        bars = always_redraw(update_bars)

        self.add(axes.x_axis, counter_label, counter_value, normal_curve, bars)
        self.add(ax2)

        rate_func=rate_func_log(n_max, n_max_end)
        self.play(
            tracker.animate(run_time=12, rate_func=rate_func).set_value(n_max_end),
            # counter_value.animate(run_time=8, rate_func=rate_func).set_value(n_max_end),

            Succession(Wait(2),
                       AnimationGroup(
                           self.camera.phi_tracker.animate.set_value(70 * DEGREES), # view from above
                           self.camera.theta_tracker.animate.set_value(-60 * DEGREES),
                           bar_shift_val.animate.set_value(1),
                           VGroup(axes, normal_curve, counter_label, ax2).animate.shift(bar_shift),
                           rect1.animate.shift(bar_shift).set_fill(opacity=0.3)
                           # run_time=2.
                       )),
        )
        print(hist.bar_weights/hist.bar_weights.sum())
        self.wait(1)


class EmpiricalPrimesHistogramLog(CramerPrimesHistogramLog):
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

