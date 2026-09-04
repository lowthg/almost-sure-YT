from __future__ import annotations

import os

import numpy as np
from scipy.special import expi
from manim import *
import primecountpy as primecount

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
        self.bars.next_to(RIGHT * bin_width * x_scale * (1 - rel_width)/2, UR, buff=0)
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


class EmpiricalCount:
    def __init__(self, n_max=1000, n_min = 1, nstep=100):
        """One Cramér path, with no residue-class or Chebyshev weighting."""
        is_prime = np.ones(n_max + 1, dtype=bool)
        is_prime[:2] = False
        is_prime[4::2] = False

        for p in range(3, int(np.sqrt(n_max)) + 1, 2):
            if is_prime[p]:
                is_prime[p * p:: 2 * p] = False

        count = np.cumsum(is_prime, dtype=np.int64)[1:]

        n = np.arange(1, n_max+1)
        x = n[1:]

        li_x = li(x)
        mean = li(x) - li(np.sqrt(x))/2 - li(np.cbrt(x))/3

        mean = np.concatenate([[0], mean])
        errors = (count - mean) / np.concatenate([[1], np.sqrt(li_x)])

        self.count = errors
        self.weights = x_sampling_weights(True, 1, n_max)
        self.x0 = float(n_min)
        self.n_max = n_max
        self.nstep = nstep

    def initial_samples(self, x_min):
        n_min = int(math.ceil(x_min))
        assert 1 <= n_min < self.n_max
        return self.count[n_min-1:], self.weights[n_min-1:], np.arange(n_min, self.n_max+1)

    def new_samples(self, x, do_xvals=False):
        count = []
        weights = []
        xvals = []
        if self.x0 < self.n_max:
            n_max = min(int(round(x)) - 1, len(self.count))
            n0 = int(round(self.x0)) - 1
            count.append(self.count[n0:n_max])
            weights.append(self.weights[n0:n_max])
            assert n_max <= len(self.count)
            if do_xvals: xvals.append(np.arange(n0+1, n_max+1))
            print('len', len(xvals[0]), len(weights[0]), len(count[0]))

        # assert self.n_max >= x
        if self.n_max < x:
            x0 = max(self.x0, self.n_max)
            xvec0 = np.exp(np.linspace(np.log(x0), np.log(x), self.nstep+1))
            xvec = xvec0[1:]
            livec = li(xvec)
            mean = livec - li(np.sqrt(xvec))/2 - li(np.cbrt(xvec))/3

            count_vec = np.fromiter((primecount.prime_pi(int(x)) for x in xvec), dtype=np.int64)

            errors = mean - count_vec

            weight_vec = np.log(xvec0)

            count_norm = errors / np.sqrt(livec)
            count.append(count_norm)
            weights.append(weight_vec[1:] - weight_vec[:-1])
            if do_xvals: xvals.append(xvec)
            print('len', len(xvec), len(weight_vec)-1, len(count_norm))

        self.x0 = x

        if do_xvals:
            return np.concatenate(count), np.concatenate(weights), np.concatenate(xvals)
        return np.concatenate(count), np.concatenate(weights)


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
        hist.bars.shift(axes.c2p(bin_min, 0)).set_z_index(1)
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
        xlen = 12
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
        ).set_z_index(5)
        tick = Line(ORIGIN, DOWN*0.1, stroke_width=4, stroke_color=WHITE).set_z_index(5)
        ticks = VGroup(*[tick.copy().shift(axes.c2p(i)) for i in [-2, -1, 0, 1, 2]]).set_z_index(5)
        xlabels = VGroup(*[MathTex('{}'.format(i), font_size=40)[0] for i in [-2, -1, 0, 1, 2]])
        for t, l in zip(ticks[:], xlabels): mh.align_sub(l, l[-1], t, DOWN, buff=0.1)

        ax2 = Axes(x_range=[0,1], y_range=[bin_min,bin_max], x_length=10, y_length=xlen).rotate(PI/2).set_opacity(0)
        rect0 = SurroundingRectangle(ax2, buff=0, stroke_width=0, stroke_opacity=0, fill_color=GREY,
                                     fill_opacity=0.)
        rect_txt = (Tex(r'\sf normalized error', color=col_txt, stroke_width=2, font_size=70)
                    .move_to(ax2.c2p(0.5,bin_max*0.9)).set_opacity(0).rotate(90*DEGREES).shift(OUT*0.05))
        rect1 = VGroup(rect0, rect_txt.set_z_index(5))
        hist.bars.shift(axes.c2p(bin_min, 0))
        axes.y_axis.set_opacity(0)

        n_max = 1000
        tracker = ValueTracker(n_max)
        bar_shift_val = ValueTracker(0.)
        counter_label = MathTex(r"n=", font_size=50, stroke_width=2).rotate(90*DEGREES, RIGHT)
        counter_label[0][0].set_color(col_var)
        counter_label.move_to(axes.c2p(-1.4, 0.2))
        counter_value = always_redraw(
            lambda: Integer(round(tracker.get_value()), color=col_num, group_with_commas=True, stroke_width=2)
            .rotate(90 * DEGREES, axis=RIGHT).next_to(counter_label, RIGHT, buff=0.12).set_z_index(10)
        )

        VGroup(axes, hist.bars, ticks, xlabels).rotate(90*DEGREES, RIGHT, about_point=ORIGIN)
        VGroup(ax2, rect1).next_to(axes.x_axis, DOWN, buff=0)
        self.camera.set_phi(90*DEGREES)

        normal_curve = axes.plot(normal_density,
            x_range=[bin_min, bin_max, 0.02],
            color=ORANGE, stroke_width=4).set_z_index(4)
        area = axes.get_area(normal_curve, (bin_min, bin_max), color=ORANGE, opacity=0.2).set_z_index(4)

        bar_shift = 5*UP + 2*LEFT
        def update_bars():
            x = tracker.get_value()
            errors, weights = self.counter.new_samples(x)
            hist.add_data(errors, weights)
            res = hist.bars.copy().shift(bar_shift_val.get_value()*bar_shift)
            xvals = np.linspace(10, x, 800)
            yvals = np.interp(xvals, self.counter.xvals, self.counter.values)
            op = bar_shift_val.get_value()
            plt = ax2.plot_line_graph(xvals/xvals[-1], -yvals, add_vertex_dots=False, stroke_width=4,
                                      stroke_color=BLUE, stroke_opacity=op).set_z_index(10)
            return VGroup(res, plt['line_graph'])

        bars = always_redraw(update_bars)

        self.add(axes.x_axis, counter_label, counter_value, normal_curve, area, bars, ticks, xlabels)
        self.add(ax2)

        rate_func=rate_func_log(n_max, n_max_end)
        self.play(
            tracker.animate(run_time=12, rate_func=rate_func).set_value(n_max_end),

            Succession(Wait(2),
                       AnimationGroup(
                           self.camera.phi_tracker.animate.set_value(70 * DEGREES), # view from above
                           self.camera.theta_tracker.animate.set_value(-60 * DEGREES),
                           bar_shift_val.animate.set_value(1),
                           VGroup(axes, normal_curve, area, counter_label, ax2).animate.shift(bar_shift),
                           VGroup(ticks, xlabels).animate.shift(bar_shift).set_opacity(0),
                           rect1.animate.shift(bar_shift).set_fill(opacity=0.3)
                           # run_time=2.
                       )),
        )
        # self.remove(bars)
        # bars = update_bars()
        self.play(
            self.camera.phi_tracker.animate.set_value(90 * DEGREES),  # view from above
            self.camera.theta_tracker.animate.set_value(-90 * DEGREES),
            VGroup(axes, normal_curve, area, counter_label, bars[0]).animate.shift(-bar_shift),
            VGroup(rect1, bars[1]).animate.shift(-bar_shift).set_opacity(0),
            VGroup(ticks, xlabels).animate.shift(-bar_shift).set_opacity(1),
        )
        self.wait(1)


class CramerHistogramLog(Scene):
    animation_seconds = 14
    y_max = 0.7
    bin_count = 11
    n_max = 1500
    n_max_end = 100_000_000_000_000
    bin_min = -2.5
    bin_max = 2.5
    counter = CramerCount(seed=1, n_min=500, n_max=1_000_000, nstep=500)
    rel_width = 0.9

    def normalized_errors(self) -> np.ndarray:
        raise NotImplementedError

    def construct(self) -> None:
        xlen = 12
        ylen = 5.5
        x_scale = xlen / (self.bin_max - self.bin_min)
        hist = Histogram(self.bin_min, self.bin_max, self.bin_count, x_scale=x_scale,
                         y_scale=ylen / self.y_max, rel_width=self.rel_width)

        axes = Axes(x_range=[self.bin_min, self.bin_max], y_range=[0, self.y_max],
            x_length=xlen, y_length=ylen, tips=False,
            axis_config={"include_ticks": False},
        ).shift(DOWN * 0.45).set_z_index(5)
        tick = Line(ORIGIN, DOWN*0.1, stroke_width=4, stroke_color=WHITE).set_z_index(5)
        ticks = VGroup(*[tick.copy().shift(axes.c2p(i)) for i in [-2, -1, 0, 1, 2]]).set_z_index(5)
        xlabels = VGroup(*[MathTex('{}'.format(i), font_size=40)[0] for i in [-2, -1, 0, 1, 2]])
        for t, l in zip(ticks[:], xlabels): mh.align_sub(l, l[-1], t, DOWN, buff=0.1)

        hist.bars.shift(axes.c2p(self.bin_min, 0))

        tracker = ValueTracker(self.n_max)
        counter_label = MathTex(r"n=", font_size=50, stroke_width=2)
        counter_label[0][0].set_color(col_var)
        counter_label.move_to(axes.c2p(0.8 * self.bin_min, 5/7 * self.y_max))
        counter_value = always_redraw(
            lambda: Integer(round(tracker.get_value()), color=col_num, group_with_commas=True, stroke_width=2)
            .next_to(counter_label[0][-1], RIGHT, buff=0.15).set_z_index(10)
        )

        def update_bars(group: VGroup) -> None:
            errors, weights = self.counter.new_samples(tracker.get_value())
            hist.add_data(errors, weights)
            # print('weights', hist.bar_weights)

        # hist.bars.set_z_index(6)

        hist.bars.add_updater(update_bars)

        normal_curve = axes.plot(normal_density,
            x_range=[self.bin_min, self.bin_max, 0.02],
            color=ORANGE, stroke_width=4).set_z_index(4)
        area = axes.get_area(normal_curve, (self.bin_min, self.bin_max), color=ORANGE, opacity=0.2).set_z_index(4)

        self.add(axes.x_axis, ticks, xlabels, hist.bars, normal_curve, area, counter_label, counter_value)

        self.play(
            tracker.animate.set_value(self.n_max_end),
            run_time=self.animation_seconds,
            rate_func=rate_func_log(self.n_max, self.n_max_end),
        )
        # print(hist.bar_weights/hist.bar_weights.sum())
        self.wait(1)


class EmpiricalHistogramLog(CramerHistogramLog):
    animation_seconds = 14
    y_max = 4.6
    bin_count = 31
    n_max = 1500
    n_max_end = 100_000_000
    bin_min = -1.5
    bin_max = 1.5
    rel_width = 0.8
    counter = EmpiricalCount(n_min=500, n_max=1_000_000, nstep=500)

class EmpiricalVarPlot(Scene):
    def construct(self):
        axes = Axes(x_range=[0, 1], y_range=[0, 1.1],
            x_length=12, y_length=6, tips=False,
            axis_config={"include_ticks": False, "stroke_width": 4},
                    y_axis_config={'include_tip': True,
                                   "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                                   "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                                   }
        ).set_z_index(5)

        labelx = MathTex(r'N', stroke_width=1.5, font_size=40, color=col_x)
        labely = MathTex(r'\mathbb E[Z^2]', stroke_width=1.5, font_size=40)
        labely[0][0].set_color(col_WVD)
        labely[0][2:4].set_color(col_p)
        labely.next_to(axes.y_axis.get_end(), RIGHT, buff=0.2)
        labelx.next_to(axes.x_axis.get_end(), UR, buff=0.14)
        title = Tex(r'\sf Expected Square Error ', r'$(N_0=500)$', stroke_width=2, font_size=60)
        title[0].set_color(col_txt)
        title[1][1:3].set_color(col_x)
        title[1][4:7].set_color(col_num)
        title.to_edge(UP, buff=0.4).shift(RIGHT)

        # mh.align_sub(title, title[0], mh.pos(UP), DOWN, buff=0.2)

        eq1 = Tex(r'\sf Cram\'er prediction', color=ORANGE, font_size=50, stroke_width=2)
        eq1.next_to(axes.c2p(0.5, 1), DOWN, buff=0.1)
        eq2 = Tex(r'\sf Empirical value', color=BLUE, font_size=50, stroke_width=2)
        eq2.move_to(axes.c2p(0.5, 0.2))
        arr1 = Arrow(eq2[0][-1].get_right()+RIGHT*0.1, axes.c2p(0.7, 0.02), buff=0, color=BLUE, stroke_width=8, path_arc=-PI/6,
                     max_tip_length_to_length_ratio=10, max_stroke_width_to_length_ratio=20)

        xtickvals = np.log([1e3, 1e4, 1e5, 1e6, 1e7, 1e9])
        xtickstrs = ['1\,000', r'10\,000', r'100\,000', r'\!1\,000\,000', r'10\,000\,000', r'1\,000\,000\,000']
        xtickvals1 = (xtickvals - xtickvals[0]) / (xtickvals[3] - xtickvals[0])
        xtickvals2 = (xtickvals - xtickvals[0]) / (xtickvals[-1] - xtickvals[0])
        xticks1 = mh.get_xticks(axes, vals=xtickvals1, label_color=col_num, strs=xtickstrs, buff=0.4)
        xticks2 = mh.get_xticks(axes, vals=xtickvals2, label_color=col_num, strs=xtickstrs, buff=0.4)
        VGroup(xticks2[1], xticks2[3]).set_opacity(0)
        xticks1[3][1].to_edge(RIGHT, buff=0.2)
        xticks2[-1][1].to_edge(RIGHT, buff=0.2)

        ytickvals = np.array([0., 1e-4, 0.001, 0.01, 0.1, 1.])
        ystrs = [r'0', r'10^{-4}', r'10^{-3}', r'10^{-2}', r'.1', r'1']
        ytickvals2 = np.log(ytickvals[1:] * 1e5) / np.log(1e5)
        tick_width = (axes.get_left() - mh.pos(LEFT))[0] - 0.4
        yticks1 = mh.get_yticks(axes, vals=ytickvals, strs=ystrs, max_width=tick_width, label_color=col_num)
        yticks2 = mh.get_yticks(axes, vals=ytickvals2, strs=ystrs[1:], max_width=tick_width, label_color=col_num)
        yticks1[1:-1].set_opacity(0)
        ylines1 = VGroup(*[Line(axes.c2p(0, y), axes.c2p(1, y), stroke_width=3, stroke_color=GREY, stroke_opacity=0)
                           for y in ytickvals[1:-1]])
        ylines2 = VGroup([Line(axes.c2p(0, y), axes.c2p(1, y), stroke_width=3, stroke_color=GREY, stroke_opacity=0.5)
                          for y in ytickvals2[:-1]])

        line1 = DashedLine(axes.c2p(0,1), axes.c2p(1,1), stroke_color=ORANGE, stroke_width=6, dash_length=0.15, dashed_ratio=0.7).set_z_index(2)

        counter = EmpiricalCount(n_min=500, n_max=1_000_000, nstep=100_000)

        samples, weights, xvals = counter.new_samples(1e9, do_xvals=True)
        cumweights = np.cumsum(weights)
        samples_exp2 = np.cumsum(samples * samples * weights) / cumweights

        xvals_log = np.log(xvals)
        n1 = 1000
        xplot_1 = np.linspace(np.log(1000), np.log(1e6), n1)
        xplot_2 = np.linspace(np.log(1e6), np.log(1e9), 1000)[1:]
        xplot = np.concatenate([xplot_1, xplot_2])
        yplot = np.interp(xplot, xvals_log, samples_exp2)
        xplot_scale = (xplot - xplot_1[0]) / (xplot_1[-1] - xplot_1[0])

        yplotlog = np.log(yplot * 1e5) / np.log(1e5)

        plt1 = axes.plot_line_graph(xplot_scale[:n1], yplot[:n1], add_vertex_dots=False, stroke_color=BLUE, stroke_width=8)
        plt2 = axes.plot_line_graph(xplot_scale[:n1], yplotlog[:n1], add_vertex_dots=False, stroke_color=BLUE, stroke_width=8)
        plt3 = axes.plot_line_graph(xplot_scale, yplotlog, add_vertex_dots=False, stroke_color=BLUE, stroke_width=8)
        box1 = Rectangle(width=3, height=2, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=1)
        box1.next_to(plt2.get_right(), RIGHT, buff=0).set_z_index(2)

        tracker = ValueTracker(0.)
        x0 = xplot_scale[n1-1]
        x1 = xplot_scale[-1]

        def get_tracker_obj():
            t = tracker.get_value()
            x2 = x1 * (1-t) + x0 * t
            x3 = x0 / x2 * x1
            y3 = np.interp(x3, xplot_scale, yplot)
            y3log = np.log(y3*1e5) / np.log(1e5)
            # y = yplot[n1-1]
            # ylog = np.log(y * 1e5) / np.log(1e5)
            val_right = DecimalNumber(y3, 4, font_size=40, stroke_width=1.5, color=BLUE)
            val_right[1:].next_to(axes.c2p(1, y3log), RIGHT, buff=0.1).set_z_index(3)
            return val_right[1:]

        print('yvals', yplot[0], yplot[-1])

        self.add(axes, xticks1, yticks1, ylines1, labely, labelx, title)

        self.play(FadeIn(eq1), Create(line1, rate_func=linear))
        self.wait(0.1)

        self.play(Create(plt1, rate_func=linear, run_time=2),
                  Succession(Wait(1), FadeIn(eq2, arr1)))
        self.wait(0.1)
        self.play(Succession(Wait(0.5),
                             AnimationGroup(mh.rtransform(plt1, plt2, yticks1[1:], yticks2[:], ylines1, ylines2),
                             eq2.animate.move_to(axes.c2p(0, 0.61), coor_mask=UP))
                             ),
                  FadeOut(yticks1[0]),
                  FadeOut(arr1),
                  )
        self.remove(plt2)
        self.add(plt3, box1)
        tracker_obj = always_redraw(get_tracker_obj)
        self.play(FadeIn(tracker_obj))

        xplot_scale2 = (xplot - xplot[0]) / (xplot[-1] - xplot[0])
        plt4 = axes.plot_line_graph(xplot_scale2, yplotlog, add_vertex_dots=False, stroke_color=BLUE, stroke_width=8)

        self.play(mh.rtransform(plt3, plt4, xticks1, xticks2),
                  tracker.animate.set_value(1.), run_time=3)



        self.wait()
