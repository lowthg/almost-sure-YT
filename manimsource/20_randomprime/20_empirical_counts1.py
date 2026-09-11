import numpy as np
from mpmath import nzeros
from scipy.special import expi
from manim import *
import primecountpy as primecount

import sys

sys.path.append('../../')
import manimhelper as mh
from common.wigner import *
from common_randomprime import *
import mpmath as mp
from pathlib import Path

col_txt = ManimColor( r'#FFAC2B')

def compose(*functions):
    def result(x):
        for function in reversed(functions):
            x = function(x)
        return x
    return result


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
        self.seed = seed
        self.rng = None
        self.count = None
        self.weights = None
        self.count0 = None
        self.mean0 = None

        self.x0 = float(n_min)
        self.n_max = n_max
        self.nstep = nstep
        self.uniform = uniform
        self.store = store
        self.values = self.xvals = None

    def init(self):
        self.rng = rng = np.random.default_rng(self.seed)
        n_max = self.n_max
        uniform = self.uniform
        n = np.arange(1, n_max+1)

        probabilities = np.zeros(n_max, dtype=float)
        probabilities[1] = 1.0
        probabilities[2:] = 1.0 / np.log(n[2:])

        selected = rng.random(n_max) < probabilities
        count = np.cumsum(selected)
        mean = np.cumsum(probabilities)
        self.count = (count - mean) / np.sqrt(mean.clip(1))
        self.weights = x_sampling_weights(not uniform, 1, n_max)
        self.count0 = count[-1] - mean[-1]
        self.mean0 = mean[-1]
        if self.store:
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
    def __init__(self, n_max=1000, n_min = 2, nstep=100, new_norm=False, offset=True, store=False):
        self.count = None
        self.weights = None
        self.x0 = float(n_min-1)
        self.n_max = n_max
        self.nstep = nstep
        self.new_norm = new_norm
        self.offset = offset
        self.store = store
        self.values = self.xvals = None

    def init(self):
        n_max = self.n_max
        count = build_prime_count(n_max)[1:]

        n = np.arange(1, n_max+1)
        x = n[1:]

        li_x = li(x)
        mean = li_x.copy()
        if self.new_norm:
            if self.offset:
                mean += np.sqrt(x) / np.log(x) - li(np.sqrt(x))/2 - li(np.cbrt(x))/3
        else:
            mean += - li(np.sqrt(x))/2 - li(np.cbrt(x))/3

        mean = np.concatenate([[0], mean])
        norm = np.log(x) / np.sqrt(x) if self.new_norm else 1. / np.sqrt(li_x)
        errors = (count - mean) * np.concatenate([[1], norm])
        self.count = errors
        self.weights = x_sampling_weights(True, 1, n_max)
        print('len count', len(self.count))
        if self.store:
            self.values = self.count[::10]
            self.xvals = n[::10]
            assert len(self.values) == len(self.xvals)

    def new_samples(self, x, do_xvals=False):
        count = []
        weights = []
        xvals = []
        # print('new_samples:', self.x0, self.n_max, x)
        if self.x0 < self.n_max:
            n_max = min(int(round(x)), len(self.count))
            n0 = int(round(self.x0))
            # print(np.array([n0+1, n_max]))
            count.append(self.count[n0:n_max])
            weights.append(self.weights[n0:n_max])
            assert n_max <= len(self.count)
            if do_xvals: xvals.append(np.arange(n0+1, n_max+1))
            # print('len', len(xvals[0]), len(weights[0]), len(count[0]))

        # assert self.n_max >= x
        if self.n_max < x:
            x0 = max(self.x0, self.n_max)
            xvec0 = np.exp(np.linspace(np.log(x0), np.log(x), self.nstep+1))
            xvec = xvec0[1:]
            # print(xvec)
            livec = li(xvec)
            mean = livec.copy()
            if self.new_norm:
                if self.offset:
                    mean += np.sqrt(xvec) / np.log(xvec) - li(np.sqrt(xvec))/2 - li(np.cbrt(xvec))/3
            else:
                mean += -li(np.sqrt(xvec))/2 - li(np.cbrt(xvec))/3

            count_vec = np.fromiter((primecount.prime_pi(int(x)) for x in xvec), dtype=np.int64)

            errors = count_vec - mean

            weight_vec = np.log(xvec0)

            norm = np.log(xvec) / np.sqrt(xvec) if self.new_norm else 1. / np.sqrt(livec)

            count_norm = errors * norm
            count.append(count_norm)
            weights.append(weight_vec[1:] - weight_vec[:-1])
            if do_xvals: xvals.append(xvec)
            # print('len', len(xvec), len(weight_vec)-1, len(count_norm))

        self.x0 = x

        if len(count) == 0:
            if do_xvals:
                return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float)
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        count_full = np.concatenate(count)

        if do_xvals:
            xvals_full = np.concatenate(xvals)
            if self.store:
                self.xvals = np.concatenate((self.xvals, xvals_full[4::10]))
                self.values = np.concatenate((self.values, count_full[4::10]))
                assert len(self.values) == len(self.xvals)

            return count_full, np.concatenate(weights), xvals_full
        return count_full, np.concatenate(weights)


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

        self.counter.init()
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
    counter = CramerCount(seed=1, n_min=500, n_max=1_000_000, nstep=1000)
    rel_width = 0.9
    xlen = 12
    ylen = 5.5

    def do_histogram(self, axes):
        x_scale = self.xlen / (self.bin_max - self.bin_min)
        hist = Histogram(self.bin_min, self.bin_max, self.bin_count, x_scale=x_scale,
                         y_scale=self.ylen / self.y_max, rel_width=self.rel_width)
        hist.bars.shift(axes.c2p(self.bin_min, 0))

        tracker = ValueTracker(self.n_max)
        counter_label = MathTex(r"n=", font_size=50, stroke_width=2)
        counter_label[0][0].set_color(col_var)
        counter_label.move_to(axes.c2p(0.9 * self.bin_min + 0.1 * self.bin_max, 5/7 * self.y_max))
        counter_value = always_redraw(
            lambda: Integer(round(tracker.get_value()), color=col_num, group_with_commas=True, stroke_width=2)
            .next_to(counter_label[0][-1], RIGHT, buff=0.15).set_z_index(10)
        )

        def update_bars(group: VGroup) -> None:
            errors, weights = self.counter.new_samples(tracker.get_value())
            hist.add_data(errors, weights)
            # print('weights', hist.bar_weights)

        self.counter.init()

        update_bars(hist.bars)
        self.play(FadeIn(counter_label, counter_value, hist.bars))

        hist.bars.add_updater(update_bars)
        print('running histogram')
        self.play(
            tracker.animate.set_value(self.n_max_end),
            run_time=self.animation_seconds,
            rate_func=rate_func_log(self.n_max, self.n_max_end),
        )


    def construct(self) -> None:
        xlen = self.xlen
        ylen = self.ylen

        axes = Axes(x_range=[self.bin_min, self.bin_max], y_range=[0, self.y_max],
            x_length=xlen, y_length=ylen, tips=False,
            axis_config={"include_ticks": False},
        ).shift(DOWN * 0.45).set_z_index(5)
        tick = Line(ORIGIN, DOWN*0.1, stroke_width=4, stroke_color=WHITE).set_z_index(5)
        ticks = VGroup(*[tick.copy().shift(axes.c2p(i)) for i in [-2, -1, 0, 1, 2]]).set_z_index(5)
        xlabels = VGroup(*[MathTex('{}'.format(i), font_size=40)[0] for i in [-2, -1, 0, 1, 2]])
        for t, l in zip(ticks[:], xlabels): mh.align_sub(l, l[-1], t, DOWN, buff=0.1)

        normal_curve = axes.plot(normal_density,
            x_range=[self.bin_min, self.bin_max, 0.02],
            color=ORANGE, stroke_width=4).set_z_index(4)
        area = axes.get_area(normal_curve, (self.bin_min, self.bin_max), color=ORANGE, opacity=0.2).set_z_index(4)

        self.add(axes.x_axis, ticks, xlabels, normal_curve, area)
        self.do_histogram(axes)
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

class EmpiricalHistogramNew(CramerHistogramLog):
    bin_max = 0.1
    bin_min = -2.1
    bin_count = 21
    y_max = 2.
    n_max = 700_000
    n_max_end = 100_000_000_000
    counter = EmpiricalCount(n_min=500_000, n_max=1_000_000, nstep=500, new_norm=True, offset=True)
    # counter = EmpiricalCount(n_min=500_000, n_max=1_000_000, nstep=5, new_norm=True)
    animation_seconds = 14
    # animation_seconds = 1


    def construct(self):
        xlen = self.xlen
        ylen = self.ylen
        bin_max = self.bin_max
        bin_min = self.bin_min
        y_max = self.y_max
        nzeros = 100
        ntheory = 100_000
        theory_center = -1.

        rng = np.random.default_rng(4)

        print('theory samples')
        gammas = np.array([float(mp.im(mp.zetazero(k))) for k in range(1, nzeros + 1) ])
        coeff = 2.0 / np.sqrt(0.25 + gammas ** 2)
        print(coeff[0])

        oscillation = np.zeros(ntheory, dtype=float)
        for a in coeff:
            theta = rng.uniform(0.0, 2.0 * np.pi, ntheory)
            oscillation += a * np.cos(theta)
        variance_total = 2.0 + np.euler_gamma - np.log(4.0 * np.pi)
        variance_explicit = 0.5 * np.sum(coeff ** 2)
        variance_tail = variance_total - variance_explicit
        print('tail width', np.sqrt(variance_tail))

        print('building curve')
        ntheoryplot = 200
        xtheory = np.linspace(bin_min, bin_max, ntheoryplot, dtype=float)
        ytheory = np.zeros(ntheoryplot, dtype=float)
        xtheory2 = xtheory - theory_center
        for x in oscillation:
            ytheory += np.exp(-(xtheory2-x)**2 / (2*variance_tail))
            ytheory += np.exp(-(xtheory2 + x) ** 2 / (2 * variance_tail))
        ytheory /= np.sqrt(2*np.pi*variance_tail) * ntheory * 2

        print('built curve')

        axes = Axes(x_range=[bin_min, bin_max], y_range=[0, y_max],
            x_length=xlen, y_length=ylen, tips=False,
            axis_config={"include_ticks": False, 'stroke_width': 4},
        ).set_z_index(2)
        xticks = mh.get_xticks(axes, vals=[-2, -1, 0], label_color=col_num)
        yticks = mh.get_yticks(axes, [1, 2, 3, 4], label_color=col_num, side=RIGHT)

        plt = axes.plot_line_graph(xtheory, ytheory, add_vertex_dots=False, stroke_width=6, stroke_color=ORANGE).set_z_index(4)
        plt_ = axes.plot_line_graph(xtheory, ytheory, add_vertex_dots=False, stroke_width=0, stroke_opacity=0,
                                   fill_color=ORANGE, fill_opacity=0.2).set_z_index(3.9)
        var = variance_total
        ynormal = np.exp(-(xtheory-theory_center)**2/(2*var)) / np.sqrt(2*np.pi*var)
        plt2 = axes.plot_line_graph(xtheory, ynormal, add_vertex_dots=False, stroke_width=8, stroke_color=BLUE).set_z_index(5)

        self.add(axes, plt, xticks, yticks)
        self.play(Create(plt, rate_func=linear, run_time=1.5),
                  Succession(Wait(0.8), FadeIn(plt_)))
        self.wait(0.1)

        self.play(FadeIn(plt2))
        self.wait(0.1)
        self.play(FadeOut(plt2))
        self.wait(0.1)
        self.do_histogram(axes)
        self.wait()

class EmpiricalHistogramNew2(EmpiricalHistogramNew):
    n_max = 700_000
    n_max_end = 100_000_000_000
    counter = EmpiricalCount(n_min=500_000, n_max=1_000_000, nstep=500, new_norm=True, offset=False)
    # counter = EmpiricalCount(n_min=500_000, n_max=1_000_000, nstep=5, new_norm=True)
    animation_seconds = 14
    # animation_seconds = 1
    # print((li(np.sqrt(1e9))/2+li(np.cbrt(1e9))/3)/np.sqrt(1e9)*np.log(1e9)-1)


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
        xtickstrs = [r'1\,000', r'10\,000', r'100\,000', r'\!1\,000\,000', r'10\,000\,000', r'1\,000\,000\,000']
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
        counter.init()

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

class EmpiricalVarNew(Scene):
    def construct(self):
        n_max1 = 1_000_000
        n_max2 = 100_000_000_000
        axes = Axes(x_range=[0, 1], y_range=[0, 0.105],
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

        variance = 2.0 + np.euler_gamma - np.log(4.0 * np.pi)
        print('variance', variance)
        eq1 = Tex(r'\sf prediction', color=ORANGE, font_size=50, stroke_width=2)
        eq1.next_to(axes.c2p(0.5, variance), DOWN, buff=0.1)
        eq2 = Tex(r'\sf Empirical value', color=BLUE, font_size=50, stroke_width=2)
        eq2.move_to(axes.c2p(0.5, 0.06))

        xtickvals = np.log([1e3, 1e4, 1e5, 1e6, 1e8, 1e11])
        xtickstrs = [r'1\,000', r'10\,000', r'100\,000', r'\!1\,000\,000', r'100\,000\,000', r'100\,000\,000\,000']
        xtickvals1 = (xtickvals - xtickvals[0]) / (xtickvals[3] - xtickvals[0])
        xtickvals2 = (xtickvals - xtickvals[0]) / (xtickvals[-1] - xtickvals[0])
        xticks1 = mh.get_xticks(axes, vals=xtickvals1, label_color=col_num, strs=xtickstrs, buff=0.4)
        xticks2 = mh.get_xticks(axes, vals=xtickvals2, label_color=col_num, strs=xtickstrs, buff=0.4)
        VGroup(xticks2[1], xticks2[3]).set_opacity(0)
        xticks1[3][1].to_edge(RIGHT, buff=0.2)
        xticks2[-1][1].to_edge(RIGHT, buff=0.2)

        ytickvals = np.array([0., variance, 0.1])
        ystrs = [r'0', r'.046', r'0.1']
        # ytickvals2 = np.log(ytickvals[1:] * 1e5) / np.log(1e5)
        tick_width = (axes.get_left() - mh.pos(LEFT))[0] - 0.4
        yticks1 = mh.get_yticks(axes, vals=ytickvals, strs=ystrs, max_width=tick_width, label_color=col_num)
        yticks1[1].set_color(color=ORANGE)
        # yticks2 = mh.get_yticks(axes, vals=ytickvals2, strs=ystrs[1:], max_width=tick_width, label_color=col_num)
        # yticks1[1:-1].set_opacity(0)
        ylines1 = VGroup(*[Line(axes.c2p(0, y), axes.c2p(1, y), stroke_width=3, stroke_color=GREY, stroke_opacity=0)
                           for y in ytickvals[1:]])
        # ylines2 = VGroup([Line(axes.c2p(0, y), axes.c2p(1, y), stroke_width=3, stroke_color=GREY, stroke_opacity=0.5)
        #                   for y in ytickvals2[:-1]])

        line1 = DashedLine(axes.c2p(0,variance), axes.c2p(1,variance), stroke_color=ORANGE, stroke_width=6, dash_length=0.15, dashed_ratio=0.7).set_z_index(2)

        counter = EmpiricalCount(n_min=500, n_max=1_000_000, nstep=100_000, new_norm=True, offset = True)
        counter.init()

        samples, weights, xvals = counter.new_samples(n_max2, do_xvals=True)
        samples += 1
        cumweights = np.cumsum(weights)
        samples_exp2 = np.cumsum(samples * samples * weights) / cumweights

        self.add(axes, xticks1, yticks1, ylines1, labely, labelx, title)

        self.play(FadeIn(eq1), Create(line1, rate_func=linear))
        self.wait(0.1)

        xvals_log = np.log(xvals)
        n1 = 1000
        xplot_1 = np.linspace(np.log(n1), np.log(n_max1), n1)
        xplot_2 = np.linspace(np.log(n_max1), np.log(n_max2), 1000)[1:]
        xplot = np.concatenate([xplot_1, xplot_2])
        yplot = np.interp(xplot, xvals_log, samples_exp2)
        xplot_scale = (xplot - xplot_1[0]) / (xplot_1[-1] - xplot_1[0])

        plt1 = axes.plot_line_graph(xplot_scale[:n1], yplot[:n1], add_vertex_dots=False, stroke_color=BLUE, stroke_width=8)

        self.play(Create(plt1, rate_func=linear, run_time=2),
                  Succession(Wait(0.6), FadeIn(eq2)))
        self.wait(0.1)

        plt3 = axes.plot_line_graph(xplot_scale, yplot, add_vertex_dots=False, stroke_color=BLUE, stroke_width=8)
        box1 = Rectangle(width=3, height=2, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=1)
        box1.next_to(plt1.get_right(), RIGHT, buff=0).set_z_index(2)

        tracker = ValueTracker(0.)
        x0 = xplot_scale[n1-1]
        x1 = xplot_scale[-1]
        print(x0, x1)

        def get_tracker_obj():
            t = tracker.get_value()
            x2 = x1 * (1-t) + x0 * t
            x3 = x0 / x2 * x1
            y3 = np.interp(x3, xplot_scale, yplot)
            val_right = DecimalNumber(y3, 3, font_size=50, stroke_width=1.5, color=BLUE)
            val_right[1:].next_to(axes.c2p(1, y3), RIGHT, buff=0.1).set_z_index(3)
            return val_right[1:]

        self.remove(plt1)
        self.add(plt3, box1)
        tracker_obj = always_redraw(get_tracker_obj)
        self.play(FadeIn(tracker_obj))

        xplot_scale2 = (xplot - xplot[0]) / (xplot[-1] - xplot[0])
        plt4 = axes.plot_line_graph(xplot_scale2, yplot, add_vertex_dots=False, stroke_color=BLUE, stroke_width=8)

        self.play(mh.rtransform(plt3, plt4, xticks1, xticks2),
                  tracker.animate.set_value(1.), run_time=3)

        self.wait()

class EmpiricalPath1(Scene):
    counter = EmpiricalCount(n_min=2, n_max=1_000_000, nstep=1000, store=True)
    def construct(self):
        print('init start')
        self.counter.init()
        print('init done')
        bin_max = 0.5
        bin_min = -0.5
        n_max_end=1_000_000

        ax2 = Axes(x_range=[0,1], y_range=[bin_min,bin_max], x_length=10, y_length=6)

        n_max = 1000
        tracker = ValueTracker(n_max)
        counter_label = MathTex(r"n=", font_size=50, stroke_width=2)
        counter_label[0][0].set_color(col_var)
        counter_value = always_redraw(
            lambda: Integer(round(tracker.get_value()), color=col_num, group_with_commas=True, stroke_width=2)
            .next_to(counter_label, RIGHT, buff=0.12).set_z_index(10)
        )

        def update_bars():
            x = tracker.get_value()
            self.counter.new_samples(x, do_xvals=True)
            xvals = np.linspace(10, x, 800)
            yvals = np.interp(xvals, self.counter.xvals, self.counter.values)
            plt = ax2.plot_line_graph(xvals/xvals[-1], -yvals, add_vertex_dots=False, stroke_width=4,
                                      stroke_color=BLUE, stroke_opacity=1).set_z_index(10)
            return VGroup(plt['line_graph'])

        print('setup bars updater')
        bars = always_redraw(update_bars)

        print('bars updater done')

        self.add(counter_label, counter_value, bars)
        self.add(ax2)

        rate_func=rate_func_log(n_max, n_max_end)
        print('main anim')
        self.play(
            tracker.animate(run_time=12, rate_func=rate_func).set_value(n_max_end),
        )
        self.wait(1)

def count_mean(x):
    return expi(np.log(x)) - expi(np.log(x)/2)/2 - expi(np.log(x)/3)/3 - expi(np.log(2)) * (1-1/2-1/3)


def zeros(i): return r'0' * (i % 3) + r'\,000' * (i // 3)

def get_tick_strs(i): return [r'5' + zeros(i-1), r'1' + zeros(i)]

def animation_scale_redraw(x_scale, y_scale, obj_scale, obj1x, obj2x, obj1y, obj2y, origin=ORIGIN):
    anim_tracker = ValueTracker(0.)
    animationx = mh.rtransform(obj1x, obj2x, rate_func=linear)
    animationy = mh.rtransform(obj1y, obj2y, rate_func=linear)
    animationx.begin()
    animationy.begin()

    def obj_func1():
        u = anim_tracker.get_value()
        scalex = np.exp(np.log(x_scale) * u)
        scaley = np.exp(np.log(y_scale) * u)
        scale = np.array([scalex, scaley, 1])
        tx = (1-scalex) / (1-x_scale)
        ty = (1-scaley) / (1-y_scale)
        obj_scale2 = obj_scale.copy().apply_points_function_about_point(lambda p: p * scale, origin)
        animationx.interpolate(tx)
        animationy.interpolate(ty)
        return VGroup(obj_scale2, obj1x.copy(), obj1y.copy())
    return anim_tracker, always_redraw(obj_func1)

class EmpiricalPath(Scene):
    def construct(self):
        nplt = 1000
        ax_args = {'x_range': [0, 1.05], 'y_range': [0, 1.05], 'x_length': 12,
                   'axis_config': {'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                                  "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                                  "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                                  },
        }
        ax = Axes(y_length=6, **ax_args).set_z_index(1).shift(RIGHT*0.2)
        origin = ax.coords_to_point(0,0)
        eqx = MathTex(r'x', stroke_width=1.5, font_size=60, color=col_x).next_to(ax.x_axis.get_right(), RIGHT, buff=0.2).set_z_index(4)

        prime_count = build_prime_count(1200001)

        x, y = prime_counting_vectors(prime_count, 1200001)
        ticksy = mh.get_yticks(ax, [0]).set_z_index(0.5).set_opacity(0)

        box1 = Rectangle(width=2, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.3)
        box1.next_to(ax.c2p(1., 0.), UR, buff=0).to_edge(DOWN, buff=0)#next_to(ax.x_axis.tip, UP, buff=0.01, coor_mask=UP)
        box2 = Rectangle(width=2, height=config.frame_height/2, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.6)
        box2.next_to(ax.c2p(1.06, 0.02), DR, buff=0)
        # box3 = Rectangle(height=2, width=config.frame_width, stroke_width=0, stroke_opacity=0,
        #                  fill_color=BLACK, fill_opacity=1).set_z_index(0.6)
        # box3.next_to(ax.c2p(0.1, 1.0), UR, buff=0)
        # box4 = Rectangle(height=2, width=config.frame_width/4, stroke_width=0, stroke_opacity=0,
        #                  fill_color=BLACK, fill_opacity=1).set_z_index(0.4)
        # box4.next_to(ax.c2p(0,1.04), UL, buff=0)

        scalex3 = 1/100
        scaley3 = 3./100

        ticks3 = mh.get_xticks(ax, [2, 3, 5, 7, 11, 13, 17, 19, 50, 100, 500, 1000], scalex=scalex3)
        ticks3[:-4].set_opacity(0)
        ticksy3 = mh.get_yticks(ax, [1, 2, 3, 4, 5, 6, 7, 8, prime_count[50], prime_count[100],
                                  prime_count[500], prime_count[1000]], scaley=scaley3)
        ticksy3[:-4].set_opacity(0)

        self.add(ax, eqx, box1, box2, ticks3, ticksy3)

        """
        plot prime counting and Li
        """

        plt7 = ax.plot_line_graph(x[:127]*scalex3, y[:127]*scaley3, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(0.2)

        label_size = 55
        eq_pi1 = MathTex(r'\pi(x)', stroke_width=1.5, color=BLUE, font_size=label_size).move_to(ax.c2p(0.7, 0.45))
        eq_li1 = MathTex(r'{\rm Li}(x)', stroke_width=1.5, color=ORANGE, font_size=label_size).move_to(ax.c2p(0.4, 0.58))
        eq_pi1 = mh.eq_shadow(eq_pi1, bg_z_index=5, fg_z_index=6)
        eq_li1 = mh.eq_shadow(eq_li1, bg_z_index=5, fg_z_index=6)

        self.play(Create(plt7, rate_func=linear), FadeIn(eq_pi1))

        xvals2 = np.linspace(4., 101., 1000)
        yvals4 = expi(np.log(xvals2)) - expi(np.log(2.))

        plt_line4 = ax.plot_line_graph(xvals2 * scalex3, yvals4 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)

        self.wait(0.1)
        self.play(Create(plt_line4), FadeIn(eq_li1))

        """
        initial zoom out
        """

        scalex4 = 1/1000
        scalex_new = 1/10000
        scaley5 = 8/10000
        scaley4 = np.sqrt(scaley5/scaley3)*scaley3

        xvals3 = np.linspace(4., 1001., 4000)
        yvals6 = expi(np.log(xvals3)) - expi(np.log(2.))

        i = np.searchsorted(x, 1050., side='right')

        plt8 = ax.plot_line_graph(x[:i]*scalex3, y[:i]*scaley3, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(0.2)
        plt_line7 = ax.plot_line_graph(xvals3 * scalex3, yvals6 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.19)

        ticks4 = mh.get_xticks(ax, [50, 100, 500, 1000, 5000, 10000],
                               [item for i in [2,3,4] for item in get_tick_strs(i)], scalex4)
        ticks4[:1].set_opacity(0)
        ticksy4 = mh.get_yticks(ax, [prime_count[50], prime_count[100], prime_count[500], prime_count[1000],
                                  prime_count[5000], prime_count[10000]], scaley=scaley4)
        ticksy4[0].set_opacity(0)

        rate_func = lambda t: 10 * (1 - np.exp(-np.log(10) * t)) / 9
        pi_shift=0.5 * UP

        self.wait(0.1)
        self.remove(plt7, plt_line4, ticks3, ticksy3)

        tracker, plt = animation_scale_redraw(scalex4/scalex3, scaley4/scaley3, VGroup(plt_line7, plt8),
                                                 ticks3[-4:], ticks4[-6:-2].copy(),
                                                 ticksy3[-4:], ticksy4[:-2].copy(), origin=origin)

        self.add(plt)
        dt = 1.
        self.play(tracker.animate.set_value(1),
                  eq_pi1.animate.shift(pi_shift),
                  run_time=1.2 * dt, rate_func=mh.rate_func_quad(0.2, 0.))
        self.remove(plt)

        """
        further zoom out
        """

        xvals_new = np.linspace(4., 10010, nplt)
        yvals8 = expi(np.log(xvals_new)) - expi((np.log(2)))  # Li up to 10k

        i = np.searchsorted(x, 10050., side='right')
        plt10 = ax.plot_line_graph(x[:i]*scalex4, y[:i]*scaley4, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(0.2)
        plt_line13 = ax.plot_line_graph(xvals_new * scalex4, yvals8 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.19)

        tick_vals_new = [100, 500, 1000, 5000, 10_000, 50_000, 100_000]
        tick_strs_new = ['100'] + [item for i in [3, 4, 5] for item in get_tick_strs(i)]
        ticks5 = mh.get_xticks(ax, tick_vals_new, tick_strs_new, scalex_new)
        ticks5[:2].set_opacity(0)
        ticks5[4][1].next_to(box2, LEFT, coor_mask=RIGHT, buff=0.01)
        ticksy5 = mh.get_yticks(ax, [prime_count[100], prime_count[500], prime_count[1000],
                                  prime_count[5000], prime_count[10000]], scaley=scaley5)
        ticksy5[:2].set_opacity(0)

        tracker, plt = animation_scale_redraw(scalex_new/scalex4, scaley5/scaley4, VGroup(plt_line13, plt10),
                                                 ticks4[1:], ticks5[:-2].copy(),
                                                 ticksy4[1:], ticksy5[:].copy(), origin=origin)

        self.add(plt)

        self.play(tracker.animate.set_value(1),
                  eq_pi1.animate.shift(pi_shift),
                  run_time=1.4 * dt, rate_func=mh.rate_func_quad(0, 0.4))
        self.remove(plt)
        plt_line14, plt11 = tuple(plt[0][:])
        self.add(ticksy5, ticks5[:-2], plt11, plt_line14)

        self.wait(0.1)

        """
        do diff with Li
        """

        scaley6 = 3/100
        yvals9 = np.interp(xvals_new+0.5, x, y, left=0, right=y[-1])  # pi up to 10k
        yvals10 = yvals9 - yvals8  # pi - Li up to 20k
        plt13 = ax.plot_line_graph(xvals_new*scalex_new, yvals10*scaley6+0.8, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(0.2)
        plt_line17 = ax.plot_line_graph(xvals_new * scalex_new, xvals_new * 0 + 0.8, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.19)

        ticksy6 = mh.get_yticks(ax, [-20, -10, 0, 10], scaley=scaley6, center=0.8)
        ticksy6[-1].set_opacity(0)

        eq_pi2 = MathTex(r'\pi(x)', r'-', r'{\rm Li}(x)', stroke_width=1.5, color=BLUE, font_size=label_size).move_to(ax.c2p(0.4, 0.2))
        eq_pi2 = mh.eq_shadow(eq_pi2, bg_z_index=5, fg_z_index=6)

        # self.play(FadeOut(plt11), FadeIn(plt12))
        self.play(AnimationGroup(mh.rtransform(plt11, plt13, plt_line14, plt_line17, ticksy[0], ticksy6[-2]),
                  FadeOut(ticksy5),
                  mh.rtransform(eq_pi1[0], eq_pi2[0], eq_li1[0], eq_pi2[2]), run_time=2),
                  Succession(Wait(1.5), FadeIn(eq_pi2[1])),
                  Succession(Wait(1.5), FadeIn(ticksy6[:-2])),
                  run_time=2)

        self.wait(0.1)

        """
        bias calculation
        """

        pos1 = ax.c2p(0.67, 0.9)
        pos2 = ax.c2p(0.67, 0.55)
        eq_li2 = MathTex(r'-\frac12{\rm Li}(\sqrt{x})-\frac13{\rm Li}(\sqrt[3]{x})', stroke_width=1.5, color=ORANGE,
                         font_size=label_size).move_to(pos2)
        eq_li2 = mh.eq_shadow(eq_li2, bg_z_index=5, fg_z_index=6)

        yvals11 = count_mean(xvals_new) - yvals8

        plt_line19 = ax.plot_line_graph(xvals_new * scalex_new, yvals11 * scaley6 + 0.8, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.19)

        plt_line17_ = plt_line17.copy().set_stroke(color=GREY).set_opacity(0.48)
        self.add(plt_line17_)
        self.play(mh.rtransform(plt_line17, plt_line19), FadeIn(eq_li2, shift=pos2-pos1))
        self.wait(0.1)

        """
        subtract bias
        """

        eq_pi3 = MathTex(r'\pi(x)', r'-', r'\left(', r'{\rm Li}(x)', r'-\frac12{\rm Li}(\sqrt{x})-\frac13{\rm Li}(\sqrt[3]{x})', r'\right)',
                         stroke_width=1.5, color=BLUE, font_size=label_size)
        eq_pi3 = mh.eq_shadow(eq_pi3, bg_z_index=5, fg_z_index=6)
        eq_pi3.next_to(ax.c2p(0,0.8), RIGHT, buff=0.4)
        eq_pi4 = MathTex(r'\pi(x)', r'-', r'\hat\pi(x)', stroke_width=1.5, color=BLUE, font_size=label_size)
        eq_pi4 = mh.eq_shadow(eq_pi4, bg_z_index=5, fg_z_index=6)
        mh.align_sub(eq_pi4, eq_pi4[1], eq_pi3[1]).move_to(ax.c2p(0.3,0), coor_mask=RIGHT)

        yvals_new = yvals10 - yvals11
        t_5 = 1e4
        scaley = 1. / 1.1
        scaley_new = np.log(t_5) / np.sqrt(t_5) * scaley
        ticksy7 = mh.get_yticks(ax, [-0.5, 0, 0.5], scaley=scaley, center=0.5)

        plt14 = ax.plot_line_graph(xvals_new*scalex_new, yvals_new*scaley_new+0.5, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(0.2)
        plt_line20 = ax.plot_line_graph(xvals_new * scalex_new, xvals_new * 0 + 0.5, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.19)
        self.play(AnimationGroup(mh.rtransform(plt13, plt14, plt_line19, plt_line20, ticksy6[2], ticksy7[1]),
                                plt_line17_.animate.shift(ax.c2p(0, 0.5 - 0.8)-origin),
                  mh.rtransform(eq_pi2[:2], eq_pi3[:2], eq_pi2[2], eq_pi3[3], eq_li2[0], eq_pi3[4]),
                  FadeOut(ticksy6[:2], ticksy6[-1]),
                  FadeIn(ticksy7[::2]),
                  run_time=1.6),
                  Succession(Wait(1.2), FadeIn(eq_pi3[2], eq_pi3[-1]))
                  )
        self.remove(plt_line17_)
        self.wait(0.1)
        eq_pi4_ = eq_pi4[2].copy().move_to(eq_pi3[2:], coor_mask=RIGHT)
        self.play(FadeOut(eq_pi3[2:]), FadeIn(eq_pi4_))
        self.play(mh.rtransform(eq_pi3[:2], eq_pi4[:2], eq_pi4_, eq_pi4[2]))
        self.wait(0.1)

        """
        bring in theoretical sample path
        """

        sw = 5
        ax2 = Axes(y_length=3.5, **ax_args).set_z_index(1).to_edge(DOWN, buff=0.5)
        ax2.shift((origin - ax2.c2p(0,0))*RIGHT)
        ax3 = Axes(y_length=3.5, **ax_args).set_z_index(1).to_edge(DOWN, buff=0.5)
        ax3.next_to(ax2, UP, buff=0.1)

        plt_new = ax2.plot_line_graph(xvals_new*scalex_new, yvals_new*scaley_new+0.5, line_color=BLUE, stroke_width=sw, add_vertex_dots=False).set_z_index(0.2)
        plt_line21 = ax2.plot_line_graph(xvals_new[::] * scalex_new, xvals_new * 0 + 0.5, add_vertex_dots=False, stroke_width=sw, line_color=ORANGE).set_z_index(0.19)
        plt_line22 = ax3.plot_line_graph(xvals_new[::] * scalex_new, xvals_new * 0 + 0.5, add_vertex_dots=False, stroke_width=sw, line_color=ORANGE).set_z_index(0.19)
        ticksy8 = mh.get_yticks(ax2, [-0.5, 0, 0.5], scaley=scaley, center=0.5)
        ticksy9 = mh.get_yticks(ax3, [-0.5, 0, 0.5], scaley=scaley, center=0.5)
        ticks_new = mh.get_xticks(ax2, tick_vals_new, tick_strs_new, scalex_new, buff=0.15, font_size=40, length=0.1)
        ticks_new[:2].set_opacity(0)

        txt1 = Tex(r'\sf Empirical:\ ', r'$\pi(x)-\hat\pi(x)$', font_size=label_size, stroke_width=2)
        txt2 = Tex(r'\sf Random sample path', r':\ random walk', font_size=label_size, stroke_width=2)
        VGroup(txt2, txt1[0]).set_color(col_txt)
        VGroup(txt1[1][0], txt1[1][5:7]).set_color(col_WVD)
        VGroup(txt1[1][2], txt1[1][8]).set_color(col_x)
        txt1 = mh.eq_shadow(txt1, bg_z_index=5, fg_z_index=6)
        txt2 = mh.eq_shadow(txt2, bg_z_index=5, fg_z_index=6)
        txt1.next_to(ax2.c2p(0,0.95), DR, buff=0).shift(RIGHT*0.3)
        txt2.next_to(ax3.c2p(0,0.95), DR, buff=0).shift(RIGHT*0.3)

        self.play(mh.rtransform(ax, ax2, plt14, plt_new, plt_line20, plt_line21, ticksy7, ticksy8, ticks5, ticks_new,
                                ax.y_axis.copy(), ax3.y_axis, ticksy7.copy(), ticksy9, plt_line20.copy(), plt_line22),
                  mh.rtransform(eq_pi4[0][:], txt1[1][:4], eq_pi4[1][0], txt1[1][4], eq_pi4[2][:], txt1[1][5:]),
                  eqx.animate.next_to(ax2.x_axis.get_right(), RIGHT, buff=0.2),
                  box2.animate.next_to(ax2.c2p(1.06, 0.02), DR, buff=0),
                  Succession(Wait(0.4), FadeIn(txt1[0], txt2[0]))
        )

        """
        random walk
        """

        rng = np.random.default_rng(4)
        nzeros = 2000

        print('theory samples')
        variance_total = 2.0 + np.euler_gamma - np.log(4.0 * np.pi)
        print('std dev', np.sqrt(variance_total))
        path_cols = [BLUE]
        path_z = [.2]
        path_ops = [1]
        noise = [np.cumsum(np.random.normal(0, 1, size=nplt-1) * np.sqrt((xvals_new[1] - xvals_new[0])*variance_total)) for _ in path_cols]
        yvec_scale = np.sqrt(xvals_new) / np.log(xvals_new)
        yvals_theory = [np.concat(([0], _))/np.log(xvals_new) for _ in noise]

        plt = [ax3.plot_line_graph(xvals_new * scalex_new, yvals_ * scaley_new + 0.5, line_color=col, stroke_opacity=op,
                                      stroke_width=sw, add_vertex_dots=False).set_z_index(z)
               for yvals_, op, col, z in zip(yvals_theory, path_ops, path_cols, path_z)]
        self.play(*[Create(_, rate_func=linear) for _ in plt],
                  FadeIn(txt2[1], rate_func=linear, run_time=0.5))
        self.wait(0.1)

        """
        zeta zeros sample path
        """

        gammas = load_gammas(nzeros)
        coeffs = 2.0 / np.sqrt(0.25 + gammas ** 2)
        thetas = [rng.uniform(0.0, 2.0 * np.pi, nzeros) for _ in path_cols]
        # thetas = np.acos(coeffs/4)
        variance_explicit = 0.5 * np.sum(coeffs ** 2)
        variance_tail = variance_total - variance_explicit
        print('tail width', np.sqrt(variance_tail))
        t = 1e4
        yvals_theory = [_*np.sqrt(variance_tail/variance_total) for _ in yvals_theory]
        for j, yval in enumerate(yvals_theory):
            yval *= np.sqrt(variance_tail/variance_total)
            for i in range(nzeros):
                yval += coeffs[i]*np.cos(gammas[i]*np.log(xvals_new)+thetas[j][i]) * yvec_scale
        plt_theory = [ax3.plot_line_graph(xvals_new * scalex_new, yvals_ * scaley_new + 0.5, line_color=col, stroke_opacity=op,
                                      stroke_width=sw, add_vertex_dots=False).set_z_index(z)
               for yvals_, op, col, z in zip(yvals_theory, path_ops, path_cols, path_z)]
        self.play(*[mh.rtransform(p1, p2) for p1, p2 in zip(plt, plt_theory)],
                  FadeOut(txt2[1], rate_func=linear, run_time=0.5))

        """
        final zooming out
        """

        self.wait(0.1)

        for i_exp in [5]:#, 6, 7, 8, 9, 10, 11]:
            xvals_old = xvals_new
            plt_old = plt_new
            ticks_old = ticks_new
            scalex_old = scalex_new
            scaley_old = scaley_new
            tick_vals_old = tick_vals_new
            tick_strs_old = tick_strs_new
            yvals_old = yvals_new
            yvals_theory_old = yvals_theory
            plt_theory_old = plt_theory
            noise_old = noise

            t_new = 10**i_exp
            scalex_new = 1/t_new
            scaley_new = np.log(t_new) / np.sqrt(t_new) * scaley

            xvals_new2 = np.linspace(0., t_new * 10.01, nplt*10)
            # xvals_new2 = np.linspace(4., t_new * 1.001, nplt)
            xvals_new2 = xvals_new2 * (xvals_old[1] - xvals_old[0]) / xvals_new2[1] + 4
            xvals_new = xvals_new2[::10]
            assert len(xvals_new) == nplt

            tick_vals_new = tick_vals_old[2:] + [t_new*5, t_new*10]
            tick_strs_new = tick_strs_old[2:] + get_tick_strs(i_exp+1)
            ticks_new = mh.get_xticks(ax2, tick_vals_new, tick_strs_new, scalex_new, buff=0.15, font_size=40, length=0.1)
            ticks_new[:2].set_opacity(0)
            ticks_new[4][1].next_to(box2, LEFT, coor_mask=RIGHT, buff=0.01)

            if i_exp <= 6:
                yvals_new3 = np.interp(xvals_new2[nplt:]+0.5, x, y, left=0, right=y[-1])
            else:
                yvals_new3 = np.fromiter((primecount.prime_pi(int(x)) for x in xvals_new2[nplt:]), dtype=np.int64)
            yvals_new2 = np.concatenate((yvals_old, yvals_new3 - count_mean(xvals_new2[nplt:])))
            yvals_new = yvals_new2[::10]
            plt_new2 = ax2.plot_line_graph(xvals_new2*scalex_old, yvals_new2*scaley_old+0.5, line_color=BLUE, stroke_width=sw, add_vertex_dots=False).set_z_index(.2)
            plt_new = ax2.plot_line_graph(xvals_new*scalex_new, yvals_new*scaley_new+0.5, line_color=BLUE, stroke_width=sw, add_vertex_dots=False).set_z_index(.2)

            yvals_theory2 = [np.zeros(nplt*10)]
            noise = [np.random.normal(0, 1, size=nplt*10-nplt) * np.sqrt((xvals_new2[1] - xvals_new2[0]) * variance_tail) + _[-1] for _ in noise_old]
            yvec_scale = np.sqrt(xvals_new2) / np.log(xvals_new2)
            for i in range(len(yvals_theory_old)):
                yvals_theory2[i][nplt:] = noise / np.log(xvals_new2[nplt:])
                for j, yval in enumerate(yvals_theory2):
                    for k in range(nzeros):
                        # print(len(yval), len(xvals_new2[nplt:]), len(yvec_scale[nplt:]))
                        yval[nplt:] += coeffs[k] * np.cos(gammas[k] * np.log(xvals_new2[nplt:]) + thetas[j][k]) * yvec_scale[nplt:]
                yvals_theory2[i][:nplt] = yvals_theory_old[i]
            yvals_theory = [_[::10] for _ in yvals_theory2]
            plt_theory2 = [
                ax3.plot_line_graph(xvals_new2 * scalex_old, yvals_ * scaley_old + 0.5, line_color=col, stroke_opacity=op,
                                    stroke_width=sw, add_vertex_dots=False).set_z_index(z)
                for yvals_, op, col, z in zip(yvals_theory2, path_ops, path_cols, path_z)]
            plt_theory = [
                ax3.plot_line_graph(xvals_new * scalex_new, yvals_ * scaley_new + 0.5, line_color=col, stroke_opacity=op,
                                    stroke_width=sw, add_vertex_dots=False).set_z_index(z)
                for yvals_, op, col, z in zip(yvals_theory, path_ops, path_cols, path_z)]

            self.remove(plt_old, *plt_theory_old)
            self.add(plt_new2, *plt_theory2)
            self.play(mh.rtransform(plt_new2, plt_new, ticks_old[2:], ticks_new[:-2]),
                      *[mh.rtransform(p1, p2) for p1, p2 in zip(plt_theory2, plt_theory)],
                                    run_time=3., rate_func = rate_func)

        self.wait()

def save_gammas(n):
    gammas = np.array([float(mp.im(mp.zetazero(k))) for k in range(1, n + 1)])
    path = Path("cached_gammas.npy")
    np.save(path, gammas)

def load_gammas(n):
    gammas = np.load(Path("cached_gammas.npy"))
    print('saved gamma size:', len(gammas), 'using first', n)
    assert len(gammas) >= n
    return gammas[:n]

# if __name__ == "__main__":
#     save_gammas(2000)
