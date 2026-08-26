import colorsys

import numpy as np
from manim import *
import sys
from manim import ManimColor
from scipy.special import expi

sys.path.append('../../')
import manimhelper as mh
from common.wigner import *
import matplotlib
import matplotlib.cm as cm

col_pi = col_special * 0.5 + ORANGE * 0.5
col_trig = PURPLE_A#*0.5+WHITE*0.5
col_txt = ManimColor( r'#FFAC2B')


def rate_func_quad(a=0., b=0.):
    assert a + b <= 1.

    def f(t):
        if t < a:
            res = t * t / (2 - a - b) / a
        elif t > 1 - b:
            res = 1. - (1 - t) * (1 - t) / (2 - a - b) / b
        else:
            res = (a + (t - a) * 2) / (2 - a - b)
        return res

    return f


class Eratosthenes(Scene):
    def construct(self):
        n_rows = 5
        n_cols = 20
        max_rows = 15

        n = n_cols * n_rows
        n_max = n_cols * max_rows
        buff = 0.02
        width = config.frame_width / n_cols * (1-2*buff)

        eq_num = MathTex(*[r'{}'.format(i) for i in range(1, n+1)], stroke_width=1.5, font_size=40, color=WHITE)
        height = eq_num.height * 1.8
        eq_rows = VGroup(*[eq_num[i:i+n_cols] for i in range(0, n, n_cols)]).set_z_index(5)
        for i in range(n_rows):
            for j in range(n_cols):
                eq_rows[i][j].move_to(RIGHT*j*width + DOWN * i * height)
        mh.align_sub(eq_rows, eq_rows[0][0], config.frame_width*(buff-0.5)*RIGHT+width/2*RIGHT)
        eq_rows.to_edge(DOWN, buff=0.4)

        pt0 = eq_rows[0].get_center() * UP + height/2 * UP + RIGHT * config.frame_width * (buff-0.5)

        eq1 = Tex(r'\sf Sieve of Eratosthenes', color=col_txt, stroke_width=2, font_size=55)
        eq1.next_to(pt0, UP, buff=0.2, coor_mask=UP).set_z_index(5)

        pt1 = eq1.get_top() * UP + pt0 * RIGHT + 0.2*UP
        lines = []
        line_args = {'stroke_width': 5}
        for i in range(n_rows+1):
            lines.append(Line(pt0 + i * height*DOWN, pt0 + i * height*DOWN + width*n_cols*RIGHT, **line_args))
        for i in range(n_cols+1):
            lines.append(Line(pt0 + i * width*RIGHT, pt0 + i * width*RIGHT + height*n_rows*DOWN, **line_args))
        lines.append(Line(pt0, pt1, **line_args))
        lines.append(Line(pt0+n_cols*width*RIGHT, pt1 + n_cols*width*RIGHT, **line_args))
        lines.append(Line(pt1, pt1+n_cols*width*RIGHT, **line_args))
        lines = VGroup(*lines).set_z_index(4)

        box = Rectangle(width=width, height=height, stroke_width=0, stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK)
        box.set_z_index(0.5)
        box2 = Rectangle(width=width*n_cols, height=(pt0-pt1)[1], stroke_width=0, stroke_opacity=0, fill_opacity=0.5,
                         fill_color=BLACK)
        box2.next_to(pt0, UR, buff=0)
        cross_args = {'stroke_color': RED, 'stroke_width': 6, 'buff': 0.1}
        cross = VGroup(
            Line(box.get_corner(UL), box.get_corner(DR), **cross_args),
            Line(box.get_corner(UR), box.get_corner(DL), **cross_args),
        ).set_z_index(6).set_opacity(0)

        col_prime = BLUE
        sieve = np.ones(n_max + 1, dtype=bool)
        sieve[:2] = False  # not prime
        for i in range(2, n_max + 1):
            if i*i > n_max:
                break
            if sieve[i]:
                sieve[i*i::i] = False

        boxes = []
        crosses = []
        for i in range(n_rows):
            for j in range(n_cols):
                boxes.append(box.copy().move_to(pt0 + (j+0.5) * width * RIGHT + (i+0.5) * height * DOWN))
                crosses.append(cross.copy().move_to(boxes[-1]))

        box.next_to(eq1, RIGHT, buff=1).set_fill(color=col_prime, opacity=1).set_z_index(1)
        eq_prime = MathTex(r'\sf prime', color=WHITE, stroke_width=1.5).set_z_index(4).next_to(box, RIGHT, buff=0.2)
        VGroup(eq1, eq_prime, box).move_to(ORIGIN, coor_mask=RIGHT)

        boxes = VGroup(*boxes)
        crosses = VGroup(*crosses)
        cross.move_to(boxes[0])
        eq_num[0].set_opacity(0.3)

        boxes_prime = VGroup()

        self.add(eq_rows, lines, eq1, boxes, crosses[0].set_opacity(1), box, eq_prime, box2)
        for i in range(2, n+1):
            if i * i > n:
                boxes2 = VGroup()
                for j in range(i, n + 1):
                    if sieve[j]:
                        boxes2.add(boxes[j-1])
                self.play(boxes2.animate.set_fill(color=col_prime))
                boxes_prime.add(*boxes2)
                break
            if sieve[i]:
                boxes_prime.add(boxes[i-1])
                self.play(boxes[i-1].animate.set_fill(col_prime), run_time=0.6, rate_func=linear)
                self.play(crosses[i*i-1::i].animate.set_opacity(1),
                          eq_num[i*i-1::i].animate.set_opacity(0.3), run_time=0.6, rate_func=linear)

        box3 = Rectangle(width=n_cols*width, height=max_rows*height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=0.7)
        box4 = Rectangle(width=n_cols*width, height=n_rows*height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=0.7)
        box4.next_to(pt0, DR, buff=0)

        # box3.next_to(pt0, DR, buff=0)
        # box3.to_edge(DOWN, buff=0.1)

        self.wait(0.1)
        self.play(FadeOut(box2, lines, eq_prime, eq1, box, crosses),
                  FadeIn(box4), rate_func=linear)
        boxes.set_opacity(0)
        boxes_prime.set_opacity(0.7)

        eq_max = MathTex(*[r'{}'.format(i) for i in range(1, n_max+1)], stroke_width=1.5, font_size=40, color=WHITE)
        eq_rows2 = VGroup(*[eq_max[i:i+n_cols] for i in range(0, n_max, n_cols)]).set_z_index(5)
        for i in range(1, n_max+1):
            if not sieve[i]:
                eq_max[i-1].set_opacity(0.3)
        pt2 = box3.get_corner(UL)
        for i in range(max_rows):
            for j in range(n_cols):
                eq_rows2[i][j].move_to(pt2 + RIGHT*(j+0.5)*width + DOWN * (i+0.5) * height)

        boxes_prime2 = VGroup()
        for i in range(n+1, n_max+1):
            if sieve[i]:
                boxes_prime2.add(boxes_prime[0].copy().move_to(eq_max[i-1]))

        shift = eq_rows2[:n_rows].get_center() - eq_rows.get_center()
        self.play(mh.rtransform(eq_rows[:], eq_rows2[:n_rows]),
                  boxes_prime.animate.shift(shift),
                  boxes_prime2.shift(-shift).animate.shift(shift),
                  eq_rows2[n_rows:].shift(-shift).animate.shift(shift),
                  mh.rtransform(box4, box3),
                  run_time=2, rate_func=smooth)

        self.wait()


def build_prime_count(limit=1_000_000):
    # is_prime[n] is 1 if n is prime
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"

    # Sieve of Eratosthenes
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            start = p * p
            count = ((limit - start) // p) + 1
            is_prime[start : limit + 1 : p] = b"\x00" * count

    # prime_count[n] = number of primes <= n
    prime_count = [0] * (limit + 1)
    total = 0

    for n in range(limit + 1):
        total += is_prime[n]
        prime_count[n] = total

    return prime_count


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


def get_xticks(ax, vals=[], strs=None, scalex=1.):
    if strs is None:
        strs = [r'{}'.format(_) for _ in vals]
    tick_eqs = MathTex(*strs, font_size=50, stroke_width=1.5, color=col_num)
    origin = ax.c2p(0, 0)
    tick_eqs.next_to(origin, DOWN, buff=0.3)
    tick0 = Line(origin, origin + DOWN * 0.2, stroke_width=6, stroke_color=WHITE)
    ticks = [tick0.copy().shift(ax.c2p(_ * scalex, 0) - origin) for _ in vals]
    for _ in range(len(vals)): tick_eqs[_].move_to(ticks[_], coor_mask=RIGHT)
    return VGroup(*[VGroup(tick, eq) for tick, eq in zip(ticks, tick_eqs[:])]).set_z_index(0.5)

def get_yticks(ax, vals=[], strs=None, scaley=1., max_width=0.9, center=0.):
    if strs is None:
        strs = [r'{}'.format(_) for _ in vals]
    tick_eqs = [MathTex(str, font_size=50, stroke_width=1.5, color=col_num)[0] for str in strs]
    origin = ax.c2p(0, 0)
    for eq in tick_eqs: eq.next_to(origin, LEFT, buff=0.3)
    tick0 = Line(origin, origin + LEFT * 0.2, stroke_width=6, stroke_color=WHITE)
    ticks = [tick0.copy().shift(ax.c2p(0, _ * scaley + center) - origin) for _ in vals]
    for _ in range(len(vals)):
        tick_eqs[_].move_to(ticks[_], coor_mask=UP)
        w = tick_eqs[_].width
        if w > max_width:
            tick_eqs[_].scale(max_width/w, about_edge=RIGHT)
    return VGroup(*[VGroup(tick, eq) for tick, eq in zip(ticks, tick_eqs[:])]).set_z_index(0.3)


class PiPlot(Scene):
    def construct(self):
        ax = Axes(x_range=[0, 1.05], y_range=[0, 1.05], x_length=12, y_length=6,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(1).shift(RIGHT*0.2)
        origin = ax.coords_to_point(0,0)
        eqx = MathTex(r'x', stroke_width=1.5, font_size=60, color=col_x).next_to(ax.x_axis.get_right(), RIGHT, buff=0.2).set_z_index(4)

        prime_count = build_prime_count(1200001)

        x, y = prime_counting_vectors(prime_count, 1200001)
        scalex1 = 0.1
        scaley1 = 0.25

        ticks = get_xticks(ax, [2, 3, 5, 7, 11, 13, 17, 19], scalex=scalex1)
        ticksy = get_yticks(ax, [0, 1, 2, 3, 4, 5, 6, 7, 8], scaley=scaley1)
        ticksy[0].set_z_index(0.5)

        nplt = 1000

        m = 15
        eps = 0.01
        plt1 = ax.plot_line_graph(x[:m+1]*scalex1, y[:m+1]*scaley1, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt2 = plt1.copy()
        plt3 = plt1.copy()
        box1 = Rectangle(width=2, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(3)
        box1.next_to(ax.c2p(1., 0.), UR, buff=0).next_to(ax.x_axis.tip, UP, buff=0.01, coor_mask=UP)
        box2 = Rectangle(width=2, height=config.frame_height/2, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.6)
        box2.next_to(ax.c2p(1.05, 0.02), DR, buff=0)
        box3 = Rectangle(height=2, width=config.frame_width, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.6)
        box3.next_to(ax.c2p(0.1, 1.0), UR, buff=0)
        box4 = Rectangle(height=2, width=config.frame_width/4, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.4)
        box4.next_to(ax.c2p(0,1.04), UL, buff=0)
        box5 = Rectangle(height=2, width=config.frame_width/4, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.4)
        box5.next_to(ax.c2p(0,0), DL, buff=0)

        self.add(ax, eqx, box1, box2, box3, box4, box5, ticksy[0])

        plt_args = {'line_color': BLUE, }


        self.play(Create(plt1, run_time=1.5, rate_func=lambda t: (t+eps)*3/m),
                  FadeIn(ticks[0], ticksy[1], run_time=1))
        self.wait(0.1)
        self.remove(plt1)
        self.play(Create(plt2, run_time=1., rate_func=lambda t: (t+eps)*2/m+3/m),
                  FadeIn(ticks[1], ticksy[2], run_time=1))
        self.wait(0.1)
        self.remove(plt2)
        t0 = (5+eps*2)/m
        self.play(Create(plt3, rate_func=lambda t: t0 + (1-t0)*t, run_time=2),
                  Succession(Wait(0.15), FadeIn(ticks[2], ticksy[3])),
                  Succession(Wait(0.6), FadeIn(ticks[3], ticksy[4])),
                  )

        scalex2 = 1/20
        scaley2 = 2./20
        plt4 = ax.plot_line_graph(x[:29]*scalex1, y[:29]*scaley1, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt5 = ax.plot_line_graph(x[:29]*scalex2, y[:29]*scaley2, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)

        ticks2 = get_xticks(ax, [2, 3, 5, 7, 11, 13, 17, 19, 50, 100], scalex=scalex2)
        ticksy2 = get_yticks(ax, [1, 2, 3, 4, 5, 6, 7, 8, prime_count[50], prime_count[100]], scaley=scaley2)

        self.remove(plt3)
        self.add(plt4)

        self.play(mh.rtransform(plt4, plt5, ticks[:], ticks2[:-2], ticksy[1:], ticksy2[:-2]),
                  run_tim1=1.5)

        self.wait(0.1)

        scalex3 = 1/100
        scaley3 = 3./100
        plt6 = ax.plot_line_graph(x[:127]*scalex2, y[:127]*scaley2, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt7 = ax.plot_line_graph(x[:127]*scalex3, y[:127]*scaley3, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)

        ticks3 = get_xticks(ax, [2, 3, 5, 7, 11, 13, 17, 19, 50, 100, 500, 1000], scalex=scalex3)
        ticks3[:-4].set_opacity(0)
        ticksy3 = get_yticks(ax, [1, 2, 3, 4, 5, 6, 7, 8, prime_count[50], prime_count[100],
                                  prime_count[500], prime_count[1000]], scaley=scaley3)
        ticksy3[:-4].set_opacity(0)

        self.remove(plt5)
        self.add(plt6)

        eps = 0.1

        self.play(mh.rtransform(plt6, plt7, ticks2, ticks3[:-2],
                                ticksy2[:], ticksy3[:-2], run_time=3., rate_func=rate_func_quad(0.2, 0.5)))

        xvals1 = np.linspace(0., 60., 1000)
        yvals1 = xvals1

        plt_line1 = ax.plot_line_graph(xvals1 * scalex3, yvals1 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.5)

        self.play(Create(plt_line1, run_time=1.4, rate_func=linear))

        xvals2 = np.linspace(4., 101., 1000)
        yvals3 = xvals2 / np.log(xvals2)
        yvals4 = expi(np.log(xvals2)) - expi(np.log(2.))

        plt_line2 = ax.plot_line_graph(xvals2 * scalex3, xvals2 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)
        plt_line3 = ax.plot_line_graph(xvals2 * scalex3, yvals3 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        plt_line4 = ax.plot_line_graph(xvals2 * scalex3, yvals4 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)

        self.play(mh.rtransform(plt_line2, plt_line3))
        self.wait(0.1)
        self.play(mh.rtransform(plt_line3.copy(), plt_line4))
        self.wait(0.1)

        scalex4 = 1/1000
        scaley4 = 5/1000
        xvals3 = np.linspace(4., 1001., 4000)
        xvals4 = np.linspace(0., 250., 1000)
        yvals5 = xvals3 / np.log(xvals3)
        yvals6 = expi(np.log(xvals3)) - expi(np.log(2.))

        i = np.searchsorted(x, 1050., side='right')

        plt8 = ax.plot_line_graph(x[:i]*scalex3, y[:i]*scaley3, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt9 = ax.plot_line_graph(x[:i]*scalex4, y[:i]*scaley4, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt_line5 = ax.plot_line_graph(xvals3 * scalex3, yvals5 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        plt_line6 = ax.plot_line_graph(xvals3 * scalex4, yvals5 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        plt_line7 = ax.plot_line_graph(xvals3 * scalex3, yvals6 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line8 = ax.plot_line_graph(xvals3 * scalex4, yvals6 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line9 = ax.plot_line_graph(xvals4 * scalex3, xvals4 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)
        plt_line10 = ax.plot_line_graph(xvals4 * scalex4, xvals4 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)

        ticks4 = get_xticks(ax, [50, 100, 500, 1000, 5000, 10000], ['50', '100', '500', r'1\,000', r'5\,000', r'10\,000'], scalex4)
        ticks4[:1].set_opacity(0)
        ticksy4 = get_yticks(ax, [prime_count[50], prime_count[100], prime_count[500], prime_count[1000],
                                  prime_count[5000], prime_count[10000]], scaley=scaley4)
        ticksy4[0].set_opacity(0)

        self.remove(plt7, plt_line3, plt_line4, plt_line1)
        self.add(plt8, plt_line5, plt_line7, plt_line9)

        self.play(mh.rtransform(plt8, plt9, plt_line5, plt_line6, plt_line7, plt_line8, plt_line9, plt_line10,
                                ticks3[-4:], ticks4[-6:-2], ticksy3[-4:], ticksy4[:-2],
                                run_time=3., rate_func=rate_func_quad(0.2, 0.2)))
        self.wait(0.1)

        i = np.searchsorted(x, 10050., side='right')

        scalex5 = 1/10000
        scaley5 = 8/10000

        xvals5 = np.linspace(4., 10010, nplt)
        xvals6 = np.linspace(0., 2000., 100)
        yvals7 = xvals5 / np.log(xvals5)  # x/logx up yo 10k
        yvals8 = expi(np.log(xvals5)) - expi((np.log(2)))  # Li up to 10k

        plt10 = ax.plot_line_graph(x[:i]*scalex4, y[:i]*scaley4, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt11 = ax.plot_line_graph(x[:i]*scalex5, y[:i]*scaley5, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt_line11 = ax.plot_line_graph(xvals5 * scalex4, yvals7 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        plt_line12 = ax.plot_line_graph(xvals5 * scalex5, yvals7 * scaley5, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        plt_line13 = ax.plot_line_graph(xvals5 * scalex4, yvals8 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line14 = ax.plot_line_graph(xvals5 * scalex5, yvals8 * scaley5, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line15 = ax.plot_line_graph(xvals6 * scalex4, xvals6 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)
        plt_line16 = ax.plot_line_graph(xvals6 * scalex5, xvals6 * scaley5, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)

        scaley6 = 3/100
        # yvals9 = y[:i] - expi(np.log(x[:i].clip(2.))) + expi(np.log(2.))

        yvals9 = np.interp(xvals5+0.5, x, y, left=0, right=y[-1])  # pi up to 10k
        yvals10 = yvals9 - yvals8  # pi - Li up to 20k
        plt12 = ax.plot_line_graph(xvals5*scalex5, yvals9*scaley5, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt13 = ax.plot_line_graph(xvals5*scalex5, yvals10*scaley6+0.8, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt_line17 = ax.plot_line_graph(xvals5 * scalex5, xvals5 * 0 + 0.8, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)

        ticks5 = get_xticks(ax, [100, 500, 1000, 5000, 10_000, 50_000, 100_000],
                            ['100', '500', '1\,000', r'5\,000', r'10\,000', r'50\,1000', r'100\,1000'], scalex5)
        ticks5[:2].set_opacity(0)
        ticks5[4][1].shift(LEFT*0.1)
        ticksy5 = get_yticks(ax, [prime_count[100], prime_count[500], prime_count[1000],
                                  prime_count[5000], prime_count[10000]], scaley=scaley5)
        ticksy5[:2].set_opacity(0)

        ticksy6 = get_yticks(ax, [-20, -10, 0, 10], scaley=scaley6, center=0.8)
        ticksy7 = get_yticks(ax, [-20, -10, 0, 10], scaley=scaley6, center=0.7)
        ticksy6[-1].set_opacity(0)
        # print('error', prime_count[10_000] - expi(np.log(1e4)) + expi(np.log(2)))

        self.remove(plt9, plt_line6, plt_line8, plt_line10)
        self.add(plt10, plt_line11, plt_line13, plt_line15)

        self.play(mh.rtransform(plt10, plt11, plt_line11, plt_line12, plt_line13, plt_line14, plt_line15, plt_line16,
                                ticks4[1:], ticks5[:-2], ticksy4[1:], ticksy5[:],
                                run_time=3., rate_func=rate_func_quad(0.2, 0.2)))

        self.wait(0.1)
        self.play(FadeOut(plt_line12, plt_line16),FadeOut(plt11), FadeIn(plt12))
        self.play(mh.rtransform(plt12, plt13, plt_line14, plt_line17, ticksy[0], ticksy6[-2]),
                  FadeOut(ticksy5), Succession(Wait(0.5), FadeIn(ticksy6[:-2])))

        self.wait(0.1)
        yvals11 = -(expi(np.log(xvals5)/2)-expi(np.log(2)))/2 # -Li(sqrt x)/2
        yvals12 = yvals11 - (expi(np.log(xvals5)/3)-expi(np.log(2)))/3  # -Li(sqrt x)/2 - Li(x^{1/3))/3

        plt_line18 = ax.plot_line_graph(xvals5 * scalex5, yvals11 * scaley6 + 0.8, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line19 = ax.plot_line_graph(xvals5 * scalex5, yvals12 * scaley6 + 0.8, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)

        plt_line17_ = plt_line17.copy().set_stroke(color=GREY).set_opacity(0.48)
        self.add(plt_line17_)
        self.play(mh.rtransform(plt_line17, plt_line18))
        self.wait(0.1)
        self.play(mh.rtransform(plt_line18.copy(), plt_line19))
        self.wait(0.1)
        self.play(FadeOut(plt_line19))
        self.wait(0.1)

        yvals13 = yvals10 - yvals11
        plt14 = ax.plot_line_graph(xvals5*scalex5, yvals13*scaley6+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt_line20 = ax.plot_line_graph(xvals5 * scalex5, xvals5 * 0 + 0.7, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        self.play(mh.rtransform(plt13, plt14, plt_line18, plt_line20, ticksy6, ticksy7),
                                plt_line17_.animate.shift(ax.c2p(0, 0.7 - 0.8)-origin))
        self.remove(plt_line17_)

        self.wait(0.1)

        scalex6 = 1/100000
        scaley7 = 3/100
        xvals7 = np.linspace(0., 100100, nplt*10)
        xvals8 = np.linspace(4., 100100, nplt)
        xvals7 = xvals7 * (xvals5[1] - xvals5[0]) / xvals7[1] + 4

        ticks6 = get_xticks(ax, [1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
                            [r'1\,000', r'5\,000', r'10\,000', r'50\,000', r'100\,000', r'500\,000', r'1\,000\,000'], scalex6)
        ticks6[:2].set_opacity(0)
        ticks6[4].shift(LEFT*0.3)
        ticksy8 = get_yticks(ax, [-40, -20, -10, 0, 10, 20], scaley=scaley7, center=0.7)

        yvals14 = (np.interp(xvals7+0.5, x, y, left=0, right=y[-1])
                   - expi(np.log(xvals7)) + expi(np.log(xvals7)/2)/2 + expi(np.log(2))/2)  # pi-Li+Li_2 up to 100k
        yvals15 = (np.interp(xvals8+0.5, x, y, left=0, right=y[-1])
                   - expi(np.log(xvals8)) + expi(np.log(xvals8)/2)/2 + expi(np.log(2))/2)  # pi-Li+Li_2 up to 100k
        plt15 = ax.plot_line_graph(xvals7*scalex5, yvals14*scaley6+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt17 = ax.plot_line_graph(xvals8*scalex6, yvals15*scaley7+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        self.wait(0.1)
        self.remove(plt14)
        self.add(plt15)
        self.play(mh.rtransform(plt15, plt17, ticks5[2:], ticks6[:-2], ticksy7[:], ticksy8[1:-1],
                                run_time=3., rate_func = rate_func_quad(0.2, 0.2)))

        scalex7 = 1/1e6
        scaley8 = 1.4/100

        xvals9 = np.linspace(0., 1001000, nplt*10)
        xvals10 = np.linspace(4., 1001000, nplt)
        xvals9 = xvals9 * (xvals8[1] - xvals8[0]) / xvals9[1] + 4.

        yvals16 = (np.interp(xvals9+0.5, x, y, left=0, right=y[-1])
                   - expi(np.log(xvals9)) + expi(np.log(xvals9)/2)/2 + expi(np.log(2))/2)  # pi-Li+Li_2 up to 1m
        yvals17 = (np.interp(xvals10+0.5, x, y, left=0, right=y[-1])
                   - expi(np.log(xvals10)) + expi(np.log(xvals10)/2)/2 + expi(np.log(2))/2)  # pi-Li+Li_2 up to 1m

        ticks7 = get_xticks(ax, [10_000, 50_000, 100_000, 500_000, 1_000_000],
                            [r'10\,000', r'50\,000', r'100\,000', r'500\,000', r'1\,000\,000'], scalex7)
        ticks7[:2].set_opacity(0)
        ticks7[4][1].shift(LEFT*0.4)
        # ticksy9 = get_yticks(ax, [-150, -100, -40, -20, -10, 0, 10, 20], scaley=scaley8, center=0.7)
        ticksy9 = get_yticks(ax, [-100, -40, -20, -10, 0, 10, 20], scaley=scaley8, center=0.7)
        VGroup(ticksy9[-4], ticksy9[-2]).set_opacity(0)

        plt18 = ax.plot_line_graph(xvals9*scalex6, yvals16*scaley7+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt19 = ax.plot_line_graph(xvals10*scalex7, yvals17*scaley8+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)

        self.wait(0.1)
        self.remove(plt17)
        self.add(plt18)

        self.play(mh.rtransform(plt18, plt19, ticks6[2:], ticks7[:], ticksy8, ticksy9[1:],
                                run_time=3., rate_func = rate_func_quad(0.2, 0.2)))


        scaley9 = 5/1000
        yvals18 = yvals17 - expi(np.log(xvals10)/2)/2 + expi(np.log(2))/2
        plt20 = ax.plot_line_graph(xvals10*scalex7, yvals18*scaley9+0.8, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)

        # ticksy10 = get_yticks(ax, [-150, -100, -50, -20, 0, 20], scaley=scaley9, center=0.8)
        ticksy10 = get_yticks(ax, [-100, -40, -20, 0, 20], scaley=scaley9, center=0.8)
        VGroup(ticksy10[-1], ticksy10[-4:-2]).set_opacity(0)

        self.play(mh.rtransform(plt19, plt20, ticksy9[:3], ticksy10[:3], ticksy9[-3::2], ticksy10[-2::]),
                  #               ticksy9[:2], ticksy10[:2], ticksy9[3], ticksy10[3], ticksy9[-3],
                  #               ticksy10[-2], ticksy9[-1], ticksy10[-1], ticksy9[2][0], ticksy10[2][0],
                  #               ticksy9[2][1][0], ticksy10[2][1][0], ticksy9[2][1][2:], ticksy10[2][1][2:]),
                  # mh.stretch_replace(ticksy9[2][1][1], ticksy10[2][1][1]),
                  plt_line20.animate.shift(ax.c2p(0, 0.8 - 0.7) - origin))

        self.wait()
