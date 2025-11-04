from manim import *
import numpy as np
import math
import sys
import scipy as sp

sys.path.append('../')
import manimhelper as mh


def bm_path(n=100, seed=1):
    np.random.seed(seed)
    t_vals = np.linspace(0, 1, n)
    y_vals = np.zeros(n)
    y_vals[1:] = np.random.normal(scale=np.sqrt(t_vals[1]), size=n - 1).cumsum()
    return t_vals, y_vals


def bm_path_hierarchy(depth=7, seed=1):
    np.random.seed(seed)
    n = 1 << depth
    t_vals = np.linspace(0, 1, n+1)
    y_vals = np.zeros(n)
    y_vals[-1] = np.random.normal()
    #rands = np.random.normal(size=n)
    for level in range(depth-1, -1, -1):
        start = 1 << level
        step = start << 1
        len = 1 << (depth-level-1)
        y_vals[start:step:n] = (y_vals[0:step:n] + y_vals[step:step]) * 0.5 + np.random.normal(scale=1, size=len)

class BMReflection(ThreeDScene):
    def construct(self):
        #t_vals, y_vals = bm_path_hierarchy(depth=5, seed=4)
        t_vals, y_vals = bm_path(n=400, seed=4)
        ymin = -1
        ymax = 1.5
        xlen = 10
        ylen = 6
        a = 1.

        ax = Axes(x_range=[0, 1.1], y_range=[ymin, ymax], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)

        plt1 = ax.plot_line_graph(t_vals, y_vals, line_color=BLUE, stroke_width=5, add_vertex_dots=False).set_z_index(5)
        line1 = DashedLine(ax.coords_to_point(0, a), ax.coords_to_point(1, a), stroke_width=4, stroke_color=GREY).set_z_index(1)
        origin = ax.coords_to_point(0, 0)
        self.add(ax, plt1, line1)
        self.wait(0.1)
        gp = VGroup(ax, plt1, line1)
        #self.play(Rotate(gp, angle=PI/6, about_point=origin, axis=UP), run_time=1)
        self.play(Rotate(plt1, angle=PI, about_point=origin, axis=RIGHT),
                  run_time=3)

        self.wait()


class BMPathIntro(Scene):
    bgcol = (BLACK, GREY)

    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = self.bgcol[0]
        else:
            config.background_color = self.bgcol[1]
        super().__init__(*args, **kwargs)

    def construct(self):
        tmax = 2.
        ymax = 1.5
        ymin = -1.5
        n = 800
        i1 = 500
        seeds = [11, 15, 20, 23, 25, 26, 32, 39, 40, 41, 51]
        seed = seeds[-1]
        np.random.seed(seed)
        ax = Axes(x_range=[0, tmax * 1.1], y_range=[ymin, ymax], x_length=8, y_length=4,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "stroke_opacity": 1,
                               },
                  ).set_z_index(2)
        box1 = SurroundingRectangle(ax, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.7,
                                    buff=0.05, corner_radius=0.2).set_opacity(0)
        VGroup(ax, box1).to_edge(DOWN, buff=0.1)
        times = np.linspace(0., tmax, n)
        t1 = times[i1]

        self.add(ax, box1)

        dt = times[1]
        dev = math.sqrt(dt)
        bmpath = np.zeros(n)
        while True:
            bmpath[1:] = np.random.normal(scale=dev, size=n - 1).cumsum()
            zeros1 = [i for i in range(1, i1) if bmpath[i] * bmpath[i-1] < 0]
            zeros2 = [i for i in range(i1, n) if bmpath[i] * bmpath[i-1] < 0]
            if len(zeros1) != 0 and len(zeros2) != 0:
                if bmpath[i1] < 0: bmpath = -bmpath
                i2 = zeros1[-1]
                i3 = zeros2[0]
                if not (200 < i2 < 300) or not (650 < i3 < 750):
                    continue
                if abs(bmpath[i2-1]) < abs(bmpath[i2]): i2 -=1
                if abs(bmpath[i3-1]) < abs(bmpath[i2]): i3 -=1
                bmpath[i2] = 0.
                bmpath[i3] = 0.
                t2 = times[i2]
                t3 = times[i3]
                print(t3)
                break

        bmpath[:i2] *= -1

        sw = 3
        sw2 = 3

        bmplot1 = ax.plot_line_graph(times, bmpath, add_vertex_dots=False, stroke_width=sw, stroke_color=YELLOW).set_z_index(3)
        col1 = YELLOW
        col2 = RED
        col3 = GREEN
        col4 = BLUE
        self.play(Create(bmplot1, rate_func=linear, run_time=2))
        self.wait(0.1)
        graph1 = VGroup(ax, bmplot1)
        scale = 0.7
        gp1 = VGroup(graph1, box1).copy().scale(scale).to_corner(UL, buff=0.02)
        gp1[1].set_opacity(0.65)
        #self.play(VGroup(graph1, box1).animate.scale(scale).to_corner(UL, buff=0.1))
        self.play(mh.transform(graph1, gp1[0], box1, gp1[1]))
        line1 = DashedLine(ax.coords_to_point(t1, ymin), ax.coords_to_point(t1, ymax),
                           color=GREY, stroke_width=2, dash_length=(ymax-ymin)/40).set_z_index(1)

        bm_func = sp.interpolate.make_interp_spline(times, bmpath, k=1)

        dot1 = Dot(radius=0.1, color=YELLOW).set_z_index(6)
        t_val = ValueTracker(0.)
        def redraw():
            t = t_val.get_value()
            h = t1 * t
            t_l = t1 - h
            t_r = min(t1 + h, t3)
            op_r = min(1, (t3 - t_r)*8)
            op_l = min(1, t_l*8)
            pt_l = ax.coords_to_point(t_l, bm_func(t_l))
            pt_r = ax.coords_to_point(t_r, bm_func(t_r))
            i_l = round(t_l / tmax * (n-1))
            i_r = round(t_r / tmax * (n-1))
            i_l1 = max(i_l, i2)
            dot_l = dot1.copy().move_to(pt_l).set_opacity(op_l).set_color(col2)
            dot_r = dot1.copy().move_to(pt_r).set_opacity(op_r)
            if t > 0:
                dot_r.set_color(col4)
                if i_l < i_l1:
                    dot_l.set_color(col2)
                else:
                    dot_l.set_color(col3)
            res = [dot_l, dot_r,
                   ax.plot_line_graph(times[i1:i_r + 1], bmpath[i1:i_r + 1], add_vertex_dots=False, stroke_width=sw,
                                      stroke_color=col4).set_z_index(5),
                   ax.plot_line_graph(times[i_r:], bmpath[i_r:], add_vertex_dots=False, stroke_width=sw,
                                      stroke_color=col1).set_z_index(5),
                   ax.plot_line_graph(times[i_l1:i1 + 1], bmpath[i_l1:i1 + 1], add_vertex_dots=False, stroke_width=sw,
                                         stroke_color=col3).set_z_index(5),
                   ax.plot_line_graph(times[:i_l + 1], bmpath[:i_l + 1], add_vertex_dots=False, stroke_width=sw,
                                         stroke_color=col1).set_z_index(5)]
            if i_l < i_l1:
                res.append(ax.plot_line_graph(times[i_l:i_l1 + 1], bmpath[i_l:i_l1 + 1], add_vertex_dots=False, stroke_width=sw,
                                         stroke_color=col2).set_z_index(5))

            return VGroup(*res)

        draw_obj = always_redraw(redraw)
        self.play(FadeIn(draw_obj, line1), run_time=0.5)
        self.remove(bmplot1)
        self.play(t_val.animate(rate_func=linear, run_time=2.5).set_value(1.))
        self.remove(draw_obj)
        draw_obj = redraw()
        self.add(draw_obj)

        #graph1 = VGroup(ax, line1, draw_obj)

        br1 = ax.plot_line_graph(times[:i2+1], bmpath[:i2+1], add_vertex_dots=False, stroke_width=sw,
                                 stroke_color=col2).set_z_index(2.5)
        scale2 = 1
        ax2 = ax.copy().scale(scale2).to_corner(DL)
        s = tmax / times[i2]
        br_times = times[:i2+1] * s
        br_values = bmpath[:i2+1] * math.sqrt(s)
        br2 = ax2.plot_line_graph(br_times, br_values, add_vertex_dots=False, stroke_width=sw2,
                                 stroke_color=col2).set_z_index(2.5)

        exc1 = ax.plot_line_graph(times[i2:i3+1], bmpath[i2:i3+1], add_vertex_dots=False, stroke_width=sw,
                                 stroke_color=col4).set_z_index(2.6)
        exc1_1 = ax.plot_line_graph(times[i2:i1+1], bmpath[i2:i1+1], add_vertex_dots=False, stroke_width=sw,
                                 stroke_color=col3).set_z_index(2.6)
        exc1_2 = ax.plot_line_graph(times[i1:i3+1], bmpath[i1:i3+1], add_vertex_dots=False, stroke_width=sw,
                                 stroke_color=col4).set_z_index(2.6)
        ax3 = ax.copy().scale(scale2).to_corner(UR)
        shift = ax3.coords_to_point(0, ymin) - ax3.coords_to_point(0, 0)
        ax3.x_axis.shift(shift)
        s = tmax / (times[i3] - times[i2])
        exc_times = (times[i2:i3+1] - times[i2]) * s
        exc_values = bmpath[i2:i3+1] * math.sqrt(s) + ymin
        exc2 = ax3.plot_line_graph(exc_times, exc_values, add_vertex_dots=False, stroke_width=sw2,
                                 stroke_color=col4).set_z_index(2.6)
        exc2_1 = ax3.plot_line_graph(exc_times[:i1-i2+1], exc_values[:i1-i2+1], add_vertex_dots=False, stroke_width=sw2,
                                 stroke_color=col3).set_z_index(2.6)
        exc2_2 = ax3.plot_line_graph(exc_times[i1-i2:], exc_values[i1-i2:], add_vertex_dots=False, stroke_width=sw2,
                                 stroke_color=col4).set_z_index(2.6)


        mea1 = ax.plot_line_graph(times[i2:i1+1], bmpath[i2:i1+1], add_vertex_dots=False, stroke_width=sw,
                                 stroke_color=col3).set_z_index(2.7)
        ax4 = ax.copy().scale(scale2).to_corner(DR)
        ax4.x_axis.shift(shift)
        s = tmax / (times[i1] - times[i2])
        mea_times = (times[i2:i1+1] - times[i2]) * s
        mea_values = bmpath[i2:i1+1] * math.sqrt(s) + ymin
        mea2 = ax4.plot_line_graph(mea_times, mea_values, add_vertex_dots=False, stroke_width=sw2,
                                 stroke_color=col3).set_z_index(2.6)

        box3 = Rectangle(width=config.frame_x_radius*2, height=config.frame_y_radius*2, fill_color=BLACK, fill_opacity=1,
                         stroke_width=0, stroke_opacity=0)

        print('box:', box1.width, box1.height)
        self.play(#graph1.animate.scale(scale).to_corner(UL),
                  mh.rtransform(br1, br2, exc1_1, exc2_1, exc1_2, exc2_2, mea1, mea2),
                  mh.rtransform(ax.copy(), ax2, ax.copy(), ax3, ax.copy(), ax4),
                  VGroup(ax, draw_obj, line1).animate.to_corner(UL),
                  FadeIn(box3), FadeOut(box1),
                  run_time=2)

        fs = 40
        eq1 = MathTex(r'\mathbb E[X^s]', r'=', r'2^{-\frac s2}s(1-2^{1-s})\Gamma(\!{}^\frac s2\!)\zeta(s)', font_size=40).set_z_index(10)
        eq1.move_to(ax2.coords_to_point(tmax * 0.6, ymax * -0.6))
        eq1[2][-8:-5].move_to(eq1[1], coor_mask=UP)
        eq2 = MathTex(r'\mathbb E[X^s]', r'=', r'2^{-\frac s2}s(s-1)\Gamma({}^\frac s2)\zeta(s)', font_size=40).set_z_index(10)
        eq2.move_to(ax3.coords_to_point(tmax * 0.55, ymax * 0.8))
        eq2[2][-8:-5].move_to(eq2[1], coor_mask=UP)
        eq3 = MathTex(r'\mathbb E[X^s]', r'=', r'2^{\frac s2}s(1\!-\!2^{1-s})\Gamma({}^\frac s2)\zeta(s)', font_size=40).set_z_index(10)
        eq3.move_to(ax4.coords_to_point(tmax * 0.55, ymax * 0.8))
        eq3[2][-8:-5].move_to(eq3[1], coor_mask=UP)

        tmax1 = br_times[np.argmax(br_values)]
        ymax1 = max(br_values)
        arr1 = Arrow(ax2.coords_to_point(tmax1, 0), ax2.coords_to_point(tmax1, ymax1-0.05), color=YELLOW, buff=0).set_z_index(10)
        larr1 = Line(arr1.get_start(), arr1.get_end()+DOWN*0.1, stroke_width=arr1.stroke_width, stroke_color=arr1.stroke_color).set_z_index(10)
        tmax2 = exc_times[np.argmax(exc_values)]
        ymax2 = max(exc_values)
        arr2 = Arrow(ax3.coords_to_point(tmax2, ymin), ax3.coords_to_point(tmax2, ymax2-0.05), color=YELLOW, buff=0).set_z_index(10)
        larr2 = Line(arr2.get_start(), arr2.get_end()+DOWN*0.1, stroke_width=arr2.stroke_width, stroke_color=arr2.stroke_color).set_z_index(10)
        tmax3 = mea_times[np.argmax(mea_values)]
        ymax3 = max(mea_values)
        arr3 = Arrow(ax4.coords_to_point(tmax3, ymin), ax4.coords_to_point(tmax3, ymax3-0.05), color=YELLOW, buff=0).set_z_index(10)
        larr3 = Line(arr3.get_start(), arr3.get_end()+DOWN*0.1, stroke_width=arr3.stroke_width, stroke_color=arr3.stroke_color).set_z_index(10)
        eq4 = MathTex(r'XXX', font_size=40)[0].set_z_index(10)
        eq4[0].next_to(larr1, LEFT, buff=0.1)
        eq4[1].next_to(larr2, LEFT, buff=0.1)
        eq4[2].next_to(larr3, LEFT, buff=0.1)
        eq4.shift(DOWN*0.2)

        self.play(Create(larr1), Create(larr2), Create(larr3),
                  FadeIn(arr1.tip, arr2.tip, arr3.tip, eq4))
        self.wait(0.1)
        self.play(FadeIn(eq1, eq2, eq3))
        self.wait(0.1)
        circ1 = mh.circle_eq(eq1[-1][-2:-1].copy().stretch(1.1, 0))
        circ2 = mh.circle_eq(eq2[-1][-1:])
        circ3 = mh.circle_eq(eq3[-1][-1:])
        self.play(eq1[-1][-4:].animate.scale(1.4, about_edge=LEFT),
                  eq2[-1][-4:].animate.scale(1.4, about_edge=LEFT),
                  eq3[-1][-4:].animate.scale(1.4, about_edge=LEFT),
                  LaggedStart(Create(circ1), Create(circ2), Create(circ3), run_time=1., lag_ratio=0.5), rate_func=linear)


        self.wait()


class Test(Scene):
    def construct(self):
        tmax = 2.
        ymax = 1.5
        ymin = -1.5
        ax = Axes(x_range=[0, tmax * 1.1], y_range=[ymin, ymax], x_length=8, y_length=4,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "stroke_opacity": 1,
                               },
                  ).set_z_index(2)
        n = 20
        times = np.linspace(0., tmax, n)
        yvals = np.random.uniform(ymin, ymax, n)
        plt = ax.plot_line_graph(times, yvals, add_vertex_dots=True, stroke_width=2,
                                 stroke_color=BLUE).set_z_index(2.6)
        self.add(ax, plt)

class CountT(Scene):
    def construct(self):
        t = 0.49
        eq1 = MathTex(r'T =', r'0.88', font_size=50)

        tval = ValueTracker(0.)
        def f():
            t = tval.get_value()
            tstr = r'{:.2f}'.format(t)
            eq2 = MathTex(tstr, font_size=50)[0]
            mh.align_sub(eq2, eq2[0], eq1[1][0])
            return VGroup(eq1[0], eq2)

        obj = always_redraw(f)
        self.add(obj)
        self.wait(0.1)
        self.play(tval.animate.set_value(t), rate_func=linear, run_time=t*6)
        self.wait()


class MellinT(Scene):
    def construct(self):
        eq1 = MathTex(r'\mathbb E[T^s]', r'=', r'2s\pi^{-s}(1-2^{1-2s})\Gamma(\!s\!)\zeta(2s)', font_size=50).set_z_index(2)
        eq1[2][-7].move_to(eq1[1], coor_mask=UP)
        box = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_opacity=0.4, fill_color=BLACK,
                                   corner_radius=0.1, buff=0.1)
        circ1 = mh.circle_eq(eq1[2][-3:]).set_z_index(2).shift(RIGHT*0.1+DOWN*0.05)
        self.add(eq1, box)
        self.wait(0.1)
        self.play(eq1[-1][-5:].animate.scale(1.4, about_edge=LEFT),
                  Create(circ1))

        self.wait()


class Particles(BMPathIntro):
    dims = (5.73, 2.87)
    time = 30.
    scale = 0.3

    def construct(self):
        width, height = self.dims
        w2 = width * 1.2
        h2 = height * 1.2
        ndots = 100
        time=self.time
        seeds = [1]
        np.random.seed(seeds[-1])
        pts = [[RIGHT * np.random.uniform(0., w2) + UP * np.random.uniform(0., h2)] for _ in range(ndots)]
        nframes = round(time * 30)
        s = self.scale * math.sqrt(1./30)
        for _ in range(nframes):
            for i in range(ndots):
                dB = np.random.normal(0., s, 2)
                p = pts[i][-1].copy()
                p += RIGHT * dB[0] + UP * dB[1]
                pts[i].append(p)
        for i in range(nframes+1):
            for j in range(ndots):
                k = round(pts[j][i][0] / w2)
                l = round(pts[j][i][1] / h2)
                pts[j][i] -= k * w2 * RIGHT + l * h2 * UP

        dots = []
        for p in pts:
            d = np.random.uniform(1., 1.5)
            col = ManimColor(np.random.uniform(0.5, 1, size=3))
            dots.append(Dot(radius=0.07/d, color=col).move_to(p[0]).set_z_index(1/d))
        dotg = VGroup(dots)

        tval = ValueTracker(0.)
        def f(obj: VGroup):
            t = tval.get_value()
            i = round(t * 30)
            print(i, pts[0][i])
            for j in range(ndots):
                obj[j].move_to(pts[j][i])

        box = Rectangle(width=width, height=height, fill_color=BLACK, fill_opacity=0.65, stroke_width=0,
                        stroke_opacity=0)

        self.add(dotg, box)
        dotg.add_updater(f)
        self.play(tval.animate.set_value(time), run_time=time, rate_func=linear)
        dotg.remove_updater(f)

class ParticlesFull(Particles):
    bgcol = (BLACK, BLACK)
    dims = (config.frame_x_radius*2, config.frame_y_radius*2)
    scale = 0.2
    time = 4

class Zeta(BMPathIntro):
    def construct(self):
        eq1 = MathTex(r'\zeta(s)', r'=', r'1+\frac1{2^s}+\frac1{3^3}', r'+\frac1{4^s}+\frac1{5^s}+\cdots',
                      font_size=60).set_z_index(2)
        eq1[3].next_to(eq1[2], DOWN, buff=0.2).align_to(eq1[2], LEFT)
        box = SurroundingRectangle(eq1, stroke_opacity=0, stroke_width=0, fill_opacity=0.65, fill_color=BLACK,
                                   corner_radius=0.2, buff=0.2)
        self.add(eq1[:2], box)
        self.wait(0.1)
        self.play(LaggedStart(
            FadeIn(eq1[2][0], run_time=0.4, rate_func=linear),
            FadeIn(eq1[2][1:6], run_time=0.4, rate_func=linear),
            FadeIn(eq1[2][6:11], run_time=0.4, rate_func=linear),
            FadeIn(eq1[3][:5], run_time=0.4, rate_func=linear),
            FadeIn(eq1[3][5:10], run_time=0.4, rate_func=linear),
            FadeIn(eq1[3][10:], run_time=0.4, rate_func=linear),
            lag_ratio=0.6))
        self.wait()

class RH(Scene):
    def construct(self):
        eq = MathTex(r'{\sf zeros\ occur\ at\ }', r'x=\frac12', r'\sf ?', stroke_width=2)
        self.add(eq)


def mathtex(*args, t2c: dict=None, **kwargs):
    if t2c is None:
        t2c = {}
    if 't2c' in mathtex.defaults:
        t2c = mathtex.defaults['t2c']
    res = MathTex(*args, **kwargs)
    for i, arg in enumerate(args):
        subtex = MathTex(arg, substrings_to_isolate=list(t2c.keys()))
        length = 0
        for elem in subtex[:]:
            j = len(elem[:])
            tex = elem.get_tex_string()
            if tex in t2c:
                res[i][length:length+j].set_color(t2c[tex])
            length += j
        assert length == len(res[i][:])

    return res

mathtex.defaults = {}

def __mathtex_set_defaults__(**kwargs):
    for key, val in kwargs.items():
        mathtex.defaults[key] = val

mathtex.set_defaults = __mathtex_set_defaults__

class Xi(Scene):
    def construct(self):
        MathTex.set_default(font_size=100)
        #mathtex.set_defaults(t2c={'s': YELLOW, 'X': BLUE, r'\xi': BLUE, r'\zeta': BLUE})
        eq1 = MathTex(r'\zeta(s)', font_size=120)
        eq1_1 = MathTex(r'\zeta(s)', r'=', r'1+2^{-s}+3^{-s}+4^{-s}+\cdots')
        eq1_1[1:].scale(0.9, about_edge=LEFT)
        eq1_1.move_to(ORIGIN, coor_mask=RIGHT)
        eq2 = MathTex(r'=', r'\pi^{-\frac s2}\Gamma(\!{}^{\frac s2}\!)\zeta(s)')
        eq2[1][-8:-5].move_to(eq2[0], coor_mask=UP)
        mh.align_sub(eq2[1], eq2[1][-1], eq1[0][-1], coor_mask=UP).move_to(RIGHT*2.4, coor_mask=RIGHT)
        eq3 = MathTex(r'=', r's(s-1)', r'\pi^{-\frac s2}\Gamma(\!{}^{\frac{s}{2} }\!)\zeta(s)')
        eq3[2][-8:-5].move_to(eq3[0], coor_mask=UP)
        mh.align_sub(eq3[1:], eq3[2][-1], eq1[0][-1], coor_mask=UP).move_to(ORIGIN, coor_mask=RIGHT)
        eq4 = MathTex(r'\xi(s)', r'=', r'\frac12', r's(s-1)', r'\pi^{-\frac s2}\Gamma(\!{}^{\frac{s}{2} }\!)\zeta(s)')
        eq4[4][-8:-5].move_to(eq4[1], coor_mask=UP)
        mh.align_sub(eq4, eq4[4][-1], eq1[0][-1], coor_mask=UP).move_to(ORIGIN, coor_mask=RIGHT)
        eq4_2 = MathTex(r'\xi(s)', r'=', r'\mathbb E[X^s]')
        mh.align_sub(eq4_2, eq4_2[1], eq4[1]).move_to(ORIGIN, coor_mask=RIGHT)
        eq5 = MathTex(r'2\xi(s)', r'=', r's(s-1)', r'\pi^{-\frac s2}\Gamma(\!{}^{\frac{s}{2} }\!)\zeta(s)')
        eq5[-1][-8:-5].move_to(eq5[1], coor_mask=UP)
        mh.align_sub(eq5, eq5[1], eq4[1], coor_mask=UP).move_to(ORIGIN, coor_mask=RIGHT)
        eq6 = MathTex(r'2\xi(s)', r'=', r'\mathbb E[X^s]')
        mh.align_sub(eq6, eq6[1], eq5[1], coor_mask=UP)
        eq7 = MathTex(r'2\xi(s)', r'=', r'\mathbb E[e^{s\log X}]')
        mh.align_sub(eq7, eq7[1], eq5[1], coor_mask=UP)
        eq8 = MathTex(r'2\xi(s)', r'=', r'\mathbb E[X^s]', r'=', r'\int_0^\infty p_X(x)x^s\,dx')
        eq8[4].scale(0.8, about_edge=LEFT)
        mh.align_sub(eq8, eq8[1], eq5[1], coor_mask=UP).move_to(ORIGIN, coor_mask=RIGHT)

        VGroup(eq1[0][0], eq2[1][-4], eq3[2][-4], eq4[0][0], eq4[-1][-4], eq5[0][1],
              eq5[-1][-4], eq6[0][1], eq6[-1][-3], eq7[0][1], eq7[-1][-2], eq1_1[0][0],
               eq4_2[0][0], eq4_2[2][-3], eq8[0][1], eq8[2][-3], eq8[4][4], eq8[4][6],
               eq8[4][-1], eq8[4][8]).set_color(BLUE)
        VGroup(eq1[0][2], eq2[-1][-2], eq2[-1][2], eq2[-1][7], eq3[1][0], eq3[1][2],
              eq3[2][-2], eq3[2][2], eq3[2][7], eq4[0][2], eq4[3][0], eq4[3][2], eq4[4][-2],
              eq4[4][2], eq4[4][7], eq5[0][3], eq5[2][0], eq5[2][2], eq5[3][-2], eq5[3][2],
               eq5[3][7], eq6[0][3], eq6[2][3], eq7[0][3], eq7[2][3], eq1_1[0][2],
               eq1_1[2][4], eq1_1[2][8], eq1_1[2][12], eq4_2[0][2], eq4_2[2][3], eq8[0][3],
               eq8[2][3], eq8[4][9]).set_color(YELLOW)
        eq4_1 = eq4[2:].copy().move_to(ORIGIN, coor_mask=RIGHT)

        self.add(eq1_1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1_1[0][:], eq1[0][:]),
                  FadeOut(eq1_1[1:]), run_time=1.4)
        self.add(eq1)
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq1[0][:], eq2[1][-4:], run_time=1),
                  FadeIn(eq2[1][:-4], run_time=1.4),
                              lag_ratio=0.2))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq2[1][:], eq3[2][:], run_time=1),
                  FadeIn(eq3[1], run_time=1.4),
                              lag_ratio=0.2))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq3[1:], eq4_1[1:], run_time=1),
                  FadeIn(eq4_1[0], run_time=1.4),
                              lag_ratio=0.2))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq4_1, eq4[2:], run_time=1),
                  FadeIn(eq4[:2], run_time=1.4),
                              lag_ratio=0.2))
        self.wait(0.1)
        shift = mh.diff(eq4[1], eq4_1[1])
        eq4_3 = eq4.copy()
        self.play(mh.rtransform(eq4[:2], eq4_2[:2]),
                  FadeOut(eq4[2:]), FadeIn(eq4_2[2], shift=shift), run_time=1.4)
        eq4 = eq4_3
        self.wait(0.1)
        cross = [Line(eq4_2[2].get_corner(x), eq4_2[2].get_corner(y), stroke_width=8, stroke_color=RED).set_opacity(2)
                 for x, y in ((UL, DR), (DL, UR))]
        self.play(Create(cross[0]), run_time=0.6, rate_func=linear)
        self.play(Create(cross[1]), run_time=0.6, rate_func=linear)
        self.wait(0.1)
        self.play(mh.rtransform(eq4_2[:2], eq4[:2]),
                  FadeIn(eq4[2:]), FadeOut(eq4_2[2], *cross), run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq4[3:], eq5[2:], eq4[1], eq5[1], eq4[0][:], eq5[0][1:],
                                eq4[2][2], eq5[0][0]),
                  FadeOut(eq4[2][:2]),
                  run_time=1.6)
        self.wait(0.1)
        self.play(FadeOut(eq5[2:]),
                  FadeIn(eq6[2]),
                  mh.rtransform(eq5[:2], eq6[:2]),
                  run_time=2)
        self.wait(0.1)
        eq6_1 = eq6.copy()
        self.play(mh.rtransform(eq6[:2], eq7[:2], eq6[2][:2], eq7[2][:2],
                                eq6[2][3], eq7[2][3], eq6[2][4], eq7[2][-1]),
                  mh.stretch_replace(eq6[2][2], eq7[2][-2]),
                  FadeIn(eq7[2][2], target_position=eq6[2][2]),
                  FadeIn(eq7[2][4:-2], shift=mh.diff(eq6[2][3], eq7[2][3])),
                  run_time=4, rate_func=there_and_back_with_pause)
        eq7.set_opacity(0)
        self.add(eq6_1)
        self.wait(0.1)
        self.play(mh.rtransform(eq6_1[:], eq8[:3]),
                  FadeIn(eq8[3:]),
                  run_time=1.4)
        self.wait()

class GammaDef(Scene):
    border = False
    def construct(self):
        sc = BLACK if self.border else WHITE
        eq1 = MathTex(r'\Gamma(s)', r'=', r'\int_0^\infty x^{s-1}e^{-x}dx', stroke_width=8, stroke_color=sc)
        if not self.border:
            VGroup(eq1[0][2], eq1[2][4]).set_color(YELLOW)
        self.add(eq1)

class SReflect(Scene):
    def construct(self):
        eq1 = MathTex(r's\to1-s', stroke_width=2, font_size=100)[0]
        VGroup(eq1[0], eq1[-1]).set_color(YELLOW)
        self.add(eq1)

class FunctionalEq(Scene):
    def construct(self):
        eq1 = MathTex(r'\xi(1-s)', r'=', r'\xi(s)', font_size=100, stroke_width=1)
        VGroup(eq1[0][0], eq1[2][0]).set_color(BLUE)
        VGroup(eq1[0][4], eq1[2][2]).set_color(YELLOW)
        self.add(eq1)

def psi(x, eps=0.001, maxn=10):
    if x < 1e-6:
        return 0
    if x < 1:
        return psi(1.0/x, eps, maxn) / (x * x * x)
    n = 1
    y = x * x
    p = 0.0
    while True:
        m = n * n
        e = math.exp(- PI * m * y)
        p += m * (PI * 2 * m * y - 3) * e
        if n > maxn and e < eps:
            break
        n += 1
    return PI * 4 * x * p

def phi2(x, eps=0.001, maxn=10):
    n = 1
    y = x * x
    p = 0.0
    while True:
        m = (n - 0.5)*(n - 0.5)
        e = math.exp(-m * y * PI)
        p += (PI * 2 * m * y - 1) * e
        if n > maxn and e < eps:
            break
        n += 1
    return y * 2 * p


def phi(x, eps=0.001, maxn=10):
    if x < 1e-6:
        return 0
    if x < 1:
        return phi2(1.0/x, eps, maxn)
    n = 1
    y = x * x
    p = 0.0
    sgn = 1
    while True:
        m = n * n
        e = math.exp(-m * y * PI)
        p += sgn * m * e
        if n > maxn and e < eps:
            break
        n += 1
        sgn = -sgn
    return PI * 4 * x * p

class Kuiper(Scene):
    scale=1
    text_scale = 1
    #scale = 0.6
    #text_scale = 0.7

    def get_p(self):
        a = math.sqrt(2/PI)
        return lambda x: psi(x * a)

    def get_eq(self):
        eq = MathTex(r'p_X(x)', font_size=50 * self.text_scale)
        VGroup(eq[0][1], eq[0][3]).set_color(BLUE)
        return eq

    def construct(self):
        s = self.scale
        s2 = self.text_scale
        xlen = 8.8 * s
        ylen = 4.4 * s
        xmax = 2.5
        ymax = 1.9
        n = 200
        ax = Axes(x_length=xlen, y_length = ylen, x_range=(0, xmax * 1.05, 1), y_range=(0, ymax * 1.1),
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  x_axis_config={'include_ticks': True}
                  )
        eqs = MathTex(r'012', font_size=45 * s2)[0]
        for i, x in enumerate([0, 1, 2]):
            eqs[i].next_to(ax.coords_to_point(x, 0), DOWN, buff=0.2)
        eq1 = MathTex(r'x', font_size=50*s).next_to(ax.x_axis.get_right(), UP, buff=0.25).shift(LEFT*0.15)
        eq2 = self.get_eq().next_to(ax.y_axis.get_top(), RIGHT, buff=0.25).shift(DOWN*0.2)
        VGroup(eq1[0][0]).set_color(BLUE)
        p = self.get_p()

        plot = ax.plot(p, x_range=(0, xmax), stroke_color=BLUE, stroke_width=5).set_z_index(3)
        area = ax.get_area(plot, color=ManimColor(BLUE.to_rgb()*0.5), opacity=1, x_range=(0., xmax), stroke_width=0, stroke_opacity=0).set_z_index(1)

        self.add(ax, eqs, eq1, eq2)
        self.wait(0.1)
        self.play(LaggedStart(Create(plot, run_time=2, rate_func=linear),
                              FadeIn(area, run_time=2, rate_func=linear), lag_ratio=0.4))
        self.wait()

class Kuiper2(Kuiper):
    scale = 0.6
    text_scale = 0.7

    def get_eq(self):
        eq = MathTex(r'p_V(x)', font_size=50 * self.text_scale)
        VGroup(eq[0][1], eq[0][3]).set_color(BLUE)
        return eq

class Kolmogorov(Kuiper):
    scale = 0.6
    text_scale = 0.7

    def get_eq(self):
        eq = MathTex(r'p_D(x)', font_size=50 * self.text_scale)
        VGroup(eq[0][1], eq[0][3]).set_color(BLUE)
        return eq

    def get_p(self):
        a = math.sqrt(2/PI)
        return lambda x: phi(x * a) * 0.88

class KolmogorovP(Scene):
    def construct(self):
        eq1 = MathTex(r'p_D(x)', r'=', r'8xe^{-2x^2}\!\!\!-32xe^{-8x^2}\!\!\!+\cdots',
                      r'=', r'8x\sum_{n=1}^\infty (-1)^{n-1}n^2e^{-2n^2x^2}')
        eq1[3:].next_to(eq1[2], DOWN).align_to(eq1[1], LEFT)
        self.add(eq1[:3])
        self.wait(0.1)
        self.play(FadeIn(eq1[3:]), run_time=1.2)
        self.wait()


class KuiperP(Scene):
    def construct(self):
        eq1 = MathTex(r'p_V(x)', r'=', r'8x(3-4x^2)e^{-2x^2}\!\!\!', r'+32x(3-16x^2)e^{-8x^2}\!\!\!+\cdots',
                      r'=', r'8x\sum_{n=1}^\infty n^2(3-4n^2x^2)e^{-2n^2x^2}')
        eq1[3].next_to(eq1[2], DOWN, buff=0.).align_to(eq1[2][2], LEFT)
        eq1[4:].next_to(eq1[3], DOWN).align_to(eq1[1], LEFT)
        eq1.move_to(ORIGIN)
        VGroup(eq1[3], eq1[4:]).shift(LEFT)
        self.add(eq1[:4])
        self.wait(0.1)
        self.play(FadeIn(eq1[4:]), run_time=1.2)
        self.wait()

class Measure(Scene):
    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = WHITE
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        fs = 50
        MathTex.set_default(font_size=fs)
        eq1 = Tex(r'{\sf Heights:}\ ', r'$X_1, X_2, X_3,\ldots,X_n$', font_size=fs)

        eq2 = MathTex(r'F_n(x)', r'=', r'\frac1n\#\{{\sf elements\ in\ sample} \le x\}', font_size=fs)
        eq3 = MathTex(r'F_n(x)', r'=', r'\frac1n\sum_{k=1}^nI(X_k\le x)', font_size=fs)
        eq4 = MathTex(r'\mathbb P(X_k\le x)', r'=', r'F(x)', font_size=fs)

        eq2.next_to(eq1, DOWN).align_to(eq1, LEFT)
        mh.align_sub(eq3, eq3[1], eq2[1])
        VGroup(eq1, eq2, eq3).to_edge(DL, buff=0.6).next_to(mh.pos((-0.83, -0.2)), DR, buff=0)
        eq4.next_to(eq3, DOWN).align_to(eq1, LEFT)

        eq1_1 = eq1[1][:2].copy().to_edge(DR, buff=0.8).move_to(mh.pos((0.82, -0.2)))
        eq1_2 = eq1[1][3:5].copy().to_edge(DR, buff=0.8).move_to(mh.pos((0.82, -0.2)))
        eq1_3 = eq1[1][6:8].copy().to_edge(DR, buff=0.8).move_to(mh.pos((0.82, -0.2)))

        self.add(eq1[0])
        self.wait(0.1)
        self.play(FadeIn(eq1_1), run_time=0.7)
        self.wait(0.1)
        self.play(ReplacementTransform(eq1_1, eq1[1][:2]),
                  FadeIn(eq1[1][2], rate_func=rush_into), run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq1_2), run_time=0.7)
        self.wait(0.1)
        self.play(ReplacementTransform(eq1_2, eq1[1][3:5]),
                  FadeIn(eq1[1][5], rate_func=rush_into), run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq1_3), run_time=0.7)
        self.wait(0.1)
        self.play(ReplacementTransform(eq1_3, eq1[1][6:8]),
                  FadeIn(eq1[1][8], rate_func=rush_into), run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq1[1][9:]))
        self.wait(0.1)
        self.play(FadeIn(eq2))
        self.wait(0.1)

        n_sample = 15
        xlen = 5
        ylen = 2.8
        seeds = [5, 6, 10, 17, 18]
        xmax = 1.2
        x0 = 0.1
        np.random.seed(seeds[0])
        ax = Axes(x_range=[0, xmax * 1.05], y_range=[0, 1.2], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)

        x_rand = np.random.beta(2, 2, n_sample) + x0
        def f(x): return sp.special.betainc(2, 2, min(max(x-x0, 0),1))

        ax.next_to(mh.pos((0.1, -0.15)), DR, buff=0)

        ticks = [ax.x_axis.get_tick(x) for x in x_rand]
        eqax1 = MathTex(r'x', font_size=45).next_to(ax.x_axis.get_right(), UP, buff=0.15)
        eqx = MathTex(r'X_1', r'X_2', r'X_3', r'X_n', font_size=32)
        eqx.next_to(ax.x_axis, DOWN, buff=0.1)
        for i in [0, 1, 2, -1]:
            eqx[i].move_to(ticks[i], coor_mask=RIGHT)
        eqx[2].move_to(ticks[3], coor_mask=RIGHT)
        axline = DashedLine(ax.coords_to_point(0, 1), ax.coords_to_point(xmax, 1), stroke_width=4, stroke_color=GREY).set_z_index(1)
        axlabel = MathTex(r'0', r'1', font_size=32)
        axlabel[0].move_to(ax.coords_to_point(0, 0), aligned_edge=RIGHT).shift(LEFT*0.1)
        axlabel[1].move_to(ax.coords_to_point(0, 1), aligned_edge=RIGHT).shift(LEFT*0.1)

        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq2[:2], eq3[:2], eq2[2][-3:-1], eq3[2][-3:-1], eq2[2][:3], eq3[2][:3]),
                  mh.fade_replace(eq2[2][-1], eq3[2][-1]),
                  mh.fade_replace(eq2[2][4], eq3[2][8]),
                  FadeOut(eq2[2][5:-3], eq2[2][3]),
                  FadeIn(eq3[2][9:-3], eq3[2][3:8]), run_time=1),
                  FadeIn(ax, eqax1, axline, axlabel, run_time=1), lag_ratio=0.3))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[1][:2].copy(), eqx[0][:], eq1[1][3:5].copy(), eqx[1][:],
                                eq1[1][6:8].copy(), eqx[2][:], eq1[1][-2:].copy(), eqx[-1][:]),
                  FadeIn(*ticks),
                  run_time=1.5)

        ps = [(0,0)]
        x_rand.sort()
        for x in x_rand:
            y = ps[-1][1]
            ps.append((x, y))
            ps.append((x, y+1/n_sample))
        ps.append((xmax, 1))

        tval = ValueTracker(0.)
        def emp():
            x1 = tval.get_value() * xmax
            lines = []
            for i in range(0, n_sample*2 + 1, 2):
                if i > 0:
                    lines.append(Line(ax.coords_to_point(*ps[i-1]), ax.coords_to_point(*ps[i]), stroke_color=BLUE,
                                  stroke_width=2).set_z_index(4))
                x, y = ps[i+1]
                lines.append(Line(ax.coords_to_point(*ps[i]), ax.coords_to_point(min(x, x1), y), stroke_color=BLUE,
                                  stroke_width=7).set_z_index(5))
                if x1 < x:
                    break
            return VGroup(*lines)


        self.wait(0.1)

        axlabel2 = MathTex(r'F_n(x)', font_size=40, color=BLUE).move_to(ax.coords_to_point(0.5, 0.8)).set_z_index(3)
        axlabel3 = MathTex(r'F(x)', font_size=40, color=YELLOW).move_to(ax.coords_to_point(0.8, 0.55)).set_z_index(3)

        plt2 = always_redraw(emp)
        self.add(plt2)
        self.play(tval.animate.set_value(1.), FadeIn(axlabel2), rate_func=linear, run_time=1.5)
        plt2.clear_updaters()

        self.wait(0.1)
        self.play(FadeIn(eq4))

        plt1 = ax.plot(f, (0, xmax), stroke_width=5, stroke_color=YELLOW).set_z_index(3)

        self.play(Create(plt1), FadeIn(axlabel3), rate_func=linear, run_time=1.5)
        self.wait(0.1)
        circ2 = mh.circle_eq(eq4).set_z_index(4)
        txt1 = Tex(r'\sf Null Hypothesis', color=RED, stroke_width=4).next_to(circ2, UR).shift(LEFT*1.45+DOWN*0.35).set_z_index(3)
        self.play(LaggedStart(Create(circ2, run_time=1, rate_func=linear),
                              FadeIn(txt1, run_time=1), lag_ratio=0.5))
        self.wait(0.1)

        gp1 = VGroup(ax, axlabel, axlabel2, axlabel3, axline, plt1, plt2, eqx, *ticks).copy()
        self.play(LaggedStart(FadeOut(txt1, circ2, eq1, eq3, eq4, run_time=1),
                  gp1.animate(run_time=1.5).next_to(mh.pos((-0.83, -0.15)), DR, buff=0),
                  lag_ratio=0.3))
        self.wait(0.1)

        y0 = 0.5
        s = 2.6
        ys = []
        lines = []
        for i in range(0, n_sample * 2 + 1, 2):
            x, y = ps[i]
            #lines.append(ax.plot(lambda x: y0, x_range=(ps[i-1][0], ps[i][0]), stroke_color=BLUE, stroke_width=7).set_z_index(5))
            if i > 0:
                ys += [s*(ps[i-1][1]-f(x)), s*(y-f(x))]
                lines.append(Line(ax.coords_to_point(x, ys[-2]+y0), ax.coords_to_point(x, y0+ys[-1]), stroke_color=BLUE,
                                  stroke_width=2).set_z_index(4))
            lines.append(ax.plot(lambda x: y0 + s*(y - f(x)), x_range=(ps[i][0], ps[i+1][0]), stroke_color=BLUE, stroke_width=7).set_z_index(5))

        lines = VGroup(*lines)
        plt3 = ax.plot(lambda _: y0, (0, xmax), stroke_width=5, stroke_color=YELLOW).set_z_index(3)
        axlabel4 = MathTex(r'F_n(x)', r'-', r'F(x)', font_size=40, color=BLUE).move_to(ax.coords_to_point(0.32, 1)).set_z_index(3)

        self.play(FadeOut(eqx, *ticks, axlabel, axline, ax.x_axis, eqax1),
                  mh.rtransform(axlabel2[0], axlabel4[0], axlabel3[0], axlabel4[2]),
                  FadeIn(axlabel4[1]),
                  mh.transform(plt2, lines, plt1, plt3, run_time=2))

        self.wait(0.1)

        statD = max(ys)
        tval.set_value(0.7)

        def boxgen():
            t = tval.get_value()
            pt0 = ax.coords_to_point(0, y0 - t)
            pt1 = ax.coords_to_point(xmax, y0 + max(t, statD))
            box1 = Rectangle(width=pt1[0]-pt0[0], height=pt1[1]-pt0[1], stroke_width=0, stroke_opacity=0,
                             fill_color=GREY, fill_opacity=0.6).set_z_index(1)
            box1.next_to(pt0, UR, buff=0)
            return box1

        box1 = always_redraw(boxgen)
        self.play(FadeIn(box1), run_time=0.8)
        self.wait(0.1)
        self.play(tval.animate.set_value(statD), run_time=0.8)
        self.wait(0.1)
        arr1 = Arrow(ax.coords_to_point(0.85, y0), ax.coords_to_point(0.85, y0+statD), buff=0).set_z_index(5)
        eq5 = MathTex(r'D_n').set_z_index(5).next_to(arr1, RIGHT, buff=0)
        self.play(FadeIn(arr1, eq5))

        eq6 = MathTex(r'D_n', r'=', r'\max_x\lvert F_n(x)-F(x)\rvert').set_z_index(5)
        eq6.next_to(ax, UP, buff=0.05, coor_mask=UP)
        self.play(FadeIn(eq6))
        self.wait(0.1)

        eq6_1 = MathTex(r'\sqrt nD_n', r'=', r'\sqrt n\max_x\lvert F_n(x)-F(x)\rvert').set_z_index(5)
        mh.align_sub(eq6_1, eq6_1[0][-2], eq6[0][-2])
        self.play(mh.rtransform(eq6[0][:], eq6_1[0][-2:], eq6[1], eq6_1[1], eq6[2][:], eq6_1[2][3:]),
                  FadeIn(eq6_1[0][:-2], eq6_1[2][:3]))
        self.wait(0.1)

        eq7 = MathTex(r'\sqrt nD_n', r'\to', r'D', r'\ {\sf(in\ distribution)}')
        mh.align_sub(eq7, eq7[0][-2], eq6[0][-2])
        self.play(mh.rtransform(eq6_1[0], eq7[0]),
                  #FadeIn(eq7[0][:-2], shift=mh.diff(eq6[0][:], eq7[0][-2:])),
                  mh.fade_replace(eq6_1[1], eq7[1]),
                  FadeOut(eq6_1[2]),
                  FadeIn(eq7[2:]),
                  run_time=1.5)
        self.wait(0.1)

        eq8 = MathTex(r'\mathbb E[D^s]', r'=', r's', r'2^{-\frac s2}', r'(1-2^{1-s})', r'\Gamma({}^{\frac s2})', r'\zeta(s)')
        eq8[5][2:5].move_to(eq8[1], coor_mask=UP)
        mh.align_sub(eq8, eq8[0][-2], eq7[0][-2], coor_mask=UP)

        VGroup(eq8[0][-2], eq8[2][0], eq8[3][2], eq8[4][-2], eq8[5][2], eq8[6][2]).set_color(YELLOW)
        self.play(mh.rtransform(eq7[2][0], eq8[0][2]),
                  FadeIn(eq8[0][:2], eq8[0][-2:], shift=mh.diff(eq7[2][0], eq8[0][2])),
                  FadeOut(eq7[:2], eq7[3]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq8[1:3]))
        self.wait(0.1)
        self.play(FadeIn(eq8[3]))
        self.wait(0.1)
        self.play(FadeIn(eq8[4]))
        self.wait(0.1)
        self.play(FadeIn(eq8[5]))
        self.wait(0.1)
        self.play(FadeIn(eq8[6]))
        self.wait(0.1)

        eq10 = MathTex(r'V_n', r'=', r'\max_x( F_n(x)-F(x)) + \max_x(F(x)-F_n(x))').set_z_index(5)
        eq10.next_to(ax, UP, buff=0.05, coor_mask=UP)
        self.play(FadeIn(eq10), FadeOut(eq8))
        self.wait(0.1)

        statV = -min(ys)
        self.play(tval.animate.set_value(statV), run_time=0.8)
        self.wait(0.1)
        arr2 = Arrow(ax.coords_to_point(1.15, y0-statV), ax.coords_to_point(1.15, y0+statD), buff=0).set_z_index(5)
        eq9_1 = MathTex(r'V_n').set_z_index(5).next_to(arr2, RIGHT, buff=0).shift(UP*0.2)
        self.play(FadeIn(arr2, eq9_1))
        self.wait(0.1)

        eq11 = MathTex(r'\sqrt nV_n', r'\to', r'V', r'\ {\sf(in\ distribution)}')
        mh.align_sub(eq11, eq11[1], eq10[1], coor_mask=UP)
        self.play(mh.rtransform(eq10[0][:], eq11[0][-2:]),
                  mh.fade_replace(eq10[1], eq11[1]),
                  FadeIn(eq11[0][:-2], shift=mh.diff(eq10[0][:], eq11[0][-2:])),
                  FadeIn(eq11[2:]),
                  FadeOut(eq10[2:]),
                  run_time=1.5)
        self.wait(0.1)

        eq12 = MathTex(r'\mathbb E[V^s]', r'=', r'2^{-\frac s2}', r's(s-1)', r'\Gamma({}^{\frac s2})', r'\zeta(s)')
        eq12[4][2:5].move_to(eq12[1], coor_mask=UP)
        mh.align_sub(eq12, eq12[0][-2], eq11[0][-2], coor_mask=UP)
        VGroup(eq12[0][-2], eq12[2][-3], eq12[3][0], eq12[3][2], eq12[4][2], eq12[5][2]).set_color(YELLOW)

        self.play(mh.rtransform(eq11[2][0], eq12[0][2]),
                  FadeIn(eq12[0][:2], eq12[0][-2:], shift=mh.diff(eq11[2][0], eq11[0][2])),
                  FadeOut(eq11[:2], eq11[3]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq12[1:3]))
        self.wait(0.1)
        self.play(FadeIn(eq12[3]))
        self.wait(0.1)
        self.play(FadeIn(eq12[4]))
        self.wait(0.1)
        self.play(FadeIn(eq12[5]))
        self.wait(0.1)

        eq13 = MathTex(r'\mathbb E[V^s]', r'=', r'\left(\frac\pi2\right)^{\frac s2}', r'\pi^{-\frac s2}', r's(s-1)', r'\Gamma({}^{\frac s2})', r'\zeta(s)')
        eq13[5][2:5].move_to(eq12[1], coor_mask=UP)
        mh.align_sub(eq13, eq13[1], eq12[1], coor_mask=UP)
        VGroup(eq13[0][-2], eq13[2][-3], eq13[3][-3], eq13[4][0], eq13[4][2], eq13[5][2], eq13[6][2]).set_color(YELLOW)
        self.play(mh.rtransform(eq12[:2], eq13[:2], eq12[3:], eq13[4:],
                  eq12[2][-4:], eq13[3][-4:], eq12[2][-3:].copy(), eq13[2][-3:], eq12[2][0], eq13[2][3]),
                  FadeIn(eq13[2][0], eq13[2][2], eq13[2][-4]),
                  FadeIn(eq13[2][1], target_position=eq12[2][0]),
                  FadeIn(eq13[3][0], target_position=eq12[2][0]),
                  run_time=1.5)

        eq14 = MathTex(r'\mathbb E[V^s]', r'=', r'\left(\frac\pi2\right)^{\frac s2}', r'2\xi(s)')
        VGroup(eq14[0][-2], eq14[2][-3], eq14[3][3]).set_color(YELLOW)
        mh.align_sub(eq14, eq14[1], eq13[1])
        self.play(mh.rtransform(eq13[:3], eq14[:3]),
                  FadeOut(eq13[3:]),
                  FadeIn(eq14[3:]),
                  run_time=1.5)
        self.wait(0.1)
        eq15 = MathTex(r'\mathbb E[({}^{\sqrt{\frac2\pi} }V)^s]', r'=', r'2\xi(s)')
        eq15[0][3:8].move_to(eq15[1], coor_mask=UP)
        VGroup(eq15[0][-2], eq15[2][3]).set_color(YELLOW)
        mh.align_sub(eq15, eq15[1], eq14[1])
        self.play(mh.rtransform(eq14[0][:2], eq15[0][:2], eq14[0][-3], eq15[0][-4], eq14[0][-2:], eq15[0][-2:],
                                eq14[1], eq15[1], eq14[3], eq15[2]),
                  mh.stretch_replace(eq14[2][5], eq15[0][-2], eq14[2][0], eq15[0][2], eq14[2][1], eq15[0][-5],
                                     eq14[2][2], eq15[0][-6], eq14[2][3], eq15[0][-7], eq14[2][4], eq15[0][-3]),
                  FadeOut(eq14[2][6:8], shift=mh.diff(eq14[2][5], eq15[0][-2])),
                  FadeIn(eq15[0][3:-7], rate_func=rush_into),
                  run_time=1.8)
        self.wait(0.1)

        eq15.generate_target().next_to(mh.pos((-0.83, -0.23)), DR, buff=0)
        self.play(FadeOut(gp1), MoveToTarget(eq15), FadeIn(eq8), run_time=1.6)
        self.wait(0.1)
        eq16 = MathTex(r'\mathbb E[({}^{\sqrt{\frac2\pi} }D)^s]', r'=', r'(1-2^{1-s})', r's', r'\pi^{-\frac s2}',
                       r'\Gamma({}^{\frac s2})', r'\zeta(s)')
        eq16[0][3:8].move_to(eq16[1], coor_mask=UP)
        eq16[-2][2:5].move_to(eq16[1], coor_mask=UP)
        mh.align_sub(eq16, eq16[1], eq8[1], coor_mask=UP)
        VGroup(eq16[0][-2], eq16[2][-2], eq16[3][0], eq16[4][-3], eq16[5][2], eq16[6][2]).set_color(YELLOW)
        self.play(mh.rtransform(eq8[0][:2], eq16[0][:2], eq8[0][-3], eq16[0][-4], eq8[0][-2:], eq16[0][-2:],
                                eq8[1], eq16[1], eq8[2], eq16[3], eq8[4], eq16[2], eq8[5], eq16[5], eq8[6], eq16[6],
                                eq8[3][-4:], eq16[4][-4:]),
                  FadeIn(eq16[4][0], target_position=eq8[3][0]),
                  mh.stretch_replace(eq8[3][0], eq16[0][-7]),
                  FadeIn(eq16[0][-6:-4], shift=mh.diff(eq8[3][0], eq16[0][-7])),
                  FadeIn(eq16[0][2:-7], eq16[0][-3], rate_func=rush_into),
                  run_time=1.8)
        self.wait(0.1)
        eq17 = MathTex(r'()', r'(s-1)^{-1}', r's', r'(s-1)', r'\pi')
        VGroup(eq17[1][1], eq17[3][1]).set_color(YELLOW)
        mh.align_sub(eq17, eq17[2], eq16[3])
        eq17_1 = eq17[1].copy()
        mh.align_sub(eq17_1, eq17_1[:-2], eq17[2], coor_mask=RIGHT)
        self.play(FadeIn(eq17[1], target_position=eq17_1),
                  FadeIn(eq17[3], target_position=eq17_1[:-2]),
                  eq16[:3].animate.shift(mh.diff(eq16[2][-1], eq17[0][-1])*RIGHT),
                  eq16[4:].animate.shift(mh.diff(eq16[4][0], eq17[4][0])*RIGHT),
                  run_time=1.5
                  )
        self.wait(0.1)
        eq18 = MathTex(r'(s-1)^{-1}', r'2\xi(s)')
        VGroup(eq18[0][1], eq18[1][3]).set_color(YELLOW)
        mh.align_sub(eq18, eq18[0], eq17[1])
        self.play(mh.rtransform(eq17[1], eq18[0]),
                  FadeOut(eq17[3], eq16[3:]),
                  FadeIn(eq18[1]),
                  run_time=1.6)
        self.wait(0.1)

        eq19 = MathTex(r'\mathbb E[({}^{\sqrt{\frac2\pi} }D)^s]', r'=', r'\frac{1-2^{1-s} }{s-1}', r'2\xi(s)')
        VGroup(eq19[0][-2], eq19[2][5], eq19[2][-3], eq19[-1][-2]).set_color(YELLOW)
        eq19[0][3:-4].move_to(eq19[1], coor_mask=UP)
        eq19.next_to(eq15, DOWN).align_to(eq15, LEFT)
        self.play(mh.rtransform(eq16[:2], eq19[:2], eq16[2][1:-1], eq19[2][:6],
                                eq18[0][1:-3], eq19[2][-3:],
                                eq18[1], eq19[-1]),
                  FadeOut(eq16[2][0], eq16[2][-1], shift=mh.diff(eq16[2][1:-1], eq19[2][:6])),
                  FadeOut(eq18[0][0], eq18[0][-3:], shift=mh.diff(eq18[0][1:-3], eq19[2][-3:])),
                  FadeIn(eq19[2][-4], rate_func=rush_into),
                  run_time=2)

        self.wait()

class ConfidenceLevel(Scene):
    def construct(self):
        MathTex.set_default(font_size=60)
        eq1 = MathTex(r'\mathbb P(D_n\le\alpha)', r'=', r'99\%')
        eq2 = Tex(r'$D_n > \alpha$', r'\ $\Rightarrow$\ ', r'reject null hypothesis')
        VGroup(eq1[0][2:4], eq1[0][-2], eq2[0][:2], eq2[0][-1]).set_color(BLUE)
        eq2[2].set_color(RED)
        self.add(eq1)


        eq2.next_to(eq1, DOWN)

        self.play(FadeIn(eq2))

class BMDefs(Scene):
    def construct(self):
        xlen = config.frame_x_radius * 1.05
        ylen = config.frame_y_radius
        ax = Axes(x_range=[0, 1.05], y_range=[-1, 1], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)
#        self.add(ax)
        eq1 = MathTex(r't')[0].next_to(ax.x_axis.get_right(), UL, buff=0.2)
        eq2 = MathTex(r'B_t')[0].next_to(ax.y_axis.get_top(), DR, buff=0.2)
        tarr = np.linspace(0.1, 0.95, 4)
        self.add(eq1, eq2)

        marks = [ax.x_axis.get_tick(t) for t in tarr]
        eq_marks = [MathTex(r't_{}'.format(i), font_size=40).next_to(marks[i], DOWN, buff=0.05) for i in range(4)]
        eq_dB = []
        self.wait(0.5)
        for i in range(len(tarr)):
            tmp = []
            if i > 0:
                pos = (marks[i-1].get_bottom() + marks[i].get_bottom()) * 0.5
                tmp.append(MathTex(r'B_{{t_{} }} - B_{{ t_{} }}'.format(i, i-1), font_size=38)
                           .next_to(pos, DOWN, buff=0.85))
                eq_dB += tmp
            self.play(FadeIn(marks[i], eq_marks[i], *tmp), run_time=1)

        self.wait()
        t0 = 0.6
        mark = ax.x_axis.get_tick(t0)
        eq_mark = MathTex(r't', font_size=40).next_to(mark, DOWN, buff=0.05)
        eq3 = MathTex(r'B_t\sim N(0, t)')[0].next_to(ax.get_bottom(), UP)
        self.play(FadeOut(*marks, *eq_marks, *eq_dB),
                  FadeIn(eq3[:3], mark, eq_mark),
                  run_time=1.5)
        self.wait(0.2)
        self.play(FadeIn(eq3[3:5], eq3[6], eq3[8]), run_time=1)
        self.wait(0.2)
        self.play(FadeIn(eq3[5]), run_time=0.5)
        self.wait(0.2)
        self.play(FadeIn(eq3[7]), run_time=0.5)
        self.wait(0.2)
        self.play(FadeOut(eq3, mark, eq_mark), run_time=0.5)
        self.wait()
        eq4 = MathTex(r'{\rm Var}(B_t){{=}}t').next_to(ax.get_bottom(), UP).shift(LEFT*1.2)
        eq5 = MathTex(r'{\rm Var}(B_{Nt}){{=}}Nt')
        eq6 = MathTex(r'{\rm Var}(B_{Nt}){{=}}N{\rm Var}(B_t)')
        eq7 = MathTex(r'{\rm std\,dev}(B_{Nt}){{=}}\sqrt{N}\,{\rm std\,dev}(B_t)')
        eq5.next_to(eq4[1], ORIGIN, submobject_to_align=eq5[1])
        eq6.next_to(eq4[1], ORIGIN, submobject_to_align=eq6[1])
        eq7.next_to(eq4[1], ORIGIN, submobject_to_align=eq7[1])
        self.play(FadeIn(eq4), run_time=0.6)
        self.wait(0.1)
        self.play(LaggedStart(ReplacementTransform(eq4[0][:5] + eq4[0][5] + eq4[0][6] + eq4[1] + eq4[2][0],
                                                   eq5[0][:5] + eq5[0][6] + eq5[0][7] + eq5[1] + eq5[2][1]),
                              FadeIn(eq5[0][5], eq5[2][0]), lag_ratio=0.3),
                  run_time=1)
        self.wait(0.2)
        self.play(ReplacementTransform(eq5[2][0], eq6[2][0]),
                  FadeOut(eq5[2][1]),
                  FadeIn(eq6[2][1:]),
                  run_time=1)
        self.wait(0.2)
        self.play(ReplacementTransform(eq5[0][-5:] + eq5[1] + eq6[2][0],
                                       eq7[0][-5:] + eq7[1] + eq7[2][2]),
                  FadeOut(eq5[0][:-5] + eq6[2][1:]),
                  FadeIn(eq7[0][:-5] + eq7[2][3:] + eq7[2][:2]),
                  run_time=1.5)
        self.wait(0.5)
        self.play(FadeOut(eq7), run_time=0.5)

        self.wait()


class Bridge(Scene):
    def get_axes(self, scale=1., xlen = 0.9, ylen=0.95, scale_neg=1.):
        ymax = 1.5 / scale
        xlen *= 2 * config.frame_x_radius
        ylen *= 2 * config.frame_y_radius
        ax = Axes(x_range=[0, 1.05], y_range=[-ymax*scale_neg, ymax], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(1.9)
        eqt = MathTex(r't').next_to(ax.x_axis.get_right(), UP, buff=0.2)
        mark1 = ax.x_axis.get_tick(1, size=0.1).set_stroke(width=6).set_z_index(11)
        line1 = DashedLine(ax.coords_to_point(1, -ymax*scale_neg), ax.coords_to_point(1, ymax), color=GREY).set_z_index(10).set_opacity(0.6)
        eq2 = MathTex(r'T', font_size=60).set_z_index(3).next_to(mark1, DR, buff=0.05)
        eq6 = MathTex(r'1', font_size=60).set_z_index(3).next_to(mark1, DR, buff=0.05)

        return ax, eqt, xlen, ylen, ymax, mark1, line1, eq2, eq6

    def construct(self):
        seeds = [3, 4, 18]
        npts = 1920
        ndt = npts - 1

        np.random.seed(seeds[0])

        ax, eqt, xlen, ylen, ymax, mark1, line1, eq2, eq6 = self.get_axes()

        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])
        b_vals = -np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))

        path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(2)

        tarr = np.linspace(0.1, 0.95, 4)
        marks = [ax.x_axis.get_tick(t, size=0.1).set_stroke(width=6) for t in tarr]
        eq_marks = [MathTex(r't_{}'.format(i), font_size=60).set_z_index(3).next_to(marks[i], DOWN, buff=0.05) for i in range(4)]
        anims = []
        eqargs = []
        eqs = []
        for i in range(len(tarr)):
            tmp = [marks[i], eq_marks[i]]
            if i > 0:
                pos = (marks[i-1].get_bottom() + marks[i].get_bottom()) * 0.5
                eqstr = r'B_{{t_{} }} - B_{{ t_{} }}'.format(i, i-1)
                tmp.append(MathTex(eqstr, font_size=60).set_z_index(3)
                           .next_to(pos, DOWN, buff=0.85))
                eqs.append(tmp[-1])
                if i > 1:
                    eqargs.append(r',')
                eqargs.append(eqstr)
            anims.append(FadeIn(*tmp, run_time=0.5))
        eq1 = MathTex(*eqargs, font_size=60).next_to(VGroup(*eqs), DOWN)
        br1 = BraceLabel(eq1, r'\sf independent', label_constructor=mh.mathlabel_ctr2, font_size=80,
                         brace_config={'color': RED}).set_z_index(2)

        eq3 = MathTex(r'B_t', r'\sim', r'N(0,t)', font_size=80).set_z_index(3)
        eq3.move_to(eq1, coor_mask=UP).shift(DOWN*0.5)


        self.add(ax, eqt)
        self.wait(0.1)
        self.play(Create(path1, rate_func=linear, run_time=4),
                  Succession(Wait(1), *anims))
        self.play(LaggedStart(mh.rtransform(eqs[0][0], eq1[0], eqs[1][0], eq1[2], eqs[2][0], eq1[4]),
                  FadeIn(eq1[1], eq1[3], br1), lag_ratio=0.2))
        self.wait(0.1)
        self.play(FadeOut(*eq_marks, *marks, eq1, br1),
                  FadeIn(eq3))
        self.wait(0.1)

        # bridge

        np.random.seed(seeds[1])
        b_vals2 = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
        b_vals2 -= t_vals * b_vals2[-1]
        path2 = ax.plot_line_graph(t_vals, b_vals2, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(2)

        self.play(FadeIn(eq2, mark1, line1), path1.animate.set_stroke(opacity=0.3, color=BLUE).set_z_index(1.5))
        self.play(Create(path2, rate_func=linear, run_time=3), FadeOut(eq3, run_time=1))
        self.wait(0.1)
        eps = 0.15
        pts = [ax.coords_to_point(1, eps), ax.coords_to_point(1, -eps)]
        origin = ax.coords_to_point(0, 0)
        yup = ax.coords_to_point(0, ymax) - origin
        yright = ax.coords_to_point(1, 0) - origin
        box1 = Rectangle(width=0.4, height=yup[1] -(pts[0]-origin)[1], color=GREY, fill_opacity=0.5, stroke_opacity=0)
        box1.next_to(pts[0], UR, buff=0)
        box2 = box1.copy().next_to(pts[1], DR, buff=0)
        eq3 = MathTex(r'\varepsilon', font_size=60).set_z_index(4).next_to(pts[0], LEFT, buff=0.1)
        eq4 = MathTex(r'-\varepsilon', font_size=60).set_z_index(4).next_to(pts[1], LEFT, buff=0.1)
        eq4_1 = eq4.copy().set_stroke(width=12, color=BLACK).set_z_index(3.5).next_to(pts[1], LEFT, buff=0.1)
        eq3_1 = eq3.copy().set_stroke(width=12, color=BLACK).set_z_index(3.5).next_to(pts[0], LEFT, buff=0.1)
        self.play(FadeIn(box1, box2, eq3, eq4, eq4_1, eq3_1))
        self.wait(0.1)

        s = yup[1]/ymax/yright[0] * eps
        self.play(path2.animate.apply_matrix([[1, 0], [-s, 1]], about_point=origin), run_time=0.5)
        self.play(path2.animate.apply_matrix([[1, 0], [1.8*s, 1]], about_point=origin), run_time=0.9)
        self.wait(0.1)
        self.play(path2.animate.apply_matrix([[1, 0], [-0.8*s, 1]], about_point=origin),
                  box1.animate.stretch(factor=ymax/(ymax-eps), dim=1, about_edge=UP),
                  box2.animate.stretch(factor=ymax/(ymax-eps), dim=1, about_edge=DOWN),
                  VGroup(eq3, eq3_1).animate.shift(DOWN*eps*yup[1]/ymax),
                  VGroup(eq4, eq4_1).animate.shift(UP*eps*yup[1]/ymax),
                  run_time=1)
        self.wait(0.1)


        self.play(path2.animate.set_stroke(opacity=0.3, color=GREEN).set_z_index(1.51))
        np.random.seed(seeds[2])
        b_vals3 = -np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
        path3 = ax.plot_line_graph(t_vals, b_vals3, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(2)
        self.play(FadeOut(eq3, eq4, eq3_1, eq4_1, box1, box2, run_time=0.8, rate_func=linear),
                  Create(path3, rate_func=linear, run_time=3))
        self.wait(0.1)
        pt1 = ax.coords_to_point(1, b_vals3[-1])
        line2 = Line(origin, pt1, stroke_color=GREEN, stroke_width=6).set_z_index(3)
        line3 = Line(origin, ax.coords_to_point(1, 0), stroke_color=GREEN, stroke_width=6, stroke_opacity=0).set_z_index(3)
        self.wait(0.1)
        self.play(FadeIn(line2))
        self.wait(0.1)

        #

        b_vals4 = b_vals3 - t_vals * b_vals3[-1]
        path3_1 = path3.copy()
        path3.set_stroke(opacity=0.3, color=ORANGE)
        path4 = ax.plot_line_graph(t_vals, b_vals4, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(2)
        self.play(mh.rtransform(path3_1, path4, line2, line3), rate_func=linear, run_time=1.4)

        # scaling
        self.wait(0.1)
        a = 0.5
        mark2 = ax.x_axis.get_tick(a, size=0.1).set_stroke(width=6)
        line2 = DashedLine(ax.coords_to_point(a, -ymax), ax.coords_to_point(a, ymax), color=GREY).set_z_index(10).set_opacity(0.6)
        eq5 = MathTex(r'aT', font_size=60).set_z_index(3).next_to(mark2, DR, buff=0.05)
        self.play(FadeIn(mark2, line2, eq5))
        paths = VGroup(path1, path2, path3, path4)
        self.play(paths.animate.apply_matrix([[a, 0], [0, math.sqrt(a)]], about_point=origin),
                  run_time=2)
        self.wait(0.1)
        self.play(paths.animate.apply_matrix([[1/a, 0], [0, 1/math.sqrt(a)]], about_point=origin),
                  FadeOut(mark2, line2, eq5, eq2, rate_func=linear),
                  FadeIn(eq6, rate_func=linear),
                  run_time=2)
        self.wait(0.1)
        path1.set_z_index(3)
        self.play(FadeOut(path2, path3, path4),
                  path1.animate.set_stroke(opacity=1),
                  rate_func=linear, run_time=1)

        tknots = [0, 0.19, 0.5, 0.7, 1]
        cols = [RED, TEAL, YELLOW, PURPLE]
        nk = len(tknots)
        iknots = [round(i * ndt) for i in tknots]
        tknots = [t_vals[i] for i in iknots]
        yknots = [b_vals[i] for i in iknots]
        pts = [ax.coords_to_point(t, y) for t, y in zip(tknots, yknots)]
        dots = [Dot(radius=0.15, fill_color=GREEN).set_z_index(5).move_to(pt) for pt in pts]
        lines = [Line(p, q, stroke_color=GREEN, stroke_width=5).set_z_index(2) for p, q in zip(pts[:-1], pts[1:])]
        self.play(FadeIn(*dots, *lines))
        self.wait(0.1)

        t_vals_arr = [t_vals[i:j+1] for i, j in zip(iknots[:-1], iknots[1:])]
        b_vals_arr = [b_vals[i:j+1] for i, j in zip(iknots[:-1], iknots[1:])]
        paths_arr = [ax.plot_line_graph(t_vals_arr[i], b_vals_arr[i], add_vertex_dots=False, stroke_color=cols[i],
                                        stroke_width=4).set_z_index(2) for i in range(nk-1)]
        paths_arr = VGroup(*paths_arr)
        #self.play(FadeIn(paths_arr))
        self.add(*paths_arr)
        self.play(FadeOut(path1))
        self.wait(0.1)

        ax2 = Axes(x_range=[0, 1.1, 1], y_range=[-ymax, ymax], x_length=xlen*0.4, y_length=ylen*0.4,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                   x_axis_config={'include_ticks': True}
                  ).set_z_index(2).move_to(mh.pos(UP * 0.5+LEFT*0.2))
        VGroup(ax2.x_axis.ticks, ax2.y_axis).set_z_index(5)
        self.wait(0.1)

        paths_arr2 = []
        for i in range(nk-1):
            t = t_vals_arr[i]-t_vals_arr[i][0]
            a = 1/t[-1]
            t *= a
            b = (b_vals_arr[i] - b_vals_arr[i][0]) * math.sqrt(a) * 1.3
            b -= t * b[-1]
            paths_arr2.append(ax2.plot_line_graph(t, b,
                                                  add_vertex_dots=False, stroke_color=cols[i],
                                                  stroke_width=3).set_z_index(2 + i/10))
        paths_arr2 = VGroup(*paths_arr2)

        paths_arr3 = paths_arr.copy()
        self.play(FadeIn(ax2), mh.rtransform(paths_arr, paths_arr2), run_time=1.6)
        self.wait(0.1)
        self.play(FadeOut(ax2), mh.rtransform(paths_arr2, paths_arr3), run_time=1.6)
        self.play(FadeOut(*dots, *lines), FadeIn(path1),
                  rate_func=linear)
        self.remove(paths_arr3)


        self.wait()

class BridgeMax(Bridge):
    def construct(self):
        seeds = [31, 25, 21]
        npts = 1920
        ndt = npts - 1
        np.random.seed(seeds[0])

        ax, eqt, xlen, ylen, ymax, mark1, line1, _, eq1 = self.get_axes(1.)
        self.add(ax, eqt, mark1, line1, eq1)

        origin = ax.coords_to_point(0, 0)
        right = ax.coords_to_point(1, 0) - origin
        up = ax.coords_to_point(0, 1) - origin

        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])
        b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
        b_vals -= t_vals * b_vals[-1]
        if min(b_vals) + max(b_vals) > 0:
            b_vals = -b_vals
        path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(2)

        self.play(Create(path1, run_time=2, rate_func=linear))
        self.wait(0.1)

        bmax = max(b_vals) * 1.05
        line2 = DashedLine(ax.coords_to_point(0, bmax), ax.coords_to_point(1, bmax), color=GREY).set_z_index(1.5)
        eq2 = MathTex(r'M', font_size=65).next_to(ax.coords_to_point(0.9, bmax), UP, buff=0.1).set_z_index(5)
        eq3 = MathTex(r'\mathbb P(M > x)', r'=', r'e^{-2x^2}', font_size=70).move_to(ax.coords_to_point(0.5, ymax*0.6)).set_z_index(5)
        eq5 = MathTex(r'\mathbb E[({}^{\sqrt{\frac2\pi} }D)^s]', r'=', r'\frac{1-2^{1-s} }{s-1}', r'2\xi(s)', font_size=65).set_z_index(5)
        eq5[0][3:-4].move_to(eq5[1], coor_mask=UP)
        eq7 = MathTex(r'\mathbb E[({}^{\sqrt{\frac2\pi} }V)^s]', r'=', r'2\xi(s)', font_size=70).set_z_index(5)
        eq7[0][3:-4].move_to(eq7[1], coor_mask=UP)
        eq5.move_to(ax.coords_to_point(0.4, ymax * 0.5)).next_to(origin, RIGHT, buff=0.1, coor_mask=RIGHT)
        eq7.move_to(ax.coords_to_point(0.4, ymax * 0.2)).next_to(origin, RIGHT, buff=0.5, coor_mask=RIGHT)

        eq3_1 = eq3.copy().set_stroke(color=BLACK, width=8).set_z_index(4.9)
        eq3 = VGroup(VGroup(a, b) for a, b in zip(eq3[:], eq3_1[:]))
        eq5_1 = eq5.copy().set_stroke(color=BLACK, width=8).set_z_index(4.9)
        eq7_1 = eq7.copy().set_stroke(color=BLACK, width=8).set_z_index(4.9)

        self.play(FadeIn(line2, eq2))
        self.wait(0.1)
        self.play(FadeIn(eq3[0]))
        self.wait(0.1)
        self.play(FadeIn(eq3[1:]))
        self.wait(0.1)

        bmin = -min(b_vals) * 1.03

        boxes = [Rectangle(width=right[0], height=h*up[1], stroke_width=0, stroke_opacity=0,
                           fill_color=GREY, fill_opacity=0.6).set_z_index(1)
                 for h in (2 * ymax, 2 * bmin, bmin + bmax)]
        boxes[0].next_to(origin, RIGHT, buff=0)
        boxes[1].next_to(origin, RIGHT, buff=0)
        boxes[2].next_to(boxes[1].get_corner(DL), UR, buff=0)
        arr1 = Arrow(ax.coords_to_point(0.6, -bmin), ax.coords_to_point(0.6, 0), buff=0).set_z_index(5)
        eq4 = MathTex(r'D', stroke_width=2).set_z_index(5).next_to(arr1, RIGHT, buff=0).shift(UP*0.5)
        arr2 = Arrow(ax.coords_to_point(0.75, -bmin), ax.coords_to_point(0.75, bmax), buff=0).set_z_index(5)
        eq6 = MathTex(r'V', stroke_width=2).set_z_index(5).next_to(arr2, RIGHT, buff=0).shift(DOWN*1)

        self.play(FadeIn(boxes[0]), eq3.animate.scale(0.8).to_edge(UP, buff=0.5))
        self.wait(0.1)
        self.play(mh.rtransform(*boxes[:2]), run_time=1.6)
        self.wait(0.1)
        self.play(FadeIn(eq4, arr1))
        self.wait(0.1)
        self.play(FadeIn(eq5[0][3:-3], eq5_1[0][3:-3]))
        self.wait(0.1)
        self.play(FadeIn(eq5[0][:3], eq5[0][-3:], eq5_1[0][:3], eq5_1[0][-3:]))
        self.wait(0.1)
        self.play(FadeIn(eq5[1:3], eq5_1[1:3]))
        self.wait(0.1)
        self.play(FadeIn(eq5[3:], eq5_1[3:]))
        self.wait(0.1)
        self.play(VGroup(eq5, eq5_1).animate.scale(0.9).next_to(eq3, RIGHT, buff=0.5).next_to(origin, RIGHT).to_edge(UP, buff=0.4),
                  eq3.animate.to_edge(RIGHT, buff=0.5))
        self.play(mh.rtransform(*boxes[1:]), run_time=1.6)
        self.play(FadeIn(arr2, eq6))
        self.wait(0.1)
        self.play(FadeIn(eq7[0][3:-3], eq7_1[0][3:-3]))
        self.wait(0.1)
        self.play(FadeIn(eq7[0][:3], eq7[0][-3:], eq7_1[0][:3], eq7_1[0][-3:]))
        self.wait(0.1)
        self.play(FadeIn(eq7[1:], eq7_1[1:]))
        self.wait(0.1)
        gp = VGroup(eq7, eq7_1)
        eq7_2 = gp.copy().scale(0.9)
        mh.align_sub(eq7_2, eq7_2[0][1], eq5[1]).to_edge(RIGHT, buff=0.4)
        self.play(FadeOut(ax, eqt, line1, path1, eq1, boxes[-1], arr1, arr2, mark1, line2, eq2, eq4, eq6, eq3),
                  mh.rtransform(gp, eq7_2),
                  eq5.animate.to_edge(LEFT, buff=0.4),
                  rate_func=linear, run_time=1.6)
        self.wait()


class BridgeLimit(Scene):
    def construct(self):
        seeds = [3]
        np.random.seed(seeds[0])
        xmin=0.1
        xmax = 1.1 + xmin
        xlen = 6
        ylen = 4
        ymax2 = 1.
        nf = 100
        n0 = 4

        ax1 = Axes(x_range=[0, xmax * 1.05], y_range=[0, 1.2], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)

        ax2 = Axes(x_range=[0, xmax * 1.05], y_range=[-ymax2, ymax2], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)

        ax2.next_to(ax1, RIGHT, buff=1)
        eq1 = Tex(r'{{\sf nsamples}} $= {} $'.format(n0), font_size=60).to_edge(DOWN)
        VGroup(ax1, ax2).move_to(ORIGIN).next_to(eq1, UP)
        eq1x = MathTex(r'x', font_size=50).next_to(ax1.x_axis.get_right(), UP, buff=0.2)
        eq2x = eq1x.copy().next_to(ax2.x_axis.get_right(), UP, buff=0.15)

        self.add(ax1, ax2, eq1x, eq2x, eq1)

        xvals1 = np.concatenate(([0], np.linspace(xmin, xmin+1, nf), [xmax]))
        yvals1 = (xvals1 - xmin).clip(0, 1)
        path1 = ax1.plot_line_graph(xvals1, yvals1, stroke_width=5, stroke_color=YELLOW, add_vertex_dots=False).set_z_index(3)

        eq3 = MathTex(r'F_n', color=BLUE, font_size=50).move_to(ax1.coords_to_point(xmin+0.5, 0.84))
        eq4 = MathTex(r'F', color=YELLOW, font_size=50).move_to(ax1.coords_to_point(xmin+0.7, 0.5))
        eq5 = MathTex(r'\sqrt n(F_n-F)', color=BLUE, font_size=45).set_z_index(6).move_to(ax2.coords_to_point(0.3, 0.9))

        self.wait(0.1)
        self.play(Create(path1, rate_func=linear, run_time=1), FadeIn(eq4))
        self.wait(0.1)

        def emp(x_rand, t=1.1):
            x1 = t * xmax
            n_sample = len(x_rand)
            ps = [(0, 0), (xmin, 0)]
            ps2 = [(0, 0), (xmin, 0)]
            s = math.sqrt(n_sample)
            for x in x_rand + xmin:
                y = ps[-1][1]
                ps.append((x, y))
                ps.append((x, y + 1 / n_sample))
                ps2.append((x, (y - x + xmin)*s))
                ps2.append((x, (y + 1 / n_sample - x + xmin)*s))
            ps.append((xmin+1, 1))
            ps2.append((xmin+1, 0))
            ps.append((xmax, 1))
            ps2.append((xmax, 0))

            lines = []
            lines2 = []
            # assert len(ps) == n_sample*2 + 4
            for i in range(n_sample*2 + 3):
                x, y = ps[i + 1]
                stop = x1 < x
                if stop:
                    x = x1
                if n_sample*2 >= i > 1 and i % 2 ==0:
                    lines.append(Line(ax1.coords_to_point(*ps[i]), ax1.coords_to_point(x, y), stroke_color=BLUE,
                                  stroke_width=2).set_z_index(4))
                    lines2.append(Line(ax2.coords_to_point(*ps2[i]), ax2.coords_to_point(*ps2[i+1]), stroke_color=BLUE,
                                      stroke_width=2).set_z_index(4))
                else:
                    lines.append(Line(ax1.coords_to_point(*ps[i]), ax1.coords_to_point(x, y), stroke_color=BLUE,
                                      stroke_width=7).set_z_index(5))
                    lines2.append(Line(ax2.coords_to_point(*ps2[i]), ax2.coords_to_point(*ps2[i+1]), stroke_color=BLUE,
                                      stroke_width=7).set_z_index(5))
                if stop:
                    break
            if t <= 1.:
                return VGroup(VGroup(*lines), VGroup(*lines2))

            ys = [p[1] for p in ps2]
            bmin = min(ys)
            bmax = max(ys)
            p = ax2.coords_to_point(0., bmin)
            q = ax2.coords_to_point(xmax, bmax) - p
            box = Rectangle(width=q[0], height=q[1], fill_color=DARK_GREY, fill_opacity=1, stroke_opacity=0).set_z_index(1).next_to(p, UR, buff=0)
            return VGroup(VGroup(*lines), VGroup(*lines2), box)

        x_rand = np.random.uniform(0, 1, n0)
        x_rand.sort()

        track = ValueTracker(0.)
        def f():
            return emp(x_rand, track.get_value())[0]

        obj = always_redraw(f)
        self.add(obj)
        self.play(track.animate.set_value(1.), FadeIn(eq3), rate_func=linear, run_time=1.6)
        obj.clear_updaters()
        self.wait(0.1)

        lines1 = emp(x_rand)
        self.add(lines1[0])
        self.play(mh.rtransform(obj, lines1[1]), FadeIn(lines1[-1], eq5), run_time=1.6)
        self.wait(0.1)

        eq1_1 = eq1[0][-1:]
        imax = 96
        dts = [0.25, 0.2, 0.17, 0.14, 0.12, 0.1]
        for i in range(imax):
            x = np.random.uniform(0, 1)
            k = np.searchsorted(x_rand, x)
            x_rand = np.insert(x_rand, k, x)
            lines2 = emp(x_rand)
            eq2 = Tex(r'$={}$'.format(len(x_rand)), font_size=60)
            mh.align_sub(eq2, eq2[0][0], eq1[0][-2])

            j = min(i, imax-i-1)
            dt = dts[min(j, len(dts) - 1)]

            self.play(FadeIn(lines2[:-1], eq2[0][1:]), FadeOut(lines1[:-1], eq1_1),
                      ReplacementTransform(lines1[-1], lines2[-1]),
                      run_time=dt,
                      rate_func=linear)
            eq1_1 = eq2[0][1:]
            lines1 = lines2

        self.wait(0.1)


        def gfunc(ax):
            p1 = ax.coords_to_point(xmin, 0)[0]
            q1 = ax.coords_to_point(xmin + 1, 0)[0]
            def g(p):
                q = p.copy()
                if q1-0.005 > q[0] > p1+0.005:
                    x = (q[0] - p1) / (q1 - p1)
                    y = math.acos(1-2*x)/PI
                    #y = x * 0.99
                    q[0] = p1 *(1-y) + q1 * y
                    #q[0] -= 0.2
                    #print(x, y)
                return q
            return g

        self.play(ApplyPointwiseFunction(gfunc(ax1), VGroup(path1, lines1[0])),
                  ApplyPointwiseFunction(gfunc(ax2), lines1[1]),
                  run_time=3,
                  rate_func=there_and_back_with_pause
                  )

        self.wait()


class BridgeDev(Bridge):
    def construct(self):
        seeds = [40]
        npts = 1920
        ndt = npts - 1
        np.random.seed(seeds[0])

        ax, eqt, xlen, ylen, ymax, mark1, line1, _, eq1 = self.get_axes(1.6, ylen=0.75)
        ax.y_axis.set_z_index(10)
        gp = VGroup(ax, eqt, mark1, line1, eq1).to_edge(DOWN, buff=0.2)
        self.add(gp)
        self.wait(0.1)
        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])
        b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
        b_vals -= b_vals[-1] * t_vals
        y0 = -ymax * 0.7
        p = ax.coords_to_point(0, y0)
        bbar = np.average(b_vals[1:])
        b_vals2 = (b_vals-bbar)**2 * 1.8 + y0

        tracker = ValueTracker(0.)

        def f():
            t = tracker.get_value()
            b_vals3 = b_vals2 * t + b_vals * (1-t)
            path4 = ax.plot_line_graph(t_vals, b_vals3, add_vertex_dots=False, stroke_color=YELLOW,
                                       stroke_width=4).set_z_index(2)
            y = t * y0
            path2 = ax.plot_line_graph(t_vals, b_vals3.clip(y), add_vertex_dots=False, stroke_color=YELLOW,
                                       stroke_width=4).set_z_index(2)
            path3 = ax.plot_line_graph(t_vals, b_vals3.clip(max=y), add_vertex_dots=False, stroke_color=YELLOW,
                                       stroke_width=4).set_z_index(2)
            path2.set_stroke(opacity=0).set_fill(opacity=0.6, color=GREEN).set_z_index(1)
            path3.set_stroke(opacity=0).set_fill(opacity=0.6, color=RED).set_z_index(1)
            return VGroup(path4, path2, path3)

        path = always_redraw(f)

        eq2 = MathTex(r'b_t', font_size=60, color=YELLOW).move_to(ax.coords_to_point(0.67, ymax*0.5))

        self.play(LaggedStart(Create(path[0], run_time=2, rate_func=linear), FadeIn(eq2, rate_func=linear), lag_ratio=0.6))
        self.wait(0.1)

        self.play(FadeIn(*path[1:], rate_func=linear))
        self.wait(0.1)
        line2 = DashedLine(ax.coords_to_point(0, bbar), ax.coords_to_point(1, bbar), stroke_width=5, color=WHITE).set_z_index(5)
        eq3 = MathTex(r'\bar b', r'=', r'\int_0^1b_t\,dt', font_size=55).next_to(line2, DOWN, buff=0.1).next_to(line1, LEFT, buff=0.4, coor_mask=RIGHT).set_z_index(15)
        eq3_1 = eq3[0].copy().next_to(line2, DOWN, coor_mask=UP)
        self.play(FadeIn(line2, eq3_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq3_1, eq3[0]), FadeIn(eq3[1:]))
        self.wait(0.1)

        eq4 = MathTex(r'(b_t-\bar b)^2', color=YELLOW, font_size=60).move_to(ax.coords_to_point(0.5, ymax * -0.2))

        self.remove(path[0], *path[1:])
        path = always_redraw(f)
        self.add(path)
        self.play(LaggedStart(AnimationGroup(tracker.animate.set_value(1.),
                  line2.animate.move_to(p, coor_mask=UP).set_stroke(opacity=0),
                  VGroup(ax.x_axis, mark1, eq1, eqt).animate.move_to(p + DOWN*0, coor_mask=UP),
                  mh.rtransform(eq2[0][:], eq4[0][1:3], eq3[0][:], eq4[0][4:6]),
                  FadeOut(eq3[1:], rate_func=rush_from), run_time=2),
                  FadeIn(eq4[0][0], eq4[0][3], eq4[0][6:]) ,lag_ratio=0.5),
                  )
        path.clear_updaters()
        self.wait(0.1)

        eq5 = MathTex(r'D', r'\sim', r'\pi\sqrt{\int_0^1(b_t-\bar b)^2\,dt}', font_size=60).move_to(ax.coords_to_point(0.42, ymax*0.35)).set_z_index(15)
        eq5[2][6:14].set_color(YELLOW)
        self.play(LaggedStart(mh.rtransform(eq4[0][:], eq5[2][6:14], run_time=1.2),
                  FadeIn(eq5[2][3:6], eq5[2][14:]), lag_ratio=0.5))
        self.wait(0.1)
        self.play(FadeIn(eq5[2][1:3]))
        self.wait(0.1)
        self.play(FadeIn(eq5[:2], eq5[2][0]))
        self.wait(0.1)
        #box = Rectangle(width=2*config.frame_x_radius, height=2*config.frame_y_radius,
        #                stroke_width=0, stroke_opacity=0, fill_opacity=0.5, fill_color=BLACK).set_z_index(12)
        self.play(VGroup(ax, line1, eq1, mark1, eqt).animate.set_opacity(0.6),
                  path[0].animate.set_stroke(opacity=0.5),
                  path[1].animate.set_fill(opacity=0.42),
                  path[2].animate.set_fill(opacity=0.42),
                  eq5.animate.set_opacity(0.7), run_time=1, rate_func=linear)
        self.wait(0.1)
        eq6 = MathTex(r'\frac{\pi}2\sqrt{T}', r'\sim', r'D', r'\sim', r'\pi\sqrt{\int_0^1(b_t-\bar b)^2dt}', font_size=70).set_z_index(15)
        VGroup(eq6[2][6:14]).set_color(YELLOW)
        eq6[4][1:].scale(6/7, about_edge=LEFT)
        eq6.move_to(ORIGIN).to_edge(DOWN)
        self.play(FadeOut(path, ax, line1, eq1, mark1, eqt, rate_func=linear),
                  mh.rtransform(eq5[:], eq6[2:]), run_time=1.5)

        self.wait()


class TDist(BMPathIntro):
    def construct(self):
        eq1 = MathTex(r'\frac{\pi}2\sqrt{T}', r'\sim', r'D', r'\sim', r'\pi\sqrt{\int_0^1(b_t-\bar b)^2dt}', font_size=70).set_z_index(2)
        eq1[4][1:].scale(6/7, about_edge=LEFT)
        mh.align_sub(eq1, eq1[:3], mh.pos(89/960 * LEFT + 205/540 * DOWN))

        self.add(eq1[0])
        self.wait(0.1)
        self.play(FadeIn(eq1[1:3]))
        self.wait(0.1)
        eq2 = eq1.copy().move_to(ORIGIN).to_edge(DOWN)
        box = SurroundingRectangle(eq2, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   corner_radius=0.15, buff=0.2)
        eq1[3:].set_opacity(0)
        self.play(mh.rtransform(eq1[:3], eq2[:3]), FadeIn(box), run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq2[3:]))
        self.wait(0.1)

        eq3 = MathTex(r'2\left(\frac{X}{\pi}\right)^2', font_size=70).set_z_index(2)
        eq4 = MathTex(r'\mathbb E\left[e^{s2\left(\frac{X}{\pi}\right)^2 }\right]', r'=', r'\frac{\sqrt s}{\sin \sqrt s', font_size=70).set_z_index(2)
        eq4.move_to(box)
        eq3.move_to(box)
        eq3.move_to(eq4[0][3:-1], coor_mask=RIGHT)
        self.play(FadeOut(eq2), FadeIn(eq3))
        self.wait(0.1)
        r = ((eq3[0][i], eq4[0][i+4]) for i in range(7))
        self.play(
                  mh.stretch_replace(*(r1 for r2 in r for r1 in r2)),
                  FadeIn(eq4[0][:4], eq4[0][-1]))
        self.wait(0.1)
        self.play(FadeIn(eq4[1:], run_time=1.6))

        self.wait()


class MGF(Scene):
    def construct(self):
        eq1 = MathTex(r'2\left(\frac{X}{\pi}\right)^2', font_size=70).set_z_index(2)
        eq2 = MathTex(r'e^{s2\left(\frac{X}{\pi}\right)^2 }', r'=', r'\frac{\sqrt s}{\sin \sqrt s', font_size=70).set_z_index(2)
        eq1.move_to(eq2[0][1:], coor_mask=RIGHT)
        box = SurroundingRectangle(eq2, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   corner_radius=0.15)
        self.add(box, eq1)
        self.wait(0.1)
        r = ((eq1[0][i], eq2[0][i+2]) for i in range(7))
        self.play(
                  mh.stretch_replace(*(r1 for r2 in r for r1 in r2)),
                  FadeIn(eq2[0][:2]))
        self.wait(0.1)
        self.play(FadeIn(eq2[1:], run_time=1.6))
        self.wait()


class Excursion(Bridge):
    def construct(self):
        seeds = [42, 43, 46, 49]
        npts = 1920
        ndt = npts - 1
        np.random.seed(seeds[0])

        ax, eqt, xlen, ylen, ymax, mark1, line1, _, eq1 = self.get_axes(1, ylen=0.75, scale_neg=0.5)
        ax.y_axis.set_z_index(10)
        gp = VGroup(ax, eqt, mark1, line1, eq1).to_edge(DOWN, buff=0.2)
        self.add(gp)
        self.wait(0.1)
        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])
        b_vals1 = np.random.normal(scale=s, size=ndt).cumsum()
        b_vals1 -= b_vals1[-1] * t_vals[1:]
        i = np.argmin(b_vals1)
        b_vals1 = np.concatenate((b_vals1[i:], b_vals1[:i+1])) - b_vals1[i]

        path1 = ax.plot_line_graph(t_vals, b_vals1, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)

        txt1 = Tex(r'\sf Brownian excursion', color=YELLOW, font_size=100)
        txt1.move_to(ax.coords_to_point(0.5, -0.25 * ymax))

        self.play(Create(path1, rate_func=linear, run_time=4), FadeIn(txt1, run_time=1, rate_func=linear))
        self.wait(0.1)

        np.random.seed(seeds[1])
        b_vals2 = np.concatenate(([0], np.random.normal(scale=s, size=ndt).cumsum()))
        b_vals2 -= b_vals2[-1] * t_vals

        path2 = ax.plot_line_graph(t_vals, b_vals2, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)

        self.play(path1.animate(run_time=0.5, rate_func=linear).set_stroke(opacity=0.3, color=BLUE).set_z_index(1.5),
                  FadeOut(txt1, run_time=0.5, rate_func=linear),
                  Create(path2, run_time=2, rate_func=linear))

        i = np.argmax(b_vals2)
        b_vals3 = np.concatenate((np.maximum.accumulate(b_vals2[:i]), np.maximum.accumulate(b_vals2[::-1][:npts-i])[::-1]))
        b_vals3 = b_vals3 * 2 - b_vals2

        path3 = ax.plot_line_graph(t_vals, b_vals3, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)

        self.play(mh.rtransform(path2, path3))
        self.wait(0.1)

        np.random.seed(seeds[3])
        b_vals5 = np.concatenate(([0], np.random.normal(scale=s, size=ndt).cumsum()))
        if max(b_vals5) + min(b_vals5) < 0:
            b_vals5 = -b_vals5


        # break out excursions
        ax2 = Axes(x_range=[0, 1.05], y_range=[0, 2.], x_length=4, y_length=2.5,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                   x_axis_config={'include_ticks': True},
                  ).set_z_index(10)
        ax2.move_to(ax.coords_to_point(0.4, ymax*0.65))
        box = SurroundingRectangle(ax2, stroke_opacity=0, stroke_width=0, fill_color=BLACK, fill_opacity=0.7,
                                   corner_radius=0.15).set_z_index(3)
        min_n = 80
        i = 1
        pairs = []
        while i < ndt:
            j = np.argmax(b_vals5[i+1:] <= 0.) if b_vals5[i] > 0 else np.argmax(b_vals5[i+1:] >= 0.)
            k = int(i + j)
            if k > ndt:
                break
            if j > min_n:
                pairs.append((i-1, k))
            i = k + 1

        pairs3 = []
        for i, (j, k) in enumerate(pairs):
            b_vals5[j] = 0.
            b_vals5[k] = 0.
            i0 = 0 if i == 0 else pairs[i-1][1]
            if j > i0:
                pairs3.append((i0, j))
        if pairs[-1][1] < ndt:
            pairs3.append((pairs[-1][1], ndt))

        print(pairs)
        print(pairs3)

        path5 = ax.plot_line_graph(t_vals, b_vals5, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)
        self.play(path3.animate(run_time=0.5, rate_func=linear).set_stroke(opacity=0.3, color=ORANGE).set_z_index(1.5),
                  Create(path5, run_time=2, rate_func=linear))
        self.wait(0.1)


        paths = []
        paths2 = []
        paths3 = []
        colors = [BLUE, GREEN, TEAL, ORANGE, PURPLE, MAROON, GOLD]
        for idx, (i, j) in enumerate(pairs):
            col = colors[idx % len(colors)]
            t = t_vals[i:j+1]
            b = b_vals5[i:j+1]
            paths.append(ax.plot_line_graph(t, b, add_vertex_dots=False, stroke_color=col,
                               stroke_width=4).set_z_index(5))
            t = t - t[0]
            a = 1./t[-1]
            t *= a
            b = b * math.sqrt(a)
            if b[1] < 0:
                b = -b
            paths2.append(ax2.plot_line_graph(t, b, add_vertex_dots=False, stroke_color=col,
                               stroke_width=4).set_z_index(5 + idx * 0.001))

        for i, j in pairs3:
            paths3.append(ax.plot_line_graph(t_vals[i:j+1], b_vals5[i:j+1], add_vertex_dots=False, stroke_color=YELLOW,
                               stroke_width=4).set_z_index(4))

        gp1 = VGroup(*paths)
        gp1_1 = gp1.copy()
        gp2 = VGroup(*paths2)
        gp3 = VGroup(*paths3)
        self.play(FadeIn(gp1, gp3), FadeOut(path5))
        self.wait(0.1)



        self.play(FadeIn(box, ax2, run_time=1),
                  mh.rtransform(gp1, gp2, run_time=2)
                  )
        self.wait(0.1)
        self.play(mh.rtransform(gp2, gp1_1), FadeOut(box, ax2, run_time=1), run_time=2)
        self.play(FadeOut(gp3, gp1_1), FadeIn(path5))
        self.wait(0.1)

        np.random.seed(seeds[2])
        b_vals4 = np.concatenate(([0], np.random.normal(scale=s, size=ndt).cumsum()))
        b_vals4 -= b_vals4[-1] * t_vals
        b_vals4 = np.maximum.accumulate(b_vals4) * 2 - b_vals4

        path4 = ax.plot_line_graph(t_vals, b_vals4, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)

        self.play(path5.animate(run_time=0.5, rate_func=linear).set_stroke(opacity=0.3, color=GREEN).set_z_index(1.5),
                  Create(path4, run_time=2, rate_func=linear))
        self.wait(0.1)

        self.wait()

class ThreeProcs(Bridge):
    def construct(self):
        seeds = [129, 115, 103] # 140
        npts = 1920
        ndt = npts - 1
        np.random.seed(seeds[0])

        t0 = 0.7

        t_vals = np.linspace(0, 1., npts)
        i0 = round(t0 * ndt)
        t0 = t_vals[i0]

        ax, eqt, xlen, ylen, ymax, mark1, line1, eqt, eq1 = self.get_axes(1.6, ylen=0.75)
        ax.y_axis.set_z_index(10)
        gp = VGroup(ax, eqt, mark1, line1, eqt).to_edge(DOWN, buff=0.2)
        line2 = line1.copy()
        mark2 = mark1.copy()
        VGroup(mark1, line1, eqt).shift(ax.coords_to_point(t0, 0)-mark1.get_center())
        self.add(gp)
        self.wait(0.1)

        s = np.sqrt(t_vals[1])

        while True:
            b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
            if b_vals[i0] < 0:
                b_vals = -b_vals
            i1 = i0 - np.argmax(b_vals[i0::-1] < 0)
            if i1 <= 100 or i1 > i0 - 100:
                continue
            if abs(b_vals[i1+1]) < abs(b_vals[i1]):
                i1 += 1
            b_vals[i1] = 0
            i2 = np.argmax(b_vals[i0:] < 0) + i0
            if i2 > ndt or b_vals[i2] >= 0 or i2 < i0 + 100:
                continue
            if abs(b_vals[i2-1]) < abs(b_vals[i2]):
                i2 -= 1
            if i0 - i1 < ndt/4 or i1 < ndt/4:
                continue
            b_vals[i2] = 0
            break


        path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)
        self.play(Create(path1, rate_func=linear), run_time=3)
        self.wait(0.1)

        path2 = ax.plot_line_graph(t_vals[:i1+1], b_vals[:i1+1], add_vertex_dots=False, stroke_color=RED,
                                   stroke_width=4).set_z_index(3)
        path3 = ax.plot_line_graph(t_vals[i1:i0+1], b_vals[i1:i0+1], add_vertex_dots=False, stroke_color=GREEN,
                                   stroke_width=4).set_z_index(3)
        path4 = ax.plot_line_graph(t_vals[i0:i2+1], b_vals[i0:i2+1], add_vertex_dots=False, stroke_color=BLUE,
                                   stroke_width=4).set_z_index(3)
        path5 = ax.plot_line_graph(t_vals[i2:npts], b_vals[i2:npts], add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2.5)

        tscale=5
        self.play(Create(path2), rate_func=linear, run_time=t_vals[i1] * tscale)
        self.wait(0.1)
        self.play(Create(path3), rate_func=linear, run_time=(t_vals[i0]-t_vals[i1]) * tscale)
        self.wait(0.1)
        self.play(Create(path4), rate_func=linear, run_time=(t_vals[i2]-t_vals[i0]) * tscale)
        self.wait(0.1)
        self.add(path5)
        self.remove(path1)
        self.play(FadeIn(path5), FadeOut(path1))
        self.wait(0.1)

        scale = 0.47

        gax = VGroup(ax, line2, mark2, eq1.next_to(mark2, DR, buff=0.1))
        gax2 = gax.copy().scale(scale, about_edge=DL)
        gax3 = gax.copy().scale(scale, about_edge=UR)
        gax4 = gax.copy().scale(scale, about_edge=DR)
        ax2, ax3, ax4 = (gax2[0], gax3[0], gax4[0])

        t_vals2 = t_vals[:i1+1].copy()
        a = 1. / t_vals2[-1]
        t_vals2 *= a
        b_vals2 = b_vals[:i1+1] * math.sqrt(a)
        path6 = ax2.plot_line_graph(t_vals2, b_vals2, add_vertex_dots=False, stroke_color=RED,
                                   stroke_width=4).set_z_index(2.5)

        t_vals4 = t_vals[i1:i0+1] - t_vals[i1]
        a = 1. / t_vals4[-1]
        t_vals4 *= a
        b_vals4 = b_vals[i1:i0+1] * math.sqrt(a)
        path8 = ax4.plot_line_graph(t_vals4, b_vals4, add_vertex_dots=False, stroke_color=GREEN,
                                   stroke_width=4).set_z_index(2.5)

        t_vals3 = t_vals[i1:i2+1] - t_vals[i1]
        a = 1. / t_vals3[-1]
        t_vals3 *= a
        b_vals3 = b_vals[i1:i2+1] * math.sqrt(a)
        path7 = ax3.plot_line_graph(t_vals3[:i0-i1+1], b_vals3[:i0-i1+1], add_vertex_dots=False, stroke_color=GREEN,
                                   stroke_width=4).set_z_index(2.5)
        path7_1 = ax3.plot_line_graph(t_vals3[i0-i1:i2-i1+1], b_vals3[i0-i1:i2-i1+1], add_vertex_dots=False, stroke_color=BLUE,
                                   stroke_width=4).set_z_index(2.6)

        gp2 = VGroup(gp, path2, path3, path4, path5)

        eqb = MathTex(r'b_t', color=RED, font_size=60, stroke_width=2).move_to(ax2.coords_to_point(0.5, ymax*0.6)).set_z_index(10)
        eqe = MathTex(r'e_t', color=BLUE, font_size=60, stroke_width=2).move_to(ax3.coords_to_point(0.5, ymax*0.6)).set_z_index(10)
        eqm = MathTex(r'm_t', color=GREEN, font_size=60, stroke_width=2).move_to(ax4.coords_to_point(0.5, ymax*0.7)).set_z_index(10)

        self.play(gp2.animate.scale(0.45, about_edge=UL),
                  mh.rtransform(gax.copy().set_opacity(0), gax2, gax.copy().set_opacity(0), gax3,
                                gax.copy().set_opacity(0), gax4,
                                path2.copy(), path6,
                                path3.copy(), path8,
                                path3.copy(), path7,
                                path4.copy(), path7_1),
                  FadeIn(eqb, shift=mh.diff(path2, path6)),
                  FadeIn(eqe, shift=mh.diff(path3, path7)),
                  FadeIn(eqm, shift=mh.diff(path3.copy(), path8)),
                  run_time=2)
        self.wait(0.1)
        eqb2 = Tex(r'\sf bridge', font_size=80, stroke_width=2, color=RED).next_to(ax2.coords_to_point(0.5, 0), DOWN, buff=0.7).set_z_index(10).align_to(ax2, DOWN)
        eqe2 = Tex(r'\sf excursion', font_size=80, stroke_width=2, color=BLUE).next_to(ax3.coords_to_point(0.5, 0), DOWN, buff=0.1).set_z_index(10)
        eqm2 = Tex(r'\sf meander', font_size=80, stroke_width=2, color=GREEN).next_to(ax4.coords_to_point(0.5, 0), DOWN, buff=0.3).set_z_index(10)
        self.play(FadeIn(eqb2))
        self.wait(0.1)
        self.play(FadeIn(eqe2))
        self.wait(0.1)
        self.play(FadeIn(eqm2))
        self.wait(0.1)

        self.play(FadeOut(gp2, rate_func=linear))

        self.wait()

class Vervaat(Bridge):
    def construct(self):
        seeds = [170, 180, 174, 169]
        npts = 1920
        ndt = npts - 1
        np.random.seed(seeds[0])

        ax, eqt, xlen, ylen, ymax, mark1, line1, _, eq1 = self.get_axes(1.3, ylen=0.75, scale_neg=0.6)
        ymax2 = 0.9375
        print(ymax)
        ax.y_axis.set_z_index(10)
        gp = VGroup(ax, eqt, mark1, line1, eq1).to_edge(DOWN, buff=0.2)
        txt1 = Tex(r'\sf Brownian bridge', color=YELLOW, font_size=100)
        txt1.move_to(ax.coords_to_point(0.5, 0.6*ymax))
        self.add(gp, txt1)
        self.wait(0.1)
        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])
        while True:
            b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
            b_vals -= b_vals[-1] * t_vals
            if ndt * 0.4 < np.argmin(b_vals) < ndt * 0.6 and 0.5 * ymax2 < -min(b_vals)\
                    and max(b_vals) - min(b_vals) < ymax2 * 1.05 and max(b_vals) > 0.2 * ymax2:
                break

        path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)
        path1_1 = path1.copy().set_stroke(color=ManimColor(RED.to_rgb()*0.3)).set_z_index(1)

        eqb = MathTex(r'b_t', stroke_width=1, color=YELLOW)
        eqb.move_to(ax.coords_to_point(0.47, -ymax*0.35))

        self.play(Create(path1, rate_func=linear, run_time=2), FadeIn(eqb, run_time=1))
        self.wait(0.1)

        i0 = np.argmin(b_vals)
        t0 = t_vals[i0]
        y0 = b_vals[i0]

        line2 = DashedLine(ax.coords_to_point(0, y0), ax.coords_to_point(1, y0), color=GREY).set_z_index(4)
        dot = Dot(radius=0.1, color=GREY).set_z_index(5).move_to(ax.coords_to_point(t0, y0))
        self.play(FadeIn(line2, dot))
        self.wait(0.1)

        path2 = ax.plot_line_graph(t_vals[:i0+1], b_vals[:i0+1], add_vertex_dots=False, stroke_color=RED,
                                   stroke_width=4).set_z_index(2.2)
        path3 = ax.plot_line_graph(t_vals[i0:], b_vals[i0:], add_vertex_dots=False, stroke_color=BLUE,
                                   stroke_width=4).set_z_index(2.1)

        self.play(Create(path2, run_time=2 * t0, rate_func=linear),
                  eqb.animate(run_time=0.8).set_color(RED),
                  FadeOut(txt1, rate_func=linear))
        self.wait(0.1)
        self.play(Create(path3, run_time=2 * (1-t0), rate_func=linear), FadeOut(dot, run_time=0.5))
        self.remove(path1)
        self.wait(0.1)

        origin = ax.coords_to_point(0, 0)
        rshift = ax.coords_to_point(1-t0, 0) - origin
        lshift = origin - ax.coords_to_point(t0, 0)
        ushift = origin - ax.coords_to_point(0, y0)

        self.add(path1_1)
        self.play(path2.animate.shift(rshift), path3.animate.shift(lshift),
                  eqb.animate(run_time=1).set_color(ManimColor(RED.to_rgb()*0.4)),
                  run_time=3)
        self.wait(0.1)

        eqe = MathTex(r'e_t', stroke_width=1, color=BLUE)
        eqe.move_to(ax.coords_to_point(0.9, ymax * 0.63))
        txt2 = Tex(r'\sf excursion', font_size=100, color=BLUE)
        txt2.move_to(ax.coords_to_point(0.5, 0.8 * ymax))
        gp1 = VGroup(path2, path3)
        self.play(gp1.animate.shift(ushift),
                  FadeIn(eqe, rate_func=rush_into),
                  run_time=1.2)
        self.wait(0.1)
        self.play(FadeIn(txt2))
        self.wait(0.1)

        path2_1 = VGroup(path2, path3).copy().set_stroke(color=ManimColor(BLUE.to_rgb()*0.3)).set_z_index(1)
        self.add(path2_1)
        self.play(path2.animate(run_time=2).shift(-rshift-ushift),
                  path3.animate(run_time=2).shift(-lshift-ushift),
                  eqe.animate(run_time=1).set_color(ManimColor(BLUE.to_rgb()*0.4)),
                  FadeOut(txt2, run_time=1))
        self.wait(0.1)

        p0 = ax.coords_to_point(0, y0)
        y1 = max(b_vals)
        p1 = ax.coords_to_point(1, y1)
        box1 = Rectangle(width=(p1-p0)[0], height=(p1-p0)[1], stroke_width=0, stroke_opacity=0,
                  fill_color=GREY, fill_opacity=0.5).set_z_index(1)
        box1.next_to(p0, UR, buff=0)
        t1 = 0.75
        arr1 = Arrow(ax.coords_to_point(t1, y0), ax.coords_to_point(t1, y1), color=WHITE, buff=0).set_z_index(10)
        eqv = MathTex(r'V', stroke_width=2).next_to(arr1, RIGHT, buff=0).set_z_index(5).shift(DOWN*0.2)

        self.play(FadeIn(box1, arr1, eqv))
        self.wait(0.1)
        self.play(path2.animate.shift(rshift), path3.animate.shift(lshift),
                  run_time=2)
        gp2 = VGroup(path2, path3, box1, arr1, eqv)
        self.play(gp2.animate.shift(ushift), run_time=1.2)
        self.wait(0.1)

        self.play(path2.animate.shift(-rshift-ushift), path3.animate.shift(-lshift-ushift),
                  VGroup(box1, arr1, eqv).animate.shift(-ushift).set_opacity(0),
                  run_time=2)
        self.wait(0.1)

        #self.play(Rotate(path2, PI, about_point=ax.coords_to_point(t0/2, y0/2)), run_time=2)
        self.play(path2.animate.stretch(-1, dim=0), run_time=2)
        self.wait(0.1)
        self.play(path3.animate.shift(ushift))
        self.play(VGroup(path2, path3).animate.shift(ushift))
        txt3 = Tex(r'meander', font_size=100, color=GREEN).set_z_index(10)
        txt3.move_to(ax.coords_to_point(0.38, 0.9 * ymax))
        self.play(FadeIn(txt3))
        self.wait()

class MeanderTfm(Bridge):
    def construct(self):
        seeds = [170]
        npts = 1920
        ndt = npts - 1
        np.random.seed(seeds[0])
        scale_neg = 0.3
        b_scale=0.5

        ax, eqt, xlen, ylen, ymax, mark1, line1, _, eq1 = self.get_axes(1.3, ylen=0.75, scale_neg=scale_neg)
        ax.y_axis.set_z_index(10)
        gp = VGroup(ax, eqt, mark1, line1, eq1).to_edge(DOWN, buff=0.2)
        txtbm = Tex(r'\sf Brownian bridge', font_size=70, color=YELLOW)[0].set_z_index(5).move_to(ax.coords_to_point(0.45, -ymax*0.15))
        self.add(gp, txtbm)
        self.wait(0.1)

        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])
        b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum())) * b_scale
        b_vals -= b_vals[-1] * t_vals

        b_abs = np.abs(b_vals)
        i_s = [0]
        i_t = []

        while True:
            i = i_s[-1]
            j = np.argmax(b_abs[i+1:]) + i+1
            if j >= ndt or j < i + 5:
                print(1, i, j)
                break
            k = np.argmax(b_vals[j+1:] < 1e-6) if b_vals[j] > 0 else np.argmax(b_vals[j+1:] > -1e-6)
            k += j+1
            if k >= ndt:
                print(2, i, j, k)
                break
            if b_abs[k-1] < b_abs[k]:
                k -= 1
            if k < j + 1:
                print(3, i, j, k)
                break
            b_vals[k] = 0
            i_t.append(j)
            i_s.append(k)

        i_s[-1] = ndt

        path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)

        eqb = MathTex(r'b_t', color=YELLOW).move_to(ax.coords_to_point(0.14, -ymax *0.14)).set_z_index(5)
        self.play(Create(path1, rate_func=linear, run_time=2), FadeIn(eqb, run_time=0.8))
        self.wait(0.1)

        slines = [
            DashedLine(ax.coords_to_point(t_vals[i], -ymax*scale_neg), ax.coords_to_point(t_vals[i], ymax), color=GREY).set_z_index(1).set_opacity(0.6)
            for i in i_s[1:]
        ]
        tlines = [
            DashedLine(ax.coords_to_point(t_vals[i], -ymax*scale_neg), ax.coords_to_point(t_vals[i], ymax), color=GREY).set_z_index(1).set_opacity(0.6)
            for i in i_t
        ]
        mt = len(tlines)

        t_paths = []
        s_paths = []
        s_eqs = []
        t_eqs = []
        for i in range(mt):
            j, k = (i_s[i], i_t[i])
            l = i_s[i+1]
            tpath = ax.plot_line_graph(t_vals[j:k+1], b_vals[j:k+1], add_vertex_dots=False, stroke_color=RED,
                                   stroke_width=4).set_z_index(3 - i/100)
            t_paths.append(tpath)
            s_paths.append(ax.plot_line_graph(t_vals[k:l+1], b_vals[k:l+1], add_vertex_dots=False, stroke_color=BLUE,
                                   stroke_width=4).set_z_index(3 - (i+0.5)/100))
            s_eqs.append(MathTex(r'S_{{ {} }}'.format(i), font_size=40).next_to(ax.coords_to_point(t_vals[j], -ymax*0.15), DR, buff=0.1).set_z_index(5))
            t_eqs.append(MathTex(r'T_{{ {} }}'.format(i+1), font_size=40).next_to(ax.coords_to_point(t_vals[k], -ymax*0.15), DR, buff=0.1).set_z_index(5))

        t_paths2 = [p.copy().set_stroke(color=YELLOW) for p in t_paths]
        s_paths2 = [p.copy().set_stroke(color=YELLOW) for p in s_paths]


        eqmax = MathTex(r'\max\lvert b_t\rvert')[0].set_z_index(5)
        eqmax[-3:-1].set_color(RED)
        i0 = i_t[0]
        eqmax.move_to(ax.coords_to_point(t_vals[i0] - 0.1, b_vals[i0] + 0.4 * ymax))
        arr1 = Arrow(eqmax.get_bottom(), ax.coords_to_point(t_vals[i0], b_vals[i0]) + UL*0.03, buff=0, color=RED).set_z_index(5)

        self.play(FadeIn(eqmax, arr1))
        self.wait(0.1)

        self.remove(path1)
        self.add(*t_paths2, *s_paths2)
        self.play(FadeIn(s_eqs[0]))

        i1 = i_s[1]
        eqmaxm = MathTex(r'\max m_t')[0].set_z_index(5)
        eqmaxm[-2:].set_color(BLUE)
        eqmaxm.move_to(ax.coords_to_point(t_vals[i1] - 0.2, b_vals[i0]*2 + 0.15 * ymax))
        arr2 = Arrow(eqmaxm.get_right()+RIGHT*0.05, ax.coords_to_point(t_vals[i1], b_vals[i0]*2) + LEFT*0.03, buff=0, color=BLUE).set_z_index(5)
        eqm = MathTex(r'm_t', color=BLUE).move_to(ax.coords_to_point(0.46, 0.67 * ymax)).set_z_index(5)

        rate = 4.
        m = 2
        newcol = ManimColor(RED.to_rgb()*0.3)
        newcol2 = ManimColor(RED.to_rgb()*0.45)
        for i in range(m):
            dt = (t_vals[i_t[i]] - t_vals[i_s[i]])*rate
            anims = [Create(t_paths[i], rate_func=linear), FadeIn(tlines[i], t_eqs[i], rate_func=rush_into)]
            if i == 0:
                anims += [FadeOut(txtbm, run_time=1)]
            self.play(*anims, run_time=dt)
            t_paths2[i].set_stroke(color=newcol)
            self.wait(0.1)
            dt = (t_vals[i_s[i+1]] - t_vals[i_t[i]])*rate
            anims = [Create(s_paths[i], rate_func=linear), FadeIn(slines[i], s_eqs[i+1], rate_func=rush_into)]
            self.play(*anims, run_time=dt)
            s_paths2[i].set_stroke(color=newcol)
            self.wait(0.1)
        self.play(FadeIn(*t_paths[m:], *s_paths[m:], *tlines[m:], *slines[m:]))
        VGroup(*t_paths2[m:], *s_paths2[m:]).set_stroke(color=newcol)
        #self.remove(*t_paths2, *s_paths2)
        self.wait(0.1)


        #self.play(FadeIn(*slines, *tlines, *t_paths, *s_paths), FadeOut(path1))
        #self.wait(0.1)

        origin = ax.coords_to_point(0, 0)
        anims = []
        for i in range(mt):
            if b_vals[i_t[i]] < 0:
                anims.append(VGroup(s_paths[i], t_paths[i]).animate.stretch(-1, dim=1, about_point=origin))
        self.play(anims[0], run_time=2)
        self.wait(0.1)
        self.play(*anims[1:], run_time=1)
        self.wait(0.1)

        anims = []
        t_paths3 = [t.copy() for t in t_paths]
        for i in range(mt):
            tpath = t_paths3[i]['line_graph']
            p = 0.5*(tpath.get_start() + tpath.get_end())
            q = tpath.get_end()
            dt = (0.5)**i
            t_paths3[i].rotate(PI, about_point=p)
            anims = [Rotate(t_paths[i], PI, about_point=p, run_time=2.5 * dt)]
            if i == 0:
                anims.append(eqb.animate.set_color(newcol2))
            self.play(*anims)
            if i < 2:
                #anims.append(Wait(0.1))
                self.wait(0.1)
            gp2 = VGroup(*t_paths[i+1:], *s_paths[i:])
            anims = [gp2.animate(run_time=1.5*dt).stretch(-1, dim=1, about_point=q)]
            if i == 0:
                txtbm[8:].set_color(newcol2).move_to(ax.coords_to_point(0.35, 0.14 * ymax))
                anims.append(FadeIn(txtbm[8:], run_time=1.5*dt))
            self.play(*anims)
            VGroup(*t_paths3[i+1:]).stretch(-1, dim=1, about_point=q)
            if i == 0:
                #anims.append(FadeIn(eqmaxm, arr2, eqm, run_time=1))
                self.play(FadeIn(eqmaxm, arr2, eqm))
            if i < 2:
                #anims.append(Wait(0.1))
                self.wait(0.1)

        #self.play(Succession(*anims))

        self.wait(0.1)

        y0 = b_vals[i0]
        lines1 = [
            DashedLine(ax.coords_to_point(0, y0), ax.coords_to_point(1, y0),
                       color=GREY).set_z_index(1).set_opacity(0.6),
            DashedLine(ax.coords_to_point(0, y0*2), ax.coords_to_point(1, y0*2),
                       color=GREY).set_z_index(1).set_opacity(0.6)
        ]
        self.play(FadeIn(*lines1))

        self.wait()

class MeanderTxt(Scene):
    def construct(self):
        eq = Tex(r'\sf meander', color=BLUE, font_size=70)
        self.add(eq)

class MaxExc(Scene):
    def construct(self):
        eq1 = MathTex(r'\max e_t', r'\sim V', r'\sim{\rm range}\,b_t', stroke_width=1, font_size=60)
        eq1[0][-2:].set_color(BLUE)
        eq1[2][-2:].set_color(RED)
        self.add(eq1[0])
        self.wait(0.1)
        self.play(FadeIn(eq1[1]))
        self.wait(0.1)
        self.play(FadeIn(eq1[2]))
        self.wait()

class SDExc(Scene):
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1)
        eq1 = MathTex(r'{\rm s.d.}(e_t)', r'=\sqrt{\int(e_t-\bar e)^2dt}')
        eq2 = MathTex(r'{\rm s.d.}(e_t)', r'\sim\pi^{-1} D', r'\sim\ {\rm s.d.}(b_t)')
        mh.align_sub(eq2, eq2[0], eq1[0])
        VGroup(eq1[0][-3:-1], eq2[0][-3:-1], eq1[1][5:7], eq1[1][8:10]).set_color(BLUE)
        eq2[2][-3:-1].set_color(RED)
        self.add(eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0], eq2[0]),
                  AnimationGroup(FadeOut(eq1[1]), FadeIn(eq2[1]), rate_func=linear))
        self.wait(0.1)
        self.play(FadeIn(eq2[2]))
        self.wait()

class MeanderMax(Scene):
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1)
        eq1 = MathTex(r'\max m_t', r'\sim 2D', r'\sim 2\max\lvert b_t\rvert')
        eq1[0][-2:].set_color(GREEN)
        eq1[2][-3:-1].set_color(RED)
        self.add(eq1[0])
        self.wait(0.1)
        self.play(FadeIn(eq1[1]))
        self.wait(0.1)
        self.play(FadeIn(eq1[2]))
        self.wait()

class BridgeExcursion(Scene):
    def construct(self):
        MathTex.set_default(font_size=70)
        eq1 = MathTex(r'\max e_t', r'=', r'{\rm range}\,b_t', r'\sim V').set_z_index(2)
        eq2 = MathTex(r'{\rm s.d.}(e_t)', r'=', r'{\rm s.d.}(b_t)', r'\sim\pi^{-1}D').set_z_index(2)
        VGroup(eq1[0][-2:], eq2[0][-3:-1]).set_color(BLUE)
        VGroup(eq1[2][-2:], eq2[2][-3:-1]).set_color(RED)
        eq2.next_to(eq1, DOWN, buff=0.1)
        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=RIGHT)
        box = SurroundingRectangle(VGroup(eq1, eq2), fill_color=BLACK, fill_opacity=0.7,
                                   stroke_width=0, stroke_opacity=0, buff=0.1, corner_radius=0.15)
        self.add(eq1, eq2, box)

class BridgeMeander(Scene):
    def construct(self):
        MathTex.set_default(font_size=70)
        eq1 = MathTex(r'\max m_t', r'=', r'2\,\max\lvert b_t\rvert', r'\sim 2V').set_z_index(2)
        eq1[0][-2:].set_color(GREEN)
        eq1[2][-3:-1].set_color(RED)
        box = SurroundingRectangle(eq1, fill_color=BLACK, fill_opacity=0.7,
                                   stroke_width=0, stroke_opacity=0, buff=0.1, corner_radius=0.15)
        self.add(box, eq1)

class KolDistribution(Scene):
    def construct(self):
        MathTex.set_default(font_size=60)
        eq1 = MathTex(r'\mathbb P(D > x)', r'=', r'2e^{-2x^2}-2e^{-8x^2}+2e^{-18x^2}-\cdots',
                      r'=', r'2\sum_{n=1}^\infty (-1)^{n-1}e^{-2n^2x^2}').set_z_index(2)
        mh.align_sub(eq1[3:], eq1[3], eq1[1]).next_to(eq1[2], DOWN, buff=0.2, coor_mask=UP)
        eq1.move_to(ORIGIN)
        box = SurroundingRectangle(eq1, fill_color=BLACK, fill_opacity=0.7, corner_radius=0.15,
                                   stroke_width=0, stroke_opacity=0, buff=0.2)
        VGroup(eq1, box).to_edge(DOWN, buff=0.1)
        self.add(box, eq1[:3])
        self.wait(0.1)
        self.play(FadeIn(eq1[3:]))
        self.wait(0.1)
        eq2 = eq1[:3].copy().to_edge(DOWN).shift(DOWN*37*config.frame_y_radius/540).set_color(GREY)
        box2 = SurroundingRectangle(eq2, fill_color=BLACK, fill_opacity=0.7, corner_radius=0.15,
                                   stroke_width=0, stroke_opacity=0, buff=0.2)
        self.play(FadeOut(eq1[3:]), mh.rtransform(box, box2, eq1[:3], eq2[:]))
        self.wait(0.1)
        self.play(FadeOut(box2))
        self.wait(0.1)
        for eq in (eq2[2][:6], eq2[2][7:13], eq2[2][14:21]):
            eq_1 = eq.copy()
            eq_2 = eq.copy().scale(1.3).set_color(WHITE)
            self.play(mh.transform(eq, eq_2))
            self.play(mh.transform(eq, eq_1))
            self.wait(0.1)

        self.wait()

class Reflection(Bridge):
    def construct(self):
        seeds = [205]
        npts = 1920
        ndt = npts - 1

        np.random.seed(seeds[0])
        yellow2 = ManimColor(YELLOW.to_rgb()*0.5)
        yellow3 = ManimColor(YELLOW.to_rgb()*0.3)
        blue3 = ManimColor(BLUE.to_rgb()*0.3)

        ax, eqt, xlen, ylen, ymax, mark1, line1, _, eq1 = self.get_axes(scale_neg=0.5)
        ax.y_axis.set_z_index(20)
        origin = ax.coords_to_point(0, 0)
        right = ax.coords_to_point(1, 0) - origin
        up = ax.coords_to_point(0, ymax) - origin

        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])

        self.add(ax, mark1, line1, eqt, eq1)
        self.wait(0.1)

        attempts = 0
        while True:
            attempts += 1
            b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
            if max(b_vals) + min(b_vals) < 0:
                b_vals = -b_vals
            level = max(b_vals) * 0.8
            i0 = np.argmax(b_vals > level)
            if b_vals[-1] > level:
                b_vals[i0:] = 2*level - b_vals[i0:]
            b_vals_r = b_vals.copy()
            b_vals_r[i0:] = 2*level - b_vals_r[i0:]
            i1 = ndt - np.argmax(b_vals[::-1] > level)
            if not 0.4 * ndt < i0 < 0.6 * ndt < i1 < 0.8 * ndt or level < 0.35 * ymax\
                or max(b_vals) > ymax or max(b_vals_r) > ymax:# or abs(b_vals[-1]) < ymax * 0.08\
            #    or min(b_vals[:i0]) > -ymax * 0.1:
                continue
            if abs(b_vals[i0-1] - level) < abs(b_vals[i0] - level): i0 -= 1
            b_vals[i0] = level
            print('attempts: ', attempts)
            break

        b_vals[:i0-20] -= t_vals[:i0-20] * ymax * 0.05 # nudge away from barrier

        path1_1 = ax.plot_line_graph(t_vals[:i0+1], b_vals[:i0+1], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(3)
        path1_2 = ax.plot_line_graph(t_vals[i0:], b_vals[i0:], add_vertex_dots=False, stroke_color=yellow2, stroke_width=5).set_z_index(2.9)
        path1_3 = ax.plot_line_graph(t_vals[i0:], 2*level-b_vals[i0:], add_vertex_dots=False, stroke_color=BLUE, stroke_width=5).set_z_index(4)

        dt = 8
        t0 = t_vals[i0]
        line2 = Line(ax.coords_to_point(0, level), ax.coords_to_point(1, level), stroke_width=5, stroke_color=RED).set_z_index(2)
        txt2 = Tex(r'\sf level', stroke_width=1, color=RED, font_size=60).move_to(ax.coords_to_point(0.1, level + ymax * 0.2)).set_z_index(2)
        p = txt2.get_right()
        arr1 = CurvedArrow(p + 0.005 * right, p * RIGHT + 0.05 * right + line2.get_center() * UP + 0.01 * up, color=RED, radius=-1.6, stroke_width=6)
        self.play(Create(path1_1, rate_func=linear, run_time = dt * t0),
                  Succession(Wait(2), FadeIn(line2, txt2, arr1)))
        self.wait(0.1)

        txt3 = Tex(r'\sf original process', font_size=60, color=YELLOW, stroke_width=1).set_z_index(10)
        txt4 = Tex(r'\sf reflected', font_size=60, color=BLUE, stroke_width=1).set_z_index(10)
        txt3.move_to(ax.coords_to_point(0.58, 0.15 * ymax))
        txt4.move_to(ax.coords_to_point(0.58, 0.67 * ymax))
        txt3_1 = txt3.copy().set_z_index(9.9).set_color(BLACK).set_stroke(width=4)

        self.play(AnimationGroup(Create(path1_2, rate_func=linear), Create(path1_3, rate_func=linear), run_time=dt*(1-t0)),
                  FadeIn(txt3, txt3_1, txt4, rate_func=linear, run_time=1.5))
        self.wait(0.1)
        #self.play(path1_1.animate.set_stroke(color=BLUE), path1_2.animate.set_stroke(yellow3))

        path1_4 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(5)
        self.play(FadeIn(path1_4), path1_3.animate.set_stroke(color=blue3), txt4.animate.set_color(blue3), rate_func=linear)

        self.wait(0.1)
        path1_2.set_stroke(color=yellow3)
        self.play(FadeOut(path1_4), txt3.animate.set_color(yellow3),
                  path1_3.animate.set_stroke(color=BLUE),
                  path1_1.animate.set_stroke(color=BLUE),
                  txt4.animate.set_color(BLUE), rate_func=linear)



        self.wait()

class ReflectBridge(Bridge):
    def construct(self):
        seeds = [205]
        npts = 1920
        ndt = npts - 1

        np.random.seed(seeds[0])
        yellow2 = ManimColor(YELLOW.to_rgb()*0.5)
        yellow3 = ManimColor(YELLOW.to_rgb()*0.3)
        blue2 = ManimColor(BLUE.to_rgb()*0.5)
        blue3 = ManimColor(BLUE.to_rgb()*0.3)
        scale_neg=0.5

        ax, eqt, xlen, ylen, ymax, mark1, line1, _, eq1 = self.get_axes(scale_neg=scale_neg)
        ax.y_axis.set_z_index(20)
        origin = ax.coords_to_point(0, 0)
        right = ax.coords_to_point(1, 0) - origin
        up = ax.coords_to_point(0, ymax) - origin

        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])

        self.add(ax, mark1, line1, eqt, eq1)
        self.wait(0.1)

        attempts = 0
        eps = ymax * 0.07
        while True:
            attempts += 1
            b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
            b_vals -= t_vals * (b_vals[-1] - eps/2)
            if max(b_vals) + min(b_vals) < 0:
                b_vals = -b_vals
            level = np.max(b_vals) * 0.8
            i0 = np.argmax(b_vals > level)
            if abs(b_vals[i0-1] - level) < abs(b_vals[i0] - level): i0 -=1
            b_vals[i0] = level
            break

        #path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(3)
        path1_1 = ax.plot_line_graph(t_vals[:i0+1], b_vals[:i0+1], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(4)
        path1_2 = ax.plot_line_graph(t_vals[i0:], b_vals[i0:], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(3)
        path1_3 = ax.plot_line_graph(t_vals[i0:], 2*level-b_vals[i0:], add_vertex_dots=False, stroke_color=BLUE, stroke_width=5).set_z_index(5)
        path1_4 = path1_3.copy().set_stroke(color=blue2).set_z_index(2.9)


        pts = [ax.coords_to_point(1, eps), ax.coords_to_point(1, -eps)]
        box1 = Rectangle(width=0.4, height=up[1] -(pts[0]-origin)[1], color=DARK_GREY, fill_opacity=1, stroke_opacity=0).set_z_index(1)
        box2 = Rectangle(width=0.4, height=up[1]*scale_neg -(pts[0]-origin)[1], color=DARK_GREY, fill_opacity=1, stroke_opacity=0).set_z_index(1)
        box1.next_to(pts[0], UR, buff=0)
        box2.next_to(pts[1], DR, buff=0)
        eq3 = MathTex(r'\varepsilon', font_size=60).set_z_index(4).next_to(pts[0], LEFT, buff=0.1)
        eq4 = MathTex(r'-\varepsilon', font_size=60).set_z_index(4).next_to(pts[1], LEFT, buff=0.1)
        eq4_1 = eq4.copy().set_stroke(width=12, color=BLACK).set_z_index(3.5).next_to(pts[1], LEFT, buff=0.1)
        eq3_1 = eq3.copy().set_stroke(width=12, color=BLACK).set_z_index(3.5).next_to(pts[0], LEFT, buff=0.1)
        eqb1 = MathTex(r'B_t', font_size=60, color=YELLOW)[0].set_z_index(5)
        eqb2 = MathTex(r'B^r_t', font_size=60, color=BLUE)[0].set_z_index(5)
        eqb3 = MathTex(r'b_t', font_size=60, color=YELLOW)[0].set_z_index(5)
        eqb1.move_to(ax.coords_to_point(0.48, 0.13 * ymax))
        eqb2.move_to(ax.coords_to_point(0.48, level*2 - 0.11 * ymax))
        eqb3.move_to(eqb1)

        self.play(Create(VGroup(path1_1, path1_2), rate_func=linear, run_time=5),
                  Succession(Wait(3), FadeIn(box1, box2, eq3, eq4, eq3_1, eq4_1)),
                  Succession(Wait(2.7), FadeIn(eqb1, rate_func=linear)))
        self.wait(0.1)

        line2 = Line(ax.coords_to_point(0, level), ax.coords_to_point(1, level), stroke_width=5, stroke_color=RED).set_z_index(2)
        eq5 = MathTex(r'x').next_to(line2.get_left(), UR, buff=0.1)
        self.play(FadeIn(line2, eq5))
        self.wait(0.1)
        self.play(mh.rtransform(path1_2.copy(), path1_3, run_time=1.6),
                  Succession(Wait(0.6), FadeIn(eqb2, rate_func=linear)))
        self.wait(0.1)

        p = path1_2['line_graph'].get_end()
        arr1 = Arrow(p+RIGHT, p, color=WHITE, buff=0).set_z_index(6)
        self.add(path1_4)
        self.play(FadeIn(arr1), FadeOut(path1_3), eqb2.animate.set_color(blue2))
        self.wait(0.1)
        line3 = DashedLine(ax.coords_to_point(0, level*2), ax.coords_to_point(1, level*2), stroke_width=5, stroke_color=RED).set_z_index(2)
        eq6 = MathTex(r'2x').next_to(line3.get_left(), UR, buff=0.1)
        box3 = Rectangle(width=0.4, height=(level-eps)*2*up[1]/ymax, color=DARK_GREY, fill_opacity=1, stroke_opacity=0).set_z_index(1)
        box4 = Rectangle(width=0.4, height=(ymax-2*level-eps)*up[1]/ymax, color=DARK_GREY, fill_opacity=1, stroke_opacity=0).set_z_index(1)
        box3.align_to(box1, DL)
        box4.align_to(box1, UL)
        arr2 = arr1.copy()

        self.add(box3, box4)
        self.play(FadeIn(line3, eq6),
                  FadeIn(path1_3),
                  eqb2.animate.set_color(BLUE),
                  eqb1.animate.set_color(yellow2),
                  path1_2.animate.set_stroke(yellow2),
                  path1_1.animate.set_stroke(BLUE),
                  FadeOut(box1, rate_func=linear),
                  arr1.animate.next_to(path1_3['line_graph'].get_end(), RIGHT, buff=0),
                  rate_func=linear, run_time=1.2
                  )
        self.remove(path1_4)
        self.wait(0.1)

        b_vals2 = b_vals - t_vals*(1-t_vals)
        path1_1c = path1_1.copy().set_color(YELLOW)
        path1_2c = path1_2.copy()
        path2_1 = ax.plot_line_graph(t_vals[:i0+1], b_vals2[:i0+1], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(4)
        path2_2 = ax.plot_line_graph(t_vals[i0:], b_vals2[i0:], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(3)
        self.play(mh.transform(path1_1, path2_1, path1_2, path2_2),
                  FadeOut(path1_3, arr1), FadeIn(arr2),
                  eqb1.animate.shift(-up * 0.25/ymax).set_color(YELLOW),
                  eqb2.animate.shift(-up * 0.5/ymax))
        self.wait(0.1)
        self.play(mh.transform(path1_1, path1_1c, path1_2, path1_2c),
                  FadeIn(path1_3, arr1), FadeOut(arr2),
                  eqb1.animate.shift(up * 0.25/ymax).set_color(yellow2),
                  eqb2.animate.shift(up * 0.5/ymax)
                  )
        self.wait(0.1)
        self.add(path1_4)
        self.play(FadeOut(path1_3, arr1),
                  FadeIn(arr2),
                  eqb2.animate.set_color(blue2),
                  eqb1.animate.set_color(YELLOW),
                  path1_2.animate.set_stroke(YELLOW),
                  rate_func=linear, run_time=1
                  )
        self.wait(0.1)
        self.play(FadeIn(path1_3, arr1),
                  FadeOut(arr2),
                  path1_2.animate.set_stroke(yellow2),
                  path1_1.animate.set_stroke(BLUE),
                  eqb2.animate.set_color(BLUE),
                  eqb1.animate.set_color(yellow2),
                  rate_func = linear, run_time = 1)
        self.wait(0.1)
        path1_4.set_color(blue3)
        b_vals3 = b_vals - t_vals * b_vals[-1]
        path4_1 = ax.plot_line_graph(t_vals[:i0+1], b_vals3[:i0+1], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(4)
        path4_2 = ax.plot_line_graph(t_vals[i0:], b_vals3[i0:], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(4)
        path4_3 = ax.plot_line_graph(t_vals[i0:], 2*level-b_vals3[i0:], add_vertex_dots=False, stroke_color=blue3, stroke_width=4).set_z_index(3)
        box5 = box2.copy().stretch(ymax * scale_neg / (ymax * scale_neg-eps), 1).align_to(box2, DL)
        box6 = box3.copy().stretch(level / (level-eps), 1).align_to(line3.get_end(), UL)
        box7 = box4.copy().stretch((ymax-2*level) / (ymax-2*level-eps), 1).align_to(line3.get_end(), DL)
        self.play(FadeOut(eqb1, eqb2),
                  FadeIn(eqb3),
                  mh.rtransform(path1_1, path4_1, path1_2, path4_2,
                                path1_3, path4_3.copy().set_z_index(5).set_stroke(opacity=0),
                                path1_4, path4_3),
                  FadeOut(arr1, eq3, eq4, eq3_1, eq4_1),
                  mh.rtransform(box2, box5, box3, box6, box4, box7))
        self.wait(0.1)
        self.play(FadeOut(box5, box6, box7, path4_3, line3, eq6), rate_func=linear)
        self.wait(0.1)

        line4 = line2.copy().next_to(ax.coords_to_point(0, -level), RIGHT, buff=0)
        eq7 = MathTex(r'-x').next_to(line4.get_left(), UR, buff=0.1)

        self.play(mh.rtransform(line2.copy(), line4, eq5[0][0].copy(), eq7[0][1]),
                  FadeIn(eq7[0][0], shift=mh.diff(eq5[0][0], eq7[0][1])),
                  run_time=1.6)
        self.wait(0.1)
        eqU = MathTex(r'U', stroke_width=1, color=RED).next_to(line2.get_left(), DR, buff=0.15).set_z_index(10)
        eqL = MathTex(r'L', stroke_width=1, color=RED).next_to(line4.get_left(), DR, buff=0.15).set_z_index(10)
        self.play(FadeIn(eqU))
        self.wait(0.1)
        self.play(FadeIn(eqL))
        self.wait(0.1)

        # double barriers

        scale_neg2 = 0.3
        scale=0.7
        ax2, eqt2, xlen2, ylen2, ymax2, mark2, line1_2, _, eq1_2 = self.get_axes(scale=scale, scale_neg=scale_neg2)
        ax2.y_axis.set_z_index(20)

        level2 = level * 0.6
        np.random.seed(215)

        attempts = 0
        while True:
            attempts += 1
            b_vals4 = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
            b_vals4 -= t_vals * b_vals4[-1]
            if max(abs(b_vals4)) < level2 * 1.2:
                continue
            i0 = np.argmax(abs(b_vals4) > level2)
            if b_vals4[i0] < 0: b_vals4 = -b_vals4
            if not ndt * 0.25 < i0 < ndt * 0.45:
                continue
            if abs(b_vals4[i0-1]-level2) < abs(b_vals4[i0]-level2): i0 -= 1
            b_vals4[i0] = level2
            break
        print('attempts:', attempts)

        attempts = 0
        while True:
            attempts += 1
            b_vals_1 = np.random.normal(scale=s, size=ndt - i0).cumsum()
            b_vals_1 += (t_vals[i0+1:] - t_vals[i0]) * (level2 - b_vals_1[-1]) / (1-t_vals[i0])
            if max(b_vals_1) < 2 * level2:
                continue
            i1 = np.argmax(abs(b_vals_1) > 2 * level2)
            if ndt * 0.6 < i0 + i1 < ndt * 0.75:
                continue
            if abs(b_vals_1[i1-1]-2*level2) < abs(b_vals_1[i1]-2*level2): i1 -= 1
            b_vals_1[i1] = level2*2
            b_vals4[i0+1:] = level2 - b_vals_1
            i1 += i0 + 1
            break
        print('attempts:', attempts)

        attempts = 0
        while True:
            attempts += 1
            b_vals_1 = np.random.normal(scale=s, size=ndt - i1).cumsum()
            b_vals_1 += (t_vals[i1+1:] - t_vals[i1]) * (level2 - b_vals_1[-1]) / (1 - t_vals[i1])
            if max(b_vals_1) > 2 * level2:
                continue
            b_vals4[i1+1:] = b_vals_1 - level2
            break

        print('attempts:', attempts)

        path5_1 = ax2.plot_line_graph(t_vals[:i0+1], b_vals4[:i0+1], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(4)
        path5_2 = ax2.plot_line_graph(t_vals[i0:i1+1], b_vals4[i0:i1+1], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(4)
        path5_3 = ax2.plot_line_graph(t_vals[i1:], b_vals4[i1:], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(4)

        path5_4 = ax2.plot_line_graph(t_vals[i0:i1+1], 2*level2-b_vals4[i0:i1+1], add_vertex_dots=False, stroke_color=BLUE, stroke_width=5).set_z_index(3)
        path5_5 = ax2.plot_line_graph(t_vals[i1:], 4*level2+b_vals4[i1:], add_vertex_dots=False, stroke_color=BLUE, stroke_width=5).set_z_index(3)

        pts = [ax2.coords_to_point(0, level2), ax2.coords_to_point(0, -level2)]
        self.play(FadeOut(path4_1, path4_2, eqb3),
                  mh.rtransform(ax, ax2, eqt, eqt2, eq1, eq1_2, mark1, mark2, line1, line1_2),
                  VGroup(line2, eq5, eqU).animate.shift(pts[0] - line2.get_left()),
                  VGroup(line4, eq7, eqL).animate.shift(pts[1] - line4.get_left()),
                  run_time=1.2)
        self.wait(0.1)

        t0 = t_vals[i0]
        t1 = t_vals[i1]
        dt = 4
        self.play(Create(path5_1, rate_func=linear, run_time=dt*t0))
        self.wait(0.1)
        self.play(Create(path5_2, rate_func=linear, run_time=dt*(t1-t0)),
                  Create(path5_4, rate_func=linear, run_time=dt * (t1 - t0)),
                  )
        self.wait(0.1)

        line5 = line3.copy().move_to(ax2.coords_to_point(0, 3*level2), coor_mask=UP)
        eq8 = MathTex(r'3x').next_to(line5.get_left(), UR, buff=0.1)

        self.play(mh.fade_replace(line4.copy(), line5),
                  mh.rtransform(eq7[0][1].copy(), eq8[0][1]),
                  FadeIn(eq8[0][0], target_position=eq7[0][0]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(Create(path5_3, rate_func=linear, run_time=dt*(1-t1)),
                  Create(path5_5, rate_func=linear, run_time=dt * (1 - t1)),
                  )
        self.wait(0.1)

        line6 = line3.copy().move_to(ax2.coords_to_point(0, 4*level2), coor_mask=UP)
        eq9 = MathTex(r'4x').next_to(line6.get_left(), UR, buff=0.1)

        self.play(mh.fade_replace(line5.copy(), line6),
                  mh.rtransform(eq8[0][1].copy(), eq9[0][1]),
                  mh.fade_replace(eq8[0][0].copy(), eq9[0][0]),
                  run_time=1)
        self.wait(0.1)
        self.play(FadeOut(path5_1, path5_2, path5_3, path5_4, path5_5, line5, line6, eq8, eq9))
        self.wait(0.1)

        # lots of reflections

        scale_neg3 = 0.2
        scale=0.6
        ax3, eqt3, xlen3, ylen3, ymax3, mark3, line1_3, _, eq1_3 = self.get_axes(scale=scale, scale_neg=scale_neg3)
        ax3.y_axis.set_z_index(20)

        n = 4
        level3 = level2 * 0.5
        pts = [ax3.coords_to_point(0, level3), ax3.coords_to_point(0, -level3),
               ax.coords_to_point(0, 2 * n * level3)]
        self.play(
            mh.rtransform(ax2, ax3, eqt2, eqt3, eq1_2, eq1_3, mark2, mark3, line1_2, line1_3),
            VGroup(line2, eq5).animate.shift(pts[0] - line2.get_left()),
            VGroup(line4, eq7).animate.shift(pts[1] - line4.get_left()),
            eqU.animate.next_to(pts[0], DR, buff=0.1).shift(UP*0.05).scale(0.9, about_edge=UL),
            eqL.animate.next_to(pts[1], DR, buff=0.1).shift(UP * 0.05).scale(0.9, about_edge=UL),
        )
        print('level3', level3)

        levels = [level3 * (1+2*i) for i in range(n)] + [2 * n * level3]
        level_end = levels[-1]
        lines = [line3.copy().move_to(ax3.coords_to_point(0, levels[i]), coor_mask=UP) for i in range(1, n+1)]

        np.random.seed(223)
        np.random.seed(224)
        attempts=0
        while True:
            attempts += 1
            b_vals6 = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
            b_vals6 += t_vals * (level_end - b_vals6[-1])
            iarr = []
            i0 = 0
            steps = []
            for i in range(n):
                level4 = levels[i]
                i1 = np.argmax(b_vals6[i0+1:] > level4) + i0 + 1
                if abs(b_vals6[i1-1] - level4) < abs(b_vals6[i1] - level4): i1 -= 1
                b_vals6[i1] = level4
                iarr.append(i1)
                dt = t_vals[i1] - t_vals[i0]
                if i > 0: dt /= 2
                steps.append(dt)
                i0 = i1
            steps.append(1-t_vals[iarr[-1]])
            s1 = min(steps)
            s2 = max(steps)
            if s2/s1 > 2:
                continue
            if min(b_vals6[:iarr[0]]) < -level3 * 0.9:
                continue
            print(s2/s1)
            break
        iarr.append(ndt)
        print('attempts:', attempts)

        b_vals5 = b_vals6.copy()
        for i in iarr:
            b_vals5[i:] = 2 * b_vals5[i] - b_vals5[i:]


        paths6 = []
        paths7 = []
        i0 = 0
        for i1 in iarr:
            paths6.append(ax3.plot_line_graph(t_vals[i0:i1+1], b_vals5[i0:i1+1], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(4))
            paths7.append(ax3.plot_line_graph(t_vals[i0:i1+1], b_vals6[i0:i1+1], add_vertex_dots=False, stroke_color=BLUE, stroke_width=5).set_z_index(3))
            i0 = i1

        dot = Dot(radius=0.1, color=RED).set_z_index(8)
        dots1 = [dot.copy().move_to(paths6[i]['line_graph'].get_end()) for i in range(n)]
        dots2 = [dot.copy().move_to(paths7[i]['line_graph'].get_end()) for i in range(n)]
        t = 4
        i0 = 0
        print([t_vals[i] for i in iarr])
        for i in range(n+1):
            i1 = iarr[i]
            dt = t * (t_vals[i1] - t_vals[i0])
            #dt = (paths6[i]['line_graph'].get_right()-paths6[i]['line_graph'].get_left()) * t
            self.play(Create(paths6[i], run_rime=dt, rate_func=linear),
                      Create(paths7[i], run_rime=dt, rate_func=linear),
                      run_time=dt)
            anims = [Wait(0.5)]
            if i > 0:
                anims.append(FadeIn(lines[i-1]))
            if i < n:
                anims.append(FadeIn(dots1[i], dots2[i]))
            self.play(*anims, run_time=0.5)
            i0 = i1
        self.wait(0.1)
        eq10 = MathTex(r'2nx').next_to(lines[-1].get_left(), UR, buff=0.1).set_z_index(8)
        self.play(FadeIn(eq10))
        self.wait(0.1)


        arrs1 = []
        arrs3 = []
        kwargs = {'buff': 0.05, 'max_stroke_width_to_length_ratio': 20, 'max_tip_length_to_length_ratio': 0.5}
        for i in range(n+1):
            obj = paths6[i]['line_graph']
            p, q = (obj.get_start(), obj.get_end())
            arrs1.append(Arrow(p, q, color=PURE_RED, stroke_width=12, tip_length=0.35, **kwargs).set_z_index(7))
            obj = paths7[i]['line_graph']
            p, q = (obj.get_start(), obj.get_end())
            arrs3.append(Arrow(p, q, color=PURE_RED, stroke_width=12, tip_length=0.35, **kwargs).set_z_index(7))

        arrs1 = VGroup(arrs1)
        self.play(Create(arrs1), run_time=2, rate_func=linear)
        self.wait(0.1)

        arrs2 = arrs1.copy()
        shift = ax3.coords_to_point(0, level3*4) - ax3.coords_to_point(0, 0)
        for i in range(1,n+1, 2):
            self.play(arrs2[i+1:].animate.shift(shift),
                      mh.rtransform(arrs2[i], arrs3[i]),
                      run_time=1)

        #self.add(path6, path7, *lines)


        self.wait()

class ReflectAssym(Bridge):
    def construct(self):
        scale_neg3 = 0.2
        scale=0.8
        ax, eqt3, xlen3, ylen3, ymax3, mark3, line1_3, _, eq1_3 = self.get_axes(scale=scale, scale_neg=scale_neg3)
        ax.y_axis.set_z_index(20)
        levely = 0.166
        levelx = levely * 2

        line2 = Line(ax.coords_to_point(0, levely), ax.coords_to_point(1, levely), stroke_width=5, stroke_color=RED).set_z_index(2)
        line3 = line2.copy().move_to(ax.coords_to_point(0, -levely), coor_mask=UP)

        eq1 = MathTex(r'x').next_to(line2.get_left(), UR, buff=0.1)[0].set_z_index(11)
        eq2 = MathTex(r'-x').next_to(line3.get_left(), UR, buff=0.1)[0].set_z_index(11)
        eqU = MathTex(r'U', stroke_width=1, color=RED).next_to(line2.get_left(), DR, buff=0.15).set_z_index(10)
        eqU2 = eqU.copy().set_stroke(width=4, color=BLACK).set_z_index(9)
        eqL = MathTex(r'L', stroke_width=1, color=RED).next_to(line3.get_left(), DR, buff=0.15).set_z_index(10)
        eq2_1 = MathTex(r'-y').next_to(line3.get_left(), UR, buff=0.1).shift(DOWN*0.1)[0].set_z_index(11)

        gp = VGroup(eq1, eqU, line2, eqU2)
        gp.generate_target().shift(ax.coords_to_point(0, levelx) - ax.coords_to_point(0, levely))

        npts = 1920
        ndt = npts - 1

        np.random.seed(300)

        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])

        n = 3
        level_end = 2 * levelx + (n-1) * (levelx + levely)
        levels = [levelx + i*(levelx+levely) for i in range(n)] + [level_end]
        levels2 = []
        for i in range(n):
            if i % 2 == 0:
                levels2.append(levelx)
            else:
                levels2.append(-levely)

        attempts = 0
        while True:
            attempts += 1
            b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
            b_vals += t_vals * (level_end - b_vals[-1])
            iarr = []
            i0 = 0
            steps = []
            for i in range(n):
                level4 = levels[i]
                i1 = np.argmax(b_vals[i0+1:] > level4) + i0 + 1
                if abs(b_vals[i1-1] - level4) < abs(b_vals[i1] - level4): i1 -= 1
                b_vals[i1] = level4
                iarr.append(i1)
                dt = t_vals[i1] - t_vals[i0]
                if i > 0: dt /= 2
                steps.append(dt)
                i0 = i1
            steps.append(1-t_vals[iarr[-1]])
            s1 = min(steps)
            s2 = max(steps)
            if s2/s1 > 2:
                continue
            #if min(b_vals[:iarr[0]]) < -level3 * 0.9:
            #    continue
            print(s2/s1)
            break
        print('attempts', attempts)

        b_vals2 = b_vals.copy()
        print(levelx, levely)
        for i in iarr:
            b_vals[i+1:] = 2 * b_vals[i] - b_vals[i+1:]
        iarr.append(ndt)

        self.add(ax, eqt3, mark3, line1_3, eq1_3, line2, line3, eq1, eq2, eqU, eqL, eqU2)
        self.wait(0.1)
        self.play(MoveToTarget(gp), mh.fade_replace(eq2[1], eq2_1[1]), mh.rtransform(eq2[0], eq2_1[0]), run_time=1.6)

        path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(3)
        path2 = ax.plot_line_graph(t_vals, b_vals2, add_vertex_dots=False, stroke_color=BLUE, stroke_width=5).set_z_index(2.5)
        dot = Dot(radius=0.1, color=RED).set_z_index(8)
        paths1 = []
        dots1 = []
        t = 4
        i0 = 0
        for i in range(n+1):
            i1 = iarr[i]
            paths1.append(ax.plot_line_graph(t_vals[i0:i1+1], b_vals[i0:i1+1], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=5).set_z_index(3))
            dt = t_vals[i1] - t_vals[i0]
            anims = [Succession(Wait(0.1), Create(paths1[i], rate_func=linear, run_time=dt * t))]#, rate_func=linear, run_time=dt * t+0.1)]
            if i > 0:
                anims.append(FadeIn(dots1[i-1], run_time=0.4))
            self.play(*anims)
            if i < n:
                dots1.append(dot.copy().move_to(paths1[i]['line_graph'].get_end()))
                #self.play(FadeIn(dots1[i]), run_time=0.5)
            i0 = i1

        self.wait(0.1)

        dash1 = DashedLine(ax.coords_to_point(0, 0), ax.coords_to_point(1, 0), stroke_width=5, stroke_color=RED).set_z_index(2)


        paths2 = [path.copy().set_stroke(color=BLUE).set_z_index(2.9) for path in paths1]
        dots2 = [dot.copy() for dot in dots1]
        b_vals3 = b_vals.copy()
        for i in range(n):
            i1 = iarr[i]
            b_vals3[i1+1:] = b_vals3[i1] * 2 - b_vals3[i1+1:]

            j0 = i1
            anims=[]
            for j in range(i+1, n+1):
                print('i,j:', i, j)
                j1 = iarr[j]
                newpath = ax.plot_line_graph(t_vals[j0:j1+1], b_vals3[j0:j1+1], add_vertex_dots=False,
                                             stroke_color=BLUE, stroke_width=5).set_z_index(3)
                anims.append(mh.transform(paths2[j], newpath))

                if j < n:
                    newdot = dots2[j].copy().move_to(newpath['line_graph'].get_end())
                    anims.append(mh.transform(dots2[j], newdot))

                j0 = j1
            if i < n-1:
                p = paths2[i+1]['line_graph']
                line = dash1.copy().move_to(p.get_end(), coor_mask=UP)
                line.generate_target().move_to(2*p.get_start() - p.get_end(), coor_mask=UP)
                anims.append(MoveToTarget(line))
            self.play(*anims)
            self.wait(0.2)
        line_end = dash1.copy().move_to(ax.coords_to_point(0, level_end), coor_mask=UP)

        eq4 = MathTex(r'n(x+y)+x-y').next_to(line_end.get_left(), UR, buff=0.1)[0].set_z_index(9)
        mh.align_sub(eq4, eq4[0], line_end.get_left(), direction=UR, buff=0.1)
        self.play(FadeIn(line_end, eq4))
        self.wait(0.1)

        box = Rectangle(width=2*config.frame_x_radius, height=2*config.frame_y_radius, stroke_width=0, stroke_opacity=0,
                        fill_color=BLACK, fill_opacity=0.7).set_z_index(30)

        eq5 = MathTex(r'\mathbb P(UL\cdots U)', r'=', r'e^{-\frac12(n(x+y)+x-y)^2}', stroke_width=1, font_size=80).set_z_index(51)
        (eq5[0][2:4] + eq5[0][7]).set_color(RED)
        eq5.next_to(ax.coords_to_point(0.01, (levels[-3]+levels[-2])/2), RIGHT, buff=0)
        eq5_1 = eq5[2].copy().set_z_index(50).set_stroke(color=BLACK, width=12)
        eq4_1 = eq4.copy().set_z_index(50).set_stroke(color=BLACK, width=8)
        eq6 = Tex(r'($n$\sf\ odd)', font_size=80).set_z_index(51).next_to(eq5[2], DOWN, buff=0.5)
        self.play(FadeIn(eq5[:2], eq5[2][:6], eq5[2][16:], eq5_1[:6], eq5_1[16:]),
                  mh.stretch_replace(VGroup(eq4.copy().set_z_index(51), eq4_1), VGroup(eq5[2][6:16], eq5_1[6:16])),
                  FadeIn(box, rate_func=linear),
                  FadeIn(eq6)
                  #VGroup(ax, eq1_3, eqt3, mark3, line1_3, eqL, eqU, lin).animate.set_opacity(0),
                  #VGroup(*paths1, *paths2).animate.set_stroke(opacity=0)
                  )
        self.wait(0.1)

        eq7 = MathTex(r'\mathbb P(UL\cdots L)', r'=', r'e^{-\frac12(n(x+y))^2}', stroke_width=1, font_size=80).set_z_index(51)
        (eq7[0][2:4] + eq7[0][7]).set_color(RED)
        mh.align_sub(eq7, eq7[1], eq5[1])
        mh.align_sub(eq7[0][:-1], eq7[0][0], eq5[0][0])
        eq7_1 = eq7[2].copy().set_z_index(50).set_stroke(color=BLACK, width=12)
        eq8 = Tex(r'($n$\sf\ even)', font_size=80).set_z_index(51)
        mh.align_sub(eq8, eq8[0][1], eq6[0][1])
        self.play(mh.rtransform(eq5[0][:-2], eq7[0][:-2], eq5[0][-1], eq7[0][-1], eq5[1], eq7[1],
                                eq5_1[:12], eq7_1[:12], eq5_1[-2:], eq7_1[-2:],
                                eq5[2][:12], eq7[2][:12], eq5[2][-2:], eq7[2][-2:]),
                  FadeOut(eq5[2][12:-2], eq5_1[12:-2]),
                  mh.fade_replace(eq5[0][-2], eq7[0][-2]),
                  mh.rtransform(eq6[0][:2], eq8[0][:2], eq6[0][-1], eq8[0][-1]),
                  mh.fade_replace(eq6[0][2:-1], eq8[0][2:-1]),
                  )
        self.wait(0.1)
        eq9 = MathTex(r'\mathbb P(\max b_t > x, \min b_t < -y)', r'=', r'\sum_{n=1}^\infty(-1)^{n-1}',
                      r'\left(\mathbb P(LU\cdots)+\mathbb P(UL\cdots)\right)', font_size=80, stroke_width=1).set_z_index(51)
        eq9[2:].scale(7/8)
        VGroup(eq9[0][5:7], eq9[0][13:15]).set_color(BLUE)
        VGroup(eq9[3][3:5], eq9[3][12:14]).set_color(RED)
        eq9[2:].next_to(eq9[0], DOWN, buff=0.2).align_to(eq9[0], LEFT).shift(RIGHT*0.2)
        mh.align_sub(eq9, eq9[0][0], eq7[0][0])
        eq9_1 = eq9.copy().set_z_index(50).set_stroke(color=BLACK, width=12)

        self.play(FadeOut(eq8, eq7, eq7_1), FadeIn(eq9, eq9_1))
        self.wait(0.1)
        self.wait()



class BridgeCalcMax(Scene):
    def eqfinal(self):
        eq21 = MathTex(r'\mathbb P\left(\max\,\lvert b_t\rvert > x\right)', r'=',
                       r'2\sum_{n=1}^\infty(-1)^{n-1} e^{-2n^2x^2}')
        eq21[0][6:8].set_color(YELLOW)
        return eq21

    def construct(self):
        MathTex.set_default(font_size=60)
        eq1 = MathTex(r'\mathbb P\left(\max B_t > x, \lvert B_1\rvert < \epsilon\right)',
                      r'=', r'\mathbb P\left(\lvert B^r_1-2x\rvert < \epsilon\right)')
        eq2 = MathTex(r'\mathbb P\left(\max B_t > x\vert\; \lvert B_1\rvert < \epsilon\right)',
                      r'=', r'\frac{\mathbb P\left(\lvert B^r_1-2x\rvert < \epsilon\right)}'
                      r'{\mathbb P\left(\lvert B_1\rvert < \epsilon\right)}')
        eq3 = MathTex(r'\mathbb P\left(\max b_t > x\right)',
                      r'=', r'\frac{p(2x)}{p(0)}')
        eq4 = MathTex(r'=', r'e^{-\frac12(2x)^2')
        eq5 = MathTex(r'\mathbb P\left(\max b_t > x\right)', r'=', r'e^{-2x^2}')

        VGroup(eq1[0][5:7], eq1[0][11:13], eq2[0][5:7], eq2[0][11:13], eq2[2][-6:-4]).set_color(YELLOW)
        VGroup(eq1[2][3:6], eq2[2][3:6]).set_color(BLUE)
        VGroup(eq3[0][5:7], eq5[0][5:7]).set_color(YELLOW)

        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)
        VGroup(eq1, eq2).to_edge(DOWN, buff=0.1)
        mh.align_sub(eq3, eq3[1], eq2[0], coor_mask=UP)
        mh.align_sub(eq4, eq4[0], eq3[1])
        mh.align_sub(eq5, eq5[1], eq4[0])

        self.wait(0.1)
        self.play(FadeIn(eq1[0]))
        self.wait(0.1)
        self.play(FadeIn(eq1[1:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][:9], eq2[0][:9], eq1[0][10:], eq2[0][10:], eq1[1], eq2[1],
                                eq1[2][:], eq2[2][:13], eq1[0][:2].copy(), eq2[2][14:16],
                                eq1[0][-7:].copy(), eq2[2][-7:]),
                  FadeIn(eq2[2][13], rate_func=rush_into),
                  mh.fade_replace(eq1[0][9], eq2[0][9], coor_mask=RIGHT),
                  run_time=2)
        self.wait(0.1)
        eq3_1 = eq3[0][5:7].copy().move_to(eq2[0][5:7], coor_mask=RIGHT)
        eq3_2 = eq3[2][:5].copy().move_to(eq2[2], coor_mask=RIGHT)
        eq3_3 = eq3[2][-4:].copy().move_to(eq2[2], coor_mask=RIGHT)
        self.play(FadeOut(eq2[0][-8:-1]),
                  mh.fade_replace(eq2[0][5], eq3_1[0]),
                  mh.rtransform(eq2[0][6], eq3_1[1]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(FadeOut(eq2[2][:7], eq2[2][9:13], eq2[2][-9:]),
                  FadeIn(eq3_2[:2], eq3_2[-1], eq3_3),
                  mh.rtransform(eq2[2][7:9], eq3_2[2:-1]),
                  run_time=1.5)
        self.play(mh.rtransform(eq2[0][:5], eq3[0][:5], eq3_1, eq3[0][5:7],
                                eq2[0][7:9], eq3[0][7:9], eq2[0][-1], eq3[0][-1],
                                eq2[1], eq3[1], eq3_2, eq3[2][:5], eq2[2][13], eq3[2][5],
                                eq3_3, eq3[2][-4:]))
        self.wait(0.1)
        self.play(mh.stretch_replace(eq3[2][1:5], eq4[1][5:9]),
                  FadeOut(eq3[2][0], eq3[2][5:]),
                  FadeIn(eq4[1][:5], eq4[1][9:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq5[:2], eq4[1][:2], eq5[2][:2],
                                eq4[1][4], eq5[2][2], eq4[1][7], eq5[2][3],
                                eq4[1][9], eq5[2][4]),
                  mh.rtransform(eq4[1][6], eq5[2][2]),
                  FadeOut(eq4[1][2:4], eq4[1][5], eq4[1][8]))
        self.wait(0.1)
        whitef = ManimColor(WHITE.to_rgb()*0.7)
        yellowf = ManimColor(YELLOW.to_rgb()*0.7)
        eq5.generate_target().to_edge(UP, buff=0.1)
        VGroup(eq5.target[0][:5], eq5.target[0][7:], eq5.target[1:]).set_color(whitef)
        eq5.target[0][5:7].set_color(yellowf)
        #self.play(eq5.animate.to_edge(UP, buff=0.1).set_opacity(0.7), run_time=1.4)
        self.play(MoveToTarget(eq5), run_time=1.4)
        self.wait(0.1)

        # do absolute maximum

        eq6 = MathTex(r'\mathbb P\left(\max\,\lvert b_t\rvert > x\right)', r'=', r'\mathbb P(L{\sf\ or\ }U)')
        eq7 = MathTex(r'\mathbb P(L)', r'=', r'\mathbb P(U)', r'=', r'\mathbb P\left(\max b_t > x\right)', r'=', r'e^{-2x^2}')
        eq8 = MathTex(r'\mathbb P(L)', r'=', r'\mathbb P(U)', r'=', r'e^{-2x^2}')
        eq9 = MathTex(r'\mathbb P\left(\max\,\lvert b_t\rvert > x\right)', r'=', r'\mathbb P(L)+\mathbb P(U)',
                      r'-', r'\mathbb P(L{\sf\ and\ }U)').set_z_index(2)

        eq10 = MathTex(r'\mathbb P(U{\sf\ then\ }L)', r'=', r'p(4x)/p(0)')
        eq11 = MathTex(r'=', r'e^{-\frac12(4x)^2}')
        eq12 = MathTex(r'=', r'e^{-8x^2}')

        eq13 = MathTex(r'\mathbb P(U{\sf\ then\ }L{\sf\ then}\ldots)', r'=', r'p(2nx)/p(0)').set_z_index(2)
        eq14 = MathTex(r'=', r'e^{-\frac12(2nx)^2}').set_z_index(2)
        eq15 = MathTex(r'=', r'e^{-2n^2x^2}').set_z_index(2)

        eq16 = MathTex(r'\mathbb P(ULUL\cdots)', r'=', r'e^{-2n^2x^2}').set_z_index(2).set_z_index(2)

        VGroup(eq6[0][6:8], eq7[4][5:7], eq9[0][6:8]).set_color(YELLOW)
        VGroup(eq7[0][2], eq7[2][2], eq8[0][2], eq8[2][2], eq6[2][2], eq6[2][5],
               eq9[2][2], eq9[2][7], eq9[4][2], eq9[4][6], eq10[0][2], eq10[0][-2],
               eq13[0][2], eq13[0][7], eq16[0][2:6]).set_color(RED)
        eq6.next_to(eq5, DOWN, buff=0.2)
        mh.align_sub(eq7, eq7[3], eq5[1], coor_mask=UP)
        mh.align_sub(eq8, eq8[3], eq5[1], coor_mask=UP)
        mh.align_sub(eq9, eq9[1], eq6[1], coor_mask=UP)
        eq10.next_to(eq9, DOWN)
        mh.align_sub(eq11, eq11[0], eq10[1])
        mh.align_sub(eq12, eq12[0], eq10[1])
        mh.align_sub(eq13, eq13[1], eq10[1], coor_mask=UP)
        mh.align_sub(eq14, eq14[0], eq13[1])
        mh.align_sub(eq15, eq15[0], eq13[1])
        mh.align_sub(eq16, eq16[0], eq13[1], coor_mask=UP)

        gp = VGroup(eq10, eq11, eq12).set_z_index(2)
        box1 = SurroundingRectangle(gp, fill_color=BLACK, fill_opacity=0.6, stroke_width=0, stroke_opacity=0,
                                    corner_radius=0.15, buff=0.2)

        eq5_1 = eq5[0].copy()
        eq6_1 = eq6[0].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq5_1[:5], eq6_1[:5], eq5_1[5:7], eq6_1[6:8],
                                eq5_1[7:], eq6_1[9:]))
        self.play(FadeIn(eq6_1[5], eq6_1[8]))
        self.wait(0.1)
        eq7_1 = eq7[2:].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq5[:], eq7_1[2:]),
                  FadeIn(eq7_1[:2]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq7_1, eq7[2:]),
                  FadeIn(eq7[:2]))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:4], eq8[:4], eq7[-1], eq8[-1]),
                  mh.rtransform(eq7[-2], eq8[3]),
                  FadeOut(eq7[-3]))
        self.wait(0.1)
        self.play(mh.rtransform(eq6_1, eq6[0]),
                  FadeIn(eq6[1:]),
                  run_time=1.2)
        self.wait(0.1)
        eq9_1 = eq9[:3].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq6[:2], eq9_1[:2], eq6[2][:3], eq9_1[2][:3],
                                eq6[2][-2:], eq9_1[2][-2:], eq6[2][:2].copy(), eq9_1[2][-4:-2]),
                  FadeOut(eq6[2][3:-2]),
                  FadeIn(eq9_1[2][3], shift=mh.diff(eq6[2][:3], eq9_1[2][:3])),
                  FadeIn(eq9_1[2][4]), run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq9_1, eq9[:3]), FadeIn(eq9[3:]), run_time=1.4)
        self.wait(0.1)

        top_buff=0.6
        #self.play(VGroup(eq8, eq9).animate.set_opacity(0.7))
        eq9.generate_target().to_edge(UP, buff=top_buff).set_opacity(0.7)
        self.play(MoveToTarget(eq9), FadeOut(eq8))


        self.play(FadeIn(box1, eq10[0]))
        self.wait(0.1)
        self.play(FadeIn(eq10[1:]))
        self.wait(0.1)
        self.play(mh.stretch_replace(eq10[2][1:5], eq11[1][5:9]),
                  FadeOut(eq10[2][0], eq10[2][5:]),
                  FadeIn(eq11[1][:5], eq11[1][-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[1][:2], eq12[1][:2], eq11[1][7], eq12[1][3], eq11[1][9], eq12[1][4]),
                  FadeOut(eq11[1][2:6], eq11[1][8]),
                  mh.fade_replace(eq11[1][6], eq12[1][2]))
        self.wait(0.1)

        self.remove(box1)
        self.play(mh.rtransform(eq10[0][:8], eq13[0][:8], eq10[0][-1], eq13[0][-1],
                                eq10[1], eq13[1]),# box1, box2),
                  FadeIn(eq13[0][8:-1]),
                  mh.fade_replace(eq12[1], eq13[2]))
        self.wait(0.1)
        self.play(mh.stretch_replace(eq13[2][1:6], eq14[1][5:10]),
                  FadeOut(eq13[2][0], eq13[2][6:]),
                  FadeIn(eq14[1][:5], eq14[1][-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq14[1][:2], eq15[1][:2], eq14[1][4], eq15[1][2],
                                eq14[1][10], eq15[1][6], eq14[1][10].copy(), eq15[1][4]),
                  mh.rtransform(eq14[1][6:8], eq15[1][2:4], eq14[1][8], eq15[1][5]),
                  FadeOut(eq14[1][2:4], eq14[1][5], eq14[1][9]))
        self.wait(0.1)

        self.play(mh.rtransform(eq13[0][:3], eq16[0][:3], eq13[0][7], eq16[0][3],
                                eq13[0][-4:], eq16[0][-4:], eq13[1], eq16[1], eq15[1], eq16[2]),
                  FadeOut(eq13[0][3:7], eq13[0][8:12]),
                  FadeIn(eq16[0][4:6]))

        br1 = BraceLabel(eq16[0][2:-1], r'n \sf\ terms', label_constructor=mh.mathlabel_ctr, font_size=60,
                         brace_config={'color': WHITE}).set_z_index(2)
        self.play(FadeIn(br1))
        self.wait(0.1)
        gp = VGroup(eq16, br1)
        #gp.generate_target().next_to(eq8, DOWN, buff=0.2)
        gp.generate_target().to_edge(UP, top_buff)
        gp.target[0].set_opacity(0.7)
        gp.target[1].set_opacity(0)
        eq9.generate_target().set_opacity(1).next_to(gp.target[0], DOWN, buff=0.3)
        self.play(MoveToTarget(gp), MoveToTarget(eq9), run_time=1.4)
        self.wait(0.1)

        # expand series
        eq16 = MathTex(r'\mathbb P\left(\max\,\lvert b_t\rvert > x\right)', r'=', r'\mathbb P(L)+\mathbb P(U)',
                      r'-', r'\mathbb P(LU{\sf\ or\ }UL)').set_z_index(2)
        eq16[3:].next_to(eq16[2], DOWN, buff=0.1).align_to(eq16[2], LEFT)
        eq17 = MathTex(r'{}-', r'\mathbb P(LU)-\mathbb P(UL)', r'+', r'\mathbb P(LUL{\sf\ or\ }ULU)').set_z_index(2)
        eq17[2:].next_to(eq17[1], DOWN, buff=0.1).align_to(eq17[0], LEFT)
        eq18 = MathTex(r'{}+', r'\mathbb P(LUL)+\mathbb P(ULU)', r'-', r'\mathbb P(LULU{\sf\ or\ }ULUL)')
        eq18[2:].next_to(eq18[1], DOWN).align_to(eq18[0], LEFT)

        eq19 = MathTex(r'{}=', r'\sum_{n=1}^\infty(-1)^{n-1}\left(\mathbb P(LU\cdots)+\mathbb P(UL\cdots)\right)')
        eq20 = MathTex(r'{}=', r'2\sum_{n=1}^\infty(-1)^{n-1} e^{-2n^2x^2}')

        VGroup(eq16[2][2], eq16[2][7], eq16[4][2:4], eq16[4][6:8],
               eq17[1][2:4], eq17[1][8:10], eq17[3][2:5], eq17[3][7:10],
               eq18[1][2:5], eq18[1][9:12], eq18[3][2:6], eq18[3][8:12],
               eq19[1][15:17], eq19[1][24:26]).set_color(RED)
        VGroup(eq16[0][6:8]).set_color(YELLOW)

        mh.align_sub(eq16, eq16[1], eq9[1]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq17, eq17[1], eq16[1]).move_to(ORIGIN, coor_mask=RIGHT)
        eq17.next_to(eq16[2], DOWN, buff=0.1).align_to(eq16[3], LEFT)
        eq18.next_to(eq17[1], DOWN, buff=0.1).align_to(eq17[0], LEFT)
        eq19.next_to(eq18, DOWN, buff=0.2).align_to(eq18[-1], RIGHT).shift(RIGHT*0.4)
        VGroup(eq16, eq17, eq18, eq19).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq20, eq20[0], eq19[0], coor_mask=UP)

        self.play(mh.rtransform(eq9[:4], eq16[:4], eq9[4][:3], eq16[4][:3], eq9[4][6], eq16[4][3],
                                eq9[4][2].copy(), eq16[4][7], eq9[4][6].copy(), eq16[4][6],
                                eq9[4][7], eq16[4][8]),
                  mh.fade_replace(eq9[4][3:6], eq16[4][4:7]),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(eq16[3], eq17[0], eq16[4][:4], eq17[1][:4], eq16[4][-3:], eq17[1][-3:]),
                  mh.fade_replace(eq16[4][4:-3], eq17[1][5]),
                  FadeIn(eq17[1][4], shift=mh.diff(eq16[4][:4], eq17[1][:4])),
                  FadeIn(eq17[1][6:8], shift=mh.diff(eq16[4][-3:], eq17[1][-3:])),
                  run_time=1.8)
        self.wait(0.1)
        self.play(FadeIn(eq17[2:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq17[2], eq18[0], eq17[3][:5], eq18[1][:5],
                                eq17[3][-4:], eq18[1][-4:]),
                  FadeIn(eq18[1][5], shift=mh.diff(eq17[3][:5], eq18[1][:5])),
                  mh.fade_replace(eq17[3][5:7], eq18[1][6], coor_mask=RIGHT),
                  FadeIn(eq18[1][7:9], shift=mh.diff(eq17[3][-4:], eq18[1][-4:])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(FadeIn(eq18[2:]))
        self.wait(0.1)
        self.play(FadeIn(eq19), run_time=1.5)
        self.wait(0.1)
        br2 = BraceLabel(eq19[1][15:20], r'n \sf\ terms', label_constructor=mh.mathlabel_ctr, font_size=60,
                         brace_config={'color': WHITE}).set_z_index(2)
        br3 = BraceLabel(eq19[1][24:29], r'n \sf\ terms', label_constructor=mh.mathlabel_ctr, font_size=60,
                         brace_config={'color': WHITE}).set_z_index(2)
        self.play(FadeIn(br2, br3))
        self.wait(0.1)
        gp1 = VGroup(eq19[1][13:30], br2, br3).copy()

        self.play(gp[0][2].animate.set_opacity(1).scale(1.4))
        self.play(gp[0][2].animate.scale(1/1.4))

        eq19_1 = gp[0][2].copy().set_z_index(20)
        eq19_2 = gp[0][2].copy().set_z_index(20)
        eq19_1.generate_target().move_to(eq19[1][13:21])
        eq19_2.generate_target().move_to(eq19[1][22:30])
        self.wait(0.1)
        self.add(eq19_1)
        gp[0][2].set_opacity(0.7)
        eq19_3 = MathTex(r'=e')[0]
        mh.align_sub(eq19_3, eq19_3[0], eq19[0][0])
        mh.align_sub(eq19_1.target, eq19_1.target[0], eq19_3[0], coor_mask=UP)
        mh.align_sub(eq19_2.target, eq19_2.target[0], eq19_3[0], coor_mask=UP)
        self.play(AnimationGroup(MoveToTarget(eq19_1), MoveToTarget(eq19_2), run_time=3),
                  Succession(Wait(2), FadeOut(eq19[1][13:21], eq19[1][22:30], br2, br3)))
        self.wait(0.1)
        self.play(mh.rtransform(eq19[0], eq20[0], eq19[1][:12], eq20[1][1:13],
                                eq19_1[:], eq20[1][13:]),
                  mh.rtransform(eq19_2[:], eq20[1][13:]),
                  FadeIn(eq20[1][0], shift=mh.diff(eq19[1][0], eq20[1][1])*RIGHT),
                  FadeOut(eq19[1][12], shift=mh.diff(eq19[1][0], eq20[1][1])*RIGHT),
                  FadeOut(eq19[1][21], target_position=eq20[1][13]),
                  FadeOut(eq19[1][30], shift=mh.diff(eq19_2, eq20[1][13:])),
                  run_time=2)
        self.wait(0.1)

        eq21 = self.eqfinal()
        self.play(mh.rtransform(eq16[:2], eq21[:2], eq20[1], eq21[2]),
                  mh.rtransform(eq20[0], eq21[1]),
                  FadeOut(gp[0], eq16[2], eq17[:2], eq18))  # maybe fade eq8 out
        self.wait(0.1)
        circ = mh.circle_eq(eq21[2][-7:-2]).shift(RIGHT*0.55+DOWN*0.14)
        gp1.next_to(eq21, DOWN, buff=0.4).to_edge(RIGHT, buff=0.2)

        self.play(FadeIn(gp1, run_time=1.5), Create(circ, rate_func=linear, run_time=0.6))

        self.wait()


class MomentsCalc(BridgeCalcMax):
    def construct(self):
        MathTex.set_default(font_size=60)
        eq1 = self.eqfinal()
        self.add(eq1)
        self.wait(0.1)

        #eq1.generate_target().to_edge(UP)

        eq2 = MathTex(r'\mathbb P(X > x)', r'=', r'2\sum_{n=1}^\infty(-1)^{n-1}e^{-2n^2x^2}')
        eq3 = MathTex(r'\mathbb E\left[f(X)\right]', r'=', r'\int_0^\infty\mathbb P(X > x) f^\prime(x)\,dx', r'+f(0)')
        eq4 = MathTex(r'f(x)=x^s', r'=', r'f^\prime(x)=sx^{s-1}')
        eq5 = MathTex(r'\mathbb E[X^s]', r'=', r'\int_0^\infty\mathbb P(X > x)sx^{s-1}\,dx')
        eq6 = MathTex(r'\mathbb E[X^s]', r'=', r'\int_0^\infty2\sum_{n=1}^\infty(-1)^{n-1}e^{-2n^2x^2}sx^s\,\frac{dx}{x}')
        eq7 = MathTex(r'\mathbb E[X^s]', r'=', r'2s\sum_{n=1}^\infty(-1)^{n-1}\int_0^\infty e^{-2n^2x^2}x^s\,\frac{dx}{x}')

        eq8 = MathTex(r'y=', r'2n^2x^2')

        eq9 = MathTex(r'\mathbb E[X^s]', r'=', r'2s\sum_{n=1}^\infty(-1)^{n-1}\int_0^\infty e^{-y}'
                      r'2^{-\frac s2}n^{-s}y^{\frac s2}\frac{dy}{y}')

        eq10 = MathTex(r'x^2=', r'2^{-1}n^{-2}y')
        eq11 = MathTex(r'x=', r'2^{-\frac12}n^{-1}y^{\frac12}')
        eq12 = MathTex(r'\log x=', r'{}^{\frac12}\log y+\,{\sf const}')
        eq12[1][:3].move_to(eq12[0][-1], coor_mask=UP)
        eq13 = MathTex(r'\frac{dx}{x}=', r'{}^{\frac12}\frac{dy}{y}')
        eq13[1][:3].move_to(eq13[0][-1], coor_mask=UP)
        r1 = eq13[1][:3].get_right()
        eq13[1][:3].scale(1.4, about_edge=LEFT)
        eq13[1][3:].shift((eq13[1][:3].get_right()-r1)*RIGHT)

        eq14 = MathTex(r'\mathbb E[X^s]', r'=', r'2^{-\frac s2}s\sum_{n=1}^\infty(-1)^{n-1}n^{-s}\int_0^\infty e^{-y}'
                      r'y^{\frac s2}\frac{dy}{y}')
        eq15 = MathTex(r'=', r'\Gamma({}^{\frac s2})')
        eq15[1][2:5].move_to(eq15[0], coor_mask=UP)

        eq16 = MathTex(r'1^{-s}-2^{-s}+3^{-s}-4^{-s}+5^{-s}-6^{-s}+\cdots')
        #eq16[0].scale(1.4)
        #eq16[1].next_to(eq16[0], DOWN, buff=0.5).move_to(ORIGIN, coor_mask=RIGHT)
        #eq16[0].align_to(eq16[1], LEFT).shift(LEFT*0.4)

        VGroup(eq3[0][4], eq3[2][5], eq2[0][2], eq5[0][2], eq5[2][5], eq6[0][2],
               eq7[0][2], eq9[0][2], eq14[0][2]).set_color(YELLOW)
        VGroup(eq3[2][7], eq3[2][12], eq3[2][-1],
               eq2[0][-2], eq2[2][-2], eq4[0][2], eq4[0][-2], eq4[2][3],
               eq4[2][-4], eq5[2][7], eq5[2][10], eq5[2][-1], eq6[2][-9], eq6[2][-3],
               eq6[2][-1], eq6[2][-6], eq7[2][-1], eq7[2][-3], eq7[2][-6], eq7[2][-8],
               eq8[1][3], eq10[0][0], eq11[0][0], eq12[0][3], eq13[0][1], eq13[0][-2]).set_color(BLUE).set_stroke(width=1)
        VGroup(eq4[0][-1], eq4[2][-5], eq4[2][-3], eq5[0][-2], eq5[2][-5], eq5[2][-7],
               eq6[0][-2], eq6[2][-5], eq6[2][-7], eq7[0][-2], eq7[2][1],
               eq9[0][-2], eq9[2][1], eq9[2][-7], eq9[2][-9], eq9[2][-14],
               eq14[0][-2], eq14[2][2], eq14[2][5], eq14[2][20], eq14[2][28],
               eq15[1][2], eq16[0][2], eq16[0][6], eq16[0][10],
               eq16[0][14], eq16[0][18], eq16[0][22]).set_color(RED)
        VGroup(eq8[0][0], eq9[2][19], eq10[1][-1], eq11[1][-4], eq12[1][6],
               eq13[1][4], eq13[1][-1], eq9[2][-1], eq9[2][-3], eq9[2][-8],
               eq14[2][26:28], eq14[2][-3], eq14[2][-1]).set_color(GREEN).set_stroke(width=1)

        eq2.to_edge(UP)
        eq3.next_to(eq2, DOWN)
        eq4[2].shift(RIGHT)
        eq4.next_to(eq3, DOWN, buff=1).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[1], eq3[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[1], eq3[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[1], eq3[1], coor_mask=UP)
        eq8.next_to(eq7, DOWN, buff=0.6)
        mh.align_sub(eq9, eq9[1], eq7[1])
        mh.align_sub(eq10, eq10[0][-1], eq8[0][-1])
        mh.align_sub(eq11, eq11[0][1], eq8[0][1])
        mh.align_sub(eq12, eq12[0][-1], eq8[0][-1])
        mh.align_sub(eq13, eq13[0][-1], eq8[0][-1])
        mh.align_sub(eq14, eq14[1], eq7[1], coor_mask=UP)
        mh.align_sub(eq15, eq15[0], eq7[0], coor_mask=UP)

        self.play(mh.rtransform(eq1[1:], eq2[1:], eq1[0][:2], eq2[0][:2], eq1[0][-3:], eq2[0][-3:]),
                  mh.fade_replace(eq1[0][2:-3], eq2[0][2]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq3), run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq4[0]))
        self.wait(0.1)
        self.play(FadeIn(eq4[2]))
        self.wait(0.1)
        eq5_1 = eq5[0][-3].copy()
        eq5_2 = eq5[2][10].copy()
        self.play(mh.rtransform(eq3[0][:2], eq5[0][:2], eq3[0][-1], eq5[0][-1], eq3[1], eq5[1],
                                eq3[2][:9], eq5[2][:9], eq3[0][4], eq5[0][2], eq4[0][-1], eq5[0][-2],
                                eq3[2][12], eq5_2, eq4[2][-5:], eq5[2][9:14], eq3[2][-2:], eq5[2][-2:]),
                  FadeOut(eq3[0][2:4], eq3[0][5], shift=mh.diff(eq3[0][4], eq5[0][2])),
                  FadeOut(eq3[2][9:12], eq3[2][13], shift=mh.diff(eq3[2][12], eq5_2)),
                  mh.stretch_replace(eq4[0][-2], eq5_1),
                  FadeOut(eq3[3], eq4[0][:5], eq4[2][:6]),
                  run_time=1.6)
        self.remove(eq5_1, eq5_2)
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:2], eq6[:2], eq5[2][:3], eq6[2][:3],
                                eq5[2][-2:], eq6[2][-4:-2],
                                eq5[2][-7:-4], eq6[2][-7:-4],
                                eq5[2][-6].copy(), eq6[2][-1],
                                eq2[2][:], eq6[2][3:-7]),
                  FadeOut(eq5[2][3:-7], eq2[:2]),
                  FadeOut(eq5[2][-4:-2], shift=mh.diff(eq5[2][-7:-4], eq6[2][-7:-4])),
                  FadeIn(eq6[2][-2]),
                  run_time=1.8)
        self.wait(0.1)
        VGroup(eq7[2][1], eq6[2][-7]).set_z_index(1)
        self.play(mh.rtransform(eq6[:2], eq7[:2], eq6[2][:3], eq7[2][14:17],
                                eq6[2][3], eq7[2][0], eq6[2][4:16], eq7[2][2:14],
                                eq6[2][-7], eq7[2][1], eq6[2][16:-7], eq7[2][17:-6],
                                eq6[2][-6:], eq7[2][-6:]),
                  run_time=1.8)
        self.wait(0.1)
        self.play(FadeIn(eq8))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:2], eq9[:2], eq7[2][:19], eq9[2][:19],
                                eq8[0][0].copy(), eq9[2][19]),
                  FadeOut(eq7[2][19:24], shift=mh.diff(eq7[2][18], eq9[2][18])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(eq8[0][-1], eq10[0][-1], eq8[0][0], eq10[1][-1],
                                eq8[1][-2:], eq10[0][:2], eq8[1][0], eq10[1][0],
                                eq8[1][1], eq10[1][3], eq8[1][2], eq10[1][5]),
                  FadeIn(eq10[1][1:3]),
                  FadeIn(eq10[1][4], shift=mh.diff(eq8[1][2], eq10[1][5])),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq10[0][-1], eq11[0][-1], eq10[0][0], eq11[0][0],
                                eq10[1][:3], eq11[1][:3], eq10[1][3:5], eq11[1][5:7],
                                eq10[1][6], eq11[1][8], eq10[0][1], eq11[1][4],
                                eq10[0][1].copy(), eq11[1][-1]),
                  mh.fade_replace(eq10[1][5], eq11[1][7]),
                  FadeIn(eq11[1][3]),
                  FadeIn(eq11[1][-3:-1]),
                  run_time=1.5)
        self.wait(0.1)
        eq11_1 = eq11[1].copy()
        shift = eq9.get_center() * LEFT
        eq9[2][20:].shift(shift)
        self.play(mh.rtransform(eq11_1[:2], eq9[2][20:22], eq11_1[3:5], eq9[2][23:25],
                                eq7[2][25], eq9[2][22],
                                eq11_1[5:7], eq9[2][25:27], eq7[2][25].copy(), eq9[2][27],
                                eq11_1[8], eq9[2][28], eq11_1[10:12], eq9[2][30:32],
                                eq7[2][25].copy(), eq9[2][29]),
                  FadeOut(eq11_1[2], target_position=eq9[2][22]),
                  FadeOut(eq11_1[7], target_position=eq9[2][27]),
                  FadeOut(eq11_1[9], target_position=eq9[2][29]),
                  FadeOut(eq7[2][24], shift=mh.diff(eq7[2][24], eq9[2][28])*RIGHT),
                  (eq9[:2]+eq9[2][:20]).animate.shift(shift),
                  eq7[2][-4:].animate.shift(-shift),
                  run_time=2)
        self.wait(0.1)
        self.play(mh.rtransform(eq11[0][-2:], eq12[0][-2:], eq11[1][-3:], eq12[1][:3],
                                eq11[1][-4], eq12[1][6]),
                  FadeIn(eq12[0][:-2]),
                  FadeOut(eq11[1][:8]),
                  FadeIn(eq12[1][3:6], eq12[1][7:]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq12[0][-1], eq13[0][-1], eq12[0][3], eq13[0][1],
                                eq12[0][3].copy(), eq13[0][3], eq12[1][:3], eq13[1][:3],
                                eq12[1][6], eq13[1][4], eq12[1][6].copy(), eq13[1][6]),
                  FadeOut(eq12[0][:3], eq12[1][3:6]),
                  FadeIn(eq13[0][0], shift=mh.diff(eq12[0][3], eq13[0][1])),
                  FadeIn(eq13[1][3], shift=mh.diff(eq12[1][6], eq13[1][4])),
                  FadeOut(eq12[1][7:]),
                  FadeIn(eq13[0][2], eq13[1][5]),
                  run_time=1.4)
        self.wait(0.1)
        eq13_1 = eq13[1]
        self.play(mh.rtransform(eq13_1[-4:], eq9[2][-4:]),
                  mh.rtransform(eq7[2][-2], eq9[2][-2], eq7[2][-4], eq9[2][-4]),
                  FadeOut(eq13_1[:3], target_position=eq9[2][0]),
                  FadeOut(eq9[2][0]),
                  FadeOut(eq7[2][-3], shift=mh.diff(eq7[2][-3], eq9[2][-3])*RIGHT),
                  FadeOut(eq7[2][-1], shift=mh.diff(eq7[2][-1], eq9[2][-1])*RIGHT),
                  FadeOut(eq13[0]),
                  run_time=1.8)
        self.wait(0.1)
        self.play(mh.rtransform(eq9[:2], eq14[:2], eq9[2][20:25], eq14[2][:5],
                                eq9[2][1:14], eq14[2][5:18], eq9[2][25:28], eq14[2][18:21],
                                eq9[2][14:20], eq14[2][21:27], eq9[2][-8:], eq14[2][-8:]),
                  run_time=1.8)
        self.wait(0.1)
        circ = mh.circle_eq(eq14[2][-14:]).shift(RIGHT*0.2)
        self.play(Create(circ), rate_func=linear, run_time=1)
        self.wait(0.1)
        eq15[1].move_to(eq14[2][-14:], coor_mask=RIGHT)
        self.play(FadeOut(eq14[2][-14:-7], eq14[2][-4:]),
                  mh.rtransform(eq14[2][-7:-4], eq15[1][2:5]),
                  FadeIn(eq15[1][:2], eq15[1][5:]))
        self.wait(0.1)
        self.play(FadeOut(circ), eq14[2][6:21].animate.move_to((eq14[2][5].get_right()+eq15[1].get_left())*0.5),
                  run_time=1)
        circ = mh.circle_eq(eq14[2][6:21])
        self.play(Create(circ), run_time=1, rate_func=linear)
        self.wait(0.1)

        eq16.next_to(circ, DOWN, coor_mask=UP, buff=1)
        self.play(FadeIn(eq16))
        self.wait(0.1)
        eq16_1 = eq16[0][3:27:8]
        eq16_2 = MathTex(r'1+1+1+1')[0][1::2]
        eq16_3 = MathTex(r'=\eta(s)')[0]
        eq16_3 = mh.align_sub(eq16_3[1:], eq16_3[0], eq14[1]).move_to(eq14[2][6:21]).scale(1.4)
        eq16_3[2].set_color(RED)
        #eq16_3 = mh.align_sub(eq16[0][:-1].copy(), eq16[0][-1], eq14[1]).move_to(eq14[2][6:21])
        for x, y in zip(eq16_1, eq16_2): y.move_to(x)
        self.play(FadeOut(eq16_1), FadeIn(eq16_2), rate_func=linear)
        self.wait(0.1)
        self.play(FadeOut(eq16_2), FadeIn(eq16_1), rate_func=linear)
        self.wait(0.1)
        self.play(FadeIn(eq16_3), FadeOut(eq14[2][6:21]), rate_func=linear)
        self.wait(0.1)
        self.play(FadeOut(eq16_1), FadeIn(eq16_2), rate_func=linear)
        self.wait(0.1)

        # take off even terms
        eq19 = MathTex(r'{}-2(', r'2^{-s}1^{-s}', r'+2^{-s}2^{-s}', r'+2^{-s}3^{-s}', r'+\cdots)')
        #mh.align_sub(eq19, eq19[0][0], eq17[0][0], coor_mask=RIGHT)
        eq19.next_to(eq16, DOWN, buff=0.6)
        mh.align_sub(eq19[:2], eq19[1][-1], eq16[0][6], coor_mask=RIGHT)
        mh.align_sub(eq19[2], eq19[2][-1], eq16[0][14], coor_mask=RIGHT)
        mh.align_sub(eq19[3], eq19[3][-1], eq16[0][22], coor_mask=RIGHT)
        mh.align_sub(eq19[4], eq19[4][0], eq16[0][-4], coor_mask=RIGHT)

        eq17 = MathTex(r'{}-2\,.\,2^{-s}', r'-2\,.\,4^{-s}', r'-2\,.\,6^{-s}', r'\,-\cdots')
        mh.align_sub(eq17, eq17[-1][-3:], eq19[-1][-4:-1])
        for i in range(3):
            mh.align_sub(eq17[i], eq17[i][-1], eq19[i+1][-1], coor_mask=RIGHT)
            eq17[i][-1].set_color(RED)

        eq18 = MathTex(r'{}-2(', r'2^{-s}', r'+4^{-s}', r'+6^{-s}', r'\,+\cdots)')
        mh.align_sub(eq18, eq18[0], eq19[0])
        for i in range(1, 4):
            mh.align_sub(eq18[i], eq18[i][-1], eq19[i][-1], coor_mask=RIGHT)
            eq18[i][-1].set_color(RED)
        for i in range(2, 4):
            eq18[i][0].move_to(eq19[i][0], coor_mask=RIGHT)
        mh.align_sub(eq18[-1], eq18[-1][-1], eq19[-1][-1])

        eq20 = MathTex(r'{}-2^{1-s}(', r'1^{-s}', r'+2^{-s}', r'+3^{-s}', r'+\cdots)')
        mh.align_sub(eq20, eq20[0][-1], eq19[0][-1])
        eq20[0][-2].set_color(RED)
        for i in range(1, 4):
            mh.align_sub(eq20[i], eq20[i][-1], eq19[i][-1], coor_mask=RIGHT)
            #eq20[i][0].move_to(eq19[i][0], coor_mask=RIGHT)
            eq20[i][-1].set_color(RED)
        for i in range(2, 4):
            eq20[i][0].move_to(eq19[i][0], coor_mask=RIGHT)
        eq20[-1].move_to(eq19[-1], coor_mask=RIGHT)

        eq16_4 = eq16[0].copy()
        anims = []
        for i in range(3):
            anims += [
                mh.rtransform(eq16_4[3+8*i], eq17[i][0], eq16_4[4+8*i:7+8*i], eq17[i][-3:]),
                FadeIn(eq17[i][1:3], target_position=eq16_4[4+8*i])
            ]
        self.play(mh.fade_replace(eq16_4[-4], eq17[-1][0]),
                  mh.rtransform(eq16_4[-3:], eq17[-1][-3:]),
                  *anims, run_time=1.4)
        self.wait(0.1)

        anims = []
        for i in range(3):
            anims += [
                mh.rtransform(eq17[i][:2], eq18[0][:2]),
                FadeOut(eq17[i][2], shift=mh.diff(eq17[i][1], eq18[0][0])),
                mh.rtransform(eq17[i][-3:], eq18[i+1][-3:])
            ]
        for i in range(1, 3):
            anims += [
                mh.fade_replace(eq17[i][0], eq18[i+1][0])
            ]
        self.play(mh.fade_replace(eq17[-1][0], eq18[-1][0]),
                  mh.rtransform(eq17[-1][1:4], eq18[-1][1:4]),
                  FadeIn(eq18[0][-1], eq18[-1][-1], rate_func=rush_into),
                  *anims, run_time=2)
        self.wait(0.1)

        anims = []
        for i in range(1, 4):
            anims += [
                mh.rtransform(eq18[i][-2:], eq19[i][-2:], eq18[i][-2:].copy(), eq19[i][-5:-3]),
                mh.fade_replace(eq18[i][-3], eq19[i][-3]),
                mh.fade_replace(eq18[i][-3].copy(), eq19[i][-6]),
            ]
            VGroup(eq19[i][-1], eq19[i][-4]).set_color(RED)
        for i in range(2, 4):
            anims += [mh.rtransform(eq18[i][0], eq19[i][0])]
        self.play(mh.rtransform(eq18[0], eq19[0], eq18[-1], eq19[-1]),
                  *anims, run_time=1.8)
        self.wait(0.1)

        anims = []
        for i in range(1, 4):
            anims += [
                mh.rtransform(eq19[i][-3:], eq20[i][-3:], eq19[i][-5:-3], eq20[0][3:5], eq19[i][-6], eq20[0][1])
            ]
        for i in range(2, 4):
            anims += [mh.rtransform(eq19[i][0], eq20[i][0])]
        self.play(mh.rtransform(eq19[0][:2], eq20[0][:2], eq19[0][-1], eq20[0][-1]),
                  FadeIn(eq20[0][2], shift=mh.diff(eq19[0][1], eq20[0][1])),
                  mh.rtransform(eq19[-1], eq20[-1]),
                  *anims, run_time=2)
        self.wait(0.1)

        eq21 = MathTex(r'(1-2^{1-s})', r'(1^{-s}+2^{-s}+3^{-s}+\cdots)')
        (eq21[1][3:15:4] + eq21[0][-2]).set_color(RED)
        eq21.move_to(VGroup(eq16, eq20), coor_mask=UP)

        shift = mh.diff(eq16[0][0], eq21[1][1])
        self.play(mh.rtransform(eq16_2[:2], eq21[1][4:20:8],
                                eq16[0][:3], eq21[1][1:4], eq16[0][4:11], eq21[1][5:12],
                                eq20[0][-1], eq21[1][0]),
                  mh.rtransform(eq20[1][:], eq21[1][1:4], eq20[2][:], eq21[1][4:8],
                                eq20[3][:], eq21[1][8:12], eq20[4][:], eq21[1][12:],
                                eq20[0][:-1], eq21[0][2:-1]),
                  FadeOut(eq16_2[2:], eq16[0][12:19], eq16[0][20:], shift=shift),
                  FadeIn(eq21[0][-1], target_position=eq20[0][-1].get_center()+LEFT*0.2),
                  FadeIn(eq21[0][:2], shift=mh.diff(eq20[0][0], eq21[0][1])),
                  run_time=2)
        self.wait(0.1)

        eq22 = MathTex(r'(1-2^{1-s})', r'\zeta(s)')
        VGroup(eq22[0][-2], eq22[1][-2]).set_color(RED)
        eq23 = eq22.copy()

        mh.align_sub(eq22, eq22[0], eq21[0])
        eq22[1].move_to(eq21[1], coor_mask=RIGHT).shift(LEFT)

        self.play(mh.rtransform(eq21[0], eq22[0]), FadeOut(eq21[1]), FadeIn(eq22[1]),
                  run_time=1.4, rate_func=linear)
        self.wait(0.1)

        eq24 = MathTex(r'=', r'\zeta(s)')
        mh.align_sub(eq24, eq24[0], eq14[1])
        mh.align_sub(eq23, eq23[1], eq24[1]).move_to(eq16_3, coor_mask=RIGHT)
        self.play(mh.rtransform(eq22, eq23),
                  FadeOut(eq16_3, circ),
                  run_time=2.2)
        self.wait(0.1)

        eq25 = MathTex(r'\mathbb E[X^s]', r'=', r'2^{-\frac s2}s', r'(1-2^{1-s})', r'\Gamma({}^{\frac s2})', r'\zeta(s)')
        eq25[-2][2:5].move_to(eq25[1], coor_mask=UP)
        eq25[0][2].set_color(YELLOW)
        VGroup(eq25[0][-2], eq25[2][-1], eq25[2][2], eq25[3][-2], eq25[4][2], eq25[5][2]).set_color(RED)
        mh.align_sub(eq25, eq25[1], eq14[1]).move_to(ORIGIN, coor_mask=RIGHT)
        eq25.scale(1.2)
        self.play(mh.rtransform(eq14[:2], eq25[:2], eq14[2][:6], eq25[2][:], eq23[0], eq25[3],
                                eq23[1], eq25[5], eq15[1], eq25[-2]),
                  run_time=1.2)
        self.wait()

def font_size_sub(eq: Mobject, index: int, font_size: float):
    n = len(eq[:])
    eq_1 = eq[index].copy()
    pos = eq.get_center()
    eq[index].set(font_size=font_size).align_to(eq_1, RIGHT)
    eq[index:].align_to(eq_1, LEFT)
    return eq.move_to(pos, coor_mask=RIGHT)

class NormalDensity(Scene):
    def construct(self):
        eq1=MathTex(r'p(x)', r'=', r'\frac1{\sqrt{2\pi}}', r'e^{-\frac12x^2}', font_size=60)
        font_size_sub(eq1, 2, 45)
        self.add(eq1)

class EpsToZero(Scene):
    def construct(self):
        col = ManimColor('#FFAC2B')
        eq2 = Tex(r'\sf send $\varepsilon$ to zero', font_size=70, color=col, stroke_width=2)
        self.add(eq2)

class ExpXs(Scene):
    def construct(self):
        eq = MathTex(r'\mathbb E\left[X^s\right]', font_size=120)[0]
        eq[2].set_color(YELLOW)
        eq[3].set_color(RED)
        self.add(eq)

class JointMaxMin(Scene):
    def construct(self):
        eq1 = MathTex(r'\mathbb P(\max b_t > x, \min b_t < -y)', r'=', r'\sum_{n=1}^\infty(-1)^{n-1}',
                      r'\left(\mathbb P(LU\cdots)+\mathbb P(UL\cdots)\right)', font_size=70)
        VGroup(eq1[0][5:7], eq1[0][13:15]).set_color(BLUE)
        VGroup(eq1[3][3:5], eq1[3][12:14]).set_color(RED)
        eq1[2:].next_to(eq1[0], DOWN, buff=0.2).align_to(eq1, LEFT).shift(RIGHT*2)
        eq1.move_to(ORIGIN).to_edge(LEFT)

        self.add(eq1)

if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        MeanderTfm().render()