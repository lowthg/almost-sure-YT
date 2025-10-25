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
    def get_axes(self, scale=1., xlen = 0.9, ylen=0.95, scale_neg=1):
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
        self.play(FadeOut(path1), FadeIn(paths_arr))
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
        self.play(FadeOut(paths_arr3, *dots, *lines),
                  FadeIn(path1), rate_func=linear)


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

        self.wait()


class TDist(Scene):
    def construct(self):
        eq1 = MathTex(r'\pi\sqrt{T}', r'\sim', r'2D', font_size=70).set_z_index(2)
        box = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   corner_radius=0.15)
        self.add(eq1[0], box)
        self.wait(0.1)
        self.play(FadeIn(eq1[1:]))
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

        self.play(Create(path1), rate_func=linear, run_time=4)
        self.wait(0.1)

        np.random.seed(seeds[1])
        b_vals2 = np.concatenate(([0], np.random.normal(scale=s, size=ndt).cumsum()))
        b_vals2 -= b_vals2[-1] * t_vals

        path2 = ax.plot_line_graph(t_vals, b_vals2, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=4).set_z_index(2)

        self.play(path1.animate(run_time=0.5, rate_func=linear).set_stroke(opacity=0.3, color=BLUE).set_z_index(1.5),
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

if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        BridgeDev().render()