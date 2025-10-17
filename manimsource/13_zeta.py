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
    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = BLACK
        else:
            config.background_color = GREY
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
    def construct(self):
        width = 5.73
        height = 2.87
        w2 = width * 1.2
        h2 = height * 1.2
        scale = 0.3
        ndots = 100
        time=30.
        seeds = [1]
        np.random.seed(seeds[-1])
        pts = [[RIGHT * np.random.uniform(0., w2) + UP * np.random.uniform(0., h2)] for _ in range(ndots)]
        nframes = round(time * 30)
        s = scale * math.sqrt(1./30)
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
        self.wait()


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
        eq1 = MathTex('\zeta(s)', font_size=120)
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
    def construct(self):
        xlen = 8.8
        ylen = 4.4
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
        eqs = MathTex(r'012', font_size=45)[0]
        for i, x in enumerate([0, 1, 2]):
            eqs[i].next_to(ax.coords_to_point(x, 0), DOWN, buff=0.2)
        eq1 = MathTex(r'x', font_size=50).next_to(ax.x_axis.get_right(), UP, buff=0.25).shift(LEFT*0.15)
        eq2 = MathTex(r'p_X(x)', font_size=50).next_to(ax.y_axis.get_top(), RIGHT, buff=0.25).shift(DOWN*0.2)
        VGroup(eq1[0][0], eq2[0][1], eq2[0][3]).set_color(BLUE)
        a = math.sqrt(2/PI)
        xvals = np.linspace(0, xmax, n)
        def p(x):
            return psi(x*a)

        plot = ax.plot(p, x_range=(0, xmax), stroke_color=BLUE, stroke_width=5).set_z_index(3)
        area = ax.get_area(plot, color=ManimColor(BLUE.to_rgb()*0.5), opacity=1, x_range=(0., xmax), stroke_width=0, stroke_opacity=0).set_z_index(1)

        self.add(ax, eqs, eq1, eq2)
        self.wait(0.1)
        self.play(LaggedStart(Create(plot, run_time=2, rate_func=linear),
                              FadeIn(area, run_time=2, rate_func=linear), lag_ratio=0.4))
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
        seeds = [0, 4, 5]
        xmax = 1.2
        x0 = 0.1
        np.random.seed(seeds[-1])
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
        #eqx[2].move_to(ticks[3], coor_mask=RIGHT)
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
        s = 3
        lines = []
        for i in range(0, n_sample * 2 + 1, 2):
            x, y = ps[i]
            #lines.append(ax.plot(lambda x: y0, x_range=(ps[i-1][0], ps[i][0]), stroke_color=BLUE, stroke_width=7).set_z_index(5))
            if i > 0:
                lines.append(Line(ax.coords_to_point(x, y0 + s*(ps[i-1][1]-f(x))), ax.coords_to_point(x, y0+s*(y-f(x))), stroke_color=BLUE,
                                  stroke_width=2).set_z_index(4))
            lines.append(ax.plot(lambda x: y0 + s*(y - f(x)), x_range=(ps[i][0], ps[i+1][0]), stroke_color=BLUE, stroke_width=7).set_z_index(5))

        lines = VGroup(*lines)
        plt3 = ax.plot(lambda _: y0, (0, xmax), stroke_width=5, stroke_color=YELLOW).set_z_index(3)
        axlabel4 = MathTex(r'F_n(x)', r'-', r'F(x)', font_size=40, color=BLUE).move_to(ax.coords_to_point(0.32, 1)).set_z_index(3)

        self.play(FadeOut(eqx, *ticks, axlabel, axline, ax.x_axis, eqax1),
                  mh.rtransform(axlabel2[0], axlabel4[0], axlabel3[0], axlabel4[2]),
                  FadeIn(axlabel4[1]),
                  mh.transform(plt2, lines, plt1, plt3, run_time=2))

        self.wait()


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        Xi().render()