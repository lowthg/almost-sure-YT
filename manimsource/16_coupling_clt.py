from fontTools.unicodedata import block
from manim import *
import numpy as np
import math
import sys
import scipy as sp
from torch.utils.jit.log_extract import run_test

sys.path.append('../')
import manimhelper as mh

class CLT2(Scene):
    XCol = BLUE
    indexCol = GREEN
    opCol = YELLOW
    paramCol = ORANGE
    exponentCol = TEAL
    numCol = TEAL
    bgCol = GREY

    def __init__(self, *args, **kwargs):
        if not config.transparent:
            config.background_color = self.bgCol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)

        eq1 = MathTex(r'X_1', r', X_2', r', X_3', r', \ldots').set_z_index(4)
        eq2 = MathTex(r'\mathbb E[X_n]', r'=', r'0', r',\ ',
                      r'\mathbb E[X_n^2]', r'=', r'\sigma^2').set_z_index(4)
        eq3 = MathTex(r'S_n', r'=', r'X_1+X_2+\cdots+X_n').set_z_index(4)
        eq4 = MathTex(r'\frac{S_n}{\sqrt{n}}', r'\to', r'N(0,\sigma^2)').set_z_index(4)

        VGroup(eq1[0][0], eq1[1][1], eq1[2][1], eq2[0][2], eq2[4][2],
               eq3[0][0], eq3[2][0], eq3[2][3], eq3[2][-2],
               eq4[0][0]).set_color(self.XCol)
        VGroup(eq1[0][1], eq1[1][2], eq1[2][2], eq2[0][3], eq2[4][4],
               eq3[0][1], eq3[2][1], eq3[2][4], eq3[2][-1],
               eq4[0][1], eq4[0][-1]).set_color(self.indexCol)
        VGroup(eq2[0][0], eq2[4][0], eq4[2][0]).set_color(self.opCol)
        VGroup(eq2[-1][-2], eq4[2][-3]).set_color(self.paramCol)
        VGroup(eq2[4][-3], eq2[6][-1], eq4[2][-2]).set_color(self.exponentCol) # exponent
        VGroup(eq3[0][1], eq4[2][2]).set_color(TEAL) # numbers

        eq2.next_to(eq1, DOWN, buff=0.4)
        mh.align_sub(eq3, eq3[2][0], eq1[0][0], coor_mask=UP)
        gp1 = VGroup(eq1, eq2, eq3)
        box = SurroundingRectangle(gp1, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                   fill_opacity=0.6, corner_radius=0.2, buff=0.1)
        VGroup(gp1, box).to_edge(DOWN, buff=0.1)

        eq4.next_to(eq3, UP)

        box2 = SurroundingRectangle(eq4, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                   fill_opacity=0.6, corner_radius=0.2, buff=0.1)


        self.add(box)
        fades = [FadeIn(_) for _ in eq1]
        self.play(LaggedStart(*fades, lag_ratio=0.5))
        eq2_1 = eq2[:3].copy().move_to(eq1, coor_mask=RIGHT)
        self.play(FadeIn(eq2_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1, eq2[:3]),
                  Succession(Wait(0.4), FadeIn(eq2[3:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][:], eq3[2][:2], eq1[1][1:], eq3[2][3:5],
                                eq1[3][1:], eq3[2][6:9]),
                  mh.fade_replace(eq1[1][0], eq3[2][2], coor_mask=RIGHT),
                  mh.fade_replace(eq1[2][:] + eq1[3][0], eq3[2][5], coor_mask=RIGHT),
                  FadeIn(eq3[:2]),
                  FadeIn(eq3[2][-3:], shift=mh.diff(eq1[3][1:], eq3[2][6:9])*RIGHT))
        self.wait(0.1)
        self.play(FadeIn(eq4, box2))
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

class BMDef(CLT2):
    bgCol = BLACK
    tCol = RED
    BCol = YELLOW
    textCol = BLUE

    #def get_plot(self):

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

        return ax, eqt, xlen, ylen, ymax

    def get_plot(self):
        ax, eqt, xlen, ylen, ymax, mark1 = self.get_axes()

        return VGroup(ax, eqt, mark1)

    def stopping_time(self, eq1):
        eq2 = Tex(r'$t_1$', r'\sf\ is a stopping time', font_size=80)
        eq3 = MathTex(r'\mathbb E[t_1]', r'<', r'\infty', font_size=80)
        eq2[1].set_color(self.textCol)
        VGroup(eq2[0][0], eq3[0][2]).set_color(self.tCol)
        VGroup(eq2[0][1], eq3[0][3]).set_color(self.indexCol)
        eq3[0][0].set_color(self.opCol)
        eq3[-1].set_color(self.numCol)

        eq2.move_to(mh.pos(RIGHT*0.35)).move_to(eq1, coor_mask=UP)
        eq3.next_to(eq2, DOWN)

        self.play(eq1.animate(run_time=1.6).shift(LEFT*4),
                  Succession(Wait(0.6), FadeIn(eq2)))
        self.wait(0.1)
        gp1 = VGroup(eq2.copy(), eq3).move_to(eq2, coor_mask=UP)
        self.play(mh.transform(eq2, gp1[0]), Succession(Wait(0.5), FadeIn(eq3)))

    def construct(self):
        self.do_anim()

    def do_anim(self, just_plot=False):
        MathTex.set_default(stroke_width=1.5)
        seeds = [3, 4, 18]
        npts = 1920
        ndt = npts - 1

        np.random.seed(seeds[0])

        ax, eqt, xlen, ylen, ymax = self.get_axes()

        eqt.set_color(self.tCol)

        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])
        b_vals = -np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))

        path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(2)

        tarr = np.linspace(0.1, 0.95, 4)
        marks = [ax.x_axis.get_tick(t, size=0.1).set_stroke(width=6) for t in tarr]
        eq_marks = [MathTex(r't_{}'.format(i), font_size=60).set_z_index(3).next_to(marks[i], DOWN, buff=0.05) for i in range(4)]
        for _ in eq_marks:
            _[0][0].set_color(self.tCol)
            _[0][1].set_color(self.indexCol)
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
                VGroup(tmp[-1][0][0], tmp[-1][0][4]).set_color(self.BCol)
                VGroup(tmp[-1][0][1], tmp[-1][0][5]).set_color(self.tCol)
                VGroup(tmp[-1][0][2], tmp[-1][0][6]).set_color(self.indexCol)
                eqs.append(tmp[-1])
                if i > 1:
                    eqargs.append(r',')
                eqargs.append(eqstr)
            anims.append(FadeIn(*tmp, rate_func=linear, run_time=0.5))
        eq1 = MathTex(*eqargs, font_size=60).next_to(VGroup(*eqs), DOWN)[:]
        for i in range(3):
            eq1[2*i] = eqs[i][0].copy().move_to(eq1[2*i])
        br1 = BraceLabel(eq1, r'\sf independent', label_constructor=mh.mathlabel_ctr2, font_size=80,
                         brace_config={'color': RED}).set_z_index(2)

        eq3 = MathTex(r'B_t', r'\sim', r'N(0,t)', font_size=80).set_z_index(3)
        eq3.move_to(eq1, coor_mask=UP).shift(DOWN*0.5)
        eq3[0][0].set_color(self.BCol)
        eq3[0][1].set_color(self.tCol)
        eq3[2][0].set_color(self.opCol)
        eq3[2][2].set_color(self.numCol)
        eq3[2][4].set_color(self.tCol)

        eqB1 = MathTex(r'B', color=YELLOW, font_size=80, stroke_width=2)
        eqB1.move_to(ax.coords_to_point(0.6, 0.5))

        i1 = int(npts * 0.75)
        t1 = t_vals[i1]
        b1 = b_vals[i1]
        p1 = ax.coords_to_point(t1, 0)
        p2 = ax.coords_to_point(t1, b1)
        eq7 = MathTex(r't_1', font_size=60).set_z_index(3).next_to(p1, DOWN, buff=0.15)
        eq8 = MathTex(r'B_{t_1}', font_size=60).set_z_index(3).next_to(p2, UP, buff=0.4)
        line2 = DashedLine(p1, p2, color=YELLOW, stroke_width=5)
        eq9 = MathTex(r'B_{t_1}', r'\sim', r'X_1', font_size=80).set_z_index(3)
        eq9.move_to(eq1, coor_mask=UP).shift(DOWN*0.5)

        VGroup(eq7[0], eq8[0][1], eq9[0][1]).set_color(self.tCol)
        VGroup(eq8[0][0], eq9[0][0]).set_color(self.BCol)
        VGroup(eq7[0][1], eq8[0][2], eq9[0][2], eq9[2][1]).set_color(self.indexCol)
        VGroup(eq9[2][0]).set_color(self.XCol)

        if just_plot:
            return VGroup(ax, eqt, path1, eq7, eq8, eq9, line2, eqB1), t_vals, b_vals, i1

        self.add(ax, eqt)
        self.wait(0.1)
        self.play(Create(path1, rate_func=linear, run_time=4),
                  Succession(Wait(3), FadeIn(eqB1, rate_func=linear)))
        self.play(Succession(*anims))
        self.play(LaggedStart(mh.rtransform(eqs[0][0], eq1[0], eqs[1][0], eq1[2], eqs[2][0], eq1[4]),
                  FadeIn(eq1[1], eq1[3], br1), lag_ratio=0.2))
        self.wait(0.1)
        self.play(FadeOut(*eq_marks, *marks, eq1, br1),
                  FadeIn(eq3))
        self.wait(0.1)
        self.play(FadeOut(eq3))

        # t1


        self.wait(0.1)
        self.play(FadeIn(eq7), Create(line2),
                  Succession(Wait(0.8), FadeIn(eq8)),
                  )
        self.play(mh.rtransform(eq8[0].copy(), eq9[0], run_time=1.8),
                  Succession(Wait(1.2), FadeIn(eq9[1:])))

        self.stopping_time(eq9)

        self.wait()

class BMTimes(BMDef):
    def construct(self):
        gp1, t_vals, b_vals, i_t1 = self.do_anim(just_plot=True)
        ax = gp1[0]
        #gp1 = gp1[:-1]
        eqsim1 = gp1[5]

        npts = len(t_vals)
        i_t2 = int(0.3 * npts)
        i_t3 = int(0.5 * npts)

        t1 = t_vals[i_t1]
        b1 = b_vals[i_t1]
        t2 = t_vals[i_t2]
        t3 = t_vals[i_t3]

        eqB1 = gp1[7]
        origin = ax.coords_to_point(0, 0)
        p1 = ax.coords_to_point(t1, b1)
        s = np.sqrt(t_vals[1])
        np.random.seed(107)
        b2_vals = np.concatenate((b_vals[i_t1:], np.random.normal(scale=s, size=i_t1).cumsum() + b_vals[-1])) - b_vals[i_t1]
        b2 = b2_vals[i_t2]
        p2 = ax.coords_to_point(t2, b2)

        np.random.seed(200)
        b3_vals = np.concatenate((b2_vals[i_t2:], np.random.normal(scale=s, size=i_t2).cumsum() + b2_vals[-1])) - b2_vals[i_t2]
        b3 = b3_vals[i_t3]
        p3 = ax.coords_to_point(t3, b3)

        gp2 = VGroup(gp1[3], gp1[4], gp1[6])
        path1 = gp1[2]
        path2 = ax.plot_line_graph(t_vals, b2_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(5)
        path3 = ax.plot_line_graph(t_vals, b3_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(8)

        pt2 = ax.coords_to_point(t2, 0)
        pb2 = ax.coords_to_point(t2, b2)
        eqb2diff = MathTex(r'B^{(2)}_t', '=', r'B_{t_1+t}-B_{t_1}', font_size=80)
        eqt2 = MathTex(r't_2', font_size=60).set_z_index(6).next_to(pt2, DOWN, buff=0.15)
        eqb2 = MathTex(r'B^{(2)}_{t_2}', font_size=60).set_z_index(6).next_to(pb2, UP, buff=0.4)
        line2 = DashedLine(pt2, pb2, color=YELLOW, stroke_width=5)
        eqsim2 = MathTex(r'B^{(2)}_{t_2}', r'\sim', r'X_2', font_size=80).set_z_index(3)
        eqsim2.next_to(eqsim1, RIGHT, buff=0.5)
        eqb2diff.move_to(mh.pos(UP*0.62+RIGHT*0.3))
        VGroup(eqt2[0], eqb2[0][-2], eqsim2[0][-2],
               eqb2diff[0][-1], eqb2diff[2][1], eqb2diff[2][4], eqb2diff[2][7]).set_color(self.tCol)
        VGroup(eqb2[0][0], eqsim2[0][0],
               eqb2diff[0][0], eqb2diff[2][0], eqb2diff[2][6]).set_color(self.BCol)
        VGroup(eqt2[0][1], eqb2[0][-1], eqsim2[0][-1], eqsim2[2][1], eqb2[0][1:4], eqsim2[0][1:4],
               eqb2diff[0][1:4], eqb2diff[2][2], eqb2diff[2][8]).set_color(self.indexCol)
        VGroup(eqsim2[2][0]).set_color(self.XCol)

        pt3 = ax.coords_to_point(t3, 0)
        pb3 = ax.coords_to_point(t3, b3)
        eqb3diff = MathTex(r'B^{(3)}_t', '=', r'B^{(2)}_{t_2+t}-B^{(2)}_{t_2}', font_size=80).set_z_index(9)
        eqt3 = MathTex(r't_3', font_size=60).set_z_index(9).next_to(pt3, UP, buff=0.15)
        eqb3 = MathTex(r'B^{(3)}_{t_3}', font_size=60).set_z_index(9).next_to(pb3, UR, buff=0)
        line3 = DashedLine(pt3, pb3, color=YELLOW, stroke_width=5)
        eqsim3 = MathTex(r'B^{(3)}_{t_3}', r'\sim', r'X_3', font_size=80).set_z_index(3)
        eqb3diff.move_to(mh.pos(UP*0.31+LEFT*0.23))
        VGroup(eqt3[0], eqb3[0][-2], eqsim3[0][-2],
               eqb3diff[0][-1], eqb3diff[2][4], eqb3diff[2][7], eqb3diff[2][13]).set_color(self.tCol)
        VGroup(eqb3[0][0], eqsim3[0][0],
               eqb3diff[0][0], eqb3diff[2][0], eqb3diff[2][9]).set_color(self.BCol)
        VGroup(eqt3[0][1], eqb3[0][-1], eqsim3[0][-1], eqsim3[2][1], eqsim3[0][1:4], eqb3[0][1:4],
               eqb3diff[0][1:4], eqb3diff[2][5], eqb3diff[2][14],
               eqb3diff[2][1:4], eqb3diff[2][10:14]).set_color(self.indexCol)
        VGroup(eqsim3[2][0]).set_color(self.XCol)


        self.add(gp1)
        self.wait()
        shift = p1 - origin
        self.play(path2.shift(shift).animate(run_time=2).shift(-shift),
                  path1.animate().set_stroke(opacity=0.3, color=BLUE),
                  gp2.animate().set_opacity(0.1),
                  Succession(Wait(1.3), FadeIn(eqb2diff)),
                  FadeOut(eqB1))
        self.wait(0.1)

        gp = VGroup(eqsim1.copy(), eqsim2).move_to(eqsim1, coor_mask=RIGHT)
        eqsim1_1 = gp[0]
        self.play(FadeIn(eqt2, run_time=0.5, rate_func=linear), Create(line2, run_time=0.5),
                  Succession(Wait(0.4), FadeIn(eqb2)),
                  )
        self.play(mh.rtransform(eqb2[0].copy(), eqsim2[0], run_time=1.8),
                  mh.transform(eqsim1, eqsim1_1),
                  Succession(Wait(1.2), FadeIn(eqsim2[1:])))
        self.wait(0.1)

        shift = p2 - origin
        gp3 = VGroup(eqt2, line2, eqb2)
        eq_tmp = eqb3diff.copy().set_color(BLACK).set_stroke(width=16).set_z_index(8.5)
        self.play(path3.shift(shift).animate(run_time=2).shift(-shift),
                  path2.animate().set_stroke(opacity=0.3, color=GREEN),
                  gp3.animate().set_opacity(0.1),
                  Succession(Wait(1.3), FadeIn(eqb3diff, eq_tmp)),
                  FadeOut(eqb2diff))
        self.wait(0.1)

        eqb3_1 = eqb3.copy().set_stroke(width=14).set_color(BLACK).set_z_index(8.5)
        self.play(FadeIn(eqt3, run_time=0.5, rate_func=linear), Create(line3, run_time=0.7),
                  Succession(Wait(0.5), FadeIn(eqb3, eqb3_1)),
                  )
        eqsim3.next_to(eqsim2, RIGHT, buff=0.5)
        gp_1 = VGroup(eqsim1, eqsim2)
        gp = VGroup(gp_1.copy(), eqsim3).move_to(gp_1, coor_mask=RIGHT).shift(DOWN*0.4)
        gp_2 = gp[0]
        self.play(mh.rtransform(eqb3[0].copy(), eqsim3[0], run_time=1.4),
                  mh.transform(gp_1, gp_2),
                  Succession(Wait(1), FadeIn(eqsim3[1:])))
        self.wait(0.1)


        self.wait()

