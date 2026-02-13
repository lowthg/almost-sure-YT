from fontTools.unicodedata import block
from manim import *
import numpy as np
import math
import sys
import scipy as sp
from torch.utils.jit.log_extract import run_test

sys.path.append('../')
import manimhelper as mh

def eq_shadow(eq: VGroup, fg_z_index=4., bg_z_index=0., bg_color=BLACK, bg_stroke_width=10.):
    res = VGroup()
    for eq1 in eq:
        elem = VGroup()
        for eq2 in eq1:
            elem.add(VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res

class CLT2(Scene):
    XCol = BLUE
    indexCol = GREEN
    opCol = YELLOW
    paramCol = ORANGE
    exponentCol = TEAL
    numCol = TEAL
    tCol = RED
    BCol = YELLOW
    textCol = BLUE
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
        eq5 = MathTex(r'\mathbb P\Big(', r'\frac{S_n}{\sqrt{n}}', r' > x\Big)', r'\to', r'\mathbb P(', r'N(0,\sigma^2)', r' > x\right)').set_z_index(4)
        ht = eq5[1].height
        eq5[0][1].stretch_to_fit_height(ht)
        eq5[2][-1].stretch_to_fit_height(ht)

        VGroup(eq1[0][0], eq1[1][1], eq1[2][1], eq2[0][2], eq2[4][2],
               eq3[0][0], eq3[2][0], eq3[2][3], eq3[2][-2],
               eq4[0][0]).set_color(self.XCol)
        VGroup(eq1[0][1], eq1[1][2], eq1[2][2], eq2[0][3], eq2[4][4],
               eq3[0][1], eq3[2][1], eq3[2][4], eq3[2][-1],
               eq4[0][1], eq4[0][-1]).set_color(self.indexCol)
        VGroup(eq2[0][0], eq2[4][0], eq4[2][0], eq5[0][0], eq5[4][0]).set_color(self.opCol)
        VGroup(eq2[-1][-2], eq4[2][-3]).set_color(self.paramCol)
        VGroup(eq2[4][-3], eq2[6][-1], eq4[2][-2]).set_color(self.exponentCol) # exponent
        VGroup(eq3[0][1], eq4[2][2], eq5[2][1], eq5[6][1], eq2[2]).set_color(self.numCol) # numbers
        copy_colors(eq4[0], eq4[2], eq5[1], eq5[5])

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

        mh.align_sub(eq5, eq5[3], eq4[1], coor_mask=UP)
        box3 = SurroundingRectangle(eq5, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                   fill_opacity=0.6, corner_radius=0.2, buff=0.1)
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0], eq5[1], eq4[1], eq5[3], eq4[2], eq5[5]),
                  mh.rtransform(box2, box3),
                  Succession(Wait(0.5), FadeIn(eq5[0], eq5[2], eq5[4], eq5[6])))
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
        eq3.move_to(eq1, coor_mask=UP).shift(DOWN*1)
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
        line2 = DashedLine(p1, p2, color=WHITE, stroke_width=5)
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

def copy_colors(*objs: Mobject):
    """
    :param objs: equal length sources then targets
    """
    m = len(objs)
    assert m % 2 == 0 and m > 0
    m = m // 2
    for j in range(m):
        n = len(objs[j][:])
        assert n == len(objs[j+m][:])
        for i in range(n):
            objs[j+m][i].set_color(objs[j][i].color)

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
        line2 = DashedLine(pt2, pb2, color=WHITE, stroke_width=5)
        eqsim2 = MathTex(r'B^{(2)}_{t_2}', r'\sim', r'X_2', font_size=80).set_z_index(3)
        eqsim2.next_to(eqsim1, RIGHT, buff=0.5)
        mh.align_sub(eqsim2, eqsim2[0][0], eqsim1[0][0], aligned_edge=DOWN, buff=0, coor_mask=UP)
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
        eqb3 = MathTex(r'B^{(3)}_{t_3}', font_size=60).set_z_index(9).next_to(pb3, UR, buff=0).shift(DOWN*0.15)
        line3 = DashedLine(pt3, pb3, color=WHITE, stroke_width=5)
        eqsim3 = MathTex(r'B^{(3)}_{t_3}', r'\sim', r'X_3', font_size=80).set_z_index(3)
        eqb3diff.move_to(mh.pos(UP*0.31+LEFT*0.23))
        VGroup(eqt3[0], eqb3[0][-2], eqsim3[0][-2],
               eqb3diff[0][-1], eqb3diff[2][4], eqb3diff[2][7], eqb3diff[2][13]).set_color(self.tCol)
        VGroup(eqb3[0][0], eqsim3[0][0],
               eqb3diff[0][0], eqb3diff[2][0], eqb3diff[2][9]).set_color(self.BCol)
        VGroup(eqt3[0][1], eqb3[0][-1], eqsim3[0][-1], eqsim3[2][1], eqsim3[0][1:4], eqb3[0][1:4],
               eqb3diff[0][1:4], eqb3diff[2][5], eqb3diff[2][14],
               eqb3diff[2][1:4], eqb3diff[2][10:13]).set_color(self.indexCol)
        VGroup(eqsim3[2][0]).set_color(self.XCol)

        mask_box = (Rectangle(width=2, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                  fill_color=BLACK, fill_opacity=1)
                    .next_to(ax.coords_to_point(1, 0), RIGHT, buff=0).set_z_index(100))
        x_ax_2 = Intersection(ax.x_axis.copy().set_z_index(101), mask_box, color=WHITE)
        mask = VGroup(mask_box, gp1[1].copy().set_z_index(102), ax.x_axis.copy().set_z_index(101), x_ax_2)

        self.add(gp1)
        self.wait()
        shift = p1 - origin
        self.add(mask)
        self.play(path2.shift(shift).animate(run_time=2).shift(-shift),
                  path1.animate().set_stroke(opacity=0.3, color=BLUE),
                  gp2.animate().set_opacity(0.1),
                  Succession(Wait(1.3), FadeIn(eqb2diff)),
                  FadeOut(eqB1))
        self.remove(mask)
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
        eqb3diffb = eqb3diff.copy().set_color(BLACK).set_stroke(width=16).set_z_index(8.5)
        self.add(mask)
        self.play(path3.shift(shift).animate(run_time=2).shift(-shift),
                  path2.animate().set_stroke(opacity=0.3, color=GREEN),
                  gp3.animate().set_opacity(0.1),
                  Succession(Wait(1.3), FadeIn(eqb3diff, eqb3diffb)),
                  FadeOut(eqb2diff))
        self.remove(mask)
        self.wait(0.1)

        eqb3_1 = eqb3.copy().set_stroke(width=14).set_color(BLACK).set_z_index(8.5)
        self.play(FadeIn(eqt3, run_time=0.5, rate_func=linear), Create(line3, run_time=0.7),
                  Succession(Wait(0.5), FadeIn(eqb3, eqb3_1)),
                  )
        eqsim3.next_to(eqsim2, RIGHT, buff=0.5)
        mh.align_sub(eqsim3, eqsim3[0][0], eqsim2[0][0], aligned_edge=DOWN, buff=0, coor_mask=UP)
        gp_1 = VGroup(eqsim1, eqsim2)
        gp = VGroup(gp_1.copy(), eqsim3).move_to(gp_1, coor_mask=RIGHT).shift(DOWN*0.4)
        gp_2 = gp[0]
        self.play(mh.rtransform(eqb3[0].copy(), eqsim3[0], run_time=1.4),
                  mh.transform(gp_1, gp_2),
                  Succession(Wait(1), FadeIn(eqsim3[1:])))
        self.wait(0.1)

        eq_tmp = MathTex(r'=', font_size=80)[0].set_z_index(3)
        eq_tmp = [eq_tmp.move_to(eqsim1[1]), eq_tmp.copy().move_to(eqsim2[1]), eq_tmp.copy().move_to(eqsim3[1])]
        self.play(FadeIn(*eq_tmp), FadeOut(eqsim1[1], eqsim2[1], eqsim3[1]))
        eqsim1 = VGroup(eqsim1[0], eq_tmp[0], eqsim1[2])
        eqsim2 = VGroup(eqsim2[0], eq_tmp[1], eqsim2[2])
        eqsim3 = VGroup(eqsim3[0], eq_tmp[2], eqsim3[2])

        self.wait(0.1)
        self.play(FadeOut(eqb3diff, eqb3diffb))

        # join back together to single BMs
        t0_vals = np.concatenate((t_vals[:i_t1], t_vals[:i_t2] + t1, t_vals + t1 + t2))
        b0_vals = np.concatenate((b_vals[:i_t1], b2_vals[:i_t2] + b1, b3_vals + b1 + b2))
        tmax = t0_vals[-1]
        scalex = 1/tmax
        scaley = 1.
        scale_dir = scalex*RIGHT + scaley*UP
        t0_vals *= scalex
        b0_vals *= scaley
        paths = VGroup(*[_['line_graph'] for _ in [path1, path2, path3]])
        i_s2 = i_t1 + i_t2
        path1_1 = ax.plot_line_graph(t0_vals[:npts], b0_vals[:npts], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(5)
        path2_1 = ax.plot_line_graph(t0_vals[i_t1:i_t1+npts], b0_vals[i_t1:i_t1+npts], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(5)
        path3_1 = ax.plot_line_graph(t0_vals[i_s2:i_s2+npts], b0_vals[i_s2:], add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(5)
        shift1 = (path1_1['line_graph'].get_start() - origin)
        shift2 = (path2_1['line_graph'].get_start() - origin)
        shift3 = (path3_1['line_graph'].get_start() - origin)

        path1_1.generate_target()
        path2_1.generate_target()
        path3_1.generate_target()
        gp4 = gp2
        eqb2.set_z_index(9)
        eqb2_1 = eqb2.copy().set_stroke(width=14).set_color(BLACK).set_z_index(8.5)
        gp5 = VGroup(eqb2, eqt2, line2, eqb2_1)
        gp6 = VGroup(eqb3, eqt3, line3, eqb3_1)
        self.play(mh.rtransform(path1, path1_1.shift(-shift1).set_stroke(opacity=0.3, color=BLUE),
                               path2, path2_1.shift(-shift2).set_stroke(opacity=0.3, color=GREEN),
                               path3, path3_1.shift(-shift3)),
                  gp4.animate().shift((gp4.get_center()-origin) * (scale_dir-1)),
                  gp5.animate().shift((pt2-origin) * (scale_dir-1)),
                  gp6.animate().shift((pt3-origin) * (scale_dir-1)),
                  run_time=1.5)
        self.wait(0.2)
        tline2 = Arrow(origin, ax.coords_to_point(t2*scalex, 0), buff=0, stroke_color=WHITE, stroke_opacity=0., stroke_width=5).set_z_index(10).set_opacity(0)
        tline3 = Arrow(origin, ax.coords_to_point(t3*scalex, 0), buff=0, stroke_color=WHITE, stroke_opacity=0., stroke_width=5).set_z_index(10).set_opacity(0)
        gp5.add(tline2)
        gp6.add(tline3)
        self.play(MoveToTarget(path3_1, run_time=1.5),
                  gp6.animate(run_time=1.5).shift(shift3).set_opacity(1),
                  Succession(Wait(0.3), MoveToTarget(path2_1, run_time=1.2)),
                  Succession(Wait(0.3), gp5.animate(run_time=1.2).shift(shift2).set_opacity(1)),
                  Succession(Wait(0.6), MoveToTarget(path1_1, run_time=0.9)),
                  Succession(Wait(0.6), gp4.animate(run_time=0.9).shift(shift1).set_opacity(1)),
                  )

        # u times
        i_u2 = i_t1 + i_t2
        i_u3 = i_u2 + i_t3
        u1 = t0_vals[i_t1]
        u2 = t0_vals[i_u2]
        u3 = t0_vals[i_u3]
        pu2 = ax.coords_to_point(u2, 0)
        pu3 = ax.coords_to_point(u3, 0)
        line4 = DashedLine(line2.get_start(), pu2, color=WHITE, stroke_width=5).set_z_index(10)
        line5 = DashedLine(line3.get_end(), pu3, color=WHITE, stroke_width=5).set_z_index(10)
        equ2 = MathTex(r'u_2', r'\!=\!t_1+t_2', font_size=60).set_z_index(10).next_to(pu2, DOWN, buff=0.15)
        equ3 = MathTex(r'u_3', r'\!=\!t_1+t_2+t_3', font_size=60).set_z_index(10).next_to(pu3, DOWN, buff=0.15)
        equ2[1].scale(0.8, about_edge=LEFT)
        equ3[1].scale(0.8, about_edge=LEFT)
        mh.align_sub(equ2, equ2[0], pu2, DOWN, buff=0.15)
        mh.align_sub(equ3, equ3[0], pu3, DOWN, buff=0.15)
        equnsum = MathTex(r'u_n', r'=', r't_1+t_2+\cdots+t_n', font_size=80).set_z_index(10)
        equnsum.next_to(VGroup(eqsim2, eqsim3), UP, buff=0.5)
        eqsimn = MathTex(r'B_{t_n}^{(n)}', r'=', r'X_n', font_size=80).set_z_index(10)
        eqbdiff = MathTex(r'B_{u_n} - B_{u_{n-1}}', r'=', r'X_n', font_size=80)
        mh.align_sub(eqsimn, eqsimn[1], eqsim2[1])
        mh.align_sub(eqbdiff, eqbdiff[1], eqsimn[1]).move_to(eqsimn, coor_mask=RIGHT)

        eqbdiff1 = MathTex(r'B_{u_1}', r'=', r'X_1', font_size=60)
        eqbdiff2 = MathTex(r'B_{u_2} - B_{u_1}', r'=', r'X_2', font_size=60)
        eqbdiff3 = MathTex(r'B_{u_3} - B_{u_2}', r'=', r'X_3', font_size=60)
        eqbdiffn = MathTex(r'B_{u_n} - B_{u_{n-1}}', r'=', r'X_n', font_size=60)
        eqbdots = MathTex(r'\vdots', font_size=60)

        mh.align_sub(eqbdiffn, eqbdiffn[1], eqbdiff[1]).to_edge(DOWN, buff=0.1).shift(RIGHT*0.9)
        eqbdots.next_to(eqbdiffn[1], UP, buff=0.3)
        mh.align_sub(eqbdiff3, eqbdiff3[1], eqbdiffn[1])
        mh.align_sub(eqbdiff3, eqbdiff3[1], eqbdots, UP, coor_mask=UP, buff=0.3)
        mh.align_sub(eqbdiff2, eqbdiff2[1], eqbdiff3[1]).next_to(eqbdiff3, UP, coor_mask=UP, buff=0.1)
        mh.align_sub(eqbdiff1, eqbdiff1[1], eqbdiff2[1]).next_to(eqbdiff2, UP, coor_mask=UP, buff=0.1)
        for _ in [eqbdiff1, eqbdiff2, eqbdiff3]:
            _[0][:3].align_to(eqbdiffn, LEFT)
        for _ in [eqbdiff2, eqbdiff3]:
            _[0][3].align_to(eqbdiffn[0][3], LEFT)
            _[0][4:].align_to(eqbdiffn[0][4], LEFT)

        eqxsum = MathTex(r'B_{u_n}', r'=', r'X_1+X_2+\cdots+X_n', font_size=80)
        eqssum = MathTex(r'B_{u_n}', r'=', r'S_n', font_size=80)
        eqxsum.move_to(mh.pos(DOWN*0.6+RIGHT*0.1))
        mh.align_sub(eqssum, eqssum[1], eqxsum[1]).move_to(eqxsum, coor_mask=RIGHT)

        VGroup(equ2[0][0], equ3[0][0], equ2[1][1], equ2[1][4],
               equ3[1][1], equ3[1][4], equ3[1][7]).set_color(self.tCol)
        VGroup(equ2[0][1], equ3[0][1], equ2[1][2], equ2[1][5],
               equ3[1][2], equ3[1][5], equ3[1][8],
               equnsum[2][1], equnsum[2][-1], equnsum[2][4], equnsum[0][1]).set_color(self.indexCol)
        VGroup(equnsum[0][0], equnsum[2][0], equnsum[2][-2], equnsum[2][3]).set_color(self.tCol)
        VGroup(eqbdiff[0][-1]).set_color(self.numCol)
        copy_colors(*eqsim2[:], *eqsimn[:])
        copy_colors(eqsim1[0], eqsim1[0], eqsim2[2], eqbdiff[0][:3], eqbdiff[0][4:7], eqbdiff[2])
        copy_colors(*eqbdiff, *eqbdiffn)
        copy_colors(eqbdiff[0][:-2], eqbdiff[2], eqbdiff3[0], eqbdiff3[2])
        copy_colors(eqbdiff[0][:-2], eqbdiff[2], eqbdiff2[0], eqbdiff2[2])
        copy_colors(eqbdiff[0][:3], eqbdiff[2], eqbdiff1[0], eqbdiff1[2])
        copy_colors(eqbdiff1[0], eqbdiff1[2], eqbdiff1[2], eqbdiff1[2],
                    eqxsum[0], eqxsum[2][:2], eqxsum[2][3:5], eqxsum[2][-2:])
        copy_colors(eqxsum[0], eqxsum[2][:2], eqssum[0], eqssum[2])

        lines = [
            Line(obj.get_corner(DL)+LEFT*0.2, obj.get_corner(UR), stroke_color=RED, stroke_width=6).set_z_index(12)
            for obj in [eqbdiff1[0], eqbdiff2[0][:2], eqbdiff2[0][-3:-1], eqbdiff3[0][:2], eqbdiff3[0][-3:-1], eqbdiffn[0][-5:-3]]
        ]

        self.wait(0.1)
        self.play(Create(line4, rate_func=linear, run_time=0.7),
                  Succession(Wait(0.35), FadeIn(equ2)))
        self.play(Create(line5, rate_func=linear, run_time=0.5),
                  Succession(Wait(0.25), FadeIn(equ3)))
        self.wait(0.1)

        self.play(mh.rtransform(equ3[0][0].copy(), equnsum[0][0], equ3[1][0], equnsum[1],
                                equ3[1][1:7], equnsum[2][:6], equ3[1][-3].copy(), equnsum[2][-3],
                                equ3[1][-2], equnsum[2][-2]),
                  mh.stretch_replace(equ3[0][1].copy(), equnsum[0][1]),
                  mh.stretch_replace(equ3[1][-1], equnsum[2][-1]),
                  FadeIn(equnsum[2][6:9], target_position=equ3[1][6]),
                  FadeOut(equ2[1:]))
        self.wait(0.1)

        self.play(mh.rtransform(eqsim2[1], eqsimn[1], eqsim2[0][:2], eqsimn[0][:2],
                                eqsim2[0][3:5], eqsimn[0][3:5], eqsim2[2][0], eqsimn[2][0]),
                  mh.fade_replace(eqsim2[0][2], eqsimn[0][2]),
                  mh.fade_replace(eqsim2[0][5], eqsimn[0][5]),
                  mh.fade_replace(eqsim2[2][1], eqsimn[2][1]),
                  FadeOut(eqsim1, eqsim3),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eqsimn[0][0], eqbdiff[0][0], eqsimn[0][5], eqbdiff[0][2],
                                eqsimn[0][0].copy(), eqbdiff[0][4], eqsimn[0][5].copy(), eqbdiff[0][6],
                                eqsimn[1:], eqbdiff[1:]),
                  mh.fade_replace(eqsimn[0][4], eqbdiff[0][1]),
                  mh.fade_replace(eqsimn[0][4].copy(), eqbdiff[0][5]),
                  FadeOut(eqsimn[0][1:4], shift=mh.diff(eqsimn[0][0], eqbdiff[0][4])),
                  FadeIn(eqbdiff[0][3], shift=mh.diff(eqsimn[0], eqbdiff[0][3])*RIGHT),
                  FadeIn(eqbdiff[0][7:9], shift=mh.diff(eqsimn[0][5], eqbdiff[0][6])),
                  run_time=1.4)
        self.wait(0.1)
        eq_1 = eqbdiff.copy()
        eq_2 = eqbdiff.copy()
        eq_3 = eqbdiff.copy()
        self.play(
            equnsum.animate(run_time=1.4).scale(0.6).move_to(mh.pos(LEFT*0.5+UP*0.75)),
            Succession(Wait(0.4), AnimationGroup(
            mh.rtransform(eqbdiff, eqbdiffn),
            mh.rtransform(eq_1[0][:2], eqbdiff3[0][:2], eq_1[0][3:6], eqbdiff3[0][3:6], eq_1[1], eqbdiff3[1],
                          eq_1[2][0], eqbdiff3[2][0]),
            mh.fade_replace(eq_1[0][2], eqbdiff3[0][2]),
            mh.fade_replace(eq_1[0][6:9], eqbdiff3[0][6]),
            mh.fade_replace(eq_1[2][1], eqbdiff3[2][1]),
            mh.rtransform(eq_2[0][:2], eqbdiff2[0][:2], eq_2[0][3:6], eqbdiff2[0][3:6], eq_2[1], eqbdiff2[1],
                          eq_2[2][0], eqbdiff2[2][0]),
            mh.fade_replace(eq_2[0][2], eqbdiff2[0][2]),
            mh.fade_replace(eq_2[0][6:9], eqbdiff2[0][6]),
            mh.fade_replace(eq_2[2][1], eqbdiff2[2][1]),
            mh.rtransform(eq_3[0][:2], eqbdiff1[0][:2], eq_3[1], eqbdiff1[1],
                          eq_3[2][0], eqbdiff1[2][0]),
            mh.fade_replace(eq_3[0][2], eqbdiff1[0][2]),
            mh.fade_replace(eq_3[2][1], eqbdiff1[2][1]),
        run_time=1.6)),
                       Succession(Wait(1.), FadeIn(eqbdots)))
        self.wait(0.1)
        self.play(Create(lines[0]), Create(lines[2]), run_time=0.5)
        self.wait(0.1)
        self.play(Create(lines[1]), Create(lines[4]), run_time=0.5)
        self.wait(0.1)
        self.play(Create(lines[3]), run_time=0.5)
        self.play(Create(lines[5]), run_time=0.5)
        self.wait(0.1)
        self.play(FadeOut(*lines, eqbdiff1[0], eqbdiff2[0], eqbdiff3[0], eqbdiffn[0][3:]))
        self.wait(0.1)
        self.play(mh.rtransform(eqbdiffn[0][:3], eqxsum[0][:], eqbdiff3[1], eqxsum[1],
                                eqbdiff1[2][:], eqxsum[2][:2], eqbdiff2[2][:], eqxsum[2][3:5],
                                eqbdiffn[2][-2:], eqxsum[2][-2:], eqbdots[0][:], eqxsum[2][6:9]),
                  mh.rtransform(eqbdiff1[1][0], eqxsum[1][0].copy().set_opacity(0)),
                  mh.rtransform(eqbdiff2[1][0], eqxsum[1][0].copy().set_opacity(0)),
                  mh.rtransform(eqbdiffn[1][0], eqxsum[1][0].copy().set_opacity(0)),
                  FadeIn(eqxsum[2][2], target_position=VGroup(eqbdiff1[2][0], eqbdiff2[1][0])),
                  FadeIn(eqxsum[2][5], target_position=VGroup(eqbdiff2[2][0], eqbdiff3[1][0])),
                  FadeIn(eqxsum[2][-3], target_position=eqbdiffn[2][0].get_top()),
                  FadeOut(eqbdiff3[2], target_position=eqxsum[2][6]),
                  run_time=2
                  )
        self.wait(0.1)
        self.play(FadeOut(eqxsum[2]),
                  FadeIn(eqssum[2]),
                  run_time=1.4)
        self.play(mh.rtransform(eqxsum[:2], eqssum[:2]))
        self.wait(0.1)
        self.play(FadeOut(line4, line5, gp4, gp5, gp6, equ2[0], equ3[0]))
        self.wait(0.1)


        # u_n

        npts0 = len(t0_vals)
        i_un = int(npts0 * 0.94)
        un = t0_vals[i_un]
        bn = b0_vals[i_un]
        pun = ax.coords_to_point(un, 0)
        pbn = ax.coords_to_point(un, bn)
        equn = MathTex(r'u_n', font_size=60).next_to(pun, DOWN, buff=0.15)
        linen = DashedLine(pun, pbn, color=WHITE, stroke_width=5).set_z_index(12)
        eqbn = MathTex(r'B_{u_n}', font_size=60).set_z_index(12).next_to(pbn, UP, buff=0.2)
        eqbnb = eqbn.copy().set_stroke(width=16).set_color(BLACK).set_z_index(11.5)

        eqbtilde = MathTex(r'\tilde B_t', r'=', r'B_{nt}/\sqrt{n}', font_size=80).set_z_index(12)
        eqbtilde.next_to(eqxsum, UP, coor_mask=UP, buff=0.4)

        copy_colors(equ2[0], equn[0])
        VGroup(eqbn[0][0], eqbtilde[0][:2], eqbtilde[2][0]).set_color(self.BCol)
        VGroup(eqbtilde[0][2], eqbtilde[2][2], eqbn[0][1]).set_color(self.tCol)
        VGroup(eqbn[0][2], eqbtilde[2][1], eqbtilde[2][-1]).set_color(self.indexCol)

        np.random.seed(300)
        scale = 3
        nptsn = (npts0-1) * scale + 1
        tn_vals = np.linspace(0, t0_vals[-1] * 3, nptsn)
        s = np.sqrt(t_vals[1])
        bn_vals = np.concatenate((b0_vals, np.random.normal(scale=s, size=nptsn - npts0).cumsum() + b0_vals[-1]))
        pathn = ax.plot_line_graph(tn_vals, bn_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(10)
        bn_vals /= math.sqrt(scale)
        tn_vals /= scale
        pathn2 = ax.plot_line_graph(tn_vals, bn_vals, add_vertex_dots=False, stroke_color=YELLOW, stroke_width=4).set_z_index(10)

        un2 = un/scale
        bn2 = bn / math.sqrt(scale)
        pun2 = ax.coords_to_point(un2, 0)
        pbn2 = ax.coords_to_point(un2, bn2)
        equn2 = MathTex(r'\frac{u_n}{n}', font_size=60).next_to(pun2, DOWN, buff=0.15).set_z_index(12)
        equn2[0][2].next_to(equn2[0][1], DOWN, coor_mask=UP, buff=0.1)
        equn2[0][3].next_to(equn2[0][2], DOWN, coor_mask=UP, buff=0.1)
        linen2 = DashedLine(pun2, pbn2, color=WHITE, stroke_width=5).set_z_index(12)
        eqbn2 = MathTex(r'\tilde B_{\frac{u_n}{n}}', font_size=60).set_z_index(12).next_to(pbn2, UP, buff=0.2)
        eqbn2.shift(DOWN*0.2+RIGHT*0.1)
        eqbn2b = eqbn2.copy().set_stroke(width=16).set_color(BLACK).set_z_index(11.5)

        eqbn2[0][:2].set_color(self.BCol)
        VGroup(equn2[0][0], eqbn2[0][2]).set_color(self.tCol)
        VGroup(equn2[0][1], equn2[0][-1], eqbn2[0][3], eqbn2[0][-1]).set_color(self.indexCol)

        self.play(FadeIn(equn),
                  Succession(Wait(0.5), Create(linen, rate_func=linear, run_time=0.3)),
                  Succession(Wait(0.6), FadeIn(eqbn, eqbnb))
                  )
        self.wait(0.1)
        gp_1 = VGroup(eqssum.copy(), eqbtilde).move_to(eqssum, coor_mask=UP)
        self.play(mh.transform(eqssum, gp_1[0]),
                  Succession(Wait(0.3), FadeIn(eqbtilde)))
        self.wait(0.1)
        self.add(pathn, mask)
        self.remove(path1_1, path2_1, path3_1)
        self.play(mh.transform(pathn, pathn2),
                  mh.stretch_replace(linen, linen2),
                  mh.rtransform(equn[0][:], equn2[0][:2]),
                  FadeIn(equn2[0][2:], shift=mh.diff(equn[0][:], equn2[0][:2])),
                  mh.rtransform(eqbn[0][:], eqbn2[0][1:4],
                                eqbnb[0][:], eqbn2b[0][1:4]),
                  FadeIn(eqbn2[0][0], eqbn2b[0][0], shift=mh.diff(eqbn[0][0], eqbn2[0][1])),
                  FadeIn(eqbn2[0][-2:], eqbn2b[0][-2:], shift=mh.diff(eqbn[0][2], eqbn2[0][3])),
                  run_time=3.)
        self.remove(mask)
        self.wait(0.1)

        # final rearrangement
        eqbtilde2 = MathTex(r'B_t', r'/\sqrt{n}', r'=', r'\tilde B_{\frac tn}', font_size=80).set_z_index(12)
        mh.align_sub(eqbtilde2, eqbtilde2[2], eqbtilde[1]).move_to(eqbtilde, coor_mask=RIGHT)
        eqssum2 = MathTex(r'S_n', r'/\sqrt{n}', r'=', r'B_{u_n}', r'/\sqrt{n}', font_size=80).set_z_index(12)
        mh.align_sub(eqssum2, eqssum2[2], eqssum[1]).move_to(eqbtilde2, coor_mask=RIGHT)
        eqssum3 = MathTex(r'S_n', r'/\sqrt{n}', r'=', r'\tilde B_{\frac{u_n}n}', font_size=80).set_z_index(12)
        mh.align_sub(eqssum3, eqssum3[2], eqssum2[2])
        eqssum4 = MathTex(r'\frac{S_n}{\sqrt{n} }', r'=', r'\tilde B_{\frac{u_n}n}', font_size=100).set_z_index(12)
        eqssum4.move_to(mh.pos(DOWN*0.5+RIGHT*0.05))

        VGroup(eqbtilde2[0][0], eqbtilde2[3][:2], eqssum3[3][:2]).set_color(self.BCol)
        VGroup(eqbtilde2[0][1], eqbtilde2[3][-3], eqssum3[3][2]).set_color(self.tCol)
        VGroup(eqbtilde2[1][-1], eqbtilde2[3][-1], eqssum3[3][3], eqssum3[3][5]).set_color(self.indexCol)
        copy_colors(eqssum[2], eqssum[0], eqbtilde2[1], eqbtilde2[1],
                    eqssum2[0], eqssum2[3], eqssum2[1], eqssum2[-1])
        copy_colors(*eqssum2[:2], *eqssum3[:2])
        copy_colors(eqssum3[0], eqssum3[1], eqssum3[3], eqssum4[0][:2], eqssum4[0][2:], eqssum4[2])

        self.play(mh.rtransform(eqbtilde[1], eqbtilde2[2], eqbtilde[0][:], eqbtilde2[3][:3],
                                eqbtilde[2][0], eqbtilde2[0][0], eqbtilde[2][2], eqbtilde2[0][1],
                                eqbtilde[2][1], eqbtilde2[3][-1], eqbtilde[2][3:], eqbtilde2[1][:]),
                  FadeIn(eqbtilde2[3][-2], shift=mh.diff(eqbtilde[0][2], eqbtilde2[3][2])),
                  run_time=1.7)
        self.wait(0.1)
        self.play(mh.rtransform(eqssum[0], eqssum2[3], eqssum[1], eqssum2[2], eqssum[2], eqssum2[0],
                                run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eqssum2[1], eqssum2[-1])))
        self.wait(0.1)
        eqtmp = eqssum3[3][2].copy()
        self.play(mh.rtransform(eqssum2[:3], eqssum3[:3],
                                eqbtilde2[3][:2], eqssum3[3][:2],
                                eqssum2[3][1:3], eqssum3[3][2:4],
                                eqbtilde2[3][3:], eqssum3[3][4:]),
                  mh.fade_replace(eqbtilde2[3][2], eqtmp),
                  FadeOut(eqbtilde2[:3], eqssum2[4]),
                  run_time=1.4)
        self.remove(eqssum2[3][0], eqtmp)
        self.wait(0.1)
        source, target = (eqssum3[1][0], eqssum4[0][2])
        target2 = Line(target.get_left(), target.get_right(), stroke_width=5, stroke_color=WHITE).set_z_index(12)
        source2 = Line(source.get_corner(DL), source.get_corner(UR), stroke_width=5, stroke_color=WHITE).set_z_index(12)
        self.remove(source)
        self.play(mh.rtransform(eqssum3[0][:], eqssum4[0][:2], eqssum3[1][1:], eqssum4[0][3:],
                                eqssum3[2], eqssum4[1], eqssum3[3], eqssum4[2], source2, target2),
                  run_time=1.4)
        self.remove(target2)
        self.add(target)
        self.wait(0.1)

        # LLN

        self.play(equnsum.animate.scale(1/.6, about_point=equnsum.get_corner(UL)))
        self.wait(0.1)

        eqtmean = MathTex(r'\mathbb E[t_k]', r'=', r'\sigma^2', font_size=80).set_z_index(12)
        eqtmean.next_to(equnsum, DR, buff=0.3).shift(LEFT*1)
        eqlln = MathTex(r'\frac{u_n}{n}', r'\to', r'\sigma^2', font_size=80).set_z_index(12)
        mh.align_sub(eqlln, eqlln[1], equnsum[1]).align_to(equnsum, LEFT).shift(RIGHT)
        eqlln[0][-1].shift(UP*0.2)
        eqlln[0][:2].shift(DOWN*0.1)
        eqslim = MathTex(r'\frac{S_n}{\sqrt{n} }', r'=', r'\tilde B_{\frac{u_n}{n} }', r'\to', r'\tilde B_{\sigma^2}',
                         font_size=100).set_z_index(12)
        mh.align_sub(eqslim, eqslim[1], eqssum4[1]).shift(DOWN*0.5+LEFT*2)
        eqslim2 = MathTex(r'\frac{S_n}{\sqrt{n} }', r'\to', r'\tilde B_{\sigma^2}',
                          r'\sim', r'N(0,\sigma^2)',
                         font_size=100).set_z_index(12)
        mh.align_sub(eqslim2, eqslim2[0], eqslim[0])

        VGroup(eqtmean[0][0], eqslim2[4][0]).set_color(self.opCol)
        VGroup(eqtmean[0][2], eqlln[0][0]).set_color(self.tCol)
        VGroup(eqtmean[0][3], eqlln[0][1], eqlln[0][3]).set_color(self.indexCol)
        VGroup(eqtmean[2][0], eqlln[2][0], eqslim[4][2], eqslim2[4][4]).set_color(self.paramCol)
        VGroup(eqtmean[2][1], eqlln[2][1], eqslim[4][-1], eqslim2[4][5]).set_color(self.exponentCol)
        eqslim[4][:2].set_color(self.BCol)
        eqslim2[4][2].set_color(self.numCol)
        copy_colors(*eqssum4, *eqslim[:3])
        copy_colors(eqslim[0], eqslim[4], eqslim2[0], eqslim2[2])

        self.play(FadeIn(eqtmean))
        self.wait(0.1)
        self.play(mh.rtransform(equnsum[0][:], eqlln[0][:2]),
                  mh.fade_replace(equnsum[1], eqlln[1]),
                  FadeIn(eqlln[0][2:], shift=mh.diff(equnsum[0][:], eqlln[0][:2])*RIGHT),
                  FadeOut(equnsum[2]),
                  FadeIn(eqlln[2]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eqssum4[:], eqslim[:3]),
                  Succession(Wait(0.5), FadeIn(eqslim[3:], run_time=1.4)))
        self.wait(0.1)
        self.play(mh.rtransform(eqslim[0], eqslim2[0], eqslim[3:5], eqslim2[1:3], run_time=1.6),
                  FadeOut(eqslim[1:3]),
                  Succession(Wait(0.6), FadeIn(eqslim2[3:])))


        self.wait()

class BrownianScaled(BMDef):
    bgCol = GREY
    def construct(self):
        MathTex.set_default(font_size=100, stroke_width=2)
        eq1 = MathTex(r'\tilde B_t', r'=', r'B_{nt}', r'/\sqrt{n}').set_z_index(2)

        eq2 = MathTex(r'B_t', r'=', r'\sqrt{n}', r'\tilde B_{\frac tn}').set_z_index(2)
        eq3 = MathTex(r'S_n/\sqrt{n}', r'\to', r'\tilde B_{\sigma^2}').set_z_index(2)
        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)

        gp = VGroup(eq1, eq2)
        box = SurroundingRectangle(gp, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   corner_radius=0.2, buff=0.2)
        VGroup(box, eq1, eq2).to_edge(DOWN, buff=0.15)

        eq3.next_to(eq2, UP, buff=0.3)
        box2 = SurroundingRectangle(VGroup(gp, eq3), stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   corner_radius=0.2, buff=0.2)

        VGroup(eq1[0][:2], eq1[2][0], eq2[0][0], eq2[3][:2], eq3[2][:2]).set_color(self.BCol)
        VGroup(eq1[0][2], eq1[2][2], eq2[0][1], eq2[3][2]).set_color(self.tCol)
        VGroup(eq1[2][1], eq1[3][-1], eq2[2][-1], eq2[3][-1], eq3[0][1], eq3[0][-1], eq3[2][-1]).set_color(self.indexCol)
        VGroup(eq3[0][0]).set_color(self.XCol)
        eq3[2][2].set_color(self.paramCol)

        self.add(box, eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[2][0], eq2[0][0], eq1[2][2], eq2[0][1],
                                eq1[1], eq2[1], eq1[3][1:], eq2[2][:],
                                eq1[0][:], eq2[3][:3], eq1[2][1], eq2[3][-1]),
                  FadeOut(eq1[3][0], target_position=eq2[2][0]),
                  FadeIn(eq2[3][-2], shift=mh.diff(eq1[0][2], eq2[3][-3])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(box, box2),
                  Succession(Wait(0.5), FadeIn(eq3)))


        self.wait()

class Skorokhod(CLT2):
    bgCol = BLACK
    def write(self, *txt, run_time=2.):
        txts = []
        for elem in txt:
            txts += list(elem)
        txt1 = VGroup(*txts)
        tval = ValueTracker(.0)
        n = len(txt1)
        def txt_write():
            t = tval.get_value()
            i = min(math.ceil(t*n), n)
            res = txt1[:i].copy()
            if i > 0:
                res[-1].set_opacity(t*n-i+1)
            return res

        txt2 = always_redraw(txt_write)
        self.add(txt2)
        self.play(tval.animate.set_value(1.), rate_func=linear, run_time=run_time)
        self.remove(txt2)
        self.add(*txt)

    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        col = ManimColor('#ffac2b')
        txt1 = Tex(r'\sf Skorokhod stopping problem', font_size=90, color=col, stroke_width=3)[0]
        txt1.to_edge(UP, buff=0.6)

        MathTex.set_default(font_size=70, stroke_width=1.5, color=self.textCol)
        txt2 = Tex(r'\sf For a random variable ', r'$X$', r' and standard ')
        txt3 = Tex(r'\sf Brownian motion ', r'$B$', r', find a stopping time ', r'$t$')
        txt4 = Tex(r'\sf with finite mean such that ', r'$B_t$', r' has the same')
        txt5 = Tex(r'\sf distribution as ', r'$X$', r'.')
        VGroup(txt2[1], txt5[1]).set_color(GREEN)
        VGroup(txt3[1], txt4[1][0]).set_color(self.BCol)
        VGroup(txt3[-1], txt4[1][1]).set_color(self.tCol)
        gp = VGroup(txt2, txt3, txt4, txt5).arrange_in_grid(cols=1, buff=0.5, cell_alignment=LEFT).next_to(txt1, DOWN, buff=0.6).to_edge(LEFT, buff=0.4)
        txt2.shift(RIGHT)

        self.write(txt1, run_time=2)
        self.wait(0.1)
        self.write(*txt2, *txt3, *txt4, *txt5, run_time=4)

        MathTex.set_default(font_size=80, stroke_width=1.5, color=WHITE)
        eq1 = MathTex(r'\mathbb E[X]', r'=', r'0')
        eq2 = MathTex(r'\mathbb E[X^2]', r'<', r'\infty')
        eq3 = MathTex(r'\mathbb E[t]', r'=', r'\mathbb E[X^2]', r'=', r'\sigma^2')
        eq4 = MathTex(r'\mathbb E[B_t^2-t]', r'=', r'0')
        eq5 = MathTex(r'\mathbb E[t]', r'=', r'\mathbb E[B_t^2]')
        eq6 = MathTex(r'=', r'\mathbb E[X^2]')
        eq7 = MathTex(r'=', r'\sigma^2')

        VGroup(eq1[0][0], eq2[0][0], eq3[0][0], eq3[2][0], eq4[0][0],
               eq5[0][0], eq5[2][0], eq6[1][0]).set_color(self.opCol)
        VGroup(eq1[2], eq2[2], eq4[2]).set_color(self.numCol)
        VGroup(eq1[0][2], eq2[0][2], eq3[2][2], eq6[1][2]).set_color(GREEN)
        VGroup(eq2[0][3], eq3[2][3], eq3[4][1], eq4[0][3], eq5[2][3], eq6[1][3], eq7[1][1]).set_color(self.exponentCol)
        VGroup(eq3[0][2], eq4[0][4], eq4[0][6], eq5[0][2], eq5[2][4]).set_color(self.tCol)
        VGroup(eq3[4][0], eq7[1][0]).set_color(self.paramCol)
        VGroup(eq4[0][2], eq5[2][2]).set_color(self.BCol)

        eq1.move_to((gp.get_bottom()+mh.pos(DOWN))/2, coor_mask=UP)
        mh.align_sub(eq2, eq2[0][0], eq1[0][0])
        eq2.next_to(eq1, RIGHT, buff=1)
        gp2 = VGroup(eq1, eq2).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq3, eq3[1], eq1[1], coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq1[1], coor_mask=UP)
        mh.align_sub(eq5, eq5[1], eq4[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[0], eq5[1])
        mh.align_sub(eq7, eq7[0], eq5[1])

        self.play(FadeIn(eq1, eq2))
        circ = mh.circle_eq(gp2).set_z_index(4)
        txt6 = Tex(r'\sf\emph{necessary and sufficient}', color=RED, font_size=80)
        txt6.move_to(circ.get_corner(UR)).shift(LEFT*1.5+UP*0.15)
        txt6 = eq_shadow(txt6, fg_z_index=6, bg_z_index=5)
        self.play(Create(circ))
        self.play(FadeIn(txt6))
        self.wait(0.1)
        self.play(FadeOut(circ, eq1, eq2, txt6))
        self.play(FadeIn(eq3[:3]))
        self.play(FadeIn(eq3[3:]))
        self.wait(0.1)
        self.play(FadeOut(eq3), FadeIn(eq4[0][2:-1]))
        self.wait(0.1)
        self.play(FadeIn(eq4[0][:2], eq4[0][-1], eq4[1:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][:5], eq5[2][:5], eq4[0][-1], eq5[2][-1],
                                eq4[1], eq5[1], eq4[0][6], eq5[0][2]),
                  mh.rtransform(eq4[0][:2].copy(), eq5[0][:2], eq4[0][-1].copy(), eq5[0][-1]),
                  FadeOut(eq4[2]),
                  FadeOut(eq4[0][5]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq5[2][:2], eq6[1][:2], eq5[2][3], eq6[1][3],
                                eq5[2][-1], eq6[1][-1]),
                  mh.fade_replace(eq5[2][2], eq6[1][2]),
                  FadeOut(eq5[2][4], shift=mh.diff(eq5[2][2], eq6[1][2])))
        self.wait(0.1)
        eq7_1 = eq7[1].copy().move_to(eq6[1][2:4], coor_mask=RIGHT)
        eq7_2 = VGroup(*eq5[:2].copy(), eq7[1]).move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq6[1][3], eq7_1[1]),
                  mh.fade_replace(eq6[1][2], eq7_1[0], coor_mask=RIGHT),
                  FadeOut(eq6[1][:2]),
                  FadeOut(eq6[1][-1]))
        self.play(mh.rtransform(eq5[:2], eq7_2[:2], eq7_1, eq7_2[2]))

        self.wait()

class NormDensity(CLT2):
    bgCol = BLACK

    def construct(self):
        xmax = 2.
        xlen = 6
        ylen = 2
        ax = Axes(x_range=[-xmax, xmax * 1.1], y_range=[0., 1.2], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)
        eq1 = MathTex(r'p_{B_t}(x)', r'=', r'\frac{1}{\sqrt{2\pi t} }', r'\,e^{-\frac{x^2}{2t}}', stroke_width=1.5, font_size=50).set_z_index(6)
        eq1[0][1].set_color(self.BCol)
        VGroup(eq1[0][2], eq1[3][6], eq1[2][-1]).set_color(self.tCol)
        VGroup(eq1[0][-2], eq1[3][2]).set_color(ManimColor(self.XCol.to_rgb()*1.5))
        eq1[3][3].set_color(self.exponentCol)
        VGroup(eq1[3][5], eq1[2][-3:-1], eq1[2][0]).set_color(self.numCol)
        VGroup(eq1[0][0], eq1[3][0]).set_color(self.opCol)


        self.add(ax)
        self.wait(0.1)
        def f(x):
            return math.exp(-x*x/2)

        path = ax.plot(f, (-xmax, xmax), stroke_color=BLUE, stroke_width=6).set_z_index(3)
        area = ax.get_area(path, color=ManimColor(BLUE.to_rgb()*0.5), opacity=1, x_range=(-xmax, xmax),
                           stroke_width=0, stroke_opacity=0).set_z_index(1)

        p = ax.coords_to_point(0, 0.5)
        mh.align_sub(eq1, eq1[1], p)
        eq1.move_to(p)
        eq1 = eq_shadow(eq1, bg_z_index=5, fg_z_index=6)

        self.play(Create(path, rate_func=linear),
                  Succession(Wait(0.5), FadeIn(area, rate_func=linear)),
                  Succession(Wait(0.7), FadeIn(eq1)),
                  )
        self.wait()

if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        Skorokhod().render()