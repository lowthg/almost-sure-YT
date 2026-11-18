import math

import numpy as np
from manim import *
import sys
from manim import ManimColor
from matplotlib.pyplot import yscale, xscale
from scipy.special import expi

import mpmath as mp

sys.path.append('../../')
import manimhelper as mh
from common.wigner import *

col_txt = ManimColor( r'#FFAC2B')
col_pi = col_special * 0.5 + ORANGE * 0.5
col_int = col_op

def rate_func_log(x):
    """
    x1 * exp(at), x1 * exp(a) = x2
    """
    a = np.log(x)
    b = x - 1
    return lambda t: (np.exp(a*t) - 1) / b

class IntroSum(Scene):
    bgcol = GREY
    trcol = BLACK
    def __init__(self, *args, **kwargs):
        config.background_color = self.trcol if config.transparent else self.bgcol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        self.do_anim()

    def get_series(self):
        eq1 = MathTex(r'\sum_{n=1}^\infty', r'\frac{1}{n^3\sin^2n}')
        VGroup(eq1[0][1], eq1[1][1]).set_color(col_op)
        VGroup(eq1[1][-5:-2]).set_color(col_trig)
        VGroup(eq1[0][2], eq1[1][-1], eq1[1][-7]).set_color(col_x)
        VGroup(eq1[0][-1], eq1[1][0], eq1[1][3], eq1[1][-2]).set_color(col_num)
        VGroup(eq1[0][0]).set_color(col_special)
        VGroup(eq1[0][3]).set_color(WHITE*0.85)
        return eq1

    def do_anim(self, just_eq=False):
        MathTex.set_default(font_size=100, stroke_width=2)
        eq1 = self.get_series()
        # eq1 = MathTex(r'\sum_{n=1}^\infty', r'\frac{1}{n^3\sin^2n}')
        args = [r'\frac1{\sin^21}', r'+']
        for n in range(2, 6):
            args += [r'\frac1{{ {}^3\sin^2{} }}'.format(n, n), r'+']
        eq2 = MathTex(*args, stroke_width=1.5)
        eq3 = MathTex(r'\sum_{n=1}^\infty', r'\frac{1}{n^3\sin^2n}', r'<', r'\infty\,', r'???')
        eq4 = MathTex(r'\sum_{n=1}^\infty', r'\frac{1}{n^3\sin^2n}', r'=', r'\infty\,', r'???')

        mh.rtransform.copy_colors = True
        VGroup(eq3[3]).set_color(col_special)
        VGroup(eq3[-1]).set_color(RED)

        for i in [0,2,4,6, 8]:
            eq2[i][2].set_color(col_x)
            eq2[i][3].set_color(col_num)
            VGroup(eq2[i][0], eq2[i][-2]).set_color(col_num)
            VGroup(eq2[i][1]).set_color(col_op)
            VGroup(eq2[i][-5:-2]).set_color(col_trig)
            eq2[i][-1].set_color(col_x)

        eq1 = mh.eq_shadow(eq1, bg_stroke_width=15)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=13)
        eq3 = mh.eq_shadow(eq3, bg_stroke_width=13)
        eq4 = mh.eq_shadow(eq4, bg_stroke_width=13)

        eq1.to_edge(DOWN, buff=0.4)
        mh.align_sub(eq2, eq2[2][1], eq1[1][1]).to_edge(LEFT, buff=3)
        mh.align_sub(eq3, eq3[1], eq1[1])
        mh.align_sub(eq4, eq4[1], eq1[1])

        if just_eq:
            return eq1

        # box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.5,
        #                             corner_radius=0.2, buff=0.3)

        self.add(eq2)
        run_time=4.
        slowdown = 1.
        rate_func = mh.rate_func_quad(0., slowdown / run_time)
        self.play(eq2.animate(run_time=run_time, rate_func=rate_func).to_edge(RIGHT))

        self.play(AnimationGroup(*[AnimationGroup(mh.rtransform(eq2[i][:2], eq1[1][:2], eq2[i][3:8], eq1[1][3:8]),
                  mh.fade_replace(eq2[i][2], eq1[1][2].copy(), coor_mask=RIGHT),
                  mh.fade_replace(eq2[i][-1], eq1[1][-1].copy(), coor_mask=RIGHT)) for i in (4,6,8)],
                  run_time=2),
                  FadeOut(eq2[:4], eq2[5], eq2[7], eq2[-1]),
                  Succession(Wait(1.2), FadeIn(eq1[0]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:2], eq3[:2]),
                  FadeIn(eq3[2:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:-3], eq4[:-3], eq3[-2:], eq4[-2:]),
                  mh.stretch_replace(eq3[-3], eq4[-3]))
        self.wait(0.1)
        self.play(FadeOut(eq4[-3:]))

        self.wait()

class IntroSum2(IntroSum):
    def construct(self):
        eq1 = self.do_anim(just_eq=True)
        eq2 = MathTex(r'\sum_{n=1}^\infty', r'\frac{1}{n^3\sin^2n}', stroke_width=1, font_size=50).set_color(BLACK)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=0.1)
        col_txt

        eq2.move_to(eq1)

        self.add(eq1)
        self.play(ReplacementTransform(eq1, eq2))
        self.wait()


class IntroPi(IntroSum):
    def construct(self):
        eq1 = self.do_anim(just_eq=True)
        eq2 = MathTex(r'\pi', font_size=400)

        VGroup(eq2).set_color(col_pi)

        eq2 = mh.eq_shadow(eq2, bg_stroke_width=15)
        eq2.move_to(eq1)
        self.add(eq2)
        self.wait()

class SumLook(IntroSum):
    def construct(self):
        xmax = 20.

        eq1 = self.do_anim(just_eq=True)
        eq2 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3}', r'=', r'\zeta(3)', r'\approx', r'1.202\ldots')
        eq3 = MathTex(r'\sin x', font_size=60, stroke_width=1.5)
        eq4 = MathTex(r'x', font_size=60, stroke_width=1)

        ax = Axes(x_range=[0, xmax * 1.02], x_length=12, y_length=3, y_range=[-1.1, 1.25], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': True,
            "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
            "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
        }).set_z_index(1)

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq2[3][0]).set_color(col_psi)
        VGroup(eq2[3][2], eq2[-1]).set_color(col_num)
        VGroup(eq4).set_color(col_x)
        ax.to_edge(UP, buff=1.5)
        eq3.move_to(ax.c2p(9.5, 1.1))
        eq4.next_to(ax.x_axis, RIGHT, buff=0.2)

        eq2 = mh.eq_shadow(eq2, bg_stroke_width=13)
        eq3 = mh.eq_shadow(eq3, bg_stroke_width=10)

        yticks1 = mh.get_yticks(ax, [-1, 1], label_color=col_num)
        xticks1 = mh.get_xticks(ax, [np.pi * i for i in range(1,7)],
                                [r'\pi'] + [r'{}\pi'.format(i) for i in range(2,7)], label_color=col_pi).set_z_index(1)
        xticks2 = mh.get_xticks(ax, range(1, int(xmax)), label_color=col_num, dir=UP, font_size=40).set_z_index(1)
        for _ in xticks1:
            _[1] = mh.eq_shadow(_[1], bg_stroke_width=10, bg_z_index=5, fg_z_index=6)
        for _ in xticks2:
            _[1] = mh.eq_shadow(_[1], bg_stroke_width=10, bg_z_index=5, fg_z_index=6)

        box1 = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1)

        mh.align_sub(eq2, eq2[0][1], eq1[0][1], coor_mask=UP).to_edge(RIGHT, buff=0.2)
        # mh.align_sub(eq2, eq2[:4], ORIGIN, coor_mask=RIGHT)

        self.add(eq1)
        eq1_1 = eq1.copy()
        self.play(FadeOut(eq1[1][-5:]),
                  eq1[1][-7:-5].animate.move_to(eq1[1][0], coor_mask=RIGHT))
        self.wait(0.1)
        eq2_1 = eq2[:-2].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq1[0], eq2_1[0], eq1[1][:2], eq2_1[1][:2], eq1[1][-7:-5], eq2_1[1][-2:]),
                  Succession(Wait(0.5), FadeIn(eq2_1[2:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1, eq2[:-2]),
                  Succession(Wait(0.5), FadeIn(eq2[-2:])))
        self.wait(0.1)
        eq1 = eq1_1
        self.play(FadeOut(eq2[2:]),
                  mh.rtransform(eq2[0], eq1[0], eq2[1][:4], eq1[1][:4]),
                  Succession(Wait(0.5), FadeIn(eq1[1][4:])))
        self.wait(0.1)
        self.play(FadeIn(ax, yticks1, box1, eq4, rate_func=linear, run_time=1),
                  eq1.animate.scale(0.8, about_edge=DOWN))

        plt = ax.plot(np.sin, (0, xmax), stroke_width=8, stroke_color=BLUE).set_z_index(4)

        self.play(Create(plt, rate_func=linear, run_time=2),
                  # Succession(Wait(0.5), FadeIn(eq3)),
                  AnimationGroup(
                      mh.rtransform(eq1[1][4:7].copy(), eq3[0][:3]),
                      mh.stretch_replace(eq1[1][-1].copy(), eq3[0][-1]),
                      run_time=1.5)
                  )
        self.wait(0.1)
        self.play(FadeIn(xticks1))
        self.wait(0.1)
        self.play(FadeIn(xticks2))

        self.wait()

class SumSeries(IntroSum):
    bgcol = BLACK
    def construct(self):
        eq0 = self.do_anim(just_eq=True)
        mh.rtransform.copy_colors = False
        eq0.scale(0.8, about_edge=DOWN)

        ax1 = Axes(x_range=[0, 1.05], x_length=12, y_length=3, y_range=[0, 1.1], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': True,
            "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
            "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
        }).set_z_index(1)
        ax2 = Axes(x_range=[0, 1.05], x_length=12, y_length=3, y_range=[0, 1.1], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': True,
            "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
            "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
        }).set_z_index(1)

        ax2.shift(ax1.c2p(0,0)-ax2.c2p(0,0))
        ax2.next_to(ax1.y_axis, DOWN, buff=0.2, coor_mask=UP)

        MathTex.set_default(stroke_width=1.5, font_size=50)
        eq1 = MathTex(r'\frac1{n^3\sin^2 n}')
        eq2 = MathTex(r'\sum_{n\le N}', r'\frac1{n^3\sin^2 n}')
        eq3 = MathTex(r'N', r'n', font_size=60, color=col_x).set_z_index(20)

        VGroup(eq1[0][0], eq1[0][3], eq1[0][-2]).set_color(col_num)
        VGroup(eq1[0][1], eq2[0][0]).set_color(col_op)
        VGroup(eq1[0][2], eq1[0][-1], eq2[0][1], eq2[0][-1]).set_color(col_x)
        VGroup(eq1[0][-5:-2]).set_color(col_trig)

        n = 1800 # by this point, converged to with 1/2 pixel (0.03) and correct to 2dp = 30.31
        xvec = np.linspace(1., n, n)
        svec = np.sin(xvec)
        termvec = 1. / (xvec ** 3 * svec * svec)
        sumvec = np.cumsum(termvec)

        n1 = 10
        xscale1 = 1/(n1+1)
        xtick_arr1 = list(range(0, n1+1))
        xticks1 = mh.get_xticks(ax2, xtick_arr1, label_color=col_num, scalex=xscale1)
        VGroup(ax1, ax2, xticks1).move_to(ORIGIN)
        xticks_old = VGroup(*xticks1[1:])
        origin1 = ax1.c2p(0,0)
        origin2 = ax2.c2p(0,0)
        xstep = ax1.c2p(1,0) - origin1
        ystep2 = ax2.c2p(0,1) - origin2

        eq3[0].next_to(ax1.x_axis.get_right(), RIGHT, buff=0.2)
        eq3[1].next_to(ax2.x_axis.get_right(), RIGHT, buff=0.2)

        box1 = Rectangle(width=3, height=config.frame_height, stroke_width=0, stroke_opacity=0, fill_opacity=1,
                         fill_color=BLACK).set_z_index(9).next_to(ax2.c2p(1,0), UR, buff=0).shift(DOWN*0.2)
        box2 = Rectangle(width=3, height=2, stroke_width=0, background_stroke_width=0, fill_opacity=1,
                         fill_color=BLACK).set_z_index(9).next_to(ax2.x_axis.get_right(), DR, buff=0)

        # inter1 = Intersection(ax1.x_axis.tip, box1).set_fill(color=WHITE, opacity=1).set_stroke(color=WHITE).set_z_index(10)
        ax1.x_axis.tip.set_z_index(10)
        ax2.x_axis.tip.set_z_index(10)
        line1 = Line(ax1.c2p(0.999,0), ax1.c2p(1.05,0), stroke_color=WHITE, stroke_width=6).set_z_index(10)
        line2 = Line(ax2.c2p(0.999,0), ax2.c2p(1.05,0), stroke_color=WHITE, stroke_width=6).set_z_index(10)
        box1 = VGroup(box1, line1, line2, box2)
        print('sl', ax1.x_axis.submobjects)

        eq1.move_to(ax2.c2p(0.6, 0.5))
        eq2.move_to(ax1.c2p(0, 0.4)).align_to(eq1, RIGHT)

        eq1 = mh.eq_shadow(eq1, bg_stroke_width=10, bg_z_index=9.9, fg_z_index=10)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=10, bg_z_index=9.9, fg_z_index=10)

        xticks1_1 = VGroup(_[0].copy() for _ in xticks_old).shift(origin1-origin2)

        ymax1_1 = sumvec[n1-1]
        print('ymax1:', ymax1_1)
        yscale1_1 = 1./ymax1_1
        yscale1_2 = yscale1_1#1./ ymax1
        eps = 0.006
        yticks_arr_old = yticks_arr_old_2 = [1,2,3]
        yticks1_2 = mh.get_yticks(ax2, [0] + yticks_arr_old_2, label_color=col_num, scaley=yscale1_1)
        yticks1_1 = mh.get_yticks(ax1, [0] + yticks_arr_old, label_color=col_num, scaley=yscale1_1)

        print(termvec[:n1])

        bluefill = BLUE*0.5
        orangefill = ORANGE*0.6

        recs1 = [Rectangle(width=xstep[0]*xscale1, height=termvec[i] * ystep2[1]*yscale1_2,
                          stroke_width=0, stroke_opacity=0,
                          fill_color=orangefill, fill_opacity=1).set_z_index(2)
                .next_to(ax2.c2p((i+1)*xscale1,eps), UR, buff=0) for i in range(n1)]
        recs2 = [recs1[i].copy().set_fill(bluefill)
                 .next_to(ax1.c2p((i+1)*xscale1, sumvec[i]*yscale1_1), DR, buff=0) for i in range(n1)]

        # transform equation to plots
        self.add(eq0)
        yticks1_2_old = yticks1_2[1:]
        self.play(mh.rtransform(eq0[1], eq1[0]),
                  FadeOut(eq0[0], rate_func=linear),
                  FadeIn(ax1, ax2, xticks1[0], xticks_old, xticks1_1, yticks1_2[0], yticks1_2_old, eq3, rate_func=linear),
                  run_time=1.5)
        self.add(box1)
        self.wait(0.1)

        # fill in plots
        xvals1 = np.concatenate([[i, i] for i in range(1, n1+1)] + [[n1+1]], dtype=float)
        yvals1 = np.concatenate([[0]]+[[termvec[i], termvec[i]] for i in range(n1)], dtype=float)
        xvals1_1 = np.append(xvals1, xvals1[-1])
        yvals1_1 = np.append(yvals1, 0)

        plt1 = ax2.plot_line_graph(xvals1*xscale1, yvals1*yscale1_2+eps, add_vertex_dots=False, stroke_width=4,
                                   stroke_color=ORANGE).set_z_index(6)
        area1 = ax2.plot_line_graph(xvals1_1*xscale1, yvals1_1*yscale1_2+eps, add_vertex_dots=False, stroke_opacity=0,
                                   fill_color=orangefill, stroke_width=0, fill_opacity=1).set_z_index(5)
        plt2 = ax2.plot_line_graph(xvals1*xscale1, yvals1 * 0 + eps, add_vertex_dots=False, stroke_width=0,
                                   stroke_color=ORANGE).set_z_index(6)
        area2 = ax2.plot_line_graph(xvals1_1*xscale1, yvals1_1 * 0 + eps, add_vertex_dots=False, stroke_opacity=0,
                                   fill_color=orangefill, stroke_width=0, fill_opacity=1).set_z_index(5)

        yvals2 = np.concatenate([[0]]+[[sumvec[i], sumvec[i]] for i in range(n1)], dtype=float)
        yvals2_1 = np.append(yvals2, 0)
        plt3 = ax1.plot_line_graph(xvals1*xscale1, yvals2*yscale1_1, add_vertex_dots=False, stroke_width=6,
                                   stroke_color=BLUE).set_z_index(6)
        area3 = ax1.plot_line_graph(xvals1_1*xscale1, yvals2_1 * yscale1_1, add_vertex_dots=False, stroke_opacity=0,
                                   fill_color=bluefill, stroke_width=0, fill_opacity=1).set_z_index(5)

        self.play(mh.rtransform(plt2, plt1, area2, area1))
        self.wait(0.1)

        yticks1_1_old = yticks1_1[1:]
        self.play(AnimationGroup(mh.rtransform(VGroup(*recs1), VGroup(*recs2), plt1.copy(), plt3),
                  mh.rtransform(eq1[0].copy(), eq2[1], copy_colors=True),
                  FadeIn(eq2[0], shift=mh.diff(eq1[0], eq2[1])),
                  run_time=2),
                  Succession(Wait(1.2), FadeIn(yticks1_1[0], yticks1_1_old)))
        self.wait(0.1)

        total = DecimalNumber(sumvec[n1-1], num_decimal_places=2, color=BLUE, font_size=50, stroke_width=1.5).set_z_index(20)
        total.next_to(ax1.c2p(1., sumvec[n1-1]*yscale1_1), RIGHT, buff=0.2)

        # fade in cumulative fill
        self.play(FadeIn(area3, total, rate_func=linear))
        self.remove(*recs2)

        self.wait(0.1)

        xticks1_arr = xtick_arr1[1:]

        def rate_func0(a):
            def rate_func(t):
                return (a - math.pow(a, 1-t)) / (a-1)
            return rate_func

        rate_func10 = rate_func0(10.)
        rate_func38 = rate_func0(100/38)
        rate_func2 = mh.rate_func_quad(0., 0.3)
        rate_func3 = mh.rate_func_quad(1., 0.)
        total_old = total

        for n2, yscale2_2, xticks2_arr, run_time, rate_func, yticks2_1_arr, yticks2_2_arr in [
            (20, yscale1_2, list(range(2,22,2)), 2, mh.rate_func_quad(0,0.25), None, None),
            (30, yscale1_2, list(range(2, 32, 2)), 2, mh.rate_func_quad(0.2, 0.25), [1, 2, 3, 4], None),
            (100, yscale1_2, list(range(10, 102, 10)), 2, mh.rate_func_quad(0.2, 0.25), None, None),
            (200, yscale1_2, list(range(20, 202, 20)), math.log(2)*4, mh.rate_func_quad(0.2,0.2), None, None),
            (320, yscale1_2, list(range(40, 322, 40)), math.log(1.6)*4, mh.rate_func_quad(0.2, 0.2), None, None),
            (380, 1/31, list(range(40, 382, 40)), 2, smooth, [10, 20, 30], [10,20,30]),
            (1000, 1/31, list(range(200, 1001, 200)), 4 * math.log(100/38)/math.log(10), lambda t: rate_func38(rate_func3(t)), None, None),
            (10_000,        1 / 31, [1000,2000,4000,6000,10000], 2, rate_func10, None, None),
            (100_000,       1 / 31, None, 2, rate_func10, None, None),
            (1_000_000,     1 / 31, None, 2, rate_func10, None, None),
            (10_000_000,    1 / 31, None, 2, rate_func10, None, None),
            (100_000_000,   1 / 31, None, 2, rate_func10, None, None),
            (1_000_000_000, 1 / 31, None, 2.6, lambda t: rate_func10(rate_func2(t)), None, None),
        ]:
            # take it to 20
            xscale2 = 1/(n2+1)
            if n2 > 500e6:
                xscale2 *= 0.998
            yscale2_1 = 1/sumvec[min(n2, n-1)]
            # ymax2_2 = max(termvec[:n2])
            # yscale2_2 = 1 / ymax2_2
            # yscale2_1 = yscale1_1
            # yscale2_2 = yscale1_2
            if n2 > 1000:
                box2.set_opacity(0)

            if xticks2_arr is None:
                xticks2_arr = [n2//10, n2//2, n2]
            xarr_old = [x for x in xticks1_arr if x not in xticks2_arr]
            xarr = xarr_old + xticks2_arr
            xstr = ['{:,}'.format(_).replace(",", r"\,") for _ in xarr]
            xticks2_0 = mh.get_xticks(ax2, xarr, xstr, label_color=col_num, scalex=xscale1)
            xticks2 = mh.get_xticks(ax2, xarr, xstr, label_color=col_num, scalex=xscale2)
            xticks2[:len(xarr_old)].set_opacity(0)
            # if xticks2[-1][1].get_right()[0] > box2.get_left()[0]:
            #     xticks2[-1][1].next_to(box2, LEFT, buff=0, coor_mask=RIGHT)
            # if xticks2_0[len(xarr_old)][1].get_right()[0] > box2.get_left()[0]:
            #     xticks2_0[len(xarr_old)][1].next_to(box2, LEFT, buff=0, coor_mask=RIGHT)

            xticks2_2 = VGroup(*[_[0].copy() for _ in xticks2_0]).shift(origin1 - origin2)
            xticks2_3 = VGroup(*[_[0].copy() for _ in xticks2]).shift(origin1 - origin2)

            if yticks2_1_arr is None:
                yticks2_1_arr = yticks_arr_old
            yarr_new = [y for y in yticks2_1_arr if y not in yticks_arr_old]
            yticks2_1 = mh.get_yticks(ax1, yarr_new + yticks_arr_old, label_color=col_num, scaley=yscale1_1)
            yticks2_2 = mh.get_yticks(ax1, yarr_new + yticks_arr_old, label_color=col_num, scaley=yscale2_1)
            yticks2_1[:len(yarr_new)].set_opacity(0)
            for i, y in enumerate(yticks_arr_old):
                if y not in yticks2_1_arr:
                    yticks2_2[i+len(yarr_new)].set_opacity(0)

            if yticks2_2_arr is None:
                yticks2_2_arr = yticks_arr_old_2
            yarr_new = [y for y in yticks2_2_arr if y not in yticks_arr_old_2]
            yticks2_3 = mh.get_yticks(ax2, yarr_new + yticks_arr_old_2, label_color=col_num, scaley=yscale1_2)
            yticks2_4 = mh.get_yticks(ax2, yarr_new + yticks_arr_old_2, label_color=col_num, scaley=yscale2_2)
            yticks2_3[:len(yarr_new)].set_opacity(0)
            for i, y in enumerate(yticks_arr_old_2):
                if y not in yticks2_2_arr:
                    yticks2_4[i+len(yarr_new)].set_opacity(0)

            ivals = range(min(n2, n))
            xvals = np.concatenate([[i+1, i+1] for i in ivals] + [[n2+1]], dtype=float)
            yvals = np.concatenate([[0]]+[[termvec[i], termvec[i]] for i in ivals], dtype=float)
            xvals_1 = np.append(xvals, xvals[-1])
            yvals_1 = np.append(yvals, 0)
            plt1_1 = ax2.plot_line_graph(xvals*xscale1, yvals*yscale1_2+eps, add_vertex_dots=False, stroke_width=4,
                                       stroke_color=ORANGE).set_z_index(6)
            area1_1 = ax2.plot_line_graph(xvals_1*xscale1, yvals_1*yscale1_2+eps, add_vertex_dots=False, stroke_opacity=0,
                                       fill_color=orangefill, stroke_width=0, fill_opacity=1).set_z_index(5)
            plt4 = ax2.plot_line_graph(xvals*xscale2, yvals*yscale2_2+eps, add_vertex_dots=False, stroke_width=4,
                                       stroke_color=ORANGE).set_z_index(6)
            area4 = ax2.plot_line_graph(xvals_1*xscale2, yvals_1*yscale2_2+eps, add_vertex_dots=False, stroke_opacity=0,
                                       fill_color=orangefill, stroke_width=0, fill_opacity=1).set_z_index(5)

            yvals = np.concatenate([[0]]+[[sumvec[i], sumvec[i]] for i in ivals], dtype=float)
            newtotal = yvals[-1]
            yvals_1 = np.append(yvals, 0)
            plt3_1 = ax1.plot_line_graph(xvals*xscale1, yvals*yscale1_1, add_vertex_dots=False, stroke_width=6,
                                       stroke_color=BLUE).set_z_index(6)
            area3_1 = ax1.plot_line_graph(xvals_1*xscale1, yvals_1 * yscale1_1, add_vertex_dots=False, stroke_opacity=0,
                                       fill_color=bluefill, stroke_width=0, fill_opacity=1).set_z_index(5)
            plt5 = ax1.plot_line_graph(xvals*xscale2, yvals*yscale2_1, add_vertex_dots=False, stroke_width=6,
                                       stroke_color=BLUE).set_z_index(6)
            area5 = ax1.plot_line_graph(xvals_1*xscale2, yvals_1 * yscale2_1, add_vertex_dots=False, stroke_opacity=0,
                                       fill_color=bluefill, stroke_width=0, fill_opacity=1).set_z_index(5)

            self.remove(plt1, area1, xticks1_1, plt3, area3, xticks_old, yticks1_1_old, yticks1_2_old, total_old)
            # xticks1[1:].set_opacity(0)

            valtrack = ValueTracker(0.)
            def get_obj():
                res = total.copy()
                t = valtrack.get_value()
                n3 = int(n1 * n2 / (n2 * (1-t) + n1 * t))
                newtot = sumvec[min(n3, n)-1]
                res.set_value(newtot)
                return res.set_z_index(30)

            total_old = always_redraw(get_obj)
            self.add(total_old)

            self.play(mh.rtransform(plt1_1, plt4, area1_1, area4,
                                    xticks2_0, xticks2, xticks2_2, xticks2_3,
                                    yticks2_1, yticks2_2, yticks2_3, yticks2_4,
                                    plt3_1, plt5, area3_1, area5),
                      *([FadeOut(xticks1[0][1])] if n2 == 100_000 else []),
                      valtrack.animate.set_value(1.),
                      run_time=run_time, rate_func=rate_func)

            xticks1_arr = xticks2_arr.copy()
            xticks_old = xticks2
            xticks1_1 = xticks2_3
            yticks1_1_old = yticks2_2
            yticks_arr_old = yticks2_1_arr
            yticks1_2_old = yticks2_4
            yticks_arr_old_2 = yticks2_2_arr
            plt1 = plt4
            plt3 = plt5
            area1 = area4
            area3 = area5
            xscale1 = xscale2
            yscale1_1 = yscale2_1
            yscale1_2 = yscale2_2
            n1 = n2

        total_old.clear_updaters()
        print('total pos:', total_old.get_center())
        print('eq pos', eq2.get_center())
        self.wait(0.1)
        self.remove(total_old)
        eq2.set_opacity(0)

        self.wait()


class FHSum(Scene):
    def construct(self):
        ax = Axes(x_range=[0, 1.1], x_length=12, y_length=6, y_range=[0,1.1], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': True,
            "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
            "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
        }).set_z_index(1)

        n = 1_000_000
        xvec = np.linspace(1., n, n)
        svec = np.sin(xvec)
        termvec = 1. / (xvec ** 3 * svec * svec)
        sumvec = np.cumsum(termvec)

        for i in [3, 22, 355, n]:
            print(i, sumvec[i-2], termvec[i-1], sumvec[i-1])
        total = sumvec[-1]
        print('100', sumvec[99])
        print('200', sumvec[199])
        print('300', sumvec[299])
        print('1 000', sumvec[999])
        print('10 000', sumvec[9999])
        print('100 000', sumvec[99999])
        print('1 000 000', sumvec[1_000_000-1])
        print(sumvec[-1])
        idx = np.searchsorted(sumvec, 30.306, side="right")
        print(idx, sumvec[idx-5:idx+5])



        i0 = 10
        xscale = 1. / xvec[i0-1]
        yscale = 1. / sumvec[i0-1]
        plt = ax.plot_line_graph(xvec[:i0] * xscale, sumvec[:i0] * yscale, add_vertex_dots=False,
                                 stroke_width=6, stroke_color=BLUE).set_z_index(4)

        self.add(ax, plt)
        self.wait()

class SumTotal(IntroSum):
    def construct(self):
        eq1 = MathTex(r'\sum_{n\le N}', r'\frac1{n^3\sin^2 n}', stroke_width=1.5, font_size=50)
        eq2 = DecimalNumber(30.31, num_decimal_places=2, color=BLUE, font_size=50, stroke_width=1.5).set_z_index(10)
        MathTex.set_default(stroke_width=1.5, font_size=70)
        eq3 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3\sin^2n}', r'\approx', r'30.31', r'\,??').set_z_index(2)

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq1[1][0], eq1[1][3], eq1[1][-2]).set_color(col_num)
        VGroup(eq1[1][1], eq1[0][0]).set_color(col_op)
        VGroup(eq1[1][2], eq1[1][-1], eq1[0][1], eq1[0][-1]).set_color(col_x)
        VGroup(eq1[1][-5:-2]).set_color(col_trig)
        eq3[-1].set_color(RED)

        eq1.move_to(0.37170545 *RIGHT + 1.52226026 * UP)
        eq2.move_to(6.22086595  * RIGHT + 3.1586239 * UP)
        mh.align_sub(eq3, eq3[:-1], ORIGIN)
        eq3.to_edge(DOWN, buff=0.4)
        eq1 = mh.eq_shadow(eq1, bg_stroke_width=10, bg_z_index=9.9, fg_z_index=10)
        eq3_1 = mh.eq_shadow(eq3[:2], bg_stroke_width=0, bg_z_index=9.9, fg_z_index=10)

        box1 = SurroundingRectangle(VGroup(eq3, eq3[-1].copy().next_to(eq3, LEFT)), stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2).move_to(ORIGIN, coor_mask=RIGHT)

        self.add(eq1, eq2)
        self.play(AnimationGroup(mh.rtransform(eq1[1], eq3_1[1], eq1[0][:3], eq3_1[0][1:4]),
                  mh.fade_replace(eq1[0][3], eq3_1[0][4].set_color(col_num)),
                  FadeIn(eq3_1[0][0].set_color(col_special), shift=eq3_1[0][1].get_top()-eq1[0][0].get_top()),
                  ReplacementTransform(eq2, eq3[3].set_color(col_num)),
                                 run_time=2),
                  Succession(Wait(1.2), FadeIn(box1, eq3[2]))
                  )
        self.remove(eq3_1)
        self.add(eq3[:2])
        self.wait(0.1)
        self.play(FadeIn(eq3[-1]))

        self.wait()

class RandomTerms(IntroSum):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=70)
        eq1 = self.get_series().set_z_index(2)
        eq2 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3\sin^2u_n}').set_z_index(2) # replace n with u_n in sine term
        eq3 = MathTex(r'u_n', r'=', r'{\sf distance\ from\ }', r'n', r'{\sf\ to\ nearest\ multiple\ of\ }', r'\pi', font_size=60).set_z_index(2)
        eq4 = MathTex(r'u_n', r'=', r'\min\nolimits_{q\in\mathbb Z}', r'\lvert', r'n-q\pi', r'\rvert', font_size=60).set_z_index(2)
        eq5 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3\sin^2U_n}').set_z_index(2) # replace n with u_n in sine term
        eq6 = MathTex(r'u_n', r'{\sf\ is\ in\ interval\ }', r'[0,\pi/2]', font_size=60).set_z_index(2)
        eq7 = MathTex(r'U_n', r'{\sf\ is\ uniform\ on\ }', r'[0,\pi/2]', font_size=60).set_z_index(2)

        mh.rtransform.copy_colors     = True  # make rtransform copy colors from eq1 to eq2
        mh.stretch_replace.copy_colors = True
        VGroup(eq2[1][-2], eq3[0][0]).set_color(col_p)
        VGroup(eq2[1][-1], eq3[3], eq3[0][1], eq4[2][-3], eq4[4][2]).set_color(col_x)
        VGroup(eq3[2], eq3[4], eq6[1], eq7[1]).set_color(col_txt)
        VGroup(eq3[-1]).set_color(col_pi)
        VGroup(eq4[3], eq4[5], eq6[2][4]).set_color(col_op)
        VGroup(eq4[2][-1]).set_color(col_special)
        VGroup(eq4[2][:3]).set_color(col_op)
        VGroup(eq6[2][1], eq6[2][5]).set_color(col_num)
        eq4[2][-2].set_color(WHITE*0.85)


        eq1.to_edge(DOWN, buff=0.4)
        mh.align_sub(eq2, eq2[0], eq1[0])  # align first term with first term of eq1
        eq3.align_to(eq2, DOWN)
        eq2_1 = eq2.copy().next_to(eq3, UP, buff=0.4, coor_mask=UP)
        mh.align_sub(eq4, eq4[0], eq3[0], coor_mask=UP)
        mh.align_sub(eq5, eq5[0], eq2_1[0])
        mh.align_sub(eq6, eq6[0], eq3[0], coor_mask=UP)
        mh.align_sub(eq7, eq7[0][1], eq6[0][1], coor_mask=UP)

        box1 = SurroundingRectangle(eq2, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)

        eq3 = mh.eq_shadow(eq3)
        eq4 = mh.eq_shadow(eq4)
        eq6 = mh.eq_shadow(eq6)
        eq7 = mh.eq_shadow(eq7)

        self.add(eq1, box1)

        # put a red line round the sine term in manim to highlight it
        circ1 = mh.circle_eq(eq2[1][-6:], scale=0.6).set_z_index(4).shift(DOWN*0.1)
        self.play(Create(circ1, rate_func=linear, run_time=0.6))

        self.wait(0.1)

        # replace n with u_n in sine term
        self.play(mh.rtransform(eq1[1][:-1], eq2[1][:-2], eq1[0], eq2[0], run_time=1.5),
                  mh.fade_replace(eq1[1][-1], eq2[1][-2:], coor_mask=RIGHT),
                  )

        self.wait(0.1)
        # fade in eq3 while shifting eq2 up to make room for eq3
        self.play(mh.rtransform(eq2, eq2_1),
                  VGroup(circ1, box1).animate.shift(mh.diff(eq2, eq2_1)),
                  Succession(Wait(0.5), FadeIn(eq3)))
        self.wait(0.1)
        self.play(Succession(Wait(0.5), AnimationGroup(
            mh.rtransform(eq3[:2], eq4[:2]),
            mh.rtransform(eq3[3][0], eq4[4][0], eq3[5][0], eq4[4][3]),
            run_time=1.7)),
                  Succession(Wait(1.2), FadeIn(eq4[2:4], eq4[4][1:3], eq4[5])),
              FadeOut(eq3[2], eq3[4]),
              )
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0], eq6[0], eq4[4][-1], eq6[2][3]),
                  mh.fade_replace(eq4[3], eq6[2][0]),
                  mh.fade_replace(eq4[5], eq6[2][-1]),
                  mh.fade_replace(eq4[4][0], eq6[2][1], coor_mask=RIGHT),
                  mh.fade_replace(eq4[4][1], eq6[2][2], coor_mask=RIGHT),
                  FadeOut(eq4[4][2], shift=mh.diff(eq4[4][3], eq6[2][3])),
                  FadeOut(eq4[1:3]),
                  FadeIn(eq6[2][4:6]),
                  FadeIn(eq6[1])
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1[0], eq5[0], eq2_1[1][:-2], eq5[1][:-2], eq2_1[1][-1], eq5[1][-1]),
                  mh.stretch_replace(eq2_1[1][-2], eq5[1][-2]),
                  )
        self.play(mh.rtransform(eq6[0][1], eq7[0][1], eq6[2], eq7[2]),
                  mh.stretch_replace(eq6[0][0], eq7[0][0]),
                  mh.fade_replace(eq6[1], eq7[1], coor_mask=RIGHT),)

        self.wait()

class Equidistribution(IntroSum):
    bgcol = GREY

    def construct(self):
        nsteps = 5
        MathTex.set_default(stroke_width=1.5, font_size=50)

        ax = Axes(x_range=[0, nsteps * 1.05], x_length=12, y_length=0.1, y_range=[0, 1], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': True,
            "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
            "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
        }).set_z_index(1)

        xstep = ax.c2p(1,0) - ax.c2p(0,0)

        ax.shift(DOWN*2.5-ax.c2p(nsteps/2, 0))
        ticks = mh.get_xticks(ax, range(nsteps+1), label_color=col_num)

        x = (math.sqrt(5)-1)/2
        # add line segment of length x starting from origin, then repeatedly move to the end of the previous position
        dy = UP * 0.2
        p0 = ax.c2p(0, 0) + dy
        p1 = ax.c2p(x, 0) + dy
        seg  = Line(p0, p1, stroke_width=8, color=RED).set_z_index(4)
        eqx = MathTex(r'x', color=col_p, font_size=50).next_to(seg, UP, buff=0.2)
        dot = Dot(ax.c2p(x, 0), color=RED, radius=0.12).set_z_index(5)

        gp = VGroup(eqx, ticks, ax.x_axis)
        h = gp.height + 0.8
        w = gp.get_right()[0] * 2 + 0.8
        box1 = RoundedRectangle(width=w, height=h, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                corner_radius=0.2).move_to(gp, coor_mask=UP)
        self.add(ax.x_axis, ticks, box1)
        self.play(FadeIn(seg, eqx, dot))

        nx = int(nsteps / x)
        dx = ax.c2p(x, 0) - ax.c2p(0, 0)

        # animate writing the multiples of x above each dot, and moving the line segment and dot to the next position
        eqxs = [MathTex(r'x')] + [MathTex(r'{}x'.format(i), color=col_num) for i in range(1, nx)]
        for i in range(nx):
            mh.align_sub(eqxs[i], eqxs[i][0][-1].set_color(col_p), ax.c2p((i+1)*x, 0), UP, buff=0.3)

        dots_line = [dot.copy()]
        self.add(dots_line[-1])
        for i in range(0, nx-1):
            anims = [VGroup(seg, eqx, dot).animate.shift(RIGHT*dx)]
            if i > 0: anims.append(FadeIn(eqxs[i-1]))
            self.play(*anims, run_time=0.5)
            dots_line.append(dot.copy())
            self.add(dots_line[-1])
        self.play(VGroup(seg, eqx).animate.shift(RIGHT*dx),
            dot.animate.shift(RIGHT*dx).set_opacity(0),
                  FadeIn(eqxs[-2]), run_time=0.5)
        self.play(VGroup(seg, eqx, dot).animate(run_time=0.6).next_to(mh.pos(RIGHT), RIGHT, buff=0, coor_mask=RIGHT),
                  FadeIn(eqxs[-1], run_time=0.5))
        self.remove(seg, eqx, dot)

        # draw numberline above the existing numberline, aligned to the left
        ax2 = Axes(x_range=[0, 1], x_length=xstep[0]*3, y_length=0.1, y_range=[0, 1], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': False}).set_z_index(1)
        ax2.shift(ax.c2p(0, 0) - ax2.c2p(0, 0) + UP * 2.75).move_to(ORIGIN, coor_mask=RIGHT)
        ticks2 = mh.get_xticks(ax2, [0, 1], label_color=col_num)

        # move each unit segment of the original x-axis to the new one,
        axes = []
        for i in range(nsteps):
            ax3 = Axes(x_range=[0, 1], x_length=xstep[0], y_length=0.1, y_range=[0, 1], axis_config={
                'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': False}).set_z_index(1)
            ax3.shift(ax.c2p(i, 0) - ax3.c2p(0, 0))
            ticks3 = mh.get_xticks(ax3, [0, 1], label_color=col_num)
            if i == 0:
                axes.append(VGroup(ax3.x_axis, ticks3))
            else:
                axes.append(VGroup(ax3.x_axis, ticks3[0][0], ticks3[1][0]))

        # move dots to new axis
        eq4 = MathTex(r'{\sf fractional\ part\ of\ }', r'nx', stroke_width=1.5, font_size=60).set_z_index(1)
        eq4[0].set_color(col_txt)
        eq4[1][-1].set_color(col_p)
        eq4[1][0].set_color(col_num)
        eq4.next_to(ax2.x_axis, UP, buff=1)

        dot.set_opacity(1)
        ndots = 30
        dots2 = [dot.copy().move_to(ax2.c2p((i+1) * x % 1)) for i in range(ndots)]
        eqx1 = eqxs[0]
        eqx2 = eqx1.copy().next_to(ax2.c2p(x % 1), UP, buff=0.3)

        anims2 = []
        for i in range(nsteps):
            if i == 0: anims = [ReplacementTransform((axes[0] + eqx1).copy(), VGroup(ax2.x_axis, ticks2) + eqx2)]
            else: anims = [ReplacementTransform(axes[i], VGroup(ax2.x_axis, ticks2[0][0], ticks2[1][0]))]
            i0 = int(i / x)
            i1 = int((i + 1) / x)
            anims.append(ReplacementTransform(VGroup(*dots_line[i0:i1]).copy(), VGroup(*dots2[i0:i1])))
            anims2.append(Succession(Wait(i*0.5), AnimationGroup(*anims, run_time=1.5)))

        box2 = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=1)
        self.play(FadeOut(*eqxs[1:]), *anims2,
                  FadeIn(eq4, box2))
        self.remove(box1)
        self.wait(0.1)
        op = [1] * (nx+1) + list(np.linspace(1, 0.5, ndots - nx))
        for i in range(ndots):
            dots2[i].set_opacity(op[i+1])

        self.play(FadeIn(*dots2[i1:]))

        # wrap the axis into a circle, and move the dots to the circle
        r = 1.8
        # circle = Circle(radius=2, color=WHITE, stroke_width=6).set_z_index(1).rotate(-PI/2)
        center = ax2.x_axis.get_center() * UP + mh.pos(RIGHT) + LEFT * (1 + r) + UP*0.5

        p0 = ax2.c2p(0, 0)
        p1 = ax2.c2p(1,0)

        dr0 = (p0 - eqx2.get_center())[1]
        ticks2_1 = ticks2[0].copy()
        alphaval = ValueTracker(0.)
        def bend():
            alpha = alphaval.get_value()
            theta = TAU * alpha
            b = alpha**2
            len = (p1-p0)[0] * (1-b) + TAU * r * b
            c = 1 - (1-alpha)**3

            if theta < 1e-6: return VGroup()

            radius = len / theta
            cent = p0 + UP * radius
            cent += c * (center + r * DOWN + UP * 0.2 - p0)
            dots = [dot.copy().move_to(cent + radius * np.array([np.sin(theta*(i*x % 1)), -np.cos(theta*(i*x % 1)), 0])).set_opacity(op[i]) for i in range(1, ndots + 1)]
            dr = dr0 * (1-alpha) + 0.4 * alpha
            posx = np.array([np.sin(theta*(x % 1)), -np.cos(theta*(x % 1)), 0]) * (radius + dr) + cent
            eqx3 = eqx2.copy().move_to(posx)
            shape = Arc(
                radius=radius,
                start_angle=-PI / 2,
                angle=theta,
                arc_center=cent,
                num_components=64,
                stroke_width=6,
            )

            return VGroup(shape, ticks2_1.copy().shift(shape.get_start() - p0), eqx3, *dots).set_z_index(10)

        bendy = always_redraw(bend)
        self.add(bendy)

        ax2.y_axis.set_opacity(0)
        gp1 = VGroup(ax2, *dots2, ticks2, eqx2)
        gp2 = gp1.copy().to_edge(LEFT, buff=1)
        eq4_1 = eq4.copy()
        self.play(
            alphaval.animate.set_value(1),
            Transform(gp1, gp2),
            FadeOut(eq4, shift=mh.diff(gp1, gp2)),
            VGroup(ax2, *dots2, ticks2, eqx2).animate.to_edge(LEFT, buff=1),
            run_time=3,
            rate_func=smooth,
        )

        bendy.clear_updaters()

        self.remove(*dots_line, *dots2, *eqxs, eqx2, bendy)
        self.add(bendy[:2])

        xval = ValueTracker(x)

        def get_obj():
            x0 = xval.get_value()
            center = bendy[0].get_center()
            eqx.next_to(ax.c2p(x0, 0), UP, buff=0.3)
            eqx2 = eqx.copy().next_to(ax2.c2p(x0, 0), UP, buff=0.3)
            eqx3 = eqx.copy().move_to(center + (r+0.4)*(math.cos(x0*TAU)*DOWN + math.sin(x0*TAU)*RIGHT))
            nx1 = int(nsteps/x0) + 1
            dots = [dot.copy().move_to(ax.c2p(i*x0)) for i in range(1, nx1+1)]
            dots[-1].set_opacity(1-(nx1*x0-nsteps)*4)

            dots2 = [dot.copy().move_to(ax2.c2p(i*x0 % 1)).set_opacity(op[i]) for i in range(1, ndots+1)]
            dots3 = [dot.copy().move_to(center+r*(DOWN*math.cos(i*x0*TAU)+RIGHT*math.sin(i*x0*TAU))).set_opacity(op[i]) for i in range(1, ndots+1)]

            return VGroup(VGroup(eqx, eqx2, eqx3), *dots, *dots2, *dots3).set_z_index(10)

        obj = get_obj()
        self.add(obj)
        self.wait(0.1)
        self.remove(obj)
        obj = always_redraw(get_obj)
        self.add(obj)
        eq1 = MathTex(r'x', r'=', r'1/4', font_size=80, stroke_width=2).set_z_index(10)
        eq1.move_to((ax.c2p(0,0) + ax2.c2p(0,0))/2, coor_mask=UP).shift(DOWN*0.2)
        eq1.next_to(ax2.x_axis, UP*1.5, buff=1.2)
        eq2 = MathTex(r'x', r'=', r'2/5', font_size=80, stroke_width=2).set_z_index(10)
        eq3 = MathTex(r'x', r'=', r'1/\sqrt2', font_size=80, stroke_width=2).set_z_index(10)
        eq1[0].set_color(col_p)
        VGroup(eq1[2][::2], eq2[2][::2], eq3[2][0], eq3[2][-1]).set_color(col_num)
        VGroup(eq1[2][1], eq2[2][1], eq3[2][1:-1]).set_color(col_op)
        mh.align_sub(eq2, eq2[1], eq1[1])
        mh.align_sub(eq3, eq3[1], eq1[1])

        box = SurroundingRectangle(VGroup(eq1, eq2, eq3[-1][-1]), stroke_width=4, stroke_color=YELLOW,
                                   fill_color=GREY*0.4, fill_opacity=1, buff=0.25, corner_radius=0.2)

        self.play(xval.animate.set_value(0.25), FadeIn(eq1, box), run_time=2)
        self.wait(0.1)
        self.play(xval.animate.set_value(0.4),
                  mh.rtransform(eq1[2][1], eq2[2][1]),
                  mh.fade_replace(eq1[2][0], eq2[2][0]),
                  mh.fade_replace(eq1[2][2], eq2[2][2]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(xval.animate.set_value(1/math.sqrt(2)),
                  mh.rtransform(eq2[2][1], eq3[2][1]),
                  mh.fade_replace(eq2[2][0], eq3[2][0]),
                  mh.fade_replace(eq2[2][-1], eq3[2][-1]),
                  FadeIn(eq3[2][2:-1], shift=mh.diff(eq2[2][-1], eq3[2][-1])),
                  run_time=2.)
        obj.clear_updaters()
        self.wait(0.1)

        eq5 = MathTex(r'\{nx\}', r'\sim', r'{\sf uniform\ on\ }', r'[0,1]', font_size=60).set_z_index(10)
        eq5[0][::3].set_color(col_op)
        eq5[0][1].set_color(col_x)
        eq5[0][2].set_color(col_p)
        eq5[2].set_color(col_txt)
        eq5[3][1::2].set_color(col_num)
        eq5.move_to(ax2.x_axis).shift(DOWN*1.4)
        self.play(FadeIn(eq5), obj[0].animate.set_opacity(0))
        circ = mh.circle_eq(eq5[0], scale=0.6).set_z_index(4)
        self.wait(0.1)
        eq4_1.move_to(ax2.x_axis).shift(UP*1)
        line1 = Line(eq4_1[0][6].get_bottom()+DOWN*0.1, circ.get_top()+UP*0.1, stroke_width=8, color=YELLOW).set_z_index(30)
        self.play(Create(circ, rate_func=linear, run_time=0.6),
                  Succession(Wait(0.15),FadeIn(eq4_1, line1)))

        eq6 = MathTex(r'f(', r'\{nx\}', r')', font_size=60).set_z_index(10)
        eq6[0][0].set_color(col_psi)
        mh.align_sub(eq6, eq6[1], eq5[0]).move_to(ax2.x_axis, coor_mask=RIGHT)

        self.wait(0.1)
        self.play(FadeOut(eq4_1, line1, eq5[1:]),
                  mh.rtransform(eq5[0], eq6[1], run_time=1., copy_colors=True),
                  FadeOut(circ, shift=mh.diff(eq5[0], eq6[1])),
                  Succession(Wait(0.5), FadeIn(eq6[0], eq6[2])),)

        eq7 = MathTex(r'\frac1N', r'\sum_{n=1}^N', r'f(', r'\{nx\}', r')', r'\to',
                      r'\int_0^1', r'f(u)', r'\,du', font_size=60).set_z_index(10)
        mh.align_sub(eq7, eq7[3], eq6[1])
        VGroup(eq7[0][1], eq7[1][1], eq7[6][0], eq7[8][0]).set_color(col_op)
        VGroup(eq7[0][0], eq7[1][4], eq7[6][1:]).set_color(col_num)
        VGroup(eq7[0][2], eq7[1][0], eq7[1][2], eq7[7][2], eq7[8][1]).set_color(col_x)
        eq7[7][0].set_color(col_psi)

        self.wait(0.1)
        self.play(mh.rtransform(eq6[:], eq7[2:5], copy_colors=True), FadeIn(eq7[0], eq7[1]), run_time=1.)
        self.wait(0.1)
        self.play(FadeIn(eq7[5:]))

        eq8 = MathTex(r'\frac1N', r'\sum_{n=1}^N', r'f(', r'\{nx\}', r')', r'\to', r'\mathbb E[',
                      r'f(U)', r']', font_size=60).set_z_index(10)
        mh.align_sub(eq8, eq8[3], eq7[3])
        eq8[6][0].set_color(col_txt2)
        self.wait(0.1)

        self.play(mh.rtransform(eq7[:6], eq8[:6], eq7[7][:2], eq8[7][:2], eq7[7][-1], eq8[7][-1], copy_colors=True),
                  mh.stretch_replace(eq7[7][2], eq8[7][2], copy_colors=True),
                  FadeIn(eq8[6]),
                  FadeOut(eq7[6]),
                  FadeOut(eq7[8], shift=mh.diff(eq7[7][-1], eq8[7][-1])),
                  FadeIn(eq8[8], shift=mh.diff(eq7[7][-1], eq8[7][-1])),
                  run_time=1.5)

        self.wait(0.1)

        circ = mh.circle_eq(eq8[7][2], scale=0.6).set_z_index(20).shift(RIGHT*0.15)

        eq7 = MathTex(r'{\sf uniform\ on\ }', r'[0,1]', font_size=60).set_z_index(10)
        eq7[0].set_color(col_txt)
        eq7[1][1::2].set_color(col_num)
        eq7.move_to(eq4_1).shift(RIGHT*2.6)
        line = Line(eq7[0][6].get_bottom()+DOWN*0.1, circ.get_top()+UP*0.1, stroke_width=8, color=YELLOW).set_z_index(30)
        self.play(Create(circ, rate_func=linear, run_time=0.6),
                  Succession(Wait(0.15), FadeIn(eq7, line)))

        self.wait()

def SurroundingRectangleSym(mobject, buff=0.2, **kwargs):
    p0 = mobject.get_corner(UR) + buff * UR
    p1 = mobject.get_corner(DL) + buff * DL
    rect = RoundedRectangle(width=max(p0[0], -p1[0]) * 2, height=p0[1] - p1[1], **kwargs)
    return rect.move_to(mobject, coor_mask=UP)

class RandomSum(IntroSum):
    def construct(self):
        self.do_anim()

    def do_anim(self, final_eq=False):
        MathTex.set_default(stroke_width=1.5, font_size=90)
        eq1 = self.get_series().set_z_index(1)
        eq2 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3\sin^2(\pi\frac n\pi)}').set_z_index(1)
        eq3 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3\sin^2(\pi\{\frac n\pi\})}').set_z_index(1)
        eq4 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3\sin^2(\pi U_n)}').set_z_index(1)
        eq5 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3\sin^2(\frac{\pi}2 U_n)}').set_z_index(1)
        eq6 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3(\frac{\pi}2 U_n)^2}').set_z_index(1)
        eq7 = MathTex(r'\frac4{\pi^2}', r'\sum_{n=1}^\infty', r'\frac1{n^3U_n^2}').set_z_index(1)
        eq8 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3U_n^2}').set_z_index(1)
        # eq9 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3U_n^2}', font_size=120).set_z_index(1)
        eq10 = MathTex(r'\mathbb E\left[', r'\sum_{n=1}^\infty', r'\frac1{n^3U_n^2}', r'\right]').set_z_index(1)
        eq11 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3}', r'\mathbb E\left[', r'U_n^{-2}', r'\right]').set_z_index(1)
        eq12 = MathTex(r'\sum_{n=1}^\infty', r'\frac1{n^3}', r'\int_0^1', r'u^{-2}', r'\,du').set_z_index(1)
        eq13 = MathTex(r'\sum_{n=1}^\infty', r'\min\left(', r'\frac1{n^3U_n^2}', r',1\right)').set_z_index(1)
        eq14 = MathTex(r'\sum_{n=1}^\infty', r'\mathbb E\left[', r'\min\left(', r'\frac1{n^3U_n^2}', r',1\right)', r'\right]').set_z_index(1)
        MathTex.set_default(stroke_width=1.5, font_size=60)
        eq15 = MathTex(r'\sum_{n=1}^\infty', r'\mathbb E\left[', r'\min\left(', r'\frac1{n^3U_n^2}', r',1\right)', r'\right]').set_z_index(1)
        eq16 = MathTex(r'=', r'\sum_{n=1}^\infty', r'\mathbb E\left[', r'\min\left(', r'\frac1{n^3U_n^2}', r',1\right)', r'\right]').set_z_index(1)

        eq15.to_edge(UL, buff=0.8)
        eq16.next_to(eq15, DOWN, buff=0.4).to_edge(RIGHT, buff=2)
        VGroup(eq15, eq16).move_to(ORIGIN, coor_mask=UP)
        if final_eq:
            VGroup(eq15[0][1], eq15[2][:3], eq15[3][1]).set_color(col_op)
            eq15[0][0].set_color(col_special)
            VGroup(eq15[0][-1], eq15[3][0], eq15[3][3::2], eq15[4][1]).set_color(col_num)
            VGroup(eq15[0][2], eq15[3][2::4]).set_color(col_x)
            eq15[3][4].set_color(col_p)
            eq15[1][0].set_color(col_txt2)
            mh.copy_colors_eq(eq15[:], eq16[1:])
            return eq15, eq16

        eq1.to_edge(DOWN, buff=0.6)
        mh.align_sub(eq2, eq2[0], eq1[0])  # align first term with first term of eq1
        mh.align_sub(eq3, eq3[0], eq1[0])  # align first term with first term of eq1
        mh.align_sub(eq4, eq4[0], eq1[0])  # align first term with first term of eq1
        mh.align_sub(eq5, eq5[0], eq1[0])  # align first term with first term of eq1
        mh.align_sub(eq6, eq6[0], eq1[0])  # align first term with first term of eq1
        mh.align_sub(eq7, eq7[1], eq1[0], coor_mask=UP)  # align first term with first term of eq1
        mh.align_sub(eq8, eq8[0], eq1[0], coor_mask=UP)  # align first term with first term of eq1
        mh.align_sub(eq10, eq10[1], eq1[0], coor_mask=UP)  # align first term with first term of eq1
        mh.align_sub(eq11, eq11[0], eq1[0], coor_mask=UP)  # align first term with first term of eq1
        mh.align_sub(eq12, eq12[0], eq1[0], coor_mask=UP)  # align first term with first term of eq1
        mh.align_sub(eq13, eq13[0], eq1[0], coor_mask=UP)  # align first term with first term of eq1
        mh.align_sub(eq14, eq14[0], eq1[0], coor_mask=UP)  # align first term with first term of eq1

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq2[1][-3], eq3[1][-6], eq3[1][-2], eq5[1][-5], eq12[4][0]).set_color(col_op)
        VGroup(eq12[2][0]).set_color(col_int)
        VGroup(eq2[1][-5], eq2[1][-2]).set_color(col_pi)
        VGroup(eq4[1][-3], eq12[4][1]).set_color(col_p)
        VGroup(eq4[1][-2]).set_color(col_x)
        VGroup(eq5[1][-4], eq12[2][1:], eq13[3][1]).set_color(col_num)
        VGroup(eq10[0][0], eq14[1][0]).set_color(col_txt2)
        VGroup(eq13[1][:3]).set_color(col_op)


        box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box2 = SurroundingRectangle(eq2, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box3 = SurroundingRectangle(eq3, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box4 = SurroundingRectangle(eq4, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box6 = SurroundingRectangle(eq6, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box7 = SurroundingRectangle(eq7, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box8 = SurroundingRectangle(eq8, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box11 = SurroundingRectangle(VGroup(eq11, eq10), stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box12 = SurroundingRectangle(VGroup(eq12, eq10), stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box13 = SurroundingRectangle(eq13, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box14 = SurroundingRectangle(eq14, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.2, buff=0.2)
        box15 = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=1)

        self.add(eq1, box1)
        self.play(mh.rtransform(eq1[1][:-1], eq2[1][:-6], eq1[0], eq2[0], eq1[1][-1], eq2[1][-4], box1, box2),
                  Succession(Wait(0.5), FadeIn(eq2[1][-3:], eq2[1][-6:-4])))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[1][:-5], eq3[1][:-7], eq2[0], eq3[0], eq2[1][-4:-1], eq3[1][-5:-2],
                                eq2[1][-1], eq3[1][-1], eq2[1][-5], eq3[1][-7], box2, box3),
                  Succession(Wait(0.4), FadeIn(eq3[1][-6], eq3[1][-2])))
        self.wait(0.1)

        circ = mh.circle_eq(eq3[1][-5:-1], scale=0.6).set_z_index(4).shift(LEFT*0.2)
        txt1 = Tex(r'\sf uniform on ', r'$[0,1]$', font_size=60, color=RED, stroke_width=2)
        txt1.next_to(circ, UR, buff=0.1).shift(LEFT * 1.5)
        txt1 = mh.eq_shadow(txt1, bg_z_index=4.9, fg_z_index=5)
        self.play(Create(circ, rate_func=linear, run_time=0.6),
                  Succession(Wait(0.15), FadeIn(txt1)))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0], eq4[0], eq3[1][:-7], eq4[1][:-4], eq3[1][-7], eq4[1][-4], eq3[1][-1], eq4[1][-1],
                                box3, box4),
                  mh.fade_replace(eq3[1][-6:-1], eq4[1][-3:-1], coor_mask=RIGHT),
                  circ.animate.shift(LEFT*0.2))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0], eq5[0], eq4[1][:-3], eq5[1][:-5], eq4[1][-3:], eq5[1][-3:]),
                  Succession(Wait(0.3), FadeIn(eq5[-1][-5:-3])),
                  FadeOut(circ, txt1))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[0], eq6[0], eq5[1][:4], eq6[1][:4], eq5[1][-7:], eq6[1][-8:-1],
                                eq5[1][7], eq6[1][-1], box4, box6),
                  FadeOut(eq5[1][4:7]),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq6[0], eq7[1], eq6[1][:4], eq7[2][:4], eq6[1][5], eq7[0][2], eq6[1][6], eq7[0][1],
                                eq6[1][-4], eq7[2][-3], eq6[1][-3], eq7[2][-1], eq6[1][-1], eq7[2][-2],
                                box6, box7),
                  mh.stretch_replace(eq6[1][7], eq7[0][0]),
                  FadeIn(eq7[0][3].set_color(col_num), shift=mh.diff(eq6[1][5], eq7[0][2])),
                  FadeOut(eq6[1][4], shift=mh.diff(eq6[1][3], eq7[2][3])),
                  FadeOut(eq6[1][-2]),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(eq7[1:], eq8[:], box7, box8), FadeOut(eq7[0]))
        self.wait(0.1)
        eq8_1 = eq8.copy()
        self.play(mh.rtransform(eq8[:], eq10[1:-1], box8, box11, run_time=1.2),
                  Succession(Wait(0.6), FadeIn(eq10[0], eq10[-1])))
        eq8 = eq8_1
        self.wait(0.1)
        self.play(mh.rtransform(eq10[1], eq11[0], eq10[0], eq11[2],
                                eq10[2][:4], eq11[1][:], eq10[2][-2:], eq11[3][-2:], eq10[2][-3], eq11[3][-4]),
                  FadeIn(eq11[3][-3].set_color(col_op), shift=mh.diff(eq10[2][-3], eq11[3][-4])),
                  mh.stretch_replace(eq10[-1], eq11[-1]))

        # transform eq11 to eq12
        self.play(mh.rtransform(eq11[:2], eq12[:2], eq11[3][1:3], eq12[3][1:3], box11, box12),
                  mh.stretch_replace(eq11[3][0], eq12[3][0]),
                  mh.fade_replace(eq11[2], eq12[2], coor_mask=RIGHT),
                  mh.fade_replace(eq11[4], eq12[4], coor_mask=RIGHT),
                  FadeOut(eq11[3][-1], shift=mh.diff(eq11[3][0], eq12[3][0])))

        circ1 = mh.circle_eq(eq12[2:], scale=0.7).set_z_index(20).shift(DOWN*0.1)
        txt1 = Tex(r'\sf infinite', font_size=70, color=RED, stroke_width=1.5)
        txt1.next_to(circ1, UR, buff=-0.3)
        txt1 = mh.eq_shadow(txt1, bg_z_index=20.9, fg_z_index=21)
        self.wait(0.1)
        self.play(Create(circ1, rate_func=linear, run_time=1.),
                  Succession(Wait(0.4), FadeIn(txt1)))

        self.wait(0.1)
        self.play(mh.rtransform(eq12[0], eq8[0], eq12[1][:4], eq8[1][:4], eq12[3][-1], eq8[1][-2]),
                  mh.stretch_replace(eq12[3][0], eq8[1][-3]),
                  FadeOut(eq12[3][-2], shift=mh.diff(eq12[3][-1], eq8[1][-2])),
                  FadeIn(eq8[1][-1], shift=mh.diff(eq12[3][-3], eq8[1][-4])),
                  FadeOut(eq12[2], eq12[4]),
                  FadeOut(circ1, txt1),)
        self.wait(0.1)
        # transform eq8 to eq13
        self.play(mh.rtransform(eq8[0], eq13[0], eq8[1], eq13[2], box12, box13, run_rime=1.3),
                  Succession(Wait(0.6), FadeIn(eq13[1], eq13[-1])))
        self.wait(0.1)
        # transform eq13 to eq14
        self.play(mh.rtransform(eq13[0], eq14[0], eq13[1:4], eq14[2:5], box13, box14),
                  Succession(Wait(0.5), FadeIn(eq14[1], eq14[-1])))
        self.wait(0.1)

        self.play(mh.rtransform(eq14[:].copy(), eq15[:], eq14[:], eq16[1:]),
                  FadeIn(eq16[0], shift=mh.diff(eq14[0], eq16[0])),
                  FadeIn(box15, rate_func=linear),
                  run_time=2)

        self.wait()

class IntegrateSum(RandomSum):
    bgcol = BLACK

    def construct(self):
        eq1, eq2 = self.do_anim(final_eq=True)
        eq3 = MathTex(r'=', r'\sum_{n=1}^\infty', r'\mathbb E\left[', r'\frac1{\max(n^3U^2,1)}', r'\right]')
        eq4 = MathTex(r'=', r'\sum_{n=1}^\infty', r'\int_0^1', r'\frac{du}{\max(n^3u^2,1)}')
        eq5 = MathTex(r'\int_0^{n^{-3/2}}', r'\frac{du}{1}', r'+', r'\int_{n^{-3/2}}^1', r'\frac{du}{n^3u^2}')
        eq6 = MathTex(r'n^{-3/2}', r'-', r'\left[', r'\frac1{n^3u}', r'\right]_{n^{-3/2}}^1')
        eq7 = MathTex(r'n^{-3/2}', r'+', r'\frac1{n^{3/2}}', r'-', r'\frac1{n^3}')
        eq8 = MathTex(r'\le', r'\sum_{n=1}^\infty', r'\frac2{n^{3/2}}')
        eq9 = MathTex(r'\le', r'\sum_{n=1}^\infty', r'\frac2{n^{3/2}}', r'<', r'\infty')
        eq10 = MathTex(r'\le', r'\sum_{n=1}^\infty', r'\frac2{n^{3/2}}', r'=2\zeta(', r'{\scriptstyle\frac32}', r')', r'<', r'\infty')
        mh.font_size_sub(eq10, 4, 70)
        eq10[4].move_to(eq10[5][0], coor_mask=UP)

        mh.align_sub(eq3, eq3[0], eq2[0])
        mh.align_sub(eq4, eq4[0], eq2[0])
        gp1 = VGroup(eq1, eq4).copy().to_edge(UP, buff=0.8)
        eq4_1 = gp1[1]
        eq5.next_to(eq4_1, DOWN, buff=0.5).to_edge(RIGHT, buff=1)
        mh.align_sub(eq6, eq6[1], eq5[2])
        eq6[0].move_to(eq5[:2], coor_mask=RIGHT)
        mh.align_sub(eq7, eq7[0], eq6[0])
        mh.align_sub(eq8, eq8[1], eq4_1[1])
        mh.align_sub(eq9, eq9[1], eq4_1[1])
        mh.align_sub(eq10, eq10[1], eq4_1[1])

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq3[2:5], eq4[2][0], eq4[3][0], eq10[4][1]).set_color(col_op)
        VGroup(eq4[2][1:], eq10[3][1], eq10[4][::2]).set_color(col_num)
        VGroup(eq4[3][1]).set_color(col_p)
        eq10[3][2].set_color(col_psi)

        self.add(eq1, eq2)
        self.play(mh.rtransform(eq2[:3], eq3[:3], eq2[4][:2], eq3[3][:2], eq2[4][2:6], eq3[3][6:10],
                                eq2[5][:2], eq3[3][10:12], eq2[-1][-1], eq3[-1][-1], eq2[3][0], eq3[3][2]),
                  mh.fade_replace(eq2[3][1], eq3[3][3]),
                  mh.fade_replace(eq2[3][2], eq3[3][4]),
                  mh.stretch_replace(eq2[3][3], eq3[3][5]),
                  FadeOut(eq2[4][-1], shift=mh.diff(eq2[4][-3], eq3[3][-5])),
                          mh.stretch_replace(eq2[5][-1], eq3[3][-1]),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[3][1:8], eq4[3][2:9], eq3[3][9:], eq4[3][10:]),
                  mh.fade_replace(eq3[2], eq4[2], coor_mask=RIGHT),
                  mh.fade_replace(eq3[3][0], eq4[3][:2]),
                  mh.stretch_replace(eq3[3][8], eq4[3][9]),
                  FadeOut(eq3[-1]))
        self.wait(0.1)
        box1 = SurroundingRectangle(eq4[2:], stroke_width=6, stroke_opacity=1, stroke_color=RED, fill_opacity=0,
                                    buff=0.05, corner_radius=0.15).flip(UP)
        self.play(Create(box1, rate_func=linear), run_time=0.8)
        gp2 = VGroup(eq1, eq4)
        self.play(gp2.animate.move_to(gp1), box1.animate.shift(mh.diff(gp2, gp1)))
        self.wait(0.1)
        eq4_ = eq4[2:].copy()
        self.play(mh.rtransform(eq4_[0][0], eq5[0][0], eq4_[0][-1], eq5[0][-1], eq4_[1][:3], eq5[1][:3],
                                eq4_[1][-2], eq5[1][-1], eq4_[1][7], eq5[0][1], eq4_[1][8], eq5[0][3],
                                eq4_[1][10], eq5[0][5]),
                  FadeIn(eq5[0][2].set_color(col_op), shift=mh.diff(eq4_[1][8], eq5[0][3])),
                  FadeIn(eq5[0][4].set_color(col_op), shift=mh.diff(eq4_[1][10], eq5[0][5])),
                  run_time=1.6)
        self.wait(0.1)
        eq4_ = eq4[2:].copy()
        self.play(AnimationGroup(mh.rtransform(eq4_[0][:2], eq5[3][:2], eq4_[1][:3], eq5[4][:3], eq4_[1][7:11].copy(), eq5[4][3:],
                                eq4_[1][7], eq5[3][2], eq4_[1][8], eq5[3][4], eq4_[1][10], eq5[3][6]),
                  FadeIn(eq5[3][3].set_color(col_op), shift=mh.diff(eq4_[1][8], eq5[3][4])),
                    FadeIn(eq5[3][5].set_color(col_op), shift=mh.diff(eq4_[1][10], eq5[3][6])),
                  run_time=1.6),
                  Succession(Wait(0.7), FadeIn(eq5[2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[0][1:-1], eq6[0][:]),
                  FadeOut(eq5[0][0], eq5[0][-1], eq5[1]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.fade_replace(eq5[2], eq6[1]),
                  mh.fade_replace(eq5[3][0], eq6[2], coor_mask=RIGHT),
                  mh.rtransform(eq5[3][1], eq6[-1][1], path_arc=-45*DEGREES),
                  mh.rtransform(eq5[3][2:], eq6[-1][2:], path_arc=45*DEGREES),
                  mh.fade_replace(eq5[4][:2], eq6[3][0].set_color(col_num), coor_mask=RIGHT),
                  mh.rtransform(eq5[4][2:6], eq6[3][1:5]),
                  FadeOut(eq5[4][-1], shift=mh.diff(eq5[4][-2], eq6[3][-1])),
                  FadeIn(eq6[-1][0], shift=(eq5[4][2].get_right() - eq6[3].get_right())*LEFT),
                  run_time=1.6)
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq6[0], eq7[0], eq6[3][:4], eq7[2][:4]),
                  mh.rtransform(eq6[4][2], eq7[2][2], eq6[4][4:], eq7[2][3:],
                                eq6[1].copy(), eq7[3], eq6[3][:4].copy(), eq7[4][:4]
                                ),
                  mh.fade_replace(eq6[1], eq7[1]),
                  FadeOut(eq6[3][4], target_position=eq7[2][2]),
                  FadeOut(eq6[4][3], shift=mh.diff(eq6[4][4], eq7[2][3])),
                  FadeOut(eq6[-1][1], target_position=eq7[4][3]),
                                 run_time=1.7),
                  FadeOut(eq6[2], eq6[-1][0])
                  )
        self.wait(0.1)
        eq8[2].move_to(eq4[2:], coor_mask=RIGHT)
        self.play(AnimationGroup(mh.stretch_replace(eq4[0], eq8[0]),
                  mh.rtransform(eq4[1], eq8[1], eq7[2][1:], eq8[2][1:]),
                  mh.rtransform(eq7[0][0], eq8[2][2], eq7[0][2:], eq8[2][3:]),
                  mh.fade_replace(eq7[2][0], eq8[2][0].set_color(col_num)),
                  FadeOut(eq7[0][1], shift=mh.diff(eq7[0][2], eq8[2][3])),
                  FadeOut(eq4[2:]),
                                 FadeOut(eq7[1], eq7[3:]),
                  run_time=1.8),)
        self.wait(0.1)
        eq9[-1].set_color(col_special)
        self.play(mh.rtransform(eq8[:3], eq9[:3]), FadeOut(box1),
                  Succession(Wait(0.3), FadeIn(eq9[3:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq9[:3], eq10[:3], eq9[3:], eq10[6:]),
                  Succession(Wait(0.3), FadeIn(eq10[3:6])))
        # self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2], eq4[3], eq3[-1], eq4[-1], eq3[3][:6], eq4[4][:6],
        #                         eq3[3][1].copy(), eq4[2][1], eq3[3][8:11], eq4[4][6:9],
        #                         eq3[3][-1], eq4[4][-1], eq3[3][6].copy(), eq4[4][-4], eq3[3][7].copy(), eq4[4][-2]),
        #           FadeOut(eq3[3][-2], shift=mh.diff(eq3[3][-2], eq4[4][-4])*RIGHT),
        #           mh.rtransform(eq3[3][6:8], eq4[2][2:], path_arc=-80*DEGREES),
        #           mh.rtransform(eq3[3][0].copy(), eq4[2][0], path_arc=80*DEGREES),
        #           FadeIn(eq4[4][-3].set_color(col_op), shift=mh.diff(eq3[3][7], eq4[4][-2])),
        #           run_time=1.7
        #           )
        # self.wait(0.1)
        # self.play(mh.rtransform(eq4[:4], eq5[:4], eq4[4][:7], eq5[4][:7], eq4[4][7], eq5[4][-1],
        #                         eq4[4][8:12], eq5[4][7:11], eq4[-1], eq5[-1], eq4[4][-1], eq5[4][-2],
        #                         eq4[4][7].copy(), eq5[4][-3]),
        #           FadeIn(eq5[4][-4].set_color(col_op)),)
        # self.wait(0.1)
        # self.play(mh.rtransform(eq5[:3], eq6[:3], eq5[4][1:6], eq6[4][2:7], eq5[4][7:], eq6[4][8:]),
        #           mh.fade_replace(eq5[3], eq6[3], coor_mask=RIGHT),
        #           mh.fade_replace(eq5[4][0], eq6[4][:2], coor_mask=RIGHT),
        #           FadeOut(eq5[-1]),
        #           mh.stretch_replace(eq5[4][6], eq6[4][7]),)
        # self.wait(0.1)
        # # split up integral in eq6 into two integrals in eq7
        # eq6_ = eq6[3:].copy()
        # self.play(mh.rtransform(eq6_[0][0], eq7[0][0], eq6_[0][-1], eq7[0][-1],
        #                         eq6_[1][:3], eq7[1][:3], eq6_[1][9:14], eq7[0][1:6],
        #                         eq6_[1][9:12].copy(), eq7[1][3:]),)

        self.wait(1)

if __name__ == "__main__":
    n = 1_000_000
    xvec = np.linspace(1., n, n)
    svec = np.sin(xvec)
    terms = 1. / (xvec**3 * svec * svec)
    print(sum(terms))