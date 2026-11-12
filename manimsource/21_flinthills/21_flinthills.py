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

class IntroSum(Scene):
    bgcol = GREY
    trcol = BLACK
    def __init__(self, *args, **kwargs):
        config.background_color = self.trcol if config.transparent else self.bgcol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        self.do_anim()

    def do_anim(self, just_eq=False):
        MathTex.set_default(font_size=100, stroke_width=2)
        eq1 = MathTex(r'\sum_{n=1}^\infty', r'\frac{1}{n^3\sin^2n}')
        args = [r'\frac1{\sin^21}', r'+']
        for n in range(2, 6):
            args += [r'\frac1{{ {}^3\sin^2{} }}'.format(n, n), r'+']
        eq2 = MathTex(*args, stroke_width=1.5)
        eq3 = MathTex(r'\sum_{n=1}^\infty', r'\frac{1}{n^3\sin^2n}', r'<', r'\infty\,', r'???')
        eq4 = MathTex(r'\sum_{n=1}^\infty', r'\frac{1}{n^3\sin^2n}', r'=', r'\infty\,', r'???')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][1], eq1[1][1]).set_color(col_op)
        VGroup(eq1[1][-5:-2]).set_color(col_trig)
        VGroup(eq1[0][2], eq1[1][-1], eq1[1][-7]).set_color(col_x)
        VGroup(eq1[0][-1], eq1[1][0], eq1[1][3], eq1[1][-2]).set_color(col_num)
        VGroup(eq1[0][0], eq3[3]).set_color(col_special)
        VGroup(eq1[0][3]).set_color(WHITE*0.85)
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

        ax2.move_to(ax1.c2p(0,0)-ax2.c2p(0,0))
        ax2.next_to(ax1.y_axis, DOWN, buff=0.2, coor_mask=UP)

        MathTex.set_default(stroke_width=1.5, font_size=50)
        eq1 = MathTex(r'\frac1{n^3\sin^2 n}')
        eq2 = MathTex(r'\sum_{n\le N}', r'\frac1{n^3\sin^2 n}')

        VGroup(eq1[0][0], eq1[0][3], eq1[0][-2]).set_color(col_num)
        VGroup(eq1[0][1], eq2[0][0]).set_color(col_op)
        VGroup(eq1[0][2], eq1[0][-1], eq2[0][1], eq2[0][-1]).set_color(col_x)
        VGroup(eq1[0][-5:-2]).set_color(col_trig)

        n = 1_000
        xvec = np.linspace(1., n, n)
        svec = np.sin(xvec)
        termvec = 1. / (xvec ** 3 * svec * svec)
        sumvec = np.cumsum(termvec)

        n1 = 10
        xscale1 = 1/(n1+1)
        xticks1 = mh.get_xticks(ax2, range(0, n1+1), label_color=col_num, scalex=xscale1)
        VGroup(ax1, ax2, xticks1).move_to(ORIGIN)
        origin1 = ax1.c2p(0,0)
        origin2 = ax2.c2p(0,0)
        xstep = ax1.c2p(1,0) - origin1
        ystep2 = ax2.c2p(0,1) - origin2

        eq1.move_to(ax2.c2p(0.6, 0.5))
        eq2.move_to(ax1.c2p(0, 0.4)).align_to(eq1, RIGHT)

        eq1 = mh.eq_shadow(eq1, bg_stroke_width=10, bg_z_index=9.9, fg_z_index=10)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=10, bg_z_index=9.9, fg_z_index=10)

        xticks1_1 = VGroup(_[0].copy() for _ in xticks1[1:]).shift(origin1-origin2)

        ymax1_1 = sumvec[n1-1]
        print('ymax1:', ymax1_1)
        yscale1_1 = 1./ymax1_1
        yscale1_2 = yscale1_1#1./ ymax1
        eps = 0.006
        yticks1_2 = mh.get_yticks(ax2, [0, 1, 2, 3], label_color=col_num, scaley=yscale1_1)
        yticks1_1 = mh.get_yticks(ax1, [0, 1, 2, 3], label_color=col_num, scaley=yscale1_1)

        print(termvec[:n1])

        bluefill = BLUE*0.5
        orangefill = ORANGE*0.6

        recs1 = [Rectangle(width=xstep[0]*xscale1, height=termvec[i] * ystep2[1]*yscale1_2,
                          stroke_width=0, stroke_opacity=0,
                          fill_color=orangefill, fill_opacity=1).set_z_index(2)
                .next_to(ax2.c2p((i+1)*xscale1,eps), UR, buff=0) for i in range(n1)]
        recs2 = [recs1[i].copy().set_fill(bluefill)
                 .next_to(ax1.c2p((i+1)*xscale1, sumvec[i]*yscale1_1), DR, buff=0) for i in range(n1)]


        self.add(ax1, ax2, xticks1, xticks1_1, eq1, yticks1_2)

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

        # self.play(Create(plt1, rate_func=linear),
        #           FadeIn(area1))
        # self.play(mh.rtransform(recs0, recs1))
        self.play(mh.rtransform(plt2, plt1, area2, area1))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(VGroup(*recs1), VGroup(*recs2), plt1.copy(), plt3),
                  mh.rtransform(eq1[0].copy(), eq2[1], copy_colors=True),
                  FadeIn(eq2[0], shift=mh.diff(eq1[0], eq2[1])),
                  run_time=2),
                  Succession(Wait(1.2), FadeIn(yticks1_1)))
        self.wait(0.1)
        self.play(FadeIn(area3, rate_func=linear))

        self.wait()


class FHSum(Scene):
    def construct(self):
        ax = Axes(x_range=[0, 1.1], x_length=12, y_length=6, y_range=[0,1.1], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': True,
            "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
            "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
        }).set_z_index(1)

        n = 1_000_000_000
        xvec = np.linspace(1., n, n)
        svec = np.sin(xvec)
        termvec = 1. / (xvec ** 3 * svec * svec)
        sumvec = np.cumsum(termvec)

        for i in [3, 22, 355, n]:
            print(i, sumvec[i-2], termvec[i-1], sumvec[i-1])
        print('100', sumvec[99])
        print('200', sumvec[199])
        print('300', sumvec[299])
        print('1 000', sumvec[999])
        print('10 000', sumvec[9999])
        print('100 000', sumvec[99999])
        print('1 000 000', sumvec[1_000_000-1])
        print(sumvec[-1])

        i0 = 10
        xscale = 1. / xvec[i0-1]
        yscale = 1. / sumvec[i0-1]
        plt = ax.plot_line_graph(xvec[:i0] * xscale, sumvec[:i0] * yscale, add_vertex_dots=False,
                                 stroke_width=6, stroke_color=BLUE).set_z_index(4)

        self.add(ax, plt)
        self.wait()


if __name__ == "__main__":
    n = 1_000_000
    xvec = np.linspace(1., n, n)
    svec = np.sin(xvec)
    terms = 1. / (xvec**3 * svec * svec)
    print(sum(terms))