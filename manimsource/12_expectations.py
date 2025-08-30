from manim import *
import numpy as np
import math
import sys
import scipy as sp

sys.path.append('../')
import manimhelper as mh

class ExpectedValue(Scene):
    fs1 = 100
    def construct(self):
        MathTex.set_default(font_size=self.fs1)
        eq1 = Tex(r'random variable $X$')[0]
        eq1[-1].set_color(RED)
        eq2 = Tex(r'expected value $\mathbb E[X]$')[0]
        eq2[-2].set_color(RED)
        eq3 = MathTex(r'\mathbb E[X]', r'=', r'\sum_x p(x) x')
        eq3[0][2].set_color(RED)
        eq3[2][-1].set_color(RED)
        eq3.next_to(eq1, DOWN)
        self.add(eq1)
        self.wait(0.1)
        eq2.next_to(eq1, DOWN)
        gp = VGroup(eq1, eq2).copy().move_to(ORIGIN)
        self.play(mh.rtransform(eq1[:-1], gp[0][:-1], eq1[-1], gp[1][-2]),
                  FadeIn(gp[1][:-2], target_position=eq2[:-2]),
                  FadeIn(gp[1][-1], target_position=eq2[-1]),
                  run_time=0.7)
        eq1, eq2 = gp[0], gp[1]
        eq3.next_to(eq2, DOWN)
        self.wait(0.1)
        gp = VGroup(eq2[:-4].copy().move_to(ORIGIN, coor_mask=RIGHT), eq3).move_to(ORIGIN)
        self.play(mh.rtransform(eq2[:-4], gp[0], eq2[-4:], eq3[0][:]),
                  FadeIn(eq3[1:], rate_func=rush_into),
                  FadeOut(eq1[:-1], shift=mh.diff(eq2[0], gp[0][0])*UP),
                  run_time=1.6)
        eq2 = gp[0]
        self.wait(0.1)

        eq4 = MathTex(r'\mathbb E[X]', r'=', r'\int p(x) x\,dx')
        eq4[0][2].set_color(RED)
        eq4[2][-3].set_color(RED)
        mh.align_sub(eq4, eq4[1], eq3[1])
        eq2_1 = eq2.copy().next_to(eq4, UP, coor_mask=UP)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2][2:7], eq4[2][1:6]),
                  mh.transform(eq2, eq2_1),
                  FadeIn(eq4[2][6:], shift=mh.diff(eq3[2][6], eq4[2][5])),
                  FadeOut(eq3[2][1]),
                  #mh.rtransform(eq3[2][0], eq4[2][0]),
                  mh.stretch_replace(eq3[2][0], eq4[2][0]),
                  run_time=2)

        self.wait(0.1)

        def p(x):
            return math.exp(-x*x/2)

        xmin = -2.5
        xmax = 2.5
        ax = Axes(x_range=[xmin, xmax * 1.1], y_range=[0, 1.1], x_length=12, y_length=3,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "stroke_opacity": 1,
                               },
                  ).set_z_index(2)
        ax.next_to(eq4, DOWN)
        gp = VGroup(eq2.copy(), eq4.copy(), ax).move_to(ORIGIN, coor_mask=UP)
        plt1 = ax.plot(p, (xmin, xmax), stroke_color=BLUE, stroke_width=6).set_z_index(4)
        fill1 = ax.get_area(plt1, color=BLUE, opacity=0.5).set_z_index(3).set_stroke(opacity=0)

        eq5 = Tex(r'probability density', font_size=60)[0].next_to(gp[1][2][-2], DOWN, buff=0.7)
        arr1 = Arrow(eq5[1].get_top(), gp[1][2][1].get_center() + DOWN*0.2, buff=0.1, stroke_width=4)
        arr2 = Arrow(eq5[1].get_bottom(), ax.coords_to_point(.5, p(.5)), buff=0.1, stroke_width=4)

        self.play(LaggedStart(mh.transform(eq2, gp[0], eq4, gp[1], run_time=1.2),
                  AnimationGroup(FadeIn(ax), Create(plt1, rate_func=linear), run_time=1.6),
                  AnimationGroup(FadeIn(fill1), FadeIn(eq5, arr1, arr2), run_time=1, rate_func=linear),
                              lag_ratio=0.3))


        self.wait()

class ExpectedAB(Scene):
    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = WHITE
        Scene.__init__(self, *args, **kwargs)

    fs1 = 100

    def construct(self):
        cx = RED
        cy = BLUE
        MathTex.set_default(font_size=self.fs1)
        eq1 = MathTex(r'A, B', r'\sim', r'N(0,1)')
        eq1[0][0].set_color(cx)
        eq1[0][-1].set_color(cy)
        self.add(eq1)
        self.wait(0.1)
        eq2 = MathTex(r'{\rm Corr}(A, B)', r'=', r'1/2', font_size=80)
        eq2.next_to(eq1, DOWN)
        eq2[0][-4].set_color(cx)
        eq2[0][-2].set_color(cy)
        gp = Group(eq1.copy(), eq2).move_to(ORIGIN)
        self.play(FadeIn(eq2), mh.transform(eq1, gp[0]), run_time=1)
        self.wait()

class ABexp(Scene):
    def construct(self):
        MathTex.set_default(font_size=100)
        eq1 = MathTex(r'X', r'=', r'e^{\frac12A^2}')
        eq2 = MathTex(r'Y', r'=', r'e^{\frac12B^2}')
        eq1[0][0].set_color(RED)
        eq1[2][-2].set_color(RED)
        eq2[0][0].set_color(BLUE)
        eq2[2][-2].set_color(BLUE)
        gp = VGroup(eq1, eq2).arrange(RIGHT, buff=1).move_to(ORIGIN)
        self.add(gp)


class ExpectedXY(Scene):
    fs1 = 100
    def construct(self):
        cx = RED
        cy = BLUE
        MathTex.set_default(font_size=self.fs1)
        eq1 = MathTex(r'\mathbb E[Y\vert X]', r'=', r'\int p(y\vert X)y\,dy')
        eq1[0][2].set_color(cy)
        eq1[0][4].set_color(cx)
        eq1[2][3].set_color(cy)
        eq1[2][5].set_color(cx)
        eq1[2][7].set_color(cy)
        eq1[2][9].set_color(cy)
        self.add(eq1)
        self.wait(0.1)
        eq2 = MathTex(r'\mathbb E[Y\vert X]', r'=', r'2X')
        eq2[0][2].set_color(cy)
        eq2[0][4].set_color(cx)
        eq2[2][1].set_color(cx)
        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)
        eq2_1 = eq2[2].copy()
        mh.align_sub(eq2_1, eq2_1[1], eq1[2][5], coor_mask=RIGHT)
        self.play(FadeOut(eq1[2][:5], eq1[2][6:]),
                  FadeIn(eq2_1[0]), mh.rtransform(eq1[2][5], eq2_1[1]),
                  run_time=1.4)
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq2_1, eq2[2]),
                 run_time=1.5)
        self.wait(0.1)
        eq3 = MathTex(r'\mathbb E[X\vert Y]', r'=', r'2Y')
        eq3[0][2].set_color(cx)
        eq3[0][4].set_color(cy)
        eq3[2][1].set_color(cy)
        eq3.next_to(eq2, DOWN, buff=0.8)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=RIGHT)
        eq2_1 = eq2.copy()
        gp = VGroup(eq2.copy(), eq3).align_to(eq1, UP).shift(DOWN*0.2)
        self.play(mh.rtransform(eq2_1[0][:2], eq3[0][:2], eq2_1[0][3], eq3[0][3], eq2_1[0][5], eq3[0][5],
                                eq2_1[1], eq3[1], eq2_1[2][0], eq3[2][0]),
                  mh.transform(eq2, gp[0]),
                  mh.fade_replace(eq2_1[0][2], eq3[0][2]),
                  mh.fade_replace(eq2_1[0][4], eq3[0][4]),
                  mh.fade_replace(eq2_1[2][1], eq3[2][1]),
                  run_time=1.6
                  )


        return
        eq3 = MathTex(r'\mathbb E[Y\vert X]', r'=', r'2X')
        eq3[0][2].set_color(cy)
        eq3[0][4].set_color(cx)
        eq3[2][1].set_color(cx)
        gp = VGroup(eq1.copy(), eq2.copy(), eq3)
        gp[1].next_to(gp[0], RIGHT, buff=0.6)
        gp[2].next_to(gp[:2], DOWN, buff=0.8)
        gp.move_to(ORIGIN)
        eq3_1 = eq3[0].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(LaggedStart(mh.transform(eq1, gp[0], eq2, gp[1], run_time=2),
                  FadeIn(eq3_1, run_time=2),
                              lag_ratio=0.3))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq3_1, eq3[0], run_time=1.4),
                  FadeIn(eq3[1:], run_time=1.4), lag_ratio=0.2))
        self.wait(0.1)
        eq4 = MathTex(r'\mathbb E[X\vert Y]', r'=', r'2Y')
        eq4[0][2].set_color(cx)
        eq4[0][4].set_color(cy)
        eq4[2][1].set_color(cy)
        mh.align_sub(eq4, eq4[1], eq3[1]).next_to(eq3, DOWN, coor_mask=UP, buff=0.4)
        gp1 = VGroup(eq1, eq2, eq3)
        gp = VGroup(gp1.copy(), eq4).move_to(ORIGIN)
        eq3_1 = eq3.copy()
        self.play(mh.transform(gp1, gp[0]),
                  mh.rtransform(eq3_1[0][:2], eq4[0][:2], eq3_1[0][3], eq4[0][3], eq3_1[0][5], eq4[0][5],
                                eq3_1[1], eq4[1], eq3_1[2][0], eq4[2][0]),
                  mh.fade_replace(eq3_1[0][2], eq4[0][2]),
                  mh.fade_replace(eq3_1[0][4], eq4[0][4]),
                  mh.fade_replace(eq3_1[2][1], eq4[2][1]),
                  run_time=2)


        self.wait()