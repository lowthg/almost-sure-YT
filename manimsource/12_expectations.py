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
            return math.exp(-x*x)

        xmin = -2
        xmax = 2
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
