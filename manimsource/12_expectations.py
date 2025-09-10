from manim import *
import numpy as np
import math
import sys
import scipy as sp

sys.path.append('../')
import manimhelper as mh

H = LabeledDot(Text("H", color=BLACK, font='Helvetica', weight=SEMIBOLD), radius=0.35, color=BLUE).scale(1.5)
T = LabeledDot(Text("T", color=BLACK, font='Helvetica', weight=SEMIBOLD), radius=0.35, color=YELLOW).scale(1.5)


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
        gp = VGroup(eq1.copy(), eq2).arrange(RIGHT, buff=1).move_to(ORIGIN)
        mh.align_sub(eq1.copy(), eq1[1], gp[0][1], coor_mask=UP)
        eq1_1 = eq1.copy()
        self.add(eq1)
        self.wait(0.1)
        self.play(mh.transform(eq1, gp[0]),
                  mh.fade_replace(eq1_1[0], eq2[0]),
                  mh.rtransform(eq1_1[1], eq2[1], eq1_1[2][:4], eq2[2][:4], eq1_1[2][-1], eq2[2][-1]),
                  mh.fade_replace(eq1_1[2][-2], eq2[2][-2]),
                  run_time=1.6)
        self.wait()


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
        eq1_1 = eq1[2][1:7].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.add(eq1_1)
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq1_1, eq1[2][1:7], run_time=1.6),
                  FadeIn(eq1[:2], eq1[2][0], eq1[2][7:], run_time=1.6),
                    lag_ratio=0.5))
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


class Head(Scene):
    def construct(self):
        self.add(H.scale(2))

class Tail(Scene):
    def construct(self):
        self.add(T.scale(2))

class Alicea(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    eq_str = r'a'
    eq_col = RED

    def construct(self):
        eq1 = MathTex(self.eq_str, font_size=120, stroke_width=6, color=self.eq_col).set_z_index(1)
        eq2 = MathTex(self.eq_str, font_size=120, stroke_width=12, stroke_color=BLACK).set_z_index(0)
        self.add(eq1, eq2)

class Bobb(Alicea):
    eq_str = r'b'
    eq_col = BLUE

class AliceExpected(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=80)
        eq1 = MathTex(r'p(b)', r'=', r'\left(\frac12\right)^{b+1}')
        eq1_1 = eq1.copy()
        eq2 = Tex(r'\sf Alice receives ', r'$4^b$')

        for x in eq2[0][:5]: x.set(stroke_width=2)
        eq2.next_to(eq1, DOWN)
        eq3 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\sum_b4^bp(b)')
        mh.align_sub(eq3, eq3[0][2:-3], eq2[0], coor_mask=UP)
        eq4 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\sum_{b=0}^{a-1}4^b\left(\frac12\right)^{b+1}')
        eq4.move_to(VGroup(eq1, eq2, eq3), coor_mask=UP)
        VGroup(eq4[0][-2]).set_color(RED)
        VGroup(eq4[2][-3], eq4[2][1], eq4[2][3]).set_color(BLUE)
        eq4_1 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\sum_b4^b')
        mh.align_sub(eq4_1, eq4_1[1], eq4[1])
        eq5 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\sum_{b=0}^{a-1}\left(\frac42\right)^b\frac12')
        eq6 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\frac12\sum_{b=0}^{a-1}2^b')
        mh.align_sub(eq6, eq6[1], eq4[1])
        mh.align_sub(eq5, eq5[1], eq4[1])
        eq7 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\frac12\left(2^a-1\right)')
        mh.align_sub(eq7, eq7[1], eq4[1], coor_mask=UP)
        VGroup(eq5[0][-2], eq5[2][0], eq6[0][-2], eq6[2][3], eq7[0][-2], eq7[2][5],
               eq3[0][-2], eq2[0][:5], eq4[2][0]).set_color(RED)
        VGroup(eq5[2][-4], eq5[2][4], eq6[2][-1], eq6[2][7], eq4[2][4], eq4[2][8], eq4[2][-3], eq4_1[2][1],
               eq3[2][-2], eq3[2][-5], eq3[2][-7], eq2[1][-1], eq1[2][-3], eq1[0][2]).set_color(BLUE)
        for _ in VGroup(eq3, eq4, eq4_1, eq5, eq6, eq7): _[0][-2].set(stroke_width=2)
        eqs = VGroup(eq1, eq2, eq3, eq4, eq4_1, eq5, eq6, eq7).set_z_index(1)

        box = SurroundingRectangle(eqs, corner_radius=0.2, fill_color=BLACK, fill_opacity=0.85, stroke_width=0)

        self.add(eq1, box)
        self.wait(0.1)
        self.play(FadeIn(eq2), run_time=1)
        self.wait(0.1)
        self.play(LaggedStart(AnimationGroup(eq2[0].animate.move_to(eq3[0][2:-3]),
                  mh.rtransform(eq2[1][:], eq3[2][2:4]), run_time=1.2),
                  FadeIn(eq3[0][:2], eq3[0][-3:], eq3[1], eq3[2][:2], eq3[2][4:], run_time=1.2),
                              lag_ratio=0.5)
                  )
        self.wait(0.1)
        self.play(FadeOut(eq1[:2]),
                  AnimationGroup(mh.rtransform(eq1[2][:], eq4[2][-8:], eq3[2][:2], eq4_1[2][:2],
                                eq3[2][2:4], eq4[2][7:9], eq3[1], eq4[1],
                                eq3[0][:2], eq4[0][:2], eq3[0][-3:], eq4[0][-3:]),
                  eq2[0].animate.move_to(eq4[0][2:-3]),
                  FadeOut(eq3[2][-4:], shift=mh.diff(eq3[2][2], eq4[2][7]) * UP), run_time=1.4))
        self.wait(0.1)
        self.play(mh.rtransform(eq4_1[2][:2], eq4[2][3:5]),
                  FadeIn(eq4[2][:3], eq4[2][5:7]))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[2][:7], eq5[2][:7], eq4[2][7], eq5[2][8],
                                eq4[2][8], eq5[2][12], eq4[2][9], eq5[2][7],
                                eq4[2][13], eq5[2][11], eq4[2][11:13].copy(), eq5[2][9:11],
                                eq4[2][10:13], eq5[2][13:], eq4[1], eq5[1],
                                eq4[0][:2], eq5[0][:2], eq4[0][-3:], eq5[0][-3:]),
                  mh.rtransform(eq4[2][14], eq5[2][12]),
                  FadeOut(eq4[2][15:]),
                  eq2[0].animate.move_to(eq5[0][2:-3])
                  )
        self.wait(0.1)
        eq6_1 = mh.align_sub(eq6[2][10:12].copy(), eq6[2][10], eq5[2][9], coor_mask=RIGHT)
        self.play(FadeOut(eq5[2][7], eq5[2][11], eq5[2][9]),
                  eq5[2][12].animate.move_to(eq6_1[1]),
                  eq5[2][10].animate.move_to(eq6_1[0]),
                  FadeOut(eq5[2][8], target_position=eq6_1[0]))
        self.play(mh.rtransform(eq5[2][:7], eq6[2][3:10], eq5[2][10], eq6[2][10],
                                eq5[2][12], eq6[2][11], eq5[2][-3:], eq6[2][:3]), run_time=1.2)
        self.wait(0.1)
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq6[2][:3], eq7[2][:3], eq6[2][10], eq7[2][4],
                                eq6[2][3], eq7[2][5], eq5[1], eq7[1], eq5[0][:2], eq7[0][:2], eq5[0][-3:], eq7[0][-3:]),
                  eq2[0].animate.move_to(eq7[0][2:-3]),
                  FadeOut(eq6[2][4:10]),
                  FadeOut(eq6[2][11], target_position=eq7[2][5]),
                    run_time=1.6),
                  FadeIn(eq7[2][3], eq7[2][6:], run_time=1.4), lag_ratio=0.5))
        self.wait(0.5)

        eq7_1 = eq7.copy().align_to(eq1_1, UP).shift(DOWN*0.1)

        eq10 = MathTex(r'\mathbb P(b > a\vert a)', r'=', r'\left(\frac12\right)^{a+1}')
        eq10[0][2].set_color(BLUE)
        VGroup(eq10[0][4], eq10[0][6], eq10[2][-3]).set_color(RED)
        eq10.next_to(eq7_1, DOWN, buff=0.1)

        eq11 = MathTex(r'\mathbb E[{\sf Alice\ pays}\vert a]', r'=', r'4^a\left(\frac12\right)^{a+1}')
        mh.align_sub(eq11, eq11[1], eq10[1], coor_mask=UP)
        VGroup(eq11[2][1], eq11[2][-3]).set_color(RED)
        eq12 = MathTex(r'\mathbb E[{\sf Alice\ pays}\vert a]', r'=', r'\frac12\left(\frac42\right)^a')
        mh.align_sub(eq12, eq12[1], eq11[1], coor_mask=UP)
        VGroup(eq12[0][2:7], eq12[0][-2], eq12[2][-1]).set_color(RED)
        eq13 = MathTex(r'\mathbb E[{\sf Alice\ pays}\vert a]', r'=', r'\frac122^a')
        VGroup(eq13[0][2:7], eq13[0][-2], eq13[2][-1]).set_color(RED)
        mh.align_sub(eq13, eq13[1], eq12[1], coor_mask=UP)
        reds = [_[0][-2] for _ in [eq10, eq11, eq12, eq13]]
        for _ in [eq11, eq12, eq13]: reds += _[0][2:7]
        for _ in reds:
            _.set(stroke_width=2).set_color(RED)
        VGroup(eq7_1, eq10, eq11, eq12, eq13).set_z_index(1)

        self.play(mh.transform(eq7[1:], eq7_1[1:], eq7[0][:2], eq7_1[0][:2], eq7[0][-3:], eq7_1[0][-3:]),
                  eq2[0].animate.move_to(eq7_1[0][2:-3]), FadeIn(eq10[0]))
        self.wait(0.1)
        self.play(FadeIn(eq10[1:]))
        self.wait(0.1)
        self.play(mh.fade_replace(eq10[0][0], eq11[0][0]),
                  mh.stretch_replace(eq10[0][1], eq11[0][1]),
                  mh.stretch_replace(eq10[0][-1], eq11[0][-1]),
                  mh.rtransform(eq10[1], eq11[1], eq10[0][-3:-1], eq11[0][-3:-1],
                                eq10[2][:], eq11[2][2:]),
                  FadeIn(eq11[2][:2], shift=mh.diff(eq10[2][0], eq11[2][2]) * RIGHT),
                  FadeOut(eq10[0][2:-3], target_position=eq11[0][2:-3]),
                  FadeIn(eq11[0][2:-3]))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[:2], eq12[:2], eq11[2][0], eq12[2][4],
                                eq11[2][1], eq12[2][8], eq11[2][2], eq12[2][3],
                                eq11[2][6], eq12[2][7], eq11[2][3:6], eq12[2][:3],
                                eq11[2][4:6].copy(), eq12[2][5:7]),
                  mh.rtransform(eq11[2][-3], eq12[2][-1]),
                  FadeOut(eq11[2][-2:], shift=mh.diff(eq11[2][-3], eq12[2][-1])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(eq12[:2], eq13[:2], eq12[2][-1], eq13[2][-1], eq12[2][:3], eq13[2][:3],
                                eq12[2][6], eq13[2][3]),
                  FadeOut(eq12[2][5], eq12[2][3], eq12[2][7]),
                  FadeOut(eq12[2][4], target_position=eq13[2][3]))
        self.wait(0.1)

        eq20 = MathTex(r'\mathbb E[{\sf Alice\ profit}\vert a]', r'=', r'\frac12(2^a-1) - \frac122^a').set_z_index(1)
        eq20[2][9:].next_to(eq20[2][3:9], DOWN, buff=0.1)
        eq20.move_to(ORIGIN)
        mh.align_sub(eq20, eq20[1], eq7[1], coor_mask=UP)
        VGroup(eq20[0][2:7], eq20[0][-2], eq20[2][5], eq20[2][-1]).set_color(RED)
        for x in eq20[0][2:7]: x.set(stroke_width=2)

        eq21 = MathTex(r'\mathbb E[{\sf Alice\ profit}\vert a]', r'=', r'-\frac12').set_z_index(1)
        VGroup(eq21[0][2:7], eq21[0][-2]).set_color(RED)
        for x in eq21[0][2:7]: x.set(stroke_width=2)
        for _ in (eq20, eq21): _[0][-2].set(stroke_width=2)

        eq20_1 = eq20[0][7:-3].copy()
        self.play(mh.rtransform(eq2[0][:5], eq20[0][2:7], eq7[0][:2], eq20[0][:2], eq7[0][-3:], eq20[0][-3:],
                                eq7[1], eq20[1], eq7[2][:], eq20[2][:9], eq13[2][:], eq20[2][10:]),
                  mh.rtransform(eq13[0][:7], eq20[0][:7], eq13[0][-3:], eq20[0][-3:]),
                  mh.fade_replace(eq13[0][7:-3], eq20[0][7:-3]),
                  mh.fade_replace(eq13[1][0], eq20[2][9]),
                  mh.fade_replace(eq2[0][5:], eq20_1)
                  )
        self.remove(eq20_1)
        self.wait(0.1)
        del1 = eq20[2][4:6]
        del2 = eq20[2][-2:]
        line1 = Line(del1.get_corner(DL), del1.get_corner(UR), stroke_width=7, stroke_color=GREEN).set_z_index(5)
        line2 = Line(del2.get_corner(DL), del2.get_corner(UR), stroke_width=7, stroke_color=GREEN).set_z_index(5)
        self.play(Create(line1, rate_func=linear, run_time=0.75))
        self.play(Create(line2, rate_func=linear, run_time=0.75))
        self.wait(0.1)
        self.play(FadeOut(eq20[2][4:6], eq20[2][-6:], line1, line2))
        self.wait(0.1)
        self.play(mh.rtransform(eq20[:2], eq21[:2], eq20[2][:3], eq21[2][1:4], eq20[2][6], eq21[2][0]),
                  mh.rtransform(eq20[2][7], eq21[2][1]),
                  FadeOut(eq20[2][8]),
                  FadeOut(eq20[2][3], shift=mh.diff(eq20[2][1], eq21[2][2]) * RIGHT),
                  run_time=1.5)
        self.wait(0.1)
        eq30 = MathTex(r'\mathbb E[{\sf Bob\ profit}\vert b]', r'=', r'-\frac12').set_z_index(1)
        blues = eq30[0][2:5] + [eq30[0][-2]]
        for _ in blues: _.set_color(BLUE).set(stroke_width=2)
        mh.align_sub(eq30, eq30[1], eq21[1]).next_to(eq21, DOWN, coor_mask=UP)
        self.play(FadeIn(eq30), run_time=1)
        self.wait(0.1)

        eq40 = MathTex(r'{\sf Alice\ profit}', r'=', r'W')
        eq41 = MathTex(r'{\sf Bob\ profit}', r'=', r'-W')
        eq42 = MathTex(r'\mathbb E[W\vert a]', r'=-\frac12')
        eq43 = MathTex(r'\mathbb E[W\vert b]', r'=\frac12')
        for _ in eq40[0][:5] + eq42[0][-2]: _.set_color(RED).set(stroke_width=2)
        for _ in eq41[0][:3] + eq43[0][-2]: _.set_color(BLUE).set(stroke_width=2)
        eq41.next_to(eq40, RIGHT, buff=0.7)
        mh.align_sub(VGroup(eq40, eq41), eq40[1], eq21[1]).move_to(ORIGIN, coor_mask=RIGHT).shift(DOWN*0.3)
        eq42.next_to(eq40, DOWN)
        eq43.next_to(eq41, DOWN)
        mh.align_sub(eq43, eq43[1][0], eq42[1][0], coor_mask=UP)
        eqs = VGroup(eq40, eq41, eq42, eq43).set_z_index(1)
        box2 = SurroundingRectangle(eqs, corner_radius=0.2, fill_color=BLACK, fill_opacity=0.85, stroke_width=0,
                                    buff=0.2)


        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq21[0][2:12], eq40[0][:], eq30[0][2:10], eq41[0][:],
                                                           box, box2),
                  FadeOut(eq21[0][:2], eq21[0][-3:], eq21[1:], eq30[0][:2], eq30[0][-3:], eq30[1:]),
                                 run_time=1.6),
                  FadeIn(eq40[1:], eq41[1:], run_time=1.2), lag_ratio=0.5))
        self.wait(0.1)
        self.play(FadeIn(eq42))
        self.wait(0.1)
        self.play(FadeIn(eq43))


        self.wait()