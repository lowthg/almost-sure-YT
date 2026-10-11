import numpy as np
from manim import *
import sys
from manim import ManimColor
from scipy.special import expi
from sympy import floor

from common_randomprime import *
import mpmath as mp

sys.path.append('../../')
import manimhelper as mh
from common.wigner import *

col_pi = col_special * 0.5 + ORANGE * 0.5
col_trig = PURPLE_A#*0.5+WHITE*0.5
col_txt = ManimColor( r'#FFAC2B')
col_prime = BLUE
col_zero = PINK * 0.7 + WHITE *0.3

def get_kwargs(**kwargs):
    return kwargs

class Chebyshev(Scene):
    bgcolor = GREY
    trcolor = BLACK
    def __init__(self, *args, **kwargs):
        config.background_color = self.trcolor if config.transparent else self.bgcolor
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)

        eq1 = MathTex(r'\pi(x)', r'=', r'{\sf number\ of\ primes}', r'{}\le x')
        eq2 = MathTex(r'\pi(x)', r'=', r'\sum_{p\le x}', r'1')
        eq3 = Tex(r'\sf sum over primes $p$', font_size=50, color=RED)
        eq4 = MathTex(r'\psi(x)', r'=', r'\sum_{p\le x}', r'\log p')
        eq5 = MathTex(r'\delta\pi(x)', r'\sim', r'\delta x/\log x')
        eq6 = MathTex(r'\delta\psi(x)', r'\sim', r'\delta x')
        eq7 = MathTex(r'{\sf prime\ number\ theorem\!\!:\ }', r'\psi(x)', r'\sim', r'x')
        eq8 = MathTex(r'{\sf Cram\acute{e}r\ model\!\!:\ }', r'{\rm Var}[\pi_R(x)]', r'\sim', r'x\log x')
        eq9 = MathTex(r'\psi(x)', r'=', r'\sum_{p\le x}', r'\log p', r'+', r'\sum_{p^2\le x}', r'\log p',
                      r'+', r'\sum_{p^3\le x}', r'\log p', r'+\cdots')
        eq10 = MathTex(r'\psi(x)', r'=', r'\sum_{p^r\le x}', r'\log p')
        eq11 = Tex(r'\sf prime powers, exponent $r\ge1$', font_size=50)
        eq11[0][:-3].set_color(RED)


        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)
        eq3.next_to(eq2[2][1], DOWN, buff=0.6).shift(RIGHT)

        VGroup(eq1, eq2, eq3).to_edge(DOWN, buff=0.4)

        arr1 = Arrow(eq3[0][4].get_top(), eq2[2][1].get_bottom(), stroke_width=6, stroke_color=RED, buff=0.1,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4)
        mh.align_sub(eq4, eq4[1], eq2[1], coor_mask=UP)
        eq5.next_to(eq4, DOWN, buff=0.3)
        mh.align_sub(eq6, eq6[1], eq5[1])
        mh.align_sub(eq7, eq7[2], eq6[1], coor_mask=UP)
        mh.align_sub(eq8, eq8[2], eq7[2], coor_mask=UP)
        mh.align_sub(eq9, eq9[1], eq4[1], coor_mask=UP)
        mh.align_sub(eq10, eq10[1], eq9[1], coor_mask=UP)
        eq11.next_to(eq10[2][1], DOWN, buff=0.6).shift(RIGHT)
        arr2 = Arrow(eq11[0][8].get_top(), eq10[2][1].get_bottom(), stroke_width=6, stroke_color=RED, buff=0.1,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4)

        VGroup(eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, arr1, arr2).set_z_index(4)

        eq9_2 = eq9.copy()
        eq9_1 = eq9[:10].copy().move_to(ORIGIN, coor_mask=RIGHT)
        eq9[:7].move_to(ORIGIN, coor_mask=RIGHT)

        gp1 = VGroup(eq1, eq2, eq4, eq5, eq6, eq7)
        box_kwargs = get_kwargs(stroke_width=0, stroke_opacity=0,
                                    fill_color=BLACK, fill_opacity=0.6, buff=0.2, corner_radius=0.2)
        box1 = SurroundingRectangle(gp1, **box_kwargs)
        box2 = SurroundingRectangle(VGroup(gp1, eq8), **box_kwargs)
        box3 = SurroundingRectangle(VGroup(gp1, eq9_1), **box_kwargs)
        box4 = SurroundingRectangle(VGroup(gp1, eq9_2), **box_kwargs)
        box5 = box1.copy()

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq1[0][0], eq4[0][0], eq5[0][1], eq6[0][1], eq8[1][4:6]).set_color(col_WVD)
        VGroup(eq1[0][2], eq1[3][1], eq5[0][3], eq5[2][1], eq5[2][-1], eq8[1][7], eq8[3][0], eq8[3][-1],
               ).set_color(col_x)
        VGroup(eq1[2], eq7[0], eq8[0]).set_color(col_txt)
        VGroup(eq2[2][0], eq5[0][0], eq5[2][:3:2]).set_color(col_op)
        VGroup(eq2[2][1], eq3[0][-1], eq4[3][3]).set_color(col_p)
        VGroup(eq2[-1], eq9[5][2], eq9[8][2], eq11[0][-1]).set_color(col_num)
        VGroup(eq4[3][:3], eq8[3][1:4], eq5[2][3:6]).set_color(col_trig)
        VGroup(eq8[1][:3]).set_color(col_txt2)
        VGroup(eq10[2][2], eq11[0][-3]).set_color(RED*0.5+WHITE*0.5)

        self.add(eq1, box1)
        self.play(FadeOut(eq1[2]),
                  AnimationGroup(mh.rtransform(eq1[:2], eq2[:2], eq1[3][0], eq2[2][-2]),
                  mh.stretch_replace(eq1[3][-1], eq2[2][-1]),
                  FadeIn(eq2[2][1], shift=mh.diff(eq1[3][0], eq2[2][-2])),
                                 run_time=1.5),
                  Succession(Wait(1), FadeIn(eq2[2][0], eq2[3], eq3, arr1))
                  )
        self.wait(0.1)
        self.play(FadeOut(arr1, eq3))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[0][1:], eq4[0][1:], eq2[1:3], eq4[1:3]),
                  mh.fade_replace(eq2[0][0], eq4[0][0], coor_mask=RIGHT),
                  mh.fade_replace(eq2[-1], eq4[-1], coor_mask=RIGHT))
        self.wait(0.1)
        self.play(FadeIn(eq5))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[0][0], eq6[0][0], eq5[0][2:], eq6[0][2:], eq5[1], eq6[1],
                                eq5[2][:2], eq6[2][:]),
                  mh.fade_replace(eq5[0][1], eq6[0][1], coor_mask=RIGHT),
                  FadeOut(eq5[2][2:], shift=mh.diff(eq5[2][1], eq6[2][1])),
                  )
        self.wait(0.1)
        eq7_1 = eq7[1:].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq6[0][1:], eq7_1[0][:], eq6[1], eq7_1[1], eq6[2][1], eq7_1[2][0]),
                  FadeOut(eq6[0][0], shift=mh.diff(eq6[0][1], eq7_1[0][0])),
                  FadeOut(eq6[2][0], shift=mh.diff(eq6[2][1], eq7_1[2][0])))
        self.wait(0.1)
        self.play(mh.rtransform(eq7_1, eq7[1:], run_time=1.6),
                  Succession(Wait(0.9), FadeIn(eq7[0])))
        self.wait(0.1)
        self.play(FadeOut(eq7))
        self.wait(0.1)
        self.play(mh.rtransform(box1, box2),
                  Succession(Wait(0.5), FadeIn(eq8)))
        self.wait(0.1)
        self.play(FadeOut(eq8))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq4[:4], eq9[:4], eq4[3].copy(), eq9[6],
                                eq4[2][:2].copy(), eq9[5][:2], eq4[2][-2:].copy(), eq9[5][-2:]),
                  FadeIn(eq9[5][2], shift=mh.diff(eq4[2][1], eq9[5][1])),
                                 run_time=1.4),
                  Succession(Wait(0.7), FadeIn(eq9[4])),
                  )
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq9[:7], eq9_1[:7], eq9[6].copy(), eq9_1[9],
                                eq9[5][:2].copy(), eq9_1[8][:2], eq9[5][-2:].copy(), eq9_1[8][-2:],
                                               box2, box3),
                  mh.stretch_replace(eq9[5][2].copy(), eq9_1[8][2]),
                                 run_time=1.3),
                  Succession(Wait(0.8), FadeIn(eq9_1[7]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq9_1[:10], eq9_2[:10], box3, box4),
                  Succession(Wait(0.4), FadeIn(eq9_2[10:])))
        eq9 = eq9_2
        self.wait(0.1)
        eq10_1 = eq10[2][2].copy()
        self.play(AnimationGroup(mh.rtransform(eq9[:2], eq10[:2], eq9[2][:2], eq10[2][:2], eq9[2][-2:], eq10[2][-2:],
                                eq9[3], eq10[3], box4, box5),
                  mh.rtransform(eq9[5][:2], eq10[2][:2], eq9[5][-2:], eq10[2][-2:], eq9[6], eq10[3]),
                  mh.fade_replace(eq9[5][2], eq10[2][2]),
                  mh.rtransform(eq9[8][:2], eq10[2][:2], eq9[8][-2:], eq10[2][-2:], eq9[9], eq10[3]),
                  mh.fade_replace(eq9[8][2], eq10_1),
                  FadeOut(eq9[-1], shift=mh.diff(eq9[9], eq10[3])),
                  FadeOut(eq9[4], shift=mh.diff(eq9[4], eq10[3])*RIGHT),
                  FadeOut(eq9[7], shift=mh.diff(eq9[6], eq10[3])*RIGHT),
                                 run_time=1.3),
                  Succession(Wait(1.1), FadeIn(eq11, arr2))
                  )
        self.remove(eq10_1)
        self.wait(0.1)
        self.play(FadeOut(eq11, arr2),
                  Succession(Wait(0.2), eq10.animate.move_to(box5, coor_mask=UP).shift(DOWN*0.1)))
        self.wait()

class ReSGeOne(Chebyshev):
    def construct(self):
        eq = MathTex(r'\Re[s] > 1', font_size=80, stroke_width=2)
        eq[0][-1].set_color(col_num)
        eq[0][-4].set_color(col_angle)
        eq[0][0].set_color(col_op)
        self.add(eq)

class EulerProduct(Chebyshev):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=70)
        eq = MathTex(r'\zeta(s)', r'=', r'\prod_{p{\sf\ prime} }', r'\frac1{1-\frac1{p^s}}')
        eq.set_z_index(1).to_edge(DOWN, buff=0.4)
        VGroup(eq[0][0]).set_color(col_WVD)
        VGroup(eq[0][2], eq[3][7]).set_color(col_angle)
        VGroup(eq[2][0], eq[3][5]).set_color(col_op)
        VGroup(eq[2][1], eq[3][6]).set_color(col_p)
        VGroup(eq[3][:6:2]).set_color(col_num)
        eq[2][-5:].set_color(col_txt*0.4+WHITE*0.6)
        box = SurroundingRectangle(eq, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   buff=0.2, corner_radius=0.2)

        self.add(eq, box)

col_zero = PINK * 0.7 + WHITE *0.3

class Explicit(Chebyshev):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)

        eq1 = MathTex(r'\psi(x)', r'=', r'x - \sum_{\rho}\frac{x^\rho}{\rho} - \log 2\pi')
        eq2 = Tex(r'\sf sum over zeta zeros')
        eq3 = MathTex(r'\psi(x)', r'=', r'x -\sum_{\rho}\frac{x^\rho}{\rho}',
                      r'-\sum_{n=1}^\infty\frac{x^{-2n} }{-2n} }', r'-\log2\pi')
        eq4 = Tex(r'\sf non-trivial zeta zeros')
        eq5 = MathTex(r'\psi(x)', r'=', r'x -\sum_{\rho}\frac{x^\rho}{\rho}',
                      r'+', r'\frac12', r'\sum_{n=1}^\infty\frac{(x^{-2})^n}{n} }', r'-\log2\pi')
        mh.font_size_sub(eq5, 4, 50)
        eq6 = MathTex(r'\psi(x)', r'=', r'x -\sum_{\rho}\frac{x^\rho}{\rho}',
                      r'-', r'\frac12', r'\log(1-x^{-2})', r'-\log2\pi')
        mh.font_size_sub(eq6, 4, 50)
        eq7 = MathTex(r'\psi(x)-x', r'\approx', r'-\sum_\rho\frac{x^\rho}{\rho}')
        eq8 = MathTex(r'\rho', r'=', r'\sigma+i\gamma')
        eq9 = MathTex(r'\psi(x)-x', r'\approx', r'-\sum_\rho\frac{x^\sigma x^{i\gamma} }{\rho}')
        eq10 = MathTex(r'\psi(x)-x', r'\approx', r'-\sum_\rho\frac{x^\sigma e^{i\gamma\log x} }{\rho}')
        eq11 = MathTex(r'\rho', r'=', r'\sigma+i\gamma', r'=', r'-\lvert\rho\rvert e^{i\theta}')
        eq12 = MathTex(r'\psi(x)-x', r'\approx', r'\sum_\rho\frac{x^\sigma e^{i\gamma\log x} }{\lvert\rho\rvert e^{i\theta} }')
        eq13 = MathTex(r'\psi(x)-x', r'\approx', r'\sum_\rho\frac{x^\sigma}{\lvert\rho\rvert} e^{i(\gamma\log x-\theta)}')
        eq14 = MathTex(r'\psi(x)-x', r'\approx', r'\sum_{\rho\colon\gamma > 0}', r'\frac{2x^\sigma}{\lvert\rho\rvert}', r'\cos(\gamma\log x-\theta)')
        eq15 = MathTex(r'\rho', r'=', r'{\scriptstyle\frac12}+i\gamma', r'=', r'-\lvert\rho\rvert e^{i\theta}')
        eq16 = MathTex(r'\psi(x)-x', r'\approx', r'\sum_{\rho\colon\gamma > 0}', r'\frac{2x^{\frac12} }{\lvert\rho\rvert}', r'\cos(\gamma\log x-\theta)')
        eq17 = MathTex(r'\psi(x)-x', r'\approx', r'\sqrt x', r'\sum_{\rho\colon\gamma > 0}', r'\frac{2}{\lvert\rho\rvert}', r'\cos(\gamma\log x-\theta)')
        eq18 = MathTex(r'\psi(x)-x', r'\approx', r'\sqrt x', r'\sum_{\rho\colon\gamma > 0}', r'\frac{2}{\lvert\rho\rvert}', r'\cos(\gamma y-\theta)')

        eq1.to_edge(DOWN, buff=0.6)
        eq2.next_to(eq1[2][3], DOWN, buff=0.4)
        mh.align_sub(eq3, eq3[1], eq1[1], coor_mask=UP)
        arr1 = Arrow(eq2[0][5].get_top(), eq1[2][3].get_bottom(), stroke_width=6, stroke_color=RED, buff=0.1,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4)
        eq4.next_to(eq3[2][3], DOWN, buff=0.4)
        arr2 = Arrow(eq4[0][6].get_top(), eq3[2][3].get_bottom(), stroke_width=6, stroke_color=RED, buff=0.1,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4)
        mh.align_sub(eq5, eq5[1], eq3[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[1], eq3[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[1], eq3[1], coor_mask=UP)
        eq8.next_to(eq7, DOWN, buff=0.3)
        mh.align_sub(eq9, eq9[1], eq3[1], coor_mask=UP)
        mh.align_sub(eq10, eq10[1], eq3[1], coor_mask=UP)
        mh.align_sub(eq11, eq11[1], eq8[1], coor_mask=UP)
        mh.align_sub(eq12, eq12[1], eq10[1])
        eq12[2].align_to(eq10[2][1:], LEFT)
        mh.align_sub(eq13, eq13[1], eq10[1], coor_mask=UP)
        mh.align_sub(eq14, eq14[1], eq10[1], coor_mask=UP)
        mh.align_sub(eq15, eq15[1], eq11[1], coor_mask=UP)
        mh.align_sub(eq16, eq16[1], eq10[1], coor_mask=UP)
        mh.align_sub(eq17, eq17[1], eq10[1], coor_mask=UP)
        mh.align_sub(eq18, eq18[1], eq10[1], coor_mask=UP)

        gp = VGroup(eq1, eq2, eq3, arr1, eq4, arr2, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16,
                    eq17, eq18)
        gp.set_z_index(2).to_edge(DOWN)
        boxargs = get_kwargs(stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    buff=0.2, corner_radius=0.2)
        box1 = SurroundingRectangle(eq1, eq3[3][1], **boxargs)
        box2 = SurroundingRectangle(VGroup(eq1, eq3), **boxargs)
        box3 = SurroundingRectangle(VGroup(eq1, eq3, eq5), **boxargs)
        box4 = SurroundingRectangle(VGroup(eq1, eq3, eq6), **boxargs)
        box5 = Rectangle(width=config.frame_width*1.1, height=config.frame_height*1.1, fill_color=BLACK, fill_opacity=1,
                         stroke_width=0)
        box6 = SurroundingRectangle(VGroup(eq3[3][1], eq7), **boxargs)
        box7 = SurroundingRectangle(VGroup(eq3[3][1], eq10, eq8), **boxargs)
        box8 = SurroundingRectangle(VGroup(eq3[3][1], eq13, eq8), **boxargs)
        box9 = SurroundingRectangle(VGroup(eq3[3][1], eq14, eq8), **boxargs)
        box10 = SurroundingRectangle(VGroup(eq3[3][1], eq17, eq8), **boxargs)

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq1[0][0]).set_color(col_WVD)
        VGroup(eq1[0][2], eq1[2][0], eq1[2][4], eq5[5][6:9]).set_color(col_x)
        VGroup(eq1[2][-5:-2], eq6[5][:3], eq10[2][8:11], eq14[4][:3]).set_color(col_trig)
        VGroup(eq1[2][-2:]).set_color(col_pi)
        VGroup(eq1[2][2], eq1[2][6], eq5[4][1], eq11[4][1:4:2], eq15[2][1]).set_color(col_op)
        VGroup(eq1[2][3], eq1[2][5], eq1[2][7], eq8[0], eq11[4][2]).set_color(col_zero)
        VGroup(eq2, eq4).set_color(RED)
        VGroup(eq3[3][1], eq10[2][5], eq11[4][4]).set_color(col_special)
        VGroup(eq3[3][3], eq3[3][7:10], eq3[3][11:14]).set_color(col_p)
        VGroup(eq3[3][5], eq5[4][::2], eq6[5][4], eq15[2][:4:2]).set_color(col_num)
        VGroup(eq8[2][2], eq11[4][5]).set_color(col_i)
        VGroup(eq8[2][-1], eq14[2][-3]).set_color(col_p)
        VGroup(eq8[2][0], eq14[2][-1]).set_color(col_num)
        VGroup(eq11[4][-1]).set_color(col_angle)

        mh.copy_colors_eq(eq15[2][:3], eq16[3][2:5])

        eq2 = mh.eq_shadow(eq2, bg_stroke_width=14)
        eq4 = mh.eq_shadow(eq4, bg_stroke_width=14)

        self.add(eq1, box1)
        self.play(FadeIn(eq2, arr1))
        self.wait(0.1)
        self.play(FadeOut(eq2, arr1))
        self.wait(0.1)

        circ = mh.circle_eq(eq1[2][0], scale=0.6)
        txt = Tex(r'\sf prime number theorem', font_size=50, color = RED)
        txt.next_to(circ, UP, buff=0.2)
        txt = mh.eq_shadow(txt, bg_stroke_width=12)
        self.play(Create(circ, rate_func=linear, run_time=0.4),
                  Succession(Wait(0.3), FadeIn(txt)))
        self.wait(0.1)
        self.play(FadeOut(circ, txt))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq1[:2], eq3[:2], eq1[2][:8], eq3[2][:8],
                                eq1[2][8:], eq3[4][:], eq1[2][1:3].copy(), eq3[3][:4:2],
                                eq1[2][4].copy(), eq3[3][6], eq1[2][6].copy(), eq3[3][10]),
                  mh.fade_replace(eq1[2][3].copy(), eq3[3][3], coor_mask=RIGHT),
                  mh.fade_replace(eq1[2][5].copy(), eq3[3][7:10], coor_mask=RIGHT),
                  mh.fade_replace(eq1[2][7].copy(), eq3[3][11:14], coor_mask=RIGHT),
                  FadeIn(eq3[3][4:6], shift=mh.diff(eq1[2][3], eq3[3][3])),
                  FadeIn(eq3[3][1], shift=mh.diff(eq1[2][2], eq3[3][2])),
                  mh.rtransform(box1, box2),
                  run_time=1.5),
                  Succession(Wait(1.2), FadeIn(eq4, arr2))
                  )
        self.wait(0.1)
        eq5_1 = eq3[3][0].copy().move_to(eq5[3][0])
        self.play(mh.rtransform(eq3[:3], eq5[:3], eq3[-1], eq5[-1],
                                eq3[3][1:6], eq5[5][:5], eq3[3][9], eq5[5][10],
                                eq3[3][10], eq5[5][11], eq3[3][13], eq5[5][12],
                                box2, box3),
                  mh.rtransform(eq3[3][6:9], eq5[5][6:9], copy_colors=False),
                  mh.fade_replace(eq3[3][0], eq5[3][0]),
                  mh.fade_replace(eq3[3][11], eq5_1),
                  mh.stretch_replace(eq3[3][12], eq5[4][2], copy_colors=False),
                  eq5[4][:2].set_opacity(-2).animate.set_opacity(1).shift(mh.diff(eq3[3][2], eq5[5][1])),
                  FadeIn(eq5[5][5:13:4], shift=mh.diff(eq3[3][6:9], eq5[5][5:8])*RIGHT),
                  VGroup(arr2, eq4).animate.shift(mh.diff(eq3[2][3], eq5[2][3])),
                  run_time=1.4
                  )
        self.remove(eq5_1)
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq5[:3], eq6[:3], eq5[-1], eq6[-1], eq5[4], eq6[4],
                                eq5[5][5], eq6[5][3], eq5[5][6:10], eq6[5][6:10],
                                               box3, box4),
                  mh.fade_replace(eq5[3], eq6[3]),
                                 VGroup(arr2, eq4).animate.shift(mh.diff(eq5[2][3], eq6[2][3])),
                                 run_time=1.3),
                  FadeOut(eq5[5][:5], eq5[5][10:]),
                  Succession(Wait(0.6), FadeIn(eq6[5][:3], eq6[5][4:6]))
                  )
        self.wait(0.1)
        circ = mh.circle_eq(eq6[4:])
        txt = Tex(r'\sf bounded', color=RED, font_size=60)
        txt.next_to(circ, UP, buff=0.2).shift(RIGHT)
        txt = mh.eq_shadow(txt, bg_stroke_width=14)
        self.play(Create(circ, rate_func=linear),
                  Succession(Wait(0.7), FadeIn(txt)))
        self.wait(0.1)
        self.play(FadeOut(circ, txt, arr2, eq4))
        self.wait(0.1)
        gp2 = VGroup(eq6, box4)
        gp2_ = gp2.copy()
        self.play(FadeIn(box5, rate_func=linear), gp2.animate.to_edge(DOWN, buff=0))
        self.wait(0.1)
        self.play(FadeOut(box5), gp2.animate.move_to(gp2_))
        self.wait(0.1)
        self.play(FadeIn(circ, txt))
        self.wait(0.1)
        self.play(FadeOut(circ, txt, eq6[3:]),
                  Succession(Wait(0.5), AnimationGroup(mh.rtransform(
                      eq6[0][:], eq7[0][:4], eq6[1], eq7[1], eq6[2][0], eq7[0][-1],
                      eq6[2][1:], eq7[2][:], box4, box6
                  ),
                             FadeIn(eq7[0][4], shift=mh.diff(eq6[0][:4], eq7[0][:4])), run_time=1.6)),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(box6, box7),
                  Succession(Wait(0.6), FadeIn(eq8)))
        self.wait(0.1)
        self.play(Succession(Wait(1), AnimationGroup(
            mh.rtransform(eq7[:2], eq9[:2], eq7[2][:4], eq9[2][:4], eq7[2][-2:], eq9[2][-2:],
                                eq7[2][3].copy(), eq9[2][5]),
                  FadeOut(eq7[2][4], shift=mh.diff(eq7[2][4], eq9[2][4:8])*RIGHT),
            run_time=1)),
                  AnimationGroup(
                  mh.rtransform(eq8[2][0].copy(), eq9[2][4]),
                      mh.stretch_replace(eq8[2][-2:].copy(), eq9[2][6:8]),
                      run_time=2
                  ))
        self.wait(0.1)
        self.play(mh.rtransform(eq9[:2], eq10[:2], eq9[2][:5], eq10[2][:5],
                                eq9[2][6:8], eq10[2][6:8], eq9[2][8:], eq10[2][12:]),
                  mh.stretch_replace(eq9[2][5], eq10[2][11]),
                  FadeIn(eq10[2][5], target_position=eq9[2][5]),
                  FadeIn(eq10[2][8:11], shift=mh.diff(eq9[2][7], eq10[2][7])),
                  run_time=1.2)
        self.wait(0.1)
        self.play(mh.rtransform(eq8[:3], eq11[:3]),
                  Succession(Wait(0.6), FadeIn(eq11[3:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq10[:2], eq12[:2], eq10[2][1:13], eq12[2][:12],
                                eq11[4][1:].copy(), eq12[2][12:]),
                  FadeOut(eq10[2][0]),
                  FadeOut(eq10[2][13]))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq12[:2], eq13[:2], eq12[2][:4], eq13[2][:4],
                                eq12[2][-7:-3], eq13[2][4:8], eq12[2][4:6], eq13[2][8:10],
                                eq12[2][6:11], eq13[2][11:16], eq12[2][-1], eq13[2][17],
                                               box7, box8),
                  mh.rtransform(eq12[2][-3:-1], eq13[2][8:10]),
                  FadeIn(eq13[2][16], shift=mh.diff(eq12[2][-1], eq13[2][17])),
                                 run_time=1.3),
                  Succession(Wait(0.7), FadeIn(eq13[2][10], eq13[2][18]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq13[:2], eq14[:2], eq13[2][:2], eq14[2][:2],
                                eq13[2][2:8], eq14[3][1:7],
                                box8, box9),
                  mh.fade_replace(eq13[2][-11:-9], eq14[4][:3], coor_mask=RIGHT),
                  mh.stretch_replace(eq13[2][-9:], eq14[4][-9:]),
                  FadeIn(eq14[3][0].set_color(col_num), shift=mh.diff(eq13[2][2], eq14[3][1])),
                  FadeIn(eq14[2][2:], shift=mh.diff(eq13[2][1], eq14[2][1])))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[:2], eq15[:2], eq11[2][1:], eq15[2][3:], eq11[3:], eq15[3:]),
                  mh.rtransform(eq14[:3], eq16[:3], eq14[4:], eq16[4:],
                                eq14[3][:2], eq16[3][:2], eq14[3][3:], eq16[3][5:]),
                  mh.fade_replace(eq11[2][0], eq15[2][:3], coor_mask=RIGHT),
                  mh.fade_replace(eq14[3][2], eq16[3][2:5], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq16[:2], eq17[:2], eq16[2], eq17[3], eq16[3][0], eq17[4][0],
                                eq16[3][1], eq17[2][-1], eq16[3][5:], eq17[4][1:], eq16[4], eq17[5],
                                box9, box10),
                  FadeOut(eq16[3][2:5], shift=mh.diff(eq16[3][1], eq17[2][-1])),
                  FadeIn(eq17[2][:-1].set_color(col_op), shift=mh.diff(eq16[3][1], eq17[2][-1])),
                  run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq17[:5], eq18[:5], eq17[5][:5], eq18[5][:5], eq17[5][9:], eq18[5][6:]),
                  mh.fade_replace(eq17[5][5:9], eq18[5][5].set_color(col_x), coor_mask=RIGHT))

        self.wait()


class CountingErrorNormal(Chebyshev):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)

        eq1 = MathTex(r'\mathcal E(x)', r'=', r'\left(', r'\pi(x) - {\rm Li}(x) +', r'\frac12', r'{\rm Li}(\sqrt x)',
                      r'\right)', r'\frac{\log x}{\sqrt x}')
        mh.font_size_sub(eq1, 4, 50)
        eq2 = MathTex(r'\mathcal E(x)', r'=', r'\pi(x) - {\rm Li}(x) +', r'\frac12', r'{\rm Li}(\sqrt x)')
        mh.font_size_sub(eq2, 3, 50)

        mh.align_sub(eq2, eq2[1], eq1[1])
        mh.rtransform.copy_colors = True
        VGroup(eq2[0][0], eq2[2][0], eq2[2][5:7], eq2[4][:2]).set_color(col_WVD)
        VGroup(eq2[0][2], eq2[2][2], eq2[2][8], eq2[4][-2], eq1[7][3], eq1[7][-1]).set_color(col_x)
        VGroup(eq2[3][::2]).set_color(col_num)
        VGroup(eq2[3][1], eq2[4][3:-2], eq1[7][4:-1]).set_color(col_op)
        VGroup(eq1[7][:3]).set_color(col_trig)

        self.add(eq2)
        self.play(mh.rtransform(eq2[:2], eq1[:2], eq2[2:5], eq1[3:6]),
                  Succession(Wait(0.5), FadeIn(eq1[2], eq1[6:])))
        # self.add(eq1[:2], eq1[3:6])
        self.wait()


class MeanSquareTheory(Chebyshev):
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        eq1 = MathTex(r'\mathbb E[\mathcal E^2]', r'=', r'\frac1{\log(N/N_0)}',
                      r'\int_{N_0}^N', r'\mathcal E(x)^2', r'\,\frac{dx}{x}')
        eq2 = MathTex(r'\mathbb E[\mathcal E^2]', r'\sim', r'\sum_\rho\frac{2}{\lvert\rho\rvert^2}',
                      r'=', r'0.046\cdots')

        mh.align_sub(eq2, eq2[0], eq1[0])

        mh.rtransform.copy_colors = True
        VGroup(eq1[4][0], eq1[0][2]).set_color(col_WVD)
        VGroup(eq1[2][0], eq1[4][-1], eq1[0][-2], eq2[2][-1], eq2[2][2], eq2[-1]).set_color(col_num)
        VGroup(eq1[2][1], eq1[2][-4], eq1[3][0], eq1[5][2], eq1[5][0], eq2[2][3:5], eq2[2][6],
               eq2[2][0]).set_color(col_op)
        VGroup(eq1[2][2:5]).set_color(col_trig)
        VGroup(eq1[2][6], eq1[2][8:10], eq1[3][1:], eq1[4][2], eq1[5][1], eq1[5][-1]).set_color(col_x)
        VGroup(eq1[0][0]).set_color(col_txt2)
        VGroup(eq2[2][1], eq2[2][-3]).set_color(col_zero)

        self.add(eq1)
        eq2_1 = eq2[2].copy().shift(RIGHT*1.6)
        self.play(mh.rtransform(eq1[0], eq2[0]),
                  mh.stretch_replace(eq1[1], eq2[1]),
                  FadeOut(eq1[2:]),
                  FadeIn(eq2_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1, eq2[2]),
                  Succession(Wait(0.5), FadeIn(eq2[3:])))
        self.wait()

class EmpiricalErrNew(Chebyshev):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)

        eq1 = MathTex(r'\mathcal E(x)', r'=', r'\pi(x)-{\rm Li}(x)')
        eq2 = MathTex(r'\mathcal E(x)', r'=', r'\left(', r'\pi(x)-{\rm Li}(x)', r'\right)',
                      r'\frac{\log x}{\sqrt x}')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][0], eq1[2][0], eq1[2][5:7], eq2[-1][3], eq2[-1][-1]).set_color(col_WVD)
        VGroup(eq1[0][-2], eq1[2][2], eq1[2][-2]).set_color(col_x)
        VGroup(eq2[-1][:3]).set_color(col_trig)
        VGroup(eq2[-1][4:-1]).set_color(col_op)

        mh.align_sub(eq1, eq1[1], eq2[1])

        eq1 = mh.eq_shadow(eq1, bg_stroke_width=15)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=15)

        self.add(eq1)
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq1[2], eq2[3]),
                  Succession(Wait(0.4), FadeIn(eq2[2], eq2[4:])),
                  )
        self.wait()

class EmpiricalArcSinSum(Chebyshev):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)

        eq1 = MathTex(r'\sum_{\Re[\rho] > 0}', r'\frac{2}{\lvert\rho\rvert}', r'\cos U_\rho')
        VGroup(eq1[0][:2], eq1[1][1:3], eq1[1][-1]).set_color(col_op)
        VGroup(eq1[0][3], eq1[1][-2], eq1[2][-1]).set_color(col_zero)
        VGroup(eq1[0][-1], eq1[1][0]).set_color(col_num)
        VGroup(eq1[2][3]).set_color(col_var)
        VGroup(eq1[2][:3]).set_color(col_trig)

        eq1 = mh.eq_shadow(eq1, bg_stroke_width=14)

        self.add(eq1)
        circ = mh.circle_eq(eq1[-1], scale=0.7)
        txt = Tex(r'\sf arcsine', color=RED, font_size=50)
        txt.next_to(circ, DR, buff=-0.1)
        self.play(Create(circ, rate_func=linear, run_time=0.6), FadeIn(txt))
        self.wait()

class TheoreticalBias(Chebyshev):
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        eq1 = MathTex(r'\pi(x)', r'=', r'{\rm Li}(x)', r'-', r'\frac12', r'{\rm Li(\sqrt x)}', r'-', r'\frac13',
                      r'{\rm Li}(\sqrt[3] x)', r'+\cdots')
        mh.font_size_sub(eq1, 4, 50)
        mh.font_size_sub(eq1, 7, 50)
        eq2 = MathTex(r'\mathcal E(x)', r'=', r'\left(', r'-', r'\frac12', r'{\rm Li}(\sqrt x)', r'-', r'\frac13',
                      r'{\rm Li}(\sqrt[3] x)', r'+\cdots', r'\right)', r'\frac{\log x}{\sqrt x}', r'+', r'{\sf noise}')
        mh.font_size_sub(eq2, 4, 50)
        mh.font_size_sub(eq2, 7, 50)
        eq3 = MathTex(r'\mathcal E(x)', r'\approx', r'\left(', r'-', r'\frac{\sqrt x}{2\log\sqrt x}', r'-',
                      r'\frac{\sqrt[3] x}{3\log\sqrt[3] x}', r'+\cdots', r'\right)', r'\frac{\log x}{\sqrt x}')
        eq4 = MathTex(r'\mathcal E(x)', r'\approx', r'-', r'1', r'-', r'\frac{\sqrt[3] x}{\sqrt x}')
        eq5 = MathTex(r'\mathcal E(x)', r'\approx', r'-', r'1', r'+', r'{\sf noise}')

        mh.rtransform.copy_colors=True
        VGroup(eq1[0][0], eq1[2][:2], eq1[5][:2], eq1[8][:2], eq2[0][0]).set_color(col_WVD)
        VGroup(eq1[0][-2], eq1[2][-2], eq1[5][-2], eq1[8][-2], eq2[0][2], eq2[-3][3], eq2[-3][-1]).set_color(col_x)
        VGroup(eq1[4][::2], eq1[7][::2], eq1[8][3], eq4[3]).set_color(col_num)
        VGroup(eq1[4][1], eq1[5][3:-2], eq1[7][1], eq1[8][4:-2], eq2[-3][4:-1]).set_color(col_op)
        VGroup(eq2[-1]).set_color(col_txt)
        VGroup(eq2[-3][:3], eq3[4][5:8], eq3[6][6:9]).set_color(col_trig)

        eq2[-2:].next_to(eq2[-3], DOWN, buff=0.3).align_to(eq2[-3], RIGHT).shift(RIGHT*0.5)
        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)
        eq2.move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq3[1])
        for i in [2, 3, 4, 5]:
            eq4[i].move_to(eq3[i+1], coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[1], eq4[1], coor_mask=UP)
        eq5.align_to(eq4[3], RIGHT).shift(RIGHT*0.3)

        eq3_1 = MathTex(r'\frac1{\log x}')
        eq1 = mh.eq_shadow(eq1, bg_stroke_width=15)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=15)
        eq3 = mh.eq_shadow(eq3, bg_stroke_width=15)
        eq4 = mh.eq_shadow(eq4, bg_stroke_width=15)
        eq5 = mh.eq_shadow(eq5, bg_stroke_width=15)
        eq3_1 = mh.eq_shadow(eq3_1, bg_stroke_width=15)[0]

        self.add(eq1)
        self.play(AnimationGroup(mh.rtransform(eq1[1], eq2[1], eq1[3:10], eq2[3:10]),
                  mh.fade_replace(eq1[0], eq2[0], coor_mask=RIGHT),
                  FadeOut(eq1[2], shift=mh.diff(eq1[2], eq2[0])*RIGHT),
                                 run_time=1.5),
                  Succession(Wait(1), FadeIn(eq2[-3:], eq2[2], eq2[10])),
                  )
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq2[:4], eq3[:4], eq2[-4:-2], eq3[-2:],
                                eq2[4][1], eq3[4][3], eq2[5][3:6], eq3[4][:3], eq2[5][3:6].copy(),
                                               eq3[4][-3:], eq2[4][2], eq3[4][4],
                                               eq2[6], eq3[5], eq2[7][1], eq3[6][4],
                                               eq2[8][3:-1], eq3[6][:4], eq2[7][2], eq3[6][5],
                                               eq2[8][3:-1].copy(), eq3[6][9:],
                                               eq2[9], eq3[7]),
                  FadeOut(eq2[4][0], shift=mh.diff(eq2[4][0], eq3[4][:3])*RIGHT),
                  FadeOut(eq2[7][0], shift=mh.diff(eq2[7][0], eq3[6][:4]) * RIGHT),
                                 run_time=1.6),
                  FadeOut(eq2[5][:3], eq2[5][-1], eq2[8][:3], eq2[8][-1]),
                  Succession(Wait(0.8), FadeIn(eq3[4][5:8], eq3[6][6:9]))
                  )
        self.wait(0.1)
        eq3_2 = eq3_1.copy()
        mh.align_sub(eq3_1, eq3_1[1], eq3[4][3])
        mh.align_sub(eq3_2, eq3_2[1], eq3[6][4])
        self.play(mh.rtransform(eq3[4][5:8], eq3_1[2:5], eq3[4][-1], eq3_1[-1],
                                eq3[6][6:9], eq3_2[2:5], eq3[6][-1], eq3_2[-1]),
                  FadeOut(eq3[4][4], eq3[4][8:-1], eq3[6][5], eq3[6][9:-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[3], eq4[2], eq3[5], eq4[4], eq3[6][:5], eq4[5][:5],
                                eq3[-1][-3:], eq4[5][-3:]),
                  FadeOut(eq3[4][:4], eq3_1[2:], eq3_2[2:]),
                  FadeOut(eq3[-1].copy(), shift=mh.diff(eq3[-1], eq3[4])*RIGHT),
                  FadeOut(eq3[-1][:5], shift=mh.diff(eq3[-1][:5], eq3[6][:5])*RIGHT),
                  FadeIn(eq4[3]),
                  FadeOut(eq3[2], eq3[8]),
                  run_time=2)
        self.wait(0.1)
        self.play(mh.rtransform(eq4[:4], eq5[:4], eq2[-2:], eq5[-2:], run_time=2),
                  FadeOut(eq4[4:], eq3[7]))

        self.wait()

class TheoreticalBias2(Chebyshev):
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        eq1 = MathTex(r'\mathcal E(x)', r'=', r'\left(', r'-', r'\frac12', r'{\rm Li}(\sqrt x)', r'-', r'\frac13',
                      r'{\rm Li}(\sqrt[3] x)', r'\right)', r'\frac{\log x}{\sqrt x}', r'+', r'{\sf noise}')
        mh.font_size_sub(eq1, 4, 50)
        mh.font_size_sub(eq1, 7, 50)
        eq2 = MathTex(r'\mathcal E(x)', r'=', r'-1', r'+1+', r'\left(', r'-', r'\frac12', r'{\rm Li}(\sqrt x)', r'-', r'\frac13',
                      r'{\rm Li}(\sqrt[3] x)', r'\right)', r'\frac{\log x}{\sqrt x}', r'+', r'{\sf noise}')
        mh.font_size_sub(eq2, 6, 50)
        mh.font_size_sub(eq2, 9, 50)
        eq3 = MathTex(r'\to0', r'{\sf\ slowly...}', font_size=60, color=RED)
        eq4 = MathTex(r'\approx', r'-0.13', r'{\sf\ at\ }', r'x=10\,000\,000\,000')
        eq5 = MathTex(r'\mathcal E(x)', r'-1+', r'\left(', r'\frac12', r'{\rm Li}(\sqrt x)', r'+', r'\frac13',
                      r'{\rm Li}(\sqrt[3] x)', r'\right)', r'\frac{\log x}{\sqrt x}')
        mh.font_size_sub(eq5, 3, 50)
        mh.font_size_sub(eq5, 6, 50)

        eq2[-2:].next_to(eq2[-3], DOWN, buff=0.3).align_to(eq2[-3], RIGHT).shift(RIGHT*0.5)
        eq2.move_to(ORIGIN)
        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)
        mh.align_sub(eq5, eq5[0], eq2[0]).shift(RIGHT)

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][0], eq1[5][:2], eq1[8][:2]).set_color(col_WVD)
        VGroup(eq1[0][2], eq1[-3][3], eq1[-3][-1], eq1[5][-2], eq1[8][-2], eq4[3][0]).set_color(col_x)
        VGroup(eq1[4][::2], eq1[7][::2], eq1[8][3], eq2[2][-1], eq2[3][1], eq4[1], eq4[3][2:]).set_color(col_num)
        VGroup(eq1[-3][4:-1], eq1[4][1], eq1[7][1], eq1[5][3:-2], eq1[8][4:-2]).set_color(col_op)
        VGroup(eq1[-1], eq4[2]).set_color(col_txt)
        VGroup(eq1[-3][:3]).set_color(col_trig)

        eq1 = mh.eq_shadow(eq1, bg_stroke_width=14)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=14)
        eq3 = mh.eq_shadow(eq3, bg_stroke_width=14)
        eq4 = mh.eq_shadow(eq4, bg_stroke_width=14)
        eq5 = mh.eq_shadow(eq5, bg_stroke_width=14)

        self.add(eq1)
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq1[2:], eq2[4:], run_time=1.5),
                  Succession(Wait(0.6), FadeIn(eq2[2:4])))
        self.wait(0.1)
        circ = mh.circle_eq(VGroup(eq2[4:-2], eq2[3][1:]))
        eq3.next_to(circ, UP, buff=0.2).shift(RIGHT)
        self.play(Create(circ, rate_func=linear),
                  Succession(Wait(0.6), FadeIn(eq3[0])))
        self.play(FadeIn(eq3[1], rate_func=linear))
        self.wait(0.1)
        eq4.move_to(eq3).align_to(eq3, LEFT).to_edge(RIGHT, buff=0.2)
        self.play(FadeOut(eq3), FadeIn(eq4))
        self.wait(0.1)
        self.play(FadeOut(circ, eq4),
                  mh.rtransform(eq2[0], eq5[0], eq2[3][1:], eq5[1][1:],
                                eq2[4], eq5[2], eq2[6:8], eq5[3:5],
                                eq2[9:13], eq5[6:]),
                  mh.fade_replace(eq2[3][0], eq5[1][0]),
                  mh.fade_replace(eq2[8], eq5[5]),
                  FadeOut(eq2[1:3], eq2[-2:]),
                  FadeOut(eq2[5], shift=mh.diff(eq2[4], eq5[2])))

        self.wait()

class ReciprocalsSumToOne(Chebyshev):
    bgcolor = BLACK
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)
        eq_zeros = []
        eq_sum = []
        eq_num = []
        zero_sum = 0.
        height = config.frame_height * 0.84 / 11 * UP
        width0 = 0.6 * RIGHT
        width1 = 3.9 * RIGHT
        width2 = 2.2 * RIGHT
        topleft1 = mh.pos(UP*0.75+LEFT*0.96)
        topleft2 = topleft1 * UL - (width0 + width1 + width2)
        nrows = 11
        fs = 44
        lines = [[], []]
        lineargs = get_kwargs(stroke_width=4, stroke_color=WHITE)
        eq_rho = []
        eq_cumsum = []
        for j, pos in [(0, topleft1), (1, topleft2)]:
            eq1 = MathTex(r'\rho', font_size=50, color=col_zero, stroke_width=1.5)
            eq1.next_to(pos+width0+width1/2, UP, buff=0.1)
            eq_rho.append(eq1)
            eq2 = Tex(r'$\sum 2/\lvert\rho\rvert$', font_size=50, stroke_width=1.5)
            VGroup(eq2[0][0], eq2[0][2:4], eq2[0][-1]).set_color(col_op)
            eq2[0][1].set_color(col_num)
            eq2[0][-2].set_color(col_zero)
            eq2.next_to(pos+width0+width1+width2/2, UP, buff=0.1)
            eq1.move_to(eq2[0][-2], coor_mask=UP)
            eq_cumsum.append(eq2)

            for i in range(nrows+1):
                lines[j].append(Line(pos - i * height, pos - i * height + width0 + width1 + width2, **lineargs))
            for pos1 in [pos, pos + width0, pos+width0+width1, pos+width0+width1+width2]:
                lines[j].append(Line(pos1, pos1 - nrows*height, **lineargs))

        # eq_rho.next_to(top)

        for i in range(1, 22):
            pos0 = topleft1 if i <= nrows else topleft2
            j = (i-1) % nrows
            eq1 = MathTex(r'{}'.format(i), font_size=fs, color=RED)
            eq_num.append(eq1)
            pos1 = pos0 + width0/2 - height * (j+1/2)
            eq1.move_to(pos1)

            gamma = float(mp.im(mp.zetazero(i)))
            eq2 = MathTex(r'1/2', r'+', r'i\,', '{:.4f}'.format(gamma), r'\cdots', font_size=fs)
            VGroup(eq2[0][::2], eq2[3:]).set_color(col_num)
            eq2[0][1].set_color(col_op)
            eq2[2].set_color(col_i)
            pos1 += (width0+width1)/2
            eq2.move_to(pos1)
            mh.align_sub(eq2, eq2[0][0], eq1[0][0].get_bottom(), UP, buff=0, coor_mask=UP)
            eq_zeros.append(eq2)

            zero_sum += 2./math.sqrt(gamma**2+0.25)
            eq3 = MathTex(r'{:.4f}'.format(zero_sum), r'\cdots', color=col_p, font_size=fs)
            eq_sum.append(eq3)
            eq3.move_to(pos1 + (width1 + width2)/2)
            mh.align_sub(eq3, eq3[-1][-1], eq2[-1][-1], coor_mask=UP)

        # self.play(FadeIn(*eq_zeros[:nrows], *eq_sum[:nrows], *eq_num[:nrows]))

        gp = VGroup(VGroup(eq_rho[0], eq_cumsum[0], *lines[0]), eq_zeros[:nrows], eq_sum[:nrows], eq_num[:nrows])
        gp1 = VGroup(eq_rho[1], eq_cumsum[1], *lines[1])
        pos = gp.get_center()
        gp.move_to(ORIGIN, coor_mask=RIGHT)
        self.add(gp[0])
        anims = []
        for i in range(nrows):
            anims.append(Succession(Wait(i*0.2), FadeIn(eq_zeros[i], eq_sum[i], eq_num[i])))
        self.play(*anims)
        self.play(gp.animate.move_to(pos),
                  mh.rtransform(gp[0].copy(), gp1))
        anims = []
        for i in range(nrows, 2*nrows-1):
            anims.append(Succession(Wait((i-nrows)*0.2), FadeIn(eq_zeros[i], eq_sum[i], eq_num[i])))
        self.play(*anims)
        self.wait(0.1)

        circ = mh.circle_eq(eq_sum[-1])
        self.play(Create(circ, rate_func=linear, run_time=0.7))

        self.wait()

class ExtremeArcSinSum(Chebyshev):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)

        eq1 = MathTex(r'\sum', r'\frac{2}{\lvert\rho\rvert}', r'\cos U_\rho').set_z_index(4)
        VGroup(eq1[0], eq1[1][2:4], eq1[1][-1]).set_color(col_op)
        VGroup(eq1[1][0]).set_color(col_num)
        VGroup(eq1[1][-2], eq1[2][-1]).set_color(col_zero)
        VGroup(eq1[2][:3]).set_color(col_trig)
        VGroup(eq1[2][3]).set_color(col_var)

        eq1.to_edge(DOWN, buff=0.4)
        circ = mh.circle_eq(eq1[-1], scale=0.8).set_z_index(6)
        eq2 = MathTex(r'\approx1', color=RED, stroke_width=1.5, font_size=60)

        box1 = SurroundingRectangle(VGroup(eq1, circ, circ.copy().scale(-1, about_point=circ.get_center()*UP)),
                                    stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)
        eq2.next_to(circ, UR, buff=0.1).set_z_index(6).align_to(box1, RIGHT).shift(LEFT*0.1)

        self.add(box1, eq1)
        self.play(Create(circ, rate_func=linear, run_time=0.6),
                  Succession(Wait(0.5), FadeIn(eq2)))
        self.wait(0.1)
        self.play(FadeOut(circ, eq2))

        self.wait()

class ProbFlip(Chebyshev):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=80)

        eq1 = MathTex(r'\mathbb P\left(\pi(x) > {\rm Li}(x)\right)', r'>', r'0')
        eq2 = MathTex(r'\mathbb P\left(\pi(x) > {\rm Li}(x)\right)', r'\approx', r'0.00000026')
        eq3 = MathTex(r'\mathbb P\left(\pi(x) < {\rm Li}(x)\right)', r'\approx', r'0.99999974')

        mh.rtransform.copy_colors=True
        VGroup(eq1[0][0]).set_color(col_txt2)
        VGroup(eq1[0][2], eq1[0][7:9]).set_color(col_WVD)
        VGroup(eq1[0][4], eq1[0][10]).set_color(col_x)
        VGroup(eq1[-1], eq2[-1], eq3[-1]).set_color(col_num)

        eq1 = mh.eq_shadow(eq1, bg_stroke_width=14)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=14)
        eq3 = mh.eq_shadow(eq3, bg_stroke_width=14)

        eq1_1 = eq1[0].copy().move_to(eq1, coor_mask=RIGHT)
        eq2.shift(LEFT*0.2)
        self.add(eq1_1)
        self.play(mh.rtransform(eq1_1, eq1[0]),
                  Succession(Wait(0.4), FadeIn(eq1[1:])))
        self.wait(0.1)
        anims = []
        for i in range(8):
            anims.append(Succession(Wait(0.3*i+0.7), FadeIn(eq2[-1][2+i])))
        self.play(mh.rtransform(eq1[0], eq2[0], eq1[-1][0], eq2[-1][0]),
                  mh.fade_replace(eq1[1], eq2[1]),
                  FadeIn(eq2[-1][1], shift=mh.diff(eq1[-1][0], eq2[-1][0])),
                  *anims
                  )

        self.play(eq2.animate.move_to(ORIGIN).to_edge(DOWN, buff=0.4))
        eq3.next_to(eq2, UP, buff=0.4).align_to(eq2, LEFT)
        self.wait(0.1)
        self.play(mh.rtransform(eq2[0][:6].copy(), eq3[0][:6], eq2[0][7:].copy(), eq3[0][7:]),
                  eq2[0][6].copy().animate.stretch(-1, dim=0).move_to(eq3[0][6]))
        anims = []
        for i in range(6):
            anims.append(Succession(Wait(0.2*i+0.6), FadeIn(eq3[-1][4+i])))
        self.play(FadeIn(eq3[-2], eq3[-1][:4]), *anims)
        self.wait()


class PathCalc(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)
        eq1 = MathTex(r'\pi(x)', r'\approx', r'{\rm Li}(x)', r'-', r'\frac12', r'{\rm Li}(\sqrt x)',
                      r'-', r'\frac13', r'{\rm Li}(\sqrt[3] x)', r'+', r'\cdots',
                      r'-', r'\frac{1}{\log x}', r'\sum_{\rho}', r'\frac{x^\rho}{\rho}')
        mh.font_size_sub(eq1, 4, 50)
        mh.font_size_sub(eq1, 7, 50)
        eq2 = MathTex(r'\hat\pi(x)', r'=', r'{\rm Li}(x)', r'-', r'\frac12', r'{\rm Li}(\sqrt x)',
                      r'-', r'\frac13', r'{\rm Li}(\sqrt[3] x)', r'+', r'\cdots')
        eq3 = MathTex(r'\pi(x)-\hat\pi(x)', r'\approx', r'-', r'\frac{1}{\log x}', r'\sum_{\rho}', r'\frac{x^\rho}{\rho}')
        eq4 = MathTex(r'\rho', r'=', r'\frac12', r'+i\gamma', r'=', r'-\lvert\rho\rvert e^{i\theta}')
        mh.font_size_sub(eq4, 2, 50)
        eq5 = MathTex(r'\pi(x)-\hat\pi(x)', r'\approx', r'\frac{\sqrt x}{\log x}', r'\sum_{\rho\colon\gamma > 0}',
                      r'\frac{2}{\lvert\rho\rvert}', r'\cos(\gamma\log x-\theta)')
        eq6 = MathTex(r'\pi(e^tx)-\hat\pi(e^tx)', r'\approx', r'\frac{\sqrt {e^tx}}{\log e^tx}', r'\sum_{\rho\colon\gamma > 0}',
                      r'\frac{2}{\lvert\rho\rvert}', r'\cos(\gamma\log e^tx-\theta)')
        eq7 = MathTex(r'\frac{e^{\frac t2}\sqrt {x}}{t+\log x}', r'\sum_{\rho\colon\gamma > 0}',
                      r'\frac{2}{\lvert\rho\rvert}', r'\cos(\gamma\log e^tx-\theta)')
        eq8 = MathTex(r'\frac{e^{\frac t2} }t', r'\sqrt {x}', r'\sum_{\rho\colon\gamma > 0}',
                      r'\frac{2}{\lvert\rho\rvert}', r'\cos(\gamma\log e^tx-\theta)')
        eq9 = MathTex(r'\frac t{e^{\frac t2} }', r'\left(', r'\pi(e^tx)-\hat\pi(e^tx)', r'\right)', r'\approx')
        eq10 = MathTex(r'\sqrt {x}', r'\sum_{\rho\colon\gamma > 0}',
                      r'\frac{2}{\lvert\rho\rvert}', r'\cos(\gamma\log x + \gamma t -\theta)')
        eq11 = MathTex(r'\sqrt {x}', r'\sum_{\rho\colon\gamma > 0}',
                      r'\frac{2}{\lvert\rho\rvert}', r'\cos(\gamma\log x + U_\rho)')
        eq12 = Tex(r'$U_\rho$', r' independent, uniform on ', r'$[0,2\pi]$')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][0], eq1[2][:2], eq1[5][:2], eq1[8][:2], eq2[0][:2]).set_color(col_WVD)
        VGroup(eq1[0][-2], eq1[2][-2], eq1[5][-2], eq1[8][-2], eq1[12][-1], eq1[14][0],
               eq2[0][-2]).set_color(col_x)
        VGroup(eq1[4][::2], eq1[7][::2], eq1[8][3], eq4[2][::2], eq1[12][0], eq5[4][0],
               eq5[3][-1], eq7[0][3], eq12[2][1]).set_color(col_num)
        VGroup(eq1[5][3:-2], eq1[8][4:-2], eq1[13][0], eq1[14][2],
               eq4[2][1], eq4[5][1], eq4[5][3], eq5[2][:2], eq7[0][2], eq1[-3][1]).set_color(col_op)
        VGroup(eq1[12][-4:-1], eq5[5][:3], eq5[5][5:8]).set_color(col_trig)
        VGroup(eq1[13][1], eq1[14][1], eq1[14][-1], eq4[0], eq4[-1][2], eq11[-1][-2], eq12[0][1]).set_color(col_zero)
        VGroup(eq4[3][1], eq4[-1][-2]).set_color(col_i)
        VGroup(eq4[3][-1], eq5[3][3], eq5[5][4]).set_color(col_p)
        VGroup(eq4[-1][-3], eq6[0][2], eq6[0][10], eq6[2][-10], eq6[2][-3], eq6[5][8]).set_color(col_special)
        VGroup(eq4[-1][-1], eq5[5][-2]).set_color(col_angle)
        VGroup(eq6[0][3], eq6[0][11], eq6[2][-9], eq6[2][-2], eq6[5][9]).set_color(col_txt2)
        VGroup(eq11[-1][-3], eq12[0][0]).set_color(col_var)
        VGroup(eq12[1]).set_color(col_txt)
        VGroup(eq12[2][-3:-1]).set_color(col_pi)
        mh.copy_colors_eq(eq1[2:11], eq2[2:])

        eq1[-4:].next_to(eq1[:-4], DOWN, buff=0.2).align_to(eq1[-5], RIGHT).shift(RIGHT)
        eq1.move_to(ORIGIN)
        mh.align_sub(eq2, eq2[1], eq1[1]).next_to(eq1, DOWN, buff=0.2, coor_mask=UP)
        gp1 = VGroup(eq1.copy(), eq2).move_to(ORIGIN, coor_mask=UP)
        eq2_2 = eq2.copy().to_edge(DOWN, buff=0.4).set_opacity(0.5)
        eq3.next_to(eq2, UP, buff=0.5, coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq5, eq5[1], eq3[1], coor_mask=UP)
        eq4.next_to(eq2_2, UP, buff=0.3, coor_mask=UP)
        eq6[:2].to_edge(LEFT, buff=0.4)
        eq6[2:].to_edge(RIGHT, buff=0.8).shift(DOWN*1.5)
        # eq6.next_to(eq4, UP, buff=0.4, coor_mask=UP)
        mh.align_sub(eq6, eq6[-1][-1], eq5[-1][-1], coor_mask=UP)
        mh.align_sub(eq7, eq7[-1], eq6[-1])
        mh.align_sub(eq8, eq8[-1], eq7[-1])
        mh.align_sub(eq9, eq9[-1], eq6[1]).align_to(eq6, LEFT)
        mh.align_sub(eq10, eq10[-1][-1], eq8[-1][-1])
        mh.align_sub(eq11, eq11[-1][-1], eq10[-1][-1])

        self.add(eq1)
        self.play(FadeIn(eq2),
                  mh.rtransform(eq1, gp1[0]))
        self.wait(0.1)
        eq2_1 = eq2[0].copy()
        self.play(FadeOut(gp1[0][2:11]),
                  eq2_1.animate(run_time=1.6).move_to(eq1[2:6]).align_to(eq1[0], DOWN))
        self.wait(0.1)
        self.play(mh.rtransform(gp1[0][0][:], eq3[0][:4], eq2_1[:], eq3[0][-5:],
                                gp1[0][1], eq3[1], gp1[0][-4:], eq3[-4:], run_time=1.6),
                  Succession(Wait(0.8), FadeIn(eq3[0][4]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq2, eq2_2),
                  FadeIn(eq4))
        self.wait(0.1)
        self.play(AnimationGroup(FadeOut(eq3[2], shift=mh.diff(eq3[1], eq5[1])),
                  FadeOut(eq3[3][0], shift=mh.diff(eq3[3][0], eq5[2][2])*RIGHT),
                  FadeIn(eq5[2][:2], shift=mh.diff(eq3[3][0], eq5[2][2])),
                  FadeIn(eq5[4][0], eq5[4][2], eq5[4][4], shift=mh.diff(eq3[5][3], eq5[4][3])),
                  mh.fade_replace(eq3[5][1], eq5[5][4]),
                  mh.rtransform(eq3[:2], eq5[:2], eq3[3][-5:], eq5[2][-5:],
                                eq3[5][0].copy(), eq5[2][2], eq3[4][:], eq5[3][:2],
                                eq3[5][2], eq5[4][1], eq3[5][3], eq5[4][3],
                                eq3[5][0], eq5[5][8]),
                                 run_time=1.7),
                  Succession(Wait(1), FadeIn(eq5[3][2:], eq5[5][:4], eq5[5][5:8], eq5[5][9:]))
                  )
        self.play(AnimationGroup(mh.rtransform(eq5[0][:2], eq6[0][:2], eq5[0][2:8], eq6[0][4:10],
                                               eq5[0][8:], eq6[0][12:], eq5[1], eq6[1]),
                  eq5[2:].animate.shift(mh.diff(eq5[3][0], eq6[3][0])*UP)),
                  eq4.animate.set_opacity(0.5),
                  Succession(Wait(0.7), FadeIn(eq6[0][2:4], eq6[0][10:12])))
        self.play(mh.rtransform(eq5[2][:-6], eq6[2][:-10], eq5[2][-6:-1], eq6[2][-8:-3],
                                eq5[2][-1], eq6[2][-1], eq5[3:5], eq6[3:5],
                                eq5[5][:8], eq6[5][:8], eq5[5][8:], eq6[5][10:]),
                  Succession(Wait(0.7), FadeIn(eq6[2][-10:-8], eq6[2][-3:-1], eq6[5][8:10])),
                  )
        self.play(mh.rtransform(eq6[3:], eq7[1:], eq6[2][:-10], eq7[0][4:-8], eq6[2][-10:-8], eq7[0][:2],
                                eq6[2][-8:-6], eq7[0][-8:-6], eq6[2][-2], eq7[0][-6],
                                eq6[2][-6:-3], eq7[0][-4:-1], eq6[2][-1], eq7[0][-1]),
                  FadeIn(eq7[0][2:4], shift=mh.diff(eq6[2][-9], eq7[0][1])),
                  mh.rtransform(eq6[2][-3], eq7[0][-5])
                  )
        self.wait(0.1)
        self.play(FadeOut(eq7[0][-5:]))
        self.play(mh.rtransform(eq7[0][:4], eq8[0][:4], eq7[0][-7:-5], eq8[0][-2:], eq7[0][4:-7], eq8[1][:],
                                eq7[1:], eq8[2:]))
        circ = mh.circle_eq(eq8[0], scale=0.6).shift(DOWN*0.05)
        self.play(Create(circ, rate_func=linear, run_time=0.7))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq6[0], eq9[2], eq6[1], eq9[4], eq8[0][:4], eq9[0][-4:], eq8[0][-1], eq9[0][0],
                          eq8[0][-2], eq9[0][1]),
            FadeOut(circ, shift=mh.diff(eq8[0], eq9[0])), run_time=1.6),
            Succession(Wait(1), FadeIn(eq9[1], eq9[-2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq8[1:4], eq10[:3], eq8[4][:8], eq10[3][:8], eq8[4][10], eq10[3][8],
                                eq8[4][4].copy(), eq10[3][10], eq8[4][9], eq10[3][11],
                                eq8[4][11:], eq10[3][12:]),
                  mh.fade_replace(eq8[4][8], eq10[3][9]))
        self.wait(0.1)
        circ = mh.circle_eq(eq10[3][-5:-1], scale=0.6)
        self.play(Create(circ, run_time=0.6, rate_func=linear))
        self.wait(0.1)
        eq11_1 = eq11[-1][-3:-1].copy().move_to(eq10[3][-5:-1], coor_mask=RIGHT)
        self.play(eq10[3][-5:-1].animate.set_opacity(0), FadeIn(eq11_1))

        # eq12.move_to((eq11.get_bottom()+eq4.get_top())/2, coor_mask=UP)
        # eq12.next_to(eq11, DOWN, buff=0.4, coor_mask=UP)
        eq12.move_to(eq4, coor_mask=UP),
        self.wait(0.1)
        self.play(Succession(Wait(0.5), FadeIn(eq12)),
                  # VGroup(eq9, eq10[:-1], eq10, eq11_1, circ).animate.shift(UP*0.7),
                  eq4.animate.move_to(eq2_2, coor_mask=UP),
                  FadeOut(eq2_2))
        mh.align_sub(eq11, eq11[0], eq10[0])
        self.play(mh.rtransform(eq10[:-1], eq11[:-1], eq10[-1][:-5], eq11[-1][:-3],
                                eq11_1, eq11[-1][-3:-1], eq10[-1][-1], eq11[-1][-1],
                                ),
                  FadeOut(circ))
        self.wait()
        circ = mh.circle_eq(eq11[-1], scale=0.7)
        txt = Tex(r'\sf arcsine', color=RED, font_size=50)
        txt.next_to(circ, UR, buff=-0.1)
        self.play(Create(circ, rate_func=linear, run_time=0.6), FadeIn(txt))
        self.wait(0.1)
        self.play(FadeOut(circ, txt))


        self.wait()

class UniformInterval(Chebyshev):
    def get_final(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)
        eq1 = MathTex(r'\psi(x)-x', r'\approx', r'\sqrt x', r'\sum_{\rho\colon\gamma > 0}', r'\frac2{\lvert\rho\rvert}',
                      r'\cos(\gamma y-\theta)')
        eq1[0][0].set_color(col_WVD)
        VGroup(eq1[0][2], eq1[0][-1], eq1[5][5], eq1[2][-1]).set_color(col_x)
        VGroup(eq1[3][0], eq1[4][1:3], eq1[4][-1], eq1[2][:-1]).set_color(col_op)
        VGroup(eq1[3][1], eq1[4][3]).set_color(col_zero)
        eq1[5][:3].set_color(col_trig)
        VGroup(eq1[3][-1], eq1[4][0]).set_color(col_num)
        VGroup(eq1[3][3], eq1[5][4]).set_color(col_p)
        eq1[5][-2].set_color(col_angle)
        eq1.to_edge(DOWN, buff=0.5)
        eq1.set_z_index(10)
        box = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                   buff=0.2, corner_radius=0.15)

        return eq1, box


    def construct(self):
        NumberLine()
        minx = 0.
        maxx = 10.
        ax = Axes(x_range=[0, 10.5], x_length=12, y_length=0.1, y_range=[0,1], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': True,
            "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
            "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
        }).set_z_index(1)
        ax.y_axis.set_opacity(0)
        ticks = mh.get_xticks(ax, np.linspace(0, 10, 11), ['0']*11)
        ticks = VGroup(*[_[0] for _ in ticks])
        gp1 = VGroup(ax, ticks).move_to(ORIGIN).to_edge(DOWN, buff=1.4)

        brace = Brace(ticks[3:5], DOWN, stroke_width=1.5, color=RED, buff=0.1).set_z_index(1)
        bracelabel = Tex(r'\sf period', color=RED, stroke_width=1.5, font_size=50).set_z_index(1)
        bracelabel.next_to(brace, DOWN, buff=0.2)

        ymax = 8.3

        line2 = Line(ax.c2p(1, 0), ax.c2p(ymax, 0), color=BLUE, stroke_width=8).set_z_index(1)
        line2_ticks = [Line(p+0.15*DOWN, p+0.15*UP, stroke_width=8, stroke_color=BLUE).set_z_index(1)
                       for p in [line2.get_start(), line2.get_end()]]
        line2 = VGroup(line2, *line2_ticks)
        yshift = UP*0.4
        line2.shift(yshift)
        eqy = MathTex(r'y', stroke_width=1.5, font_size=50, color=BLUE).set_z_index(1)
        eqy.next_to(line2[0], UP, buff=0.15)
        dot1 = Dot(radius=0.2, color=RED).set_z_index(7)
        y1 = 5.7
        p1 = ax.c2p(np.floor(y1), 0)
        p2 = ax.c2p(y1, 0)
        line3 = DashedLine(p1 + yshift, p1 + DOWN*0.6, color=RED, stroke_width=6).set_z_index(6)
        line4 = DashedLine(p2 + yshift, p2 + DOWN*0.6, color=RED, stroke_width=6).set_z_index(6)
        arr1 = Arrow(p1+DOWN*0.4, p2+DOWN*0.4+LEFT*0.05, stroke_width=6, color=RED, buff=0,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4).set_z_index(2)
        txt1 = Tex(r'\sf $\sim$ ', r'uniform', stroke_width=1.5, font_size=50)
        txt1[1].set_color(col_txt)
        txt1.next_to(arr1, DOWN, buff=0.2)
        txt1 = mh.eq_shadow(txt1, bg_stroke_width=14, bg_z_index=8, fg_z_index=9)

        radius = 1.8
        circ1 = Arc(radius=radius, stroke_width=10, stroke_color=WHITE, start_angle=-PI/2, angle=TAU).set_z_index(1)
        circ1.move_to(UP*1.2)
        circ2 = Circle(radius=radius).move_to(circ1)
        p1 = circ1.get_start()
        tick1 = Line(p1+UP*0.15, p1+DOWN*0.15, stroke_width=8, stroke_color=WHITE).set_z_index(1)
        ax2 = Axes(x_range=[-1, 1], y_range=[-1, 1], x_length=2*radius, y_length=2*radius).set_opacity(0)
        ax2.shift(circ1.get_center() - ax2.c2p(0,0))

        box_args = get_kwargs(stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                              buff=0.2, corner_radius=0.2)
        box1 = SurroundingRectangle(VGroup(ax.x_axis, txt1, eqy), **box_args).set_z_index(0.1)
        box2 = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1)

        box3 = Difference(box2, circ2, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=1)


        self.add(gp1, box1)

        self.play(FadeIn(brace, bracelabel))
        self.wait(0.1)
        self.play(FadeIn(line2, eqy))
        self.wait(0.1)
        dot1.move_to(ax.c2p(1,0)+yshift)
        self.play(FadeIn(dot1), FadeOut(brace, bracelabel), rate_func=linear, run_time=0.5)
        self.play(dot1.animate(rate_func=mh.rate_func_quad(0, 0.15), run_time=2).move_to(ax.c2p(y1, 0)+yshift))
        self.play(Create(line3), Create(line4), run_time=0.6)
        self.play(FadeIn(arr1, txt1))
        self.wait(0.1)
        """
        circle
        """
        self.play(FadeOut(line3, line4, arr1, txt1),
                  FadeIn(tick1),
                  Create(circ1, rate_func=linear, run_time=2))

        def pos_func(theta):
            scale = 1.04 + theta * 0.04
            return np.sin(theta * TAU) * scale, -np.cos(theta * TAU) * scale

        crv1 = ax2.plot_line_graph(*pos_func(np.linspace(0, ymax-1, 1000)), add_vertex_dots=False, stroke_width=5, stroke_color=BLUE)
        crv1.set_z_index(6)
        self.play(mh.rtransform(line2[0].copy(), crv1, run_time=2.5),
                  FadeIn(box3, run_time=1, rate_func=linear))
        self.remove(box1)
        self.wait(0.1)
        dot2 = dot1.copy()
        yshift2 = -0.4
        self.play(dot1.animate.move_to(ax.c2p(1,0) + yshift+yshift2),
                  dot2.animate.move_to(ax2.c2p(*pos_func(0))),
                  VGroup(ticks, ax, line2, eqy).animate.shift(yshift2))
        self.wait(0.1)

        y2 = 4.4
        crv2 = ax2.plot_line_graph(*pos_func(np.linspace(0, y2-1, 1000)))
        rate_func=mh.rate_func_quad(0.1, 0.15)
        self.play(AnimationGroup(MoveAlongPath(dot2, crv2['line_graph'], rate_func=rate_func),
                  dot1.animate(rate_func=rate_func).move_to(ax.c2p(y2,0)+yshift),
                  run_time=4),
                  FadeIn(box2))
        self.remove(box3)
        p1 = ax.c2p(np.floor(y2), 0)
        p2 = ax.c2p(y2, 0)
        line5 = DashedLine(p1 + yshift, p1 + DOWN*0.6, color=RED, stroke_width=6).set_z_index(6)
        line6 = DashedLine(p2 + yshift, p2 + DOWN*0.6, color=RED, stroke_width=6).set_z_index(6)
        arr2 = Arrow(p1+DOWN*0.4, p2+DOWN*0.4, stroke_width=6, color=RED, buff=0,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4).set_z_index(2)
        p1 = circ1.get_center()
        p2 = circ1.get_start()
        p3 = dot2.get_center()
        line7 = DashedLine(p1, p2, color=RED, stroke_width=8).set_z_index(7)
        line8 = DashedLine(p1, p3, color=RED, stroke_width=8).set_z_index(7)
        arc1 = Arc(radius = radius * 0.5, arc_center=circ1.get_center(), start_angle=-PI/2, angle=(y2 % 1) * TAU,
                   stroke_width=8, stroke_color=RED).set_z_index(4)
        tip1 = StealthTip(stroke_width=6, color=RED).rotate(y2*TAU-PI*0.04).set_z_index(4)
        tip1.shift(arc1.get_end() - tip1.get_end())
        eqU = MathTex(r'U', color=col_var, stroke_width=1.5, font_size=50).set_z_index(8)
        eqU.move_to(arc1)

        self.play(Create(line5), Create(line6), Create(line7), Create(line8))
        self.play(FadeIn(arr2), Create(arc1),
                  Succession(Wait(0.5), FadeIn(tip1)),
                  Succession(Wait(0.8), FadeIn(txt1.next_to(arc1, RIGHT, buff=0.1), eqU)))
        self.wait(0.1)

        """
        arcsine plot
        """
        gp2 = VGroup(ax2, arc1, tip1, txt1, line7, line8, dot2, circ1, tick1, crv1, eqU)
        self.play(gp2.animate.to_edge(LEFT, buff=0.5))
        self.wait(0.1)
        p1 = circ1.get_top()
        p2 = circ1.get_bottom()
        p3 = p1 * UP
        p4 = p2 * UP
        p5 = ax2.c2p(np.sin(y2*TAU), -np.cos(y2*TAU))
        p6 = p5 * UP
        p7 = circ1.get_center()
        p8 = p7 * UP
        dx = RIGHT*0.2
        line9 = Line(p3, p4, stroke_width=8, stroke_color=WHITE).set_z_index(1)
        line10 = DashedLine(p1, p3+dx, stroke_width=6, stroke_color=WHITE).set_z_index(8)
        line11 = DashedLine(p2, p4+dx, stroke_width=6, stroke_color=WHITE).set_z_index(8)
        line12 = DashedLine(p5, p6+dx, stroke_width=6, stroke_color=RED).set_z_index(9)
        line13 = DashedLine(p7, p8+dx, stroke_width=6, stroke_color=RED).set_z_index(9)
        arr3 = Arrow(p8+dx, p6+dx, stroke_width=8, color=RED, buff=0,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4).set_z_index(9)
        eq1 = MathTex(r'-\cos U', r'\sim', r'\sf arcsine', stroke_width=1.5, font_size=60).set_z_index(5)
        eq1[0][1:4].set_color(col_trig)
        eq1[0][-1].set_color(col_var)
        eq1[2].set_color(col_txt)
        eq1.next_to(arr3, RIGHT, buff=0)
        rate_func = mh.rate_func_quad(0, 0.3)
        self.play(AnimationGroup(Create(line10), Create(line11), Create(line12), Create(line13), rate_func=rate_func, run_time=1),
                  FadeIn(line9, run_time=1),
                  Succession(Wait(0.8), FadeIn(eq1[0], arr3, run_time=1)))
        self.wait(0.1)
        self.play(FadeIn(eq1[1:]))
        self.wait(0.1)

        ax3 = Axes(x_range=[-1, 1.1], y_range=[0, 1.5], x_length=5, y_length=2.5,
                  axis_config={'color': WHITE, 'stroke_width': 6, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(5)
        ax3.next_to(eq1[2], DOWN, buff=0.6)
        ax3.move_to((arr3.get_center()+mh.pos(RIGHT))/2, coor_mask=RIGHT)
        ticks2 = mh.get_xticks(ax3, [-1, 1], label_color=col_num, font_size=45).set_z_index(4)
        def p(u):
            return 1 / np.pi / np.sqrt(1-u*u)
        xvec = np.sin(np.linspace(-0.85, 0.85, 1000) * PI/2)
        yvec = p(xvec)
        plt1 = ax3.plot_line_graph(xvec, yvec, add_vertex_dots=False, stroke_width=6, stroke_color=GREEN).set_z_index(7)
        area1 = ax3.plot_line_graph(np.concatenate([[-1, -1, xvec[0]], xvec, [xvec[-1], 1, 1]]),
                                    np.concatenate([[0, yvec[0], yvec[0]], yvec, [yvec[-1], yvec[-1], 0]]),
                                                   add_vertex_dots=False, stroke_width=0, stroke_opacity=0,
                                                   fill_color=GREEN, fill_opacity=0.5).set_z_index(6)
        eq2 = MathTex(r'p(u)', r'=', r'\frac1{\pi\sqrt{1-u^2} }', font_size=50, stroke_width=1.5)
        eq2[0][0].set_color(col_txt2)
        VGroup(eq2[0][-2], eq2[2][-3:]).set_color(col_x)
        VGroup(eq2[2][0], eq2[2][-4]).set_color(col_num)
        eq2[2][1:-4].set_color(col_op)
        eq2[2][2].set_color(col_pi)
        eq2 = mh.eq_shadow(eq2, bg_z_index=8, fg_z_index=9, bg_stroke_width=10)
        eq2.move_to(ax3.c2p(0, 1))
        self.play(Create(plt1, rate_func=mh.rate_func_quad(0.2, 0.2)),
                  FadeIn(area1),
                  FadeIn(eq2),
                  FadeIn(ax3, ticks2)
                  )
        self.wait(0.1)

        eq3 = MathTex(r'\frac2{\lvert\rho\rvert}', r'\cos(\gamma y-\theta)', r'\sim', r'{\sf arcsine}',
                      stroke_width=1.5, font_size=60).set_z_index(4)
        eq3[0][-2].set_color(col_zero)
        VGroup(eq3[0][1:3], eq3[0][1]).set_color(col_op)
        eq3[0][0].set_color(col_num)
        eq3[1][4].set_color(col_p)
        eq3[1][5].set_color(col_x)
        eq3[1][-2].set_color(col_angle)
        mh.align_sub(eq3, eq3[1][:3], eq1[0][1:4])
        self.play(mh.rtransform(eq1[0][1:4], eq3[1][:3], eq1[1:], eq3[2:], copy_colors=True),
                  Succession(Wait(0.5), AnimationGroup(FadeOut(eq1[0][-1]), FadeIn(eq3[1][3:]))),
                  FadeOut(eq1[0][0]))
        self.wait(0.1)
        eq3_1 = eq3.copy().next_to(arr3, RIGHT, buff=0, coor_mask=RIGHT)
        self.play(FadeIn(eq3_1[0]), mh.rtransform(eq3[1:], eq3_1[1:]))
        self.wait(0.1)

        eq4, box4 = self.get_final()
        self.add(box4)

        self.play(FadeOut(ax, ticks, dot1, dot2, line2, circ1, tick1, line9, line10, line11, line12, line13, arr3, arc1,
                          tip1, eqU, arr2, eqy, line7, line8, crv1, line5, line6, txt1, ax3, plt1, area1, eq2, ticks2,
                          eq3_1[2:], box2, rate_func=linear),
                  mh.rtransform(eq3_1[:2], eq4[-2:], run_time=2),
                  Succession(Wait(1), FadeIn(eq4[:-2]))
                  )

        self.wait()

class TwoDistribution(UniformInterval):
    bgcolor = BLACK
    def get_final(self, shifted=False):
        MathTex.set_default(stroke_width=1.5, font_size=60)
        width = 4.5
        ax = Axes(x_range=[0, 1], x_length=width, y_length=width, y_range=[0,1], axis_config={
            'stroke_width': 6, 'color': WHITE, 'include_ticks': False, 'include_tip': False,
        }).set_z_index(4)
        xticks = mh.get_xticks(ax, [0, 1], label_color=col_num).set_z_index(4)
        yticks = mh.get_yticks(ax, [0, 1], label_color=col_num).set_z_index(4)
        eq3 = MathTex(r'\{ ay\}', r'\{ by \}', font_size=80).set_z_index(5)
        VGroup(eq3[0][1], eq3[1][1]).set_color(PURPLE)
        VGroup(eq3[0][0], eq3[0][-1], eq3[1][0], eq3[1][-1]).set_color(col_op)
        VGroup(eq3[0][2], eq3[1][2]).set_color(col_x)
        eq3_1 = eq3.copy()
        eq3_1[1].scale(0.6).next_to(ax.c2p(0, 0.5), LEFT, buff=0.2),
        eq3_1[0].scale(0.6).next_to(ax.c2p(0.5, 0), DOWN, buff=0.2),
        origin = ax.c2p(0,0)
        square1 = Rectangle(width=width, height=width, stroke_width=0, stroke_opacity=0, fill_color=BLUE, fill_opacity=0.8).set_z_index(3)
        square1.next_to(origin, UR, buff=0)
        gp = VGroup(ax, xticks, yticks, eq3_1, square1)
        if not shifted:
            gp = gp.copy()
        gp.to_edge(LEFT, buff=0.2)

        dot1 = Dot(radius=0.15, color=RED).move_to(gp[0].c2p(0,0)).set_z_index(8)

        return ax, xticks, yticks, eq3, eq3_1, square1, dot1

    def construct(self):
        eq1, box1 = UniformInterval.get_final(self)
        ax, xticks, yticks, eq3, eq3_1, square1, dot1 = self.get_final()

        eq2 = MathTex(r'\cos(\gamma_1y-\theta_1)',
                      r'\cos(\gamma_2y-\theta_2)').set_z_index(2)
        eq2[1].shift(RIGHT*1.1)
        txt1 = Tex(r'\sf fractional parts', color=RED).set_z_index(2)

        mh.align_sub(eq2, eq2[0][:3], eq1[-1][:3]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq3, eq3[0][2], eq2[0][6])
        eq3[0].move_to(eq2[0], coor_mask=RIGHT)
        eq3[1].move_to(eq2[1], coor_mask=RIGHT)
        width = box1.width
        height = 2 * (box1.get_top() - eq2.get_center())[1]
        box2 = RoundedRectangle(width=width, height=height, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                corner_radius=0.15).move_to(eq2)

        mh.rtransform.copy_colors=True
        VGroup(eq2[0][5:13:4], eq2[1][5:13:4]).set_color(col_num)
        txt1.next_to(eq3, UP, buff=0.2)

        txt1 = mh.eq_shadow(txt1, bg_stroke_width=12)
        arr1 = Arrow(txt1.get_bottom()+LEFT*0.2, eq3[0].get_corner(UR)+DL*0.1+DOWN*0.2, buff=0.1, color=RED, stroke_width=6).set_z_index(3)
        arr2 = Arrow(txt1.get_bottom()-LEFT*0.2, eq3[1].get_corner(UL)+DR*0.1+DOWN*0.2, buff=0.1, color=RED, stroke_width=6).set_z_index(3)

        self.add(eq1, box1)
        eq_1 = eq1[-1].copy()
        self.play(FadeOut(eq1[:5]), AnimationGroup(
            mh.rtransform(eq1[-1][:5], eq2[0][:5], eq1[-1][5:8], eq2[0][6:9], eq1[-1][-1], eq2[0][-1],
                          eq_1[:5], eq2[1][:5], eq_1[5:8], eq2[1][6:9], eq_1[-1], eq2[1][-1],
                          box1, box2),
            FadeIn(eq2[0][5], shift=mh.diff(eq1[-1][4], eq2[0][4])),
            FadeIn(eq2[0][9], shift=mh.diff(eq1[-1][7], eq2[0][8])),
            FadeIn(eq2[1][5], shift=mh.diff(eq1[-1][4], eq2[1][4])),
            FadeIn(eq2[1][9], shift=mh.diff(eq1[-1][7], eq2[1][8])),
            run_time=1.8)
        )
        self.wait(0.1)
        self.play(FadeOut(eq2[0][:4], eq2[0][7:], eq2[1][:4], eq2[1][7:]),
                  AnimationGroup(
                      mh.rtransform(eq2[0][6], eq3[0][2], eq2[1][6], eq3[1][2]),
                      mh.fade_replace(eq2[0][4:6], eq3[0][1], coor_mask=RIGHT),
                      mh.fade_replace(eq2[1][4:6], eq3[1][1], coor_mask=RIGHT),
                      run_time=1.5
                  ),
                  Succession(Wait(0.7), FadeIn(eq3[0][0], eq3[0][-1], eq3[1][0], eq3[1][-1])))
        self.play(FadeIn(txt1, arr1, arr2))
        self.wait(0.1)

        """
        square
        """
        box3 = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1)

        self.play(FadeOut(arr1, arr2, txt1),
                  Transform(eq3, eq3_1, run_time=1.6),
                  Succession(Wait(1), FadeIn(ax, square1, xticks, yticks, run_time=1.6)),
                  FadeIn(box3, run_time=2.6),
                  )
        self.remove(box2)
        self.wait(0.1)
        gp1 = VGroup(ax, xticks, yticks, eq3, square1)
        self.play(gp1.animate(run_time=1.5).to_edge(LEFT, buff=0.2),
                  FadeIn(dot1, run_time=1.5, target_position=ax.c2p(0,0)))


        self.wait()

class TwoDistribution3(TwoDistribution):
    bgcolor = BLACK
    do_line = True
    run_time = 2.
    a = 1.
    b = 1.
    # a = 2.
    # b = 3.
    # run_time = 2

    def construct(self):
        ax, xticks, yticks, _, eq3, square1, dot1 = self.get_final(shifted=True)
        self.add(ax, xticks, yticks, eq3, square1)
        val = ValueTracker(0.)
        eps = 1 - 1e-5
        a = self.a * eps
        b = self.b * eps

        def get_obj():
            t = val.get_value()
            p1 = ax.c2p(t * a % 1, t * b % 1)
            dot = dot1.copy().move_to(p1)
            if not self.do_line:
                return dot
            lines = []
            nx = ny = 0
            t0 = 0.
            while t0 < t:
                x0 = t0*a - nx
                y0 = t0*b - ny
                p0 = ax.c2p(x0, y0)

                t1 = (1 - x0)/a
                t2 = (1 - y0)/b
                t3 = min(t1, t2)
                if t1 < t2: nx += 1
                else: ny += 1
                x1 = x0 + min(t3,t-t0) * a
                y1 = y0 + min(t3,t-t0) * b
                p1 = ax.c2p(x1, y1)
                line = Line(p0, p1, stroke_width=8, stroke_color=RED).set_z_index(10)
                lines.append(line)
                t0 += t3

            return VGroup(dot, *lines)

        if not self.do_line:
            self.do_line = True
            val.set_value(1)
            gp = get_obj()
            val.set_value(0)
            self.do_line = False
            self.add(gp[1:])

        obj = always_redraw(get_obj)
        self.add(obj)
        self.play(val.animate(rate_func=linear, run_time=self.run_time).set_value(1))
        obj.clear_updaters()
        self.wait()

class TwoDistribution3_1(TwoDistribution3):
    do_line = False

class TwoDistribution4(TwoDistribution3):
    a = 2.
    b = 3.
    run_time = 4.

class TwoDistribution4_1(TwoDistribution4):
    do_line = False

class TwoDistribution5(TwoDistribution3):
    a = 3.
    b = 4.
    run_time = 8.

class TwoDistribution5_1(TwoDistribution5):
    do_line = False

class TwoDistribution6(TwoDistribution3):
    a = 1 * 10
    b = (1 + math.sqrt(5))/2 * 10
    run_time = 12

class TwoDistributionEq1(TwoDistribution3):
    def construct(self):
        eq = MathTex(r'a=b', stroke_width=1.5, font_size=80)[0]
        eq[::2].set_color(PURPLE)
        self.add(eq)

class TwoDistributionEq2(TwoDistribution3):
    def construct(self):
        eq = MathTex(r'a:b=2:3', stroke_width=1.5, font_size=80)[0]
        eq[:4:2].set_color(PURPLE)
        eq[4::2].set_color(col_num)
        eq[1::4].set_color(col_op)
        self.add(eq)

class TwoDistributionEq3(TwoDistribution3):
    def construct(self):
        eq = MathTex(r'a:b=3:4', stroke_width=1.5, font_size=80)[0]
        eq[:4:2].set_color(PURPLE)
        eq[4::2].set_color(col_num)
        eq[1::4].set_color(col_op)
        self.add(eq)

class TwoDistributionEq4(TwoDistribution3):
    def construct(self):
        eq = MathTex(r'a:b=1:1.61803\cdots', r' ({\sf golden\ ratio})', stroke_width=1.5, font_size=80)
        eq[0][:4:2].set_color(PURPLE)
        eq[0][4:].set_color(col_num)
        eq[0][1:9:4].set_color(col_op)
        mh.font_size_sub(eq, 1, 60)
        eq[1].next_to(eq[0], DOWN, buff=0.4)
        eq.move_to(ORIGIN, coor_mask=RIGHT)
        eq[1][1:-1].set_color(col_txt)
        self.add(eq)

class MultiDistribution(UniformInterval):
    bgcolor = BLACK
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)
        eq1 = MathTex(r'a_1,a_2,a_3,\ldots, a_r', font_size=80)
        eq2 = MathTex(r'n_1a_1+n_2a_2+n_3a_3+\cdots+n_ra_r', r'=', r'0', font_size=70)
        eq3 = Tex(r'\sf do not satisfy any linear relation of the form')
        eq4 = Tex(r'\sf for whole numbers ', r'$n_1,\ldots,n_r$', r' not all zero')
        eq5 = MathTex(r'\left(', r'\{a_1y\}', r',', r'\{a_2y\}', r',\ldots,', r'\{a_ry\}', r'\right)', r'\sim',
                      r'{\sf independent, uniform}')
        eq7 = MathTex(r'\frac1{y_1-y_0}', r'\int_{y_0}^{y_1}', r'f', r'\left(', r'a_1y', r',', r'a_2y', r',\ldots,',
                      r'a_ry', r'\right)', r'\,dy',
                      r'\to', r'\mathbb E\left[', r'f', r'(', r'U_1', r',', r'U_2', r',\ldots,', r'U_r', r')', r'\right]')
        eq8 = MathTex(r'\to', r'\int_0^1\!\!\cdots\!\int_0^1', r'f', r'(', r'u_1', r',', r'u_2', r',\ldots,', r'u_r', r')',
                      r'\,du_1\ldots du_r')

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq1[0][:9:3], eq1[0][-2], *[eq5[_][1] for _ in (1,3,5)]).set_color(PURPLE)
        VGroup(eq1[0][1:10:3], eq1[0][-1], eq2[0][1:16:5], eq4[1][1], eq4[1][-1], eq2[-1],
               *[eq5[_][2] for _ in (1,3,5)], eq7[0][0],
               eq7[15][1], eq7[17][1], eq7[19][1], eq8[1][1:3], eq8[1][-2:], eq8[-1][2], eq8[-1][-1]).set_color(col_num)
        VGroup(eq2[0][:15:5], eq2[0][-4], eq4[1][0], eq4[1][-2],
               eq7[15][0], eq7[17][0], eq7[19][0], eq8[-1][1], eq8[-1][-2]).set_color(col_var)
        VGroup(eq3, eq4[0], eq4[2], eq5[-1]).set_color(col_txt)
        VGroup(*[eq5[_][::4] for _ in (1,3,5)], eq7[1][0], eq7[10][0], eq8[1][0], eq8[1][-3],
               eq8[-1][0], eq8[-1][-3]).set_color(col_op)
        VGroup(*[eq5[_][3] for _ in (1,3,5)], eq7[0][2:4], eq7[0][-2:], eq7[1][1:], eq7[10][1]).set_color(col_x)
        VGroup(eq7[2]).set_color(col_psi)
        VGroup(eq7[12]).set_color(col_txt2)

        eq3.next_to(eq2, UP, buff=0.2)
        eq1_1 = eq1.copy().next_to(eq3, UP, buff=0.8)
        eq4.next_to(eq2, DOWN, buff=0.2)
        eq5.next_to(eq4, DOWN, buff=0.6)

        gp1 = VGroup(eq1_1, eq2, eq3, eq4)
        gp2 = gp1.copy().to_edge(UP, buff=0.4)
        gp2[1:].next_to(gp2[0], DOWN, buff=0.3, coor_mask=UP)

        mh.align_sub(eq7, eq7[2:10], gp2, DOWN, buff=1)
        eq7[11:].next_to(eq7[2:11], DOWN, buff=1.).to_edge(RIGHT, buff=1.2)
        mh.align_sub(eq8, eq8[0], eq7[11]).to_edge(RIGHT, buff=0.2)
        # eq8.to_edge(RIGHT, buff=0.2).to_edge(DOWN, buff=0.5)

        self.add(eq1)
        self.play(AnimationGroup(mh.rtransform(eq1[0][:9:3].copy(), eq2[0][2:17:5], eq1[0][-2].copy(), eq2[0][-2],
                                eq1[0][1:9:3].copy(), eq2[0][3:17:5], eq1[0][-1].copy(), eq2[0][-1],
                                               eq1[0][-6:-3].copy(), eq2[0][-8:-5], eq1, eq1_1),
                                 FadeIn(eq2[0][:2], shift=mh.diff(eq1[0][0], eq2[0][2])),
                                 FadeIn(eq2[0][5:7], shift=mh.diff(eq1[0][3], eq2[0][7])),
                                 FadeIn(eq2[0][10:12], shift=mh.diff(eq1[0][6], eq2[0][12])),
                                 FadeIn(eq2[0][-4:-2], shift=mh.diff(eq1[0][-2], eq2[0][-2])),
                                 FadeIn(eq2[0][4], shift=mh.diff(eq1[0][2], eq2[0][4])+DOWN*0.2),
                                 FadeIn(eq2[0][9], shift=mh.diff(eq1[0][5], eq2[0][9])+DOWN*0.2),
                                 FadeIn(eq2[0][14], shift=mh.diff(eq1[0][8], eq2[0][14])+DOWN*0.2),
                                 FadeIn(eq2[0][-5], shift=mh.diff(eq1[0][-3], eq2[0][-5])+DOWN*0.2),
                                 FadeIn(eq2[1:], shift=mh.diff(eq1[0][-2], eq2[0][-2])),
                                 run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq3, eq4))
                  )
        self.play(FadeIn(eq5))
        self.wait(0.1)
        self.play(Succession(Wait(0.6), AnimationGroup(
            mh.rtransform(eq5[0], eq7[3], eq5[-3], eq7[9], eq5[2:6:2], eq7[5:9:2]),
            *[mh.rtransform(eq5[i][1:-1], eq7[i+3][:]) for i in (1, 3, 5)],
            FadeIn(eq7[2], shift=mh.diff(eq5[0], eq7[3])),
            mh.rtransform(gp1, gp2),
            run_time=1.5)),
                  *[FadeOut(eq5[i][::4]) for i in (1,3,5)],
                  FadeOut(eq5[-2:])
                  )
        self.wait(0.1)
        self.play(FadeIn(eq7[:2], eq7[10]))
        self.wait(0.1)
        eq7_1 = eq7.copy()
        self.play(Succession(Wait(1), FadeIn(eq7[11:13], eq7[-1])),
                  AnimationGroup(mh.rtransform(eq7_1[2], eq7[13], eq7_1[3:11:2], eq7[14:22:2]),
                  mh.fade_replace(eq7_1[4], eq7[15]),
                  mh.fade_replace(eq7_1[6], eq7[17]),
                  mh.fade_replace(eq7_1[8], eq7[19]),
                                 run_time=1.6
                  ))
        self.wait(0.1)
        br1 = Brace(eq7[15:-2], DOWN, color=RED, buff=0.1)
        txt1 = Tex(r'\sf independent uniform on ', r'$[0,1]$')
        txt1[0].set_color(RED)
        txt1[1][1::2].set_color(col_num)
        txt1.next_to(br1, DOWN, buff=0.2).to_edge(RIGHT, buff=0.5)
        self.play(FadeIn(br1, txt1))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq7[11], eq8[0], eq7[13:15], eq8[2:4],
                                eq7[16:24:2], eq8[5:13:2], eq7[15][1], eq8[4][1],
                                eq7[17][1], eq8[6][1], eq7[19][1], eq8[8][1]),
                  mh.stretch_replace(eq7[15][0], eq8[4][0]),
                  mh.stretch_replace(eq7[17][0], eq8[6][0]),
                  mh.stretch_replace(eq7[19][0], eq8[8][0]),
                  FadeOut(eq7[21], shift=mh.diff(eq7[20], eq8[9])),
                  FadeOut(br1, txt1, shift=mh.diff(eq7[15:-2], eq8[4:-2])),
                  mh.fade_replace(eq7[12], eq8[1]),
                                 run_time=1.5),
                  Succession(Wait(0.7), FadeIn(eq8[-1])))
        self.wait()

class Latexay(Scene):
    eqstr = r'a'
    def construct(self):
        MathTex.set_default(str)
        eq1 = MathTex(r'\{' + self.eqstr + r'y\}', stroke_width=1.5, font_size=50)
        eq1[0][1].set_color(PURPLE)
        eq1[0][2].set_color(col_x)
        eq1[0][::3].set_color(col_op)
        eq1 = mh.eq_shadow(eq1, bg_stroke_width=10)
        self.add(eq1)

class Latexby(Latexay):
    eqstr = r'b'

class Simplicity(UniformInterval):
    def get_final(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)
        eq = MathTex(r'\psi(x)', r'=', r'\sum_{p^r\le x}', r'\log p')
        eq2 = MathTex(r'J(x)', r'=', r'\int_0^x', r'\frac{1}{\log u}', r'\,d\psi(u)')
        VGroup(eq[0][0], eq2[0][0], eq2[-1][1]).set_color(col_WVD)
        VGroup(eq[0][2], eq[2][-1], eq2[0][2], eq2[2][1], eq2[3][-1], eq2[4][3]).set_color(col_x)
        VGroup(eq[2][0], eq2[2][0], eq2[-1][0], eq2[3][1]).set_color(col_op)
        VGroup(eq[2][1], eq[3][-1]).set_color(col_p)
        VGroup(eq[3][:3], eq2[3][2:5]).set_color(col_trig)
        VGroup(eq[2][2], eq2[2][2], eq2[3][0]).set_color(col_num)

        eq2.next_to(eq, DOWN, buff=1)
        VGroup(eq, eq2).move_to(ORIGIN)

        return eq, eq2

    def construct(self):
        eq6, eq7 = self.get_final()
        eq1 = MathTex(r'\psi(x)-x', r'\approx', r'\sqrt x', r'\sum_{\rho\colon\gamma > 0}', r'\frac2{\lvert\rho\rvert}',
                r'\cos(\gamma y-\theta)').set_z_index(1)
        eq2 = MathTex(r'\frac1{\lvert\rho\rvert^2}', r'<', r'\infty').set_z_index(1)
        eq3 = MathTex(r'n_1\gamma_1', r'+', r'n_2\gamma_2', r'+\cdots+', r'n_r\gamma_r', r'=', r'0').set_z_index(1)
        eq4 = Tex(r"\sf nontrivial: ", r'$n_1, n_2,\ldots,n_r$', r' not all ', r'$0$').set_z_index(1)
        eq5 = MathTex(r'\psi(x)-x', r'\sim', r'\sqrt x', r'\sum_{\rho\colon\gamma > 0}', r'\frac2{\lvert\rho\rvert}',
                      r'\cos U_\rho').set_z_index(1)
        eq6_1 = eq6.copy()


        mh.align_sub(eq2, eq2[0][1], eq1[4][1])
        gp = VGroup(eq1, eq2).set_z_index(2).to_edge(DOWN, buff=0.4)

        eq3.next_to(eq1, DOWN, buff=0.2)
        gp2 = VGroup(eq1.copy(), eq3).to_edge(DOWN, buff=0.4)
        mh.align_sub(eq4, eq4[1][0], eq3[3][0][0], coor_mask=UP)
        mh.align_sub(eq5, eq5[-2], eq1[-2])
        mh.align_sub(eq6, eq6[0][0], eq5[0][0], coor_mask=UP)

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][0]).set_color(col_WVD)
        VGroup(eq1[0][2], eq1[0][-1], eq1[2][2], eq1[5][5]).set_color(col_x)
        VGroup(eq1[3][0], eq1[2][:2], eq1[4][1:3], eq1[4][-1]).set_color(col_op)
        VGroup(eq1[3][1], eq1[4][3], eq5[-1][-1]).set_color(col_zero)
        VGroup(eq1[3][3], eq1[5][4]).set_color(col_p)
        VGroup(eq1[3][-1], eq1[4][0], eq2[0][0], eq2[0][-1], eq3[0][3], eq3[2][3], eq3[4][3], eq3[-1],
               eq3[0][1], eq3[2][1], eq3[4][1], eq4[1][1], eq4[1][4], eq4[1][-1]).set_color(col_num)
        VGroup(eq1[5][:3]).set_color(col_trig)
        VGroup(eq1[5][-2]).set_color(col_angle)
        VGroup(eq2[-1]).set_color(col_special)
        VGroup(eq3[0][0], eq3[2][0], eq3[4][0], eq4[1][0], eq4[1][3], eq4[1][-2], eq4[3], eq5[-1][-2]).set_color(col_var)
        VGroup(eq4[::2]).set_color(col_txt)

        box = SurroundingRectangle(gp, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   corner_radius=0.15, buff=0.2)
        box2 = SurroundingRectangle(gp2, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   corner_radius=0.15, buff=0.2)

        self.add(box, eq1)
        eq1_1 = eq1.copy()
        self.play(mh.rtransform(eq1[4][1:], eq2[0][1:-1]),
                  FadeIn(eq2[0][-1]),
                  mh.fade_replace(eq1[4][0], eq2[0][0]),
                  FadeOut(eq1[:3], eq1[-1]),
                  FadeIn(eq2[1:]))
        self.remove(eq1[3])
        eq1= eq1_1
        self.add(eq1[3])
        self.wait(0.1)
        self.play(mh.rtransform(eq2[0][1:-1], eq1[4][1:], ),
                  FadeOut(eq2[0][-1]),
                  mh.fade_replace(eq2[0][0], eq1[4][0]),
                  FadeIn(eq1[:3], eq1[-1]),
                  FadeOut(eq2[1:]))
        self.wait(0.1)
        eq1_1 = eq1[-1][4]
        eq1_2 = eq1.copy()
        box_1 = box.copy()
        self.play(AnimationGroup(mh.rtransform(eq1, gp2[0], box, box2,
                                eq1_1.copy(), eq3[0][2], eq1_1.copy(), eq3[2][2], eq1_1.copy(), eq3[4][2]),
                  FadeIn(eq3[0][3], shift=mh.diff(eq1_1, eq3[0][2])),
                  FadeIn(eq3[2][3], shift=mh.diff(eq1_1, eq3[2][2])),
                  FadeIn(eq3[4][3], shift=mh.diff(eq1_1, eq3[4][2])),
                  run_time=1.5),
                  Succession(Wait(1.1), FadeIn(eq3[0][:2], eq3[1:7:2], eq3[2][:2], eq3[4][:2], eq3[-1]))
                  # FadeIn(eq3)
                  )
        self.wait(0.1)
        line1 = Line(eq3[0][0].get_corner(DL), eq3.get_corner(UR)+DOWN*0.1, stroke_width=8, color=RED).set_z_index(4)
        self.play(Create(line1, rate_func=linear, run_time=1))
        self.wait(0.1)
        self.play(FadeOut(eq3, line1), FadeIn(eq4))
        self.wait(0.1)
        self.play(mh.rtransform(gp2[0], eq1_2, box2, box_1),
                  FadeOut(eq4))
        self.wait(0.1)
        self.play(mh.rtransform(eq1_2[0], eq5[0], eq1_2[2:-1], eq5[2:-1], eq1_2[-1][:3], eq5[-1][:3]),
                  FadeOut(eq1_2[-1][3:]), FadeIn(eq5[-1][3:]),
                  mh.fade_replace(eq1_2[1], eq5[1]))
        self.wait(0.1)
        txt1 = Tex(r'\sf independent', color=RED, font_size=50).set_z_index(4)
        txt1.next_to(eq5[-1][-2].get_right(), buff=1).shift(DOWN*0.4)
        arr1 = Arrow(txt1.get_left()+UP*0.1, eq5[-1][-2].get_right(), buff=0.1, color=RED, stroke_width=6,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4).set_z_index(4)
        txt1 = mh.eq_shadow(txt1, bg_z_index=1, fg_z_index=2, bg_stroke_width=12)
        self.play(FadeIn(txt1, arr1))
        self.wait(0.1)
        self.play(FadeOut(txt1, arr1),
                  Succession(Wait(0.7), eq5.animate.move_to(ORIGIN, coor_mask=RIGHT)))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[0][:4], eq6[0][:4]),
                  FadeOut(eq5[0][4:], eq5[2:]),
                  mh.fade_replace(eq5[1], eq6[1]),
                  FadeIn(eq6[2:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq6, eq6_1, run_time=1.5),
                  Succession(Wait(1), FadeIn(eq7)),
                  FadeOut(box_1))

        self.wait()

class RiemannJ(Simplicity):
    bgcolor = BLACK
    def construct(self):
        self.do_anim()

    def do_anim(self, just_eq=False):
        eq1, eq2 = self.get_final()

        eq3 = MathTex(r'J(x)', r'=', r'\sum_{p^r\le x}', r'\frac1{\log p^r}', r'\log p')
        eq4 = MathTex(r'J(x)', r'=', r'\sum_{p^r\le x}', r'\frac1{r\log p}', r'\log p')
        eq5 = MathTex(r'J(x)', r'=', r'\sum_{p^r\le x}', r'\frac1{r}')
        eq6 = MathTex(r'J(x)', r'=', r'\sum_{p\le x}', r'1', r'+', r'\sum_{p^2\le x}', r'\frac12', r'+',
                      r'\sum_{p^3\le x}', r'\frac13', r'+', r'\sum_{p^4\le x}', r'\frac14',
                      r'+', r'\cdots')
        # mh.font_size_sub(eq6, 6, 50)
        # mh.font_size_sub(eq6, 9, 50)
        # mh.font_size_sub(eq6, 12, 50)
        eq7 = MathTex(r'J(x)', r'=', r'\sum_{p\le x}', r'1', r'+', r'\sum_{p\le x^{\frac12} }', r'\frac12', r'+',
                      r'\sum_{p\le x^{\frac13} }', r'\frac13', r'+', r'\sum_{p\le x^{\frac14} }', r'\frac14',
                      r'+', r'\cdots')
        eq8 = MathTex(r'J(x)', r'=', r'\pi(x)', r'+', r'\frac{\pi(x^{\frac12})}2',
                      r'+', r'\frac{\pi(x^{\frac13})}3', r'+', r'\frac{\pi(x^{\frac14})}4', r'+', r'\cdots')
        eq9 = MathTex(r'\pi(x)', r'=', r'J(x)', r'-', r'\frac{\pi(x^{\frac12})}2',
                      r'-', r'\frac{\pi(x^{\frac13})}3', r'-', r'\frac{\pi(x^{\frac14})}4', r'-', r'\cdots')
        eq10 = MathTex(r'\pi(x^{\frac12})', r'=', r'J(x^{\frac12})', r'-', r'\frac{\pi(x^{\frac14})}2', r'-', r'\cdots')
        eq11 = MathTex(r'\frac{\pi(x^{\frac12})}2', r'=', r'\frac{J(x^{\frac12})}2', r'-', r'\frac{\pi(x^{\frac14})}4', r'-', r'\cdots')
        eq12 = MathTex(r'\pi(x)', r'=', r'J(x)', r'-', r'\frac{J(x^{\frac12})}2',
                      r'-', r'\frac{\pi(x^{\frac13})}3', r'+', r'\cdots')
        eq13 = MathTex(r'\pi(x^{\frac13})', r'=', r'J(x^{\frac13})', r'-', r'\cdots')
        eq14 = MathTex(r'\pi(x)', r'=', r'J(x)', r'-', r'\frac{J(x^{\frac12})}2',
                      r'-', r'\frac{J(x^{\frac13})}3', r'+', r'\cdots')
        eq15 = MathTex(r'=', r'\sum_n', r'\mu(n)', r'\frac{J(x^{\frac1n})}n')
        eq16 = MathTex(r'\psi(x)', r'\approx', r'x', r'+', r'\sqrt{x}', r'\mathcal E(x)')
        eq17 = Tex(r'\sf for noise ', r'$\mathcal E(x)=\sum\frac2{\lvert\rho\rvert} \cos(\gamma\log x-\theta)$')
        eq18 = MathTex(r'J(x)', r'\approx', r'\int_1^x\frac1{\log u}du', r'+', r'\int_1^x', r'\frac1{\log u}', r'd(\sqrt u \mathcal E(u))')
        eq19 = MathTex(r'J(x)', r'-', r'{\rm Li}(x)', r'\approx', r'\int_1^x', r'\frac1{\log u}', r'd(\sqrt u \mathcal E(u))')
        eq20 = MathTex(r'J(x)', r'-', r'{\rm Li}(x)', r'\approx', r'\frac{\sqrt x}{\log x}', r'\mathcal E(x)',
                       r'+', r'\int_1^x', r'\sqrt u\mathcal E(u)', r'\frac{du}{u\log^2u}')
        eq21 = MathTex(r'J(x)', r'-', r'{\rm Li}(x)', r'\approx', r'\frac{\sqrt x}{\log x}', r'\mathcal E(x)')
        eq22 = MathTex(r'\pi(x)', r'=', r'{\rm Li}(x)', r'-', r'\frac{ {\rm Li}(x^{\frac12})}2',
                       r'-', r'\frac{ {\rm Li}(x^{\frac13})}3', r'+', r'\cdots',
                       r'+', r'\frac{\sqrt x}{\log x}', r'\mathcal E(x)', r'+', r'{\sf smaller\ terms}')
        eq23 = MathTex(r'\pi(x)', r'\approx', r'{\rm Li}(x)', r'-', r'\frac{ {\rm Li}(x^{\frac12})}2',
                       r'-', r'\frac{ {\rm Li}(x^{\frac13})}3', r'+', r'\cdots',
                       r'+', r'\frac{\sqrt x}{\log x}', r'\mathcal E(x)')
        eq24 = MathTex(r'\mathcal E(x)', r'=', r'\sum_{\Re[\rho] > 0}', r'\frac2{\lvert\rho\rvert}', r'\cos U_\rho')

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq3[3][-2], eq3[4][-1], eq3[2][1]).set_color(col_p)
        VGroup(eq3[3][-1], eq3[2][2], eq6[5][2], eq6[8][2], eq6[11][2], eq6[6][2], eq6[9][2], eq6[12][2],
               eq11[0][-1], eq11[2][-1], eq15[1][1], eq15[2][2], eq15[3][-6:-2:2], eq15[3][-1],
               eq17[1][6], eq18[2][2:4], eq24[2][-1]).set_color(col_num)
        VGroup(eq3[4][:3], eq17[1][11:14], eq17[1][16:19], eq18[2][5:8]).set_color(col_trig)
        VGroup(eq3[2][0], eq11[0][-2], eq11[2][-2], eq15[1][0], eq15[3][-5::3],
               eq16[4][:-1], eq17[1][5:9:2], eq17[8:12:2], eq18[2][0], eq18[2][4], eq18[2][9], eq18[6][0]).set_color(col_op)
        VGroup(eq8[2][0], eq8[4][0], eq8[6][0], eq8[8][0], eq15[3][0], eq16[-1][0], eq18[0][0],
               eq19[2][:2], eq17[1][0], eq22[2][:2], eq22[4][:2], eq22[6][:2]).set_color(col_WVD)
        VGroup(eq15[3][2], eq16[2], eq16[4][-1], eq16[5][2], eq17[1][2], eq17[1][19], eq18[2][-1],
               eq18[2][1], eq18[2][8]).set_color(col_x)
        eq15[2][0].set_color(RED_C)
        VGroup(eq17[0], eq22[-1]).set_color(col_txt)
        VGroup(eq17[1][9], eq24[2][3], eq24[-1][-1]).set_color(col_zero)
        eq17[1][15].set_color(col_p)
        eq17[1][-2].set_color(col_angle)
        VGroup(eq24[-1][-2]).set_color(col_var)
        VGroup(eq24[2][1]).set_color(col_special)
        mh.copy_colors_eq(eq17[1][:-9], VGroup(*eq24[0][:], eq24[1][0], eq24[2][0], *eq24[3][:], *eq24[4][:3]))
        eq24.to_edge(DOWN, buff=0.4)
        if just_eq:
            return eq24

        mh.align_sub(eq3, eq3[1], eq2[1])
        mh.align_sub(eq4, eq4[1], eq3[1])
        mh.align_sub(eq5, eq5[1], eq3[1])
        mh.align_sub(eq6, eq6[1], eq5[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[1], eq6[1], coor_mask=UP)
        mh.align_sub(eq8, eq8[1], eq6[1], coor_mask=UP)
        mh.align_sub(eq9, eq9[1], eq8[1], coor_mask=UP)
        mh.align_sub(eq10, eq10[1], eq9[1]).next_to(eq9, DOWN, buff=0.4, coor_mask=UP)
        gp1 = VGroup(eq1.copy(), eq9.copy(), eq10).move_to(ORIGIN, coor_mask=UP)
        mh.align_sub(eq11, eq11[1], eq10[1])
        eq11[2].move_to(eq9[4], coor_mask=RIGHT)
        eq11[3:].align_to(eq9[7], LEFT)
        mh.align_sub(eq12, eq12[1], gp1[1][1])
        eq12[-2:].align_to(eq9[-2:], LEFT)
        mh.align_sub(eq13, eq13[1], eq11[1])
        mh.align_sub(eq14, eq14[1], eq12[1])
        eq14_2 = eq14.copy()
        eq14[-2:].align_to(eq12[-2:], LEFT)
        eq14_3 = eq14.copy().move_to(ORIGIN).to_edge(DOWN, buff=1)
        mh.align_sub(eq15, eq15[0], eq14[1]).next_to(eq14, DOWN, buff=0.3, coor_mask=UP)
        mh.align_sub(eq16, eq16[1], gp1[0][1], coor_mask=UP)
        eq17.next_to(eq16, DOWN, buff=1)
        mh.align_sub(eq18, eq18[1], eq16[1], coor_mask=UP).shift(DOWN*0.5)
        mh.align_sub(eq19, eq19[1], eq18[1])
        mh.align_sub(eq20, eq20[1], eq19[1])
        eq20[-4:].align_to(eq20[4], LEFT).next_to(eq20[-4:], DOWN, buff=0.2, coor_mask=UP)
        eq21.move_to(eq20[:6]).move_to(ORIGIN, coor_mask=RIGHT).shift(DOWN*1.3)

        self.add(eq1, eq2)
        self.play(mh.rtransform(eq2[:2], eq3[:2], eq2[2][1], eq3[2][-1], eq2[3][:5], eq3[3][:5]),
                  FadeOut(eq2[2][0], eq2[2][2]),
                  mh.fade_replace(eq2[3][-1], eq3[3][-2:], coor_mask=RIGHT),
                  FadeIn(eq3[2][:-1]),
                  mh.fade_replace(eq2[4], eq3[4], coor_mask=RIGHT)
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:3], eq4[:3], eq3[3][:2], eq4[3][:2], eq3[3][2:-1], eq4[3][3:], eq3[4], eq4[4]),
                  mh.stretch_replace(eq3[3][-1], eq4[3][2]))
        self.wait(0.1)
        self.play(FadeOut(eq4[3][-4:], eq4[4]),
                  Succession(Wait(0.8), mh.rtransform(eq4[:3], eq5[:3], eq4[3][:3], eq5[3][:])))
        self.wait(0.1)
        eq_1 = eq5[2:].copy()
        eq_2 = eq_1.copy()
        eq_3 = eq_1.copy()
        eq_4 = eq_1.copy()
        self.remove(eq5[2:].set_opacity(0))
        self.play(AnimationGroup(mh.rtransform(eq5[:2], eq6[:2],
                                eq_1[0][:2], eq6[2][:2], eq_1[0][-2:], eq6[2][-2:], eq_1[1][0], eq6[3][0],
                                eq_2[0][:2], eq6[5][:2], eq_2[0][-2:], eq6[5][-2:], eq_2[1][:2], eq6[6][:2],
                                eq_3[0][:2], eq6[8][:2], eq_3[0][-2:], eq6[8][-2:], eq_3[1][:2], eq6[9][:2],
                                eq_4[0][:2], eq6[11][:2], eq_4[0][-2:], eq6[11][-2:], eq_4[1][:2], eq6[12][:2],
                                ),
                  mh.fade_replace(eq_2[0][2], eq6[5][2], coor_mask=RIGHT),
                  mh.fade_replace(eq_2[1][2], eq6[6][2], coor_mask=RIGHT),
                  mh.fade_replace(eq_3[0][2], eq6[8][2], coor_mask=RIGHT),
                  mh.fade_replace(eq_3[1][2], eq6[9][2], coor_mask=RIGHT),
                  mh.fade_replace(eq_4[0][2], eq6[11][2], coor_mask=RIGHT),
                  mh.fade_replace(eq_4[1][2], eq6[12][2], coor_mask=RIGHT),
                                 run_time=2
                  ),
                  Succession(Wait(1), FadeIn(eq6[4::3], eq6[-1], run_time=2))
                  )
        rect1 = SurroundingRectangle(eq6[2:4], stroke_width=6, stroke_color=RED, fill_opacity=0, buff=0.15, corner_radius=0.15).set_z_index(4)
        self.play(FadeIn(rect1))
        self.wait(0.1)
        rect2 = SurroundingRectangle(eq6[5:7], stroke_width=6, stroke_color=RED, fill_opacity=0, buff=0.15, corner_radius=0.15).set_z_index(4)
        self.play(mh.rtransform(rect1, rect2))
        self.wait(0.1)
        rect1 = SurroundingRectangle(eq6[8:10], stroke_width=6, stroke_color=RED, fill_opacity=0, buff=0.15, corner_radius=0.15).set_z_index(4)
        self.play(mh.rtransform(rect2, rect1))
        self.wait(0.1)
        rect2 = SurroundingRectangle(eq6[11:13], stroke_width=6, stroke_color=RED, fill_opacity=0, buff=0.15, corner_radius=0.15).set_z_index(4)
        self.play(mh.rtransform(rect1, rect2))
        self.play(FadeOut(rect2, rate_func=linear))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[:5], eq7[:5], eq6[7::3], eq7[7::3], eq6[-1], eq7[-1],
                                ),
                  *[mh.rtransform(eq6[i][:2], eq7[i][:2], eq6[i][3:], eq7[i][2:4], eq6[i][2], eq7[i][-1], eq6[i+1], eq7[i+1]) for i in (5,8,11)],
                  *[FadeIn(eq7[i][-3].set_color(col_num), eq7[i][-2].set_color(col_op)) for i in (5,8,11)]
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:2], eq8[:2], eq7[4], eq8[3], eq7[-1], eq8[-1],
                                eq7[6][1:], eq8[4][-2:], eq7[5][-3:], eq8[4][3:6], eq7[7], eq8[5],
                                eq7[9][1:], eq8[6][-2:], eq7[8][-3:], eq8[6][3:6], eq7[10], eq8[7],
                                eq7[12][1:], eq8[8][-2:], eq7[11][-3:], eq8[8][3:6], eq7[13], eq8[9],
                                ),
                  mh.stretch_replace(eq7[2][-1], eq8[2][2]),
                  mh.stretch_replace(eq7[5][-4], eq8[4][2]),
                  mh.stretch_replace(eq7[8][-4], eq8[6][2]),
                  mh.stretch_replace(eq7[11][-4], eq8[8][2]),
                  FadeOut(eq7[2][:-1], eq7[3], eq7[5][:-4], eq7[6][0], eq7[8][:-4], eq7[9][0], eq7[11][:-4], eq7[12][0]),
                  FadeIn(eq8[2][:2], eq8[2][-1], eq8[4][:2], eq8[4][6], eq8[6][:2], eq8[6][6], eq8[8][:2], eq8[8][6]),
                  run_time=1.8
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq8[1], eq9[1], eq8[4::2], eq9[4::2]),
                  mh.rtransform(eq8[0], eq9[2], eq8[2], eq9[0], path_arc=PI/3),
                  mh.fade_replace(eq8[3], eq9[3]),
                  mh.fade_replace(eq8[5], eq9[5]),
                  mh.fade_replace(eq8[7], eq9[7]),
                  mh.fade_replace(eq8[9], eq9[9]),
                  run_time=1.5
                  )
        self.wait(0.1)
        eq9_ = eq9.copy()
        mh.copy_colors_eq(eq9[4][3:6], eq10[0][3:6])
        mh.copy_colors_eq(eq9[4][3:6], eq10[2][3:6])
        self.play(mh.rtransform(eq1, gp1[0], eq9, gp1[1], eq9_[0][:3], eq10[0][:3],
                                eq9_[0][-1], eq10[0][-1], eq9_[1], eq10[1],
                                eq9_[2][:3], eq10[2][:3], eq9_[2][-1], eq10[2][-1], eq9_[3], eq10[3],
                                eq9_[4][:5], eq10[4][:5], eq9_[4][-3:], eq10[4][-3:], eq9_[5], eq10[5]
                                ),
                  FadeIn(eq10[0][3:6], shift=mh.diff(eq9[0][2], eq10[0][2])),
                  FadeIn(eq10[2][3:6], shift=mh.diff(eq9[2][2], eq10[2][2])),
                  mh.fade_replace(eq9_[4][5], eq10[4][5].set_color(col_num)),
                  FadeIn(eq10[6], shift=mh.diff(eq9[5], eq10[5]))
                  )
        eq1 = gp1[0]
        self.play(AnimationGroup(mh.rtransform(eq10[0][:], eq11[0][:-2], eq10[1], eq11[1], eq10[2], eq11[2][:-2], eq10[3], eq11[3],
                                eq10[4][:-1], eq11[4][:-1], eq10[5:], eq11[5:]),
                  mh.fade_replace(eq10[4][-1], eq11[4][-1].set_color(col_num)),
                                 run_time=1.5),
                  Succession(Wait(0.7), FadeIn(eq11[0][-2:], eq11[2][-2:])),
                  )
        self.wait(0.1)
        eq9 = gp1[1]
        eq12_1 = eq12[-2].copy()
        self.play(mh.rtransform(eq9[:4], eq12[:4], eq9[4][1:], eq12[4][1:], eq9[5:7], eq12[5:7], eq11[-1], eq12[-1]),
                  mh.rtransform(eq11[2], eq12[4], eq9[-1], eq12[-1]),
                  mh.fade_replace(eq9[-2], eq12[-2]),
                  FadeOut(eq9[4][0], eq9[7:9]),
                  FadeOut(eq11[3:5], target_position=eq9[7:9]),
                  mh.fade_replace(eq11[-2], eq12_1),
                  FadeOut(eq11[:2])
                  )
        self.remove(eq12_1)
        self.wait(0.1)
        eq12_ = eq12.copy()
        mh.copy_colors_eq(eq12[6][3:6], eq13[0][3:6])
        mh.copy_colors_eq(eq12[6][3:6], eq13[2][3:6])
        self.play(mh.rtransform(eq12_[0][:3], eq13[0][:3], eq12_[0][-1], eq13[0][-1], eq12_[1], eq13[1],
                                eq12_[2][:3], eq13[2][:3], eq12_[2][-1], eq13[2][-1], eq12_[3], eq13[3]),
                  FadeIn(eq13[4], shift=mh.diff(eq12[3], eq13[3])),
                  FadeIn(eq13[0][3:6], shift=mh.diff(eq12[0][2], eq13[0][2])),
                  FadeIn(eq13[2][3:6], shift=mh.diff(eq12[2][2], eq13[2][2])),
                  )
        self.wait(0.1)
        eq14_1 = eq14[-2].copy()
        self.play(mh.rtransform(eq12[:6], eq14[:6], eq12[-2:], eq14[-2:], eq12[6][1:], eq14[6][1:]),
                  mh.rtransform(eq13[2][:], eq14[6][:7], eq13[-1], eq14[-1]),
                  FadeOut(eq12[6][0], eq14[:2]),
                  mh.fade_replace(eq13[-2], eq14_1),
                  FadeOut(eq13[:2]),
                  run_time=1.5)
        self.remove(eq14_1)

        self.wait(0.1)
        txt1 = Tex(r'\sf M\"obius function', color=RED, font_size=50)
        txt1.next_to(eq15[2][0], DR, buff=1, aligned_edge=UL)
        arr1 = Arrow(txt1.get_left(), eq15[2][0].get_bottom(), stroke_width=6, color=RED, buff=0.1,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4)

        self.play(FadeIn(eq15, txt1, arr1), mh.rtransform(eq14, eq14_2))
        eq14 = eq14_2
        self.wait(0.1)
        self.play(FadeOut(eq15, txt1, arr1))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:2], eq16[:2]),
                  eq14.animate.set_opacity(0.4).move_to(eq14_3),
                  FadeOut(eq1[2:]),
                  FadeIn(eq16[2:])
                  )
        self.play(FadeIn(eq17))
        self.wait(0.1)
        self.play(FadeOut(eq17))
        self.wait(0.1)
        mh.copy_colors_eq(eq18[2][:3], eq18[4][:3])
        mh.copy_colors_eq(eq18[2][3:-2], eq18[5][:])
        self.play(AnimationGroup(mh.fade_replace(eq16[0][0], eq18[0][0]),
                  mh.rtransform(eq16[0][1:], eq18[0][1:], eq16[1], eq18[1], eq16[3], eq18[3],
                                eq16[4][:2], eq18[-1][2:4], eq16[-1][:2], eq18[-1][5:7],
                                eq16[-1][3], eq18[-1][8]),
                  mh.fade_replace(eq16[2], eq18[2][-1]),
                  mh.fade_replace(eq16[-1][2], eq18[-1][7].set_color(col_x)),
                  mh.fade_replace(eq16[4][2], eq18[-1][4].set_color(col_x)),
                  run_time=1.5),
                  Succession(Wait(0.7), FadeIn(eq18[2][:-1], eq18[4], eq18[5], eq18[6][:2], eq18[6][-1]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq18[0], eq19[0], eq18[1], eq19[3], eq18[4:], eq19[4:]),
                  FadeOut(eq18[2][0], eq18[2][2:]),
                  mh.stretch_replace(eq18[2][1], eq19[2][3]),
                  FadeIn(eq19[2][:3], eq19[2][4]),
                  mh.fade_replace(eq18[3], eq19[1])
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq19[:4], eq20[:4], eq19[4], eq20[7],
                                eq19[5][1:-1], eq20[4][3:-1], eq19[6][2:4], eq20[4][:2],
                                eq19[6][5:7], eq20[5][:2], eq19[6][8], eq20[5][3]),
                  mh.rtransform(eq19[6][0], eq20[-1][0], eq19[6][2:-1].copy(), eq20[-2][:],
                                eq19[5][1].copy(), eq20[-1][2], eq19[5][2:5].copy(), eq20[-1][4:7],
                                eq19[5][5].copy(), eq20[-1][8]),
                  mh.fade_replace(eq19[5][-1], eq20[4][-1].set_color(col_x)),
                  mh.fade_replace(eq19[6][4], eq20[4][2].set_color(col_x)),
                  mh.fade_replace(eq19[6][7], eq20[5][2].set_color(col_x)),
                  FadeOut(eq19[6][1], eq19[6][-1], shift=mh.diff(eq19[6][2:4], eq20[-2])),
                  FadeOut(eq19[5][0], target_position=eq20[-1][0]),
                  FadeIn(eq20[-1][1].set_color(col_x), shift=mh.diff(eq19[6][0], eq20[-1][0])),
                  FadeIn(eq20[-1][3].set_color(col_x), eq20[-1][7].set_color(col_trig), shift=mh.diff(eq19[5][2:5].copy(), eq20[-1][4:7])),
                  FadeIn(eq20[6], shift=mh.diff(eq19[4], eq20[7])),
                  run_time=2
                  )
        self.wait(0.1)
        circ = mh.circle_eq(eq20[-3:]).set_z_index(10)
        txt2 = MathTex(r'\sf small', color=RED, font_size=60)
        txt2.next_to(circ, UR, buff=-0.3)
        self.play(Create(circ, rate_func=linear, run_time=0.8), FadeIn(txt2))
        self.wait(0.1)
        self.play(FadeOut(circ, txt2, eq20[-4:]))
        self.play(mh.rtransform(eq20[:6], eq21[:]))
        self.wait(0.1)

        mh.align_sub(eq22[:9], eq22[1], eq14[1]).move_to(eq14, coor_mask=RIGHT)
        self.play(mh.rtransform(eq14[:2], eq22[:2], eq14[2][1:], eq22[2][2:], eq14[3], eq22[3],
                                eq14[4][1:], eq22[4][2:], eq14[5], eq22[5], eq14[6][1:], eq22[6][2:],
                                eq14[7:], eq22[7:9]),
                  mh.fade_replace(eq14[2][0], eq22[2][:2]),
                  mh.fade_replace(eq14[4][0], eq22[4][:2]),
                  mh.fade_replace(eq14[6][0], eq22[6][:2]),
                  )
        eq22[9:].next_to(eq22[:9], DOWN, buff=0.2)
        eq22_1 = eq22.copy().to_edge(DOWN, buff=0.8)
        self.play(AnimationGroup(mh.rtransform(eq22[:9], eq22_1[:9], eq21[4:].copy(), eq22_1[10:12]),
                                 eq21.animate.next_to(eq22_1, UP, buff=0.4, coor_mask=UP),
                                 run_time=1.6),
                  Succession(Wait(0.8), FadeIn(eq22_1[9], eq22_1[-2:])))
        self.wait(0.1)
        eq23[:9].move_to(eq22_1[:9])
        eq23[9:].move_to(eq22_1[9:-2]).shift(RIGHT*0.8)
        self.play(mh.rtransform(eq22_1[:-2], eq23[:]),
                  FadeOut(eq22_1[-2:]))
        self.wait(0.1)

        eq21_1 = eq21.copy().to_edge(UP, buff=0.4)
        eq23_1 = eq23.copy().move_to((eq21_1.get_bottom()+eq24.get_top())/2, coor_mask=UP)
        circ = mh.circle_eq(eq24[-1], scale=0.7).set_z_index(10)
        txt3 = Tex(r'\sf arcsine', color=RED).next_to(circ, UR, buff=-0.1)
        self.play(eq21.animate.move_to(eq21_1),
                  eq23.animate.move_to(eq23_1),
                  Succession(Wait(0.6), FadeIn(eq24, circ, txt3)))

        self.wait()

class DistnFinal(RiemannJ):
    bgcolor = GREY
    def construct(self):
        eq1 = self.do_anim(just_eq=True).set_z_index(4)

        eq2 = MathTex(r'{\sf Cramer\!:\ }', r'{\sf error\ }', r'\approx', r'\sqrt{{\rm Li}(x)}').set_z_index(4)
        eq3 = MathTex(r'{\sf Cramer\!:\ }', r'{\sf error\ }', r'\approx', r'\sqrt{\frac x{\log x} }').set_z_index(4)
        eq4 = MathTex(r'{\sf equidistribution\!:\ }', r'{\sf error\ }', r'\approx', r'\frac{\sqrt x}{\log x}', r'\mathcal E(x)').set_z_index(4)
        eq5 = MathTex(r'{\rm Var}\left[\mathcal E(x)\right]', r'\sim', r'\sum_{\Re[\rho] > 0}',
                      r'\frac{4}{\lvert\rho\rvert^2}', r'{\rm Var}\left[\cos U_\rho\right]').set_z_index(4)
        eq6 = MathTex(r'{\rm Var}\left[\mathcal E(x)\right]', r'\sim', r'\sum_{\Re[\rho] > 0}',
                      r'\frac{4}{\lvert\rho\rvert^2}', r'\frac12').set_z_index(4)
        eq7 = MathTex(r'{\rm Var}\left[\mathcal E(x)\right]', r'\sim', r'\sum_{\Re[\rho] > 0}',
                      r'\frac{2}{\lvert\rho\rvert^2}').set_z_index(4)
        eq8 = MathTex(r'{\rm Var}\left[\mathcal E(x)\right]', r'\sim', r'\sum_\rho',
                      r'\frac{1}{\lvert\rho\rvert^2}').set_z_index(4)
        eq9 = MathTex(r'{\rm Var}\left[\mathcal E(x)\right]', r'\sim',
                      r'2+\gamma-\log4\pi').set_z_index(4)
        eq10 = MathTex(r'{\rm Var}\left[\mathcal E(x)\right]', r'\sim',
                      r'0.046185\ldots').set_z_index(4)

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq2[:2], eq4[:2]).set_color(col_txt)
        VGroup(eq2[3][2:4], eq4[4][0]).set_color(col_WVD)
        VGroup(eq2[3][5], eq4[4][2]).set_color(col_x)
        VGroup(eq2[3][:2], eq6[-1][1]).set_color(col_op)
        VGroup(eq3[3][4:7], eq9[2][4:7]).set_color(col_trig)
        VGroup(eq5[3][-1], eq6[-1][::2], eq9[2][0]).set_color(col_num)
        VGroup(eq5[0][:3], eq5[-1][:3]).set_color(col_txt2)
        VGroup(eq9[2][2]).set_color(col_special)
        VGroup(eq9[2][-2:]).set_color(col_pi)
        mh.copy_colors_eq(eq1[2], eq5[2], eq1[3][:], eq5[3][:-1], eq1[4][:], eq5[4][4:-1])

        eq2.move_to(eq1, coor_mask=UP)
        mh.align_sub(eq3, eq3[2], eq2[2], coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq3[1]).next_to(eq3, UP, buff=0.1, coor_mask=UP)
        eq5.to_edge(DOWN, buff=0.4)
        mh.align_sub(eq6, eq6[1], eq5[1])
        mh.align_sub(eq7, eq7[1], eq6[1], coor_mask=UP)
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=UP)
        mh.align_sub(eq9, eq9[1], eq8[1], coor_mask=UP)
        mh.align_sub(eq10, eq10[1], eq9[1], coor_mask=UP)

        box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.15, buff=0.2)
        box2 = SurroundingRectangle(VGroup(eq1, eq2), stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.15, buff=0.2)
        box3 = SurroundingRectangle(VGroup(eq3, eq4), stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.15, buff=0.2)
        box4 = SurroundingRectangle(eq5, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.15, buff=0.2)
        self.add(eq1, box1)
        self.play(FadeOut(eq1), FadeIn(eq2), mh.rtransform(box1, box2))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:3], eq3[:3], eq2[3][5], eq3[3][7], eq2[3][5].copy(), eq3[3][2]),
                  mh.stretch_replace(eq2[3][:2], eq3[3][:2]),
                  mh.fade_replace(eq2[3][2:4], eq3[3][4:7]),
                  FadeOut(eq2[3][4::2], shift=mh.diff(eq2[3][5], eq3[3][7])),
                  Succession(Wait(0.4), FadeIn(eq3[3][3].set_color(col_op))))
        self.wait(0.1)
        self.play(mh.rtransform(box2, box3),
                  AnimationGroup(Wait(0.4),
            mh.rtransform(eq3[1:3].copy(), eq4[1:3], eq3[0][-1].copy(), eq4[0][-1],
                                eq3[3][2:].copy(), eq4[3][2:]),
                  FadeIn(eq4[0][:-1], shift=mh.diff(eq3[0][-1], eq4[0][-1])),
                  mh.stretch_replace(eq3[3][0].copy(), eq4[3][0]),
                  mh.stretch_replace(eq3[3][1].copy(), eq4[3][1])),
                  Succession(Wait(0.6), FadeIn(eq4[4]))
                  )
        self.wait(0.1)
        self.play(FadeOut(eq3, eq4[:-1], box3),
                  mh.rtransform(eq4[-1][:], eq5[0][4:-1], run_time=1.5),
                  Succession(Wait(1), FadeIn(eq5[0][:4], eq5[0][-1], eq5[1:], box4)))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:-1], eq6[:-1]),
                  FadeOut(eq5[-1]), FadeIn(eq6[-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[:-2], eq7[:-1], eq6[-2][1:], eq7[-1][1:]),
                  mh.fade_replace(eq6[-2][0], eq7[-1][0].set_color(col_num)),
                  FadeOut(eq6[-1], target_position=eq7[-1][0]))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:2], eq8[:2], eq7[-1][1:], eq8[-1][1:], eq7[2][0], eq8[2][0], eq7[2][3], eq8[2][1]),
                  mh.fade_replace(eq7[-1][0], eq8[-1][0].set_color(col_num)),
                  FadeOut(eq7[2][1:3], eq7[2][4:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq8[:2], eq9[:2]),
                  FadeOut(eq8[2:]), FadeIn(eq9[2]))

        txt1 = Tex(r'\sf Euler-Mascheroni constant $\approx0.577$', color=RED, font_size=50).set_z_index(5)
        txt1 = mh.eq_shadow(txt1, bg_stroke_width=12)
        txt1.next_to(eq9[2], UP, buff=0.6)
        arr1 = Arrow(txt1[0][12].get_bottom(), eq9[2][2].get_top(), buff=0.1, color=RED,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.4).set_z_index(5)
        self.play(FadeIn(txt1, arr1))
        self.wait(0.1)
        self.play(mh.rtransform(eq9[:2], eq10[:2]),
                  FadeOut(eq9[2]),
                  FadeIn(eq10[2].set_color(col_num)),
                  FadeOut(arr1, txt1))

        self.wait()


if __name__ == "__main__":
    def li(x):
        return expi(np.log(x)) - expi(np.log(2))

    def f(x):
        return 1 - (li(np.sqrt(x))/2 + li(np.pow(x, 1/3))/3)*np.log(x)/np.sqrt(x)
        # return 1 - (li(np.sqrt(x))/2)*np.log(x)/np.sqrt(x)
        # return (li(np.pow(x, 1 / 3)) / 3) * np.log(x) / np.sqrt(x)

    print(f(10_000_000_000))
