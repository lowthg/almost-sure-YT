from manim import *
import sys
from manim import ManimColor
from scipy.special import expi
from common_randomprime import *

sys.path.append('../../')
import manimhelper as mh
from common.wigner import *

col_pi = col_special * 0.5 + ORANGE * 0.5
col_trig = PURPLE_A#*0.5+WHITE*0.5
col_txt = ManimColor( r'#FFAC2B')
col_prime = BLUE

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