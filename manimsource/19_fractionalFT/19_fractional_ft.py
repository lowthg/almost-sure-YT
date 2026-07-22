import math

import numpy as np
from manim import *
import sys
import scipy as sp
from manim import ManimColor

sys.path.append('../../')
import manimhelper as mh
from common.wigner import *

col_pi = col_special * 0.5 + ORANGE * 0.5
col_trig = PURPLE_A#*0.5+WHITE*0.5
col_laser = ManimColor(r'#ED2F32')
col_txt = ManimColor( r'#FFAC2B')

def eq_shadow(eq: VGroup, fg_z_index=4., bg_z_index=0., bg_color=BLACK, bg_stroke_width=10.):
    res = VGroup()
    for eq1 in eq:
        elem = VGroup()
        for eq2 in eq1:
            elem.add(VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res

class FractionalFTEq(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=65)
        eq1 = MathTex(r'\mathcal F_\theta f(x)', r'=', r'\sqrt{\frac{1-i\cot\theta}{2\pi}}', r'\int',
                      r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}f(y)\,dy')
        eq1[2:].next_to(eq1[:2], DOWN, buff=0.5)
        eq1[:2].align_to(eq1[2], LEFT).shift(RIGHT*2)
        eq1.move_to(ORIGIN)

        self.add(eq1)


class FourierTfmEq(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=70)
        eq1 = MathTex(r'g(\nu)', r'=', r'\int', r'e^{-2\pi i\nu t}', r'f(t)', r'\,dt')
        eq2 = MathTex(r'f(t)', r'=', r'\int', r' e^{2\pi i\nu t}', r'g(\nu)', r'\,d\nu')
        eq3 = MathTex(r'x=t', r'y=2\pi\nu')
        eq4 = MathTex(r'g(', r'\frac{y}{2\pi}', r')', r'=', r'\int', r'e^{-ixy}', r'f(x)', r'\,dx')
        mh.font_size_sub(eq4, 1, 50)
        eq5 = MathTex(r'f(x)', r'=', r'\int', r' e^{ixy}', r'g(', r'\frac{y}{2\pi}', r')', r'\,\frac{dy}{2\pi}')
        mh.font_size_sub(eq5, 5, 50)
        eq6 = MathTex(r'=', r'y')
        eq7 = MathTex(r'f(x)', r'=', r'\frac1{2\pi}', r'\int', r'e^{ixy}', r'g(y)', r'\,dy')
        eq8 = MathTex(r'\sqrt{2\pi}', r'g(y)', r'=', r'\int', r'e^{-ixy}', r'f(x)', r'\,dx')
        eq9 = MathTex(r'f(x)', r'=', r'\frac1{2\pi}', r'\int', r'e^{ixy}', r'\sqrt{2\pi}', r'g(y)', r'\,dy')
        eq10 = MathTex(r'g(y)', r'=', r'\frac1{\sqrt{2\pi}}', r'\int', r'e^{-ixy}', r'f(x)', r'\,dx')
        eq11 = MathTex(r'f(x)', r'=', r'\frac1{\sqrt{2\pi}}', r'\int', r'e^{ixy}', r'g(y)', r'\,dy')
        eq12 = MathTex(r'\mathcal F f(y)', r'=', r'\frac1{\sqrt{2\pi}}', r'\int', r'e^{-ixy}', r'f(x)', r'\,dx')
        eq13 = MathTex(r'\mathcal F^{-1} g(x)', r'=', r'\frac1{\sqrt{2\pi}}', r'\int', r'e^{ixy}', r'g(y)', r'\,dy')
        eq14 = MathTex(r'\mathcal F f(x)', r'=', r'\frac1{\sqrt{2\pi}}', r'\int', r'e^{-ixy}', r'f(y)', r'\,dy')
        eq15 = MathTex(r'\mathcal F_\theta f(x)', r'=', r'\sqrt{\frac{1-i\cot\theta}{2\pi}}', r'\int',
                      r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}', r'f(y)', r'\,dy')
        mh.font_size_sub(eq15, 2, 57)
        eq16 = MathTex(r'\cot\theta=\frac{\cos\theta}{\sin\theta}', r'{}\csc\theta=\frac1{\sin\theta}', font_size=60)
        eq17 = MathTex(r'\mathcal F_{\frac\pi2}f(x)', r'=')
        eq18 = MathTex(r'\sqrt{\frac{0}{2\pi}}', font_size=57)
        eq19 = MathTex(r'e^{+01}')
        eq20 = MathTex(r'\mathcal F_{-\frac\pi2}f(x)', r'=')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][0], eq1[4][0], eq12[0][-4], eq13[0][-4]).set_color(col_psi)
        VGroup(eq1[4][2], eq1[5][1], eq1[3][-1], eq3[0][0], eq3[0][2],
               eq4[5][3], eq5[3][2], eq5[0][2], eq14[0][-2], eq15[4][6:8]).set_color(col_x)
        VGroup(eq1[0][2], eq2[5][1], eq1[3][-2], eq3[1][0], eq3[1][4],
               eq6[1], eq5[5][0], eq5[7][1], eq14[5][2], eq14[6][1], eq15[4][9:11]).set_color(col_p)
        VGroup(eq1[3][4], eq15[2][4], eq15[4][4]).set_color(col_i)
        VGroup(eq1[3][0], eq2[3][0]).set_color(col_special)
        VGroup(eq1[3][2:4], eq3[1][2:4], eq8[0][-2:]).set_color(col_pi)
        VGroup(eq1[2], eq2[2], eq1[5][0], eq2[5][0], eq8[0][:-2],
               eq4[1][1], eq5[5][1], eq5[-1][2], eq10[2][1], eq15[4][2],
               eq16[0][9], eq16[1][6]).set_color(col_op)
        VGroup(eq7[2][0], eq10[2][0], eq15[4][1], eq15[4][3],
               eq16[1][5], eq18[0][2], eq19[0][2:]).set_color(col_num)
        VGroup(eq12[0][0], eq13[0][:3]).set_color(col_ft)
        VGroup(eq15[0][1], eq15[2][8], eq15[4][15], eq15[4][-1],
               eq16[0][3], eq16[0][8], eq16[0][13], eq16[1][3], eq16[1][10], eq17[0][1:4], eq20[0][1]).set_color(col_angle)
        VGroup(eq15[2][5:8], eq15[4][12:15], eq15[4][20:23],
               eq16[0][:3], eq16[0][5:8], eq16[0][10:13], eq16[1][:3], eq16[1][7:10]).set_color(col_trig)
        mh.copy_colors_eq(eq1[0], eq2[4], eq1[4], eq2[0], eq1[3][2:], eq2[3][1:])
        mh.copy_colors_eq(eq8[0], eq9[5])

        mh.align_sub(eq2, eq2[1], eq1[1]).next_to(eq1, DOWN, buff=0.6, coor_mask=UP)
        VGroup(eq1, eq2).move_to(ORIGIN)
        eq3[1].shift(RIGHT)
        eq3.move_to(ORIGIN).next_to(eq1, UP, buff=0.8)
        mh.align_sub(eq4, eq4[3], eq1[1])
        mh.align_sub(eq5, eq5[1], eq2[1])
        mh.align_sub(eq7, eq7[1], eq5[1])
        mh.align_sub(eq8, eq8[2], eq1[1])
        mh.align_sub(eq9, eq9[1], eq2[1])
        mh.align_sub(eq10, eq10[1], eq1[1])
        mh.align_sub(eq11, eq11[1], eq2[1])
        mh.align_sub(eq12, eq12[1], eq10[1])
        mh.align_sub(eq13, eq13[1], eq11[1])
        VGroup(eq12, eq13).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq14, eq14[1], eq12[1])
        eq15[2:].next_to(eq15[:2], DOWN, buff=0.8)
        eq15[:2].align_to(eq15[2], LEFT).shift(RIGHT*2)
        eq15.move_to(UP*0.4)
        eq16[1].shift(RIGHT)
        # eq16.move_to(ORIGIN).to_edge(DOWN, buff=0.2)
        eq16.move_to((mh.pos(DOWN) + eq15.get_bottom())*0.5*UP)
        mh.align_sub(eq17, eq17[1], eq15[1])
        mh.align_sub(eq18, eq18[0][-3], eq15[2][-3])
        mh.align_sub(eq19, eq19[0][0], eq15[4][0])
        mh.align_sub(eq20, eq20[1], eq15[1])


        self.add(eq1, eq2)
        self.play(FadeIn(eq3))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[1], eq4[3], eq1[2], eq4[4], eq1[3][:2], eq4[5][:2],
                                ),
                  mh.rtransform(eq3[1][0].copy(), eq4[5][4], eq3[1][0].copy(), eq4[1][0],
                                eq3[1][2:4].copy(), eq4[1][2:4], run_time=1.6),
                  Succession(Wait(0.6), AnimationGroup(
                      mh.rtransform(eq1[3][4], eq4[5][2], eq1[0][:2], eq4[0][:2], eq1[0][-1], eq4[2][-1]),
                      FadeOut(eq1[3][2:4], eq1[3][5], eq1[0][2]),
                  FadeIn(eq4[1][1]))
                             ))
        self.play(mh.rtransform(eq3[0][0].copy(), eq4[6][2], eq1[4][:2], eq4[6][:2],
                                eq1[4][-1], eq4[6][-1], eq1[5][0], eq4[7][0],
                                eq3[0][0].copy(), eq4[7][1], run_time=1.6),
                  mh.stretch_replace(eq3[0][0].copy(), eq4[5][3], run_time=1.6),
                  FadeOut(eq1[3][6], shift=mh.diff(eq1[3][6], eq4[5][3]) * RIGHT, run_time=1.6),
                  FadeOut(eq1[4][2], shift=mh.diff(eq1[4][2], eq4[6][2])*RIGHT, run_time=1.6),
                  FadeOut(eq1[5][1], shift=mh.diff(eq1[5][1], eq4[7][1]) * RIGHT, run_time=1.6),
        )
        self.wait(0.1)
        eq5_1 = eq5[0][2].copy()
        eq5_2 = eq5[3][3].copy()
        eq5_3 = eq5[3][2].copy()
        eq5_4 = eq5[5][0].copy()
        eq5_5 = eq5[7][1].copy()
        self.play(mh.rtransform(eq2[0][:2], eq5[0][:2], eq2[0][-1], eq5[0][-1],
                                eq3[0][0], eq5[0][2],
                                eq2[1:3], eq5[1:3], eq2[3][0], eq5[3][0], eq2[3][3], eq5[3][1],
                                eq3[1][0].copy(), eq5[3][3], eq2[4][:2], eq5[4][:2],
                                eq3[1][0], eq5[5][0], eq3[1][2:4].copy(), eq5[5][2:4],
                                eq2[4][-1], eq5[6][0],
                                eq2[5][0], eq5[7][0], eq3[1][0].copy(), eq5[7][1],
                                eq3[1][2:4], eq5[7][3:5]),
                  mh.fade_replace(eq2[0][2], eq5_1),
                  mh.fade_replace(eq2[3][4], eq5_2),
                  mh.fade_replace(eq2[3][5], eq5_3),
                  mh.fade_replace(eq2[4][2], eq5_4),
                  mh.fade_replace(eq2[5][1], eq5_5),
                  mh.stretch_replace(eq3[0][0].copy(), eq5[3][2]),
                  FadeOut(eq2[3][1:3]),
                  FadeIn(eq5[5][1], eq5[7][2]),
                  FadeOut(eq3[0][1:], eq3[1][1], eq3[1][-1]),
                  run_time=1.6)
        self.remove(eq5_1, eq5_2, eq5_3, eq5_4, eq5_5)
        self.wait(0.1)
        eq6_1 = mh.align_sub(eq6, eq6[0], eq5[1])[1].copy().move_to(eq5[5], coor_mask=RIGHT)
        eq6_2 = mh.align_sub(eq6, eq6[0], eq4[3])[1].copy().move_to(eq4[1], coor_mask=RIGHT)
        self.play(FadeOut(eq5[5], eq4[1]), FadeIn(eq6_1, eq6_2))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:2], eq7[:2], eq5[2:4], eq7[3:5], eq5[4][:], eq7[5][:2], eq6_1, eq7[5][2],
                                eq5[6][0], eq7[5][3], eq5[7][:2], eq7[6][:], eq5[7][2:], eq7[2][1:]),
                  FadeIn(eq7[2][0], shift=mh.diff(eq5[7][2], eq7[2][1])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][:2], eq8[1][:2], eq6_2, eq8[1][2], eq4[2][0], eq8[1][3],
                                eq4[3:], eq8[2:], eq7[:5], eq9[:5], eq7[5:], eq9[6:], run_time=1.4),
                  Succession(Wait(0.8), FadeIn(eq8[0], eq9[5])))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq8[0][:], eq10[2][2:], eq8[1:3], eq10[:2], eq8[3:], eq10[3:],
                                eq9[:2], eq11[:2], eq9[3:5], eq11[3:5],
                                eq9[5][:], eq11[2][2:], eq9[2][:2], eq11[2][:2], eq9[6:], eq11[5:]),
                  mh.rtransform(eq9[2][2:], eq11[2][-2:]), run_time=1.8),
                  Succession(Wait(1.), FadeIn(eq10[2][:2]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq10[1:], eq12[1:], eq11[1:], eq13[1:], eq10[0][-3:], eq12[0][-3:], eq11[0][-3:], eq13[0][-3:]),
                  FadeOut(eq10[0][-4], eq11[0][-4]),
                  FadeIn(eq12[0][:-3], eq13[0][:-3]))
        self.wait(0.1)
        self.play(mh.rtransform(eq12[1:-2], eq14[1:-2], eq12[0][:3], eq14[0][:3], eq12[0][4:], eq14[0][4:],
                                eq12[-2][:2], eq14[-2][:2], eq12[-2][3], eq14[-2][3], eq12[-1][0], eq14[-1][0]),
                  mh.fade_replace(eq12[0][3], eq14[0][3], coor_mask=RIGHT),
                  mh.fade_replace(eq12[-2][2], eq14[-2][2], coor_mask=RIGHT),
                  mh.fade_replace(eq12[-1][1], eq14[-1][1], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        self.play(FadeOut(eq13))
        self.wait(0.1)
        eq15_3 = mh.align_sub(eq15[0].copy(), eq15[1], eq14[1])
        circ1 = mh.circle_eq(eq15_3[1], scale=0.5)
        self.play(Succession(Wait(0.7), Create(circ1, rate_func=linear, run_time=0.5)),
                  mh.rtransform(eq14[0][0], eq15_3[0], eq14[0][1:], eq15_3[2:]),
                  FadeIn(eq15_3[1], shift=mh.diff(eq14[0][0], eq15_3[0])),
                  )
        self.wait(0.1)
        eq15_1 = eq15[2][2].copy().move_to(eq15[2][-3], coor_mask=RIGHT)
        self.play(mh.rtransform(eq14[1], eq15[1], eq15_3, eq15[0]),
                  mh.rtransform(eq14[2][2:4], eq15[2][:2], eq14[2][-2:], eq15[2][-2:], eq14[2][1], eq15[2][-3]),
                  mh.rtransform(eq14[3], eq15[3], eq14[4][0], eq15[4][0], eq14[4][-4:], eq15[4][-8:-4],
                                eq14[5:], eq15[5:], eq14[2][0], eq15_1),
                  FadeOut(circ1, shift=mh.diff(eq15_3, eq15[0])),
                  run_time=2)
        self.wait(0.1)
        self.play(mh.rtransform(eq15_1, eq15[2][2]),
                  Succession(Wait(0.4), AnimationGroup(FadeIn(eq15[2][3:-3]), FadeIn(eq15[4][1:-8], eq15[4][-4:]))))
        self.wait(0.1)
        self.play(FadeIn(eq16))
        self.wait(0.1)
        circ1 = mh.circle_eq(eq17[0][1:-4], scale=0.5).set_z_index(10).shift(RIGHT*0.1)
        self.play(Create(circ1, run_time=0.5, rate_func=linear))
        self.wait(0.1)
        eq15_2 = eq15.copy()
        self.play(mh.rtransform(eq15[0][0], eq17[0][0], eq15[0][-4:], eq17[0][-4:]),
                  mh.fade_replace(eq15[0][1:-4], eq17[0][1:-4], coor_mask=RIGHT)
                  )
        self.wait(0.1)
        self.play(FadeOut(circ1), run_time=0.8)
        self.wait(0.1)
        eq18[0][-4].move_to(eq15[2][-7:-4], coor_mask=RIGHT)
        eq19[0][2].move_to(eq15[4][12:15], coor_mask=RIGHT)
        eq19[0][3].move_to(eq15[4][20:23], coor_mask=RIGHT)
        self.play(VGroup(eq15[2][-7:-3], eq15[4][12:16]).animate.set_opacity(0),
                  FadeIn(eq18[0][-4], eq19[0][2]))
        self.wait(0.1)
        self.play(FadeOut(eq15[4][20:24]), FadeIn(eq19[0][3]))
        self.wait(0.1)
        gp1 = eq15[2][3:5] + eq15[4][1:12]
        gp2 = VGroup(eq15[2], eq15[3], eq15[4][0])
        gp2_ = gp2.copy().next_to(eq15[4][16], LEFT, buff=0.2, coor_mask=RIGHT)
        gp2_[0][3:5].set_opacity(0)
        gp2_[0][2].move_to(gp2_[0][-3], coor_mask=RIGHT)
        self.play(FadeOut(eq18[0][-4], eq19[0][2:4]), gp1.animate.set_opacity(0))
        self.play(mh.rtransform(gp2, gp2_),
                  eq15[5:].animate.next_to(eq15[4][19], RIGHT, buff=0.2, coor_mask=RIGHT), run_time=1.2)
        self.wait(0.1)
        circ1 = mh.circle_eq(eq20[0][1:-4], scale=0.5).set_z_index(10).shift(RIGHT*0.2)
        self.play(Create(circ1, run_time=0.5, rate_func=linear))
        self.wait(0.1)
        self.play(mh.rtransform(eq17[0][0], eq20[0][0], eq17[0][1:], eq20[0][2:]),
                  FadeIn(eq20[0][1]))
        self.wait(0.1)
        self.play(FadeOut(circ1), run_time=0.8)
        eq19[0][1].move_to(eq15[4][16], coor_mask=RIGHT)
        circ1 = mh.circle_eq(eq19[0][1], scale=0.5).set_z_index(10)
        self.play(Create(circ1, run_time=0.5, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(eq15[4][16]), FadeIn(eq19[0][1]))
        self.wait(0.1)
        self.play(FadeOut(circ1))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq20[0][0], eq15_2[0][0], eq20[0][-4:], eq15_2[0][-4:]),
                  mh.fade_replace(eq20[0][2:-4], eq15_2[0][1], coor_mask=RIGHT),
                  FadeOut(eq20[0][1], shift=mh.diff(eq20[0][2:-4], eq15_2[0][1])*RIGHT),
                  mh.rtransform(eq15[1], eq15_2[1], gp2_[:2], eq15_2[2:4], gp2_[2], eq15_2[4][0],
                                eq15[4][17:20], eq15_2[4][17:20], eq15[5:], eq15_2[5:]),
                  mh.fade_replace(eq19[0][1], eq15_2[4][16]),
                                 run_time=1.8),
                  Succession(Wait(1), FadeIn(eq15_2[4][1:16], eq15_2[4][20:]))
                  )
        eq15 = eq15_2
        self.wait(0.1)

        eq21 = MathTex(r'\cot0=\frac{\cos 0}{\sin 0}', r'{}\csc0=\frac1{\sin 0}', font_size=60)
        eq22 = MathTex(r'\cot0=\frac{1}{0}', r'{}\csc0=\frac1{0}', font_size=60)
        eq23 = MathTex(r'\cot\pi=\frac{-1}{0}', r'{}\csc\pi=\frac1{0}', font_size=60)
        eq24 = MathTex(r'\mathcal F_0', r'f(x)', r'=', r'f(x)', r'\mathcal F_\pi', r'f(x)', r'=', r'f(-x)')
        eq25 = MathTex(r'\mathcal F_\theta', r'\mathcal F_\phi', r'=', r'\mathcal F_{\theta+\phi}')
        eq26 = MathTex(r'\mathcal F_{r\frac\pi2}', r'=', r'\mathcal F^r')
        eq27 = MathTex(r'\mathcal F^r', r'\mathcal F^s', r'=', r'\mathcal F^{r+s}')
        eq28 = MathTex(r'\mathcal F_{\frac\pi4}', r'=', r'\mathcal F^{\frac12}')
        eq29 = MathTex(r'\mathcal F_{\frac\pi4}', r'=', r'\sqrt{\mathcal F}')

        eq24[4:].shift(RIGHT)

        VGroup(eq21, eq22, eq23[0][5]).set_color(col_num)
        VGroup(eq23[0][3], eq23[1][3], eq26[0][1:], eq26[2][-1],
               eq28[0][-1], eq28[2][-3:]).set_color(col_angle)
        VGroup(eq26[0][0], eq26[2][0]).set_color(col_ft)
        VGroup(eq29[2][:-1]).set_color(col_op)
        mh.copy_colors_eq(eq15[0][:2], eq24[0][:], eq15[0][2:], eq24[1][:])
        mh.copy_colors_eq(eq24[1], eq24[3], eq24[0], eq24[4], eq24[1], eq24[5])
        mh.copy_colors_eq(eq24[1][:3], eq24[7][:3], eq24[1][-2:], eq24[7][-2:])
        mh.copy_colors_eq(eq24[0], eq25[0], eq24[0], eq25[1], eq24[0][:], eq25[3][:2], eq24[0][-1], eq25[3][-1])
        mh.copy_colors_eq(eq25, eq27)
        # mh.copy_colors_eq(eq16, eq21)

        mh.align_sub(eq21[0], eq21[0][4], eq16[0][4])
        mh.align_sub(eq21[1], eq21[1][4], eq16[1][4])
        mh.align_sub(eq22[0], eq22[0][4], eq16[0][4])
        mh.align_sub(eq22[1], eq22[1][4], eq16[1][4])
        eq22[0][5:].move_to(eq21[0][5:], coor_mask=RIGHT)
        eq22[1][5:].move_to(eq21[1][5:], coor_mask=RIGHT)
        mh.align_sub(eq23[0], eq23[0][4], eq16[0][4])
        mh.align_sub(eq23[1], eq23[1][4], eq16[1][4])
        eq23[0][5:7].move_to(eq22[0][5], coor_mask=RIGHT)
        eq24.move_to(eq16).move_to(ORIGIN, coor_mask=RIGHT)
        eq25.move_to(eq16).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq26, eq26[1], eq25[2]).next_to(eq25, LEFT, coor_mask=RIGHT, buff=1)
        gp3 = VGroup(eq26, eq25.copy()).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq27, eq27[2], gp3[1][2])
        mh.align_sub(eq28, eq28[1], eq26[1])
        mh.align_sub(eq28[0], eq28[0][0], eq26[0][0], coor_mask=RIGHT)
        mh.align_sub(eq29, eq29[1], eq28[1])
        mh.align_sub(eq29, eq29[2][-1], eq28[2][0], coor_mask=RIGHT)

        # self.play(mh.rtransform(eq16[0][:3], eq21[0][:3], eq16[0][4:8], eq21[0][4:8], eq16[0][94:14], eq21[0][9:14]),
        #           mh.fade_replace(eq16[0][3], eq21[0][3]),
        #           mh.fade_replace(eq16[0][8], eq21[0][8]),
        #           # mh.fade_replace(eq16[0][14], eq21[0][14]),
        #           )
        self.play(mh.rtransform(eq16, eq21))
        self.wait(0.1)
        self.play(FadeOut(eq21[0][5:9], eq21[0][10:14], eq21[1][7:], eq21[1][7:]),
                  FadeIn(eq22[0][5], eq22[0][7], eq22[1][7:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq21[0][:3], eq23[0][:3], eq21[0][4], eq23[0][4], eq22[0][5], eq23[0][6],
                                eq21[1][:3], eq23[1][:3], eq21[1][4], eq23[1][4]),
                  FadeIn(eq23[0][5], shift=mh.diff(eq22[0][5], eq23[0][6])),
                  mh.fade_replace(eq21[0][3], eq23[0][3], coor_mask=RIGHT),
                  mh.fade_replace(eq21[1][3], eq23[1][3], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        line1 = Line(eq16.get_corner(DL), eq16.get_corner(UR), stroke_width=8, stroke_color=RED).set_z_index(10)
        self.play(Create(line1, run_time=0.8, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(line1, eq23[0][:7], eq23[1][:5], eq22[0][-1], eq22[1][-1], eq21[1][5:7], eq21[0][9]))
        self.wait(0.1)
        self.play(FadeIn(eq24[:4]))
        self.wait(0.1)
        self.play(FadeIn(eq24[4:]))
        self.wait(0.1)
        self.play(FadeOut(eq24))
        self.play(FadeIn(eq25[:2]))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq25[0][:].copy(), eq25[3][:2], eq25[1][-1].copy(), eq25[3][-1]),
                  mh.rtransform(eq25[1][0].copy(), eq25[3][0]), run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq25[2], eq25[3][2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq25, gp3[1]),
                  FadeIn(eq26[0]))
        self.wait(0.1)
        self.play(FadeIn(eq26[1:]))
        self.wait(0.1)
        self.play(mh.rtransform(gp3[1][0][0], eq27[0][0], gp3[1][1][0], eq27[1][0],
                                gp3[1][2], eq27[2], gp3[1][3][0], eq27[3][0], gp3[1][3][2], eq27[3][2]),
                  mh.fade_replace(gp3[1][0][1], eq27[0][1]),
                  mh.fade_replace(gp3[1][1][1], eq27[1][1]),
                  mh.fade_replace(gp3[1][3][1], eq27[3][1]),
                  mh.fade_replace(gp3[1][3][3], eq27[3][3]),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq26[0][0], eq28[0][0], eq26[0][2:4], eq28[0][1:3],
                                eq26[1], eq28[1], eq26[2][0], eq28[2][0]),
                  FadeOut(eq26[0][1]),
                  mh.fade_replace(eq26[0][-1], eq28[0][-1]),
                  mh.fade_replace(eq26[2][-1], eq28[2][-3:], coor_mask=RIGHT)
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq28[:2], eq29[:2], eq28[2][0], eq29[2][-1]),
                  FadeOut(eq28[2][1:]),
                  FadeIn(eq29[2][:-1]))

        self.wait(0.1)
        self.play(eq15.animate.to_edge(UP, buff=1), FadeOut(eq27, eq29, rate_func=linear), run_time=1.5)

        self.wait()

class Gaussian_f(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=80)
        eq1 = MathTex(r'f(x)', r'=', r'e^{-\frac12x^2}')
        eq2 = MathTex(r'f(x)', r'=', r'e^{-\frac12x^2+i\omega x}')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][0]).set_color(col_psi)
        VGroup(eq1[0][2], eq1[2][-2:], eq2[2][-1]).set_color(col_x)
        VGroup(eq1[2][0]).set_color(col_special)
        VGroup(eq1[2][2], eq1[2][4]).set_color(col_num)
        VGroup(eq1[2][3]).set_color(col_op)
        VGroup(eq2[2][-3]).set_color(col_i)
        VGroup(eq2[2][-2]).set_color(col_p)

        mh.align_sub(eq2, eq2[1], eq1[1]).move_to(eq1, coor_mask=RIGHT)

        self.add(eq1)
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq1[2][:], eq2[2][:-4]),
                  FadeIn(eq2[2][-4:], shift=mh.diff(eq1[2][-1], eq2[2][-5])),
                  run_time=1.5)
        self.wait(0.1)
        self.play(FadeOut(eq2[2][7:], rate_func=linear))

        self.wait()


class STFTCalc(Scene):
    trcol = GREY
    bgcol = GREY
    fill_op=0.7
    def __init__(self, *args, **kwargs):
        config.background_color = self.trcol if config.transparent else self.bgcol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=75)

        eq2 = MathTex(r'\widehat f(y)', r'=',  r'\frac1{\sqrt{2\pi} }', r'\int',  r'f(x)', r'e^{-ixy}', r'\,dx')
        eq3 = Tex(r'angular frequency $y=2\pi f$', font_size=60).set_z_index(2)
        eq3_2 = Tex(r'angular wavenumber $y=2\pi/\lambda$', font_size=60).set_z_index(2)
        eq4 = MathTex(r'\widehat f(y)', r'=',  r'\frac1{\sqrt{2\pi} }', r'\int',  r'f(x)', r'w(x)', r'e^{-ixy}', r'\,dx')
        eq5 = Tex(r'window function', color=RED, font_size=60).set_z_index(2)
        eq6 = MathTex(r'\widehat f(x, y)', r'=',  r'\frac1{\sqrt{2\pi} }', r'\int',  r'f(z)', r'w(z-x)', r'e^{-izy}', r'\,dz')

        VGroup(eq2, eq4, eq6).set_z_index(1)

        mh.rtransform.copy_colors = True
        VGroup(eq2[2][-2:], eq3[0][-3:-1], eq3_2[0][-4:-2]).set_color(col_pi)
        VGroup(eq2[4][0], eq4[5][0], eq2[0][:2]).set_color(col_psi)
        VGroup(eq4[5][2], eq2[4][2], eq2[5][3], eq2[-1][-1], eq6[0][3], eq6[5][4],
               eq6[4][2], eq6[5][2], eq6[6][3], eq6[7][1]).set_color(col_x)
        VGroup(eq2[0][3], eq2[5][4], eq3[0][-5], eq3[0][-1], eq3_2[0][-1]).set_color(col_p)
        VGroup(eq2[5][0]).set_color(col_special)
        VGroup(eq2[5][2]).set_color(col_i)
        VGroup(eq2[3], eq2[-1][-2], eq2[2][1:-2], eq3_2[0][-2]).set_color(col_op)
        VGroup(eq2[2][0]).set_color(col_num)
        VGroup(eq3[0][:-5], eq3_2[0][:-6]).set_color(RED)

        w = 15
        eq2 = eq_shadow(eq2, bg_stroke_width=w)
        eq3 = eq_shadow(eq3, bg_stroke_width=w)
        eq3_2 = eq_shadow(eq3_2, bg_stroke_width=w)
        eq4 = eq_shadow(eq4, bg_stroke_width=w)
        eq5 = eq_shadow(eq5, bg_stroke_width=w)
        eq6 = eq_shadow(eq6, bg_stroke_width=w)

        # mh.align_sub(eq2, eq2[0], eq1[1], coor_mask=UP)
        eq3.move_to(eq2[0]).shift(UP*1.5+RIGHT*3)
        mh.align_sub(eq3_2, eq3_2[0][-6], eq3[0][-5])
        mh.align_sub(eq4, eq4[1], eq2[1], coor_mask=UP)

        # gp = VGroup(eq2, eq4)
        # VGroup(gp, eq3, eq3_2).to_edge(DOWN, buff=0.2)

        eq5.next_to(eq4[5], UP*2)
        mh.align_sub(eq6, eq6[1], eq4[1], coor_mask=UP)

        self.add(eq2)
        self.wait(0.1)
        args = [eq3[0][6:8].get_bottom()+DOWN*0.1, eq2[0][3].get_corner(UR)+UP*0.1]
        arr1 = Arrow(*args, color=RED, stroke_width=8, buff=0).set_z_index(8)
        # arr1_ = Arrow(*args, color=BLACK, stroke_width=20, buff=0, stroke_color=BLACK, fill_color=BLACK,
        #               max_stroke_width_to_length_ratio=7, max_tip_length_to_length_ratio=0.4).set_z_index(7)
        self.play(FadeIn(eq3, arr1))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0][:7], eq3_2[0][:7], eq3[0][-5:-1], eq3_2[0][-6:-2]),
                  FadeOut(eq3[0][7:-5]), FadeIn(eq3_2[0][7:-6]),
                  mh.fade_replace(eq3[0][-1], eq3_2[0][-1], coor_mask=RIGHT),
                  FadeIn(eq3_2[0][-2], shift=mh.diff(eq3[0][-2:], eq3_2[0][-3:])*RIGHT))
        self.wait(0.1)
        self.play(FadeOut(eq3_2, arr1, ))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:5], eq4[:5], eq2[5:], eq4[6:]),
                  Succession(Wait(0.6), FadeIn(eq4[5])))
        self.wait(0.1)
        arr1 = Arrow(eq5[0][5:7].get_bottom()+DOWN*0.1, eq4[5][0].get_top()+UP*0.1, color=RED, stroke_width=8, buff=0,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.5)
        self.play(FadeIn(eq5, arr1))
        self.wait(0.1)
        self.play(FadeOut(eq5, arr1))
        self.wait(0.1)
        eq6_1 = eq6[0].copy().align_to(eq4[0], RIGHT)
        self.play(mh.rtransform(eq4[0][:3], eq6_1[:3], eq4[0][-2:], eq6_1[-2:]),
                  Succession(Wait(0.4), FadeIn(eq6_1[3:5])))
        self.wait(0.1)
        self.play(mh.rtransform(eq6_1, eq6[0], eq4[1:4], eq6[1:4], eq4[4][:2], eq6[4][:2],
                                eq4[4][-1], eq6[4][-1], eq4[5][:2], eq6[5][:2],
                                eq4[5][-1], eq6[5][-1], eq4[6][:3], eq6[6][:3], eq4[6][-1], eq6[6][-1],
                                eq4[7][0], eq6[7][0]),
                  mh.fade_replace(eq4[4][2], eq6[4][2], coor_mask=RIGHT),
                  mh.fade_replace(eq4[5][2], eq6[5][2], coor_mask=RIGHT),
                  mh.fade_replace(eq4[6][3], eq6[6][3], coor_mask=RIGHT),
                  mh.fade_replace(eq4[7][1], eq6[7][1], coor_mask=RIGHT),
                  Succession(Wait(0.4), FadeIn(eq6[5][3:-1])))
        self.wait()


class STFTCalc2(STFTCalc):
    trcol = GREY
    bgcol = GREY
    fill_op=0.7
    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=75)

        eq5 = Tex(r'window function', color=RED, font_size=60).set_z_index(2)
        eq6 = MathTex(r'\widehat f(x, y)', r'=',  r'\frac1{\sqrt{2\pi} }', r'\int',  r'f(z)', r'w(z-x)', r'e^{-izy}', r'\,dz')
        eq6.set_z_index(1)

        mh.rtransform.copy_colors = True
        VGroup(eq6[2][-2:]).set_color(col_pi)
        VGroup(eq6[4][0], eq6[5][0], eq6[0][:2]).set_color(col_psi)
        VGroup(eq6[0][3], eq6[5][4], eq6[4][2], eq6[5][2], eq6[6][3], eq6[7][1]).set_color(col_x)
        VGroup(eq6[0][5], eq6[6][4]).set_color(col_p)
        VGroup(eq6[6][0]).set_color(col_special)
        VGroup(eq6[6][2]).set_color(col_i)
        VGroup(eq6[3], eq6[-1][0], eq6[2][1:-2]).set_color(col_op)
        VGroup(eq6[2][0]).set_color(col_num)

        w = 12
        eq5 = eq_shadow(eq5, bg_stroke_width=w)
        eq6 = eq_shadow(eq6, bg_stroke_width=w)

        eq5.next_to(eq6[5][:4], UP*2).shift(LEFT*0.3)

        p0 = eq5[0][5:7].get_bottom()+DOWN*0.1
        p1 = eq6[5][0].get_top()+UP*0.1

        arr1 = Arrow(p0, p1, color=RED, stroke_width=8, buff=0,
                     max_stroke_width_to_length_ratio=20, max_tip_length_to_length_ratio=0.5).set_z_index(1)
        arr2 = arr1.get_tip().copy().set_color(BLACK).set_z_index(0).set_stroke(width=8)
        dp = (p0 - p1) * 0.05
        arr3 = Line(p0 + dp, p1 + dp, color=BLACK, stroke_width=15)
        print(arr3)
        self.add(eq5, eq6, arr1, arr2, arr3)


class STFTWindow(STFTCalc):
    trcol = BLUE
    xmax = 1.
    def construct(self):
        self.construct_plot()

    @staticmethod
    def f(x):
        y = (np.abs(x) - 0.2).clip(min=0)
        return np.exp(-12*y*y)

    @staticmethod
    def eq():
        eq2 = MathTex(r'w(x)', font_size=50, stroke_width=1.5).set_z_index(10)
        eq2[0][0].set_color(col_psi)
        eq2[0][2].set_color(col_x)
        return eq2, (0.2, 0.6)

    def construct_plot(self):
        MathTex.set_default(stroke_width=1.5)
        xmax = self.xmax
        ax = Axes(x_range=[-xmax, xmax*1.1], y_range=[0, 1.15], x_length=6, y_length=2,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "shade_in_3d": True,
                               },
                  ).set_z_index(0.5)
        eq1 = MathTex(r'x', font_size=40, color=col_x).set_z_index(1)
        eq1.next_to(ax.x_axis.get_right(), UL, buff=0.2)
        eq2, pos = self.eq()
        eq2.move_to(ax.coords_to_point(*pos))

        eq1 = eq_shadow(eq1, bg_stroke_width=6)
        eq2 = eq_shadow(eq2, bg_stroke_width=6, bg_z_index=5, fg_z_index=6)

        x = np.linspace(-xmax, xmax, 100)
        y = self.f(x)
        crv = ax.plot_line_graph(x, y, line_color=BLUE, stroke_width=5, add_vertex_dots=False).set_z_index(4)
        crv['line_graph'].set_fill(opacity=0.5, color=BLUE)
        self.add(ax, eq1, crv, eq2)

        return x, y, crv, eq2, ax

class STFTWindowGauss(STFTWindow):
    @staticmethod
    def f(x):
        return np.exp(-0.5*x*x)
    xmax = 3.
    @staticmethod
    def eq():
        eq2 = MathTex(r'w(x) = e^{-\frac12x^2}', font_size=50, stroke_width=1.5).set_z_index(10)
        eq2[0][0].set_color(col_psi)
        VGroup(eq2[0][2], eq2[0][-2:]).set_color(col_x)
        VGroup(eq2[0][7], eq2[0][9]).set_color(col_num)
        eq2[0][5].set_color(col_special)
        eq2[0][8].set_color(col_op)
        return eq2, (0., 0.45)


    def construct(self):
        self.construct_plot()

class STFTWindowGauss2(STFTWindowGauss):
    def construct(self):
        x, y, crv, eq2, ax = self.construct_plot()
        self.play(FadeOut(eq2))

        y2 = np.exp(-x*x*4)
        crv2 = ax.plot_line_graph(x, y2, line_color=BLUE, stroke_width=5, add_vertex_dots=False).set_z_index(4)
        crv2['line_graph'].set_fill(opacity=0.5, color=BLUE)
        self.play(mh.rtransform(crv, crv2), run_time=2)
        self.wait()

class Wigner(STFTWindow):
    bgcol = BLACK
    trcol = BLACK
    def construct(self):
        MathTex.set_default(font_size=65, stroke_width=1.5)
        eq1 = MathTex(r'W(x,y)', r'=', r'\frac1{2\pi}', r'\int f(x+z/2)^*f(x-z/2)e^{iyz}\,dz')
        eq2 = MathTex(r'\left\lvert\widehat f(x,y)\right\rvert^2', r'=', r'\iint', r'W(x+u,y+v)',
                      r'\rho(u, v)', r'\,dudv')
        eq3 = MathTex(r'\rho(u,v)', r'\sim', r'e^{-u^2-v^2}')



        # eq13 = MathTex(r'{\sf if\ }', r'\psi(x)', r'\sim', r'e^{-\frac12ax^2}', font_size=80)
        # eq14 = MathTex(r'{\sf then\ }', r'W(x,p)', r'\sim', r'e^{-ax^2-p^2/a}')

        mh.rtransform.copy_colors = True
        VGroup(eq1[2][0], eq1[3][7], eq1[3][16], eq2[0][-1]).set_color(col_num)
        VGroup(eq1[0][2], eq1[3][3], eq1[3][5], eq1[3][12], eq1[3][14], eq1[3][-3], eq1[3][-1],
               eq2[0][6], eq2[3][2], eq2[3][4], eq2[4][2], eq2[5][1], eq3[2][2:4]).set_color(col_x)
        VGroup(eq1[3][-4], eq1[0][4], eq2[0][8], eq2[3][6], eq2[3][8], eq2[4][4], eq2[5][3], eq3[2][5:7]).set_color(col_p)
        VGroup(eq1[3][1], eq1[3][10], eq2[0][3:5]).set_color(col_psi)
        VGroup(eq1[3][-6], eq3[2][0]).set_color(col_special)
        VGroup(eq1[2][1], eq1[3][0], eq1[3][-2], eq2[0][:3], eq2[0][-4:-1], eq2[2][:2], eq2[5][0], eq2[5][2]).set_color(col_op)
        VGroup(eq1[0][0], eq2[3][0], eq2[4][0]).set_color(col_WVD)
        VGroup(eq1[3][9], eq1[3][-5]).set_color(col_i)
        VGroup(eq1[2][-2:]).set_color(col_pi)
        mh.copy_colors_eq(eq2[4], eq3[0])

        eq1 = eq_shadow(eq1, bg_stroke_width=14)
        eq2 = eq_shadow(eq2, bg_stroke_width=14)
        eq3 = eq_shadow(eq3, bg_stroke_width=14)

        eq2.next_to(eq1, DOWN, buff=0.7)
        gp1 = VGroup(eq1.copy(), eq2).move_to(ORIGIN, coor_mask=UP)

        self.add(eq1)
        eq2_1 = eq2[0].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(eq1.animate(run_time=1.5).move_to(gp1[0]),
                  Succession(Wait(0.5), FadeIn(eq2_1[3:-4])))
        self.wait(0.1)
        self.play(FadeIn(eq2_1[:3], eq2_1[-4:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1, eq2[0], run_time=1.5), Succession(Wait(0.6), FadeIn(eq2[3][:3], eq2[3][5:7], eq2[3][-1])))
        self.wait(0.1)
        self.play(Succession(Wait(0.4), FadeIn(eq2[1:3], eq2[3][3:5], eq2[3][7:-1], eq2[4:])))

        eq3.next_to(eq2, DOWN, buff=0.7).shift(LEFT*2)
        gp1 = VGroup(eq1, eq2)
        gp2 = VGroup(gp1.copy(), eq3).move_to(ORIGIN, coor_mask=UP)

        self.wait(0.1)
        self.play(gp1.animate.move_to(gp2[0]),
                  Succession(Wait(0.4), FadeIn(eq3)))

        # self.play(FadeIn(eq2))

        self.wait()

class Sinc(STFTWindow):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=75)
        eq1 = MathTex(r'{\rm sinc}(x)', r'=', r'\frac{\sin(x)}{x}')
        VGroup(eq1[0][-2], eq1[2][-1], eq1[2][-4]).set_color(col_x)
        VGroup(eq1[0][:4], eq1[2][:3]).set_color(col_trig)
        VGroup(eq1[2][-2]).set_color(col_op)

        eq1 = eq_shadow(eq1, bg_stroke_width=15)
        self.add(eq1)

class MomentumOp(STFTWindow):
    bgcol = GREY
    trcol = BLACK

    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=70)
        eq1 = MathTex(r'Xf(x)', r'=', r'xf(x)')
        eq2 = MathTex(r'Df(x)', r'=', r'\frac{df(x)}{dx}')
        eq3 = MathTex(r'De^{i\omega x}', r'=', r'\frac{de^{i\omega x} }{dx}')
        eq4 = MathTex(r'De^{i\omega x}', r'=', r'i\omega e^{i\omega x}')
        eq5 = MathTex(r'-iDe^{i\omega x}', r'=', r'Pe^{i\omega x}')
        eq6 = MathTex(r'P', r'=', r'-iD')

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq1[0][0], eq1[0][3], eq2[2][3], eq2[2][-1], eq2[0][3], eq3[0][-1]).set_color(col_x)
        VGroup(eq2[0][0], eq3[0][-2], eq5[2][0]).set_color(col_p)
        VGroup(eq1[0][1], eq2[2][1], eq2[0][1]).set_color(col_psi)
        VGroup(eq2[2][0], eq2[2][-2]).set_color(col_op)
        VGroup(eq3[0][-3]).set_color(col_i)
        VGroup(eq3[0][-4]).set_color(col_special)
        mh.copy_colors_eq(eq1[0], eq1[2], eq3[0][1:], eq3[2][1:5])

        eq2.next_to(eq1, DOWN, buff=0.4)

        VGroup(eq1, eq2).to_edge(DOWN, buff=0.4)
        mh.align_sub(eq3, eq3[1], eq2[1])
        mh.align_sub(eq4, eq4[1], eq2[1])
        mh.align_sub(eq5, eq5[1], eq2[1])
        mh.align_sub(eq6, eq6[1], eq2[1], coor_mask=UP)

        gp = VGroup(eq1, eq2, eq3, eq4, eq5).set_z_index(2)
        box1 = SurroundingRectangle(VGroup(gp, gp.copy().shift(-gp.get_center()*RIGHT)),
                                    stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                    buff=0.2, corner_radius=0.2)

        eq1_ = eq1.copy().move_to(gp)
        self.add(eq1_, box1)
        self.play(mh.rtransform(eq1_, eq1), FadeIn(eq2))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[0][0], eq3[0][0],
                                eq2[1], eq3[1], eq2[2][0], eq3[2][0], #eq2[2][5:], eq3[2][5:]
        ),
                  mh.fade_replace(eq2[0][1:], eq3[0][1:], coor_mask=RIGHT),
                  mh.fade_replace(eq2[2][1:5], eq3[2][1:5], coor_mask=RIGHT))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2][1:5], eq4[2][2:6]),
                  mh.stretch_replace(eq3[2][2].copy(), eq4[2][0]),
                  mh.stretch_replace(eq3[2][3].copy(), eq4[2][1]),
                  FadeOut(eq3[2][0], eq2[2][5:]),
                  run_time=1.2)
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][:], eq5[0][2:], eq4[1], eq5[1],
                                eq4[2][0], eq5[0][1]),
                  FadeIn(eq5[0][0], shift=mh.diff(eq4[2][0], eq5[0][1])),
                  run_time=1.6)
        self.wait(0.1)
        # self.play(mh.stretch_replace(eq4[2][1], eq5[2][0]))
        self.play(mh.fade_replace(eq4[2][1], eq5[2][0], coor_mask=RIGHT))
        self.wait(0.1)
        self.play(FadeOut(eq5[0][3:], eq4[2][2:]),
                  Succession(Wait(0.5), mh.rtransform(eq5[0][:3], eq6[2][:], eq5[1], eq6[1], eq5[2][0], eq6[0][0],
                                                      run_time=1.6)))
        self.wait()

class EnergyStates(STFTWindow):
    bgcol = BLACK

    def construct(self):
        MathTex.set_default(stroke_width=1.5)
        xmax = 5.
        ymin = -0.5
        ymax= 0.7
        ax = Axes(x_range=[-xmax, xmax*1.06], y_range=[ymin, ymax], x_length=10, y_length=6,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "shade_in_3d": True,
                               },
                  ).set_z_index(0.5)

        npts = 400
        npol = 11
        xvals = np.linspace(-xmax, xmax, npts)
        plts = []
        fills = []
        hermite = []
        c = PI
        for i in range(npol):
            hermite.append(sp.special.hermite(i) / math.sqrt(c))
            yvals = np.exp(-xvals * xvals * 0.5) * hermite[i](xvals)
            # plts.append(ax.plot_line_graph(xvals, yvals, add_vertex_dots=False, stroke_width=6, stroke_color=BLUE).set_z_index(2))
            plts.append(ax.plot(lambda x: hermite[i](x) * np.exp(-0.5*x*x), x_range=(-xmax, xmax), stroke_width=8, stroke_color=BLUE).set_z_index(2))
            # fills.append(plts[i].copy().set_stroke(opacity=0).set_fill(opacity=0.5, color=BLUE).set_z_index(1))
            fills.append(ax.get_area(plts[i], x_range=(-xmax, xmax), color=BLUE, opacity=0.5, stroke_width=0, stroke_opacity=0).set_z_index(1))
            c *= 2 * (i+1)


        MathTex.set_default(stroke_width=2, font_size=70)
        eq0 = MathTex(r'x', stroke_width=2, stroke_color=col_x).set_z_index(5).next_to(ax.x_axis, RIGHT, buff=0.2)
        eq1 = MathTex(r'\psi_0', stroke_width=2, stroke_color=col_psi).set_z_index(5).move_to(ax.coords_to_point(0.6, 0.65))
        eq1[0][1:].set_color(col_num)

        self.add(ax, eq0)
        self.play(Create(plts[0], rate_func=linear, run_time=2),
                  Succession(Wait(1.), FadeIn(eq1, fills[0], rate_func=linear, run_time=1.2)))
        print('hi')
        for i in range(1, npol):
            eq2 = MathTex(r'\psi_{{ {} }}'.format(i), stroke_width=2, stroke_color=col_psi).set_z_index(5)
            eq2[0][1:].set_color(col_num)
            mh.align_sub(eq2, eq2[0][0], eq1[0][0])
            plt1 = plts[i-1].copy()
            plts[i-1].set_stroke(opacity=0.3, color=RED)
            self.play(mh.rtransform(plt1, plts[i], fills[i-1], fills[i]),
                      mh.rtransform(eq1[0][0], eq2[0][0]),
                      FadeOut(eq1[0][1:]),
                      FadeIn(eq2[0][1:]))
            eq1 = eq2
            self.wait(0.2)

        self.wait()

class EigenApprox(EnergyStates):
    npol = 41
    npts = 800
    xmax = 5.
    c_r = 2.

    def basisfns(self):
        hermite = []
        psi = []
        c = math.sqrt(PI)
        npol = self.npol
        c_r = self.c_r

        def f(j):
            return lambda x: hermite[j](x) * np.exp(-0.5*x*x)

        for i in range(npol):
            hermite.append(sp.special.hermite(i) / math.sqrt(c))
            psi.append(f(i))
            c *= 2 * (i+1)

        coeffs = np.zeros(npol)
        coeffs[0] = float(1. - sp.stats.norm.cdf(-c_r)*2) * math.sqrt(2*math.sqrt(PI))
        for i in range(2, npol, 2):
            print(psi[i-1](-c_r), psi[i-1](0))
            coeffs[i] = -4*psi[i-1](c_r) / math.sqrt(2*i) + math.sqrt((i-1)/i) * coeffs[i-2]

        print(2*c_r - sum(coeffs*coeffs))

        return psi, coeffs


    def construct(self):
        psi, coeffs = self.basisfns()

        MathTex.set_default(stroke_width=1.5)
        xmax = self.xmax
        ymin = -0.2
        ymax= 1.5
        ax = Axes(x_range=[-xmax, xmax*1.06], y_range=[ymin, ymax], x_length=10, y_length=6,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "shade_in_3d": True,
                               },
                  ).set_z_index(0.5)

        npts = self.npts
        npol = self.npol
        xvals = np.linspace(-xmax, xmax, npts)

        yvals = np.zeros(npts)

        plts = []
        fills = []
        for i in range(0, npol, 2):
            yvals += coeffs[i] * psi[i](xvals)
            plt = ax.plot_line_graph(xvals, yvals, stroke_color=BLUE, add_vertex_dots=False, stroke_width=8)
            plt2 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                      np.concatenate(([0.], yvals, [0.])),
                                      add_vertex_dots=False, stroke_width=0, stroke_opacity=0, fill_opacity=0.5,
                                      fill_color=BLUE)
            # plt2.set_stroke(opacity=0).set_fill(opacity=0.5)
            plts.append(plt.set_z_index(5))
            fills.append(plt2.set_z_index(2))

        c_r = self.c_r

        plt2 = ax.plot_line_graph(np.array([-xmax, -c_r, -c_r, c_r, c_r, xmax]),
                                  np.array([0., 0., 1., 1., 0., 0.]),
                                  add_vertex_dots=False, stroke_width=6, stroke_color=ORANGE, fill_color=ORANGE, fill_opacity=0.3)
        plt2.set_z_index(1)

        self.add(ax, plt2)
        self.play(Create(plts[0], rate_func=linear, run_time=2),
                  Succession(Wait(1.), FadeIn(fills[0], rate_func=linear, run_time=1.2)))
        for i in range(1, npol//2):
            # plt1 = plts[i-1].copy()
            # plts[i-1].set_stroke(opacity=0.3, color=RED)
            plt1 = plts[i-1]
            self.play(mh.rtransform(plt1, plts[i], fills[i-1], fills[i]),
                      # mh.rtransform(eq1[0][0], eq2[0][0]),
                      # FadeOut(eq1[0][1:]),
                      # FadeIn(eq2[0][1:]))
                      run_time = 2. / (i+1)
                      )
            # eq1 = eq2
            self.wait(0.2)

        self.wait()

class RotatePt(STFTWindow):
    bgcol = BLACK
    def construct(self):
        self.do_anim()

    def do_anim(self, just_eq=False):
        MathTex.set_default(stroke_width=1.5, font_size=60)
        eq4 = MathTex(r'x^\prime', r'=', r'x\cos\theta + p\sin\theta')
        eq5 = MathTex(r'p^\prime', r'=', r'-x\sin\theta + p\cos\theta')

        eq6 = MathTex(r'X\mathcal F_\theta f', r'=', r'\mathcal F_\theta', r'(X\cos\theta + P\sin\theta)', r'f')
        eq7 = MathTex(r'X\mathcal F_\theta f(x)')
        eq8 = MathTex(r'P\mathcal F_\theta f', r'=', r'\mathcal F_\theta', r'(-X\sin\theta + P\cos\theta)', r'f')
        eq6_ = MathTex(r'X\mathcal F_\theta', r'=', r'\mathcal F_\theta', r'(X\cos\theta + P\sin\theta)')
        eq8_ = MathTex(r'P\mathcal F_\theta', r'=', r'\mathcal F_\theta', r'(-X\sin\theta + P\cos\theta)')

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq4[0], eq4[2][0], eq7[0][0], eq7[0][5]).set_color(col_x)
        VGroup(eq4[2][6], eq5[0], eq8[0][0]).set_color(col_p)
        VGroup(eq4[2][1:4], eq4[2][7:10]).set_color(col_trig)
        VGroup(eq7[0][3], eq6[4]).set_color(col_psi)
        VGroup(eq4[2][4], eq4[2][10], eq7[0][2], eq6[2][1]).set_color(col_angle)
        VGroup(eq7[0][1], eq6[2][0]).set_color(col_ft)
        mh.copy_colors_eq(eq4[2][:], eq5[2][1:])

        mh.align_sub(eq5, eq5[1], eq4[1]).next_to(eq4, DOWN, buff=0.3, coor_mask=UP)
        eq4[2].align_to(eq5[2][1], LEFT)
        VGroup(eq4, eq5).next_to(ORIGIN, RIGHT, buff=0.5)
        eq6.to_edge(RIGHT, buff=0.3).shift(DOWN*1)
        mh.align_sub(eq7, eq7[0][1], eq6[2][0]).move_to(eq6, coor_mask=RIGHT)
        eq8.next_to(eq6, DOWN, buff=0.3).align_to(eq6, RIGHT)
        mh.align_sub(eq6[:2], eq6[1], eq8[1], coor_mask=RIGHT)
        eq8_.next_to(eq6_, DOWN, buff=0.3).align_to(eq6_, RIGHT)
        mh.align_sub(eq6_[:2], eq6_[1], eq8_[1], coor_mask=RIGHT)

        gp2 = VGroup(VGroup(eq6[0][:-1], eq6[1], eq6[2], eq6[3]), VGroup(eq8[0][:-1], eq8[1], eq8[2], eq8[3])).set_z_index(1)
        gp3 = VGroup(VGroup(eq6_[0][:], eq6_[1], eq6_[2], eq6_[3]), VGroup(eq8_[0][:], eq8_[1], eq8_[2], eq8_[3])).set_z_index(1)
        gp3.move_to(ORIGIN).to_edge(DOWN, buff=0.4)

        if just_eq:
            mh.copy_colors_eq(eq4[2][:], eq6[3][1:-1], eq4[2][:], eq8[3][2:-1],
                              eq6[2], eq8[2], eq6[2][:], eq6[0][1:-1], eq6[2][:], eq8[0][1:-1])
            eq6[0][0].set_color(col_x)
            return gp2.to_edge(DOWN, buff=0.5), gp3

        ax = Axes(x_range=[-1, 1.1], y_range=[-1, 1.1], x_length=6, y_length=6,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  )
        p1 = (0.9*0.95, 0.8*0.95)
        pt1 = ax.coords_to_point(*p1)
        origin = ax.coords_to_point(0,.0)
        dot1 = Dot(pt1, radius=0.15, color=ORANGE).set_z_index(5)
        line1 = Line(origin, pt1, stroke_color=BLUE, stroke_width=8).set_z_index(2)
        theta = PI/2 * 1.1
        theta1 = math.atan(pt1[1]/pt1[0])

        pt2 = rotate_vector(pt1-origin, -theta)+origin

        eq2 = MathTex(r'(x,p)', font_size=60, stroke_width=1.5).next_to(pt1, UR, buff=0.15)
        eq3 = MathTex(r'(x^\prime,p^\prime)', font_size=60, stroke_width=1.5).next_to(pt2, DR, buff=0.15)
        VGroup(eq2[0][1], eq3[0][1:3]).set_color(col_x)
        VGroup(eq2[0][3], eq3[0][4:6]).set_color(col_p)


        arc1 = Arc(1, theta1, -theta, arc_center=origin, stroke_width=8).set_z_index(1)
        eq1 = MathTex(r'\theta', font_size=70, color=col_angle, stroke_width=2)
        eq1 = eq_shadow(eq1, bg_stroke_width=15)
        eq1.move_to(origin + RIGHT*0.6 + DOWN*0.1)
        dl = VGroup(dot1, line1).copy()
        self.add(ax, line1, dot1, eq2)
        self.wait(0.1)
        self.play(AnimationGroup(Rotate(dl, -theta, about_point=origin),
                  Create(arc1), run_time=1.6),
                  Succession(Wait(0.8), FadeIn(eq1)),
                  Succession(Wait(1.14), FadeIn(eq3))
                  )
        self.wait(0.1)
        gp1 = VGroup(ax, eq1, eq2, eq3, arc1, line1, dot1, dl)
        self.play(gp1.animate(run_time=1.5).to_edge(LEFT, buff=0.4),
                  Succession(Wait(0.8), FadeIn(eq4, eq5)))
        self.wait(0.1)
        self.play(VGroup(eq4, eq5).animate.shift(UP*1.5), Succession(Wait(0.4), FadeIn(eq7[0][-6:])))
        self.wait(1.2)
        self.play(FadeIn(eq7[0][0]))
        self.wait(0.1)
        shift = mh.diff(eq7[0][:-3], eq6[0][:])
        eq4_ = eq4.copy()
        self.play(AnimationGroup(mh.rtransform(eq7[0][:-3], eq6[0][:]),
                  FadeOut(eq7[0][-3:], shift=shift),
                                 run_time=1.4),
                  Succession(Wait(1),
                             AnimationGroup(mh.rtransform(eq4_[1], eq6[1], eq4_[2][1:6], eq6[3][2:7], eq4_[2][7:11], eq6[3][8:12]),
                                            mh.stretch_replace(eq4_[2][0], eq6[3][1]),
                                            mh.stretch_replace(eq4_[2][6], eq6[3][7]),
                                            run_time=1.8)),
                  Succession(Wait(2), FadeIn(eq6[3][0], eq6[3][-1], eq6[4], eq6[2])),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq6[0][1:].copy(), eq8[0][1:], eq6[1].copy(), eq8[1],
                                eq6[3][0].copy(), eq8[3][0], eq6[3][-1].copy(), eq8[3][-1], eq6[4].copy(), eq8[4], eq6[2].copy(), eq8[2]),
                  mh.fade_replace(eq6[0][0].copy(), eq8[0][0]),
                  Succession(Wait(0.8),AnimationGroup(
                      mh.rtransform(eq5[2][0].copy(), eq8[3][1], eq5[2][2:7].copy(), eq8[3][3:8], eq5[2][8:12].copy(), eq8[3][9:13]),
                      mh.stretch_replace(eq5[2][1].copy(), eq8[3][2]),
                      mh.stretch_replace(eq5[2][7].copy(), eq8[3][8]),
                      run_time=1.8))
                  )
        self.wait(0.1)
        self.play(VGroup(eq6[0][-1], eq8[0][-1], eq6[4], eq8[4]).animate.set_opacity(0))
        self.wait(0.1)

        eq10 = MathTex(r'X\mathcal F_\theta f(x)', r'=', r'\sqrt{\frac{1-i\cot\theta}{2\pi}}', r'\int',
                      r'x', r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}', r'f(y)', r'\,dy')
        eq12 = MathTex(r'\left(y\cos\theta+i\sin\theta\frac{\partial}{\partial y}\right)', r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}',
                       r'f(y)')
        mh.font_size_sub(eq12, 0, 55)
        eq11 = MathTex(r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}', r'\left(y\cos\theta-i\sin\theta\frac{\partial}{\partial y}\right)',
                       r'f(y)')
        mh.font_size_sub(eq11, 1, 55)
        eq13 = MathTex(r'X\cos\theta+P\sin\theta')
        eq14 = MathTex(r'\mathcal F_\theta', r'(X\cos\theta+P\sin\theta)', r'f(x)')

        eq15 = MathTex(r'P\mathcal F_\theta f(x)', r'=', r'\sqrt{\frac{1-i\cot\theta}{2\pi}}', r'\int',
                      r'\!\left(\!-i\frac{\partial}{\partial x}\right)\!',
                       r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}', r'f(y)', r'\,dy')
        mh.font_size_sub(eq15, 4, 50)
        eq16 = MathTex(r'\left(-y\sin\theta+i\cos\theta\frac{\partial}{\partial y}\right)', r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}',
                       r'f(y)')
        mh.font_size_sub(eq16, 0, 55)
        eq17 = MathTex(r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}', r'\left(-y\sin\theta-i\cos\theta\frac{\partial}{\partial y}\right)',
                       r'f(y)')
        mh.font_size_sub(eq17, 1, 55)
        eq18 = MathTex(r'-X\sin\theta+P\cos\theta')
        eq19 = MathTex(r'\mathcal F_\theta', r'(-X\sin\theta+P\cos\theta)', r'f(x)')

        VGroup(eq10[0][1], eq14[0][0]).set_color(col_ft)
        VGroup(eq10[0][3], eq10[6][0], eq14[2][0], eq19[2][0]).set_color(col_psi)
        VGroup(eq10[0][2], eq10[2][-4], eq10[5][15], eq10[5][23], eq12[0][5], eq12[0][11], eq14[0][1]).set_color(col_angle)
        VGroup(eq10[0][0], eq10[0][5], eq10[5][6:8], eq10[5][18], eq10[4],
               eq14[2][2], eq15[4][6], eq13[0][0]).set_color(col_x)
        VGroup(eq10[2][:2], eq10[2][-3], eq10[3], eq10[5][2], eq10[7][0], eq12[0][12:15], eq15[4][3:6]).set_color(col_op)
        VGroup(eq10[2][2], eq10[5][1], eq10[5][3]).set_color(col_num)
        VGroup(eq10[2][4], eq10[5][4], eq10[5][17], eq12[0][7], eq15[4][2]).set_color(col_i)
        VGroup(eq10[2][5:8], eq10[5][12:15], eq10[5][20:23], eq12[0][2:5], eq12[0][8:11]).set_color(col_trig)
        VGroup(eq10[2][-2:]).set_color(col_pi)
        VGroup(eq10[5][0]).set_color(col_special)
        VGroup(eq10[5][9:11], eq10[5][19], eq10[6][2], eq10[7][1], eq12[0][1], eq12[0][15], eq13[0][6]).set_color(col_p)
        mh.copy_colors_eq(eq10[:4], eq15[:4], eq10[5:], eq15[5:])
        mh.copy_colors_eq(eq12[0][1:], eq16[0][2:])
        mh.copy_colors_eq(eq13[0][:], eq18[0][1:])
        mh.copy_colors_eq(eq14[0], eq19[0], eq14[1][1:], eq19[1][2:], eq14[2], eq19[2])

        eq10[2:].next_to(eq10[:2], DOWN, buff=0.5)
        eq10[:2].align_to(eq10[2], LEFT).shift(RIGHT*2)
        eq10.to_edge(UP, buff=0.5).move_to(ORIGIN, coor_mask=RIGHT)
        eq14.move_to(eq10[2:], coor_mask=UP)
        eq15[:2].move_to(eq10[:2])
        eq15[2:].move_to(eq10[2:])
        eq19.move_to(eq15[2:], coor_mask=UP)

        self.play(FadeOut(gp1, eq4, eq5),
                  FadeIn(eq10), run_time=1.8)
        self.wait(0.1)
        br1 = Brace(VGroup(eq10[4:7]), DOWN, color=RED, buff=0.1)
        eq12.next_to(br1, DOWN).to_edge(RIGHT, buff=0.2)
        mh.align_sub(eq11, eq11[2], eq12[2])
        self.play(FadeIn(br1))
        self.play(AnimationGroup(mh.rtransform(eq10[5:7].copy(), eq12[1:3]),
                  FadeOut(eq10[4].copy(), target_position=eq12[0]),
                  VGroup(eq6, eq8).animate.to_edge(DOWN, buff=0.5),
                  run_time=1.5),
                  Succession(Wait(0.3), FadeIn(eq12[0], run_time=1.5)))
        self.wait(0.1)
        txt1 = Tex(r'\sf integration by parts', font_size=50, color=RED).next_to(eq12[:2], DOWN, buff=0.1)
        self.play(FadeIn(txt1))
        self.play(mh.rtransform(eq12[0][:6], eq11[1][:6], eq12[0][7:], eq11[1][7:], eq12[1], eq11[0], eq12[2], eq11[2]),
                  mh.fade_replace(eq12[0][6], eq11[1][6]),
                  run_time=1.6)
        self.wait(0.1)
        mh.align_sub(eq13, eq13[0][1], eq11[1][2]).move_to(eq11[1][1:-1], coor_mask=RIGHT)
        self.play(mh.fade_replace(eq11[1][1], eq13[0][0]),
                  mh.rtransform(eq11[1][2:6], eq13[0][1:5], eq11[1][8:12], eq13[0][7:11]),
                  mh.fade_replace(eq11[1][12:16], eq13[0][6], coor_mask=RIGHT),
                  mh.fade_replace(eq11[1][6], eq13[0][5]),
                  FadeOut(eq11[1][7]),
                  FadeOut(txt1))
        self.wait(0.1)
        self.play(FadeOut(eq10[2:], br1, eq11[0]),
                  mh.rtransform(eq13[0][:], eq14[1][1:-1], eq11[2][:2], eq14[2][:2], eq11[2][-1], eq14[2][-1]),
                  mh.stretch_replace(eq11[1][0], eq14[1][0]),
                  mh.stretch_replace(eq11[1][-1], eq14[1][-1]),
                  mh.fade_replace(eq11[2][2], eq14[2][2]),
                  FadeIn(eq14[0], shift=mh.diff(eq11[1][0], eq14[1][0])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(FadeOut(eq10[:2], eq14), FadeIn(eq15))
        self.wait(0.1)
        br1 = Brace(VGroup(eq15[4:7]), DOWN, color=RED, buff=0.1)
        eq16.next_to(br1, DOWN).to_edge(RIGHT, buff=0.2)
        mh.align_sub(eq17, eq17[2], eq16[2])
        self.play(FadeIn(br1))
        self.play(AnimationGroup(mh.rtransform(eq15[5:7].copy(), eq16[1:3]),
                  FadeOut(eq15[4].copy(), target_position=eq16[0]),
                  run_time=1.5),
                  Succession(Wait(0.3), FadeIn(eq16[0], run_time=1.5)))
        self.wait(0.1)
        self.play(mh.rtransform(eq16[0][:7], eq17[1][:7], eq16[0][8:], eq17[1][8:], eq16[1], eq17[0], eq16[2], eq17[2]),
                  mh.fade_replace(eq16[0][7], eq17[1][7]),
                  run_time=1.6)
        self.wait(0.1)
        mh.align_sub(eq18, eq18[0][2], eq17[1][3]).move_to(eq17[1][1:-1], coor_mask=RIGHT)
        self.play(mh.fade_replace(eq17[1][2], eq18[0][1]),
                  mh.rtransform(eq17[1][3:7], eq18[0][2:6], eq17[1][9:13], eq18[0][8:12], eq17[1][1], eq18[0][0]),
                  mh.fade_replace(eq17[1][13:17], eq18[0][7], coor_mask=RIGHT),
                  mh.fade_replace(eq17[1][7], eq18[0][6]),
                  FadeOut(eq17[1][8]),
                  )
        self.wait(0.1)
        self.play(FadeOut(eq15[2:], br1, eq17[0]),
                  mh.rtransform(eq18[0][:], eq19[1][1:-1], eq17[2][:2], eq19[2][:2], eq17[2][-1], eq19[2][-1]),
                  mh.stretch_replace(eq17[1][0], eq19[1][0]),
                  mh.stretch_replace(eq17[1][-1], eq19[1][-1]),
                  mh.fade_replace(eq17[2][2], eq19[2][2]),
                  FadeIn(eq19[0], shift=mh.diff(eq17[1][0], eq19[1][0])),
                  run_time=1.6)

        self.wait(0.1)
        self.play(#FadeOut(eq15[:2], eq19),
                  # VGroup(eq6, eq8).animate.move_to(ORIGIN, coor_mask=RIGHT).to_edge(DOWN, buff=0.4),
                  mh.rtransform(gp2, gp3),
                  run_time=1.2)

        self.wait()

class RotateDE(RotatePt):
    bgcol = GREY
    trcol = BLACK
    def construct(self):
        gp1, gp2 = self.do_anim(True)

        eq1 = MathTex(r'\frac{d^2}{dx^2}f(x) + x^2f(x)', r'=', r'0').set_z_index(1)
        eq2 = MathTex(r'(X^2+D^2)', r'f(x)', r'=', r'0').set_z_index(1)
        eq3 = MathTex(r'(X+iD)', r'(X-iD)', r'f(x)', r'=', r'0').set_z_index(1)
        eq4 = MathTex(r'(X+iD)', r'(X-iD)', r'f(x)', r'=', r'if(x)').set_z_index(1)
        eq5 = MathTex(r'(X-P)', r'(X+P)', r'f(x)', r'=', r'if(x)').set_z_index(1)
        eq6 = MathTex(r'\mathcal F_{-\frac\pi4}', r'(X-P)', r'(X+P)', r'f(x)', r'=', r'i\mathcal F_{-\frac\pi4}f(x)').set_z_index(1)
        eq7 = MathTex(r'\sqrt2 X', r'\mathcal F_{-\frac\pi4}', r'(X+P)', r'f(x)', r'=', r'i\mathcal F_{-\frac\pi4}f(x)').set_z_index(1)
        eq8 = MathTex(r'\sqrt2 P', r'\mathcal F_{-\frac\pi4}', r'f(x)', r'=', r'i\mathcal F_{-\frac\pi4}f(x)').set_z_index(1)
        eq9 = MathTex(r'-2ix\frac{d}{dx}', r'\mathcal F_{-\frac\pi4}', r'f(x)', r'=', r'i\mathcal F_{-\frac\pi4}f(x)').set_z_index(1)
        eq10 = MathTex(r'2x\frac{d}{dx}', r'\mathcal F_{-\frac\pi4}', r'f(x)', r'=', r'-\mathcal F_{-\frac\pi4}f(x)').set_z_index(1)
        eq11 = MathTex(r'\mathcal F_{-\frac\pi4}', r'f(x)', r'=', r'\frac1{\sqrt x}').set_z_index(1)
        eq12 = MathTex(r'f(x)', r'=', r'\mathcal F_{\frac\pi4}', r'\frac1{\sqrt x}').set_z_index(1)
        eq12_1 = MathTex(r'f(x)', r'=', r'\sqrt{\mathcal F}', r'\frac1{\sqrt x}').set_z_index(1)
        eq13 = MathTex(r'f(x)', r'=', r'a', r'\mathcal F_{\frac\pi4}', r'\frac1{\sqrt {\lvert x\rvert} }',
                       r'+', r'b', r'\mathcal F_{\frac\pi4}', r'\frac{ {\rm sign}(x)}{\sqrt {\lvert x\rvert} }').set_z_index(1)
        eq14 = MathTex(r'f(x)', r'=', r'a', r'\,\sqrt{\lvert x\rvert}J_{-\frac14}\left(', r'\frac{x^2}{2}', r'\right)', r'+',
                       r'b\;', r'{\rm sign}(x)\sqrt{\lvert x\rvert}J_{\frac14}\left(', r'\frac{x^2}{2}', r'\right)').set_z_index(1)

        box1 = SurroundingRectangle(gp1, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                    corner_radius=0.2, buff=0.2)
        box2 = SurroundingRectangle(gp2, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                    corner_radius=0.2, buff=0.2)

        VGroup(eq6[0][0]).set_color(col_ft)
        VGroup(eq6[0][1:]).set_color(col_angle)
        VGroup(eq1[0][4], eq1[0][8], eq1[0][11:13], eq1[0][15], eq7[0][3], eq9[0][7],
               eq11[3][4], eq13[8][5]).set_color(col_x)
        VGroup(eq2[0][-3:-1], eq5[0][3], eq5[1][3], eq8[0][-1]).set_color(col_p)
        VGroup(eq1[0][:4], eq1[0][5], eq7[0][:2], eq9[0][4:7], eq11[3][1:4], eq12_1[2][:2],
               eq13[4][-1], eq13[4][-3], eq13[8][-1], eq13[8][-3], eq13[8][:4]).set_color(col_op)
        VGroup(eq1[0][6], eq1[0][-4]).set_color(col_psi)
        VGroup(eq1[2], eq7[0][2], eq11[3][0]).set_color(col_num)
        VGroup(eq3[1][3], eq3[0][3], eq4[4][0], eq9[0][2]).set_color(col_i)
        VGroup(eq13[2], eq13[6]).set_color(col_var)
        mh.copy_colors_eq(eq1[0][-4:], eq4[4][1:])
        mh.copy_colors_eq(eq6[0][:], eq6[5][1:6])
        mh.copy_colors_eq(eq7[0][:-1], eq8[0][:-1])

        eq1.move_to(box2)
        mh.align_sub(eq2, eq2[2], eq1[1], coor_mask=UP)
        mh.align_sub(eq3, eq3[3], eq1[1], coor_mask=UP)
        mh.align_sub(eq4, eq4[3], eq3[3]).move_to(box2, coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[3], eq1[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[3], eq1[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[4], eq6[4])
        mh.align_sub(eq8, eq8[3], eq6[4])
        mh.align_sub(eq9, eq9[3], eq6[4], coor_mask=UP)
        mh.align_sub(eq10, eq10[3], eq6[4], coor_mask=UP)
        mh.align_sub(eq11, eq11[2], eq6[4], coor_mask=UP)
        mh.align_sub(eq12, eq12[1], eq6[4], coor_mask=UP)
        mh.align_sub(eq13, eq13[1], eq6[4], coor_mask=UP).shift(UP*0.1)
        mh.align_sub(eq12_1, eq12_1[1], eq12[1])
        mh.align_sub(eq14, eq14[1], eq13[1], coor_mask=UP)

        w = eq6.width + 0.4
        h = box2.height
        box3 = RoundedRectangle(width=w, height=h, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                fill_opacity=self.fill_op, corner_radius=0.2).move_to(box2)
        w = eq14.width + 0.4
        box4 = RoundedRectangle(width=w, height=h, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                fill_opacity=self.fill_op, corner_radius=0.2).move_to(box2)

        rect1 = SurroundingRectangle(eq14[3:6], fill_opacity=0, stroke_width=6, stroke_color=YELLOW,
                                     buff=0.1, corner_radius=0.15).set_z_index(5)
        rect2 = SurroundingRectangle(eq14[8:11], fill_opacity=0, stroke_width=6, stroke_color=BLUE,
                                     buff=0.1, corner_radius=0.15).set_z_index(5)

        self.play(mh.rtransform(gp1, gp2, box1, box2, run_time=1.2))
        self.wait(0.1)
        self.play(FadeOut(gp2), FadeIn(eq1))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq1[1:], eq2[2:], eq1[0][-4:], eq2[1][:], eq1[0][10], eq2[0][3], eq1[0][-5], eq2[0][2]),
                  mh.rtransform(eq1[0][6:10], eq2[1][:]),
                  mh.stretch_replace(eq1[0][-6], eq2[0][1]),
                  mh.fade_replace(eq1[0][:6], eq2[0][-3:-1], coor_mask=RIGHT),
                  run_time=1.6),
                  Succession(Wait(1), FadeIn(eq2[0][0], eq2[0][-1], run_time=1.2)))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq2[1:], eq3[2:], eq2[0][:2], eq3[0][:2], eq2[0][-1], eq3[1][-1],
                                eq2[0][-3], eq3[0][-2], eq2[0][3], eq3[0][2], eq2[0][1].copy(), eq3[1][1],
                                               eq2[0][-3].copy(), eq3[1][-2]),
                                 mh.fade_replace(eq2[0][3].copy(), eq3[1][2]),
                                 FadeIn(eq3[0][3], shift=mh.diff(eq2[0][4], eq3[0][4])),
                                 FadeIn(eq3[1][3], shift=mh.diff(eq2[0][4], eq3[1][4])),
                                 run_time=1.6),
                  FadeOut(eq2[0][2], eq2[0][-2]),
                  Succession(Wait(0.8), FadeIn(eq3[0][-1], eq3[1][0])))
        circ1 = mh.circle_eq(VGroup(eq3[0][-2:], eq3[1][:2])).set_z_index(5)
        self.wait(0.1)
        self.play(Create(circ1, run_time=0.6, rate_func=linear))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:-1], eq4[:-1]),
                  FadeOut(eq3[-1]), FadeIn(eq4[-1]),
                  FadeOut(circ1, shift=mh.diff(eq3[0][-1], eq4[0][-1])))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][:2], eq5[0][:2], eq4[0][-1], eq5[0][-1], eq4[1][:2], eq5[1][:2],
                                eq4[1][-1], eq5[1][-1], eq4[2:], eq5[2:]),
                  mh.fade_replace(eq4[0][2], eq5[0][2]),
                  mh.fade_replace(eq4[1][2], eq5[1][2]),
                  mh.fade_replace(eq4[0][4], eq5[0][3], coor_mask=RIGHT),
                  mh.fade_replace(eq4[1][4], eq5[1][3], coor_mask=RIGHT),
                  FadeOut(eq4[0][3], shift=mh.diff(eq4[0][4], eq5[0][3])*RIGHT),
                  FadeOut(eq4[1][3], shift=mh.diff(eq4[1][4], eq5[1][3])*RIGHT))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:4], eq6[1:5], eq5[4][0], eq6[5][0], eq5[4][1:], eq6[5][6:], box2, box3),
                  Succession(Wait(0.5), FadeIn(eq6[0], eq6[5][1:6])))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[2:], eq7[2:], eq6[0], eq7[1]),
                  FadeOut(eq6[1]),
                  FadeIn(eq7[0]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq7[3:], eq8[2:], eq7[1], eq8[1]),
                  FadeOut(eq7[2]),
                  FadeIn(eq8[0]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq7[0][2], eq9[0][1], eq8[1:], eq9[1:]),
                  mh.rtransform(eq8[0][2], eq9[0][1]),
                  FadeOut(eq7[0][:2], shift=mh.diff(eq7[0][2], eq9[0][1])),
                  FadeOut(eq8[0][:2], shift=mh.diff(eq8[0][2], eq9[0][1])),
                  mh.stretch_replace(eq7[0][3], eq9[0][3]),
                  mh.fade_replace(eq8[0][3], eq9[0][4:], coor_mask=RIGHT),
                  FadeIn(eq9[0][2], shift=mh.diff(eq8[0][-1], eq9[0][2])*RIGHT),
                  FadeIn(eq9[0][0], shift=mh.diff(eq8[0][-1], eq9[0][0])*RIGHT),
                  run_time=1.5)
        self.play(FadeOut(eq9[0][0], eq9[0][2], eq9[4][0]),
                  mh.rtransform(eq9[0][1], eq10[0][0], eq9[0][3:], eq10[0][1:], eq9[1:4], eq10[1:4],
                                eq9[4][1:], eq10[4][1:]),
                  FadeIn(eq10[4][0]))
        self.wait(0.1)
        self.play(mh.rtransform(eq10[1:4], eq11[:3]),
                  FadeOut(eq10[0], eq10[4]),
                  FadeIn(eq11[3]))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[0][0], eq12[2][0], eq11[0][2:], eq12[2][1:],
                                eq11[1:3], eq12[:2], eq11[3], eq12[3]),
                  FadeOut(eq11[0][1], shift=mh.diff(eq11[0][0], eq12[2][0])),
                  run_time=1.5)
        self.wait(0.1)
        eq12_ = eq12.copy()
        self.play(mh.rtransform(eq12[:2], eq12_1[:2], eq12[2][0], eq12_1[2][2], eq12[3], eq12_1[3]),
                  FadeOut(eq12[2][1:], shift=mh.diff(eq12[2][0], eq12_1[2][2])),
                  FadeIn(eq12_1[2][:2], shift=mh.diff(eq12[2][0], eq12_1[2][2])),
                  rate_func=there_and_back_with_pause, run_time=3)
        eq12_1.set_opacity(0)
        eq12 = eq12_
        self.add(eq12)
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq12[:2], eq13[:2], eq12[3][-1], eq13[4][-2],
                                eq12[2], eq13[3], eq12[3][:2], eq13[4][:2]),
                  mh.rtransform(eq12[3][-1].copy(), eq13[8][-2], eq12[2].copy(), eq13[7],
                                eq12[3][1].copy(), eq13[8][7]),
                  mh.stretch_replace(eq12[3][2:4], eq13[4][2:4]),
                  mh.stretch_replace(eq12[3][2:4].copy(), eq13[8][8:10]),
                  FadeIn(eq13[4][-1], eq13[4][-3], shift=mh.diff(eq12[3][-1], eq13[4][-2])),
                  FadeIn(eq13[8][-1], eq13[8][-3], shift=mh.diff(eq12[3][-1], eq13[8][-2])),
                  FadeIn(eq13[8][:8], target_position=eq12[3][0]),
                  run_time=1.6),
                  Succession(Wait(0.8), FadeIn(eq13[2], eq13[5:7]))
                  )
        self.wait(0.1)
        mh.copy_colors_eq(eq13[4][2:], eq14[3][:5], eq13[4][2:], eq14[8][7:12], eq13[8][:7], eq14[8][:7])
        mh.copy_colors_eq(eq6[0][:], eq14[3][5:10], eq13[3][:], eq14[8][12:16])
        eq14[4][:2].set_color(col_x)
        eq14[4][2].set_color(col_op)
        eq14[4][3:].set_color(col_num)
        mh.copy_colors_eq(eq14[4], eq14[9])

        self.play(mh.rtransform(eq13[:3], eq14[:3], eq13[5:7], eq14[6:8], box3, box4),
                  mh.fade_replace(eq13[3:5], eq14[3:6], coor_mask=RIGHT),
                  mh.fade_replace(eq13[7:], eq14[8:], coor_mask=RIGHT),
                  run_time=1.5)

        circ1 = mh.circle_eq(eq14[3][5:10]).set_z_index(5)
        circ2 = mh.circle_eq(eq14[8][12:16]).set_z_index(5)
        self.play(Create(circ1), Create(circ2), run_time=0.8, rate_func=linear)
        self.wait(0.1)
        self.play(FadeOut(circ1, circ2))
        self.wait(0.1)
        self.play(FadeIn(rect1, rect2), rate_func=linear)

        self.wait()

class DESolution(STFTWindow):
    bgcol = BLACK
    def construct(self):
        xmax = 10
        ymax = 1.2
        ymin = -0.9
        ax = Axes(x_range=[-1.1 * xmax, 1.1 * xmax, 5], y_range=[ymin, 1.1 * ymax, 2], x_length=13, y_length=5,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': True,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(1).to_edge(UP, buff=0.6)
        ax.shift(-ax.coords_to_point(0,0)*RIGHT)
        box = SurroundingRectangle(ax, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                   buff=0.15, corner_radius=0.15)
        eq1 = MathTex(r'x', font_size=50, color=col_x, stroke_width=1.5).next_to(ax.x_axis.get_right(), UP, buff=0.2).set_z_index(5).shift(LEFT*0.1)
        eq2 = MathTex(r'f(x)', font_size=50, stroke_width=1.5).next_to(ax.y_axis.get_top(), RIGHT, buff=0.2).set_z_index(5).shift(DOWN*0.)
        eq3 = MathTex('-10', '-5', '5', '10', color=col_num, font_size=35, stroke_width=1).next_to(ax.coords_to_point(0,0), DOWN, buff=0.2)
        eq3[0].move_to(ax.x_axis.ticks[0], coor_mask=RIGHT)
        mh.align_sub(eq3[0], eq3[0][1:], ax.x_axis.ticks[0], coor_mask=RIGHT)
        mh.align_sub(eq3[1], eq3[1][1:], ax.x_axis.ticks[1], coor_mask=RIGHT)
        eq3[2].move_to(ax.x_axis.ticks[2], coor_mask=RIGHT)
        eq3[3].move_to(ax.x_axis.ticks[3], coor_mask=RIGHT)
        eq2[0][0].set_color(color=col_psi)
        eq2[0][2].set_color(color=col_x)
        self.add(ax, box, eq1, eq2, eq3)

        n = 600
        xvals = np.linspace(1e-4, xmax * 1.2, n)
        xrt = np.sqrt(xvals)
        yvals1 = sp.special.jv(-1/4,xvals*xvals/2) * xrt
        yvals2 = sp.special.jv(1/4,xvals*xvals/2) * xrt

        plt1 = ax.plot_line_graph(xvals, yvals1, add_vertex_dots=False, stroke_width=6, stroke_color=YELLOW).set_z_index(5).set_z_index(2)
        plt2 = ax.plot_line_graph(xvals, yvals2, add_vertex_dots=False, stroke_width=6, stroke_color=BLUE).set_z_index(5).set_z_index(3)
        plt3 = ax.plot_line_graph(-xvals, yvals1, add_vertex_dots=False, stroke_width=6, stroke_color=YELLOW).set_z_index(5).set_z_index(2)
        plt4 = ax.plot_line_graph(-xvals, -yvals2, add_vertex_dots=False, stroke_width=6, stroke_color=BLUE).set_z_index(5).set_z_index(3)
        self.add(plt1, plt2)
        self.play(Create(plt1, rate_func=linear),
                  Create(plt2, rate_func=linear),
                  Create(plt3, rate_func=linear),
                  Create(plt4, rate_func=linear),
                  run_time=10.
                  )

        self.wait()

class LightIntensity(Scene):
    def construct(self):
        height = 1
        xmax = 4
        ymax = 1
        n = 401
        x0 = 0.75
        s = 1.5
        a = PI / x0 / 2 * 5
        ax = Axes(x_range=[-xmax, xmax], y_range=[0, ymax], x_length=7, y_length=2,
                  ).rotate(-PI/2)

        d0 = 10

        def f(x):
            return np.sinc(x)**2

        col = ManimColor(r'#ED2F32')
        path = ax.plot(f, (-xmax, xmax), stroke_color=col, stroke_width=5).set_z_index(1)
        path.set_fill(opacity=0.7, color=col)
        VGroup(ax, path).shift(-path.get_center())

        self.add(path)

class FresnelIntensity(EigenApprox):
    npol = 51
    def construct(self):
        ymax = 0.95

        n = 2**16 # must be even
        step = 2**1
        xmax = 400.
        xmax1 = 12.
        i0 = round((xmax - xmax1) / (2*xmax) * n)
        i1 = n - i0 - 1
        print(i0, i1, (i1 - i0+1)/step)

        ax = Axes(x_range=[-xmax1, xmax1], y_range=[0, ymax], x_length=7, y_length=2,
                  ).rotate(-PI / 2)
        # self.add(ax)

        xvals = np.linspace(-xmax, xmax, n)
        col = col_laser
        # path = ax.plot(f, (-xmax, xmax), stroke_color=col, stroke_width=5).set_z_index(1)
        xvals1 = np.concatenate(([-xmax], xvals, [xmax]))
        yvals1 = np.zeros(n+2)
        ax.shift(-ax.get_center())

        tVar = ValueTracker(0.08)

        dx = 2*xmax/n
        k_space = np.fft.fftfreq(n, dx) * (2 * np.pi)  # Momentum space grid
        # print(k_space)

        c = 2.
        zvals = np.sinc(k_space/PI*c)*math.sqrt(c)/dx * (1+0j)
        zvals[1::2] *= -1
        # zvals = np.zeros(n, dtype=complex)
        # zvals[-1] = 128.
        xvals1 = xvals[i0:i1:step]
        xvals2 = np.concatenate(([xvals1[0]], xvals1, [xvals1[-1]]))
        yvals2 = xvals2 * 0
        print(len(yvals1))

        t0 = 1.5
        t1 = 2.

        def obj_func():
            t = tVar.get_value()

            zvals1 = zvals * np.exp(-0.5j * k_space * k_space * t)
            yvals = np.fft.ifft(zvals1) * t
            yvals1 = np.abs(yvals[i0:i1:step])
            yvals1 /= max(yvals1)
            yvals1 *= yvals1

            yvals3 = np.sinc(xvals1/t/PI*c)**2
            yvals3 /= max(yvals3)

            if t > t0:
                yvals1 = ((t-t0) * yvals3 + (t1-t) *yvals1)/(t1-t0)

            path1 = ax.plot_line_graph(xvals1, yvals1, line_color=col, stroke_width=6, stroke_opacity=1,
                                       add_vertex_dots=False,
                                       fill_color=col, fill_opacity=0)
            yvals2[1:-1] = yvals1
            path2 = ax.plot_line_graph(xvals2, yvals2, line_color=col, stroke_width=0, stroke_opacity=0,
                                       add_vertex_dots=False,
                                       fill_color=col, fill_opacity=0.7)

            # path3 = ax.plot_line_graph(xvals1, yvals3, line_color=BLUE, stroke_width=6, stroke_opacity=1,
            #                            add_vertex_dots=False,
            #                            fill_color=col, fill_opacity=0).set_z_index(10)

            return VGroup(path1, path2)

        obj = always_redraw(obj_func)
        self.add(obj)
        self.play(tVar.animate.set_value(t1), rate_func=linear, run_time=137/30)
        self.wait(0.1)

class Latexvap(Scene):
    def construct(self):
        eq = MathTex(r'v=ap', font_size=80, stroke_width=2)
        eq[0][0].set_color(col_laser)
        eq[0][2].set_color(col_var)
        eq[0][3].set_color(col_p)
        self.add(eq)

class SlitEqn(Scene):
    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=2)
        eq1 = MathTex(r'x')
        eq2 = MathTex(r'vt')
        eq3 = MathTex(r'x + vt')
        eq4 = MathTex(r'x + apt', font_size=80, stroke_width=2)
        eq5 = MathTex(r'x + atp', font_size=80, stroke_width=2)
        eq6 = MathTex(r'x + \frac{atp}{\sqrt{\scriptstyle 1-(ap/c)^2}}', font_size=80, stroke_width=2)
        VGroup(eq1, eq2).arrange(DOWN, buff=0.75, aligned_edge=RIGHT)

        col_red = ManimColor(r'#ED2F32')
        mh.rtransform.copy_colors = True
        VGroup(eq1).set_color(col_x)
        VGroup(eq4[0][3], eq6[0][-5]).set_color(col_p)
        VGroup(eq4[0][2], eq2, eq6[0][-6]).set_color(col_var)
        VGroup(eq2[0][0], eq6[0][-3]).set_color(col_red)
        eq6[0][6:-9].set_color(col_op)
        VGroup(eq6[0][-9], eq6[0][-1]).set_color(col_num)
        print(YELLOW.to_rgb())

        eq3.move_to(eq1).align_to(eq1, RIGHT).shift(DOWN*0.5+LEFT*0.1)
        mh.align_sub(eq5, eq5[0][0], eq3[0][0]).align_to(eq1, RIGHT)
        mh.align_sub(eq4, eq4[0][0], eq5[0][0]).align_to(eq1, RIGHT)
        mh.align_sub(eq6, eq6[0][0], eq5[0][0]).align_to(eq1, RIGHT)


        self.add(eq1, eq2)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][0], eq3[0][0], eq2[0][:], eq3[0][-2:]),
                  Succession(Wait(0.4), FadeIn(eq3[0][1])))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0][:2], eq4[0][:2], eq3[0][-1], eq4[0][-1]),
                  mh.fade_replace(eq3[0][-2], eq4[0][2:-1], coor_mask=RIGHT))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][:3], eq5[0][:3], eq4[0][3], eq5[0][4],
                                eq4[0][4], eq5[0][3]))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[0][:5], eq6[0][:5], run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq6[0][5:])))
        self.wait()

class FresnelEquation(STFTWindow):
    bgcol = BLACK
    def construct(self):
        self.do_anim()

    def do_anim(self, just_eq=False):
        MathTex.set_default(font_size=70, stroke_width=1.5)
        eq1 = MathTex(r'g(x)', r'=', r'\frac{ke^{ikz} }{i z\sqrt{2\pi}}', r'\int', r'e^{\frac{ik}{2z} (x-x^\prime)^2}',
                      r'f(x^\prime)', r'\,dx^\prime')
        eq2 = Tex(r'wavenumber', r'\ $2\pi/\lambda$', font_size=60)
        eq3 = Tex(r'amplitude at slit', font_size=60, color=col_txt)
        eq4 = Tex(r'distance to screen', font_size=60, color=col_txt)
        eq5 = Tex(r'amplitude at screen', font_size=60, color=col_txt)

        eq6 = MathTex(r'g(x)', r'=', r'\frac{ke^{ikz} }{i z\sqrt{2\pi}}', r'\int', r'e^{\frac{ik}{2z} (x-y)^2}',
                      r'f(y)', r'\,dy')
        eq7 = MathTex(r'g(x)', r'=', r'\frac{ke^{ikz} }{i z\sqrt{2\pi}}', r'\int', r'e^{\frac{ik}{2z} (x^2+y^2-2xy)}',
                      r'f(y)', r'\,dy')
        eq7_1 = MathTex(r'\tan\theta', r'=', r'\frac zk', font_size=55)
        eq8 = MathTex(r'g(x)', r'\!=\!', r'\frac{ke^{ikz} }{i z\sqrt{2\pi}}', r'\!\int\!\!', r'e^{\frac i2 (x^2+y^2)\cot\theta-ixy\cot\theta}',
                      r'f(y)', r'dy')
        eq9 = MathTex(r'g(x)', r'\sim', r'\int', r'e^{\frac i2 (x^2+y^2)\cot\theta-ixy\cot\theta}',
                      r'f(y)', r'\,dy')

        eq10 = MathTex(r'\mathcal F_\theta f(x)', r'\sim', r'\int', r'e^{\frac i2(x^2+y^2)\cot\theta-ixy\csc\theta}',
                      r'f(y)', r'\,dy', font_size=60)
        eq11 = MathTex(r'\mathcal F_\theta f(x\cos\theta)', r'\!\sim\!\!\!', r'\int\!\!',
                       r'e^{\frac i2(x^2\cos^2\theta+y^2)\cot\theta-ix\cos\theta y\csc\theta}',
                      r'\!f(y)', r'dy', font_size=60)
        eq12 = MathTex(r'\mathcal F_\theta f(x\cos\theta)', r'\sim', r'\int', r'e^{\frac i2(x^2\cos^2\theta+y^2)\cot\theta-ixy\cot\theta}',
                      r'f(y)', r'dy', font_size=60)
        eq13 = MathTex(r'\mathcal F_\theta f(x\cos\theta)', r'\sim', r'e^{-\frac i2x^2\sin^2\theta\cot\theta}',
                       r'\int', r'e^{\frac i2(x^2+y^2)\cot\theta-ixy\cot\theta}',
                      r'f(y)', r'dy', font_size=60)
        eq14 = MathTex(r'\mathcal F_\theta f(x\cos\theta)', r'\sim', r'e^{-\frac i2x^2\sin\theta\cos\theta}',
                       r'\int', r'e^{\frac i2(x^2+y^2)\cot\theta-ixy\cot\theta}',
                      r'f(y)', r'dy', font_size=60)
        eq15 = MathTex(r'g(x)', r'\sim', r'e^{\frac i2 x^2\sin\theta\cos\theta}', r'\,\mathcal F_\theta f(x\cos\theta)')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][0], eq1[5][0]).set_color(col_psi)
        VGroup(eq1[0][2], eq1[2][4], eq1[2][7], eq1[4][5], eq1[4][7],
               eq7[4][8], eq7_1[2][0]).set_color(col_x)
        VGroup(eq1[4][-4:-2], eq1[5][2:4], eq1[6][1:3],
               eq6[4][-3], eq6[5][2], eq6[6][1], eq7[4][11]).set_color(col_p)
        VGroup(eq1[2][0], eq1[2][3], eq1[4][2], eq2[1][-1], eq7_1[2][-1]).set_color(col_var)
        VGroup(eq1[2][1], eq1[4][0]).set_color(col_special)
        VGroup(eq1[2][2], eq1[2][6], eq1[4][1]).set_color(col_i)
        VGroup(eq1[2][-4:-2], eq1[3], eq1[4][3], eq1[6][0]).set_color(col_op)
        VGroup(eq1[2][-2:], eq2[1][:2]).set_color(col_pi)
        VGroup(eq1[4][4], eq1[4][-1], eq7[4][13]).set_color(col_num)
        VGroup(eq2[0]).set_color(col_txt)
        VGroup(eq8[4][11:14], eq8[4][-4:-1],
               eq11[0][5:8], eq11[3][7:11], eq11[3][23:26],
               eq13[2][7:10], eq12[3][-2], eq14[2][-2], eq7_1[0][:3]).set_color(col_trig)
        VGroup(eq8[4][14], eq8[4][-1], eq10[0][1],
               eq11[0][8], eq11[3][11], eq11[3][26], eq7_1[0][3]).set_color(col_angle)
        VGroup(eq10[0][0]).set_color(col_ft)
        mh.copy_colors_eq(eq1[0][:], eq10[0][2:])

        eq2.next_to(eq1[2], UP, buff=0.5)
        eq3.next_to(eq1[5], UP, buff=0.6)
        eq4.next_to(eq1[2], DOWN, buff=0.2).shift(RIGHT*2.5)
        eq5.next_to(eq4, DOWN, buff=0.6).align_to(eq1, LEFT)
        mh.align_sub(eq6, eq6[1], eq1[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[1], eq6[1], coor_mask=UP)
        eq7_1.next_to(eq7, UP, buff=1, coor_mask=UP)
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=UP)
        mh.align_sub(eq9, eq9[1], eq8[1], coor_mask=UP)
        eq10.next_to(eq9, DOWN, coor_mask=UP)

        mh.align_sub(eq11, eq11[1], eq10[1], coor_mask=UP)
        mh.align_sub(eq12, eq12[1], eq11[1], coor_mask=UP)
        mh.align_sub(eq13, eq13[1], eq11[1], coor_mask=UP)
        eq13[:2].to_edge(LEFT, buff=0.2).shift(UP*0.55)
        eq13[2:].to_edge(RIGHT, buff=0.2).shift(DOWN*0.7)
        eq14[:2].move_to(eq13[:2])
        mh.align_sub(eq14[2:], eq14[-1], eq13[-1])
        mh.align_sub(eq15, eq15[1], eq9[1])

        circ3 = mh.circle_eq(eq15[2]).set_z_index(5).shift(DR*0.1)

        if just_eq:
            mh.rtransform(eq1[0], eq15[0], eq11[0][5:], eq15[3][5:],
                          eq10[0][:5], eq15[3][:5], eq10[3][:4], eq15[2][:4])
            eq15[2][0].set_color(col_special)
            eq15[2][1].set_color(col_i)
            eq15[2][2].set_color(col_op)
            eq15[2][3].set_color(col_num)
            eq15[2][4:6].set_color(col_x)
            VGroup(eq15[2][6:9], eq15[2][10:13]).set_color(col_trig)
            VGroup(eq15[2][9], eq15[2][13]).set_color(col_angle)
            return eq15, eq7_1, circ3

        self.add(eq1)

        line_arg = {'color': RED, 'stroke_width': 8}
        line1 = Line(eq3.get_bottom(), eq1[5][0].get_top() + RIGHT*0.2 + UP*0.1, **line_arg)
        line1_1 = Line(eq3.get_bottom(), eq6[5][0].get_top() + RIGHT*0.2 + UP*0.1, **line_arg)
        line2 = Line(eq4.get_left()+UL*0.1, eq1[2][7].get_bottom() + DR*0.1, **line_arg)
        line2_1 = Line(eq4.get_left()+UL*0.1, eq6[2][7].get_bottom() + DR*0.1, **line_arg)
        line3 = Line(eq2[0].get_bottom() + DOWN*0.1, eq1[2][0].get_top() + UL*0.1, **line_arg)
        line3_1 = Line(eq2[0].get_bottom() + DOWN*0.1, eq6[2][0].get_top() + UL*0.1, **line_arg)
        line4 = Line(eq5[0][:3].get_top() + UP*0.1, eq1[0][0].get_bottom() + DOWN*0.1, **line_arg)
        line4_1 = Line(eq5[0][:3].get_top() + UP*0.1, eq6[0][0].get_bottom() + DOWN*0.1, **line_arg)
        self.play(FadeIn(eq3, line1))
        self.wait(0.1)
        self.play(FadeIn(eq4, line2))
        self.wait(0.1)
        self.play(FadeIn(eq2, line3))
        self.wait(0.1)
        self.play(FadeIn(eq5, line4))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:4], eq6[:4], eq1[4][:9], eq6[4][:9],
                                eq1[4][-2:], eq6[4][-2:], eq1[5][:2], eq6[5][:2], eq1[5][-1], eq6[5][-1],
                                eq1[6][0], eq6[6][0],
                                line1, line1_1, line2, line2_1, line3, line3_1, line4, line4_1),
                  mh.fade_replace(eq1[4][9:11], eq6[4][9], coor_mask=RIGHT),
                  mh.fade_replace(eq1[5][2:4], eq6[5][2], coor_mask=RIGHT),
                  mh.fade_replace(eq1[6][1:3], eq6[6][1], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq6[:4], eq7[:4], eq6[5:], eq7[5:],
                                eq6[4][:8], eq7[4][:8], eq6[4][8], eq7[4][12],
                                eq6[4][9:11], eq7[4][15:17], eq6[4][9].copy(), eq7[4][10],
                                eq6[4][7].copy(), eq7[4][14]),
                  mh.rtransform(eq6[4][-1], eq7[4][8], eq6[4][-1].copy(), eq7[4][11],
                                copy_colors=False),
                  mh.fade_replace(eq6[4][8].copy(), eq7[4][9]),
                  mh.stretch_replace(eq6[4][-1].copy(), eq7[4][13], copy_colors=False),
                  FadeOut(line1_1, line2_1, line3_1, line4_1, eq2, eq3, eq4, eq5),
                  run_time=1.6)
        self.wait(0.1)
        self.play(FadeIn(eq7_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:4], eq8[:4], eq7[5:], eq8[5:], eq7[4][:2], eq8[4][:2],
                                eq7[4][3:5], eq8[4][2:4], eq7[4][6:12], eq8[4][4:10],
                                eq7[4][-1], eq8[4][10], eq7[4][12], eq8[4][15], eq7[4][14:16], eq8[4][17:19]),
                  mh.fade_replace(VGroup(eq7[4][2], eq7[4][5]), eq8[4][11:15], coor_mask=RIGHT),
                  mh.fade_replace(eq7[4][13], eq8[4][16], coor_mask=RIGHT),
                  mh.fade_replace(VGroup(eq7[4][2], eq7[4][5]), eq8[4][-4:], coor_mask=RIGHT),
                  run_time=1.8)
        self.wait(0.1)
        self.play(mh.rtransform(eq8[0], eq9[0], eq8[3:], eq9[2:]),
                  mh.fade_replace(eq8[1], eq9[1]),
                  FadeOut(eq8[2]))
        self.wait(0.1)
        mh.copy_colors_eq(eq9[2:], eq10[2:])
        self.play(FadeIn(eq10),
                  eq9.animate.shift(UP*0.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq10[0][:5], eq11[0][:5], eq10[0][-1], eq11[0][-1], eq10[1:3], eq11[1:3],
                                eq10[3][:7], eq11[3][:7], eq10[3][7:18], eq11[3][12:23], eq10[3][18:], eq11[3][27:],
                                eq10[4:], eq11[4:]),
                  Succession(Wait(0.7), FadeIn(eq11[0][5:-1], eq11[3][7:12], eq11[3][23:27])))
        self.wait(0.1)
        circ1 = mh.circle_eq(eq11[3][-7:-1]).set_z_index(5)
        self.play(Create(circ1), run_time=0.6)
        self.wait(0.1)
        eq12_1 = eq12[3][-3:-1].copy()
        self.play(mh.rtransform(eq11[:3], eq12[:3], eq11[4:], eq12[4:], eq11[3][:23], eq12[3][:23],
                                eq11[3][27], eq12[3][23]),
                  mh.rtransform(eq11[3][23:25], eq12[3][24:26], eq11[3][26], eq12[3][27]),
                  mh.rtransform(eq11[3][-4], eq12[3][-4], eq11[3][-1], eq12[3][-1]),
                  mh.fade_replace(eq11[3][25], eq12[3][26], coor_mask=RIGHT),
                  mh.fade_replace(eq11[3][-3:-1], eq12_1, coor_mask=RIGHT))
        self.remove(eq12_1)
        self.wait(0.1)
        self.play(FadeOut(circ1, rate_func=linear, run_time=0.5))
        circ2 = mh.circle_eq(eq12[3][8:11], scale=0.8).shift(RIGHT*0.15+DOWN*0.1)
        self.play(Create(circ2), run_time=0.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq12[:2], eq13[:2], eq12[2], eq13[3], eq12[4:], eq13[5:],
                                eq12[3][:7], eq13[4][:7], eq12[3][12:], eq13[4][7:]),
                  mh.rtransform(eq12[3][0].copy(), eq13[2][0], eq12[3][10:12], eq13[2][10:12],
                                eq12[3][1:4].copy(), eq13[2][2:5], eq12[3][5:7].copy(), eq13[2][5:7],
                                eq12[3][16:20].copy(), eq13[2][-4:]),
                  mh.fade_replace(eq12[3][7:10], eq13[2][7:10]),
                  FadeIn(eq13[2][1], shift=mh.diff(eq12[3][0], eq13[2][0])),
                  FadeOut(circ2, shift=mh.diff(eq12[3][7:10], eq13[2][7:10])),
                  run_time=1.8)
        self.wait(0.1)
        self.play(mh.rtransform(eq13[:2], eq14[:2], eq13[3:], eq14[3:], eq13[2][:10], eq14[2][:10],
                                eq13[2][11:14], eq14[2][10:13], eq13[2][15], eq14[2][14]),
                  mh.fade_replace(eq13[2][14], eq14[2][13], coor_mask=RIGHT),
                  FadeOut(eq13[2][10], shift=mh.diff(eq13[2][9], eq14[2][9])))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq9[:2], eq15[:2],
                                eq14[2][0], eq15[2][0], eq14[2][2:], eq15[2][1:],
                                eq14[0], eq15[3]),
                  FadeOut(eq14[2][1], shift=mh.diff(eq14[2][0], eq15[2][0])),
                  FadeOut(eq9[2:]),
                  FadeOut(eq14[3:]),
                  run_time=1.7),
                  FadeOut(eq14[1]))
        circ3 = mh.circle_eq(eq15[2]).set_z_index(5).shift(DR*0.1)
        self.play(Create(circ3, run_time=0.7, rate_func=linear))
        self.wait()

class FresnelEq2(FresnelEquation):
    trcol = BLACK
    bgcol = GREY
    def construct(self):
        eq1, eq2, circ = self.do_anim(just_eq=True)
        VGroup(eq1, eq2).set_z_index(5)
        circ.set_z_index(6)
        box1 = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1)
        box2 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                    fill_opacity=self.fill_op, corner_radius=0.15, buff=0.2)
        eq3 = eq1.copy().to_edge(DOWN, buff=0.6).to_edge(LEFT, buff=0.6).set_z_index(5)
        self.add(eq1, eq2, box1, circ, box2)
        self.play(eq2.animate.to_edge(UP, buff=0.6).to_edge(LEFT, buff=0.6),
                  FadeOut(box1),
                  mh.rtransform(eq1, eq3),
                  FadeOut(circ, shift=mh.diff(eq1, eq3)),
                  box2.animate.shift(mh.diff(eq1, eq3)),
                  run_time=1.5)
        self.wait()


class SkewRot(Scene):
    def construct(self):
        pts = []
        spacing = 2.5
        theta = 0.1
        lenx = config.frame_x_radius * 1.7
        leny = config.frame_y_radius * 1.7
        nx = 4
        ny = 9
        for i in range(-nx, nx+1):
            pts.append(LEFT * lenx + UP * spacing * i)
            pts.append(RIGHT * lenx + UP * spacing * i)
        for i in range(-ny, ny+1):
            pts.append(UP * leny + RIGHT * spacing * i)
            pts.append(DOWN * leny + RIGHT * spacing * i)

        kwargs = {'stroke_width': 5, 'stroke_color': BLUE}
        lines = VGroup(*[Line(pts[i], pts[i+1], **kwargs) for i in range(0, len(pts), 2)]).set_z_index(1)
        lines0 = lines.copy().set_z_index(0).set_opacity(0.5)
        lines.set_stroke(width=8)
        origin = ORIGIN

        theta = 42 *DEGREES
        a = math.tan(theta)
        b = math.cos(theta)


        for pt in pts:
            pt[0] += pt[1] * a
        lines2 = VGroup(*[Line(pts[i], pts[i+1], **kwargs) for i in range(0, len(pts), 2)]).set_z_index(1)
        for pt in pts:
            pt[0] *= b
        lines3 = VGroup(*[Line(pts[i], pts[i+1], **kwargs) for i in range(0, len(pts), 2)]).set_z_index(1)
        for pt in pts:
            pt[1] /= b
        lines4 = VGroup(*[Line(pts[i], pts[i+1], **kwargs) for i in range(0, len(pts), 2)]).set_z_index(1)
        for pt in pts:
            pt[1] -= pt[0] * a
        lines5 = VGroup(*[Line(pts[i], pts[i+1], **kwargs) for i in range(0, len(pts), 2)]).set_z_index(1)

        arr1 = Arrow(origin, origin + RIGHT * 3, buff=0).set_z_index(5)
        arr2 = Arrow(origin, origin + UP * 3, buff=0).set_z_index(5)
        MathTex.set_default(stroke_width=2)
        eq3 = MathTex(r'x', color=col_x, font_size=60, stroke_width=1.5).set_z_index(6).next_to(arr1, RIGHT, buff=0.1)
        eq4 = MathTex(r'p', color=col_p, font_size=60, stroke_width=1.5).set_z_index(6).next_to(arr2, UP, buff=0.1)
        eq5 = Tex(r'shear: ', r'$x^\prime=x + p\tan\theta$', font_size=70)
        eq6 = Tex(r'scale: ', r'$x^\prime=x\cos\theta$', font_size=70)
        eq7 = Tex(r'scale: ', r'$p^\prime=p/\cos\theta$', font_size=70)
        eq8 = Tex(r'shear: ', r'$p^\prime=p - x\tan\theta$', font_size=70)

        VGroup(eq5[0], eq6[0], eq7[0], eq8[0]).set_color(col_txt)
        VGroup(eq5[1][:2], eq5[1][3], eq6[1][:2], eq6[1][3], eq8[1][5]).set_color(col_x)
        VGroup(eq5[1][5], eq7[1][:2], eq7[1][3], eq8[1][:2], eq8[1][3]).set_color(col_p)
        VGroup(eq5[1][6:9], eq6[1][4:7], eq7[1][5:8], eq8[1][6:9]).set_color(col_trig)
        VGroup(eq5[1][9], eq6[1][-1], eq7[1][-1], eq8[1][-1]).set_color(col_angle)
        eq3 = eq_shadow(eq3, bg_stroke_width=15, bg_z_index=5, fg_z_index=6)
        eq4 = eq_shadow(eq4, bg_stroke_width=15, bg_z_index=5, fg_z_index=6)
        eq5 = eq_shadow(eq5, bg_stroke_width=15, bg_z_index=5, fg_z_index=6)
        eq6 = eq_shadow(eq6, bg_stroke_width=15, bg_z_index=5, fg_z_index=6)
        eq7 = eq_shadow(eq7, bg_stroke_width=15, bg_z_index=5, fg_z_index=6)
        eq8 = eq_shadow(eq8, bg_stroke_width=15, bg_z_index=5, fg_z_index=6)
        eq5.shift(DOWN*config.frame_y_radius/2)
        eq6.move_to(eq5)
        mh.align_sub(eq7, eq7[0], eq6[0])
        mh.align_sub(eq8, eq8[0], eq5[0])
        # for pt in pts:
        #     pt[0] -= pt[1] * theta
        # lines3 = VGroup(*[Line(pts[i], pts[i+1], **kwargs) for i in range(0, len(pts), 2)]).set_z_index(1)
        self.add(lines, lines0, arr1, arr2, eq3, eq4)
        self.wait(0.1)
        self.play(FadeIn(eq5, rate_func=linear, run_time=0.5))
        self.play(ReplacementTransform(lines, lines2, run_time=2.))
        self.wait(0.1)
        self.play(FadeOut(eq5), FadeIn(eq6), rate_func=linear, run_time=0.5)
        self.play(ReplacementTransform(lines2, lines3), run_time=1.)
        self.wait(0.1)
        self.play(FadeOut(eq6), FadeIn(eq7), rate_func=linear, run_time=0.5)
        self.play(ReplacementTransform(lines3, lines4), run_time=1.)
        self.wait(0.1)
        self.play(FadeOut(eq7), FadeIn(eq8), rate_func=linear, run_time=0.5)
        self.play(ReplacementTransform(lines4, lines5), run_time=2.)
        self.wait(0.1)
        self.play(FadeOut(eq8, rate_func=linear, run_time=0.5))
        arc_args = {'radius': 1.6, 'angle': -theta, 'arc_center': origin, 'stroke_width': 8, 'stroke_color': col_angle}
        arcs = [Arc(start_angle=angle, **arc_args) for angle in [0.,PI/2,PI,3*PI/2]]
        eq1 = MathTex(r'\theta', stroke_width=1.5, font_size=60, color=col_angle)
        eqs = [eq1.copy().move_to(a.get_center() * 0.8) for a in arcs]
        self.play(*[Create(arc) for arc in arcs],
                  Succession(Wait(0.5), FadeIn(*eqs)))
        # self.play(Rotate(lines0, -theta, about_point=origin), run_time=2.)
        # self.wait(0.1)
        # self.play(ReplacementTransform(lines2, lines3))
        self.wait()

