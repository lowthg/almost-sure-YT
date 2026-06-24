from manim import *
import numpy as np
import math
import sys

sys.path.append('../../')
import manimhelper as mh
from common.wigner import *

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
        eq15_1 = eq15[2][2].copy().move_to(eq15[2][-3], coor_mask=RIGHT)
        self.play(mh.rtransform(eq14[0][0], eq15[0][0], eq14[0][1:], eq15[0][2:], eq14[1], eq15[1]),
                  mh.rtransform(eq14[2][2:4], eq15[2][:2], eq14[2][-2:], eq15[2][-2:], eq14[2][1], eq15[2][-3]),
                  mh.rtransform(eq14[3], eq15[3], eq14[4][0], eq15[4][0], eq14[4][-4:], eq15[4][-8:-4],
                                eq14[5:], eq15[5:], eq14[2][0], eq15_1),
                  FadeIn(eq15[0][1], shift=mh.diff(eq14[0][0], eq15[0][0])),
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
        self.wait(0.1)
        eq19[0][1].move_to(eq15[4][16], coor_mask=RIGHT)
        self.play(FadeOut(eq15[4][16]), FadeIn(eq19[0][1]))
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

        self.wait()
