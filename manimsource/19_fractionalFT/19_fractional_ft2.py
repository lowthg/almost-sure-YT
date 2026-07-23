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

class RotateHarmonic(Scene):
    bgcol = GREY
    trcol = BLACK
    fill_op=0.7
    def __init__(self, *args, **kwargs):
        config.background_color = self.trcol if config.transparent else self.bgcol
        Scene.__init__(self, *args, **kwargs)


    def construct(self):
        self.do_anim()

    def do_anim(self, just_eq=False):
        MathTex.set_default(stroke_width=1.5, font_size=60)
        # eq4 = MathTex(r'x^\prime', r'=', r'x\cos\theta + p\sin\theta')
        # eq5 = MathTex(r'p^\prime', r'=', r'-x\sin\theta + p\cos\theta')
        #
        eq1 = MathTex(r'\mathcal F_\theta^{-1}X\mathcal F_\theta', r'=', r'X\cos\theta + P{}\sin\theta')
        eq2 = MathTex(r'\mathcal F_\theta^{-1}P\mathcal F_\theta', r'=', r'-X\sin\theta + P\cos\theta')
        eq3 = MathTex(r'\mathcal F_t^{-1}X\mathcal F_t', r'=', r'X\cos t + P{}\sin t')
        eq4 = MathTex(r'\mathcal F_t^{-1}P\mathcal F_t', r'=', r'-X\sin t + P\cos t')
        eq5 = MathTex(r'x_t', r'=', r'x_0\cos t + p_0\sin t')
        eq6 = MathTex(r'p_t', r'=', r'-x_0\sin t + p_0\cos t')
        eq7 = MathTex(r'dx_t/dt', r'=', r'p_t')
        eq8 = MathTex(r'dp_t/dt', r'=', r'-x_t')
        eq9 = MathTex(r'dx_t/dt', r'=', r'p_t/m')
        eq10 = MathTex(r'dp_t/dt', r'=', r'-kx_t')
        eq11 = MathTex(r'k', r'=', r'm', r'=', r'1', font_size=60)
        eq12 = MathTex(r'E', r'=', r'\frac1{2}\left(', r'\frac{p^2}{m} + kx^2', r'\right)')
        eq13 = MathTex(r'E', r'=', r'\frac1{2}\left(', r'p^2 + x^2', r'\right)')

        eq14 = MathTex(r'H', r'=', r'\frac1{2}\left(', r'P^2 + X^2', r'\right)', font_size=80)
        eq15 = MathTex(r'H', r'=', r'\frac1{2}\left(', r'X^2 + P^2', r'\right)', font_size=80)

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True

        VGroup(eq1[0][4], eq1[2][0], eq12[3][-2:]).set_color(col_x)
        VGroup(eq2[0][4], eq1[2][6], eq12[3][:2]).set_color(col_p)
        VGroup(eq1[2][1:4], eq1[2][7:10]).set_color(col_trig)
        VGroup(eq1[0][:3], eq1[0][-2]).set_color(col_ft)
        VGroup(eq1[0][3], eq1[0][-1], eq1[2][4], eq1[2][-1],
               eq5[2][1], eq5[2][8], eq6[2][2], eq6[2][9],
               eq7[0][-1], eq8[0][-1], eq7[2][-1], eq8[2][-1]).set_color(col_angle)
        VGroup(eq7[0][0], eq7[0][-2], eq8[0][0], eq8[0][-2], eq12[2][1]).set_color(col_op)
        VGroup(eq9[2][-1], eq10[2][-3], eq11[0], eq11[2], eq12[3][3], eq12[3][5]).set_color(RED)
        VGroup(eq11[4], eq12[2][0], eq12[2][2]).set_color(col_num)
        VGroup(eq12[0], eq14[0]).set_color(col_WVD)
        mh.copy_colors_eq(eq1[0][:4], eq2[0][:4], eq1[0][-2:], eq2[0][-2:], eq1[2][:], eq2[2][1:])
        mh.copy_colors_eq(eq1, eq3, eq2, eq4)

        gp1 = VGroup(eq1, eq2).set_z_index(4)

        eq2.to_edge(DOWN, buff=0.4)
        mh.align_sub(eq1, eq1[1], eq2[1]).next_to(eq2, UP, coor_mask=UP, buff=0.4)
        eq1[2].align_to(eq2[2][1], LEFT)
        eq2[2][6:].align_to(eq1[2][5:], LEFT)
        mh.align_sub(eq3, eq3[1], eq1[1])
        mh.align_sub(eq4, eq4[1], eq2[1])
        eq3[2].align_to(eq1[2], LEFT)
        mh.align_sub(eq3[0], eq3[0][4], eq1[0][4], coor_mask=RIGHT)
        mh.align_sub(eq4[0], eq4[0][4], eq2[0][4], coor_mask=RIGHT)
        eq3[2][5:].align_to(eq1[2][5], LEFT)
        eq4[2][6:].align_to(eq2[2][6], LEFT)
        box1 = SurroundingRectangle(gp1, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                    buff=0.2, corner_radius=0.2)
        mh.align_sub(eq5, eq5[1], eq1[1])
        mh.align_sub(eq6, eq6[1], eq2[1])
        eq5_1 = eq5.copy()
        eq6_1 = eq6.copy()
        eq5_1[2].align_to(eq6_1[2][1], LEFT)
        eq6_1[2][7:].align_to(eq5_1[2][6], LEFT)
        VGroup(eq6_1, eq5_1).to_edge(RIGHT, buff=1.5)
        mh.align_sub(eq7, eq7[1], eq5_1[1])
        mh.align_sub(eq8, eq8[1], eq6_1[1])
        eq7.align_to(eq5_1, LEFT)
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=RIGHT)
        eq7[2].align_to(eq8[2][1], LEFT)
        mh.align_sub(eq9, eq9[1], eq7[1])
        mh.align_sub(eq10, eq10[1], eq8[1])
        eq11.move_to(VGroup(eq9, eq10)).to_edge(RIGHT, buff=0.3)
        eq12.to_edge(UP, buff=0.5).shift(RIGHT*0.5)
        mh.align_sub(eq13, eq13[1], eq12[1]).move_to(eq12, coor_mask=RIGHT)
        mh.align_sub(eq15, eq15[1], eq14[1])

        if just_eq:
            eq15[0].set_color(col_WVD)
            eq15[2][:3].set_color(col_num)
            eq15[2][1].set_color(col_op)
            eq15[3][:2].set_color(col_x)
            eq15[3][-2:].set_color(col_p)
            return eq15

        self.add(eq1, eq2, box1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][:3], eq3[0][:3], eq1[0][4:-1], eq3[0][4:-1], eq1[1], eq3[1],
                                eq1[2][:4], eq3[2][:4], eq1[2][5:-1], eq3[2][5:-1],
                                eq2[0][:3], eq4[0][:3], eq2[0][4:-1], eq4[0][4:-1], eq2[1], eq4[1],
                                eq2[2][:5], eq4[2][:5], eq2[2][6:-1], eq4[2][6:-1],
                                ),
                  mh.fade_replace(eq1[0][3], eq3[0][3], coor_mask=RIGHT),
                  mh.fade_replace(eq1[0][-1], eq3[0][-1], coor_mask=RIGHT),
                  mh.fade_replace(eq1[2][4], eq3[2][4], coor_mask=RIGHT),
                  mh.fade_replace(eq1[2][-1], eq3[2][-1], coor_mask=RIGHT),
                  mh.fade_replace(eq2[0][3], eq4[0][3], coor_mask=RIGHT),
                  mh.fade_replace(eq2[0][-1], eq4[0][-1], coor_mask=RIGHT),
                  mh.fade_replace(eq2[2][5], eq4[2][5], coor_mask=RIGHT),
                  mh.fade_replace(eq2[2][-1], eq4[2][-1], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        mh.align_sub(eq5[0], eq5[0][0], eq3[0][4], coor_mask=RIGHT)
        eq5[2][2:].align_to(eq3[2][1:], LEFT)
        eq5[2][:2].move_to(eq3[2][0], coor_mask=RIGHT)
        eq5[2][6].move_to(eq3[2][5], coor_mask=RIGHT)
        eq5[2][7:9].move_to(eq3[2][6], coor_mask=RIGHT)
        eq5[2][9:].align_to(eq3[2][7:], LEFT)
        mh.align_sub(eq6[0], eq6[0][0], eq4[0][4], coor_mask=RIGHT)
        eq6[2][3:].align_to(eq4[2][2:], LEFT)
        eq6[2][1:3].move_to(eq4[2][1], coor_mask=RIGHT)
        eq6[2][7].move_to(eq4[2][6], coor_mask=RIGHT)
        eq6[2][8:10].move_to(eq4[2][7], coor_mask=RIGHT)
        eq6[2][10:].align_to(eq4[2][8:], LEFT)
        self.play(mh.rtransform(eq3[0][3], eq5[0][1], eq3[1], eq5[1], eq3[2][1:6], eq5[2][2:7], eq3[2][7:], eq5[2][9:]),
                  mh.rtransform(eq3[0][-1], eq5[0][-1]),
                  mh.stretch_replace(eq3[0][4], eq5[0][0]),
                  mh.stretch_replace(eq3[2][0], eq5[2][0]),
                  mh.stretch_replace(eq3[2][6], eq5[2][7]),
                  FadeIn(eq5[2][1], eq5[2][8]),
                  FadeOut(eq3[0][:3], eq3[0][-2]),
                  mh.rtransform(eq4[0][3], eq6[0][1], eq4[1], eq6[1], eq4[2][2:7], eq6[2][3:8], eq4[2][8:], eq6[2][10:],
                                eq4[2][0], eq6[2][0]),
                  mh.rtransform(eq4[0][-1], eq6[0][-1]),
                  mh.stretch_replace(eq4[0][4], eq6[0][0]),
                  mh.stretch_replace(eq4[2][1], eq6[2][1]),
                  mh.stretch_replace(eq4[2][7], eq6[2][8]),
                  FadeIn(eq6[2][2], eq6[2][9]),
                  FadeOut(eq4[0][:3], eq4[0][-2]),
                  )
        self.play(mh.rtransform(eq5, eq5_1, eq6, eq6_1),
                  FadeOut(box1))
        self.wait(0.1)
        self.play(mh.rtransform(eq5_1[0][:], eq7[0][1:3], eq5_1[1], eq7[1],
                                eq5_1[2][7], eq7[2][0], run_time=1.5),
                  Succession(Wait(0.5), FadeIn(eq7[0][0], eq7[0][3:])),
                  FadeOut(eq5_1[2][:7], eq5_1[2][9:]),
                  mh.fade_replace(eq5_1[2][8], eq7[2][1], coor_mask=RIGHT, run_time=1.5),
                  )
        self.play(mh.rtransform(eq6_1[0][:], eq8[0][1:3], eq6_1[1], eq8[1],
                                eq6_1[2][:2], eq8[2][:2], run_time=1.5),
                  Succession(Wait(0.5), FadeIn(eq8[0][0], eq8[0][3:])),
                  FadeOut(eq6_1[2][3:]),
                  mh.fade_replace(eq6_1[2][2], eq8[2][2], coor_mask=RIGHT, run_time=1.5),
                  )
        self.wait(0.1)
        eq7_1 = eq7.copy()
        eq8_1 = eq8.copy()
        self.play(mh.rtransform(eq7[:2], eq9[:2], eq7[2][:2], eq9[2][:2],
                                eq8[:2], eq10[:2], eq8[2][0], eq10[2][0], eq8[2][1:], eq10[2][2:]),
                  Succession(Wait(0.5), FadeIn(eq9[2][2:], eq10[2][1])))
        self.wait(0.1)
        self.play(FadeIn(eq11))
        self.wait(0.1)
        self.play(FadeIn(eq12))
        self.wait(0.1)
        eq7 = eq7_1
        eq8 = eq8_1
        self.play(Succession(Wait(0.5), mh.rtransform(eq9[:2], eq7[:2], eq9[2][:2], eq7[2][:2],
                                eq10[:2], eq8[:2], eq10[2][0], eq8[2][0], eq10[2][2:], eq8[2][1:],
                                eq12[:3], eq13[:3], eq12[-1], eq13[-1],
                                eq12[3][:2], eq13[3][:2], eq12[3][4], eq13[3][2],
                                eq12[3][-2:], eq13[3][-2:])),
                  FadeOut(eq9[2][2:], eq10[2][1], eq12[3][2:4], eq12[3][5]),
                  FadeOut(eq11))
        self.wait(0.1)
        self.play(mh.rtransform(eq13[1:3], eq14[1:3], eq13[3][1:3], eq14[3][1:3],
                                eq13[3][-1], eq14[3][-1], eq13[-1], eq14[-1]),
                  mh.stretch_replace(eq13[3][-2], eq14[3][-2]),
                  mh.stretch_replace(eq13[3][0], eq14[3][0]),
                  mh.fade_replace(eq13[0], eq14[0]),
                  FadeOut(eq7, eq8),
                  run_time=1.6
                  )
        self.play(mh.rtransform(eq14[:3], eq15[:3], eq14[-1], eq15[-1],
                                eq14[3][:2], eq15[3][-2:], eq14[3][-2:], eq15[3][:2],
                                eq14[3][2], eq15[3][2]))

        self.wait()


class RotateHarmonic2(RotateHarmonic):
    bgcol = BLACK
    def construct(self):
        ax = Axes(x_range=[-1.05, 1.1], y_range=[-1.05, 1.1], x_length=6, y_length=6,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).to_edge(LEFT, buff=0.4)
        p1 = (0.9*0.8, 0.8*0.8)
        pt1 = ax.coords_to_point(*p1)
        origin = ax.coords_to_point(0,.0)
        dot1 = Dot(pt1, radius=0.15, color=ORANGE).set_z_index(5)
        dir1 = pt1 - origin
        line1 = Line(origin + dir1 * 0.02, pt1, stroke_color=BLUE, stroke_width=8).set_z_index(2)
        dot2 = Dot(origin, radius=0.06, color=BLUE).set_z_index(5)

        eq2 = MathTex(r'(x,p)', font_size=60, stroke_width=1.5).next_to(pt1, UR, buff=0.15)
        VGroup(eq2[0][1]).set_color(col_x)
        VGroup(eq2[0][3]).set_color(col_p)
        eq3 = MathTex(r'x', font_size=60, stroke_width=1.5, color=col_x).next_to(ax.x_axis, RIGHT, buff=0.1)
        eq4 = MathTex(r'p', font_size=60, stroke_width=1.5, color=col_p).next_to(ax.y_axis, UP, buff=0.1)

        dl = VGroup(dot1, line1)
        self.add(ax, line1, dot1, eq2, eq3, eq4, dot2)
        self.wait(0.1)

        period = 3.
        frac = 0.25

        self.play(Rotate(dl, -2*PI*frac, about_point=origin, rate_func=lambda t: t * t, run_time=period*frac*2),
                  FadeOut(eq2))
        self.play(Rotate(dl, -2*PI*(1-frac), about_point=origin, rate_func=linear, run_time=period * (1-frac)))
        self.play(Rotate(dl, -2*PI, about_point=origin, rate_func=linear, run_time=period))

class RotateHarmonicEqs(RotateHarmonic):
    bgcol = BLACK
    def construct(self):
        eq1 = self.do_anim(just_eq=True)
        MathTex.set_default(font_size=80)
        eq2 = MathTex(r'i\frac{d}{dt}f_t(x)', r'=', r'Hf_t(x)', font_size=80)
        eq2_1 = MathTex(r'f(x)', font_size=80)
        eq3 = MathTex(r'f_t(x)', r'=', r'e^{-iHt}', r'f_0(x)')
        eq4 = MathTex(r'\frac d{dt}', r'e^{-iHt}', r'=', r'-iH', r'e^{-iHt}', font_size=70)
        eq5 = MathTex(r'X', r'f_t(x)', r'=', r'X', r'e^{-iHt}', r'f_0(x)')
        eq6 = MathTex(r'X', r'f_t(x)', r'=', r'e^{-iHt}', r'\left(', r'e^{iHt}', r'X',
                      r'e^{-iHt}', r'\right)', r'f_0(x)')
        eq7 = MathTex(r'e^{iHt}', r'X', r'e^{-iHt}', r'=', r'(1+iHt)', r'X', r'(1-iHt)', r'+O(t^2)', font_size=80)
        mh.font_size_sub(eq7, -1, 60)
        eq8 = MathTex(r'=', r'X+iHtX-XiHt')
        eq9 = MathTex(r'=', r'X+i(HX-XH)t')
        eq10 = MathTex(r'=', r'X+i[H,X]t')
        eq11 = MathTex(r'e^{iHt}', r'X', r'e^{-iHt}', r'=', r'X+i[H,X]t', r'+O(t^2)')
        eq12 = MathTex(r'[H,X]', r'=', r'\frac12[X^2,X]', r'+', r'\frac12[P^2,X]')
        eq13 = MathTex(r'[H,X]', r'=', r'\frac12[PP,X]')
        eq14 = MathTex(r'[H,X]', r'=', r'\frac12P[P,X]', r'+', r'\frac12[P,X]P')

        eq15 = MathTex(r'\frac{d}{dx}(xf(x))', r'=', r'\frac{dx}{dx}f(x)', r'+', r'x\frac{d}{dx}f(x)', font_size=70)
        eq16 = MathTex(r'DXf(x)', r'=', r'1f(x)', r'+', r'XDf(x)', font_size=70)
        eq17 = MathTex(r'DX-XD', r'=', r'1', font_size=70)
        eq18 = MathTex(r'[D,X]', r'=', r'1')
        eq19 = MathTex(r'[-iD,X]', r'=', r'-i')
        eq20 = MathTex(r'[P,X]', r'=', r'-i')

        VGroup(eq2[2][0]).set_color(col_WVD)
        VGroup(eq2_1[0][0], eq15[0][-5], eq15[2][-4], eq15[4][-4]).set_color(col_psi)
        VGroup(eq2_1[0][2], eq5[0], eq5[3],
               eq15[0][3], eq15[0][5], eq15[0][8], eq15[2][1], eq15[2][4], eq15[2][7], eq15[4][0], eq15[4][4], eq15[4][7]
               ).set_color(col_x)
        VGroup(eq16[0][0], eq16[4][1], eq20[0][1]).set_color(col_p)
        VGroup(eq2[0][1:4], eq7[-1][1],
               eq15[0][:3], eq15[2][0], eq15[2][2:4], eq15[4][1:4]).set_color(col_op)
        VGroup(eq2[0][0], eq19[0][2], eq19[2][1]).set_color(col_i)
        VGroup(eq2[0][6], eq2[2][2], eq2[0][4], eq3[3][1], eq7[-1][3:5]).set_color(col_angle)
        VGroup(eq3[2][0]).set_color(col_special)
        VGroup(eq7[4][1], eq7[6][1], eq16[2][0]).set_color(col_num)
        mh.copy_colors_eq(eq2[0][1:5], eq4[0][:])

        eq2.next_to(eq1, DOWN, buff=0.5)
        gp1 = VGroup(eq1.copy(), eq2).move_to(ORIGIN, coor_mask=UP)
        mh.align_sub(eq2_1, eq2_1[0], eq2[0][5], coor_mask=UP)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=UP)
        eq4.next_to(eq3, DOWN, buff=0.3)
        mh.align_sub(eq5, eq5[2], eq3[1])
        mh.align_sub(eq6, eq6[2], eq3[1], coor_mask=UP)
        eq7[:-1].next_to(eq6, DOWN, buff=0.7)
        eq7[-1].next_to(eq7[-2], DOWN, buff=0.4).align_to(eq7[-2], RIGHT)
        gp2 = VGroup(gp1[0].copy(), eq6.copy(), eq7)
        mh.align_sub(gp2, gp2[:2] + gp2[2][:-1], ORIGIN, coor_mask=UP)
        mh.align_sub(eq8, eq8[0], eq7[3])
        mh.align_sub(eq9, eq9[0], eq7[3])
        mh.align_sub(eq10, eq10[0], eq7[3])
        eq10[1][3:8].move_to(eq9[1][3:10], coor_mask=RIGHT)
        eq10[1][-1].move_to(eq9[1][-1], coor_mask=RIGHT)
        mh.align_sub(eq11, eq11[3], gp2[1][2], coor_mask=UP)
        eq12.next_to(eq11, DOWN, buff=0.5)
        mh.align_sub(eq13, eq13[1], eq12[1])
        eq13[2].align_to(eq12[4], RIGHT)
        mh.align_sub(eq14, eq14[1], eq12[1], coor_mask=UP)
        eq15.next_to(eq14, DOWN, buff=0.1)

        self.add(eq1)
        self.play(Succession(Wait(0.5), FadeIn(eq2_1)),
                  mh.rtransform(eq1, gp1[0]))
        eq1 = gp1[0]
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1[0][0], eq2[0][5], eq2_1[0][1:4], eq2[0][7:10],
                                eq2_1[0][0].copy(), eq2[2][1], eq2_1[0][1:4].copy(), eq2[2][3:6],
                                run_time=1.6),
                  Succession(Wait(1.), FadeIn(eq2[0][:5], eq2[0][6], eq2[1], eq2[2][0], eq2[2][2]))
                  )
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq2[0][-5:], eq3[0][:], eq2[1], eq3[1], eq2[2][0], eq3[2][3],
                                eq2[2][1], eq3[3][0], eq2[2][3:6], eq3[3][2:5],
                                eq2[0][4], eq3[2][4]),
                  mh.fade_replace(eq2[2][2], eq3[3][1]),
                  mh.stretch_replace(eq2[0][0], eq3[2][2]),
                  FadeIn(eq3[2][1], shift=mh.diff(eq2[0][0], eq3[2][2])),
                                 run_time=1.6),
                  FadeOut(eq2[0][1:4]),
                  Succession(Wait(0.9), FadeIn(eq3[2][0]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq3[2].copy(), eq4[1], eq3[2].copy(), eq4[-1],
                                run_time=1.4),
                  Succession(Wait(0.6), FadeIn(eq4[2])))
        self.play(FadeIn(eq4[0]),
                  mh.stretch_replace(eq4[4][1:-1].copy().set_z_index(5), eq4[3][:].set_z_index(5), run_time=1))
        self.wait(0.1)
        self.play(FadeOut(eq4))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq5[1:3], eq3[2:], eq5[4:], run_time=1.4),
                  Succession(Wait(0.5), FadeIn(eq5[0], eq5[3], run_time=1)))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:3], eq6[:3], eq5[-3:-1], eq6[-4:-2], eq5[-1], eq6[-1]),
                  run_time=1.5)
        eq6_1 = eq6[-3].copy()
        self.play(mh.rtransform(eq6_1.copy(), eq6[3], eq6_1[0], eq6[5][0], eq6_1[2:], eq6[5][1:], run_time=1.4),
                  Succession(Wait(1), FadeIn(eq6[4], eq6[-2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[5:8].copy(), eq7[:3], run_time=1.5),
                  Succession(Wait(1), FadeIn(eq7[3])),
                  mh.rtransform(eq1, gp2[0], eq6, gp2[1]),
                  )
        eq1 = gp2[0]
        eq6 = gp2[1]
        eq6_2 = eq6.copy()
        txt2 = Tex(r'\sf negligible terms', color=RED, font_size=55).next_to(eq7[-1], LEFT, buff=0.5).to_edge(DOWN, buff=0.2)
        arr1 = Arrow(txt2.get_right(), eq7[-1][1].get_left()+DR*0.1, buff=0.1, stroke_width=6, color=RED).set_z_index(7)
        self.play(AnimationGroup(mh.rtransform(eq6_2[6], eq7[5], eq6_2[5][2:], eq7[4][4:-1],
                                eq6_2[7][1], eq7[6][2], eq6_2[7][3:], eq7[6][4:-1]),
                  mh.stretch_replace(eq6_2[5][1], eq7[4][3]),
                  mh.stretch_replace(eq6_2[7][2], eq7[6][3]),
                  FadeIn(eq7[4][1:3], shift=mh.diff(eq6_2[5][1], eq7[4][3])),
                  FadeIn(eq7[6][1], shift=mh.diff(eq6_2[7][1], eq7[6][3])),
                  run_time=1.6),
                  Succession(Wait(1.2), FadeIn(eq7[4][0], eq7[4][-1], eq7[6][0], eq7[6][-1])),
                  Succession(Wait(1.8), FadeIn(eq7[-1], txt2, arr1))
                  )
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq7[5][0], eq8[1][0], eq7[4][2:6], eq8[1][1:5],
                                eq7[5][0].copy(), eq8[1][5], eq7[6][2], eq8[1][6],
                                eq7[6][3:6], eq8[1][8:11], eq7[5][0].copy(), eq8[1][7]),
                  FadeOut(eq7[4][1], target_position=eq8[1][0]),
                  FadeOut(eq7[6][1], target_position=eq8[1][0]),
                                 FadeOut(txt2, arr1, rate_func=linear),
                                 run_time=1.6),
                  FadeOut(eq7[4][0], eq7[4][-1], eq7[6][0], eq7[6][-1]),
                  )
        self.play(AnimationGroup(
            mh.rtransform(eq8[1][:3], eq9[1][:3], eq8[1][3], eq9[1][4], eq8[1][5], eq9[1][5],
                          eq8[1][4], eq9[1][-1], eq8[1][6:8], eq9[1][6:8], eq8[1][9], eq9[1][8]),
            mh.rtransform(eq8[1][8], eq9[1][2], eq8[1][10], eq9[1][-1]),
            run_time=1.8),
            Succession(Wait(1.2), FadeIn(eq9[1][3], eq9[1][-2]))
        )
        self.wait(0.1)
        circ = mh.circle_eq(eq9[1][4:-2]).set_z_index(6).shift(DOWN*0.1)
        txt1 = Tex(r'\sf commutator', color=RED, font_size=70).next_to(circ, DOWN, buff=0.2).shift(LEFT).set_z_index(8)
        self.play(Create(circ, rate_func=linear, run_time=0.6),
                  Succession(Wait(0.4), FadeIn(txt1)))
        self.wait(0.1)
        self.play(mh.stretch_replace(eq9[1][3], eq10[1][3]),
                  mh.stretch_replace(eq9[1][9], eq10[1][7]),
                  mh.rtransform(eq9[1][4], eq10[1][4], eq9[1][5], eq10[1][6]),
                  mh.rtransform(eq9[1][7], eq10[1][6], eq9[1][8], eq10[1][4]),
                  mh.fade_replace(eq9[1][6], eq10[1][5], coor_mask=RIGHT),
                  mh.rtransform(eq9[1][:3], eq10[1][:3], eq9[1][-1], eq10[1][-1]),
                  run_time=1.4
                  )
        self.wait(0.1)
        self.play(FadeOut(eq6),
                  AnimationGroup(mh.rtransform(eq7[:4], eq11[:4], eq10[1], eq11[4], eq7[-1], eq11[-1]),
                                 FadeOut(circ, txt1, shift=mh.diff(eq10[1][3:8], eq11[4][3:8])),
                                run_time=1.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[4][3:8].copy(), eq12[0][:], run_time=1.5),
                  Succession(Wait(0.6), FadeIn(eq12[1])))
        eq1_1 = eq1[2:].copy().set_z_index(6)
        eq12.set_z_index(6)
        self.play(
            mh.rtransform(eq12[0][0].copy(), eq12[2][3], eq12[0][0].copy(), eq12[4][3],
                          eq12[0][2:].copy(), eq12[2][6:], eq12[0][2:].copy(), eq12[4][6:], run_time=1.6),
            Succession(Wait(1), mh.rtransform(eq1_1[0][:3], eq12[2][:3], eq1_1[0][:3].copy(), eq12[4][:3],
                                eq1_1[1][:2], eq12[2][4:6], eq1_1[1][2], eq12[3][0],
                                eq1_1[1][3:5], eq12[4][4:6], run_time=2)))
        self.wait(0.1)
        line1 = Line(eq12[2].get_corner(DL), eq12[2].get_corner(UR), color=RED, stroke_width=8).set_z_index(10)
        self.play(Create(line1, run_time=0.6, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(line1, eq12[2:4]))
        self.wait(0.1)
        self.play(mh.rtransform(eq12[:2], eq13[:2], eq12[4][:5], eq13[2][:5], eq12[4][4].copy(), eq13[2][5],
                                eq12[4][6:], eq13[2][6:]),
                  FadeOut(eq12[4][5]))
        self.play(mh.rtransform(eq13[:2], eq14[:2], eq13[2][:3].copy(), eq14[2][:3], eq13[2][:5], eq14[4][:5],
                                eq13[2][4].copy(), eq14[2][3], eq13[2][3].copy(), eq14[2][4],
                                eq13[2][5:9].copy(), eq14[2][5:9], eq13[2][5], eq14[4][-1],
                                eq13[2][6:9], eq14[4][5:8], run_time=1.8),
                  Succession(Wait(1.2), FadeIn(eq14[3])))
        self.wait(0.1)
        gp3 = VGroup(eq1, eq11, eq14, eq15)
        gp3_ = gp3.copy().move_to(ORIGIN, coor_mask=UP)
        self.play(mh.rtransform(gp3[:-1], gp3_[:-1]),
                  Succession(Wait(0.5), FadeIn(gp3_[-1])))
        eq15 = gp3_[-1]
        self.wait(0.1)
        mh.align_sub(eq16, eq16[1], eq15[1])
        eq16[0][-4:].move_to(eq15[0][-5:-1], coor_mask=RIGHT)
        eq16[0][0].move_to(eq15[0][:4], coor_mask=RIGHT)
        eq16[0][1].next_to(eq16[0][2], LEFT, buff=0.1, coor_mask=RIGHT)
        eq16[2][0].move_to(eq15[2][:5], coor_mask=RIGHT)
        eq16[2][-4:].move_to(eq15[2][-4:], coor_mask=RIGHT)
        eq16[3].move_to(eq15[3], coor_mask=RIGHT)
        eq16[4][1].move_to(eq15[4][1:5], coor_mask=RIGHT)
        eq16[4][-4:].move_to(eq15[4][-4:], coor_mask=RIGHT)
        eq16[4][0].next_to(eq16[4][1], LEFT, coor_mask=RIGHT, buff=0.1)
        mh.align_sub(eq17, eq17[1], eq16[1], coor_mask=UP)
        mh.align_sub(eq18, eq18[1], eq17[1], coor_mask=UP)
        mh.align_sub(eq19, eq19[1], eq18[1])
        mh.align_sub(eq20, eq20[1], eq18[1])

        self.play(
            FadeOut(eq15[0][4], eq15[0][-1]),
            mh.rtransform(eq15[1], eq16[1], eq15[0][-5:-1], eq16[0][-4:],
                          eq15[2][-4:], eq16[2][-4:],
                          eq15[3], eq16[3],
                          eq15[4][-4:], eq16[4][-4:]),
            mh.fade_replace(eq15[0][:4], eq16[0][0], coor_mask=RIGHT),
            mh.stretch_replace(eq15[0][5], eq16[0][1]),
            mh.fade_replace(eq15[2][:5], eq16[2][0]),
            mh.stretch_replace(eq15[4][0], eq16[4][0]),
            mh.fade_replace(eq15[4][1:5], eq16[4][1], coor_mask=RIGHT),
        )
        self.wait(0.1)
        self.play(FadeOut(eq16[0][-4:], eq16[2][-4:], eq16[4][-4:]),
                  AnimationGroup(mh.rtransform(eq16[0][:2], eq17[0][:2], eq16[1], eq17[1], eq16[2][0], eq17[2][0],
                                eq16[4][:2], eq17[0][-2:]),
                  mh.fade_replace(eq16[3], eq17[0][2]),
                  run_time=1.7))
        self.play(mh.rtransform(eq17[1:], eq18[1:], eq17[0][0], eq18[0][1], eq17[0][1], eq18[0][3]),
                  mh.rtransform(eq17[0][3], eq18[0][3], eq17[0][4], eq18[0][1]),
                  mh.fade_replace(eq17[0][2], eq18[0][2], coor_mask=RIGHT),
                  FadeIn(eq18[0][0], eq18[0][-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq18[0][0], eq19[0][0], eq18[0][1:], eq19[0][3:], eq18[1], eq19[1]),
                  mh.fade_replace(eq18[2], eq19[2], coor_mask=RIGHT),
                  FadeIn(eq19[0][1:3]))
        self.wait(0.1)
        self.play(mh.rtransform(eq19[0][0], eq20[0][0], eq19[0][4:], eq20[0][2:], eq19[1:], eq20[1:]),
                  FadeOut(eq19[0][1:3]),
                  mh.fade_replace(eq19[0][3], eq20[0][1], coor_mask=RIGHT))

        self.wait()
