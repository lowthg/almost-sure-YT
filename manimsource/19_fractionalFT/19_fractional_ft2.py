import math

import numpy as np
from manim import *
import sys
import scipy as sp
from manim import ManimColor
from torch.utils.jit.log_extract import run_test

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
        self.do_anim()

    def do_anim(self, just_eq=False):
        eq1 = RotateHarmonic.do_anim(self, just_eq=True)
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
        eq21 = MathTex(r'[H,X]', r'=', r'-\frac12Pi', r'-', r'\frac12iP')
        eq22 = MathTex(r'[H,X]', r'=', r'-iP')

        eq23 = MathTex(r'e^{iHt}', r'X', r'e^{-iHt}', r'=', r'X+Pt', r'+O(t^2)')
        eq24 = MathTex(r'e^{iHt}', r'P', r'e^{-iHt}', r'=', r'P+i[H,P]t', r'+O(t^2)')

        eq25 = MathTex(r'[H,P]', r'=', r'\frac12X[X,P]', r'+', r'\frac12[X,P]X')
        eq26 = MathTex(r'[X,P]', r'=', r'i')
        eq27 = MathTex(r'[H,P]', r'=', r'iX')
        eq28 = MathTex(r'e^{iHt}', r'P', r'e^{-iHt}', r'=', r'P-Xt', r'+O(t^2)')

        eq29 = MathTex(r'e^{iH\delta t}', r'X', r'e^{-iH\delta t}', r'=', r'X+P\delta t', r'+O(\delta t^2)')
        eq30 = MathTex(r'e^{iH\delta t}', r'P', r'e^{-iH\delta t}', r'=', r'P-X\delta t', r'+O(\delta t^2)')

        eq31 = MathTex(r'e^{iHt}', r'X', r'e^{-iH t}', r'=', r'X\cos t+P\sin t')
        eq32 = MathTex(r'e^{iHt}', r'P', r'e^{-iH t}', r'=', r'P\cos t-X\sin t')

        eq32_1 = MathTex(r'\mathcal F_t', r'=', r'e^{-iHt}')

        eq33_1 = MathTex(r'f(x)', r'=', r'e^{-\frac12x^2}')
        eq33 = MathTex(r'D e^{-\frac12 x^2}', r'=', r'-xe^{-\frac12x^2}')
        eq34 = MathTex(r'D^2 e^{-\frac12 x^2}', r'=', r'(x^2-1)e^{-\frac12x^2}')
        eq35 = MathTex(r'(X^2-D^2) e^{-\frac12 x^2}', r'=', r'1\,e^{-\frac12x^2}')
        eq36 = MathTex(r'(X^2+P^2) e^{-\frac12 x^2}', r'=', r'1\,e^{-\frac12x^2}')
        eq37 = MathTex(r'H e^{-\frac12 x^2}', r'=', r'\frac12e^{-\frac12x^2}')
        eq38 = MathTex(r'e^{-iHt}', r'e^{-\frac12 x^2}', r'=', r'e^{-\frac12it}', r'e^{-\frac12x^2}')
        eq39 = MathTex(r'e^{-i(H-\frac12)t}', r'e^{-\frac12 x^2}', r'=', r'e^{-\frac12x^2}')

        eq40 = MathTex(r'H', r'=', r'\frac1{2}\left(', r'X^2 + P^2-1', r'\right)', font_size=80)
        eq41 = MathTex(r'H', r'e^{-\frac12x^2}', r'=', r'0')

        eq42 = MathTex(r'H', r'=', r'\frac1{2}\left(', r'X^2 - D^2-1', r'\right)', font_size=80)

        VGroup(eq2[2][0], eq32_1[2][3]).set_color(col_WVD)
        VGroup(eq2_1[0][0], eq15[0][-5], eq15[2][-4], eq15[4][-4],
               eq33_1[0][0]).set_color(col_psi)
        VGroup(eq2_1[0][2], eq5[0], eq5[3],
               eq15[0][3], eq15[0][5], eq15[0][8], eq15[2][1], eq15[2][4], eq15[2][7], eq15[4][0], eq15[4][4], eq15[4][7],
               eq25[2][3], eq25[2][5], eq25[4][4], eq25[4][-1],
               eq33_1[2][-2:], eq33[2][1], eq33_1[0][2]).set_color(col_x)
        VGroup(eq16[0][0], eq16[4][1], eq20[0][1],
               eq25[0][3], eq25[2][-2], eq25[4][-3],
               eq33[0][0], eq42[3][3]).set_color(col_p)
        VGroup(eq2[0][1:4], eq7[-1][1],
               eq15[0][:3], eq15[2][0], eq15[2][2:4], eq15[4][1:4],
               eq33_1[2][3], eq37[2][1]).set_color(col_op)
        VGroup(eq2[0][0], eq19[0][2], eq19[2][1], eq38[0][2], eq38[3][-2], eq32_1[2][2]).set_color(col_i)
        VGroup(eq2[0][6], eq2[2][2], eq2[0][4], eq3[3][1], eq7[-1][3:5],
               eq31[4][4], eq32[4][4], eq38[0][-1], eq38[3][-1], eq32_1[0][1], eq32_1[2][-1]).set_color(col_angle)
        VGroup(eq3[2][0], eq33[0][1], eq33_1[2][0], eq38[0][0], eq38[3][0], eq32_1[2][0]).set_color(col_special)
        VGroup(eq7[4][1], eq7[6][1], eq16[2][0], eq33_1[2][2], eq33_1[2][4],
               eq37[2][2], eq40[3][-1], eq42[3][-1]).set_color(col_num)
        VGroup(eq31[4][1:4], eq31[4][7:10], eq32[4][1:4], eq32[4][7:10]).set_color(col_trig)
        VGroup(eq32_1[0][0]).set_color(col_ft)
        mh.copy_colors_eq(eq2[0][1:5], eq4[0][:])
        mh.copy_colors_eq(eq33[0][1:], eq33[2][2:])

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

        if just_eq:
            eq1 = gp2[0]
            VGroup(eq1, eq11, eq14, eq15).move_to(ORIGIN, coor_mask=UP)
            mh.align_sub(eq40, eq40[1], eq1[1], coor_mask=UP)
            rect2 = SurroundingRectangle(eq40, fill_opacity=0, stroke_color=RED, stroke_opacity=1, stroke_width=8,
                                         corner_radius=0.2, buff=0.2)
            eq42.move_to(eq40)
            gp5 = VGroup(eq42, rect2).to_edge(LEFT, buff=0.55)
            rect3 = SurroundingRectangle(eq32_1, fill_opacity=0, stroke_width=8, stroke_color=RED,
                                         buff=0.2, corner_radius=0.15)
            gp6 = VGroup(eq32_1, rect3).move_to(gp5).to_edge(RIGHT, buff=0.55)
            mh.copy_colors_eq(eq1[:3], eq42[:3], eq1[3][:], eq42[3][:-2])
            return gp5, gp6

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
        eq14 = gp3_[2]
        eq11 = gp3_[1]
        eq1 = gp3_[0]
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
        mh.align_sub(eq21, eq21[1], eq14[1])
        mh.align_sub(eq22, eq22[1], eq21[1], coor_mask=UP)
        mh.align_sub(eq23, eq23[3], eq11[3], coor_mask=UP)
        mh.align_sub(eq24, eq24[3], eq23[3], coor_mask=UP)
        eq24.next_to(eq23, DOWN, buff=0.4, coor_mask=UP)
        eq25.next_to(eq25, DOWN, buff=0.4)
        mh.align_sub(eq27, eq27[1], eq25[1], coor_mask=UP)
        mh.align_sub(eq28, eq28[3], eq24[3], coor_mask=UP)
        mh.align_sub(eq29, eq29[3], eq23[3], coor_mask=UP)
        mh.align_sub(eq30, eq30[3], eq28[3], coor_mask=UP)
        mh.align_sub(eq31, eq31[3], eq29[3], coor_mask=UP)
        mh.align_sub(eq32, eq32[3], eq30[3], coor_mask=UP)

        eq33.next_to(eq32, DOWN, buff=0.7, coor_mask=UP)
        mh.align_sub(eq32_1, eq32_1, eq33[1], coor_mask=UP)
        mh.align_sub(eq33_1, eq33_1[1], eq33[1], coor_mask=UP)
        mh.align_sub(eq34, eq34[1], eq33[1])
        mh.align_sub(eq35, eq35[1], eq34[1], coor_mask=UP)
        mh.align_sub(eq36, eq36[1], eq35[1])
        mh.align_sub(eq37, eq37[1], eq36[1])
        mh.align_sub(eq38, eq38[2], eq37[1], coor_mask=UP)
        mh.align_sub(eq39, eq39[2], eq38[2], coor_mask=UP)
        mh.align_sub(eq40, eq40[1], eq1[1], coor_mask=UP)
        mh.align_sub(eq41, eq41[2], eq39[2], coor_mask=UP)

        rect1 = SurroundingRectangle(eq20, fill_opacity=0, stroke_color=RED,
                                     stroke_width=8, corner_radius=0.15, buff=0.2)

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
                  mh.fade_replace(eq19[0][3], eq20[0][1], coor_mask=RIGHT),
                  Succession(Wait(0.5), FadeIn(rect1))
                  )
        self.wait(0.1)
        eq21_1 = eq21[3].copy()
        eq14_1 = eq14.copy()
        self.play(AnimationGroup(
            mh.rtransform(eq14[:2], eq21[:2], eq14[2][:4], eq21[2][1:5],
                          eq20[2][0].copy(), eq21[2][0], eq20[2][-1].copy(), eq21[2][-1],
                          eq14[4][:3], eq21[4][:3], eq14[4][-1], eq21[4][-1],
                          eq20[2][0].copy(), eq21[3][0], eq20[2][-1].copy(), eq21[4][3]
                          ),
            mh.fade_replace(eq14[3], eq21_1),
            run_time=1.7),
            FadeOut(eq14[2][4:], eq14[4][3:-1]),
                  )
        self.remove(eq21_1)
        self.wait(0.1)
        self.play(mh.rtransform(eq21[:2], eq22[:2], eq21[2][0], eq22[2][0],
                                eq21[2][4], eq22[2][2], eq21[2][5], eq22[2][1]),
                  mh.rtransform(eq21[3][0], eq22[2][0], eq21[4][3:], eq22[2][1:]),
                  FadeOut(eq21[4][:3]),
                  FadeOut(eq21[2][1:4], shift=mh.diff(eq21[2][4], eq22[2][1])*RIGHT),
                  run_time=1.4)
        self.wait(0.1)
        eq22_1 = eq22[2]
        self.play(FadeOut(eq11[4][2:-1]),
                  eq22_1[2].animate.move_to(eq11[4][3:-1]).align_to(eq11[4][4], DOWN),
                  FadeOut(eq22_1[:2], target_position=eq11[4][2]),
                  FadeOut(eq22[:2]),
                  run_time=1.3)
        self.play(mh.rtransform(eq11[:4], eq23[:4], eq11[4][:2], eq23[4][:2],
                                eq22_1[2], eq23[4][2], eq11[4][-1], eq23[4][-1],
                                eq11[-1], eq23[-1]))
        self.wait(0.1)
        mh.copy_colors_eq(eq11, eq24)
        VGroup(eq24[1], eq24[4][0], eq24[4][6]).set_color(col_p)
        self.play(FadeIn(eq24))
        mh.align_sub(eq14_1, eq14_1[1], eq25[1])
        self.wait(0.1)
        gp4 = VGroup(eq20, rect1)
        self.play(FadeIn(eq14_1),
                  gp4.animate.scale(0.8).to_edge(DOWN, buff=0.1).shift(LEFT*0.25))
        eq26.scale(0.8)
        mh.align_sub(eq26, eq26[1], eq20[1])
        self.wait(0.1)
        self.play(mh.rtransform(eq14_1[0][:3], eq25[0][:3], eq14_1[0][-1], eq25[0][-1],
                                eq14_1[1], eq25[1],
                                eq14_1[2][:3], eq25[2][:3], eq14_1[2][4], eq25[2][4],
                                eq14_1[2][6], eq25[2][6], eq14_1[3], eq25[3],
                                eq14_1[2][8], eq25[2][8],
                                eq14_1[4][:4], eq25[4][:4],
                                eq14_1[4][5], eq25[4][5],
                                eq14_1[4][7], eq25[4][7]),
                  mh.fade_replace(eq14_1[0][3], eq25[0][3]),
                  mh.fade_replace(eq14_1[2][3], eq25[2][3]),
                  mh.fade_replace(eq14_1[2][5], eq25[2][5]),
                  mh.fade_replace(eq14_1[2][7], eq25[2][7]),
                  mh.fade_replace(eq14_1[4][4], eq25[4][4]),
                  mh.fade_replace(eq14_1[4][6], eq25[4][6]),
                  mh.fade_replace(eq14_1[4][8], eq25[4][8])
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq20[0][0], eq26[0][0], eq20[0][1], eq26[0][3],
                                eq20[0][2], eq26[0][2], eq20[0][3], eq26[0][1],
                                eq20[0][4], eq26[0][4], eq20[1], eq26[1],
                                eq20[2][1], eq26[2][0]),
                  FadeOut(eq20[2][0]))
        eq26_1 = MathTex(r'=', r'i', r'i')
        mh.align_sub(eq26_1, eq26_1[0], eq25[1])
        eq26_1[1].move_to(eq25[2][4:], coor_mask=RIGHT)
        eq26_1[2].move_to(eq25[4][3:-1], coor_mask=RIGHT)
        self.wait(0.1)
        self.play(mh.rtransform(eq26[2][0].copy(), eq26_1[1],
                                eq26[2][0].copy(), eq26_1[2]),
                  FadeOut(eq25[2][4:], eq25[4][3:-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq25[:2], eq27[:2], eq25[2][3], eq27[2][1],
                                eq26_1[1], eq27[2][0]),
                  mh.rtransform(eq25[4][-1], eq27[2][1], eq26_1[2], eq27[2][0]),
                  FadeOut(eq25[2][:3], shift=mh.diff(eq25[2][3], eq27[2][0])*RIGHT),
                  FadeOut(eq25[3]),
                  FadeOut(eq25[4][:3]),
                  run_time=1.6)
        self.wait(0.1)
        eq24_1 = MathTex(r'-').move_to(eq24[4][1])
        self.play(
            FadeOut(eq24[4][2:-1]),
            FadeOut(eq27[2][0], target_position=eq24[4][2]),
            eq27[2][1].animate.move_to(eq24[4][3:-1]).align_to(eq24[4][4], DOWN),
            mh.fade_replace(eq24[4][1], eq24_1[0][0]),
            FadeOut(eq27[:2], rect1, eq26),
            run_time=1.3
        )
        self.play(mh.rtransform(eq24[:4], eq28[:4], eq24[4][0], eq28[4][0],
                                eq24_1[0][0], eq28[4][1],
                                eq27[2][1], eq28[4][2],
                                eq24[4][-1], eq28[4][-1],
                                eq24[-1], eq28[-1]))
        self.wait(0.1)
        gp4 = VGroup(eq29[0][3], eq29[2][4], eq29[4][3], eq29[5][3],
                     eq30[0][3], eq30[2][4], eq30[4][3], eq30[5][3])
        gp4.set_color(col_angle)
        self.play(mh.rtransform(
            eq23[0][:3], eq29[0][:3], eq23[1], eq29[1], eq23[2][:4], eq29[2][:4],
            eq23[0][-1], eq29[0][-1], eq23[4][:3], eq29[4][:3], eq23[4][-1], eq29[4][-1],
            eq23[-1][:3], eq29[-1][:3], eq23[-1][-3:], eq29[-1][-3:], eq23[2][-1], eq29[2][-1],
            eq28[0][:3], eq30[0][:3], eq28[1], eq30[1], eq28[2][:4], eq30[2][:4],
            eq28[0][-1], eq30[0][-1], eq28[4][:3], eq30[4][:3], eq28[4][-1], eq30[4][-1],
            eq28[-1][:3], eq30[-1][:3], eq28[-1][-3:], eq30[-1][-3:], eq28[2][-1], eq30[2][-1],
            eq23[3], eq29[3], eq28[3], eq30[3]
        ), Succession(Wait(0.2), FadeIn(gp4)))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq29[0][:3], eq31[0][:3], eq29[0][-1], eq31[0][-1],
                eq29[1], eq31[1], eq29[2][:4], eq31[2][:4], eq29[2][-1], eq31[2][-1],
                eq29[3], eq31[3], eq29[4][0], eq31[4][0], eq29[4][1:3], eq31[4][5:7],
                eq29[4][4], eq31[4][-1],
                          eq30[0][:3], eq32[0][:3], eq30[0][-1], eq32[0][-1],
                          eq30[1], eq32[1], eq30[2][:4], eq32[2][:4], eq30[2][-1], eq32[2][-1],
                          eq30[3], eq32[3], eq30[4][0], eq32[4][0], eq30[4][1:3], eq32[4][5:7],
                          eq30[4][4], eq32[4][-1]
                          ),
            FadeOut(eq29[0][-2], shift=mh.diff(eq29[0][-1], eq31[0][-1])),
            FadeOut(eq29[2][-2], shift=mh.diff(eq29[2][-1], eq31[2][-1])),
            FadeOut(eq29[4][3], shift=mh.diff(eq29[4][4], eq31[4][-1])),

            FadeOut(eq30[0][-2], shift=mh.diff(eq30[0][-1], eq32[0][-1])),
            FadeOut(eq30[2][-2], shift=mh.diff(eq30[2][-1], eq32[2][-1])),
            FadeOut(eq30[4][3], shift=mh.diff(eq30[4][4], eq32[4][-1])),
            run_time=1.5),
            Succession(Wait(0.8), FadeIn(eq31[4][1:5], eq31[4][7:10])),
            Succession(Wait(0.8), FadeIn(eq32[4][1:5], eq32[4][7:10])),
            FadeOut(eq29[-1], eq30[-1]),
                  )
        self.wait(0.1)
        mh.copy_colors_eq(eq32[2], eq32_1[2])
        self.play(FadeIn(eq32_1),
                  VGroup(eq31, eq32).animate.set_opacity(0.6))
        self.wait(0.1)
        line2 = Line(eq33_1.get_corner(DL), eq33_1.get_corner(UR), stroke_width=6, stroke_color=RED).set_z_index(5)
        self.play(Create(line2, rate_func=linear, run_time=0.7))
        self.wait(0.1)
        self.play(FadeOut(line2, eq32_1))
        self.wait(0.1)
        # eq33_1 = eq33[0][1:].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(FadeIn(eq33_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq33_1[2][:], eq33[0][1:], eq33_1[2][:].copy(), eq33[2][2:], run_time=1.5),
                  Succession(Wait(0.5), FadeIn(eq33[1], eq33[0][0])),
                  FadeOut(eq33_1[:2]))
        self.play(
                  mh.rtransform(eq33[2][3].copy(), eq33[2][0]),
                  mh.stretch_replace(eq33[2][-2].copy(), eq33[2][1]))
        self.wait(0.1)
        eq34_1 = eq34[2][1].copy()
        self.play(AnimationGroup(mh.rtransform(eq33[0][0], eq34[0][0], eq33[0][1:], eq34[0][2:],
                                eq33[1], eq34[1], eq33[2][2:], eq34[2][6:],
                                eq33[2][1], eq34[2][1], eq33[2][0], eq34[2][3],
                                ),
                  FadeIn(eq34[0][1].set_color(col_p)),
                  FadeIn(eq34[2][2].set_color(col_x)),
                  mh.fade_replace(eq33[2][1].copy(), eq34[2][4].set_color(col_num), coor_mask=RIGHT),
                  mh.stretch_replace(eq33[2][-2].copy(), eq34_1),
                  run_time=1.2),
                  Succession(Wait(0.4), FadeIn(eq34[2][0], eq34[2][5]))
                  )
        self.remove(eq34_1)
        self.wait(0.1)
        self.play(mh.rtransform(eq34[0][-7:], eq35[0][-7:], eq34[1], eq35[1], eq34[2][-7:], eq35[2][-7:],
                                eq34[2][2:4], eq35[0][2:4], eq34[2][0], eq35[0][0], eq34[0][:2], eq35[0][4:6],
                                eq34[2][4], eq35[2][0], eq34[2][5], eq35[0][6]),
                  mh.stretch_replace(eq34[2][1], eq35[0][1]),
                  run_time=1.5)
        self.play(mh.rtransform(eq35[0][:3], eq36[0][:3], eq35[0][5:], eq36[0][5:], eq35[1:], eq36[1:]),
                  mh.fade_replace(eq35[0][4], eq36[0][4].set_color(col_p)),
                  mh.fade_replace(eq35[0][3], eq36[0][3].set_color(col_p)),
                  )
        self.wait(0.1)
        eq37_1 = eq37.copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(FadeOut(eq36[0][:7]),
                  FadeIn(eq37_1[0][0].set_color(col_WVD)),
                  mh.rtransform(eq36[0][-7:], eq37[0][-7:], eq36[1], eq37[1], eq36[2][-7:], eq37[2][-7:],
                                eq36[2][0], eq37[2][0]),
                  Succession(Wait(0.2), FadeIn(eq37[2][1:3])))
        self.play(mh.rtransform(eq37[0][1:], eq37_1[0][1:], eq37[1:], eq37_1[1:]))
        self.wait(0.1)
        eq37 = eq37_1
        self.wait(0.1)
        self.play(mh.rtransform(eq37[0][-7:], eq38[1][:], eq37[1], eq38[2], eq37[2][-7:], eq38[4][:],
                                eq37[0][0], eq38[0][3], eq37[2][1], eq38[3][3]),
                  mh.stretch_replace(eq37[2][0], eq38[3][2]),
                  mh.stretch_replace(eq37[2][2], eq38[3][4]),
                  Succession(Wait(0.3), FadeIn(eq38[0][:3], eq38[0][-1], eq38[3][:2], eq38[3][5:]))
                  )
        self.wait(0.1)
        circ1 = mh.circle_eq(eq38[3][2:5], scale=0.5).set_z_index(5)
        self.play(Create(circ1, rate_func=linear, run_time=0.5))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq38[0][:3], eq39[0][:3], eq38[0][3], eq39[0][4], eq38[0][-1], eq39[0][-1],
                                eq38[1:3], eq39[1:3], eq38[4], eq39[3]),
                  mh.rtransform(eq38[3][1:5].set_z_index(6), eq39[0][5:9].set_z_index(6), path_arc=PI/4),
                  run_time=1.8),
                  FadeOut(eq38[3][0], eq38[3][5:], circ1),
                  Succession(Wait(1), FadeIn(eq39[0][3], eq39[0][9]))
                  )
        self.wait(0.1)
        rect2 = SurroundingRectangle(eq40, fill_opacity=0, stroke_color=RED, stroke_opacity=1, stroke_width=8,
                                     corner_radius=0.2, buff=0.2)
        rect2.rotate(PI, UP)
        self.play(Create(rect2, rate_func=linear))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:3], eq40[:3], eq1[3][:], eq40[3][:-2], eq1[4], eq40[4],
                                run_time=1.2),
                  Succession(Wait(0.8), FadeIn(eq40[3][-2:])))
        self.wait(0.1)
        self.play(FadeOut(eq39[0][3], eq39[0][5:10]))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq39[0][4], eq41[0][0], eq39[1:3], eq41[1:3]),
                  mh.fade_replace(eq39[3], eq41[3].set_color(col_num), coor_mask=RIGHT),
                                 run_time=1.4),
                  FadeOut(eq39[0][:3], eq39[0][-1]))
        self.wait(0.1)
        self.play(FadeOut(eq41))
        self.wait(0.1)
        rect3 = SurroundingRectangle(eq32_1, fill_opacity=0, stroke_width=8, stroke_color=RED,
                                     buff=0.2, corner_radius=0.15)
        self.play(FadeIn(eq32_1))
        self.wait(0.1)
        self.play(FadeIn(rect3))
        self.wait(0.1)
        eq42.move_to(eq40)
        gp5 = VGroup(eq40, rect2)
        gp5_1 = VGroup(eq42, rect2.copy()).to_edge(LEFT, buff=0.55)
        gp6 = VGroup(eq32_1, rect3)
        self.play(#gp5.animate.to_edge(LEFT, buff=0.55),
            mh.rtransform(gp5[1], gp5_1[1], eq40[:3], eq42[:3], eq40[3][:2], eq42[3][:2],
                          eq40[3][4:], eq42[3][4:], eq40[4], eq42[4]),
            mh.fade_replace(eq40[3][2], eq42[3][2]),
            mh.fade_replace(eq40[3][3], eq42[3][3]),
            gp6.animate.move_to(gp5).to_edge(RIGHT, buff=0.55),
                  FadeOut(eq31, eq32),
                  run_time=2.5)

        self.wait()

class HarmonicSolve(RotateHarmonicEqs):
    def construct(self):
        gp1, gp2 = self.do_anim(just_eq=True)

        MathTex.set_default(font_size=80)
        eq1 = MathTex(r'\psi_0(x)', r'=', r'e^{-\frac12 x^2}')
        eq2 = MathTex(r'\psi_1(x)', r'=', r'(X-D)', r'e^{-\frac12 x^2}')
        eq3 = MathTex(r'\psi_n(x)', r'=', r'(X-D)^n', r'e^{-\frac12 x^2}')
        eq4 = MathTex(r'H', r'\psi_n(x)', r'=', r'n', r'\psi_n(x)')
        eq5 = MathTex(r'e^{-iHt}', r'\psi_n(x)', r'=', r'e^{-int}', r'\psi_n(x)')

        eq6 = MathTex(r'\psi_1(x)', r'=', r'2x', r'e^{-\frac12x^2}')
        eq7 = MathTex(r'\psi_2(x)', r'=', r'(4x^2-2)', r'e^{-\frac12x^2}')
        eq8 = MathTex(r'\psi_3(x)', r'=', r'(8x^3-12x)', r'e^{-\frac12x^2}')
        eq9 = MathTex(r'\psi_4(x)', r'=', r'(16x^4-48x^2+12)', r'e^{-\frac12x^2}')

        eq10 = MathTex(r'\psi_n(x)', r'=', r'\frac1{\pi^{\frac14}\sqrt{2^n n!}}', r'(X-D)^n', r'e^{-\frac12 x^2}')

        eq11 = MathTex(r'\lVert\psi_n\rVert^2', r'=', r'\int\lvert\psi_n(x)\rvert^2\,dx', r'=', r'1',
                       font_size=70)

        VGroup(eq4[0][0]).set_color(col_WVD)
        VGroup(eq1[0][0], eq11[0][1], eq11[2][2]).set_color(col_psi)
        VGroup(eq1[0][3], eq1[2][-2:], eq2[2][3], eq6[2][1],
               eq7[2][2:4], eq8[2][2:4], eq8[2][7], eq9[2][3:5], eq9[2][8:10],
               eq11[2][5], eq11[2][-1]).set_color(col_x)
        VGroup(eq2[2][1]).set_color(col_p)
        VGroup(eq1[2][0], eq5[0][0], eq5[3][0]).set_color(col_special)
        VGroup(eq1[2][2], eq1[2][4], eq6[2][0], eq7[2][1], eq7[2][5],
               eq8[2][1], eq8[2][5:7], eq9[2][1:3], eq9[2][6:8], eq9[2][11:13],
               eq10[2][0], eq10[2][3], eq10[2][5], eq10[2][-4], eq11[0][-1], eq11[2][-3], eq11[4]).set_color(col_num)
        VGroup(eq1[2][3], eq10[2][4], eq10[2][6:-4], eq10[2][-1], eq10[2][2],
               eq11[0][0], eq11[0][-2], eq11[2][:2], eq11[2][-4], eq11[2][-2]).set_color(col_op)
        VGroup(eq1[0][1], eq4[3], eq6[0][1], eq7[0][1], eq8[0][1], eq9[0][1],
               eq10[2][-3:-1], eq11[0][2], eq11[2][3]).set_color(col_var)
        VGroup(eq5[0][2], eq5[3][2]).set_color(col_i)
        VGroup(eq5[0][-1], eq5[3][-1]).set_color(col_angle)
        VGroup(eq10[2][2]).set_color(col_pi)

        eq1_2 = eq1.copy().next_to(gp1, DOWN, buff=0.2).to_edge(LEFT)
        mh.align_sub(eq2, eq2[1], eq1[1]).shift(DOWN*1.3)
        mh.align_sub(eq3, eq3[1], eq2[1]).shift(DOWN*1.3)
        # eq3_1 = eq3.copy().move_to(ORIGIN).shift(DOWN*0.6)
        eq10.shift(DOWN*0.6)
        # eq4.move_to(DOWN*1.2)
        eq4.to_edge(DOWN, buff=1)
        mh.align_sub(eq5, eq5[2], eq4[2]).move_to(eq4, coor_mask=RIGHT)
        eq11.next_to(eq10, DOWN, buff=0.4)

        mh.align_sub(eq6, eq6[1], eq1_2[1]).shift(DOWN)
        mh.align_sub(eq7, eq7[1], eq6[1]).shift(DOWN)
        mh.align_sub(eq8, eq8[1], eq7[1]).shift(DOWN)
        mh.align_sub(eq9, eq9[1], eq8[1]).shift(DOWN)

        self.add(gp1, gp2)
        self.play(FadeIn(eq1))
        self.wait(0.1)
        eq1_1 = eq1.copy()
        self.play(AnimationGroup(mh.rtransform(eq1_1[0][0], eq2[0][0], eq1_1[0][2:], eq2[0][2:],
                                eq1_1[1], eq2[1], eq1_1[2], eq2[3]),
                  mh.fade_replace(eq1_1[0][1], eq2[0][1].set_color(col_var)),
                  run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq2[2])))
        self.wait(0.1)
        eq2_1 = eq2.copy()
        self.add(eq2_1)
        self.play(mh.rtransform(eq2[0][0], eq3[0][0], eq2[0][2:], eq3[0][2:],
                                eq2[1], eq3[1], eq2[2][:], eq3[2][:-1], eq2[3], eq3[3]),
                  mh.fade_replace(eq2[0][1], eq3[0][1].set_color(col_var)),
                  FadeIn(eq3[2][-1].set_color(col_var), shift=mh.diff(eq2[2][:], eq3[2][:-1])),
                  run_time=1.2)
        self.wait(0.1)

        self.play(FadeOut(eq2_1), mh.rtransform(eq1, eq1_2, run_time=1.2))
        self.wait(0.1)
        eq1_ = eq1_2.copy()
        self.play(mh.rtransform(eq1_[0][0], eq6[0][0], eq1_[0][2:], eq6[0][2:], eq1_[1], eq6[1],
                                eq1_[2], eq6[3]),
                  mh.fade_replace(eq1_[0][1], eq6[0][1]),
                  FadeIn(eq6[2]))
        eq6_ = eq6.copy()
        self.play(mh.rtransform(eq6_[0][0], eq7[0][0], eq6_[0][2:], eq7[0][2:], eq6_[1], eq7[1],
                                eq6_[3], eq7[3]),
                  mh.fade_replace(eq6_[0][1], eq7[0][1]),
                  FadeIn(eq7[2]))
        eq7_ = eq7.copy()
        self.play(mh.rtransform(eq7_[0][0], eq8[0][0], eq7_[0][2:], eq8[0][2:], eq7_[1], eq8[1],
                                eq7_[3], eq8[3]),
                  mh.fade_replace(eq7_[0][1], eq8[0][1]),
                  FadeIn(eq8[2]),
                  FadeOut(eq3))
        eq8_ = eq8.copy()
        self.play(mh.rtransform(eq8_[0][0], eq9[0][0], eq8_[0][2:], eq9[0][2:], eq8_[1], eq9[1],
                                eq8_[3], eq9[3]),
                  mh.fade_replace(eq8_[0][1], eq9[0][1]),
                  FadeIn(eq9[2]))
        self.wait(0.1)

        mh.copy_colors_eq(eq3[0], eq4[1], eq3[0], eq4[-1])
        gp3 = VGroup(eq1_2, eq6, eq7, eq8, eq9)
        pt = eq1_2.get_corner(UL)
        self.play(gp3.animate(run_time=1.7).scale(0.6, about_point=pt),
                  Succession(Wait(0.8), FadeIn(eq4)))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][0], eq5[0][3], eq4[1:3], eq5[1:3],
                                eq4[3][0], eq5[3][3], eq4[4], eq5[4]),
                  Succession(Wait(0.3), FadeIn(eq5[0][:3], eq5[0][-1], eq5[3][:3], eq5[3][-1])))
        self.wait(0.1)
        mh.align_sub(eq3, eq3[1], eq10[1]).move_to(ORIGIN, coor_mask=RIGHT)
        # mh.copy_colors_eq(eq3, eq3_1)
        self.play(FadeOut(eq5, eq1_2, eq6, eq7, eq8, eq9), run_time=1.4)
        self.play(FadeIn(eq3))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq10[:2], eq3[2:], eq10[3:], run_time=1.4),
                  Succession(Wait(0.6), FadeIn(eq10[2])))
        self.wait(0.1)
        self.play(FadeIn(eq11))

        self.wait()

class HarmonicSolve2(RotateHarmonicEqs):
    def construct(self):
        gp1, gp2 = self.do_anim(just_eq=True)

        MathTex.set_default(font_size=70)

        eq1 = MathTex(r'f(x)', r'=', r'c_0', r'\psi_0(x)', r'+', r'c_1', r'\psi_1(x)', r'+', r'c_2', r'\psi_2(x)',
                      r'+', r'c_3', r'\psi_3(x)', r'+\cdots')
        eq2 = MathTex(r'c_n', r'=', r'\langle \psi_n, f\rangle', r'=', r'\int\psi_n(x)^*f(x)\,dx')
        eq3 = MathTex(r'\mathcal F_t', r'f', r'=', r'c_0', r'\mathcal F_t', r'\psi_0', r'+',
                      r'c_1', r'\mathcal F_t', r'\psi_1', r'+', r'c_2', r'\mathcal F_t', r'\psi_2',
                      r'+', r'c_3', r'\mathcal F_t', r'\psi_3')
        eq4 = MathTex(r'c_0', r'\psi_0', r'+', r'c_1', r'e^{-it}', r'\psi_1', r'+',
                      r'c_2', r'e^{-2it}', r'\psi_2',
                      r'+', r'c_3', r'e^{-3it}', r'\psi_3')

        VGroup(*(_[0] for _ in eq1[::3]),
               eq2[2][1], eq2[2][4], eq2[4][1], eq2[4][7]).set_color(col_psi)
        VGroup(*(_[-2] for _ in eq1[::3]),
               eq2[4][4], eq2[4][9], eq2[4][-1]).set_color(col_x)
        VGroup(*(_[1] for _ in eq1[3::3]),
               *(_[1] for _ in eq1[2::3]), eq2[0][1], eq2[2][2], eq2[4][2],
               *(_[2] for _ in eq4[8::4])).set_color(col_var)
        VGroup(*(_[0] for _ in eq1[2::3]), eq2[0][0]).set_color(GOLD_A)
        VGroup(eq2[4][6], *(_[-2] for _ in eq4[4::4])).set_color(col_i)
        VGroup(eq2[2][0], eq2[2][3], eq2[2][-1], eq2[4][0], eq2[4][-2]).set_color(col_op)
        VGroup(*(_[0] for _ in eq3[::4])).set_color(col_ft)
        VGroup(*(_[1] for _ in eq3[::4]),
               *(_[-1] for _ in eq4[4::4])).set_color(col_angle)
        VGroup(*(_[0] for _ in eq4[4::4])).set_color(col_special)

        eq1_1 = eq1[:2].copy().scale(1.5)
        eq1[:2].scale(1.2)
        eq1[:2].next_to(eq1[2:], UP, buff=0.6).shift(LEFT*3)
        eq1[-1].next_to(eq1[:-1], DOWN, buff=0.4).to_edge(RIGHT, buff=0.6)
        eq1.move_to(DOWN*0.5)
        eq2.next_to(eq1, DOWN, buff=0.1)
        eq3[:3].scale(1.2)
        mh.align_sub(eq3[:3], eq3[2], eq1[1])
        mh.align_sub(eq3[3:], eq3[3], eq1[2]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq4, eq4[0], eq3[3], coor_mask=UP)

        mh.align_sub(eq1_1, eq1_1[0], DOWN*0.5)[1].set_opacity(-1)

        self.add(gp1, gp2, eq1_1)
        self.play(mh.rtransform(eq1_1, eq1[:2], run_time=1.5),
                  Succession(Wait(0.7), FadeIn(eq1[2:], rate_func=linear)))
        self.wait(0.1)
        eq2_1 = eq2[:3].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(FadeIn(eq2_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1, eq2[:3], run_time=1.3),
                  Succession(Wait(0.6), FadeIn(eq2[3:])))
        self.wait(0.1)

        self.play(FadeOut(*(_[-3:] for _ in eq1[::3])))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][:-3], eq3[1][:]),
                  *(mh.rtransform(eq1[i][:-3], eq3[j][:],
                                  eq1[i-2:i], eq3[j-3:j-1])
                    for i, j in [(3,5), (6,9), (9, 13), (12, 17)]),
                  Succession(Wait(0.4), FadeIn(eq3[::4]))
                  )
        self.wait(0.1)
        self.play(*(mh.rtransform(eq3[i], eq4[j]) for i,j in ((3,0), (5,1), (6,2), (7,3), (9,5), (10,6),
                                                              (11,7), (13,9), (14,10), (15,11), (17,13))),
                  FadeOut(eq3[4], shift=mh.diff(eq3[4], eq4[:2])*RIGHT),
                  *(mh.fade_replace(eq3[i], eq4[j], coor_mask=RIGHT) for i,j in [(8,4), (12,8), (16,12)]))

        self.wait()

class GeneralOperator(Scene):
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
        eq1 = MathTex(r'\mathcal F_\theta^{-1}X\mathcal F_\theta', r'=', r'X\cos\theta + P\sin\theta')
        eq2 = MathTex(r'\mathcal F_\theta^{-1}P\mathcal F_\theta', r'=', r'-X\sin\theta + P\cos\theta')
        eq3 = MathTex(r'U^{-1}XU', r'=', r'X\cos\theta + P\sin\theta')
        eq4 = MathTex(r'U^{-1}PU', r'=', r' -X\sin\theta + P\cos\theta')

        eq5 = MathTex(r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}U^{-1}XU \\ '
                      r'U^{-1}PU\end{pmatrix}', r'=',
                      r'\begin{pmatrix}X\cos\theta + P\sin\theta \\'
                      r'-X\sin\theta + P\cos\theta\end{pmatrix}')
        eq6 = MathTex(r'U^{-1}\!', r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!U', r'=',
                      r'\begin{pmatrix}\cos\theta & \sin\theta \\'
                      r'-\sin\theta & \cos\theta\end{pmatrix}',
                      r'\begin{pmatrix}X \\ P\end{pmatrix}')
        eq7 = MathTex(r'U^{-1}\!', r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!U', r'=',
                      r'\begin{pmatrix}a & b \\'
                      r'c & d\end{pmatrix}',
                      r'\begin{pmatrix}X \\ P\end{pmatrix}')

        eq8 = MathTex(r'\int\lvert Uf(x)\rvert^2\,dx', r'=',
                      r'\int\lvert f(x)\rvert^2\,dx')
        eq9 = MathTex(r'DX', r'=', r'1+XD', font_size=80)

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True

        VGroup(eq1[0][4], eq1[2][0], eq8[0][5], eq8[0][-1], eq8[2][4], eq8[2][-1],
               eq9[0][1], eq9[2][-2], eq7[1][1]).set_color(col_x)
        VGroup(eq2[0][4], eq1[2][6], eq9[0][0], eq9[2][-1], eq7[1][2]).set_color(col_p)
        VGroup(eq1[2][1:4], eq1[2][7:10], eq7[4][1:-1]).set_color(col_trig)
        VGroup(eq1[0][:3], eq1[0][-2], eq3[0][0], eq3[0][-1], eq4[0][0], eq4[0][-1],
               eq8[0][2], eq7[0], eq7[2]).set_color(col_ft)
        VGroup(eq1[0][3], eq1[0][-1], eq1[2][4], eq1[2][-1]).set_color(col_angle)
        VGroup(eq8[0][:2], eq8[0][-2], eq8[0][-4], eq8[2][:2], eq8[2][-2], eq8[2][-4]).set_color(col_op)
        VGroup(eq8[0][3], eq8[2][2]).set_color(col_psi)
        VGroup(eq8[0][-3], eq9[2][0]).set_color(col_num)
        mh.copy_colors_eq(eq1[0][:4], eq2[0][:4], eq1[0][-2:], eq2[0][-2:], eq1[2][:], eq2[2][1:])

        gp1 = VGroup(eq1, eq2)
        VGroup(eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9).set_z_index(4)
        eq2.to_edge(DOWN, buff=0.4)
        mh.align_sub(eq1, eq1[1], eq2[1]).next_to(eq2, UP, coor_mask=UP, buff=0.2)
        eq1[2].align_to(eq2[2][1], LEFT)
        eq2[2][6:].align_to(eq1[2][5:], LEFT)
        mh.align_sub(eq3, eq3[1], eq1[1])
        mh.align_sub(eq4, eq4[1], eq2[1])
        eq3[2].align_to(eq4[2][1], LEFT)
        eq4[2][6:].align_to(eq3[2][5], LEFT)

        mh.align_sub(eq5, eq5[1], VGroup(eq3[1], eq4[1]), coor_mask=UP)
        eq5[2][1:12].align_to(eq5[2][13], LEFT)
        mh.align_sub(eq6, eq6[3], eq5[1], coor_mask=UP)
        eq6[4][1:5].align_to(eq6[4][10], LEFT)
        mh.align_sub(eq7, eq7[3], eq6[3], coor_mask=UP)
        eq7_2 = eq7.copy().to_edge(UP, buff=0.4)

        eq9.move_to(DOWN*0.5)

        if just_eq:
            mh.copy_colors_eq(eq7_2[1], eq7_2[-1])
            return eq7_2, eq9

        box1 = SurroundingRectangle(gp1, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                    buff=0.2, corner_radius=0.2)
        box2 = SurroundingRectangle(eq5, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                    buff=0.2, corner_radius=0.2)
        box3 = SurroundingRectangle(eq6, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                    buff=0.2, corner_radius=0.2)
        box4 = SurroundingRectangle(eq7, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                    buff=0.2, corner_radius=0.2)

        eq8.move_to(box1)

        self.add(eq1, eq2, box1)

        self.play(mh.rtransform(eq1[0][4], eq3[0][3], eq1[1:], eq3[1:], eq1[0][1:3], eq3[0][1:3],
                                eq2[0][4], eq4[0][3], eq2[1:], eq4[1:], eq2[0][1:3], eq4[0][1:3]),
                  mh.fade_replace(eq1[0][-2:], eq3[0][-1], coor_mask=RIGHT),
                  mh.fade_replace(eq1[0][:1] + eq1[0][3], eq3[0][0], coor_mask=RIGHT),
                  mh.fade_replace(eq2[0][-2:], eq4[0][-1], coor_mask=RIGHT),
                  mh.fade_replace(eq2[0][:1] + eq2[0][3], eq4[0][0], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        self.play(FadeOut(eq3, eq4), FadeIn(eq8))
        self.wait(0.1)
        self.play(FadeIn(eq3, eq4), FadeOut(eq8))

        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq3[1], eq5[1], eq3[0][:], eq5[0][1:6],
                                               eq4[0][:], eq5[0][6:11],
                                eq3[2][:], eq5[2][1:12], eq4[2][:], eq5[2][12:24]),
                  mh.rtransform(eq4[1], eq5[1], box1, box2),
                                 run_time=1.4),
                  Succession(Wait(0.7), FadeIn(eq5[0][0], eq5[0][-1], eq5[2][0], eq5[2][-1]))
                                               )
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(
            eq5[0][0], eq6[1][0], eq5[0][1:4], eq6[0][:], eq5[0][4], eq6[1][1], eq5[0][5], eq6[2][0],
            eq5[0][-1], eq6[1][-1], eq5[0][9], eq6[1][2], eq5[1], eq6[3],
            eq5[2][0], eq6[4][0], eq5[2][2:6], eq6[4][1:5], eq5[2][8:12], eq6[4][5:9],
            eq5[2][12], eq6[4][9], eq5[2][14:18], eq6[4][10:14], eq5[2][20:24], eq6[4][14:18],
            eq5[2][-1], eq6[4][-1], eq5[2][1], eq6[5][1], eq5[2][7], eq6[5][2],
        ),
        mh.rtransform(eq5[0][6:9], eq6[0][:], eq5[0][-2], eq6[2][0],
                      eq5[2][13], eq6[5][1], eq5[2][19], eq6[5][2],
                      box2, box3),
            run_time=1.8),
            FadeOut(eq5[2][6], eq5[2][18]),
            Succession(Wait(0.5), FadeIn(eq6[5][0], eq6[5][-1]))
        )
        self.wait(0.1)
        eq7_1 = eq7[4][1:-1].copy()
        eq7_1[0].move_to(eq6[4][1:5], coor_mask=RIGHT)
        eq7_1[1].move_to(eq6[4][5:9], coor_mask=RIGHT)
        eq7_1[2].move_to(eq7_1[0], coor_mask=RIGHT)
        eq7_1[3].move_to(eq7_1[1], coor_mask=RIGHT)
        self.play(FadeOut(eq6[4][1:-1]), FadeIn(eq7_1))
        self.play(mh.rtransform(eq6[:4], eq7[:4], eq6[-1], eq7[-1],
                                eq7_1, eq7[4][1:-1], eq6[4][0], eq7[4][0],
                                eq6[4][-1], eq7[4][-1], box3, box4))
        self.wait(0.1)
        txt1 = Tex(r'\sf real matrix', color=RED, font_size=60).next_to(eq7[4], UP, buff=0.5).set_z_index(5)
        txt1.shift(RIGHT)
        arr1 = Arrow(txt1[0][:5].get_bottom(), eq7[4].get_center()+UP*0.4,
                     color=RED, stroke_width=8, buff=0.1,
                     max_stroke_width_to_length_ratio=20,
                     max_tip_length_to_length_ratio=10).set_z_index(5)
        self.play(FadeIn(txt1, arr1))
        self.wait(0.1)
        self.play(FadeOut(txt1, arr1))
        self.wait(0.1)
        box5 = box4.copy().shift(mh.diff(eq7, eq7_2)).set_fill(opacity=0)
        box6 = Rectangle(stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=1,
                         width=config.frame_width, height=config.frame_height)
        self.play(AnimationGroup(mh.rtransform(eq7, eq7_2, box4, box5), FadeIn(box6),
                                 run_time=1.8),
                  Succession(Wait(1.4), FadeIn(eq9)))
        self.wait()

class Symplectic(GeneralOperator):
    bgcol = BLACK
    def construct(self):
        eq1, eq2 = self.do_anim(just_eq=True)
        eq3 = MathTex(r'PX', r'=', r'-i+XP', font_size=80)
        eq4 = MathTex(r'XP-PX', r'=', r'i', font_size=80)
        eq5 = MathTex(r'[X,P]', r'=', r'i', font_size=80)
        eq6 = MathTex(r'U^{-1}\!', r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!U', r'=',
                      r'\begin{pmatrix}X^\prime \\ P^\prime\end{pmatrix}', r'=',
                      r'\begin{pmatrix}a & b \\'
                      r'c & d\end{pmatrix}',
                      r'\begin{pmatrix}X \\ P\end{pmatrix}')
        eq7 = MathTex(r'[X^\prime, P^\prime]', r'=', r'i', font_size=80)
        eq8 = MathTex(r'[aX+bP', r',', r'cX+dP]', r'=', r'i', font_size=80)
        eq9 = MathTex(r'a[X,', r'cX+dP]', r'+', r'b[P,', r'cX+dP]', r'=', r'i', font_size=80)
        eq10 = MathTex(r'ac[X,X]', r'+', r'ad[X,P]', r'+', r'bc[P,X]', r'+', r'bd[P,P]', r'=', r'i', font_size=80)
        eq11 = MathTex(r'ad[X,P]', r'-', r'bc[X,P]', r'=', r'i', font_size=80)
        eq12 = MathTex(r'ad-bc', r'=', r'1', font_size=80)

        VGroup(eq6[4][1:3]).set_color(col_x)
        VGroup(eq3[0][0], eq3[2][-1], eq6[4][3:5]).set_color(col_p)
        VGroup(eq3[2][1]).set_color(col_i)
        VGroup(eq5[0][::2]).set_color(col_op)
        VGroup(eq12[2]).set_color(col_num)

        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq5, eq5[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[3], eq1[3], coor_mask=UP)
        mh.align_sub(eq7, eq7[0][3], eq5[0][2])
        mh.align_sub(eq8, eq8[1], eq5[0][2])
        eq9[2:].next_to(eq9[:2], DOWN, buff=0.4).align_to(eq9[1], LEFT)
        eq9.move_to(ORIGIN).move_to(eq8, coor_mask=UP)
        eq10[:3].move_to(eq9[:2])
        eq10[3:].move_to(eq9[2:])
        # mh.align_sub(eq9, eq9[3], eq1[3], coor_mask=UP)
        mh.align_sub(eq11, eq11[3], eq2[1], coor_mask=UP)
        mh.align_sub(eq12, eq12[1], eq2[1], coor_mask=UP)

        self.add(eq1, eq2)
        self.play(mh.rtransform(eq2[0][1], eq3[0][1], eq2[1], eq3[1],
                                eq2[2][1:3], eq3[2][2:4]),
                  mh.fade_replace(eq2[0][0], eq3[0][0]),
                  mh.fade_replace(eq2[2][-1], eq3[2][-1]),
                  mh.fade_replace(eq2[2][0], eq3[2][:2], coor_mask=RIGHT))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0][:], eq4[0][-2:], eq3[2][-2:], eq4[0][:2],
                                eq3[2][1], eq4[2][0], eq3[1], eq4[1]),
                  mh.fade_replace(eq3[2][2], eq4[0][2]),
                  FadeOut(eq3[2][0], shift=mh.diff(eq3[2][1], eq4[2][0])),
                  run_time=1.4)
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq4[0][0], eq5[0][1], eq4[0][1], eq5[0][3],
                                eq4[1:], eq5[1:]),
                  mh.rtransform(eq4[0][-2], eq5[0][3], eq4[0][-1], eq5[0][1]),
                  mh.fade_replace(eq4[0][2], eq5[0][2], coor_mask=RIGHT),
                                 run_time=1.2),
                  Succession(Wait(0.6), FadeIn(eq5[0][0], eq5[0][-1])))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:4], eq6[:4], eq1[4:], eq6[6:], eq1[3].copy(), eq6[5]),
                  Succession(Wait(0.4), FadeIn(eq6[4])))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[4][1:3].copy(), eq7[0][1:3], eq6[4][3:5].copy(), eq7[0][4:6],
                                eq5[0][::2], eq7[0][::3], eq5[1:], eq7[1:]),
                  mh.rtransform(eq5[0][1], eq7[0][1], eq5[0][3], eq7[0][4]),
                  )
        self.wait(0.1)
        eq6_ = eq6[-2:].copy()

        self.play(mh.rtransform(eq6_[0][1:3], eq8[0][1:5:3], eq6_[1][1:3].copy(), eq8[0][2:6:3],
                                eq7[0][0], eq8[0][0], eq7[0][3], eq8[1][0], run_time=1.5),
                  Succession(Wait(0.5), FadeIn(eq8[0][3])),
                  FadeOut(eq7[0][1:3]))
        self.play(mh.rtransform(eq6_[0][3:5], eq8[2][:4:3], eq6_[1][1:3], eq8[2][1:5:3],
                                eq7[0][-1], eq8[2][-1], eq7[1:], eq8[3:], run_time=1.5),
                  Succession(Wait(0.5), FadeIn(eq8[2][2])),
                  FadeOut(eq7[0][4:6]))

        self.wait(0.1)
        self.play(mh.rtransform(eq8[0][1], eq9[0][0], eq8[0][0].copy(), eq9[0][1], eq8[0][2], eq9[0][2],
                                eq8[1][0].copy(), eq9[0][3], eq8[2].copy(), eq9[1],
                                eq8[0][3], eq9[2][0], eq8[0][4], eq9[3][0], eq8[0][0], eq9[3][1],
                                eq8[0][5], eq9[3][2], eq8[2:], eq9[4:], eq8[1][0], eq9[3][3],
                                run_time=1.4
                                ))
        self.play(*[mh.rtransform(eq9[i][0].copy(), eq10[j][0], eq9[i][1:4].copy(), eq10[j][2:5],
                                eq9[i+1][0], eq10[j][1], eq9[i+1][1], eq10[j][5], eq9[i+1][-1].copy(), eq10[j][-1],
                                eq9[i+1][2], eq10[j+1][0], eq9[i][0], eq10[j+2][0], eq9[i][1:4], eq10[j+2][2:5],
                                eq9[i+1][3], eq10[j+2][1], eq9[i+1][-2:], eq10[j+2][-2:]) for i,j in [(0,0), (3,4)]],
                  mh.rtransform(eq9[2], eq10[3], eq9[-2:], eq10[-2:]),
                  run_time=1.5)
        self.wait(0.1)
        lines = [Line(_[2].get_corner(DL)+LEFT*0.1, _.get_corner(UR)+RIGHT*0.1,
                      color=RED, stroke_width=6).set_z_index(5)
                 for _ in [eq10[0], eq10[6]]]
        self.play(Create(lines[0], rate_func=linear), run_time=0.6)
        self.play(Create(lines[1], rate_func=linear), run_time=0.6)
        self.wait(0.1)
        self.play(FadeOut(eq10[:2], eq10[5:7], *lines))
        self.play(mh.rtransform(eq10[2], eq11[0], eq10[4][:3], eq11[2][:3],
                                eq10[4][3], eq11[2][5], eq10[4][4], eq11[2][4], eq10[4][5], eq11[2][3],
                                eq10[4][6], eq11[2][6], eq10[-2:], eq11[-2:]
                                ),
                  mh.fade_replace(eq10[3], eq11[1]),
                  run_time=1.5)
        self.wait(0.1)
        eq11_1 = eq11[-1].copy().move_to(eq11[0][2:], coor_mask=RIGHT)
        eq11_2 = eq11[-1].copy().move_to(eq11[2][2:], coor_mask=RIGHT)
        self.play(FadeOut(eq11[0][2:], eq11[2][2:]),
                  FadeIn(eq11_1, eq11_2))
        self.wait(0.1)
        eq12_1 = eq12[-1].copy().move_to(eq11[-1], coor_mask=RIGHT)
        self.play(FadeOut(eq11_1, eq11_2, eq11[-1]), FadeIn(eq12_1))
        self.play(mh.rtransform(eq11[0][:2], eq12[0][:2], eq11[1][0], eq12[0][2], eq11[2][:2], eq12[0][3:],
                                eq11[3], eq12[1], eq12_1, eq12[2]))

        self.wait()

def stretch_replace_each(eq1, eq2, **kwargs):
    n = len(eq1)
    assert n == len(eq2)
    res = []
    for i in range(n):
        res.append(mh.stretch_replace(eq1[i], eq2[i], **kwargs))
    return AnimationGroup(*res)

col_squeeze = col_ft * 0.5+ PURPLE * 0.5
col_chirp = BLUE_B*0.5 + col_ft*0.5

class SymplecticConstruct(GeneralOperator):
    bgcol = BLACK

    def construct(self):
        gps, cols = self.do_anims1(eq_only=False)

        MathTex.set_default(font_size=80, stroke_width=1.6)
        eq1 = MathTex(r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}c & -s \\ s & c\end{pmatrix}')
        eq2 = MathTex(r's=\sin\theta', r'c=\cos\theta')

        eq3 = MathTex(r'\renewcommand*{\arraystretch}{1.2}'
                      r'\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                      r'\!\begin{pmatrix} c & -s \\ s & c\end{pmatrix}',
                      r'=',
                      r'\begin{pmatrix} c & -s \\ s+\frac{c^2}{s} & c-\frac css\end{pmatrix}',
                      font_size=80)
        eq4 = MathTex(r'\renewcommand*{\arraystretch}{1.2}'
                      r'\begin{pmatrix} 0 & 1 \\ -1 & 0\end{pmatrix}',
                      r'\!\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                      r'\!\begin{pmatrix} c & -s \\ s & c\end{pmatrix}',
                      r'=',
                      r'\begin{pmatrix} c & -s \\ \frac 1s & 0\end{pmatrix}',
                      font_size=73)

        eq5 = MathTex(r'\renewcommand*{\arraystretch}{1.2}'
                      r'=',
                      r'\begin{pmatrix} \frac 1s & 0 \\ -c & s \end{pmatrix}',
                      font_size=73)

        eq6 = MathTex(r'\renewcommand*{\arraystretch}{1.2}'
                      r'\begin{pmatrix} s & 0 \\ 0 & \frac 1s\end{pmatrix}',
                      r'\!\begin{pmatrix} 0 & 1 \\ -1 & 0\end{pmatrix}',
                      r'\!\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                      r'\!\begin{pmatrix} c & -s \\ s & c\end{pmatrix}',
                      r'=',
                      r'\begin{pmatrix} \frac 1s & 0 \\ -c & s \end{pmatrix}',
                      font_size=62)
        eq7 = MathTex(r'\renewcommand*{\arraystretch}{1.2}'
                      r'=',
                      r'\begin{pmatrix} 1 & 0 \\ -\frac cs & 1 \end{pmatrix}',
                      font_size=62)

        eq8 = MathTex(r'\renewcommand*{\arraystretch}{1.2}'
                      r'\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                      r'\!\!\begin{pmatrix} s & 0 \\ 0 & \frac 1s\end{pmatrix}',
                      r'\!\!\begin{pmatrix} 0 & 1 \\ -1 & 0\end{pmatrix}',
                      r'\!\!\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                      r'\!\!\begin{pmatrix} c & -s \\ s & c\end{pmatrix}',
                      r'\!=\!',
                      r'\begin{pmatrix} 1 & 0 \\ -\frac cs & 1 \end{pmatrix}',
                      font_size=55)
        eq9 = MathTex(r'\renewcommand*{\arraystretch}{1.2}'
                      r'\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                      r'\!\!\begin{pmatrix} s & 0 \\ 0 & \frac 1s\end{pmatrix}',
                      r'\!\!\begin{pmatrix} 0 & 1 \\ -1 & 0\end{pmatrix}',
                      r'\!\!\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                      r'\!=\!',
                      r'\begin{pmatrix} c & s \\ -s & c\end{pmatrix}',
                      font_size=63)

        eq10 = MathTex(r'\renewcommand*{\arraystretch}{1.2}'
                       r'\!\begin{pmatrix} c & s \\ -s & c\end{pmatrix}',
                       r'=',
                       r'\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                       r'\!\begin{pmatrix} s & 0 \\ 0 & \frac 1s\end{pmatrix}',
                       r'\!\begin{pmatrix} 0 & 1 \\ -1 & 0\end{pmatrix}',
                       r'\!\begin{pmatrix} 1 & 0 \\ \frac cs & 1\end{pmatrix}',
                       font_size=63)

        eq11 = MathTex(r'U', r'=', r'C_{\frac cs}', r'S_{\frac 1s}', r'\mathcal F', r'C_{\frac cs}')
        eq12 = MathTex(r'U', r'=', r'C_{\cot\theta}', r'S_{\csc\theta}', r'\mathcal F', r'C_{\cot\theta}')
        eq12_1 = eq12.copy()

        eq13 = MathTex(r'U^{-1}', r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix} X \\ P\end{pmatrix}',
                       r'U', r'=', r'\begin{pmatrix} c & s \\ -s & c\end{pmatrix}',
                                   r'\begin{pmatrix} X \\ P\end{pmatrix}',
                       font_size=70
                       )
        eq14 = MathTex(r'\mathcal F_\theta', r'=', r'U')
        eq15 = MathTex(r'\mathcal F_\theta', r'=', r'C_{\cot\theta}', r'S_{\csc\theta}', r'\mathcal F', r'C_{\cot\theta}')
        eq16 = MathTex(r'\mathcal F_\theta', r'=', r'\sqrt{\csc\theta}', r'C_{\cot\theta}', r'S_{\csc\theta}', r'\mathcal F', r'C_{\cot\theta}')

        eq17 = MathTex(r'\mathcal F_\theta', r'e^{-\frac12x^2}', r'=', r'e^{-\frac12x^2}')
        eq18 = MathTex(r'C_{\cot\theta}', r'e^{-\frac12x^2}', r'=', r'e^{\frac12i\cot\theta\,x^2}', r'e^{-\frac12x^2}')
        eq19 = MathTex(r'C_{\cot\theta}', r'e^{-\frac12x^2}', r'=', r'e^{-\frac12(1-i\cot\theta)x^2}')
        eq20 = MathTex(r'\mathcal F', r'C_{\cot\theta}', r'e^{-\frac12x^2}', r'=', r'\mathcal F', r'e^{-\frac12(1-i\cot\theta)x^2}')
        eq21 = MathTex(r'\mathcal F', r'C_{\cot\theta}', r'e^{-\frac12x^2}', r'=', r'\frac1{\sqrt{1-i\cot\theta} }',
                       r'e^{-\frac12\frac1{1-i\cot\theta}x^2}')
        eq22 = MathTex(r'C_{\cot\theta}', r'S_{\csc\theta}', r'\mathcal F', r'C_{\cot\theta}', r'e^{-\frac12x^2}', r'=', r'\frac1{\sqrt{1-i\cot\theta} }',
                       r'e^{-\frac12x^2}')

        eq23 = MathTex(r'\mathcal F_\theta', r'=', r'\sqrt{1-i\cot\theta}\,', r'C_{\cot\theta}', r'S_{\csc\theta}', r'\mathcal F', r'C_{\cot\theta}')

        VGroup(eq1[0][1], eq1[0][3:-1], eq2[0][0], eq2[0][2:-1], eq3[0][3:6:2],
               eq6[0][1], eq6[0][6], eq8[0][3], eq8[0][5], eq18[3][5:8]).set_color(col_trig)
        VGroup(eq2[0][-1], eq14[0][-1], eq18[3][8]).set_color(col_angle)
        VGroup(eq3[0][4], eq6[0][5], eq8[0][4], eq17[1][3]).set_color(col_op)
        VGroup(eq3[0][1:3], eq3[0][6], eq4[0][1:3], eq4[0][4:6],
               eq6[0][2:5], eq8[0][1:3], eq8[0][-2], eq17[1][2:5:2]).set_color(col_num)
        mh.copy_colors_eq(eq2[0], eq2[1])
        VGroup(eq13[1][1], eq17[1][-2:]).set_color(col_x)
        VGroup(eq13[1][2]).set_color(col_p)
        mh.copy_colors_eq(eq13[1], eq13[-1])
        VGroup(eq14[0][0], eq14[2]).set_color(col_ft)
        VGroup(eq17[1][0]).set_color(col_special)
        VGroup(eq18[3][4]).set_color(col_i)
        mh.copy_colors_eq(eq14[0], eq17[0], eq17[1], eq17[3])
        mh.copy_colors_eq(eq17[1][0], eq18[3][0], eq17[1][2:5], eq18[3][1:4], eq17[1][-2:], eq18[3][-2:])
        mh.copy_colors_eq(eq14[0], eq23[0])

        eq1.move_to(DOWN)
        eq2[1].shift(RIGHT)
        eq2.move_to(ORIGIN).to_edge(DOWN, buff=0.7)
        mh.align_sub(eq3, eq3[1], eq1[0], coor_mask=UP)
        mh.align_sub(eq4, eq4[-2], eq3[-2], coor_mask=UP)
        mh.align_sub(eq5, eq5[0], eq4[-2])
        mh.align_sub(eq6, eq6[-2], eq4[-2], coor_mask=UP)
        mh.align_sub(eq7, eq7[0], eq6[-2])
        mh.align_sub(eq8, eq8[-2], eq6[-2], coor_mask=UP)
        mh.align_sub(eq9, eq9[-2], eq8[-2], coor_mask=UP)

        eq10.to_edge(DOWN, buff=0.5)

        eq11.next_to(eq10, UP, buff=0.5)
        for i in range(len(eq11)):
            eq11[i].move_to(eq10[i], coor_mask=RIGHT)
            mh.align_sub(eq12[i], eq12[i][0], eq11[i][0])
        VGroup(eq11[-1], eq12[-1]).to_edge(RIGHT, buff=0.3)

        mh.align_sub(eq12_1, eq12_1[1], eq12[1], coor_mask=UP)
        # mh.align_sub(eq13, eq13[3], eq10[1], coor_mask=UP)
        eq13.to_edge(DOWN, buff=0.4)

        mh.align_sub(eq15, eq15[1], eq13[1], coor_mask=UP)
        mh.align_sub(eq14, eq14[1], eq15[1])
        mh.align_sub(eq14, eq14[2], eq13[2], coor_mask=RIGHT)
        mh.align_sub(eq16, eq16[1], eq15[1], coor_mask=UP)
        mh.align_sub(eq17, eq17[2], eq16[1], coor_mask=UP)
        mh.align_sub(eq18, eq18[2], eq16[1], coor_mask=UP)
        mh.align_sub(eq19, eq19[2], eq16[1], coor_mask=UP)
        mh.align_sub(eq20, eq20[3], eq16[1], coor_mask=UP)
        mh.align_sub(eq21, eq21[3], eq16[1], coor_mask=UP)
        eq21[:4].to_edge(LEFT, buff=0.3).shift(UP*1.2)
        eq21[4:].to_edge(RIGHT, buff=0.3).to_edge(DOWN, buff=0.3)
        mh.align_sub(eq22, eq22[6], eq21[3]).align_to(eq21, LEFT)
        mh.align_sub(eq22[6:], eq22[7][0], eq21[5][0]).align_to(eq21, RIGHT)

        self.wait(0.1)
        self.play(FadeIn(eq1, eq2))

        circ1 = mh.circle_eq(eq1[0][5]).set_z_index(0)
        eq3.set_z_index(1)
        self.wait(0.1)
        self.play(Create(circ1, run_time=0.6, rate_func=linear))
        self.wait(0.1)
        eq3_ = mh.align_sub(eq3[:2].copy(), eq3[1], eq1)
        self.play(mh.rtransform(eq1[0], eq3_[1]),
                  FadeIn(eq3_[0]),
                  circ1.animate.shift(mh.diff(eq1[0][5], eq3_[1][5])))
        self.wait(0.1)
        self.play(mh.rtransform(eq3_, eq3[:2]),
                  FadeIn(eq3[2], shift=mh.diff(eq3_, eq3[:2])),
                  circ1.animate.shift(mh.diff(eq3_[1][5], eq3[1][5])))
        self.wait(0.1)
        eq3.set_z_index(1)
        VGroup(eq3[3][0], eq3[3][-1]).set_z_index(0)
        eq3_ = eq3[1].copy()
        # circ2 = circ1.copy().shift(mh.diff(eq3[1][5], eq3[3][10]))
        circ2 = mh.circle_eq(eq3[3][10:16], scale=0.6).set_z_index(0)
        self.play(mh.rtransform(eq3_[1:4], eq3[3][1:4], eq3_[4], eq3[3][4], eq3_[5], eq3[3][10],
                  run_time=1.4),
                  Succession(Wait(0.8), FadeIn(eq3[3][0], eq3[3][-1])),
                  FadeOut(circ1),
                  Succession(Wait(0.8), Create(circ2, run_time=0.6, rate_func=linear)))
        eq3_ = eq3.copy()
        self.play(AnimationGroup(
            mh.rtransform(eq3_[3][1], eq3[3][6], eq3_[3][3], eq3[3][15]),
                  mh.rtransform(eq3_[3][2], eq3[3][11], copy_colors=False),
                  mh.rtransform(eq3_[0][3], eq3[3][6], eq3_[0][4:6], eq3[3][8:10], path_arc=PI/8),
                  mh.rtransform(eq3_[0][3:6].copy(), eq3[3][12:15], path_arc=PI/8),
                  # FadeIn(eq3[3][7].set_color(col_trig), shift=mh.diff(eq3_[1][1], eq3[3][6])),
                  run_time=2.3),
            Succession(Wait(1.3), FadeIn(eq3[3][5], eq3[3][7].set_color(col_trig))))
        self.wait(0.1)
        self.play(FadeOut(eq3[3][13:16]))
        eq_1 = MathTex(r'{}-', r'0', color=col_num)
        mh.align_sub(eq_1, eq_1[0], eq3[3][11])[1].move_to(eq3[3][10:16], coor_mask=RIGHT)
        self.play(FadeOut(eq3[3][10], target_position=eq_1[1]),
                  mh.fade_replace(eq3[3][11], eq_1[1], coor_mask=RIGHT),
                  FadeOut(eq3[3][12], target_position=eq_1[1]),
                  FadeOut(circ2))
        eq_2 = MathTex(r'\begin{pmatrix}\frac{s^2+c^2}{s}\end{pmatrix}')
        mh.align_sub(eq_2, eq_2[0][-3], eq3[3][8]).move_to(eq3[3][4:10], coor_mask=RIGHT)
        self.play(mh.rtransform(eq3[3][4], eq_2[0][1], eq3[3][5:10], eq_2[0][3:8]),
                  FadeIn(eq_2[0][2].set_color(col_trig), shift=mh.diff(eq3[3][4], eq_2[0][1])))
        eq_3 = MathTex(r'\begin{pmatrix}\frac{1}{s}\end{pmatrix}')
        mh.align_sub(eq_3, eq_3[0][-3], eq_2[0][-3])
        circ = mh.circle_eq(eq_2[0][1:6], scale=0.8).set_z_index(10)
        self.play(Create(circ), run_time=0.5)
        self.wait(0.1)
        self.play(FadeOut(eq_2[0][1:6]), FadeIn(eq_3[0][1].set_color(col_num)))
        self.wait(0.1)
        eq4_ = eq4[1:].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq3[:3], eq4_[:3], eq3[3][:4], eq4_[3][:4], eq3[3][-1], eq4_[3][-1],
                                eq_3[0][1], eq4_[3][4], eq_2[0][-3:-1], eq4_[3][5:7], eq_1[1][0], eq4_[3][7]),
                  FadeOut(circ, shift=mh.diff(eq_3[0][1], eq4_[3][4])))
        self.wait(0.1)
        self.play(mh.rtransform(eq4_, eq4[1:], run_time=1.2),
                  Succession(Wait(0.5), FadeIn(eq4[0])))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[-1][0], eq5[-1][0], eq4[-1][-1], eq5[-1][-1],
                                eq4[-1][1], eq5[-1][6], eq4[-1][3], eq5[-1][7],
                                eq4[-1][4:7], eq5[-1][1:4], eq4[-1][7], eq5[-1][4]
                                ),
                  FadeOut(eq4[-1][2], shift=mh.diff(eq4[-1][3], eq5[-1][7])),
                  FadeIn(eq5[-1][5], shift=mh.diff(eq4[-1][1], eq5[-1][6])),
                  run_time=1.4)
        self.wait(0.1)
        eq_ = VGroup(eq5[-1][1:4], eq5[-1][-2]).copy().rotate(PI/4)
        circ1 = mh.circle_eq(eq_, scale=0.8).rotate(-PI/4).move_to(eq5[-1])
        circ2 = circ1.copy().scale(eq6[-1].width/eq5[-1].width).move_to(eq6[-1])
        self.play(Create(circ1, rate_func=linear, run_time=0.6))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[:-1], eq6[1:-1], eq5[1], eq6[-1],
                                circ1, circ2, run_time=1.2),
                  Succession(Wait(0.5), FadeIn(eq6[0])))
        self.wait(0.1)
        eq7[-1][5].set_color(col_op)
        eq7[-1][6].set_color(col_trig)
        self.play(mh.rtransform(eq6[-1][1], eq7[-1][1], eq6[-1][4], eq7[-1][2],
                                eq6[-1][5:7], eq7[-1][3:5], eq6[-1][0], eq7[-1][0], eq6[-1][-1], eq7[-1][-1]),
                  FadeOut(eq6[-1][2:4]),
                  FadeIn(eq7[-1][5:7]),
                  mh.fade_replace(eq6[-1][7], eq7[-1][7].set_color(col_num), coor_mask=RIGHT),
                  run_time=1.)
        self.wait(0.1)
        self.play(FadeOut(circ2))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[:-1], eq8[1:-1], eq7[1], eq8[-1], run_time=1.2),
                  Succession(Wait(0.5), FadeIn(eq8[0])))
        self.wait(0.1)

        eq8_ = MathTex(r'\begin{pmatrix}\frac cs\!-\!\frac cs\end{pmatrix}', font_size=55)[0][1:-1]
        mh.copy_colors_eq(eq8[-1][4:7], eq8_[:3])
        mh.align_sub(eq8_, eq8_[3], eq8[-1][3]).next_to(eq8[-1][0], RIGHT, coor_mask=RIGHT, buff=-0.05)
        self.play(mh.rtransform(eq8[-1][3:7], eq8_[3:7]),
                  mh.fade_replace(eq8[-1][1].copy(), eq8_[:3]))
        eq8_1 = MathTex(r'\begin{pmatrix}0 & 1\end{pmatrix}', font_size=55, color=col_num)[0]
        mh.align_sub(eq8_1, eq8_1[2], eq8[-1][-2])[1].move_to(eq8[-1][1], coor_mask=RIGHT)
        self.play(FadeOut(eq8_[:3], shift=mh.diff(eq8_[:3], eq8_1[1])*RIGHT),
                  FadeOut(eq8_[3], shift=mh.diff(eq8_[3], eq8_1[1])*RIGHT),
                  FadeIn(eq8_1[1]),
                  FadeOut(eq8_[4:], shift=mh.diff(eq8_[4:], eq8_1[1])*RIGHT))
        self.wait(0.1)
        self.play(Succession(Wait(0.6), AnimationGroup(mh.rtransform(eq8[:4], eq9[:4], eq8[5], eq9[4]),
                  mh.rtransform(eq8[4][:2], eq9[-1][:2], eq8[4][2:4], eq9[-1][3:5],
                                eq8[4][4], eq9[-1][2], eq8[4][-2:], eq9[-1][-2:]),
                                                       run_time=1.5)),
                  FadeOut(eq8[-1][:3], eq8[-1][-2:], eq8_1[1]))
        self.wait(0.1)

        self.play(FadeOut(eq2),
                  mh.rtransform(eq9[:-2], eq10[2:], eq9[-2:], eq10[1::-1], run_time=1.8))
        self.wait(0.1)

        eq11.set_z_index(10)
        for i,j in [(5,3), (4,1), (3,2), (2,3)]:
            print(i,j)
            eq_ = gps[j][0].copy().set_z_index(10)
            if j == 1:
                anims = [mh.rtransform(eq_[0], eq12[i][0])]
            else:
                anims = [mh.rtransform(eq_[0], eq11[i][0]), mh.fade_replace(eq_[1], eq11[i][1:])]
                eq11[i][1:].set_color(col_trig)
                eq11[i][2].set_color(col_op)
                if j == 2:
                    eq11[i][1].set_color(col_num)
            self.play(*anims, run_time=1.2)
            self.wait(0.1)
            if j != 1:
                eq12[i][1:-1].set_color(col_trig)
                eq12[i][-1].set_color(col_angle)
                self.play(mh.rtransform(eq11[i][0], eq12[i][0]), FadeOut(eq11[i][1:]), FadeIn(eq12[i][1:]))
                self.wait(0.1)

        eq12[0].set_color(col_ft)
        self.play(FadeIn(eq12[:2]))
        self.wait(0.1)
        self.play(mh.rtransform(eq12, eq12_1))
        self.wait(0.1)
        self.play(FadeOut(eq10[1:]), mh.rtransform(eq10[0], eq13[4], run_time=1.6),
                  mh.rtransform(eq12_1[0][0].copy(), eq13[0][0],
                                eq12_1[0].copy(), eq13[2], run_time=1.6),
                  FadeIn(eq13[0][1:].set_color(col_ft), shift=mh.diff(eq12_1[0][0], eq13[0][0]), run_time=1.6),
                  Succession(Wait(0.8), FadeIn(eq13[1], eq13[3], eq13[-1]))
                  )
        self.wait(0.1)
        self.play(FadeOut(eq13[:2], eq13[3:]),
                  mh.rtransform(eq13[2], eq14[2], run_time=1.4),
                  Succession(Wait(0.4), FadeIn(eq14[:2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq12_1[2:].copy(), eq15[2:], eq14[:2], eq15[:2]), FadeOut(eq14[2]))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq15[:2], eq16[:2], eq15[2:], eq16[3:]),
                  mh.stretch_replace(eq15[3][-4:].copy(), eq16[2][-4:], path_arc=-PI/2.5),
                  run_time=1.5),
                  Succession(Wait(0.7), FadeIn(eq16[2][:-4].set_color(col_op))))
        self.wait(0.1)
        line1 = Line(eq16.get_corner(DL)+DL*0.1, eq16.get_corner(UR+UR*0.1), stroke_width=8, stroke_color=RED).set_z_index(5)
        self.play(Create(line1, rate_func=linear, run_time=0.7))
        self.wait(0.1)
        self.play(FadeOut(line1, eq16))
        self.wait(0.1)
        self.play(FadeIn(eq17))
        self.wait(0.1)
        self.play(FadeOut(eq17[0]),
                  mh.rtransform(eq17[1:3], eq18[1:3], eq17[3], eq18[4], run_time=1.3),
                  Succession(Wait(0.5), mh.rtransform(eq12_1[-1].copy(), eq18[0], run_time=1.7)),
                  Succession(Wait(2), FadeIn(eq18[3])))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq18[:3], eq19[:3], eq18[3][0], eq19[3][0], eq18[3][1:4], eq19[3][2:5],
                                eq18[3][4:9], eq19[3][8:13], eq18[3][-2:], eq19[3][-2:]
                                ),
                  FadeOut(eq18[4][0]),
                  mh.rtransform(eq18[4][1], eq19[3][1], eq18[4][-2:], eq19[3][-2:]),
                  mh.fade_replace(eq18[4][2:5], eq19[3][6].set_color(col_num), coord_mask=RIGHT),
                  FadeIn(eq19[3][7], shift=mh.diff(eq18[3][4], eq19[3][8])),
                  run_time=1.3),
            Succession(Wait(0.8), FadeIn(eq19[3][5], eq19[3][-3])))
        self.wait(0.1)
        self.play(mh.rtransform(eq19[:3], eq20[1:4], eq19[3], eq20[5]),
                  Succession(Wait(0.5),
                             mh.rtransform(eq12_1[-2].copy(), eq20[0], eq12_1[-2].copy(), eq20[4],
                                           run_time=1.5)))
        self.wait(0.1)
        eq20_1 = eq20[5][6:13]
        VGroup(eq21[5][5], eq21[4][0]).set_color(col_num)
        eq21[4][2:-7].set_color(col_op)
        self.play(AnimationGroup(
            mh.rtransform(eq20[:4], eq21[:4], eq20[5][:5], eq21[5][:5],
                          eq20[5][-2:], eq21[5][-2:],
                          ),
            stretch_replace_each(eq20_1, eq21[5][7:14]),
            stretch_replace_each(eq20_1.copy(), eq21[4][-7:]),
            eq12_1.animate.shift(UP*0.5),
            # mh.rtransform(eq20[5][6:13].copy(), eq21[4][-7:]),
            # FadeIn(eq21[5][-4:-2].set_color(col_num), shift=mh.diff(eq20[5][-5], eq21[5][-3])),
            run_time=1.5),
            Succession(Wait(1), FadeIn(eq21[5][5:7])),
            Succession(Wait(1), FadeIn(eq21[4][:-7])),
            FadeOut(eq20[4]),
            FadeOut(eq20[5][5], eq20[5][13]))

        self.play(mh.rtransform(eq21[:4], eq22[2:6], run_time=1.2),
                  Succession(Wait(0.8), mh.rtransform(eq12_1[2:4].copy(), eq22[:2], run_time=1.5)))
        self.wait(0.1)
        circ1 = mh.circle_eq(eq21[-2], scale=0.6).shift(DOWN*0.15)
        self.play(Create(circ1, run_time=0.8, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(eq21[-1][5:-2]),
                  Succession(Wait(0.5), AnimationGroup(
                      mh.rtransform(eq21[-2], eq22[-2], eq21[-1][:5], eq22[-1][:5],
                                    eq21[-1][-2:], eq22[-1][-2:]),
                      circ1.animate.shift(mh.diff(eq21[-2], eq22[-2])),
                      run_time=1.2)))
        self.wait(0.1)
        self.play(FadeOut(eq22[:-2], eq22[-1], eq22[-2][:2]),
                  Succession(Wait(0.3), eq22[-2][2:].animate.shift(UP*0.36+LEFT*0.15)))
        self.wait(0.1)
        self.play(FadeOut(*gps, *cols),
                  mh.fade_replace(eq12_1[0], eq23[0]),
                  mh.rtransform(eq12_1[1], eq23[1], eq12_1[2:], eq23[3:],
                                eq22[-2][2:], eq23[2][:]),
                  FadeOut(circ1, shift=mh.diff(eq22[-2][2:], eq23[2][:])),
                  run_time=1.8)

        # self.play(FadeOut(eq1), FadeIn(eq10))
        self.wait()

    def do_anims1(self, eq_only=False):
        # eq1, _ = self.do_anim(just_eq=True)
        MathTex.set_default(stroke_width=1.5, font_size=70)
        eq1 = MathTex(r'X\mathcal F', r'=', r'\mathcal F P', font_size=80)
        eq2 = MathTex(r'P\mathcal F', r'=', r'-\mathcal F X', font_size=80)
        eq3 = MathTex(r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!\mathcal F', r'=',
                      r'\mathcal F\!', r'\begin{pmatrix} P \\ -X\end{pmatrix}', font_size=80)

        eq4 = MathTex(r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!\mathcal F', r'=',
                      r'\mathcal F\!', r'\begin{pmatrix} 0 & 1 \\ -1 & 0\end{pmatrix}',
                      r'\begin{pmatrix}X \\ P\end{pmatrix}', font_size=80)

        eq5 = MathTex(r'S_\alpha f(x)', r'=', r'f(\alpha x)', font_size=80)
        eq6 = MathTex(r'\int\lvert', r'S_\alpha f(x)', r'\rvert^2\,dx', r'=',
                      r'\int\lvert', r'f(\alpha x)', r'\rvert^2\,dx', font_size=80)
        eq7 = MathTex(r'\int\lvert', r'S_\alpha f(x)', r'\rvert^2\,dx', r'=',
                      r'\int\lvert', r'f(x)', r'\rvert^2\,\frac{dx}{\lvert\alpha\rvert}', font_size=80)
        eq8 = MathTex(r'\lVert', r'S_\alpha f', r'\rVert^2', r'=', r'\lVert', r'f', r'\rVert^2/\lvert\alpha\rvert')
        eq9 = MathTex(r'\lVert', r'\sqrt{\alpha}', r'S_\alpha f', r'\rVert^2', r'=', r'\lVert', r'f', r'\rVert^2')
        eq10 = MathTex(r'S_\alpha X', r'f(x)', r'=', r'S_\alpha x', r'f(x)', r'=', r'\alpha x', r'f(\alpha x)',
                       r'=', r'\alpha X', r'f(\alpha x)', r'=', r'\alpha XS_\alpha', r'f(x)', font_size=80)
        eq11 = MathTex(r'XS_\alpha', r'=', r'S_\alpha X', r'/\alpha', font_size=80)
        eq12 = MathTex(r'DS_\alpha', r'f(x)', r'=', r'D', r'(f(\alpha x))', r'=', r'\alpha', r'(Df)(\alpha x)',
                       r'=', r'\alpha S_\alpha', r'(Df)(x)', font_size=80)
        eq13 = MathTex(r'PS_\alpha', r'=', r'\alpha S_\alpha P', font_size=80)
        eq14 = MathTex(r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!S_\alpha', r'=',
                      r'S_\alpha\!', r'\begin{pmatrix} X/\alpha \\ \alpha P\end{pmatrix}', font_size=80)
        eq15 = MathTex(r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!S_\alpha', r'=',
                      r'S_\alpha\!', r'\begin{pmatrix} 1/\alpha & 0 \\ 0 & \alpha\end{pmatrix}',
                      r'\begin{pmatrix}X \\ P\end{pmatrix}', font_size=80)

        eq16 = MathTex(r'C_{\alpha}', r'f(x)', r'=', r'e^{\frac12i\alpha x^2}', r'f(x)', font_size=80)
        eq17 = MathTex(r'C_{\alpha}X', r'f(x)', r'=', r'e^{\frac12i\alpha x^2}x', r'f(x)', font_size=80)
        eq18 = MathTex(r'XC_{\alpha}', r'=', r'C_{\alpha}X', font_size=80)
        eq19 = MathTex(r'D', r'C_{\alpha}', r'f(x)', r'=', r'D(', r'e^{\frac12i\alpha x^2}', r'f(x)', r')', font_size=80)
        eq20 = MathTex(r'D', r'C_{\alpha}', r'f(x)', r'\!=\!', r'(\!D', r'e^{\frac12i\alpha x^2}', r'\!)', r'f(x)', r'\!+\!',
                       r'e^{\frac12i\alpha x^2}', r'\!\!D', r'f(x)', font_size=80)
        eq21 = MathTex(r'D', r'C_{\alpha}', r'f(x)', r'\!=\!', r'e^{\frac12i\alpha x^2}', r'\!i\alpha x', r'f(x)', r'\!+\!',
                       r'e^{\frac12i\alpha x^2}', r'\!\!D', r'f(x)', font_size=80)
        eq22 = MathTex(r'D', r'C_\alpha', r'=', r'C_\alpha', r'(', r'i\alpha X', r'+', r'D', r')', font_size=80)
        eq23 = MathTex(r'P', r'C_\alpha', r'=', r'C_\alpha', r'(', r'\alpha X', r'+', r'P', r')', font_size=80)
        tiny = 0.05
        for i,j in [(2,1), (2,2), (2,3), (7,1), (7,2), (7,3), (-1,1), (-1,2), (-1,3), (-2,1)]:
            (eq20[:i] + eq20[i][:j]).shift(RIGHT*tiny)
        for i,j in [(2,1), (2,2), (2,3), (6,1), (6,2), (6,3), (-1,1), (-1,2), (-1,3), (-2,1)]:
            (eq21[:i] + eq21[i][:j]).shift(RIGHT * tiny)
        eq20.move_to(ORIGIN)
        eq21.move_to(ORIGIN)
        eq24 = MathTex(r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!C_\alpha', r'=',
                      r'C_\alpha\!', r'\begin{pmatrix} X \\ \alpha X+P\end{pmatrix}', font_size=80)
        eq24[4][1].move_to(eq24[4][3], coor_mask=RIGHT)
        eq25 = MathTex(r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ '
                      r'P\end{pmatrix}', r'\!C_\alpha', r'=',
                      r'C_\alpha\!', r'\begin{pmatrix} 1 & 0 \\ \alpha & 1\end{pmatrix}',
                      r'\begin{pmatrix}X \\ P\end{pmatrix}', font_size=80)

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True
        VGroup(eq1[0][0], eq2[2][2], eq5[0][-2], eq5[2][-2], eq6[2][-1],
               eq10[0][2], eq10[1][2], eq16[3][-2:]).set_color(col_x)
        VGroup(eq1[0][1], eq1[2][0], eq2[0][1], eq2[2][1], eq10[1][0]).set_color(col_ft)
        VGroup(eq1[2][1], eq2[0][0], eq12[0][0], eq19[0][0], eq19[4][0]).set_color(col_p)
        VGroup(eq4[4][1:-1], eq6[2][-3], eq16[3][1:4:2]).set_color(col_num)
        VGroup(eq5[0][2], eq5[2][0]).set_color(col_psi)
        VGroup(eq5[0][1], eq5[2][2], eq10[0][1], eq12[0][2], eq16[0][1], eq16[3][5]).set_color(col_angle)
        VGroup(eq6[0], eq6[2][::2], eq6[4], eq7[-1][-3::2], eq16[3][2]).set_color(col_op)
        VGroup(eq16[3][4]).set_color(col_i)
        VGroup(eq5[0][0], eq10[0][0], eq10[12][2], eq12[0][1], eq12[9][1]).set_color(col_squeeze)
        VGroup(eq16[0][0]).set_color(col_chirp)
        VGroup(eq16[3][0]).set_color(col_special)
        mh.copy_colors_eq(eq6[2], eq6[6])
        mh.copy_colors_eq(eq10[1], eq12[1])
        mh.copy_colors_eq(eq5[0][-4:], eq16[1][:], eq5[0][-4:], eq16[-1][:])
        mh.copy_colors_eq(eq16[:3], eq19[1:4], eq16[3:], eq19[5:7])

        mh.align_sub(eq2, eq2[1], eq1[1]).next_to(eq1, DOWN, buff=0.5, coor_mask=UP)
        VGroup(eq1, eq2).move_to(ORIGIN)
        mh.align_sub(eq3, eq3[2], VGroup(eq1[1], eq2[1]), coor_mask=UP)
        mh.align_sub(eq4, eq4[2], eq3[2], coor_mask=UP)
        eq5.move_to(DOWN)
        mh.align_sub(eq6, eq6[3], eq5[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[3], eq6[3])
        mh.align_sub(eq8, eq8[3], eq6[3], coor_mask=UP)
        mh.align_sub(eq9, eq9[4], eq8[3])

        mh.align_sub(eq10, eq10[2], eq9[4])
        mh.align_sub(eq10, eq10[:5], ORIGIN, coor_mask=RIGHT)
        eq10[5:].align_to(eq10[2], LEFT)
        eq10[8:].align_to(eq10[5], LEFT)
        eq10[11:].align_to(eq10[8], LEFT)
        mh.align_sub(eq11, eq11[1], eq10[2], coor_mask=UP)

        eq12.next_to(eq11, DOWN, buff=0.5)
        mh.align_sub(eq12, eq12[:5], ORIGIN, coor_mask=RIGHT)
        eq12[5:].align_to(eq12[2], LEFT)
        eq12[8:].align_to(eq12[5], LEFT)

        mh.align_sub(eq13, eq13[1], eq12[2])
        mh.align_sub(eq13, eq13[1], eq11[1], coor_mask=RIGHT)

        mh.align_sub(eq14, eq14[2], VGroup(eq11[1], eq13[1]))
        mh.align_sub(eq15, eq15[2], eq14[2], coor_mask=UP)

        mh.align_sub(eq16, eq16[2], eq5[1], coor_mask=UP)
        mh.align_sub(eq17, eq17[2], eq16[2], coor_mask=UP)
        mh.align_sub(eq18, eq18[1], eq17[2], coor_mask=UP)
        eq19.next_to(eq18, DOWN, buff=0.5)
        mh.align_sub(eq20, eq20[3], eq19[3], coor_mask=UP)
        mh.align_sub(eq21, eq21[3], eq19[3], coor_mask=UP)
        mh.align_sub(eq22, eq22[2], eq18[1])
        mh.align_sub(eq22, eq22[2], eq21[3], coor_mask=UP)
        mh.align_sub(eq23, eq23[2], eq22[2])
        mh.align_sub(eq24, eq24[2], VGroup(eq18[1], eq23[2]))
        mh.align_sub(eq25, eq25[2], eq24[2], coor_mask=UP)

        # construct table
        MathTex.set_default(stroke_width=1.5, font_size=70)
        eq_t1 = MathTex(r'{\sf operator}', r'\mathcal F', r'S_\alpha', r'C_\alpha', font_size=70).set_z_index(5)
        eq_t2 = MathTex(r'{\sf matrix}', r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix} 0 & 1 \\ -1 & 0\end{pmatrix}',
                        r'\begin{pmatrix}1/\alpha & 0 \\ 0 & \alpha\end{pmatrix}',
                        r'\begin{pmatrix}1 & 0 \\ \alpha & 1\end{pmatrix}', font_size=55).set_z_index(5)
        mh.font_size_sub(eq_t1, 0, 60)
        mh.font_size_sub(eq_t2, 0, 60)
        eq_ = eq_t1[0].copy().move_to(eq_t2[0])
        VGroup(eq_, *eq_t2[1:]).arrange(RIGHT, buff=1)
        eq_t2[0].move_to(eq_)

        eq_t1.to_edge(UP, buff=0.4)
        eq_t2.next_to(eq_t1, DOWN)

        n_t = len(eq_t1)
        centers = np.zeros(n_t+1)
        widths = np.zeros(n_t)
        gps = []
        for i in range(n_t):
            eq_t1[i].move_to(eq_t2[i], coor_mask=RIGHT)
            gps.append(VGroup(eq_t1[i], eq_t2[i]))
        gps[0].set_color(col_txt)

        for i in range(n_t-1):
            centers[i+1] = (gps[i].get_right() + gps[i+1].get_left())[0]/2
        centers[0] = -centers[1] + 2* gps[0].get_center()[0]
        centers[-1] = -centers[-2] + 2*gps[-1].get_center()[0]

        centers[0] += 0.4

        widths = [centers[i+1] - centers[i] for i in range(n_t)]
        for i in range(n_t):
            gps[i].move_to((centers[i] + centers[i+1])/2*RIGHT, coor_mask=RIGHT)

        center1 = VGroup(eq_t1, eq_t2).get_center()
        rects = [Rectangle(width=widths[i], height=3, stroke_width=6, stroke_opacity=1, fill_opacity=0)
                 .move_to(gps[i].get_center()*RIGHT + center1*UP).set_z_index(4)
                 for i in range(n_t)]
        rects2 = [_.copy().set_stroke(opacity=0).set_fill(opacity=1, color=DARKER_GREY).set_z_index(0) for _ in rects]
        rects2[0].set_fill(color=DARK_GREY)
        pt0 = (eq_t1.get_bottom() + eq_t2.get_top())/2*UP
        box_lines = [Line(centers[i]*RIGHT + pt0, centers[i+1]*RIGHT+pt0, stroke_width=6, stroke_opacity=1)
                     .set_z_index(4) for i in range(n_t)]
        cols = [VGroup(rects[i], box_lines[i], rects2[i]) for i in range(n_t)]

        if eq_only:
            self.add(*cols, *gps)
            return gps, cols

        self.add(eq1, eq2)
        self.play(AnimationGroup(mh.rtransform(eq1[0][0], eq3[0][1], eq2[0][0], eq3[0][2], eq1[0][1], eq3[1],
                                eq1[1], eq3[2], eq1[2][0], eq3[3][0], eq1[2][1], eq3[4][1],
                                eq2[2][0], eq3[4][2], eq2[2][2], eq3[4][3]),
                  mh.rtransform(eq2[0][1], eq3[1], eq2[1], eq3[2], eq2[2][1], eq3[3][0]),
                  run_time=1.2),
                  Succession(Wait(0.6), FadeIn(eq3[0][0], eq3[0][-1], eq3[4][0], eq3[4][-1])))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq3[:4], eq4[:4], eq3[4][0], eq4[4][0], eq3[4][-1], eq4[4][-1],
                                eq3[4][1], eq4[5][2], eq3[4][3], eq4[5][1]),
                                 mh.rtransform(eq3[4][2], eq4[4][3], copy_colors=False),
                  FadeIn(eq4[4][2], target_position=eq3[4][1]),
                  FadeIn(eq4[4][4], target_position=eq3[4][3]),
                                 run_time=1.3),
                  Succession(Wait(0.6), FadeIn(eq4[4][1], eq4[4][5], eq4[5][0], eq4[5][-1])))
        self.wait(0.1)
        eq4.set_z_index(5)
        self.play(AnimationGroup(mh.rtransform(eq4[1], gps[1][0], eq4[4], gps[1][1]),
                  mh.rtransform(eq4[3], gps[1][0]), run_time=1.7),
                  Succession(Wait(1.), FadeIn(gps[0], cols[0], cols[1])),
                  FadeOut(eq4[0], eq4[2], eq4[5]))
        self.wait(0.1)
        self.play(FadeIn(eq5))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:], eq6[1::2], run_time=1.5),
                  Succession(Wait(0.5), FadeIn(eq6[::2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[:5], eq7[:5], eq6[5][:2], eq7[5][:2], eq6[5][-2:], eq7[5][-2:],
                                eq6[6][:], eq7[6][:4], eq6[5][2], eq7[6][-2]),
                  Succession(Wait(0.4), FadeIn(eq7[6][-1], eq7[6][-4:-2])))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq7[1][:3], eq8[1][:], eq7[2][1], eq8[2][1], eq7[3], eq8[3],
                                eq7[5][0], eq8[5], eq7[6][1], eq8[6][1], eq7[-1][-3:], eq8[-1][-3:]),
                  mh.stretch_replace(eq7[0][1], eq8[0]),
                  mh.stretch_replace(eq7[2][0], eq8[2][0]),
                  mh.stretch_replace(eq7[4][1], eq8[4]),
                  mh.stretch_replace(eq7[6][0], eq8[6][0]),
                  FadeIn(eq8[-1][-4], shift=mh.diff(eq7[-1][-3:], eq8[-1][-3:])),
                                 run_time=1.6),
                  FadeOut(eq7[0][0], eq7[1][3:], eq7[2][-2:],
                          eq7[4][0], eq7[5][1:], eq7[6][2:5]),
                  )
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq8[0], eq9[0], eq8[1:-1], eq9[2:-1], eq8[-1][:2], eq9[-1][:]),
                                mh.rtransform(eq8[-1][-2], eq9[1][-1], path_arc=-PI/4),
                  FadeOut(eq8[-1][-3::2], shift=mh.diff(eq8[-1][-2], eq9[1][-1]), path_arc=-PI/4),
                  FadeIn(eq9[1][:-1].set_color(col_op), shift=mh.diff(eq8[-1][-2], eq9[1][-1]), path_arc=-PI/4),
                                 run_time=1.4),
                  FadeOut(eq8[-1][-4]),
                  # Succession(Wait(1.1), FadeIn(eq9[1][:-1].set_color(col_op))),
                  )
        self.wait(0.1)
        eq10_1 = eq10[:2].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(FadeIn(eq10_1), FadeOut(eq9))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq10_1[:2], eq10[:2], eq10_1[1].copy(), eq10[4],
                  eq10_1[0][:2].copy(), eq10[3][:2]),
                  mh.stretch_replace(eq10_1[0][2].copy(), eq10[3][2]),
                                 run_time=1.4),
                  Succession(Wait(0.6), FadeIn(eq10[2]))
                  )
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq10[4][:2], eq10[7][:2], eq10[4][-2:], eq10[7][-2:],
                                eq10[3][2], eq10[6][1]),
                  mh.stretch_replace(eq10[3][1], eq10[6][0]),
                  mh.stretch_replace(eq10[3][1].copy(), eq10[7][2], path_arc=PI/3),
                                 run_time=1.2),
                  FadeOut(eq10[3][0]))
        self.wait(0.1)
        self.play(mh.rtransform(eq10[7], eq10[10], eq10[6][0], eq10[9][0]),
                  mh.stretch_replace(eq10[6][1], eq10[9][1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq10[10][:2], eq10[13][:2], eq10[10][-2:], eq10[13][-2:],
                                eq10[9][:2], eq10[12][:2]),
                  mh.stretch_replace(eq10[10][2], eq10[12][3]),
                  FadeIn(eq10[12][2]))
        self.wait(0.1)
        self.play(FadeOut(eq10[1], eq10[13]))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq10[0][:3], eq11[2][:], eq10[2], eq11[1], eq10[12][1:], eq11[0][:]),
                                 mh.rtransform(eq10[12][0], eq11[3][1], path_arc=PI/2), run_time=1.4),
                  Succession(Wait(0.6), FadeIn(eq11[3][0].set_color(col_op))))
        self.wait(0.1)
        eq12_2 = eq12[:2].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(FadeIn(eq12_2))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq12_2, eq12[:2], eq12_2[0][0].copy(), eq12[3][0], eq12_2[1][:2].copy(), eq12[4][1:3],
                          eq12_2[1][-2:].copy(), eq12[4][-3:-1]),
            mh.stretch_replace(eq12_2[0][2].copy(), eq12[4][3]),
                  run_time=1.4),
            Succession(Wait(0.6), FadeIn(eq12[4][0], eq12[4][-1], eq12[2])),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq12[3][0], eq12[7][1], eq12[4][1], eq12[7][2], eq12[4][0], eq12[7][0],
                                eq12[4][2:6], eq12[7][4:8]),
                  mh.rtransform(eq12[4][3].copy(), eq12[6][0], path_arc=-PI/3),
                  FadeOut(eq12[4][-1], shift=mh.diff(eq12[4][5], eq12[7][7])),
                  FadeIn(eq12[7][3], shift=mh.diff(eq12[4][1], eq12[7][2])),
                  run_time=1.2)
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq12[6][0], eq12[9][0], eq12[7][:5], eq12[10][:5],
                                               eq12[7][-2:], eq12[10][-2:]),
                  mh.stretch_replace(eq12[7][5], eq12[9][2], path_arc=-PI/3),
                  run_time=1.2),
                  Succession(Wait(0.4), FadeIn(eq12[9][1])))
        self.wait(0.1)
        self.play(FadeOut(eq12[1], eq12[10][0], eq12[10][2:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq12[9][:], eq13[2][:3], eq12[2], eq13[1], eq12[0][1:], eq13[0][1:]),
                  mh.stretch_replace(eq12[0][0], eq13[0][0]),
                  mh.stretch_replace(eq12[10][1], eq13[2][3]))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq11[0][0], eq14[0][1], eq13[0][0], eq14[0][2], eq11[0][1:3], eq14[1][:],
                                eq11[1], eq14[2], eq11[2][:2], eq14[3][:], eq11[2][2], eq14[4][1],
                                eq11[3][:], eq14[4][2:4], eq13[2][0], eq14[4][4], eq13[2][3], eq14[4][5]
                                ),
                  mh.rtransform(eq13[0][1:3], eq14[1][:], eq13[1], eq14[2], eq13[2][1:3], eq14[3][:]),
                                 run_time=1.2),
                  Succession(Wait(0.6), FadeIn(eq14[0][0], eq14[0][-1], eq14[4][0], eq14[4][-1])))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq14[:4], eq15[:4], eq14[4][0], eq15[4][0], eq14[4][-1], eq15[4][-1],
                                eq14[4][1], eq15[5][1], eq14[4][5], eq15[5][2],
                                eq14[4][2:4], eq15[4][2:4], eq14[4][4], eq15[4][6]
                                ),
                  FadeIn(eq15[4][1].set_color(col_num), shift=mh.diff(eq14[4][2:4], eq15[4][2:4])),
                                 run_time=1.3),
                  Succession(Wait(0.5), FadeIn(eq15[4][4:6].set_color(col_num), eq15[5][0], eq15[5][-1])))
        self.wait(0.1)
        eq15.set_z_index(5)
        self.play(AnimationGroup(mh.rtransform(eq15[1], gps[2][0], eq15[4], gps[2][1]),
                  mh.rtransform(eq15[3], gps[2][0]), run_time=1.7),
                  Succession(Wait(1.), FadeIn(cols[2])),
                  FadeOut(eq15[0], eq15[2], eq15[5]))

        self.wait(0.1)
        self.play(FadeIn(eq16))
        self.wait(0.1)
        self.play(mh.rtransform(eq16[0][:], eq17[0][:-1], eq16[1:3], eq17[1:3], eq16[3][:], eq17[3][:-1], eq16[4], eq17[4]),
                  Succession(Wait(0.3), FadeIn(VGroup(eq17[0][-1], eq17[3][-1]).set_color(col_x))))
        self.play(eq17[0][:-1].animate.align_to(eq17[0], RIGHT),
                  eq17[0][-1].animate.align_to(eq17[0], LEFT),
                  eq17[3][:-1].animate.align_to(eq17[3], RIGHT),
                  eq17[3][-1].animate.align_to(eq17[3], LEFT))
        self.wait(0.1)
        self.play(FadeOut(eq17[1], eq17[3:]),
                  mh.rtransform(eq17[0].copy(), eq18[2], eq17[2], eq18[1], eq17[0][-1], eq18[0][0],
                                eq17[0][:-1], eq18[0][1:], run_time=1.3
                                ))
        self.wait(0.1)
        self.play(FadeIn(eq19))
        self.wait(0.1)
        eq_ = eq19.copy()
        self.play(AnimationGroup(
            mh.rtransform(eq19[:4], eq20[:4], eq19[4][1::-1], eq20[4][:2], eq19[5], eq20[5],
                                eq19[6], eq20[7], eq19[7], eq20[6]),
                  mh.rtransform(eq_[4][0], eq20[10][0], eq_[5], eq20[9], eq_[6], eq20[11]),
            run_time=1.6),
            Succession(Wait(1), FadeIn(eq20[8])))
        self.wait(0.1)
        self.play(mh.rtransform(eq20[:4], eq21[:4], eq20[5], eq21[4],
                                eq20[7:], eq21[6:]),
                  FadeOut(eq20[4], eq20[6]),
                  *[mh.stretch_replace(eq20[5][4+i].copy(), eq21[5][i]) for i in (0,1,2)],
                  run_time=1.2)
        self.wait(0.1)
        eq_1 = [eq21[4], eq21[8]]
        eq_2 = [eq21[1].copy().move_to(_, coor_mask=RIGHT) for _ in eq_1]
        self.play(FadeOut(*eq_1), FadeIn(*eq_2))
        self.wait(0.1)
        eq_3 = MathTex(r'i\alpha X', font_size=80)
        mh.align_sub(eq_3, eq_3[0][0], eq21[5][0]).align_to(eq21[5], RIGHT)
        self.play(mh.rtransform(eq21[5][:2], eq_3[0][:2]),
                  mh.stretch_replace(eq21[5][2], eq_3[0][2]))
        self.wait(0.1)
        self.play(FadeOut(eq21[2], eq21[6], eq21[-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq21[:2], eq22[:2], eq21[3], eq22[2], eq_2[0], eq22[3], eq_3[0], eq22[5],
                                eq21[7], eq22[6], eq21[9], eq22[7]),
                  mh.rtransform(eq_2[1], eq22[3], path_arc=-PI/2.3),
                  FadeIn(eq22[-1], shift=mh.diff(eq21[9], eq22[7])),
                  FadeIn(eq22[4], shift=mh.diff(eq_3[0], eq22[5])),
                  run_time=1.7
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq22[1:5], eq23[1:5], eq22[5][1:], eq23[5][:],
                                eq22[6], eq23[6], eq22[8], eq23[8]),
                  FadeOut(eq22[5][0]),
                  mh.stretch_replace(eq22[0], eq23[0]),
                  mh.stretch_replace(eq22[7], eq23[7]))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq18[1], eq24[2], eq18[0][0], eq24[0][1], eq18[0][1:], eq24[1][:],
                                eq18[2][:2], eq24[3][:], eq18[2][2], eq24[4][1]),
            mh.rtransform(eq23[2], eq24[2], eq23[0][0], eq24[0][2], eq23[1][:], eq24[1][:],
                          eq23[3], eq24[3], eq23[5][:], eq24[4][2:4], eq23[6][0], eq24[4][4], eq23[7][0], eq24[4][5],
                          eq23[4][0], eq24[4][0], eq23[8][0], eq24[4][-1]),
            run_time=1.2),
        Succession(Wait(0.6), FadeIn(eq24[0][0], eq24[0][-1])))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq24[:4], eq25[:4], eq24[4][0], eq25[4][0], eq24[4][1], eq25[5][1], eq24[4][-1], eq25[4][-1]),
                  mh.rtransform(eq24[4][2], eq25[4][3], eq24[4][3], eq25[5][1], eq24[4][5], eq25[5][2]),
                  FadeOut(eq24[4][4], target_position=eq25[4][3:5]),
                  FadeIn(eq25[4][1].set_color(col_num), target_position=eq24[4][1]),
                  FadeIn(eq25[4][4].set_color(col_num), target_position=eq24[4][5]),
                  FadeIn(eq25[4][2].set_color(col_num), shift=mh.diff(eq24[4][5], eq25[4][4])),
            run_time=1.3),
            Succession(Wait(0.6), FadeIn(eq25[-1][0], eq25[-1][-1]))
                  )
        self.wait(0.1)
        eq25.set_z_index(5)
        self.play(AnimationGroup(mh.rtransform(eq25[1], gps[3][0], eq25[4], gps[3][1]),
                  mh.rtransform(eq25[3], gps[3][0]), run_time=1.7),
                  Succession(Wait(1.), FadeIn(cols[3])),
                  FadeOut(eq25[0], eq25[2], eq25[5]))

        return gps, cols

class FractionalFTResult(Scene):
    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=1.6)
        eq1 = MathTex(r'\mathcal F_\theta', r'=', r'\sqrt{1-i\cot\theta}\,', r'C_{\cot\theta}', r'S_{\csc\theta}',
                      r'\mathcal F', r'C_{\cot\theta}')
        eq2 = MathTex(r'\mathcal F^{-1}_\theta', r'\renewcommand*{\arraystretch}{1.2}\begin{pmatrix}X \\ P\end{pmatrix}',
                      r'\mathcal F_\theta' ,r'=', r'\begin{pmatrix}\cos\theta & \sin\theta \\ -\sin\theta & \cos\theta\end{pmatrix}',
                      r'\begin{pmatrix}X \\ P\end{pmatrix}', font_size=70)
        eq3 = MathTex(r'\mathcal F_\theta', r'e^{-\frac12x^2}', r'=', r'e^{-\frac12x^2}')

        eq4 = MathTex(r'C_{\cot\theta}', r'f(x)', r'=', r'e^{\frac12ix^2\cot\theta}', r'f(x)')
        eq5 = MathTex(r'\mathcal F', r'C_{\cot\theta}', r'f(x)', r'=',
                      r'\frac1{\sqrt{2\pi} }', r'\int', r'e^{-ixy}', r'e^{\frac12iy^2\cot\theta}', r'f(y)', r'\,dy')
        mh.font_size_sub(eq5, 4, 70)
        eq6 = MathTex(r'e^{\frac12iy^2\cot\theta-ixy}')
        eq7 = MathTex(r'S_{\csc\theta}', r'\mathcal F', r'C_{\cot\theta}', r'f(x)', r'=',
                      r'\frac1{\sqrt{2\pi} }', r'\int', r'e^{\frac12iy^2\cot\theta-ix\csc\theta\,y}', r'f(y)', r'\,dy')
        mh.font_size_sub(eq7, 5, 70)
        eq8 = MathTex(r'e^{\frac12iy^2\cot\theta-ixy\csc\theta}')
        eq9 = MathTex(r'C_{\cot\theta}', r'S_{\csc\theta}', r'\mathcal F', r'C_{\cot\theta}', r'f(x)', r'=',
                      r'\frac1{\sqrt{2\pi} }', r'\int', r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}', r'f(y)', r'\,dy')
        mh.font_size_sub(eq9, 6, 70)
        eq10 = MathTex(r'\mathcal F_\theta', r'f(x)', r'=',
                       r'\sqrt{\frac{1-i\cot\theta}{2\pi} }', r'\!\!\int\!', r'e^{\frac12i(x^2+y^2)\cot\theta-ixy\csc\theta}', r'f(y)',
                       r'\,dy', font_size=75)
        mh.font_size_sub(eq10, 3, 60)


        mh.rtransform.copy_colors = True
        VGroup(eq1[::5], eq2[0][:3]).set_color(col_ft)
        VGroup(*[eq1[i][-1] for i in [0, 2, 3, 4, 6]], eq2[0][-1]).set_color(col_angle)
        VGroup(*[eq1[i][-4:-1] for i in [2, 3, 4, 6]]).set_color(col_trig)
        VGroup(eq1[2][:-7], eq3[1][3], eq5[4][1:-2], eq5[5][0], eq5[9][-2]).set_color(col_op)
        VGroup(eq1[2][-7], eq3[1][2:5:2], eq5[4][0]).set_color(col_num)
        VGroup(eq1[2][-5], eq4[3][4], eq5[6][2], eq5[7][4]).set_color(col_i)
        VGroup(eq1[3][0], eq1[-1][0]).set_color(col_chirp)
        VGroup(eq1[4][0]).set_color(col_squeeze)
        VGroup(eq2[1][1], eq3[1][-2:], eq4[1][2], eq5[6][3]).set_color(col_x)
        VGroup(eq2[1][2], eq5[6][4], eq5[7][5:7], eq5[8][2], eq5[9][-1]).set_color(col_p)
        VGroup(eq3[1][0], eq5[6][0], eq5[7][0]).set_color(col_special)
        VGroup(eq4[1][0]).set_color(col_psi)
        VGroup(eq5[4][-2:]).set_color(col_pi)

        mh.copy_colors_eq(eq1[0], eq2[2])
        mh.copy_colors_eq(eq1[3][-4:], eq2[4][1:5])
        mh.copy_colors_eq(eq2[1], eq2[-1])
        mh.copy_colors_eq(eq2[4][1:5], eq2[4][5:9])
        mh.copy_colors_eq(eq2[4][1:5], eq2[4][10:14], eq2[4][1:5], eq2[4][14:18])
        mh.copy_colors_eq(eq1[0], eq3[0])
        mh.copy_colors_eq(eq3[1], eq3[-1])
        mh.copy_colors_eq(eq4[1], eq4[-1], eq3[1][0], eq4[3][0], eq3[1][2:5], eq4[3][1:4], eq3[1][-2:], eq4[3][5:7], eq1[-1][-4:], eq4[3][-4:])
        mh.copy_colors_eq(eq1[4][-4:], eq7[7][-5:-1])
        mh.copy_colors_eq(eq1[0], eq10[0])

        eq2.next_to(eq1, DOWN, buff=0.8, coor_mask=UP)
        gp1 = VGroup(eq1.copy(), eq2).move_to(ORIGIN, coor_mask=UP)
        eq3.next_to(gp1, DOWN, buff=0.8, coor_mask=UP)
        gp2 = VGroup(*gp1[:2].copy(), eq3).move_to(ORIGIN, coor_mask=UP)

        mh.align_sub(eq5, eq5[3], eq4[2])
        eq5[:4].shift(UP*0.5)
        eq5[4:].to_edge(RIGHT, buff=0.4).shift(DOWN*1.5)
        mh.align_sub(eq6[0], eq6[0][0], eq5[7][0]).align_to(eq5[7], RIGHT)
        mh.align_sub(eq7, eq7[4], eq5[3]).to_edge(LEFT, buff=0.3)
        mh.align_sub(eq7[5:], eq7[7][0], eq6[0][0]).align_to(eq5, RIGHT)
        mh.align_sub(eq8, eq8[0][0], eq7[7][0])
        mh.align_sub(eq9, eq9[5], eq7[4]).to_edge(LEFT, buff=0.3)
        mh.align_sub(eq9[6:], eq9[8][0], eq7[7][0]).align_to(eq7, RIGHT)
        mh.align_sub(eq10, eq10[2], eq5[3]).shift(LEFT)
        mh.align_sub(eq10[3:], eq10[-1], eq9[-1]).move_to(ORIGIN, coor_mask=RIGHT)

        self.add(eq1)
        self.play(mh.rtransform(eq1, gp1[0], run_time=1.2),
                  Succession(Wait(0.5), FadeIn(eq2)))
        self.wait(0.1)
        self.play(mh.rtransform(gp1[:2], gp2[:2]),
                  Succession(Wait(0.5), FadeIn(eq3)))
        self.wait(0.1)
        self.play(FadeOut(gp2[1], eq3))
        self.wait(0.1)
        eq1 = gp2[0]
        self.play(mh.rtransform(eq1[-1].copy(), eq4[0], run_time=1.5),
                  FadeIn(eq4[1:-2], eq4[-1]),
                  Succession(Wait(1.1), FadeIn(eq4[-2]))
                  )
        eq9_1 = eq4[3].copy()

        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq4[4][:2], eq5[8][:2], eq4[4][-1], eq5[8][-1],
                                eq4[3][:5], eq5[7][:5], eq4[3][7:], eq5[7][7:]),
                  mh.rtransform(eq1[5].copy(), eq5[0]),
                  mh.rtransform(eq4[3][6], eq5[7][6], copy_colors=False),
                  mh.fade_replace(eq4[3][5], eq5[7][5]),
                  mh.fade_replace(eq4[4][2], eq5[8][2]),
                  run_time=1.7),
                  Succession(Wait(0.7), mh.rtransform(eq4[:3], eq5[1:4], run_time=1.2)),
                  Succession(Wait(1), FadeIn(eq5[4:7], eq5[-1])))
        self.play(mh.rtransform(eq5[7][:], eq6[0][:-4]),
                  mh.rtransform(eq5[6][0], eq6[0][0]),
                  mh.rtransform(eq5[6][1:], eq6[0][-4:], path_arc=-PI/3),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:4], eq7[1:5], eq1[4].copy(), eq7[0], run_time=1.5))
        rect1 = SurroundingRectangle(eq6[0][-2], buff=0.1, corner_radius=0.1, stroke_color=RED,
                                     stroke_width=6)
        rect2 = SurroundingRectangle(eq7[7][-6:-1], buff=0.1, corner_radius=0.1, stroke_color=RED,
                                     stroke_width=6)
        self.play(FadeIn(rect1, rate_func=linear, run_time=0.4))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[4:6], eq7[5:7], eq6[0][:-1], eq7[7][:-5], eq6[0][-1], eq7[7][-1]),
                  mh.rtransform(eq5[8:], eq7[8:]),
                  Succession(Wait(0.2), FadeIn(eq7[7][-5:-1])),
                  mh.rtransform(rect1, rect2))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[7][:-5], eq8[0][:-5], eq7[7][-5:-1], eq8[0][-4:], eq7[7][-1], eq8[0][-5],
                                path_arc=-PI/3),
                  FadeOut(rect2))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:5], eq9[1:6], eq1[3].copy(), eq9[0], run_time=1.5))
        mh.align_sub(eq9_1, eq9_1[0], eq7[7][0]).to_edge(LEFT, buff=0.1)
        eq9_2 = MathTex(r'\frac{e^{\frac12ix^2\cot\theta} }{\sqrt{2\pi} }', r'\int')
        mh.align_sub(eq9_2, eq9_2[1], eq7[6])
        mh.align_sub(eq9_2, eq9_2[0][-5], eq7[5][-5], coor_mask=UP)
        mh.copy_colors_eq(eq9_1[:], eq9_2[0][:-5])
        self.play(mh.rtransform(eq7[5][-5:], eq9_2[0][-5:]),
                  FadeOut(eq7[5][0]),
                  FadeIn(eq9_2[0][:-5]))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq7[-2:], eq9[-2:], eq8[0][-12:], eq9[8][-12:], eq7[6], eq9[7],
                                eq9_2[0][-5:], eq9[6][-5:], eq8[0][:5], eq9[8][:5], eq8[0][5:7], eq9[8][9:11]),
                  mh.rtransform(eq9_2[0][7:11], eq9[8][12:16], eq9_2[0][5:7], eq9[8][6:8],
                                eq9_2[0][1:5], eq9[8][1:5]),
                  mh.fade_replace(eq9_2[0][0], eq9[6][0].set_color(col_num), coor_mask=RIGHT),
                                 run_time=1.6),
                  Succession(Wait(1), FadeIn(eq9[8][5], eq9[8][8], eq9[8][11])))
        self.wait(0.1)
        self.play(FadeOut(eq9[:4]),
                  mh.rtransform(eq9[4:6], eq10[1:3]),
                  FadeIn(eq10[0]))
        self.play(mh.rtransform(eq9[-4:], eq10[-4:], eq9[6][-2:], eq10[3][-2:],
                                eq9[6][-4:-2], eq10[3][:2], eq9[6][-5], eq10[3][-3], eq9[6][0], eq10[3][2]),
                  mh.rtransform(eq1[2][2:].copy(), eq10[3][2:9]),
                  run_time=2)

        self.wait(0.1)
        circ1 = mh.circle_eq(eq10[5][9:11], scale=0.4).shift(DR*0.005)
        self.play(Create(circ1, rate_func=linear, run_time=0.4))
        line1 = Line(eq1[-1][0].get_bottom()+DOWN*0.1, circ1.get_top(), stroke_width=8, stroke_color=col_chirp).set_z_index(1)
        self.play(Create(line1, run_time=0.7, rate_func=linear))

        circ2 = mh.circle_eq(eq10[5][18:20], scale=0.4).shift(LEFT*0.05)
        self.play(Create(circ2, rate_func=linear, run_time=0.4))
        line2 = Line(eq1[-2][0].get_bottom()+DOWN*0.1, circ2.get_top(), stroke_width=8, stroke_color=col_ft).set_z_index(1.1)
        self.play(Create(line2, run_time=0.7, rate_func=linear))

        circ3 = mh.circle_eq(eq10[5][20:], scale=0.4).shift(RIGHT*0.1+DOWN*0.05)
        self.play(Create(circ3, rate_func=linear, run_time=0.4))
        line3 = Line(eq1[-3][0].get_bottom()+DOWN*0.1, circ3.get_top(), stroke_width=8, stroke_color=col_squeeze).set_z_index(1.2)
        self.play(Create(line3, run_time=0.7, rate_func=linear))

        circ4 = mh.circle_eq(eq10[5][6:8], scale=0.4).shift(DOWN*0.05)
        self.play(Create(circ4, rate_func=linear, run_time=0.4))
        line4 = Line(eq1[-4][0].get_bottom()+DOWN*0.1, circ4.get_top(), stroke_width=8, stroke_color=col_chirp).set_z_index(1.3)
        self.play(Create(line4, run_time=0.7, rate_func=linear))

        circ5 = mh.circle_eq(eq10[3][2:])
        self.play(Create(circ5, rate_func=linear, run_time=0.6))

        self.wait()
