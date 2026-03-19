from algan.external_libraries.manim.utils.color import ManimColor
from fontTools.unicodedata import block
from manim import *
import numpy as np
import math
import sys
import scipy as sp
from matplotlib.font_manager import font_scalings
from numpy.random.mtrand import Sequence
from sorcery import switch
from torch.utils.jit.log_extract import run_test


sys.path.append('../')
import manimhelper as mh
from common.wigner import *

col_psi = (RED-WHITE)*0.8 + WHITE
col_x = (BLUE-WHITE)*0.7 + WHITE
col_p = (GREEN-WHITE)*0.9 + WHITE
col_num = (TEAL_E - WHITE) * 0.5 + WHITE
col_special = (PURPLE - WHITE) * 0.25 + (TEAL_E - WHITE) * 0.25 + WHITE
col_i = (YELLOW - WHITE) * 0.3 + WHITE
col_WVD = (ORANGE-WHITE)*0.9 + WHITE
col_op = (PURPLE-WHITE) * 0.5 + WHITE
col_var = col_special
col_eq = (col_op+WHITE)*0.5

def eq_shadow(eq: VGroup, fg_z_index=4., bg_z_index=0., bg_color=BLACK, bg_stroke_width=10.):
    res = VGroup()
    for eq1 in eq:
        elem = VGroup()
        for eq2 in eq1:
            elem.add(VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res


class LinearComb(Scene):
    fill_op = 0.7
    bgcol = GREY

    def __init__(self, *args, **kwargs):
        if not config.transparent:
            config.background_color=self.bgcol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        eq1 = MathTex(r'aX+bP', font_size=100).set_z_index(1)
        VGroup(eq1[0][1]).set_color(col_x)
        eq1[0][4].set_color(col_p)
        VGroup(eq1[0][0], eq1[0][3]).set_color(col_var)
        box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_opacity=self.fill_op,
                                    fill_color=BLACK, corner_radius=0.15, buff=0.2)
        VGroup(eq1, box1).to_edge(DOWN, buff=1)
        self.add(eq1, box1)

class Latexx(Scene):
    def construct(self):
        eq = MathTex(r'x', font_size=80, stroke_width=2)
        self.add(eq)

class Latexv(Scene):
    def construct(self):
        eq = MathTex(r'v', font_size=80, stroke_width=2)
        self.add(eq)

class LatexPsix(Scene):
    def construct(self):
        eq = MathTex(r'\psi(x)', font_size=80, stroke_width=2)
        self.add(eq)

class LatexPsiFT(Scene):
    def construct(self):
        eq = MathTex(r'\widehat\psi(p)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(x)e^{-ipx}\,dx', font_size=60, stroke_width=2)
        mh.font_size_sub(eq, 2, 50)
        eq = eq_shadow(eq)
        self.add(eq)

class LatexWxp(Scene):
    def construct(self):
        eq = MathTex(r'W(x,p)', font_size=60, stroke_width=2)
        eq = eq_shadow(eq)
        self.add(eq)

class SlitEqn(Scene):
    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=2)
        eq1 = MathTex(r'x')
        eq2 = MathTex(r'vt')
        eq3 = MathTex(r'x + vt')
        eq4 = MathTex(r'x + \frac{p}{m}t', font_size=80, stroke_width=2)
        eq5 = MathTex(r'x + \frac{t}{m}p', font_size=80, stroke_width=2)
        VGroup(eq1, eq2).arrange(DOWN, buff=1.8, aligned_edge=RIGHT)

        eq3.move_to(eq1).align_to(eq1, RIGHT).shift(DOWN*0.5)
        mh.align_sub(eq5, eq5[0][0], eq3[0][0]).align_to(eq1, RIGHT)
        mh.align_sub(eq4, eq4[0][0], eq5[0][0])


        self.add(eq1, eq2)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][0], eq3[0][0], eq2[0][:], eq3[0][-2:]),
                  Succession(Wait(0.4), FadeIn(eq3[0][1])))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0][:2], eq4[0][:2], eq3[0][-1], eq4[0][-1]),
                  mh.fade_replace(eq3[0][-2], eq4[0][2:-1], coor_mask=RIGHT))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][:2], eq5[0][:2], eq4[0][2], eq5[0][-1],
                                eq4[0][-1], eq5[0][2], eq4[0][3:-1], eq5[0][3:-1]))
        self.wait()

class Interference(Scene):
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
            y = 0j
            for x1 in [-x0, x0]:
                d = (np.sqrt((x - x1) * (x - x1) / d0 + 1) - 1) * d0
                y += np.exp(-(x - x1) * (x - x1) / (2 * s * s) + d * 1j * a)
            y = np.abs(y)
            y *= y / 4 * 1.35
            return y

        path = ax.plot(f, (-xmax, xmax), stroke_color=BLUE, stroke_width=5).set_z_index(1)
        path.set_fill(opacity=0.7, color=BLUE)
        VGroup(ax, path).shift(-path.get_center())

        self.add(path)

class FourierTomography(LinearComb):
    fill_op = 0.7

    def get_eqs(self):
        MathTex.set_default(font_size=60, stroke_width=1.5, color=col_eq)
        eq1 = MathTex(r'\varphi(u,v)', r'=', r'\iint e^{i(ux+vy)}\rho(x,y)\,dxdy',
                      r'=', r'\mathbb E[e^{i(uX+vY)}]')
        eq2 = MathTex(r'\rho(x,y)', r'=', r'\frac1{(2\pi)^2}', r'\iint e^{-i(ux+vy)}\varphi(u,v)\,dudv')
        eq3 = MathTex(r'\rho(x,y)', r'=', r'\frac1{(2\pi)^2}', r'\iint e^{-i(ux+vy)}', r'\mathbb E[e^{i(uX+vY)}]',
                      r'\,dudv')

        mh.rtransform.copy_colors = True
        VGroup(eq1[2][6], eq1[2][9], eq1[2][13], eq1[2][15], eq1[2][18], eq1[2][20],
               ).set_color(col_x)
        VGroup(eq1[0][2], eq1[0][4], eq1[2][5], eq1[2][8], eq2[3][-3], eq2[3][-1]).set_color(col_p)
        VGroup(eq2[2][0], eq2[2][-4], eq2[2][-1]).set_color(col_num)
        VGroup(eq1[2][2], eq2[2][-3]).set_color(col_special)
        VGroup(eq1[0][0], eq1[2][-10], eq1[4][0]).set_color(col_WVD)
        VGroup(eq1[2][3]).set_color(col_i)
        VGroup(eq1[2][:2], eq1[2][-4], eq1[2][-2], eq3[2][1]).set_color(col_op)
        mh.copy_colors_eq(eq1[2][2:11], eq1[4][2:-1])
        mh.copy_colors_eq(eq1[4], eq3[4])

        eq2.next_to(eq1, DOWN)
        gp = VGroup(eq1, eq2, eq3).set_z_index(1)
        box = SurroundingRectangle(gp, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                   fill_opacity=self.fill_op, corner_radius=0.2, buff=0.15)
        VGroup(gp, box).to_edge(DOWN, buff=0.1)

        return box, eq1, eq2, eq3

    def construct(self):
        box, eq1, eq2, eq3 = self.get_eqs()

        eq1_2 = eq1.copy()
        eq1.move_to(box)
        eq1_1 = eq1[:3].copy().move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=UP)

        self.add(box, eq1_1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1_1, eq1[:3]),
                  Succession(Wait(0.4), FadeIn(eq1[3:])))
        self.wait(0.1)
        self.play(Transform(eq1, eq1_2))
        self.wait(0.1)
        eq1_ = eq1[:3].copy()
        shift = mh.diff(eq1_[2][0], eq2[3][0])
        self.play(mh.rtransform(eq1_[0][:], eq2[3][-10:-4], eq1_[1], eq2[1], eq1_[2][:3], eq2[3][:3],
                                eq1_[2][3:11], eq2[3][4:12], eq1_[2][11:17], eq2[0][:],
                                eq1_[2][-4], eq2[3][-4], eq1_[2][-2], eq2[3][-2]),
                  FadeIn(eq2[3][3], shift=mh.diff(eq1[2][3], eq2[3][4])),
                  mh.fade_replace(eq1_[2][-3], eq2[3][-3]),
                  mh.fade_replace(eq1_[2][-1], eq2[3][-1]),
                  eq2[2].set_opacity(-2).shift(-shift).animate.set_opacity(1).shift(shift),
                  run_time=2)
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:3], eq3[:3], eq2[3][:12], eq3[3][:], eq2[3][-4:], eq3[5][:]),
                  Succession(Wait(0.4), AnimationGroup(mh.fade_replace(eq2[3][12:-4], eq3[4], coor_mask=RIGHT))),
                  # mh.fade_replace(eq2[3][12:-4], eq3[4], coor_mask=RIGHT)
                  )
        self.wait(0.1)
        self.play(FadeOut(eq1), eq3.animate.move_to(box))
        self.wait()

class WaveFunction(FourierTomography):
    def get_eqs(self, do_anim):
        box, _, _, _ = FourierTomography.get_eqs(self)

        eq1 = MathTex(r'\psi', r'\colon\mathbb R\to\mathbb C', font_size=80)
        eq2 = MathTex(r'\lvert\psi(x)\rvert^2', r'=', r'{\sf probability\ density}')
        eq3 = MathTex(r'\int', r'\lvert\psi(x)\rvert^2', r'dx', r'=', r'1')
        eq4 = MathTex(r'\lvert\psi(x)\rvert^2', r'=', r'\frac1{\sqrt{2\pi\sigma^2} }', r'e^{-\frac{x^2}{2\sigma^2}')
        mh.font_size_sub(eq4, 2, 50)
        eq5 = MathTex(r'\psi(x)', r'=', r'\frac1{\sqrt[4]{2\pi\sigma^2} }', r'e^{-\frac{x^2}{4\sigma^2}')
        mh.font_size_sub(eq5, 2, 50)

        VGroup(eq1, eq2, eq3, eq4, eq5).set_z_index(1)

        eq2.next_to(eq1, DOWN, buff=0.5)
        VGroup(eq1, eq2).move_to(box)
        mh.align_sub(eq3, eq3[1], eq2[0])
        mh.align_sub(eq4, eq4[0], eq3[1])
        mh.align_sub(eq5, eq5[1], eq4[1])

        if not do_anim:
            return box, eq1, eq5

        self.add(box)
        self.wait(0.1)
        eq1_ = eq1.copy().move_to(box)
        self.play(FadeIn(eq1_))
        self.wait(0.1)
        self.play(mh.rtransform(eq1_, eq1),
                  Succession(Wait(0.3), FadeIn(eq2)))
        self.wait(0.1)
        self.play(FadeOut(eq2[2]),
                  Succession(Wait(0.4), mh.rtransform(eq2[0], eq3[1], eq2[1], eq3[3])),
                  Succession(Wait(1), FadeIn(eq3[0], eq3[2], eq3[4]))
                  )
        self.wait(0.1)
        self.play(LaggedStart(FadeOut(eq3[0], eq3[2], eq3[4]),
                  mh.rtransform(eq3[1], eq4[0], eq3[3], eq4[1]),
                  FadeIn(eq4[2:]), lag_ratio=0.6))
        self.wait(0.1)
        self.play(FadeOut(eq4[0][0], eq4[0][-2:]),
                  mh.rtransform(eq4[1], eq5[1], eq4[2][:2], eq5[2][:2], eq4[2][2:], eq5[2][3:],
                                eq4[3][:5], eq5[3][:5], eq4[3][6:], eq5[3][6:]),
                  mh.fade_replace(eq4[3][5], eq5[3][5], coor_mask=RIGHT),
                  FadeIn(eq5[2][2]),
                  Succession(Wait(0.8), mh.rtransform(eq4[0][1:-2], eq5[0][:])))

        return box, eq1, eq5

    def construct(self):
        box, eq1, eq5 = self.get_eqs(do_anim=True)

        # go full screen
        scale=1.1
        eq6 = MathTex(r'\psi(x)', r'=', r'\frac1{\sqrt[4]{2\pi\sigma^2} }', r'e^{-\frac{x^2}{4\sigma^2}+ipx')
        mh.font_size_sub(eq6, 2, 50).scale(scale)

        box2 = Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1)

        eq5_1 = eq5.copy().move_to(ORIGIN).scale(scale).shift(LEFT*2.1 + UP * 1.)
        mh.align_sub(eq6, eq6[1], eq5_1[1]).move_to(eq5_1, coor_mask=RIGHT)

        self.wait(0.1)
        self.play(LaggedStart(FadeOut(eq1, run_time=1),
                  AnimationGroup(mh.rtransform(box, box2),
                                 mh.transform(eq5, eq5_1),
                                 run_time=1.5), lag_ratio=0.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:3], eq6[:3], eq5[3][:], eq6[3][:-4]),
                  FadeIn(eq6[3][-4:], shift=mh.diff(eq5[3][-1], eq6[3][-5])))

        self.wait()

class WignerDensity(LinearComb):
    bgcol = BLACK
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        eq1 = MathTex(r'W(x,p)', r'=', r'\frac1{2\pi\hbar}', r'\int \psi(x-y/2)^*\psi(x+y/2)e^{-\frac{ipy}{\hbar} }\,dy')
        eq2 = MathTex(r'W(x,p)', r'=', r'\frac1{2\pi}', r'\int \psi(x-y/2)^*\psi(x+y/2)e^{-ipy }\,dy')

        eq3 = MathTex(r'\psi(x)', r'\sim', r'e^{-\frac12ax^2}', font_size=80)
        eq4 = MathTex(r'W(x,p)', r'\sim', r'\int e^{-\frac12a(x-y/2)^2}e^{-\frac12a(x+y/2)^2}e^{-ipy }\,dy')
        eq5 = MathTex(r'W(x,p)', r'\sim', r'\int e^{-\frac12a(x-y/2)^2-\frac12a(x+y/2)^2-ipy }\,dy')
        eq6 = MathTex(r'W(x,p)', r'\sim', r'\int e^{-a(x^2+(y/2)^2)-ipy }\,dy')
        eq7 = MathTex(r'W(x,p)', r'\sim', r'e^{-ax^2}', r'\int e^{-a(y/2)^2-ipy }\,dy')
        eq8 = MathTex(r'W(x,p)', r'\sim', r'e^{-ax^2}', r'\int e^{-ay^2-2ipy }\,dy')
        eq9 = MathTex(r'W(x,p)', r'\sim', r'e^{-ax^2}', r'\int e^{-a(y^2+2ipy/a) }\,dy')


        mh.align_sub(eq1, eq1[1], eq2[1], coor_mask=UP)
        eq3.next_to(eq2, UP, buff=1)
        gp1 = VGroup(eq2.copy(), eq3).move_to(ORIGIN, coor_mask=UP)
        mh.align_sub(eq4, eq4[1], gp1[0][1], coor_mask=UP)
        mh.align_sub(eq5, eq5[1], eq4[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[1], eq5[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[1], eq6[1], coor_mask=UP)
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=UP)
        mh.align_sub(eq9, eq9[1], eq8[1], coor_mask=UP)
        mh.align_sub(eq2, eq2[1], )

        self.add(eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq1[2][:-1], eq2[2][:],
                                eq1[3][:-7], eq2[3][:-5], eq1[3][-6:-4], eq2[3][-4:-2],
                                eq1[3][-2:], eq2[3][-2:]),
                  mh.stretch_replace(eq1[3][-7], eq2[3][-5]),
                  FadeOut(eq1[2][-1], eq1[3][-4:-2]))
        self.wait(0.1)
        self.play(mh.transform(eq2, gp1[0]),
                  Succession(Wait(0.4), FadeIn(eq3)))
        self.wait(0.1)
        eq3_1 = eq3[2].copy()
        eq3_2 = eq3[2].copy()
        eq4_1 = eq4[2][1:15]
        eq4_2 = eq4[2][15:-7]
        self.play(Succession(Wait(1.), FadeOut(eq2[3][1:-7], run_time=1)),
                  mh.rtransform(eq3_1[:6], eq4_1[:6], eq3_1[6], eq4_1[7], eq3_1[7], eq4_1[-1],
                                eq3_2[:6], eq4_2[:6], eq3_2[6], eq4_2[7], eq3_2[7], eq4_2[-1],
                                eq2[0], eq4[0], eq2[3][0], eq4[2][0], eq2[3][-7:], eq4[2][-7:],
                                run_time=2),
                  FadeOut(eq2[2]),
                  mh.fade_replace(eq2[1], eq4[1], run_time=2),
                  Succession(Wait(1.5), FadeIn(eq4_1[6], eq4_1[8:-1],
                                               eq4_2[6], eq4_2[8:-1],
                                               run_time=1, rate_func=linear)),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq4[:2], eq5[:2], eq4[2][:15], eq5[2][:15],
                                eq4[2][16:29], eq5[2][15:28], eq4[2][30:], eq5[2][28:]),
                  FadeOut(eq4[2][15], eq4[2][29], eq4[2][29]))
        self.wait(0.1)
        eq6_1 = eq6[2][7].copy()
        self.play(mh.rtransform(eq5[:2], eq6[:2], eq5[2][:3], eq6[2][:3],
                                eq5[2][6:9], eq6[2][3:6], eq5[2][14], eq6[2][6],
                                eq5[2][10:13], eq6[2][9:12], eq5[2][13], eq6[2][14]),# eq5[2][14].copy(), eq6[2][13]),
                  FadeOut(eq5[2][3:6]),
                  mh.fade_replace(eq5[2][9], eq6_1),
                  FadeIn(eq6[2][8], eq6[2][12], shift=mh.diff(eq5[2][10:13], eq6[2][9:12])),
                  mh.rtransform(eq5[2][15], eq6[2][2], eq5[2][19:22], eq6[2][3:6], eq5[2][22], eq6[2][7],
                                eq5[2][23:26], eq6[2][9:12], eq5[2][26], eq6[2][14], eq5[2][27], eq6[2][13],
                                eq5[2][28:], eq6[2][15:]),
                  FadeOut(eq5[2][16:19]),
                  run_time=2
                  )
        self.remove(eq6_1)
        self.wait(0.1)
        eq7_1 = eq7[3][2].copy()
        self.play(mh.rtransform(eq6[:2], eq7[:2], eq6[2][1:4], eq7[2][:3], eq6[2][5:7], eq7[2][3:],
                                eq6[2][0], eq7[3][0], eq6[2][1:4].copy(), eq7[3][1:4],
                                eq6[2][8:14], eq7[3][4:10],
                                eq6[2][15:], eq7[3][10:]),
                  FadeOut(eq6[2][4]),
                  mh.fade_replace(eq6[2][7], eq7_1),
                  FadeOut(eq6[2][14]),
                  run_time=1.5
                  )
        self.remove(eq7_1)
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:3], eq8[:3], eq7[3][:4], eq8[3][:4],
                                eq7[3][5], eq8[3][4], eq7[3][9], eq8[3][5], eq7[3][10], eq8[3][6],
                                eq7[3][11:], eq8[3][8:]),
                  FadeOut(eq7[3][4], eq7[3][6:9]),
                  FadeIn(eq8[3][7], shift=mh.diff(eq7[3][11], eq8[3][8])),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq8[:3], eq9[:3], eq8[3][:4], eq9[3][:4],
                                eq8[3][4:6], eq9[3][5:7], eq8[3][7:11], eq9[3][8:12],
                                eq8[3][11:], eq9[3][15:]),
                  mh.fade_replace(eq8[3][6], eq9[3][7]),
                  FadeIn(eq9[3][4], shift=mh.diff(eq8[3][4], eq9[3][5])),
                  FadeIn(eq9[3][12:15], shift=mh.diff(eq8[3][4], eq9[3][5])),
                  )
        self.wait(0.1)
        self.wait()

class WignerTranslate(WignerDensity):
    def construct(self):
        MathTex.set_default(font_size=55, stroke_width=1.5)
        fs1 = 65
        eq1 = MathTex(r'\tilde\psi(x)', r'=', r'\psi(x+y)', font_size=fs1)
        eq2 = MathTex(r'\widetilde W(x,p)', r'=', r'\frac1{2\pi}', r'\int\tilde\psi', r'\left(x-\frac v2\right)^*', r'\tilde\psi',
                      r'\left(x+\frac v2\right)', r'e^{-ipv}\,dx')
        mh.font_size_sub(eq2, 2, 45)
        mh.font_size_sub(eq2, 4, 45)
        mh.font_size_sub(eq2, 6, 45)
        eq3 = MathTex(r'\widetilde W(x,p)', r'=', r'\frac1{2\pi}', r'\int\psi', r'\left((x+y)-\frac v2\right)^*',
                      r'\psi', r'\left((x+y)+\frac v2\right)', r'e^{-ipv}\,dv')
        eq4 = MathTex(r'\widetilde W(x,p)', r'=', r'W(x+y,p)')
        mh.font_size_sub(eq3, 2, 45)
        mh.font_size_sub(eq3, 4, 45)
        mh.font_size_sub(eq3, 6, 45)
        eq5 = MathTex(r'\tilde\psi(x)', r'=', r'\sqrt{a}', r'\psi(ax)', font_size=fs1)
        eq6 = MathTex(r'\widetilde W(x,p)', r'=', r'\frac1{2\pi}', r'\int\psi', r'\left(ax-\frac {av}2\right)^*',
                      r'\psi', r'\left(ax+\frac {av}2\right)', r'e^{-ipv}a\,dv')
        mh.font_size_sub(eq6, 2, 45)
        mh.font_size_sub(eq6, 4, 45)
        mh.font_size_sub(eq6, 6, 45)
        eq7 = MathTex(r'{\sf subsitute}', r'w=av', font_size=55)
        eq8 = MathTex(r'\frac w2', r'\frac w2', r'e^{-ipw/a}\,dw')
        mh.font_size_sub(eq8, 0, 45)
        mh.font_size_sub(eq8, 1, 45)
        eq9 = MathTex(r'\widetilde W(x,p)', r'=', r'W(ax,p/a)')

        eq2.to_edge(DOWN, buff=0.4)
        eq1.next_to(eq2, UP, buff=0.5)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=UP).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq4, eq4[1], eq2[1], coor_mask=UP).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[1], eq1[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[1], eq2[1], coor_mask=UP).move_to(ORIGIN, coor_mask=RIGHT)
        eq7[1].next_to(eq7[0], DOWN, buff=0.4)
        eq7.move_to(eq5).to_edge(RIGHT, buff=0.8)
        mh.align_sub(eq8[2], eq8[2][0], eq6[-1][0])
        mh.align_sub(eq8[0], eq8[0][1], eq6[4][6])
        mh.align_sub(eq8[1], eq8[1][1], eq6[6][6])
        mh.align_sub(eq9, eq9[1], eq2[1], coor_mask=UP).move_to(ORIGIN, coor_mask=RIGHT)

        sw = 10
        eq1 = eq_shadow(eq1, bg_stroke_width=sw)
        eq2 = eq_shadow(eq2, bg_stroke_width=sw)
        eq3 = eq_shadow(eq3, bg_stroke_width=sw)
        eq4 = eq_shadow(eq4, bg_stroke_width=sw)
        eq5 = eq_shadow(eq5, bg_stroke_width=sw)
        eq6 = eq_shadow(eq6, bg_stroke_width=sw)
        eq7 = eq_shadow(eq7, bg_stroke_width=sw)
        eq8 = eq_shadow(eq8, bg_stroke_width=sw)
        eq9 = eq_shadow(eq9, bg_stroke_width=sw)

        self.add(eq1, eq2)
        eq1_1 = eq1[2].copy()
        eq1_2 = eq1[2].copy()
        self.play(mh.rtransform(eq2[:3], eq3[:3], eq2[3][0], eq3[3][0], eq2[3][2], eq3[3][1],
                                eq2[4][0], eq3[4][0], eq2[4][1], eq3[4][2], eq2[4][2:], eq3[4][6:],
                                eq2[5][1], eq3[5][0], eq2[6][0], eq3[6][0], eq2[6][1], eq3[6][2], eq2[6][2:], eq3[6][6:],
                                eq2[7][:], eq3[7][:]),
                  FadeOut(eq2[3][1], shift=mh.diff(eq2[3][2], eq3[3][1])),
                  FadeOut(eq2[5][0], shift=mh.diff(eq2[5][1], eq3[5][0])),
                  mh.rtransform(eq1_1[0], eq3[3][1], eq1_1[1:], eq3[4][1:6],
                                eq1_2[0], eq3[5][0], eq1_2[1:], eq3[6][1:6]
                                ),
                  run_time=1.8)
        self.play(Succession(Wait(1), mh.rtransform(eq3[:2], eq4[:2], run_time=1.4)),
                  AnimationGroup(
                  mh.rtransform(eq3[4][2:5], eq4[2][2:5]),
                  mh.rtransform(eq3[6][2:5], eq4[2][2:5]),
                  FadeIn(eq4[2][:2], eq4[2][5:]),
                  run_time=1.6),
                  FadeOut(eq3[2], eq3[3], eq3[4][:2], eq3[4][5:], eq3[5],
                          eq3[6][:2], eq3[6][5:], eq3[7]),
                  )
        self.wait(0.1)
        self.play(Succession(Wait(0.4),
                             mh.rtransform(eq1[:2], eq5[:2], eq1[2][:2], eq5[3][:2],
                                           eq1[2][2],eq5[3][3], eq1[2][-1], eq5[3][-1])),
                  FadeOut(eq1[2][3:-1]),
                  FadeOut(eq4),
                  Succession(Wait(1), FadeIn(eq5[2], eq5[3][2])),
                  )
        self.wait(0.1)
        eq5_1 = eq5[3].copy()
        eq5_2 = eq5[3].copy()
        s1 = mh.diff(eq5_1[2], eq6[4][4])
        s2 = mh.diff(eq5_2[2], eq6[6][4])
        self.play(AnimationGroup(mh.rtransform(eq5[2][-1].copy(), eq6[-1][-3], eq5_1[0], eq6[3][1], eq5_1[2:4], eq6[4][1:3],
                                eq5_1[2].copy(), eq6[4][4], eq5_1[4], eq6[4][8],
                                eq5_2[0], eq6[5][0], eq5_2[2:4], eq6[6][1:3], eq5_2[2].copy(), eq6[6][4], eq5_2[4], eq6[6][8]
                                ),
                  mh.stretch_replace(eq5_1[1], eq6[4][0]),
                  mh.stretch_replace(eq5_2[1], eq6[6][0]),
                  FadeIn(eq6[:3], eq6[3][0], eq6[7][-2:]),
                  VGroup(eq6[4][5:-2],eq6[4][3],eq6[4][-1]).set_opacity(-1.5).shift(-s1).animate.set_opacity(1).shift(s1),
                  VGroup(eq6[6][5:-1],eq6[6][3]).set_opacity(-1.5).shift(-s2).animate.set_opacity(1).shift(s2),
                                 run_time=2.4),
                  Succession(Wait(1.4), FadeIn(eq6[7][:-3])),
        )
        self.wait(0.05)
        self.play(FadeIn(eq7))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[1][0].copy(), eq8[0][0], eq6[4][6:8], eq8[0][1:],
                                eq7[1][0].copy(), eq8[1][0], eq6[6][6:8], eq8[1][1:]),
                  FadeOut(eq6[4][4:6], eq6[6][4:6]),
                  run_time=1.6)
        self.play(mh.rtransform(eq6[7][-2], eq8[2][-2], eq7[1][0].copy(), eq8[2][-1]),
                  FadeOut(eq6[7][-3], target_position=eq8[2][-1]),
                  FadeOut(eq6[7][-1], target_position=eq8[2][-1].get_center()+RIGHT*0.),
                  run_time=1.4)
        self.play(mh.rtransform(eq6[7][:4], eq8[2][:4], eq7[1][0].copy(), eq8[2][4], eq7[1][2].copy(), eq8[2][6]),
                  FadeIn(eq8[2][5]),
                  FadeOut(eq6[7][4], eq7),
                  run_time=1.2)
        self.wait(0.1)
        self.play(Succession(Wait(1), mh.rtransform(eq6[:2], eq9[:2], run_time=1.4)),
                  AnimationGroup(
                  mh.rtransform(eq6[4][1:3], eq9[2][2:4]),
                  mh.rtransform(eq6[6][1:3], eq9[2][2:4]),
                  mh.rtransform(eq8[2][3], eq9[2][5], eq8[2][5:7], eq9[2][6:8]),
                  run_time=1.6),
                  Succession(Wait(1.4), FadeIn(eq9[2][:2], eq9[2][4], eq9[2][-1], run_time=1.2)),
                  FadeOut(eq6[2], eq6[3], eq6[4][0], eq6[4][3], eq6[4][-2:], eq6[5],
                          eq8[:2], eq6[6][0], eq6[6][3], eq6[6][-1], eq8[2][:3], eq8[2][4], eq8[2][-2:]
                  #         #eq3[6][:2], eq3[6][5:], eq3[7]
                          ),
                  )
        self.wait()




class WignerDensity2(LinearComb):
    bgcol = BLACK
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        eq1 = MathTex(r'W(x,p)', r'=', r'\frac1{2\pi\hbar}', r'\int \psi(x-y/2)^*\psi(x+y/2)e^{-\frac{ipy}{\hbar} }\,dy')
        eq2 = MathTex(r'W(x,p)', r'=', r'\frac1{2\pi}', r'\int \psi(x-y/2)^*\psi(x+y/2)e^{-ipy }\,dy')
        eq4 = MathTex(r'W(x,p)', r'\sim', r'\int e^{-\frac12a(x-y/2)^2}e^{-\frac12a(x+y/2)^2}e^{-ipy }\,dy')

        eq13 = MathTex(r'{\sf if\ }', r'\psi(x)', r'\sim', r'e^{-\frac12ax^2}', font_size=80)
        eq14 = MathTex(r'{\sf then\ }', r'W(x,p)', r'\sim', r'e^{-ax^2-p^2/a}')

        mh.align_sub(eq1, eq1[1], eq2[1], coor_mask=UP)
        eq13.next_to(eq2, DOWN, buff=1)
        eq14.next_to(eq13, DOWN, buff=1)
        gp1 = VGroup(eq2.copy(), eq13, eq14).move_to(ORIGIN, coor_mask=UP)


        self.add(eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq1[2][:-1], eq2[2][:],
                                eq1[3][:-7], eq2[3][:-5], eq1[3][-6:-4], eq2[3][-4:-2],
                                eq1[3][-2:], eq2[3][-2:]),
                  mh.stretch_replace(eq1[3][-7], eq2[3][-5]),
                  FadeOut(eq1[2][-1], eq1[3][-4:-2]))
        self.wait(0.1)
        self.play(mh.transform(eq2, gp1[0]),
                  Succession(Wait(0.6), FadeIn(eq13)), run_time=1.8)
        self.wait(0.1)
        self.play(FadeIn(eq14))
        self.wait(0.1)

        eq3_1 = eq13[3].copy()
        eq3_2 = eq13[3].copy()
        eq4_1 = eq4[2][1:15]
        eq4_2 = eq4[2][15:-7]
        self.play(Succession(Wait(1.), FadeOut(eq2[3][1:-7], run_time=1)),
                  mh.rtransform(eq3_1[:6], eq4_1[:6], eq3_1[6], eq4_1[7], eq3_1[7], eq4_1[-1],
                                eq3_2[:6], eq4_2[:6], eq3_2[6], eq4_2[7], eq3_2[7], eq4_2[-1],
                                eq2[0], eq4[0], eq2[3][0], eq4[2][0], eq2[3][-7:], eq4[2][-7:],
                                run_time=2),
                  FadeOut(eq2[2]),
                  mh.fade_replace(eq2[1], eq4[1], run_time=2),
                  Succession(Wait(1.5), FadeIn(eq4_1[6], eq4_1[8:-1],
                                               eq4_2[6], eq4_2[8:-1],
                                               run_time=1, rate_func=linear)),
                  )

        self.wait()

class WavePosition(FourierTomography):
    def construct(self):
        box, _, _, _ = self.get_eqs()

        eq6 = MathTex(r'X', r'\psi(x)', r'=', r'x\psi(x)', font_size=80)
        eq7 = MathTex(r'\mathbb E[f(X)]', r'=', r'\int f(x)\lvert\psi(x)\rvert^2\,dx', font_size=80)
        eq8 = MathTex(r'\mathbb E[f(X)]', r'=', r'\int\psi(x)^* f(X)\psi(x)\,dx', font_size=80)
        eq9 = MathTex(r'=', r'\langle \psi\vert f(X)\psi\rangle', font_size=80)
        VGroup( eq6, eq7, eq8, eq9).set_z_index(1)

        eq6.move_to(box)
        eq7.move_to(box)
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=UP)
        mh.align_sub(eq9, eq9[0], eq8[1]).next_to(eq8, DOWN, buff=0.1)
        # mh.rtransform.copy_colors = False

        self.add(box)
        self.wait(0.1)
        eq6_1 = eq6[0].copy().scale(1.2).move_to(box)
        self.play(LaggedStart(FadeIn(eq6_1), lag_ratio=0.8))
        self.wait(0.1)
        self.play(mh.rtransform(eq6_1, eq6[0]),
                  Succession(Wait(0.5), FadeIn(eq6[1:])))
        self.wait(0.1)
        self.play(FadeOut(eq6), FadeIn(eq7))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:2], eq8[:2], eq7[2][0], eq8[2][0],
                                eq7[2][6:10].copy(), eq8[2][1:5], eq7[2][1:3], eq8[2][6:8], eq7[2][4], eq8[2][9],
                                eq7[2][6:10], eq8[2][10:14], eq7[2][-2:], eq8[2][-2:]),
                  FadeIn(eq8[2][5], shift=mh.diff(eq7[2][9], eq8[2][4])),
                  mh.fade_replace(eq7[2][3], eq8[2][8], coor_mask=RIGHT),
                  FadeOut(eq7[2][5], eq7[2][10:12], shift=mh.diff(eq7[2][6:10], eq8[2][10:14])))
        self.wait(0.1)
        gp = VGroup(eq8.copy(), eq9).move_to(box)
        eq9_1 = eq9[1].copy()
        eq9_1[1:].align_to(eq8[2][1], LEFT)
        eq9_1[3:].align_to(eq8[2][6], LEFT)
        eq9_1[2].move_to((eq9_1[1].get_right()+eq9_1[3].get_left())*0.5, coor_mask=RIGHT)
        self.play(LaggedStart(mh.transform(eq8, gp[0]),
                              FadeIn(eq9[0], eq9_1[0], eq9_1[2], eq9_1[-1]),
                              lag_ratio=0.7))
        self.play(mh.rtransform(eq8[2][1].copy(), eq9_1[1], eq8[2][6:11].copy(), eq9_1[3:-1]))
        self.play(mh.rtransform(eq9_1, eq9[1]))

        self.wait()

class WaveMomentum(FourierTomography):
    def construct(self):
        box, _, _, _ = self.get_eqs()
        MathTex.set_default(font_size=80)
        eq1 = MathTex(r'\psi(x)', r'=', r'e^{ipx}', font_size=80)
        eq2 = MathTex(r'e^{ipx/\hbar }', font_size=80)
        eq3 = MathTex(r'P', r'=', r'-i', r'\frac{\partial}{\partial x}', font_size=80)
        mh.font_size_sub(eq3, 3, 70).move_to(ORIGIN, RIGHT)
        eq4 = MathTex(r'P\psi(x)', r'=', r'-i', r'\frac{\partial}{\partial x}', r'e^{ipx}', font_size=80)
        mh.font_size_sub(eq4, 3, 70)
        eq5 = MathTex(r'P\psi(x)', r'=', r'-i', r'ip', r'e^{ipx}', font_size=80)
        eq6 = MathTex(r'\psi(x)', r'=', r'c_1e^{ip_1x}', r'+', r'c_2e^{ip_2x}', r'+\cdots')
        eq7 = MathTex(r'P\psi(x)', r'=', r'p_1c_1e^{ip_1x}', r'+', r'p_2c_2e^{ip_2x}', r'+\cdots')
        eq8 = MathTex(r'\mathbb E[f(P)]', r'=', r'\int \psi(x)^* f(P)\psi(x)\,dx')
        eq9 = MathTex(r'=', r'\langle \psi\vert f(P)\psi\rangle', font_size=80)
        eq10 = MathTex(r'P^r\psi(x)', r'=', r'\Big(-i', r'\frac{\partial}{\partial x}', r'\Big)^r', r'\psi(x)')
        eq11 = MathTex(r'P^r\psi(x)', r'=', r'(-i)^r', r'\frac{\partial^r}{\partial x^r}', r'\psi(x)')
        eq12 = MathTex(r'\left(c_1P^{r_1}+c_2P^{r_2}+\cdots)\psi', r'=', r'c_1P^{r_1}\psi+c_2P^{r_2}\psi+\cdots')
        mh.font_size_sub(eq10, 3, 70)
        mh.font_size_sub(eq11, 3, 70)
        VGroup(eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12).set_z_index(1)

        eq1.move_to(box)
        mh.align_sub(eq2, eq2[0][0], eq1[2][0])
        eq3.next_to(eq1, DOWN, buff=0.2)
        gp1 = VGroup(eq1.copy(), eq3).move_to(box)
        mh.align_sub(eq4, eq4[1], eq3[1], coor_mask=UP)
        mh.align_sub(eq5, eq5[1], eq4[1])
        eq5[4].align_to(eq4[4], LEFT)
        eq5[3].move_to(eq4[3], coor_mask=RIGHT)
        mh.align_sub(eq6, eq6[1], gp1[0][1], coor_mask=UP)
        mh.align_sub(eq7, eq7[1], eq5[1], coor_mask=UP)
        eq8.move_to(box)
        mh.align_sub(eq9, eq9[0], eq8[1]).next_to(eq8, DOWN, buff=0.1)
        eq10.move_to(box)
        mh.align_sub(eq11, eq11[1], eq10[1])
        # mh.align_sub(eq12, eq12[1], eq10[1], coor_mask=UP)
        eq12[1:].next_to(eq12[0], DOWN).align_to(eq12[0][7:], LEFT)
        eq12.move_to(box)

        self.add(box, eq1)
        self.wait(0.1)
        self.play(FadeIn(eq2[0][4:]))
        self.wait(0.1)
        self.play(FadeOut(eq2[0][4:]))
        self.wait(0.1)
        eq3_1 = eq3[0].copy().move_to(box, coor_mask=RIGHT)
        self.play(mh.transform(eq1, gp1[0]), FadeIn(eq3_1))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq3_1, eq3[0]),
                  FadeIn(eq3[1:]), lag_ratio=0.5))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq3[0][0], eq4[0][0], eq3[1], eq4[1], eq3[2:4], eq4[2:4], run_time=1.5),
                  FadeIn(eq4[0][1:], eq4[4]), lag_ratio=0.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[:3], eq5[:3], eq4[4], eq5[4]),
                  mh.stretch_replace(eq4[4][1].copy(), eq5[3][0]),
                  mh.stretch_replace(eq4[4][2].copy(), eq5[3][1]),
                  FadeOut(eq4[4][3].copy(), shift=mh.diff(eq4[4][2], eq5[3][1])),
                  FadeOut(eq4[3]))
        self.wait(0.1)
        self.play(FadeOut(eq5[2], eq5[3][0]))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq1[:2], eq6[:2], eq1[2][:3], eq6[2][2:5], eq1[2][3:], eq6[2][6:],
                                eq1[2][:3].copy(), eq6[4][2:5], eq1[2][3:].copy(), eq6[4][6:]),
                  FadeIn(eq6[2][:2], shift=mh.diff(eq1[2][0], eq6[2][2])),
                  FadeIn(eq6[2][5], shift=mh.diff(eq1[2][2], eq6[2][4])),
                  FadeIn(eq6[4][:2], shift=mh.diff(eq1[2][0], eq6[4][2])),
                  FadeIn(eq6[4][5], shift=mh.diff(eq1[2][2], eq6[4][4])), run_time=1.6),
                  Succession(Wait(0.8), FadeIn(eq6[3], eq6[5])),
                  )
        self.play(AnimationGroup(mh.rtransform(eq5[:2], eq7[:2], eq5[3][1], eq7[2][0], eq5[4][:3], eq7[2][4:7],
                                eq5[4][3], eq7[2][8], eq5[3][1].copy(), eq7[4][0], eq5[4][:3].copy(), eq7[4][4:7],
                                               eq5[4][3].copy(), eq7[4][8]),
                  FadeIn(eq7[2][1], shift=mh.diff(eq5[3][1], eq7[2][0])),
                  FadeIn(eq7[2][2:4], shift=mh.diff(eq5[4][0], eq7[2][4])+LEFT*0.3),
                  FadeIn(eq7[2][7], shift=mh.diff(eq5[4][2], eq7[2][6])),
                  FadeIn(eq7[4][1], shift=mh.diff(eq5[3][1], eq7[4][0])),
                  FadeIn(eq7[4][2:4], shift=mh.diff(eq5[4][0], eq7[4][4])+LEFT*0.4),
                  FadeIn(eq7[4][7], shift=mh.diff(eq5[4][2], eq7[4][6])),
                  run_time=1.6),
                  Succession(Wait(0.8), FadeIn(eq7[3], eq7[5])),
                  )
        self.wait(0.1)
        self.play(FadeOut(eq7, eq6), FadeIn(eq8))
        self.wait(0.1)
        gp = VGroup(eq8.copy(), eq9).move_to(box)
        eq9_1 = eq9[1].copy()
        eq9_1[1:].align_to(eq8[2][1], LEFT)
        eq9_1[3:].align_to(eq8[2][6], LEFT)
        eq9_1[2].move_to((eq9_1[1].get_right()+eq9_1[3].get_left())*0.5, coor_mask=RIGHT)
        self.play(mh.transform(eq8, gp[0]))
        self.play(FadeIn(eq9[0], eq9_1[0], eq9_1[2], eq9_1[-1]),
                  mh.rtransform(eq8[2][1].copy(), eq9_1[1], eq8[2][6:11].copy(), eq9_1[3:-1]),
                  run_time=1.4)
        self.play(mh.rtransform(eq9_1, eq9[1]))
        self.wait(0.1)
        self.play(FadeOut(eq8, eq9), FadeIn(eq10))
        self.wait(0.1)
        self.play(mh.rtransform(eq10[:2], eq11[:2], eq10[2][1:3], eq11[2][1:3], eq10[4][-2:], eq11[2][-2:],
                                eq10[3][0], eq11[3][0], eq10[4][-1].copy(), eq11[3][1], eq10[3][1:4], eq11[3][2:5],
                                eq10[4][-1].copy(), eq11[3][5], eq10[5], eq11[4]),
                  mh.stretch_replace(eq10[2][0], eq11[2][0]),
                  run_time=1.6
                  )
        self.wait(0.1)
        self.play(Succession(Wait(0.5), AnimationGroup(
            mh.rtransform(eq11[0][:2], eq12[0][3:5], eq11[0][:2].copy(), eq12[0][9:11],
                          eq11[0][2], eq12[0][-1], eq11[1], eq12[1]),
            FadeOut(eq11[0][-3:], shift=mh.diff(eq11[0][2], eq12[0][-1])),
            FadeIn(eq12[0][5], shift=mh.diff(eq11[0][1], eq12[0][4])),
            FadeIn(eq12[0][1:3], shift=mh.diff(eq11[0][0], eq12[0][3])),
            FadeIn(eq12[0][11], shift=mh.diff(eq11[0][1], eq12[0][11])),
            FadeIn(eq12[0][7:9], shift=mh.diff(eq11[0][0], eq12[0][9])),
            run_time=1.6)),
                  FadeOut(eq11[2:]),
                  Succession(Wait(1.2), FadeIn(eq12[0][0], eq12[0][6], eq12[0][12:17], run_time=1)),
                  )
        eq12_ = eq12[0].copy()
        # mh.align_sub(eq12_, eq12_[1], eq12[2][0]).align_to(eq12[2], LEFT)
        # eq12_[6:].align_to(eq12[2][6], LEFT)
        # self.play(mh.rtransform(eq12[0].copy(), eq12_))
        self.play(FadeOut(eq12_[0], eq12_[-2]),
                  mh.rtransform(eq12_[1:6], eq12[2][:5], eq12_[-1], eq12[2][5],
                                eq12_[6:12], eq12[2][6:12], eq12_[-1].copy(), eq12[2][12],
                                eq12_[12:16], eq12[2][13:17]),
                  run_time=1.6)


        self.wait()

class OperatorExp(WaveMomentum):
    def construct(self):
        box, _, _, _ = self.get_eqs()
        MathTex.set_default(font_size=80)
        eq1 = MathTex(r'e^{iPy}', r'\psi(x)', r'=', r'\psi(x+y)')
        eq2 = MathTex(r'\frac{\partial}{\partial y}', r'e^{iPy}', r'=', r'iP', r'e^{iPy}')
        mh.font_size_sub(eq2, 0, 70)
        eq3 = MathTex(r'\frac{\partial}{\partial y}', r'e^{iPy}', r'\psi(x)', r'=', r'iP', r'e^{iPy}', r'\psi(x)')
        mh.font_size_sub(eq3, 0, 70)
        eq4 = MathTex(r'\psi_y(x)', r'=', r'e^{iPy}', r'\psi(x)')
        eq5 = MathTex(r'\frac{\partial}{\partial y}', r'\psi_y(x)', r'=', r'iP', r'\psi_y(x)')
        mh.font_size_sub(eq5, 0, 70)
        eq6 = MathTex(r'\frac{\partial}{\partial y}', r'\psi_y(x)', r'=', r'\frac{\partial}{\partial x}', r'\psi_y(x)')
        mh.font_size_sub(eq6, 0, 70)
        mh.font_size_sub(eq6, 3, 70)
        eq7 = MathTex(r'\psi_y(x)', r'=', r'\psi(x+y)')

        VGroup(eq1, eq2, eq3, eq4, eq5, eq6, eq7).set_z_index(1)

        eq1.move_to(box)
        mh.align_sub(eq2, eq2[2], eq1[2]).move_to(box, coor_mask=RIGHT)
        mh.align_sub(eq3, eq3[3], eq2[2])
        eq4.next_to(eq3, DOWN, buff=0.2)
        gp1 = VGroup(eq3.copy(), eq4).move_to(box, coor_mask=UP)
        mh.align_sub(eq5, eq5[2], gp1[0][3])
        mh.align_sub(eq6, eq6[2], eq5[2])
        mh.align_sub(eq7, eq7[1], eq4[1])

        eq1_ = eq1[0].copy().scale(1.4).move_to(box, coor_mask=RIGHT)
        self.add(box, eq1_)
        self.wait(0.1)
        self.play(mh.rtransform(eq1_, eq1[0]), FadeIn(eq1[1], shift=mh.diff(eq1_[-1], eq1[0][-1])*RIGHT),
                  Succession(Wait(0.5), FadeIn(eq1[2:])))
        self.wait(0.1)
        self.play(FadeOut(eq1[1:]),
                  mh.rtransform(eq1[0], eq2[1]),
                  Succession(Wait(0.5), FadeIn(eq2[0], eq2[2:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:2], eq3[:2], eq2[2:5], eq3[3:6]),
                  Succession(Wait(0.3), FadeIn(eq3[2], eq3[-1])))
        self.wait(0.1)
        self.play(Transform(eq3, gp1[0]),
                  Succession(Wait(0.4), FadeIn(eq4)))
        self.play(mh.rtransform(eq3[0], eq5[0], eq3[2][0], eq5[1][0], eq3[2][1:], eq5[1][2:],
                                eq3[3], eq5[2], eq3[4], eq5[3], eq3[6][0], eq5[4][0], eq3[6][1:], eq5[4][2:]),
                  FadeOut(eq3[1], eq3[5]),
                  FadeIn(eq5[1][1], shift=mh.diff(eq3[2][0], eq5[1][0])),
                  FadeIn(eq5[4][1], shift=mh.diff(eq3[6][0], eq5[4][0])),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:3], eq6[:3], eq5[4], eq6[4]),
                  mh.fade_replace(eq5[3], eq6[3], coor_mask=RIGHT)
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq4[:2], eq7[:2], eq4[3][:3], eq7[2][:3], eq4[3][-1], eq7[2][-1],
                                eq4[2][-1], eq7[2][-2], run_time=1.6),
                  Succession(Wait(0.6), FadeIn(eq7[2][3])),
                  FadeOut(eq4[2][:-1]))
        self.wait()

class OperatorExp2(OperatorExp):
    def construct(self):
        box, _, _, _ = self.get_eqs()
        MathTex.set_default(font_size=80)
        eq1 = MathTex(r'uX+Pv')
        eq2 = MathTex(r'\langle \phi\vert (uX+Pv)\psi\rangle', r'=',
                      r'\langle(uX+Pv)\phi\vert\psi\rangle')
        eq3 = MathTex(r'e^{i(uX+Pv)}', r'=', r'e^{iuX}e^{iPv}')
        eq4 = MathTex(r'e^{iuX}e^{iPv}', r'\psi(x)', r'=', r'e^{iuX}', r'\psi(x+v)')
        eq4_1 = MathTex(r'e^{iux}')
        eq5 = MathTex(r'e^{iPv}e^{iuX}', r'\psi(x)', r'=', r'e^{iPv}', r'e^{iux}\psi(x)')
        eq6 = MathTex(r'=', r'e^{iu(x+v)}\psi(x+v)')
        eq7 = MathTex(r'e^{i(uX+Pv)}', r'=', r'e^{iPv/2}e^{iuX}e^{iPv/2}')
        eq8 = MathTex(r'\psi_t(x)', r'=', r'e^{it(uX+Pv)}', r'\psi(x)')
        eq9 = MathTex(r'\frac{\partial}{\partial t}', r'\psi_t(x)', r'=', r'i(uX+Pv)', r'\psi(x)')
        mh.font_size_sub(eq9, 0, 70)
        eq9_1 = MathTex(r'\frac{\partial}{\partial t}', r'\psi_t(x)', r'=', r'i\Big(ux+v', r'\frac{\partial}{\partial x}', r'\Big)', r'\psi(x)')
        mh.font_size_sub(eq9_1, 0, 70)
        mh.font_size_sub(eq9, 4, 70)
        eq10 = MathTex(r'\psi_t(x)', r'=', r'e^{itPv/2}', r'e^{ituX}', r'e^{itPv/2}', r'\psi(x)')
        eq10_1 = MathTex(r'\psi(x', r'+tv/2', r')')
        mh.font_size_sub(eq10_1, 1, 60)
        eq10_1 = VGroup(VGroup(*eq10_1[0][:], *eq10_1[1][:], *eq10_1[2][:]))
        eq10_2 = MathTex(r'e^{itux}')
        eq10_3 = MathTex(r'e^{itu(x+tv/2)}', r'\psi(x', r'+tv/2+tv/2', r')')
        mh.font_size_sub(eq10_3, 2, 60)
        eq10_3 = VGroup(eq10_3[0], VGroup(*eq10_3[1][:], *eq10_3[2][:], *eq10_3[3][:]))
        eq11 = MathTex(r'\psi_t(x)', r'=', r'e^{itu(x+tv/2)}', r'\psi(x', r'+tv', r')')
        mh.font_size_sub(eq11, 4, 60)
        VGroup(eq1, eq2, eq3, eq4, eq4_1, eq5, eq6, eq7, eq8, eq8, eq9, eq10, eq10_1, eq10_2, eq10_3, eq11).set_z_index(1)

        eq1.move_to(box)
        eq2.move_to(box)
        eq1.move_to(box).move_to(eq2[0][4:9], coor_mask=UP)
        eq3.move_to(box)
        eq5.next_to(eq4, DOWN, buff=0.6)
        mh.align_sub(eq5, eq5[2], eq4[2], coor_mask=RIGHT)
        VGroup(eq4, eq5).move_to(box)
        mh.align_sub(eq4_1, eq4_1[0][0], eq4[3][0])
        mh.align_sub(eq6, eq6[0], eq5[2])
        eq7.move_to(box)
        eq9.next_to(eq8, DOWN, buff=0.4)
        VGroup(eq8, eq9).move_to(box)
        mh.align_sub(eq10, eq10[1], eq8[1], coor_mask=UP)
        mh.align_sub(eq9_1, eq9_1[1], eq9[1], coor_mask=UP)
        mh.align_sub(eq10_1, eq10_1[0][0], eq10[5][0]).align_to(eq10[4], LEFT)
        mh.align_sub(eq10_2, eq10_2[0][0], eq10[3][0])
        mh.align_sub(eq10_3, eq10_3[0][0], eq10[2][0])
        mh.align_sub(eq11, eq11[1], eq10[1])

        self.add(box, eq1)
        self.wait(0.1)
        eq2_ = mh.align_sub(eq2[0].copy(), eq2[0][4:9], eq1[0], coor_mask=RIGHT)
        self.play(mh.rtransform(eq1[0][:], eq2_[4:9]),
                  Succession(Wait(0.6), FadeIn(eq2_[:4], eq2_[9:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_.copy(), eq2[0]),
                  eq2_.animate.align_to(eq2[2], LEFT),
                  FadeIn(eq2[1]), run_time=1.6)
        self.play(mh.rtransform(eq2_[0], eq2[2][0], eq2_[1:3], eq2[2][8:10],
                                eq2_[3:10], eq2[2][1:8], eq2_[-2:], eq2[2][-2:]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(FadeOut(eq2))
        self.play(FadeIn(eq3))
        self.wait(0.1)
        p = eq3[1].get_center()
        a = 0.3
        line1 = Line(p+DL*a, p+UR*a, stroke_width=7, stroke_color=RED).set_z_index(2)
        line2 = Line(p+UL*a, p+DR*a, stroke_width=7, stroke_color=RED).set_z_index(2)
        self.play(Succession(Create(line1), Create(line2)), run_time=0.8, rate_func=linear)
        self.wait(0.1)
        self.play(FadeOut(eq3[:2], line1, line2),
                  mh.rtransform(eq3[2][:], eq4[0][:], run_time=1.6),
                  Succession(Wait(1.2), FadeIn(eq4[1])))
        self.wait(0.1)
        eq4_ = eq4[:2].copy()
        self.play(FadeIn(eq4[2]), eq4_.animate.align_to(eq4[3], LEFT))
        self.play(mh.rtransform(
            eq4_[0][:4], eq4[3][:], eq4_[1][:3], eq4[4][:3], eq4_[1][-1], eq4[4][-1],
            eq4_[1][3], eq4[4][-1], eq4_[0][-1], eq4[4][4]),
            FadeOut(eq4_[0][4:7]),
            Succession(Wait(0.4), FadeIn(eq4[4][3])),
        run_time=1.2)
        self.wait(0.1)
        # self.play(FadeOut(eq4_[0], ))
        self.play(mh.fade_replace(eq4[3][3], eq4_1[0][3], coor_mask=RIGHT),
                  mh.rtransform(eq4[3][:3], eq4_1[0][:3])
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][:4].copy(), eq5[0][4:], eq4[0][4:].copy(), eq5[0][:4],
                                eq4[1].copy(), eq5[1]))
        self.play(FadeIn(eq5[2]),
                  mh.rtransform(eq5[0][:4].copy(), eq5[3][:], eq5[0][4:7].copy(), eq5[4][:3],
                                eq5[1][:].copy(), eq5[4][4:]),
                  mh.fade_replace(eq5[0][7].copy(), eq5[4][3]),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(eq5[4][:3], eq6[1][:3], eq5[4][3], eq6[1][4],
                                eq5[4][4:7], eq6[1][8:11], eq5[4][-1], eq6[1][-1],
                                eq5[3][-1], eq6[1][6], eq5[3][-1].copy(), eq6[1][-2], run_time=1.4),
                  FadeOut(eq5[3][:-1]),
                  Succession(Wait(0.7), FadeIn(eq6[1][3], eq6[1][5], eq6[1][7], eq6[1][11])))
        self.wait(0.1)
        self.play(FadeOut(eq4[:3], eq4_1, eq4[4:], eq5[:3], eq6[1]))
        self.play(FadeIn(eq7))
        self.wait(0.1)
        self.play(FadeOut(eq7[1:]),
                  mh.rtransform(eq7[0][2:], eq8[2][3:], eq7[0][:2], eq8[2][:2]),
                  FadeIn(eq8[2][2], target_position=eq7[0][1:3]),
                  Succession(Wait(0.7), FadeIn(eq8[:2], eq8[3])),
                  run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq9))
        self.wait(0.1)
        self.play(mh.rtransform(eq9[:3], eq9_1[:3], eq9[3][2], eq9_1[3][2], eq9[3][0], eq9_1[3][0],
                                eq9[3][4], eq9_1[3][4], eq9[3][6], eq9_1[3][5],
                                eq9[4], eq9_1[6]
                                ),
                  mh.stretch_replace(eq9[3][3], eq9_1[3][3]),
                  mh.stretch_replace(eq9[3][1], eq9_1[3][1]),
                  mh.stretch_replace(eq9[3][-1], eq9_1[5]),
                  mh.fade_replace(eq9[3][5], eq9_1[4]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(LaggedStart(FadeOut(eq8[2][3], eq8[2][6], eq8[2][9]),
            AnimationGroup(mh.rtransform(eq8[:2], eq10[:2], eq8[2][:3], eq10[2][:3], eq8[2][:3].copy(), eq10[3][:3], eq8[2][:3].copy(), eq10[4][:3],
                                eq8[2][4:6], eq10[3][3:5], eq8[2][7:9], eq10[2][3:5], eq8[2][7:9].copy(), eq10[4][3:5],
                                eq8[3], eq10[5]),
                  FadeIn(eq10[2][-2:], shift=mh.diff(eq8[2][8], eq10[2][4])),
                  FadeIn(eq10[4][-2:], shift=mh.diff(eq8[2][8], eq10[4][4]))),
                              lag_ratio=0.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq10[5][:3], eq10_1[0][:3], eq10[5][-1], eq10_1[0][-1],
                                eq10[4][-3:-1], eq10_1[0][5:7], eq10[4][2], eq10_1[0][4]),
                  FadeOut(eq10[4][:2], eq10[4][3]),
                  FadeIn(eq10_1[0][3]),
                  mh.stretch_replace(eq10[4][-1], eq10_1[0][7]),
                  run_time=1.3)
        self.wait(0.1)
        self.play(mh.rtransform(eq10[3][:4], eq10_2[0][:4]),
                  mh.stretch_replace(eq10[3][4], eq10_2[0][4]))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq10_2[0][:4], eq10_3[0][:4], eq10_2[0][4], eq10_3[0][5],
                                eq10[2][2], eq10_3[0][7], eq10[2][-3:], eq10_3[0][8:11]),
                  FadeOut(eq10[2][:2], eq10[2][3]),
                  mh.rtransform(eq10_1[0][:3], eq10_3[1][:3],
                                eq10[2][2].copy(), eq10_3[1][4], eq10[2][-3:-1].copy(), eq10_3[1][5:7],
                                eq10_1[0][-6:], eq10_3[1][-6:]),
                  mh.stretch_replace(eq10[2][-1].copy(), eq10_3[1][7]),
                  run_time=1.7),
                  Succession(Wait(1), FadeIn(eq10_3[0][4], eq10_3[0][6], eq10_3[0][11], eq10_3[1][3]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq10[:2], eq11[:2], eq10_3[0], eq11[2],
                                eq10_3[1][:3], eq11[3][:], eq10_3[1][3:6], eq11[4][:]),
                  mh.rtransform(eq10_3[1][8:11], eq11[4][:3], eq10_3[1][-1], eq11[5][0]),
                  FadeOut(eq10_3[1][6:8]),
                  FadeOut(eq10_3[1][11:13], target_position=eq10_3[1][6:8]),
                  run_time=1.5)
        self.wait()

class MixedExp(Scene):
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        eq1 = MathTex(r'\frac{\partial}{\partial t}\psi_t', r'=', r'i(uX+vP)', r'\psi_t')
        eq2 = MathTex(r'\frac{\partial}{\partial t}\psi_t(x)', r'=', r'i\left(uX+vP\right)', r'\psi_t(x)')
        eq3 = MathTex(r'\frac{\partial}{\partial t}\psi_t(x)', r'=', r'i\left(ux-iv\frac{\partial}{\partial x}\right)', r'\psi_t(x)')
        eq4 = MathTex(r'\frac{\partial}{\partial t}\psi_t(x)', r'=', r'\left(iux+v\frac{\partial}{\partial x}\right)', r'\psi_t(x)')
        eq5 = MathTex(r'\left(\frac{\partial}{\partial t}-v\frac{\partial}{\partial x}\right)', r'\psi_t(x)', r'=',
                      r'iux', r'\psi_t(x)')
        self.add(eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][:], eq2[0][:-3], eq1[1], eq2[1], eq1[2], eq2[2], eq1[3][:], eq2[3][:-3]),
                  Succession(Wait(0.4), FadeIn(eq2[0][-3:], eq2[3][-3:])),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:2], eq3[:2], eq2[3], eq3[3],
                                eq2[2][0], eq3[2][0], eq2[2][2], eq3[2][2], eq2[2][5], eq3[2][6]),
                  mh.stretch_replace(eq2[2][1], eq3[2][1]),
                  mh.fade_replace(eq2[2][3], eq3[2][3], coor_mask=RIGHT),
                  mh.fade_replace(eq2[2][4], eq3[2][4], coor_mask=RIGHT),
                  mh.fade_replace(eq2[2][6], eq3[2][7:11], coor_mask=RIGHT),
                  mh.stretch_replace(eq2[2][7], eq3[2][11]),
                  FadeIn(eq3[2][5], shift=mh.diff(eq2[2][5], eq3[2][6])*RIGHT)
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2][0], eq4[2][1], eq3[2][2:4], eq4[2][2:4],
                                eq3[2][6:], eq4[2][5:], eq3[3], eq4[3]),
                  mh.stretch_replace(eq3[2][1], eq4[2][0]),
                  mh.fade_replace(eq3[2][4], eq4[2][4]),
                  FadeOut(eq3[2][5])
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq4[1], eq5[2], eq4[3], eq5[4], eq4[0][:4], eq5[0][1:5],
                                eq4[0][4:], eq5[1][:], eq4[2][0], eq5[0][0], eq4[2][5:], eq5[0][6:],
                                eq4[2][1:4], eq5[3][:]),
                  mh.fade_replace(eq4[2][4], eq5[0][5]),
                  run_time=1.8
                  )
        # self.play(mh.rtransform(eq3[0][1:5], eq4[0][1:5], eq3[0][5:], eq4[1][:], eq3[1], eq4[2],
        #                         eq3[2][5:12], eq4[0][6:13], eq3[3], eq4[4], eq3[2][2:4], eq4[3][:],
        #                         eq3[2][1], eq4[0][0]),
        #           mh.fade_replace(eq3[2][4], eq4[0][5]),
        #           FadeOut(eq3[0][0]),
        #           FadeOut(eq3[2][0]))
        self.wait()

class WignerNarration(Scene):
    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)

        txt1 = Tex(r'\sf Gaussian wavefunction ', r'$\psi(x)\sim e^{-\frac{x^2}{2} }$')
        txt1_1 = Tex(r'$e^{-\frac{x}{2} }$')
        mh.align_sub(txt1_1, txt1_1[0][-3], txt1[1][-4])
        txt1 = VGroup(txt1[0], VGroup(*txt1[1][:-2], *txt1_1[0][-2:])).move_to(ORIGIN)

        txt2 = Tex(r'\sf shift wavefunction ', r'$\psi(x)\to\psi(x-a)$')
        txt3 = Tex(r'\sf shifts distribution along $X$')

        txt4 = Tex(r'\sf scale ', r'$\psi(x)\to e^{iax}\psi(x)$')
        txt5 = Tex(r'\sf shifts distribution along $P$')
        txt6 = Tex(r'\sf squeeze wavefunction ', r'$\psi(x)\to\psi(2x)$')

        txt7 = Tex(r'\sf squeezes distribution along $X$, stretches along $P$')
        txt8 = Tex(r'\sf uncertainty principle: ', r'${\rm dev}(X)\,{\rm dev}(P)\ge1/2$')

        txt9 = Tex(r'\sf stretching wavefunction squeezes distribution along $P$')

        txt10 = Tex(r'\sf scale ', r'$\psi(x)\to e^{iax^2}\psi(x)$')
        txt11 = Tex(r'\sf skews distribution along $P$')

        txt12 = Tex(r'\sf combine stretched and squeezed wavefunctions')
        txt13 = Tex(r'\sf superposition:', r' $\psi(x) \sim \psi_1(x)+\psi_2(x)$')

        txt14 = Tex(r'\sf phase shift:', r' $\psi(x)\sim \psi_1(x) + e^{ia}\psi_2(x)$')
        txt15 = Tex(r'\sf phase shift:', r' $\psi(x)\sim \psi_1(x) -\psi_2(x)$')

        txt16 = Tex(r'\sf shift $\psi$ and scale by $e^{iax}$')

        txt17 = Tex(r'\sf conjugation $\psi\to\psi^*$ reflects through $X$-axis')

        txt18 = Tex(r'\sf Fourier transforming $\psi$ rotates distribution $90^\circ$')

        txt19 = Tex( r'\sf reflecting $\psi(x)\to\psi(-x)$ reflects distribution')

        txt20 = Tex( r'\sf combining distributions: mixed quantum state')

        txt21 = Tex(r'\sf superposition:', r' $\psi(x) \sim \psi_1(x)+\psi_2(x)$')
        txt22 = Tex(r'\sf phase shift:', r' $\psi(x)\sim \psi_1(x) + e^{ia}\psi_2(x)$')

        w = 8
        txts = [txt1, txt2, txt3, txt4, txt5, txt6, txt7, txt8, txt9, txt10,
                txt11, txt12, txt13, txt14, txt15, txt16, txt17, txt18, txt19,
                txt20, txt21, txt22]
        txt1, txt2, txt3, txt4, txt5, txt6, txt7, txt8, txt9, txt10,\
            txt11, txt12, txt13, txt14, txt15, txt16, txt17, txt18, txt19,\
            txt20, txt21, txt22 = (eq_shadow(txt, bg_stroke_width=20) for txt in txts)

        self.add(txt1)
        self.wait(0.1)
        self.play(FadeOut(txt1))
        self.wait(0.1)
        self.play(FadeIn(txt2))
        self.wait(0.1)
        self.play(txt2.animate.next_to(txt3, UP), FadeIn(txt3))
        self.wait(0.1)
        self.play(FadeOut(txt2, txt3))
        self.wait(0.1)
        self.play(FadeIn(txt4))
        self.wait(0.1)
        self.play(txt4.animate.next_to(txt5, UP), FadeIn(txt5))
        self.wait(0.1)
        self.play(FadeOut(txt4, txt5))
        self.wait(0.1)
        self.play(FadeIn(txt6))
        self.wait(0.1)
        self.play(txt6.animate.next_to(txt7, UP), FadeIn(txt7))
        self.wait(0.1)
        self.play(FadeOut(txt7), FadeIn(txt8))
        self.wait(0.1)
        self.play(FadeOut(txt6, txt8))
        self.wait(0.1)
        self.play(FadeIn(txt9))
        self.wait(0.1)
        self.play(FadeOut(txt9))
        self.wait(0.1)
        self.play(FadeIn(txt10))
        self.wait(0.1)
        self.play(txt10.animate.next_to(txt11, UP), FadeIn(txt11))
        self.wait(0.1)
        self.play(FadeOut(txt10, txt11))
        self.wait(0.1)
        self.play(FadeIn(txt12))
        self.wait(0.1)
        self.play(txt12.animate.next_to(txt13, UP), FadeIn(txt13))
        self.wait(0.1)
        self.play(mh.rtransform(txt13[0][-1], txt14[0][-1], txt13[1][:11], txt14[1][:11],
                                txt13[1][-5:], txt14[1][-5:]),
                  FadeOut(txt13[0][:-1], shift=mh.diff(txt13[0][-1], txt14[0][-1])),
                  FadeIn(txt14[0][:-1]))
        self.play(FadeIn(txt14[1][11:-5]))
        self.wait(0.1)
        self.play(mh.rtransform(txt14[0], txt15[0], txt14[1][:10], txt15[1][:10],
                                txt14[1][-5:], txt15[1][-5:]),
                  mh.fade_replace(txt14[1][10], txt15[1][10]),
                  FadeOut(txt14[1][11:-5]))
        self.wait(0.1)
        self.play(FadeOut(txt12, txt15))
        self.wait(0.1)
        self.play(FadeIn(txt16))
        self.wait(0.1)
        self.play(FadeOut(txt16))
        self.wait(0.1)
        self.play(FadeIn(txt17))
        self.wait(0.1)
        self.play(FadeOut(txt17))
        self.wait(0.1)
        self.play(FadeIn(txt18))
        self.wait(0.1)
        self.play(FadeOut(txt18))
        self.wait(0.1)
        self.play(FadeIn(txt19))
        self.wait(0.1)
        self.play(FadeOut(txt19))
        self.wait(0.1)
        self.play(FadeIn(txt20))
        self.wait(0.1)
        self.play(FadeOut(txt20))
        self.play(FadeIn(txt21))
        self.wait(0.1)
        self.play(mh.rtransform(txt21[0][-1], txt22[0][-1], txt21[1][:11], txt22[1][:11],
                                txt21[1][-5:], txt22[1][-5:]),
                  FadeOut(txt21[0][:-1], shift=mh.diff(txt21[0][-1], txt22[0][-1])),
                  FadeIn(txt22[0][:-1]))
        self.play(FadeIn(txt22[1][11:-5]))

        self.wait()

class WignerNarration2(Scene):
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)

        txt1 = Tex(r'\sf simple harmonic motion: ', r'$V(x)=\frac12x^2$')
        txt2 = Tex(r'\sf pendulum: ', r'$V(x)\sim1-\cos x$', r'${}\approx\frac12x^2-\frac1{24}x^4$')
        txt1 = eq_shadow(txt1, bg_stroke_width=20)
        txt2 = eq_shadow(txt2, bg_stroke_width=20)
        mh.align_sub(txt2, txt2[0][-1], txt1[0][-1], coor_mask=UP)
        txt3 = txt2.copy()
        txt2[:2].move_to(ORIGIN, coor_mask=RIGHT)
        self.add(txt1)
        self.wait(0.1)
        shift = mh.diff(txt1[1][8], txt2[1][10])*RIGHT
        self.play(FadeOut(txt1[0][:-1]),
                  FadeIn(txt2[0][:-1]),
                  Succession(Wait(0.4), AnimationGroup(
                  mh.rtransform(txt1[0][-1], txt2[0][-1], txt1[1][:4], txt2[1][:4], txt1[1][8], txt2[1][10]),
                  FadeOut(txt1[1][5:8], txt1[1][9], shift=shift),
                  mh.fade_replace(txt1[1][4], txt2[1][4], coor_mask=RIGHT))),
                  Succession(Wait(0.6), AnimationGroup(
                  FadeIn(txt2[1][5:10], shift=LEFT*0.2))))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(txt2[:2], txt3[:2]), FadeIn(txt3[2]), lag_ratio=0.5))
        self.wait()

class Example(ThreeDScene):
    colors = [
        ManimColor(RED_D.to_rgb() * 0.5),
        ManimColor(RED_E.to_rgb() * 0.5)
    ]

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        xmin, xmax = (-5., 5.)
        ymin, ymax = (-5., 5.)
        zmin, zmax = (-.4, .4)
        xlen = 12.
        ylen = 12.
        zlen = 6.

        ax = ThreeDAxes([xmin, xmax*1.1], [ymin, ymax*1.1], [zmin, zmax], xlen, ylen, zlen,
                        axis_config={'color': WHITE, 'stroke_width': 2, 'include_ticks': False,
                                     "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                                     "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                                     },
                        ).shift(DL*0.5)
        txt1 = MathTex(r'X', stroke_width=2, font_size=60).move_to(ax.coords_to_point(xmax*1.15, 0))
        txt1.rotate(PI/2, RIGHT)
        txt2 = MathTex(r'P', stroke_width=2, font_size=60).move_to(ax.coords_to_point(0, ymax*1.15))
        txt2.rotate(PI/2, RIGHT)
        txt2.rotate(PI/2, OUT)

        params1 = [(.5*4, 0., 0., 0., 0., 1.)]
        params2 = [(.5/4, 0., 0., 0., 0., 1.)]
        params1 = gauss_scale(params1, 1./gauss1d_norm(params1))
        params2 = gauss_scale(params2, -1./gauss1d_norm(params2))
        params = params1  + params2
        params = gauss_scale(params, 1./gauss1d_norm(params))

        #params = gauss_shift(params, 3)
        #params = gauss1d_p_shift(params, 3)
        #params = gauss_tfm(params)
        #params += params1
        #params += gauss_scale(gauss_reflect(params), -1.j)
        wig = gauss_wigner(params, params)

        origin = ax.coords_to_point(0, 0, 0)
        right = ax.coords_to_point(1, 0, 0) - origin
        up = ax.coords_to_point(0, 1, 0) - origin
        out = ax.coords_to_point(0, 0, 1) - origin

        def f(u,v):
            x = (1-u)*xmin + u*xmax
            y = (1-v)*ymin + v*ymax
            z = gauss2d_calc(wig, x, y)
            return origin + right * x + up * y + out * z

        surf = Surface(f,
            stroke_opacity=0.8, checkerboard_colors=self.colors, stroke_color=WHITE,
            resolution=(100, 100), should_make_jagged=False).set_z_index(200)

        self.add(ax, txt1, txt2, surf)
        self.wait()

class STFT(Scene):
    fill_op = 0.7
    bgcol = GREY

    def __init__(self, *args, **kwargs):
        if not config.transparent:
            config.background_color=self.bgcol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=1.5)

        eq1 = MathTex(r'\psi(t)', font_size=100)
        eq2 = MathTex(r'\phi(\omega)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s)e^{-i\omega s}\,ds')
        eq3 = MathTex(r'\phi(\omega)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s)w(s)e^{-i\omega s}\,ds')
        eq4 = MathTex(r'\phi(t,\omega)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s)w(s-t)e^{-i\omega s}\,ds')

        VGroup(eq1, eq2, eq3, eq4).set_z_index(1)

        VGroup(eq1[0][0], eq2[0][0], eq2[3][1], eq2[3][1], eq3[3][5], eq4[0][0]).set_color(col_psi)
        VGroup(eq1[0][2], eq2[3][3], eq2[3][-1], eq2[3][-3], eq3[3][7], eq4[0][2], eq4[3][9]).set_color(col_x)
        VGroup(eq2[0][2], eq2[3][-4], eq4[0][4]).set_color(col_p)
        VGroup(eq2[2][0], eq2[2][-2:], eq2[3][-5]).set_color(col_num)

        mh.copy_eq_colors(eq3[0], eq2[0])
        mh.copy_eq_colors(eq3[3][:5], eq2[3][:5])
        mh.copy_eq_colors(VGroup(*eq4[0][:2], *eq4[0][4:]), VGroup(*eq3[0][:2], *eq3[0][2:]))
        mh.copy_eq_colors(eq4[3][:8], eq3[3][:8])
        mh.copy_eq_colors(eq3[3][9:], eq2[3][5:])
        mh.copy_eq_colors(eq4[3][11:], eq3[3][9:])
        mh.copy_eq_colors(eq3[2], eq2[2])
        mh.copy_eq_colors(eq4[2], eq2[2])

        box1 = SurroundingRectangle(eq2, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                   corner_radius=0.2, buff=0.15)

        VGroup(box1, eq2).to_edge(DOWN, buff=0.1)

        eq1.move_to(box1)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq2[1], coor_mask=UP)

        box2 = SurroundingRectangle(eq3, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                   corner_radius=0.2, buff=0.15)
        box3 = SurroundingRectangle(eq4, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=self.fill_op,
                                   corner_radius=0.2, buff=0.15)

        self.add(box1, eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][:2], eq2[3][1:3], eq1[0][3], eq2[3][4]),
                  mh.fade_replace(eq1[0][2], eq2[3][3], coor_mask=RIGHT),
                  Succession(Wait(0.5), FadeIn(eq2[:3], eq2[3][0], eq2[3][5:])))
        self.wait(0.1)
        self.play(FadeOut(eq2[:3], eq2[3][0], eq2[3][5:], rate_func=linear),
                  Succession(Wait(0.5), mh.rtransform(eq2[3][1:5], eq3[3][1:5])),
                  Succession(Wait(1), FadeIn(eq3[3][5:9])))
        self.wait(0.1)
        self.play(mh.rtransform(box1, box2), Succession(Wait(0.5),
                  FadeIn(eq3[:3], eq3[3][0], eq3[3][9:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0][:2], eq4[0][:2], eq3[0][2:], eq4[0][4:], eq3[1:3], eq4[1:3],
                                eq3[3][:8], eq4[3][:8], eq3[3][8:], eq4[3][10:], box2, box3),
                  Succession(Wait(0.6), FadeIn(eq4[0][2:4], eq4[3][8:10])))
        self.wait()

class STFTWigner(STFT):
    bgcol=BLACK

    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=60)#, stroke_color=col_op, fill_color=col_op, color=col_op)
        eq1 = MathTex(r'\phi(t,\omega)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s)w(s-t)e^{-is\omega}\,ds')
        eq2 = MathTex(r'\phi(t,\omega)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s)w(s-t)^*e^{-is\omega}\,ds')
        eq3 = MathTex(r'\phi(0,0)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s)w(s-0)^*e^{-is0}\,ds')
        eq4 = MathTex(r'\phi(0,0)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s)w(s)^*\,ds')
        eq5 = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\frac1{2\pi}', r'\left(\int\psi(s)w(s)^*\,ds\right)^*', r'\int\psi(s)w(s)^*\,ds')
        eq6 = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\frac1{2\pi}', r'\int\psi(s)^*w(s)\,ds', r'\int\psi(s)w(s)^*\,ds')
        eq7 = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\frac1{2\pi}', r'\int\psi(u)^*w(u)\,du', r'\int\psi(v)w(v)^*\,dv')
        eq8 = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\frac1{2\pi}', r'\iint\psi(u)^*\psi(v)w(u)w(v)^*\,dudv')
        eq9 = MathTex(r'u', r'=', r's-\frac y2', font_size=55)
        eq10 = MathTex(r'v', r'=', r's+\frac y2', font_size=55)
        eq11 = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\iint f_s(y)g_s(y)^*\,dyds')
        eq12 = MathTex(r'f_s(y)', r'=', r'\frac1{\sqrt{2\pi} }', r'\psi(u)^*\psi(v)', font_size=55)
        eq13 = MathTex(r'g_s(y)', r'=', r'\frac1{\sqrt{2\pi} }', r'w(u)^*w(v)', font_size=55)
        mh.font_size_sub(eq12, 2, 44)
        mh.font_size_sub(eq13, 2, 44)
        eq14 = MathTex(r'g_s(y)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int e^{iyz}\hat g_s(z)\,dz', font_size=55)
        mh.font_size_sub(eq14, 2, 50)
        eq15 = MathTex(r'g_s(y)^*', r'=', r'\frac1{\sqrt{2\pi} }', r'\int e^{-iyz}\hat g_s(z)^*\,dz', font_size=55)
        mh.font_size_sub(eq15, 2, 50)
        eq16 = MathTex(r'\int f_s(y)g_s(y)^*\,dy', r'=', r'\int f_s(y)', r'\frac1{\sqrt{2\pi} }', r'\int e^{-iyz}\hat g_s(z)^*\,dz', r'dy', font_size=55)
        mh.font_size_sub(eq16, 3, 50)
        eq17 = MathTex(r'\int f_s(y)g_s(y)^*\,dy', r'=', r'\int', r'\frac1{\sqrt{2\pi} }', r'\int f_s(y)e^{-iyz}\,dy', r'\,\hat g_s(z)^*\,dz', font_size=55)
        mh.font_size_sub(eq17, 3, 50)
        eq18 = MathTex(r'\int f_s(y)g_s(y)^*\,dy', r'=', r'\int\hat f_s(z)\hat g_s(z)^*\,dz', font_size=55)
        eq19_ = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\iint \hat f_s(z)\hat g_s(z)^*\,dzds')
        eq19 = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\iint \hat f_s(z)\hat g_s(z)^*\,dsdz')
        eq20 = MathTex(r'\hat f_s(z)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int', r'f_s(y)', r'e^{-iyz}\,dy', font_size=55)
        mh.font_size_sub(eq20, 2, 50)
        eq21 = MathTex(r'\hat f_s(z)', r'=', r'\frac1{2\pi}', r'\int', r'\psi(u)^*\psi(v)', r'e^{-iyz}\,dy', font_size=55)
        mh.font_size_sub(eq21, 2, 50)
        eq22 = MathTex(r'\hat f_s(z)', r'=', r'\frac1{2\pi}', r'\int', r'\psi\left(s-\frac y2\right)^*\psi\left(s+\frac y2\right)', r'e^{-iyz}\,dy', font_size=55)
        eq23 = MathTex(r'\hat f_s(z)', r'=', r'W_\psi(s,z)')
        eq24 = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\iint W_\psi(s,z)\hat g_s(z)^*\,dsdz')
        eq25 = MathTex(r'\hat g_s(z)', r'=', r'W_w(s,z)')
        eq26 = MathTex(r'\hat g_s(z)^*', r'=', r'W_w(s,z)')
        eq27 = MathTex(r'\lvert\phi(0,0)\rvert^2', r'=', r'\iint W_\psi(s,z)W_w(s,z)\,dsdz')
        eq28 = MathTex(r'\phi(t,\omega)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s)w(s-t)e^{-is\omega}\,ds', font_size=55)
        eq29 = MathTex(r'\phi(t,\omega)', r'=', r'\frac1{\sqrt{2\pi} }', r'\int\psi(s+t)w(s)e^{-i(s+t)\omega}\,ds', font_size=55)
        eq30 = MathTex(r'\phi(t,\omega)', r'=', r'\frac{e^{-it\omega} }{\sqrt{2\pi} }', r'\int e^{-i s\omega}\psi(s+t)w(s)\,ds', font_size=55)
        eq31 = MathTex(r'\lvert\phi(t,\omega)\rvert^2', r'=', r'\iint W_\psi(t+s,\omega+z)W_w(s,z)\,dsdz')

        VGroup(eq1[0][0], eq1[3][1], eq1[3][5], eq12[3][0], eq12[3][5],
               eq23[2][1]).set_color(col_psi)
        VGroup(eq1[0][2], eq1[3][3], eq1[3][7], eq1[3][9], eq1[3][14], eq1[3][-1],
               eq7[3][3], eq7[3][8], eq7[3][11], eq7[4][3], eq7[4][7], eq7[4][11],
               eq9[0], eq9[2][0], eq9[2][2], eq12[0][1], eq12[0][3], eq12[3][2], eq12[3][7], eq14[3][3],
               eq14[3][7], eq16[0][-1], eq16[5][-1], eq23[2][3], eq29[3][5], eq29[3][17],
               eq31[2][5]).set_color(col_x)
        VGroup(eq1[0][4], eq1[3][15], eq14[3][4], eq14[3][9], eq14[3][-1], eq23[2][5], eq31[2][9]).set_color(col_p)
        VGroup(eq3[0][2], eq3[0][4], eq5[0][-1], eq31[0][-1],
               eq1[2][0], eq1[2][-2], eq9[2][4]).set_color(col_num)
        VGroup(eq1[2][-1], eq1[3][11], eq14[3][1]).set_color(col_special)
        VGroup(eq1[3][13], eq2[3][11], eq5[3][-1], eq12[3][4],
               eq11[2][12], eq15[0][-1], eq15[3][-3], eq14[3][2], eq26[0][-1]).set_color(col_i)
        VGroup(eq23[2][0], eq12[0][0], eq14[3][5:7]).set_color(col_WVD)
        VGroup(eq1[2][1], eq1[2][2:-2], eq1[3][0], eq1[3][-2], eq14[3][0], eq14[3][-2],
               eq16[0][0], eq16[0][-2], eq16[2][0], eq16[5][-2], eq20[3], eq5[0][0], eq5[0][-2]).set_color(col_op)

        mh.rtransform.copy_colors = True
        mh.copy_colors_eq(eq1[0], eq31[0][1:-2])
        mh.copy_colors_eq(eq1[2], eq12[2], eq1[2], eq14[2], eq1[2], eq20[2])
        mh.copy_colors_eq(eq9, eq10, eq12, eq13, eq12[0], eq14[0], eq12[0], eq16[0][1:6],
                          eq12[0], eq16[2][1:], eq14[3][5:11], eq18[2][1:7], eq23, eq25,
                          eq14[3][5:11], eq20[0], eq12[0], eq20[4], eq1[3][11:16], eq20[5][:5],
                          eq1[3][-2:], eq20[5][-2:], eq12[3], eq21[4], eq9[2], eq22[4][2:7],
                          eq10[2], eq22[4][11:16], eq14[3][5:11], eq25[0], eq1, eq28
        )

        VGroup(eq3[3][9], eq3[3][16]).set_color(col_num)

        mh.align_sub(eq5, eq5[1], eq4[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[1], eq5[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[1], eq6[1], coor_mask=UP)
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=UP)
        eq9.next_to(eq8, DOWN, buff=0.5)
        eq10.next_to(eq9, DOWN, buff=0.2)
        mh.align_sub(eq10, eq10[1], eq9[1], coor_mask=RIGHT)
        mh.align_sub(eq11, eq11[1], eq8[1], coor_mask=UP)
        mh.align_sub(eq12, eq12[1], eq9[1]).shift(RIGHT*1.5)
        mh.align_sub(eq13, eq13[1], eq10[1]).shift(RIGHT*1.5)
        eq14.next_to(eq8, UP, buff=0.6, coor_mask=UP)
        mh.align_sub(eq15, eq15[1], eq14[1], coor_mask=UP)
        mh.align_sub(eq16, eq16[1], eq15[1], coor_mask=UP)
        mh.align_sub(eq17, eq17[1], eq16[1])
        mh.align_sub(eq18, eq18[1], eq17[1], coor_mask=UP)
        mh.align_sub(eq19_, eq19_[1], eq11[1])
        mh.align_sub(eq19, eq19[1], eq11[1])
        mh.align_sub(eq20, eq20[1], eq18[1], coor_mask=UP)
        mh.align_sub(eq21, eq21[1], eq20[1], coor_mask=UP)
        mh.align_sub(eq22, eq22[1], eq21[1], coor_mask=UP)
        mh.align_sub(eq23, eq23[1], eq22[1], coor_mask=UP)
        mh.align_sub(eq24, eq24[1], eq19[1], coor_mask=UP)
        mh.align_sub(eq26, eq26[1], eq22[1], coor_mask=UP)
        mh.align_sub(eq25, eq25[1], eq26[1])
        mh.align_sub(eq27, eq27[1], eq24[1], coor_mask=UP)
        eq28.next_to(eq27, DOWN, buff=1)
        mh.align_sub(eq29, eq29[1], eq28[1])
        mh.align_sub(eq30, eq30[1], eq29[1])
        mh.align_sub(eq31, eq31[1], eq27[1], coor_mask=UP)

        self.add(eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:3], eq2[:3], eq1[3][:11], eq2[3][:11], eq1[3][11:], eq2[3][12:]),
                  FadeIn(eq2[3][11]))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[0][:2], eq3[0][:2], eq2[0][3], eq3[0][3], eq2[0][-1], eq3[0][-1],
                                eq2[1:3], eq3[1:3], eq2[3][:9], eq3[3][:9], eq2[3][10:16], eq3[3][10:16],
                                eq2[3][17:], eq3[3][17:]),
                  mh.fade_replace(eq2[0][2],eq3[0][2], coor_mask=RIGHT),
                  mh.fade_replace(eq2[0][4],eq3[0][4], coor_mask=RIGHT),
                  mh.fade_replace(eq2[3][9], eq3[3][9], coor_mask=RIGHT),
                  mh.fade_replace(eq2[3][16], eq3[3][16], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        self.play(FadeOut(eq3[3][8:10], eq3[3][12:17]))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:3], eq4[:3], eq3[3][:8], eq4[3][:8], eq3[3][10:12], eq4[3][8:10], eq3[3][17:], eq4[3][10:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0][:], eq5[0][1:-2], eq4[1], eq5[1], eq4[2][:2], eq5[2][:2],
                                eq4[2][-2:], eq5[2][-2:], eq4[3][:], eq5[3][1:-2],
                                eq4[3].copy().set_z_index(4), eq5[4].set_z_index(4)),
                  FadeOut(eq4[2][2:-2], shift=mh.diff(eq4[2][-2:], eq5[2][-2:])),
                  FadeIn(eq5[0][0], eq5[0][-2:], shift=mh.diff(eq4[0][:], eq5[0][1:-2])),
                  FadeIn(eq5[3][0], eq5[3][-2:], shift=mh.diff(eq4[3][:], eq5[3][1:-2])),
                  run_time=1.8
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:3], eq6[:3], eq5[4], eq6[4], eq5[3][1:6], eq6[3][:5],
                                eq5[3][6:10], eq6[3][6:10], eq5[3][11:13], eq6[3][10:12],
                                eq5[3][-1], eq6[3][5]),
                  FadeOut(eq5[3][0], shift=mh.diff(eq5[3][1], eq6[3][0])),
                  FadeOut(eq5[3][-2], shift=mh.diff(eq5[3][-3], eq6[3][-1])),
                  FadeOut(eq5[3][10], shift=mh.diff(eq5[3][9], eq6[3][9])),
                  run_time=1.5
                  )
        self.play(mh.rtransform(eq6[:3], eq7[:3], eq6[3][:3], eq7[3][:3],
                                eq6[3][4:8], eq7[3][4:8], eq6[3][9:-1], eq7[3][9:-1]),
                  mh.fade_replace(eq6[3][3], eq7[3][3]),
                  mh.fade_replace(eq6[3][8], eq7[3][8]),
                  mh.fade_replace(eq6[3][-1], eq7[3][-1]),
                  mh.rtransform(eq6[4][:3], eq7[4][:3],
                                eq6[4][4:7], eq7[4][4:7], eq6[4][8:-1], eq7[4][8:-1]),
                  mh.fade_replace(eq6[4][3], eq7[4][3]),
                  mh.fade_replace(eq6[4][7], eq7[4][7]),
                  mh.fade_replace(eq6[4][-1], eq7[4][-1]),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:3], eq8[:3], eq7[3][0], eq8[3][0],
                                eq7[3][1:6], eq8[3][2:7], eq7[3][6:10], eq8[3][11:15],
                                eq7[3][10:12], eq8[3][20:22], eq7[4][0], eq8[3][1],
                                eq7[4][1:5], eq8[3][7:11], eq7[4][5:10], eq8[3][15:20],
                                eq7[4][10:], eq8[3][22:], copy_colors=True),
                  run_time=1.8)
        self.wait(0.1)
        self.play(FadeIn(eq9, eq10))
        self.wait(0.1)
        eq11_1 = eq11[2][-4:].copy().align_to(eq8[3][-4], LEFT)
        self.play(mh.rtransform(eq9[2][0].copy(), eq11_1[-1], eq8[3][-2], eq11_1[-2],
                                eq9[2][2].copy(), eq11_1[-3], eq8[3][-4], eq11_1[-4]),
                  mh.FadeOut(eq8[3][-1], eq8[3][-3]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(VGroup(eq9, eq10).animate.next_to(eq12, LEFT, buff=2, coor_mask=RIGHT),
                  Succession(Wait(0.5), FadeIn(eq12, eq13)), run_time=1.2)
        self.wait(0.1)
        eq11_2 = eq11[2][2:7].copy().move_to(eq8[3][2:11], coor_mask=RIGHT)
        eq8_1 = MathTex(r'\frac1{\sqrt{2\pi} }', font_size=55)[0]
        mh.copy_colors_eq(eq1[2], eq8_1)
        mh.align_sub(eq8_1, eq8_1[1], eq8[2][1])
        self.play(FadeOut(eq8[3][2:11]), mh.rtransform(eq12[0][:].copy(), eq11_2),
                  mh.rtransform(eq8[2][:2], eq8_1[:2], eq8[2][-2:], eq8_1[-2:]),
                  FadeIn(eq8_1[2:-2]))
        self.wait(0.1)
        eq11_3 = eq11[2][7:13].copy().move_to(eq8[3][11:20], coor_mask=RIGHT)
        self.play(FadeOut(eq8[3][11:20]), mh.rtransform(eq13[0][:].copy(), eq11_3[:-1]),
                  FadeIn(eq11_3[-1], shift=mh.diff(eq13[0][-1], eq11_3[-2])),
                  FadeOut(eq8_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq8[:2], eq11[:2], eq8[3][:2], eq11[2][:2],
                                eq11_2, eq11[2][2:7], eq11_3, eq11[2][7:13], eq11_1, eq11[2][-4:]))
        self.wait(0.1)
        self.play(FadeIn(eq14))
        self.wait(0.1)
        self.play(mh.rtransform(eq14[0][:], eq15[0][:-1], eq14[1:3], eq15[1:3],
                                eq14[3][:2], eq15[3][:2], eq14[3][2:11], eq15[3][3:12],
                                eq14[3][11:], eq15[3][13:]),
                  Succession(Wait(0.4), FadeIn(eq15[0][-1], eq15[3][2], eq15[3][12])))
        self.wait(0.1)
        self.play(mh.rtransform(eq15[0][:], eq16[0][6:12], eq15[1], eq16[1],eq15[2], eq16[3],
                                eq15[3], eq16[4]),
                  Succession(Wait(0.6), FadeIn(eq16[0][:6], eq16[0][12:], eq16[2], eq16[5])))
        self.wait(0.1)
        self.play(mh.rtransform(eq16[:2], eq17[:2], eq16[2][:], eq17[4][:6], eq16[3], eq17[3],
                                eq16[4][0], eq17[2][0], eq16[4][1:6], eq17[4][6:11],
                                eq16[5][:], eq17[4][11:], eq16[4][6:], eq17[5][:],
                                run_time=1.8))
        self.wait(0.1)
        eq18_1 = eq18[2][1:7].copy()
        mh.align_sub(eq18_1, eq18_1[0], eq18[2][1], coor_mask=RIGHT)
        self.play(FadeOut(eq17[3:5], run_time=1.5), FadeIn(eq18_1, run_time=1.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq17[:2], eq18[:2], eq17[2][0], eq18[2][0],
                                eq18_1, eq18[2][1:7], eq17[5][:], eq18[2][7:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[:2], eq19_[:2], eq11[2][0], eq19_[2][0],
                                eq11[2][-2:], eq19_[2][-2:]),
                  FadeOut(eq11[2][1:-2]), run_time=1.5)
        self.play(mh.rtransform(eq18[2][:], eq19_[2][1:-2]),
                  FadeOut(eq18[:2]), run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq19_[:2], eq19[:2], eq19_[2][:-4], eq19[2][:-4],
                                eq19_[2][-4:-2], eq19[2][-2:], eq19_[2][-2:], eq19[2][-4:-2]))
        self.wait(0.1)
        self.play(FadeIn(eq20))
        self.wait(0.1)
        self.play(mh.rtransform(eq20[:2], eq21[:2], eq20[2][:2], eq21[2][:2],
                                eq20[3][0], eq21[3][0], eq20[5][:], eq21[5][:],
                                eq20[2][-2:], eq21[2][-2:]),
                  FadeOut(eq20[2][2:-2], shift=mh.diff(eq20[2][-2:], eq21[2][-2:])),
                  # eq20[2][2:].animate.shift(mh.diff(eq20[2][1], eq21[2][1])),
                  Succession(Wait(0.5), AnimationGroup(FadeOut(eq20[4]), FadeIn(eq21[4]),
                                                       # mh.rtransform(eq20[2][-2:], eq21[2][-2:]),
                                                       # FadeOut(eq20[2][2:-2]),
                                                       ))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq21[:2], eq22[:2], eq21[2][:], eq22[2][:],
                                eq21[3], eq22[3], eq21[4][0], eq22[4][0],
                                eq21[4][4:6], eq22[4][8:10],
                                eq21[5], eq22[5]),
                  eq21[4][2].animate.move_to(eq22[4][2:7], coor_mask=RIGHT),
                  eq21[4][7].animate.move_to(eq22[4][11:16], coor_mask=RIGHT),
                  mh.stretch_replace(eq21[4][1], eq22[4][1]),
                  mh.stretch_replace(eq21[4][3], eq22[4][7]),
                  mh.stretch_replace(eq21[4][6], eq22[4][10]),
                  mh.stretch_replace(eq21[4][-1], eq22[4][-1]),
                  )
        self.play(FadeOut(eq21[4][2], eq21[4][7]), FadeIn(eq22[4][2:7], eq22[4][11:16]))
        self.wait(0.1)
        self.play(FadeOut(eq22[2:], run_time=1.6), FadeIn(eq23[2], run_time=1.6),
                  Succession(Wait(0.5), mh.rtransform(eq22[:2], eq23[:2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq19[:2], eq24[:2], eq19[2][:2], eq24[2][:2], eq19[2][-11:], eq24[2][-11:],
                                eq23[2][:], eq24[2][2:-11]),
                  FadeOut(eq19[2][2:-11]),
                  FadeOut(eq23[:2]),
                  run_time=1.6)
        self.wait(0.1)
        self.play(FadeIn(eq25))
        self.wait(0.1)
        self.play(mh.rtransform(eq25[0][:], eq26[0][:-1], eq25[1:], eq26[1:]),
                  FadeIn(eq26[0][-1], shift=mh.diff(eq25[0][-1], eq26[0][-2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq24[:2], eq27[:2], eq24[2][:9], eq27[2][:9],
                                eq24[2][-4:], eq27[2][-4:], eq26[2][:], eq27[2][9:-4]),
                  FadeOut(eq24[2][9:-4]),
                  FadeOut(eq26[:2]),
                  run_time=1.6)
        self.wait(0.1)
        self.play(FadeOut(eq9, eq10, eq12, eq13))
        self.wait(0.1)
        self.play(FadeIn(eq28))
        self.wait(0.1)
        self.play(mh.rtransform(eq28[:3], eq29[:3], eq28[3][:4], eq29[3][:4], eq28[3][4:8], eq29[3][6:10],
                                eq28[3][10:14], eq29[3][10:14], eq28[3][14], eq29[3][15],
                                eq28[3][15:], eq29[3][19:]),
                  FadeOut(eq28[3][8:10]),
                  Succession(Wait(0.2), FadeIn(eq29[3][4:6], eq29[3][14], eq29[3][16:19])))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq29[:2], eq30[:2], eq29[2][1:], eq30[2][5:], eq29[3][0], eq30[3][0],
                          eq29[3][1:11], eq30[3][6:16], eq29[3][-2:], eq30[3][-2:]),
            mh.rtransform(eq29[3][11:14], eq30[2][:3], eq29[3][17], eq30[2][3], eq29[3][19].copy(), eq30[2][4], path_arc=PI/3),
            mh.rtransform(eq29[3][11:14].copy(), eq30[3][1:4], eq29[3][15], eq30[3][4], eq29[3][19], eq30[3][5], path_arc=PI/2),
            FadeOut(eq29[2][0]), run_time=2),
            FadeOut(eq29[3][14], eq29[3][16], eq29[3][18], run_time=1)
            )
        self.wait(0.1)
        s1 = mh.diff(eq27[2][5], eq31[2][7])
        s2 = mh.diff(eq27[2][7], eq31[2][9])
        self.play(mh.rtransform(eq27[0][:3], eq31[0][:3], eq27[0][4], eq31[0][4], eq27[0][6:], eq31[0][6:],
                                eq27[1], eq31[1], eq27[2][:5], eq31[2][:5], eq27[2][5:7], eq31[2][7:9],
                                eq27[2][7:], eq31[2][11:]),
                  mh.fade_replace(eq27[0][3], eq31[0][3], coor_mask=RIGHT),
                  mh.fade_replace(eq27[0][5], eq31[0][5], coor_mask=RIGHT),
                  eq31[2][5:7].set_opacity(-2).shift(-s1).animate.set_opacity(1).shift(s1),
                  eq31[2][9:11].set_opacity(-2).shift(-s2).animate.set_opacity(1).shift(s2),
                  run_time=1.2
                  )
        self.wait(0.1)
        self.play(FadeOut(eq30))
        self.wait()

"""
integral exp(-ax^2+bx) dx
y = sqrt(a)x-b/2/sqrt(a)
integral exp(-y^2+b^2/(4a)) dy / sqrt(a) = sqrt(pi/a) * exp(b^2/4a)
"""
if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True, 'fps': 15}):
        STFTWigner().render()

