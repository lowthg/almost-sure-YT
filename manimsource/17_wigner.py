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

