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