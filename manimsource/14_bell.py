from manim import *
import numpy as np
import math
import sys
import scipy as sp

sys.path.append('../')
import manimhelper as mh

class Prob25(Scene):
    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = GREY
        MathTex.set_default(font_size=100)
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=100)
        eq1 = MathTex(r'\mathbb P({\sf property\ fails})', r'\ge', r'1/4')
        eq1[1:].next_to(eq1[0], DOWN, buff=0.5)
        eq1[0][2:-1].set_stroke(width=2)

        eq2 = MathTex(r'\ge', r'25\%')
        mh.align_sub(eq2, eq2[0], eq1[1])

        VGroup(eq1[2], eq2[1]).set_color(BLUE)
        eq1[0][2:-1].set_color(GREEN)

        self.add(eq1)
        self.wait()
        self.play(FadeOut(eq1[2]), FadeIn(eq2[1]))
        self.wait()

class Prob15(Prob25):
    def construct(self):
        eq1 = MathTex(r'=', r'15\%')
        eq1[1].set_color(BLUE)
        self.add(eq1)

class CHSH(Scene):
    def __init__(self, *args, **kwargs):
        #if config.transparent:
        config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    #col1 = ManimColor('#FF2000')
    #col2 = ManimColor('#0080FF')
    col1 = GREEN
    col2 = GREEN
    colA = RED
    colB = PURPLE

    def construct(self):
        MathTex.set_default(font_size=65)
        eq1 = MathTex(r'\lvert', r'\langle A_1B_1\rangle', r'+', r'\langle A_1B_2\rangle',
                      r'+', r'\langle A_2B_1\rangle', r'-', r'\langle A_2B_2\rangle',
                      r'\rvert', r'\le', r'2')

        eq2 = MathTex(r'A_1', r'=\pm1', r'A_2', r'=\pm1')
        eq3 = MathTex(r'B_1', r'=\pm1', r'B_2', r'=\pm1')

        VGroup(eq1, eq2, eq3).set_z_index(2)

        VGroup(eq1[-1], eq2[1][-2:], eq2[3][-2:], eq3[1][-2:], eq3[3][-2:]).set_color(BLUE)
        VGroup(eq1[1][2], eq1[1][4], eq1[3][2], eq1[5][4],
               eq2[0][1], eq3[0][1]).set_color(self.col1)
        VGroup(eq1[3][4], eq1[5][2], eq1[7][2], eq1[7][4],
               eq2[2][1], eq3[2][1]).set_color(self.col2)
        VGroup(*[eq1[i][1] for i in [1, 3, 5, 7]],
               eq2[0][0], eq2[2][0]).set_color(self.colA)
        VGroup(*[eq1[i][3] for i in [1, 3, 5, 7]],
               eq3[0][0], eq3[2][0]).set_color(self.colB)

        box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)
        VGroup(box1, eq1).to_edge(DOWN, buff=0.1)

        for eq in [eq2, eq3]:
            eq.next_to(eq1, UP)
            eq[:2].move_to(eq1[1:4], coor_mask=RIGHT)
            eq[2:].move_to(eq1[5:8], coor_mask=RIGHT)

        box2 = SurroundingRectangle(VGroup(eq1, eq2, eq3), stroke_width=0, stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)

        self.add(box1, eq1)
        self.wait(0.1)

        eqAs = [eq1[i][1:3] for i in [1, 3, 5, 7]]
        eqBs = [eq1[i][3:5] for i in [1, 5, 3, 7]]
        boxtfm = [ReplacementTransform(box1, box2, run_time=0.8)]
        for eqs, eq in [(eqAs, eq2), (eqBs, eq3)]:
            eqs2 = [_.copy() for _ in eqs]
            self.play(*[_.animate.scale(1.2).set_stroke(width=2) for _ in eqs])
            self.wait(0.1)
            self.play(LaggedStart(AnimationGroup(mh.rtransform(eqs[0].copy(), eq[0][:], eqs[2].copy(), eq[2][:]),
                      mh.rtransform(eqs[1].copy(), eq[0][:], eqs[3].copy(), eq[2][:]), *boxtfm, run_time=1.5),
                      FadeIn(eq[1], eq[3], rate_func=linear), lag_ratio=0.3))
            boxtfm = []
            self.wait(0.1)
            self.play(*[Transform(x, y) for (x,y) in zip(eqs, eqs2)],
                      FadeOut(eq))
            self.wait(0.1)

        eq4 = MathTex(r'\langle A_1B_1\rangle', r'=', r'1{\sf\ if\ }A_1=B_1').set_z_index(2)
        eq5 = MathTex(r'\langle A_1B_1\rangle', r'=', r'\begin{cases}'
                             r'1{\sf\ if\ }A_1=B_1 \\'
                             r'-1{\sf\ if\ }A_1=-B_1'
                             r'\end{cases}').set_z_index(2)
        eq6 = MathTex(r'\langle A_1B_1\rangle', r'=', r'{\sf expected\ value\ of}', r'\begin{cases}'
                             r'1{\sf\ if\ }A_1=B_1 \\'
                             r'-1{\sf\ if\ }A_1=-B_1'
                             r'\end{cases}').set_z_index(2)
        eq7 = MathTex(r'\langle A_1B_1\rangle', r'=', r'\mathbb P(A_1=B_1)', r'-', r'\mathbb P(A_1=-B_1)').set_z_index(2)
        eq8 = MathTex(r'\langle A_1B_1\rangle', r'=', r'\mathbb P(A_1=B_1)', r'-1+', r'\mathbb P(A_1=B_1)').set_z_index(2)
        eq9 = MathTex(r'\langle A_1B_1\rangle', r'=', r'2\mathbb P(A_1=B_1)', r'-1').set_z_index(2)

        VGroup(eq6[2], eq5[2][2:4], eq5[2][11:13], eq6[3][2:4], eq6[3][11:13], eq7[2][0], eq7[4][0],
               eq8[2][0], eq8[4][0], eq9[2][1]).set_color(YELLOW)
        VGroup(eq4[2][0], eq5[2][1], eq5[2][10], eq6[3][1], eq6[3][10], eq8[3][1], eq9[3][1], eq9[2][0]).set_color(BLUE)
        VGroup(eq4[0][1], eq4[2][3], eq5[0][1], eq5[2][4], eq5[2][13], eq6[0][1], eq6[3][4], eq6[3][13],
               eq7[0][1], eq7[2][2], eq7[4][2], eq8[0][1], eq8[2][2], eq8[4][2], eq9[0][1], eq9[2][3]).set_color(self.colA)
        VGroup(eq4[0][3], eq4[2][6], eq5[0][3], eq5[2][7], eq5[2][17], eq6[0][3], eq6[3][7], eq6[3][17],
               eq7[0][3], eq7[2][5], eq7[4][6], eq8[0][3], eq8[2][5], eq8[4][5], eq9[0][3], eq9[2][6]).set_color(self.colB)
        VGroup(eq4[0][2], eq4[2][4], eq5[0][2], eq5[2][5], eq5[2][14], eq6[0][2], eq6[3][5], eq6[3][14],
               eq7[0][2], eq7[2][3], eq7[4][3], eq8[0][2], eq8[2][3], eq8[4][3], eq9[0][2], eq9[2][4]).set_color(self.col1)
        VGroup(eq4[0][4], eq4[2][7], eq5[0][4], eq5[2][8], eq5[2][18], eq6[0][4], eq6[3][8], eq6[3][18],
               eq7[0][4], eq7[2][6], eq7[4][7], eq8[0][4], eq8[2][6], eq8[4][6], eq9[0][4], eq9[2][7]).set_color(self.col2)

        eq4.next_to(eq1, UP)
        eq5.next_to(eq1, UP)
        mh.align_sub(eq6, eq6[0], eq5[0], coor_mask=UP)
        mh.align_sub(eq7, eq7[0], eq6[0], coor_mask=UP)
        mh.align_sub(eq8, eq8[0], eq6[0], coor_mask=UP)
        mh.align_sub(eq9, eq9[0], eq8[0])
        #eq8.next_to(eq1, UP)

        box3 = SurroundingRectangle(VGroup(eq1, eq4), stroke_width=0, stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)
        box4 = SurroundingRectangle(VGroup(eq1, eq5), stroke_width=0, stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)
        box5 = SurroundingRectangle(VGroup(eq1, eq6), stroke_width=0, stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)


        self.play(eq1[1].animate.scale(1.2).set_stroke(width=1.6))
        self.wait(0.1)
        self.play(mh.rtransform(box2, box3), mh.rtransform(eq1[1].copy(), eq4[0]), FadeIn(eq4[1:]))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq4[:2], eq5[:2], eq4[2][:], eq5[2][1:9], box3, box4),
                              FadeIn(eq5[2][0], eq5[2][9:]), lag_ratio=0.1), run_time=2)
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq5[:2], eq6[:2], eq5[2], eq6[3], box4, box5), FadeIn(eq6[2]), lag_ratio=0.5), run_time=1.6)
        self.wait(0.1)
        self.play(LaggedStart(FadeOut(eq6[2], eq6[3][0], eq6[3][2:4], eq6[3][11:13], rate_func=linear, run_time=0.8),
                  AnimationGroup(mh.rtransform(eq6[:2], eq7[:2], eq6[3][4:9], eq7[2][2:7], eq6[3][9], eq7[3][0], eq6[3][13:19], eq7[4][2:8]),
                  FadeOut(eq6[3][1], target_position=eq7[2][0]),
                  FadeOut(eq6[3][10], target_position=eq7[4][0]),
                  FadeIn(eq7[2][:2], eq7[2][-1], shift=mh.diff(eq6[3][4:9], eq7[2][2:7])),
                  FadeIn(eq7[4][:2], eq7[4][-1], shift=mh.diff(eq6[3][13:19], eq7[4][2:8])), run_time=1.8),
                              lag_ratio=0.3))
        self.wait(0.1)
        self.play(mh.rtransform(eq7[:3], eq8[:3], eq7[3][0], eq8[3][0], eq7[4][:5], eq8[4][:5],
                                eq7[4][6:], eq8[4][5:]),
                  FadeOut(eq7[4][5], shift=mh.diff(eq7[4][6], eq8[4][5])),
                  FadeIn(eq8[3][1:], shift=mh.diff(eq7[4][0], eq8[4][0])),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq8[:2], eq9[:2], eq8[2][:], eq9[2][1:], eq8[3][:2], eq9[3][:]),
                  mh.rtransform(eq8[4][:], eq9[2][1:]),
                  FadeOut(eq8[3][2]),
                  FadeIn(eq9[2][0], shift=mh.diff(eq8[2][0], eq9[2][1])),
                  run_time=1.6)

        self.wait()
