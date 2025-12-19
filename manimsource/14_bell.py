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
        if config.transparent:
            config.background_color=BLACK
        else:
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
        box_op = 0.6
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

        box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_opacity=box_op, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)
        VGroup(box1, eq1).to_edge(DOWN, buff=0.1)

        for eq in [eq2, eq3]:
            eq.next_to(eq1, UP)
            eq[:2].move_to(eq1[1:4], coor_mask=RIGHT)
            eq[2:].move_to(eq1[5:8], coor_mask=RIGHT)

        box2 = SurroundingRectangle(VGroup(eq1, eq2, eq3), stroke_width=0, stroke_opacity=0, fill_opacity=box_op, fill_color=BLACK,
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

        box3 = SurroundingRectangle(VGroup(eq1, eq4), stroke_width=0, stroke_opacity=0, fill_opacity=box_op, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)
        box4 = SurroundingRectangle(VGroup(eq1, eq5), stroke_width=0, stroke_opacity=0, fill_opacity=box_op, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)
        box5 = SurroundingRectangle(VGroup(eq1, eq6), stroke_width=0, stroke_opacity=0, fill_opacity=box_op, fill_color=BLACK,
                                    corner_radius=0.2, buff=0.2)


        eq1_1 = eq1[1].copy()
        self.play(eq1[1].animate.scale(1.2).set_stroke(width=1.6))
        self.wait(0.1)
        self.play(mh.rtransform(box2, box3), mh.rtransform(eq1[1].copy(), eq4[0]), FadeIn(eq4[1:]))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq4[:2], eq5[:2], eq4[2][:], eq5[2][1:9], box3, box4),
                              FadeIn(eq5[2][0], eq5[2][9:]), lag_ratio=0.1), run_time=2)
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq5[:2], eq6[:2], eq5[2], eq6[3], box4, box5), FadeIn(eq6[2]), lag_ratio=0.5), run_time=1.6)
        self.wait(0.1)
        self.play(mh.transform(eq1[1], eq1_1))
        self.wait(0.1)

        # remove abs
        np.random.seed(0)
        pts = [[eq.get_top() + UL*0.2, eq.get_top() + UR * 0.1,
                eq.get_bottom() + DR * 0.8, eq.get_bottom() + DL * 0.7,
                eq.get_top() + UL*0.1, eq.get_top() + UR * 0.1] for eq in (eq1[0], eq1[-3])]
        for p in pts:
            for q in p:
                q += np.random.uniform(-1, 1) * 0.1 * RIGHT + np.random.uniform(-1, 1) * 0.1 * UP
        circs = [ParametricFunction(bezier(p), color=RED, stroke_width=10).set_z_index(3) for p in pts]
        self.play(Create(circs[0]), rate_func=linear, run_time=0.8)
        self.wait(0.2)
        self.play(Create(circs[1]), rate_func=linear, run_time=0.8)
        self.wait(0.1)
        self.play(FadeOut(*circs), VGroup(eq1[0], eq1[-3]).animate.set_opacity(0))
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
        self.wait(0.1)

        eq10 = MathTex(r'2\mathbb P(A_1=B_1)', r'-1+', r'2\mathbb P(A_1=B_2)', r'-1+',
                       r'2\mathbb P(A_2=B_1)', r'-1+', r'2\mathbb P(A_2=-B_2)', r'-1', r'\le', r'2').set_z_index(2)
        eq11 = MathTex(r'{}+', r'\langle A_2(-B_2)\rangle')
        eq12 = MathTex(r'\le', r'2+4')
        eq13 = MathTex(r'\le', r'6')
        eq14 = MathTex(r'\le', r'6/2')
        eq15 = MathTex(r'\le', r'3')
        eq16 = MathTex(r'{\sf observation\ }', r'O_{ij}', r'=', r'\begin{cases} \{A_i=B_j\} '
                                                                r'{\sf\ if\ }i=1{\sf\ or\ }j=1 \\ '
                                                                r'\{A_i=-B_j\} {\sf\ if\ }i=j=1 \end{cases}',
                       font_size=59)
        eq17 = MathTex(r'\frac14\left(', r'\mathbb P(O_{11})', r'+', r'\mathbb P(O_{12})', r'+', r'\mathbb P(O_{21})', r'+',
                       r'\mathbb P(O_{22})', r'\right)', r'\le', r'3')
        eq18 = MathTex(r'\le', r'\frac34').set_z_index(3)
        eq19 = MathTex(r'\mathbb P(O_{ij})', r'\le', r'\frac34').set_z_index(3)
        eq20 = MathTex(r'i,j', r'{\sf\ independent\ uniform\ on\ }', r'\{1,2\}', font_size=59)
        VGroup(eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq20).set_z_index(2)

        VGroup(eq10[0][0], eq10[2][0], eq10[4][0], eq10[6][0], eq10[9][0], eq10[1][1], eq10[3][1], eq10[5][1],
               eq10[7][1], eq12[1][0], eq12[1][2], eq13[1], eq14[1][0], eq14[1][2], eq15[1],
               eq16[3][12], eq16[3][17], eq17[-1], eq16[3][-1], eq17[0][0], eq17[0][2],
               eq18[1][0], eq18[1][2], eq19[2][0], eq19[2][2], eq20[2][1], eq20[2][3]).set_color(BLUE)
        VGroup(eq10[0][1], eq10[2][1], eq10[4][1], eq10[6][1], eq16[0], eq16[3][8:10], eq16[3][13:15],
               eq17[1][0], eq17[3][0], eq17[5][0], eq17[7][0], eq16[3][26:28], eq19[0][0], eq20[1]).set_color(YELLOW)
        VGroup(eq10[0][3], eq10[2][3], eq10[4][3], eq10[6][3], eq11[1][1],
               eq16[3][2], eq17[7][2], eq16[3][19]).set_color(self.colA)
        VGroup(eq10[0][6], eq10[2][6], eq10[4][6], eq10[6][7], eq11[1][5],
               eq16[3][5], eq16[3][23]).set_color(self.colB)
        VGroup(eq10[0][4], eq10[0][7], eq10[2][4], eq10[4][7], eq16[1][1:], eq16[3][3], eq16[3][6],
               eq17[1][3:5], eq17[3][3:5], eq17[5][3:5], eq17[7][3:5], eq20[0][0], eq20[0][2],
               eq16[3][10], eq16[3][15], eq16[3][20], eq16[3][24], eq16[3][28], eq16[3][30], eq19[0][3:5]).set_color(self.col1)
        VGroup(eq10[2][7], eq10[4][4], eq10[6][4], eq10[6][8], eq11[1][2], eq11[1][6]).set_color(self.col2)
        colO = BLUE_E
        VGroup(eq16[1][0], eq17[1][2], eq17[3][2], eq17[5][2], eq17[7][2], eq19[0][2]).set_color(colO)

        eq10[4:].next_to(eq10[:4], DOWN, buff=0.6).align_to(eq10[0], LEFT)
        mh.align_sub(eq10, eq10[-2], eq1[-2]).move_to(ORIGIN, coor_mask=RIGHT)
        eq10[-2:].move_to(eq1[-2:])
        eq10.shift(UP*1.5)
        mh.align_sub(eq12, eq12[0], eq10[-2]).shift(LEFT)
        mh.align_sub(eq13, eq13[0], eq12[0])
        mh.align_sub(eq14, eq14[0], eq13[0])
        mh.align_sub(eq15, eq15[0], eq14[0])
        #mh.align_sub(eq17, eq17[-2], eq15[0], coor_mask=UP)
        eq17.align_to(box5, DOWN).shift(UP*0.2)
        eq16.generate_target().next_to(eq17[1:], UP)
        eq16.next_to(eq10, UP, buff=0.4, coor_mask=UP)
        mh.align_sub(eq18, eq18[0], eq17[-2])
        mh.align_sub(eq19, eq19[0][0], eq17[1][0]).align_to(eq17, LEFT)
        #eq20.move_to((eq19.get_right()+box5.get_right())/2).move_to(eq19, coor_mask=UP)
        eq20.next_to(box5, direction=ORIGIN, aligned_edge=RIGHT).shift(LEFT*0.4)
        mh.align_sub(eq20, eq20[1][0], eq19[0][0], aligned_edge=DOWN, coor_mask=UP)

        shift = mh.diff(eq1[-2], eq10[-2])
        self.play(mh.rtransform(eq9[2], eq10[0], eq9[3][:], eq10[1][:2], eq1[2][0], eq10[1][2]),
                  mh.rtransform(eq1[1][1:3], eq10[0][3:5], eq1[1][3:5], eq10[0][6:8]),
                  eq9[:2].animate.shift(mh.diff(eq9[2], eq10[0])).set_opacity(-1),
                  FadeOut(eq1[1][0], shift=mh.diff(eq1[1][2], eq10[0][4])),
                  FadeOut(eq1[1][5], shift=mh.diff(eq1[1][4], eq10[0][7])),
                  eq1[3:5].animate.shift(mh.diff(eq1[3][1], eq10[2][3])),
                  eq1[5].animate.shift(LEFT+shift),
                  eq1[6:].animate.shift(shift),
                  run_time=2)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[3][1:3], eq10[2][3:5], eq1[3][3:5], eq10[2][6:8], eq1[4][0], eq10[3][2]),
                  FadeOut(eq1[3][0], shift=mh.diff(eq1[3][1], eq10[2][3])),
                  FadeOut(eq1[3][5], shift=mh.diff(eq1[3][4], eq10[2][7])),
                  FadeIn(eq10[2][:3], shift=mh.diff(eq1[3][1], eq10[2][3])),
                  FadeIn(eq10[3][:2]+eq10[2][-1], shift=mh.diff(eq1[3][4], eq10[2][7])),
                  FadeIn(eq10[2][5], shift=mh.diff(eq1[3][1:5], eq10[2][5])),
                  run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[5][1:3], eq10[4][3:5], eq1[5][3:5], eq10[4][6:8]),
                  mh.fade_replace(eq1[5][0], eq10[4][2]),
                  mh.fade_replace(eq1[5][5], eq10[4][8]),
                  FadeIn(eq10[4][:2], shift=mh.diff(eq1[5][0], eq10[4][2])),
                  FadeIn(eq10[5][:2], shift=mh.diff(eq1[5][5], eq10[4][8])),
                  FadeIn(eq10[4][5], target_position=eq1[5][1:5]),
                  run_time=1.5)
        self.wait(0.1)
        mh.align_sub(eq11, eq11[1][1:3], eq1[7][1:3]).align_to(eq1[7], RIGHT)
        eq11[0].shift(LEFT*0.5)
        self.play(mh.rtransform(eq1[7][:3], eq11[1][:3], eq1[7][3:5], eq11[1][5:7], eq1[7][5], eq11[1][8],
                                eq1[6][0], eq11[1][4]),
                  mh.fade_replace(eq1[6][0].copy(), eq11[0][0]),
                  FadeIn(eq11[1][3], shift=mh.diff(eq1[7][2], eq11[1][2])),
                  FadeIn(eq11[1][7], shift=mh.diff(eq1[7][4], eq11[1][6])),
                  run_time=1.2)
        self.wait(0.1)
        self.play(mh.rtransform(eq11[0][0], eq10[5][2], eq11[1][1:3], eq10[6][3:5], eq11[1][4:7], eq10[6][6:9]),
                  mh.fade_replace(eq11[1][0], eq10[6][2]),
                  mh.fade_replace(eq11[1][-1], eq10[6][-1]),
                  FadeOut(eq11[1][3:4] + eq11[1][7], shift=mh.diff(eq11[1][4:7], eq10[6][6:9])),
                  FadeIn(eq10[6][:2], shift=mh.diff(eq11[1][0], eq10[6][2])),
                  FadeIn(eq10[7], shift=mh.diff(eq11[1][-1], eq10[6][-1])*0.5),
                  mh.rtransform(eq1[-2:], eq10[-2:]),
                  FadeIn(eq10[6][5], shift=mh.diff(eq11[1][3], eq10[6][5])),
                  run_time=1.5)
        self.wait(0.1)
        eq12_1 = [eq12[1][1].copy() for _ in range(3)]

        eq12[1][2].set_opacity(-9)
        for _ in range(1, 9, 2): eq10[_][1].set_opacity(10)  # set high opacity to delay fadeout

        self.play(mh.fade_replace(eq10[1][0], eq12[1][1]),
                  mh.fade_replace(eq10[3][0], eq12_1[0]),
                  mh.fade_replace(eq10[5][0], eq12_1[1]),
                  mh.fade_replace(eq10[7][0], eq12_1[2]),
                  mh.rtransform(eq10[8], eq12[0], eq10[9][0], eq12[1][0]),
                  *[FadeOut(eq10[_][1], target_position=eq12[1][2]) for _ in range(1, 9, 2)],
                  eq12[1][2].animate.set_opacity(1),
                  eq10[3][2].animate.shift(mh.diff(eq10[8], eq12[0])),
                  run_time=1.8)
        self.remove(*eq12_1)
        eq13_1 = eq13[1][0].copy()
        self.play(mh.rtransform(eq12[0], eq13[0]),
                  mh.fade_replace(eq12[1][0], eq13[1][0]),
                  mh.fade_replace(eq12[1][2], eq13_1),
                  FadeOut(eq12[1][1], target_position=eq13[1]),
                  run_time=0.8)
        self.remove(eq13_1)
        self.play(mh.rtransform(eq13[0], eq14[0], eq13[1][0], eq14[1][0], eq10[0][0], eq14[1][2]),
                  mh.rtransform(eq10[2][0], eq14[1][2]), mh.rtransform(eq10[4][0], eq14[1][2]),
                  mh.rtransform(eq10[6][0], eq14[1][2]), FadeIn(eq14[1][1]),
                  run_time=1.5)
        eq15_1 = eq15[1][0].copy()
        self.play(mh.rtransform(eq14[0], eq15[0]),
                  mh.fade_replace(eq14[1][0], eq15[1][0]),
                  mh.fade_replace(eq14[1][2], eq15_1),
                  FadeOut(eq14[1][1], target_position=eq15[1][0]),
                  run_time=0.8)
        self.remove(eq15_1)
        self.wait(0.1)
        self.play(FadeIn(eq16), run_time=2)
        self.wait(0.1)
        self.play(mh.rtransform(eq10[0][1:3], eq17[1][:2], eq10[0][-1], eq17[1][-1],
                                eq10[1][2], eq17[2][0], eq10[2][1:3], eq17[3][:2], eq10[2][-1], eq17[3][-1],
                                eq10[3][2], eq17[4][0], eq10[4][1:3], eq17[5][:2], eq10[4][-1], eq17[5][-1],
                                eq10[5][2], eq17[6][0], eq10[6][1:3], eq17[7][:2], eq10[6][-1], eq17[7][-1],
                                eq15[:], eq17[-2:]),
                  mh.fade_replace(eq10[0][3:-1], eq17[1][2:-1]),
                  mh.fade_replace(eq10[2][3:-1], eq17[3][2:-1]),
                  mh.fade_replace(eq10[4][3:-1], eq17[5][2:-1]),
                  mh.fade_replace(eq10[6][3:-1], eq17[7][2:-1]),
                  MoveToTarget(eq16),
                  run_time=1.2)
        self.wait(0.1)
        self.play(FadeIn(eq17[0], eq17[-3], eq18[1][1:]),
                  mh.rtransform(eq17[-1][0], eq18[1][0]))
        self.wait(0.1)
        eq19_1 = [eq19[0][3:5]] + [eq19[0][3:5].copy() for _ in range(3)]
        self.play(eq17[0].animate.shift(mh.diff(eq17[1][0], eq19[0][0])).set_opacity(-1),
                  *[mh.rtransform(eq17[_][:3], eq19[0][:3], eq17[_][-1], eq19[0][-1]) for _ in range(1, 8, 2)],
                  *[mh.fade_replace(eq17[_*2+1][3:5], eq19_1[_]) for _ in range(4)],
                  *[FadeOut(eq17[_], target_position=eq19[0][2]) for _ in range(2, 7, 2)],
                  FadeOut(eq17[-3], shift=mh.diff(eq17[-4][-1], eq19[0][-1])),
                  mh.rtransform(eq17[-2], eq19[-2], eq18[1], eq19[-1]),
                  run_time=2)
        self.remove(*eq19_1[1:])
        self.wait(0.1)
        self.play(FadeIn(eq20))
        self.wait(0.1)
        rect = RoundedRectangle(width=eq19.width + 0.4, height=eq19.height + 2*SMALL_BUFF, fill_opacity=0,
                                stroke_opacity=1, stroke_color=RED, stroke_width=8,
                                corner_radius=0.2).set_z_index(4).move_to(eq19)
        self.play(FadeIn(rect))

        self.wait()

def create_filter(width=1.3, ngaps=10):
    fcol = BLACK
    fcoledge = WHITE
    buff = 0.1
    gap = 0.7
    p1 = RoundedRectangle(width=width, height=width, corner_radius=0.05, stroke_color=fcoledge, stroke_width=2)
    polys = [p1]
    pts = list(p1.points)
    dx = (width - 2 * buff) / (ngaps * gap + (ngaps - 1) * (1 - gap))
    dl, ul = DL * (width / 2 - buff), UL * (width / 2 - buff)
    p2 = Polygram(*[np.array([dl, ul, ul + RIGHT * dx * gap, dl + RIGHT * dx * gap])], fill_opacity=0.5, fill_color=WHITE,
                  stroke_opacity=0)
    for _ in range(ngaps):
        pts.extend(list(p2.points))
        polys.append(p2.copy())
        p2.shift(RIGHT * dx)
    polys.append(Polygram(fill_color=fcol, fill_opacity=1, stroke_opacity=0).set_points(pts))
    return VGroup(*polys)

class AliceBob(ThreeDScene):
    cube_rad = 0.8
    phot_h = cube_rad * 0.8  # starting photon distance
    center = IN * 2
    filt_h = 4  # filter distance
    photon_dist = 8  # distance beyond filters
    filt_time = 0.8  # time for photon to reach filter
    phi_photon = 15 * DEGREES
    phi_filter = 5 * DEGREES
    photon_speed = 5.

    def __init__(self, *args, **kwargs):
        if config.transparent:
            print("transparent!")
            config.background_color = WHITE

        # photon emission params
        fps = config.frame_rate
        nframes1 = int((self.filt_h - self.phot_h) * fps / self.photon_speed)
        self.speed = (self.filt_h - self.phot_h)/(nframes1 + 0.5)
        self.photon_time1 = nframes1 / fps + 0.001
        self.photon_dist1 = self.filt_h - self.phot_h - self.speed * 0.5
        photon_nframes2 = int((self.photon_dist - self.filt_h) / self.speed + 0.5)
        self.photon_dist2 = photon_nframes2 * self.speed
        self.photon_time2 = photon_nframes2 / fps + 0.001
        self.axes_photon = [DOWN * math.cos(self.phi_photon) + LEFT * math.sin(self.phi_photon),
                            DOWN * math.cos(self.phi_photon) + RIGHT * math.sin(self.phi_photon)]

        rad = 0.06
        p = []
        max_r = 7.
        f = lambda r: 1/r**2
        for r in np.linspace(1.0, max_r, 15)[::-1]:
            u = f(r)
            if u > 0.02:
                p.append(Dot3D(radius=rad*r, fill_opacity=u, stroke_opacity=0, fill_color=WHITE, z_index=1))
        p = VGroup(*p)
        arr = VGroup(Arrow3D(ORIGIN, OUT*0.7, color=BLUE, z_index=1), Arrow3D(ORIGIN, IN*0.7, color=BLUE, z_index=2))
        pA = VGroup(arr.set_z_index(1), p.set_z_index(2))
        self.photons = VGroup(pA, pA.copy())

        # filter params
        self.axes_filter = [DOWN * math.cos(self.phi_filter) + LEFT * math.sin(self.phi_filter),
                            DOWN * math.cos(self.phi_filter) + RIGHT * math.sin(self.phi_filter)]

        filt = create_filter(width=1.8).set_z_index(10).shift(self.center).rotate(90 * DEGREES, axis=RIGHT)
        filtA = filt.copy().shift(DOWN*self.filt_h).rotate(self.phi_filter, IN)
        filtB = filt.shift(UP*self.filt_h).rotate(self.phi_filter, OUT)
        self.filters = VGroup(filtA, filtB)
        self.filter_angle = (0., 0.)


        ThreeDScene.__init__(self, *args, *kwargs)

    def emit_photons(self, passthru=(True, True), angle=0., filter=True, op=(1,1), op2=(1,1)):
        pA, pB = self.photons.copy()
        pA.move_to(self.center + DOWN * self.phot_h)
        pB.move_to(self.center + UP * self.phot_h)
        if op[0] == 0:
            pA.set_opacity(0)
        if op[1] == 0:
            pB.set_opacity(0)
        if angle is None:
            pA[0].set_opacity(0)
            pB[0].set_opacity(0)
            angle = 0.
        else:
            if angle != 0.:
                pA.rotate(angle, self.axes_photon[0])
                pB.rotate(angle, self.axes_photon[1])

        self.filters.set_z_index(0)
        self.add(pA, pB)
        self.play(pA.animate(rate_func=linear).shift(DOWN * self.photon_dist1),
                  pB.animate(rate_func=linear).shift(UP * self.photon_dist1),
                  run_time=self.photon_time1)
        self.filters.set_z_index(50)
        pA.shift(DOWN * self.speed)
        pB.shift(UP * self.speed)
        pA[0].set_opacity(op2[0])
        pB[0].set_opacity(op2[1])
        if not passthru[0]:
            pA.set_opacity(0)
        if not passthru[1]:
            pB.set_opacity(0)
        if filter:
            angle2 = self.filter_angle
            if angle2[0] - angle != 0.:
                pA.rotate(angle2[0] - angle, axis=self.axes_photon[0])
            if angle2[1] - angle != 0.:
                pB.rotate(angle2[1] - angle, axis=self.axes_photon[1])
        else:
            angle2 = (angle, angle)

        self.play(pA.animate(rate_func=linear).shift(DOWN * self.photon_dist2),
                  pB.animate(rate_func=linear).shift(UP * self.photon_dist2),
                  run_time=self.photon_time2)
        if angle2[0] != 0.:
            pA.rotate(-angle2[0], self.axes_photon[0])
        if angle2[1] != 0.:
            pA.rotate(-angle2[1], self.axes_photon[1])
        self.remove(pA, pB)

    def set_filters(self, angle=(0., 0.), run_time=1.):
        t = ValueTracker(0.0)
        t0 = [0.0]
        thetaA = angle[0] - self.filter_angle[0]
        thetaB = angle[1] - self.filter_angle[1]

        def f(mob):
            t1 = t.get_value()
            dt = t1 - t0[0]
            if thetaA * dt != 0.:
                mob[0].rotate(thetaA * dt, axis=self.axes_filter[0])
            if thetaB * dt != 0.:
                mob[1].rotate(thetaB * dt, axis=self.axes_filter[1])
            t0[0] = t1
            return mob

        self.filters.add_updater(f)
        self.play(t.animate.set_value(1.), run_time=run_time)
        self.filters.remove_updater(f)
        self.filter_angle = angle

    def add_machine(self):
        self.camera.light_source_start_point = 4 * DOWN + 2 * LEFT - 10 * OUT
        self.camera.light_source = Point(self.camera.light_source_start_point)
        m1 = Cube(side_length=self.cube_rad * 2, fill_opacity=1, fill_color=GREY, stroke_opacity=1, stroke_color=WHITE,
                  stroke_width=1).set_z_index(103)
        m2 = Cylinder(radius=0.5, height=1.9, direction=UP, fill_opacity=1, fill_color=BLACK, stroke_opacity=0,
                      stroke_width=1, stroke_color=WHITE, checkerboard_colors=[BLACK],
                      resolution=(1, 32), show_ends=True).set_z_index(102)
        m3 = Cylinder(radius=0.49, height=2, direction=UP, fill_opacity=1, fill_color=BLACK,
                      stroke_opacity=0, checkerboard_colors=[GREY],
                      resolution=(1, 32), show_ends=True).set_z_index(101)
        Tex.set_default(color=DARK_BLUE, font_size=20, stroke_width=1.5)
        txt = VGroup(Tex(r'\b Entanglement'), Tex(r'\b Generator')).arrange(DOWN, buff=0.15)\
            .rotate(90*DEGREES, RIGHT).rotate(90*DEGREES, OUT).shift(RIGHT*self.cube_rad+OUT*self.cube_rad*0.05).set_z_index(105)

        machine = VGroup(m1, m2, m3, txt).move_to(self.center)
        self.add(machine)

        self.set_camera_orientation(phi=80*DEGREES, theta=0*DEGREES)

    def construct(self):
        skip = False
        self.add_machine()
        self.wait(0.5)
        if not skip:
            self.emit_photons(filter=False)
            self.emit_photons(angle=90 * DEGREES, filter=False)

        self.play(FadeIn(self.filters), run_time=0.5)
        self.wait(0.1)

        if not skip:
            self.set_filters((60 * DEGREES, 60 * DEGREES))
            self.wait(0.1)
            self.set_filters((-60 * DEGREES, -60 * DEGREES))
            self.wait(0.1)
            self.set_filters()
            self.wait(0.1)

        if not skip:
            self.emit_photons((True, True))
            self.emit_photons((False, False), angle=90 * DEGREES)

        if not skip:
            self.set_filters((60 * DEGREES, 60 * DEGREES))
            self.emit_photons((True, True), angle=None)

        if not skip:
            self.set_filters((-60 * DEGREES, -60 * DEGREES))
            self.emit_photons((False, False), angle=None)

        if not skip or True:
            self.set_filters((0 * DEGREES, 60 * DEGREES))
            self.emit_photons((False, True), angle=None)

        if not skip:
            self.set_filters((60 * DEGREES, -60 * DEGREES))
            self.emit_photons((True, False), angle=None)

        if not skip:
            self.set_filters((-60 * DEGREES, 0 * DEGREES))
            self.emit_photons((True, True), angle=None)

        self.set_filters((-60 * DEGREES, 0 * DEGREES))
        self.wait(0.1)
        self.set_filters((0 * DEGREES, 60 * DEGREES))
        self.emit_photons((True, False), angle=None)

        print('done!')

        self.wait(0.1)


class AliceBob1(AliceBob):
    def construct(self):
        self.add_machine()
        self.wait(0.5)
        self.emit_photons(angle=None, filter=False, op=(1,0), op2=(0,0))
        self.wait(0.1)
        self.emit_photons(angle=None, filter=False, op=(0,1), op2=(0,0))
        self.wait()

class AliceBob2(AliceBob):
    def construct(self):
        self.add_machine()
        self.add(self.filters)
        self.wait(0.5)
        self.emit_photons(angle=0.)
        self.wait(0.1)
        self.emit_photons(angle=PI/2, passthru=(False, False))
        self.wait()

class AliceBob3(AliceBob):
    def construct(self):
        self.add_machine()
        self.add(self.filters)
        self.wait(0.1)
        self.set_filters((0 * DEGREES, -60 * DEGREES))
        self.wait(0.1)
        self.emit_photons(angle=0.)
        self.wait(0.1)
        self.emit_photons(angle=PI/2, passthru=(False, True))
        self.wait(0.1)
        self.set_filters((0 * DEGREES, 0 * DEGREES))
        self.wait()

class PhotonMeasure(Scene):
    def __init__(self, *args, **kwargs):
        if not config.transparent:
            config.background_color = GREY
        Scene.__init__(self, *args, *kwargs)

    def construct(self):
        photon = Dot(radius=0.2).set_z_index(3)
        pcol = BLUE
        pcol2 = GREY
        len = 1.7
        ver = DoubleArrow(DOWN*len, UP*len, color=BLUE, buff=0).set_z_index(2)
        hor = DoubleArrow(LEFT*len, RIGHT*len, color=BLUE, buff=0).set_z_index(2)
        fs = 35

        theta = -50 * DEGREES
        tdir = (UP*math.cos(theta) + LEFT*math.sin(theta)) * len
        pthet = ver.copy().set_z_index(2)
        #pthet2 = pthet.copy().rotate(90 * DEGREES)
        arc = Arc(start_angle=90 * DEGREES, angle=theta, radius=len * 0.7).set_z_index(1)
        eq_thet = MathTex(r'\theta', z_index=1)[0].move_to((UP*math.cos(theta/2) + LEFT*math.sin(theta/2)) * len * 0.5)
        top = tdir
        dotted1 = DashedLine(top, top * RIGHT, color=GREY).set_z_index(1)
        dotted2 = DashedLine(top, top * UP, color=GREY).set_z_index(1)
        eq_cos = MathTex(r'\cos\theta', font_size=fs).set_z_index(1).rotate(PI/2).next_to(top*UP/2+UP*0.1, LEFT, buff=0.1)
        eq_sin = MathTex(r'\sin\theta', font_size=fs).set_z_index(1).next_to(top*RIGHT/2+RIGHT*0.1, DOWN, buff=0.1)

        eq_ver = MathTex(r'\Psi_V', font_size=fs).set_z_index(1).next_to(ver.get_end(), UR, buff=0)
        eq_hor = MathTex(r'\Psi_H', font_size=fs).set_z_index(1).next_to(hor, RIGHT, buff=0.1)
        eq_pthet = MathTex(r'\Psi_\theta=\cos\theta\Psi_V', font_size=fs).next_to(top, UP, buff=0.1).align_to(hor, LEFT).shift(LEFT*0.6).set_z_index(6)
        eq_cos2 = MathTex(r'\cos^2', font_size=fs).set_opacity(0).set_z_index(6)

        eq_pthet1 = MathTex(r'+\sin\theta\Psi_H', font_size=fs).set_z_index(1).next_to(eq_pthet, DOWN, buff=0.1).align_to(eq_pthet, LEFT).shift(RIGHT*0.1)
        eqs = VGroup(eq_ver, eq_hor, eq_pthet, eq_pthet1)

        eq1 = MathTex(r'\mathbb P({\sf polarization }=\theta)', r'=', r'\cos^2\theta', stroke_width=1.5, font_size=50).set_z_index(6)
        eq2 = MathTex(r'\mathbb P(A=B)', r'=', r'\cos^2\theta', stroke_width=2, font_size=60).set_z_index(6)
        eq3 = MathTex(r'=', r'(1+\cos2\theta)/2', stroke_width=2, font_size=60).set_z_index(6)

        col1 = GREEN
        col2 = ORANGE
        col3 = PURPLE
        (eq1[0][2:14]+eq1[0][0]+eq2[0][0]).set_color(YELLOW)
        VGroup(eq1[2][3], eq_cos2[0][-1], eq2[2][3], eq3[1][-1], eq3[1][1], eq3[1][6]).set_color(BLUE)
        VGroup(eq_thet, eq_cos[0][-1], eq_sin[0][-1], eq_ver[0][-1], eq_hor[0][-1], eq_pthet[0][-1], eq_pthet[0][1],
               eq_pthet[0][-3], eq_pthet1[0][-3], eq_pthet1[0][-1], eq1[2][4], eq1[0][-2],
               eq2[2][-1], eq3[1][-4]).set_color(col1)
        VGroup(eq_cos[0][:-1], eq_sin[0][:-1], eq_pthet[0][3:6],
               eq_pthet1[0][1:4], eq1[2][:3], eq_cos2[0][-1], eq2[2][:3], eq3[1][3:6]).set_color(col2)
        VGroup(eq_ver[0][0], eq_hor[0][0], eq_pthet[0][0], eq_pthet[0][-2], eq_pthet1[0][-2],).set_color(col3)
        eq2[0][2].set_color(PURE_RED)
        eq2[0][4].set_color(PURPLE)

        mob = VGroup(ver, hor, eqs, eq_ver)
        width = 2 * (photon.get_center() - mob.get_left())[0]
        height = 2 * (mob.get_top() - photon.get_center())[1]
        rect = RoundedRectangle(width=width + 0.2, height=height + 0.1, corner_radius=0.3, stroke_color=WHITE,
                                stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK, z_index=0)
        gp1 = VGroup(photon, ver, hor, arc, eq_thet, dotted1, dotted2, eq_cos, eq_sin, eqs, pthet)
        rect.move_to(gp1, coor_mask=UP)
        VGroup(rect, gp1).to_corner(UR, buff=0.1)
        gp2 = VGroup(photon, ver, hor, arc, eq_thet, dotted1, dotted2, pthet, eq_cos, eq_sin)
        shift2 = gp2.get_center()
        gp2.move_to(rect, coor_mask=UP)
        shift2 -= gp2.get_center()
        origin = photon.get_center()

        rect2 = SurroundingRectangle(eq1, corner_radius=0.15, stroke_width=0,
                                stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK).set_z_index(5)
        VGroup(rect2, eq1).next_to(rect, DOWN, buff=0.1).align_to(rect, RIGHT)

        rect3 = SurroundingRectangle(eq2, corner_radius=0.15, stroke_width=0,
                                stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK).set_z_index(5)
        VGroup(rect3, eq2).align_to(rect2, UR)
        mh.align_sub(eq3, eq3[0], eq2[1]).align_to(eq2, RIGHT)
        eq2.generate_target().next_to(eq3, UP, buff=0.3, coor_mask=UP)
        #eq3.next_to(eq2, DOWN, buff=0.3).align_to(eq2, RIGHT)
        rect4 = SurroundingRectangle(VGroup(eq2.target, eq3), corner_radius=0.15, stroke_width=0,
                                stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK).set_z_index(5)

        self.add(rect, photon)
        self.wait(0.5)
        self.play(FadeIn(ver), run_time=1)
        self.wait(0.5)
        self.play(FadeIn(hor.set_color(pcol2)), run_time=1)
        #self.play(ver.animate.set_color(pcol2), FadeIn(hor), run_time=1)
        #self.wait(0.5)
        #self.play(ver.animate.set_color(pcol), hor.animate.set_color(pcol2), run_time=0.5)
        #self.wait(0.5)
        ver.set_color(pcol2)
        self.add(pthet)
        self.play(LaggedStart(AnimationGroup(Rotate(pthet, angle=theta, about_point=origin),
                  Create(arc), run_time=1),
                  FadeIn(eq_thet, run_time=1), lag_ratio=0.5))
        self.wait(0.5)
        self.play(FadeIn(dotted1, dotted2, eq_cos, eq_sin), run_time=1)
        self.wait(0.5)
        #gp2 = VGroup(photon, hor, ver, arc, eq_thet, dotted1, dotted2, pthet)#, pthet2, eq_cos, eq_sin)
        self.play(FadeIn(eqs), gp2.animate.shift(shift2), run_time=1)
        self.wait(0.5)
        mh.align_sub(eq_cos2, eq_cos2[0][:-1], eq_pthet[0][3:6])
        self.play(FadeIn(eq1[:2], rect2),
                  mh.rtransform(eq_pthet[0][3:6].copy(), eq1[2][:3], eq_pthet[0][6].copy(), eq1[2][-1],
                                eq_cos2[0][-1].copy(), eq1[2][3]),
                  run_time=1.8)
        self.wait(0.5)
        pos2 = eq2[0][2:-1].get_center()
        eq2[0][2:-1].move_to(eq1[0][2:-1]).set_opacity(0)
        self.play(mh.rtransform(rect2, rect3, eq1[0][:2], eq2[0][:2], eq1[0][-1], eq2[0][-1], eq1[1:], eq2[1:]),
                  eq1[0][2:-1].animate.set_opacity(-1).move_to(pos2),
                  eq2[0][2:-1].animate.set_opacity(2).move_to(pos2),
                  run_time=1.6)
        self.wait(0.5)
        self.play(FadeOut(gp1, rect))
        self.play(LaggedStart(AnimationGroup(
            mh.rtransform(rect3, rect4), MoveToTarget(eq2), run_time=1),
            FadeIn(eq3, run_time=1), lag_ratio=0.5))
        self.wait()
