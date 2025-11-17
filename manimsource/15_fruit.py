from manim import *
import numpy as np
import math
import sys
import scipy as sp

sys.path.append('../')
import manimhelper as mh


class ClickBaitEq(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = WHITE
        Scene.__init__(self, *args, **kwargs)

    @staticmethod
    def eq1():
        str0 = r' {\vbox to 1em {\vfil\hbox to 1.18em{}\vfil} } '
        str1 = r'\frac{}{' + str0 + r'+' + str0 + r'}'
        eq = MathTex(str1, r'+', str1, r'+', str1, r'=4', font_size=65, color=BLACK, stroke_color=BLACK, stroke_width=2)
        eq.shift(LEFT*0.001*config.frame_width + UP*0.053739*config.frame_height)
        return eq

    def construct(self):
        self.add(self.eq1())

class ClickBaitEq2(Scene):
    @staticmethod
    def eq1():
        eq = ClickBaitEq.eq1().set_color(WHITE)
        eq[-1][-1].set_color(BLUE)
        return eq

    def construct(self):
        self.add(self.eq1())


colx = RED
coly = YELLOW
colz = GREEN
coln = BLUE

class ClickBaitEqx(Scene):
    eqstr = r'x'
    col = colx
    def construct(self):
        eq = MathTex(self.eqstr, font_size=65, color=self.col, stroke_color=self.col, stroke_width=2)
        self.add(eq)

class ClickBaitEqy(ClickBaitEqx):
    eqstr = r'y'
    col = coly

class ClickBaitEqz(ClickBaitEqx):
    eqstr = r'z'
    col = colz

class Equation1(Scene):
    @staticmethod
    def eq1():
        eq1 = MathTex(r'\frac{x}{y+z}', r'+', r'\frac{y}{x+z}', r'+', r'\frac{z}{x+y}', r'=4',
                      font_size=65, stroke_width=2)
        VGroup(eq1[0][0], eq1[2][2], eq1[4][2]).set_color(colx)
        VGroup(eq1[0][2], eq1[2][0], eq1[4][4]).set_color(coly)
        VGroup(eq1[0][4], eq1[2][4], eq1[4][0]).set_color(colz)
        eq1[-1][-1].set_color(coln)
        return eq1

    @staticmethod
    def eq2():
        eq1 = Equation1.eq1()
        eq2 = ClickBaitEq.eq1().set_color(WHITE)
        for i, eq in enumerate(eq1[:]):
            eq.move_to(eq2[i])
        for i in (0, 2, 4):
            mh.align_sub(eq1[i], eq1[i][1], eq2[i][0])
            mh.align_sub(eq1[i][2:], eq1[i][3], eq2[i][1])
            eq1[i][1].become(eq2[i][0])
            eq1[i][3].become(eq2[i][1])
            eq1[i][2].move_to(eq1[i][1].get_left()*0.55 + eq1[i][3].get_left()*0.45, coor_mask=RIGHT)
            eq1[i][4].move_to(eq1[i][1].get_right()*0.55 + eq1[i][3].get_right()*0.45, coor_mask=RIGHT)
            eq1[i][0].shift(UP*0.2)
        return eq1

    def construct(self):
        self.add(self.eq2())
        #self.play(mh.transform(eq1[0][1], eq2[0][0]))


class Rearrange1(Scene):
    def construct(self):
        eq1 = Equation1.eq2()
        MathTex.set_default(font_size=80, stroke_width=2)
        eq2 = MathTex(r'u', r'=', r'y+z')
        eq3 = MathTex(r'v', r'=', r'x+z')
        eq4 = MathTex(r'w', r'=', r'x+y')
        eq5 = MathTex(r'uvw', r'+')
        eq6 = MathTex(r'xvw', r'+', r'yuw', r'+', r'zuv', r'=4uvw', font_size=90)
        eq7 = MathTex(r'2x', r'=', r'-u+v+w')
        eq8 = MathTex(r'2y', r'=', r'u-v+w')
        eq9 = MathTex(r'2z', r'=', r'u+v-w')
        eq10 = MathTex(r'v+w', r'=', r'2x+y+z')
        eq11 = MathTex(r'-u+v+w', r'=', r'2x')
        eq12 = MathTex(r'\sum_{\sf perms}u^2v', r'=', r'11uvw')
        VGroup(eq2[0], eq3[2][0], eq4[2][0], eq5[0][0], eq6[0][0], eq6[2][1], eq6[4][1], eq6[5][2],
               eq7[0][1], eq7[2][1], eq8[2][0], eq9[2][0], eq10[2][1], eq11[0][1], eq11[2][1],
               eq12[0][6], eq12[2][2]).set_color(colx)
        VGroup(eq2[2][0], eq3[0], eq4[2][2], eq5[0][1], eq6[0][1], eq6[2][0], eq6[4][2], eq6[5][3],
               eq7[2][3], eq8[0][1], eq8[2][2], eq9[2][2], eq10[0][0], eq10[2][3], eq11[0][3],
               eq12[0][8], eq12[2][3]).set_color(coly)
        VGroup(eq2[2][2], eq3[2][2], eq4[0], eq5[0][2], eq6[0][2], eq6[2][2], eq6[4][0], eq6[5][4],
               eq7[2][5], eq8[2][4], eq9[0][1], eq9[2][4], eq10[0][2], eq10[2][5], eq11[0][5],
               eq12[2][4]).set_color(colz)
        VGroup(eq6[-1][1], eq7[0][0], eq8[0][0], eq9[0][0], eq10[2][0], eq11[2][0], eq12[2][:2]).set_color(BLUE)

        eq2.next_to(eq1, DOWN, buff=0.6, coor_mask=UP).shift(LEFT)
        eq3.next_to(eq2, DOWN, buff=0.4, coor_mask=UP)
        eq4.next_to(eq3, DOWN, buff=0.4, coor_mask=UP)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=RIGHT)
        mh.align_sub(eq4, eq4[1], eq2[1], coor_mask=RIGHT)
        eq5 = mh.align_sub(eq5, eq5[1], eq1[0][3])[0]
        eq5[0].move_to(eq1[0][1], coor_mask=RIGHT)
        eq5[1].move_to(eq1[2][1], coor_mask=RIGHT)
        eq5[2].move_to(eq1[4][1], coor_mask=RIGHT)
        mh.align_sub(eq6, eq6[-1][0], eq1[-1][0], coor_mask=UP)

        mh.align_sub(eq7, eq7[1], eq2[1]).next_to(eq2, RIGHT, coor_mask=RIGHT, buff=1.2)
        mh.align_sub(eq8, eq8[1], eq3[1])
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=RIGHT)
        mh.align_sub(eq9, eq9[1], eq4[1])
        mh.align_sub(eq9, eq9[1], eq7[1], coor_mask=RIGHT)
        eq8[2].align_to(eq7[2][1], LEFT)
        eq9[2].align_to(eq7[2][1], LEFT)
        mh.align_sub(eq10, eq10[1], eq7[1])
        mh.align_sub(eq11, eq11[1], eq7[1])
        eq12.next_to(eq6, UP)

        self.add(eq1, eq2, eq3, eq4)
        self.wait(0.1)
        self.play(FadeOut(eq1[0][2:], eq1[2][2:], eq1[4][2:]),
                  mh.rtransform(eq2[0][0].copy().set_z_index(2), eq5[0],
                                eq3[0][0].copy().set_z_index(2), eq5[1],
                                eq4[0][0].copy().set_z_index(2), eq5[2]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq5[0], eq6[2][1], eq5[0].copy(), eq6[4][1], eq5[0].copy(), eq6[5][2],
                                eq5[1], eq6[0][1], eq5[1].copy(), eq6[4][2], eq5[1].copy(), eq6[5][3],
                                eq5[2], eq6[0][2], eq5[2].copy(), eq6[2][2].copy(), eq5[2].copy(), eq6[5][4],
                                eq1[0][0], eq6[0][0], eq1[1], eq6[1], eq1[2][0], eq6[2][0], eq1[3], eq6[3], eq1[4][0], eq6[4][0],
                                eq1[5][:2], eq6[5][:2]),
                  FadeOut(eq1[0][1], eq1[2][1], eq1[4][1]),
                  run_time=1.6)
        self.wait(0.1)
        VGroup(eq2.generate_target(), eq3.generate_target(), eq4.generate_target(),
               eq7, eq8, eq9).move_to(ORIGIN, coor_mask=RIGHT)
        eq10.align_to(eq7, LEFT)
        eq11.align_to(eq10, LEFT)

        self.play(MoveToTarget(eq2), MoveToTarget(eq3), MoveToTarget(eq4), run_time=1.4)
        self.wait(0.1)
        eq3_1 = eq3.copy()
        eq4_1 = eq4.copy()
        self.play(mh.rtransform(eq3_1[0][0], eq10[0][0], eq4_1[0][0], eq10[0][2],
                                ),
                  FadeIn(eq10[0][1], target_position=VGroup(eq3_1[0], eq4_1[0])),
                  FadeIn(eq10[1], target_position=VGroup(eq3_1[1], eq4_1[1])),
                  run_time=1.6)
        self.play(mh.rtransform(eq4_1[2][:], eq10[2][1:4]),
                  mh.rtransform(eq3_1[2][0], eq10[2][1], eq3_1[2][1:], eq10[2][4:]),
                  FadeIn(eq10[2][0], shift=(mh.diff(eq4_1[2][0], eq10[2][1])+mh.diff(eq3_1[2][0], eq10[2][1]))/2),
                  run_time=1.6)
        self.wait(0.1)
        eq2_1 = eq2[0].copy().move_to(eq10[2][3:], coor_mask=RIGHT)
        self.play(FadeOut(eq10[2][3:]), FadeIn(eq2_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1[0], eq11[0][1], eq10[2][:2], eq11[2][:], eq10[1], eq11[1],
                                eq10[0][:], eq11[0][3:]),
                  mh.fade_replace(eq10[2][2], eq11[0][0]),
                  #FadeIn(eq11[0][2], shift=mh.diff(eq10[0][0], eq11[0][3])),
                  FadeIn(eq11[0][2], rate_func=rush_into),
                  run_time=1.8)
        self.wait(0.1)
        self.play(mh.rtransform(eq11[0], eq7[2], eq11[1], eq7[1], eq11[2], eq7[0]), FadeIn(eq8), FadeIn(eq9),
                  run_time=1.8)
        self.wait(0.1)
        self.play(FadeIn(eq12))

        self.wait()



if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        ClickBaitEq().render()