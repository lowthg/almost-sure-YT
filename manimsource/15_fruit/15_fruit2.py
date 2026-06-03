from manim import *
import numpy as np
import math
import sys

sys.path.append('../../')
import manimhelper as mh
from fruit_defs import *

class Substituteuvw0(Scene):
    trcolor = BLACK
    bgcolor = BLACK

    def __init__(self, *args, **kwargs):
        config.background_color = self.trcolor if config.transparent else self.bgcolor
        Scene.__init__(self, *args, **kwargs)

    boxargs = {'stroke_width': 0, 'stroke_opacity': 0, 'fill_color': BLACK, 'fill_opacity': 0.65,
               'corner_radius': 0.15, 'buff': 0.2}

    def construct(self):
        eq1 = Equation1.eq1().set_z_index(4)
        for i in (0,2,4):
            eq1[i][0].set_opacity(0)
            eq1[i][2].set_opacity(0)
            eq1[i][4].set_opacity(0)
        box1 = SurroundingRectangle(eq1, **self.boxargs)
        VGroup(box1, eq1).to_edge(DOWN, buff=0.1)
        self.add(eq1, box1)

class Substituteuvw(Substituteuvw0):
    trcolor = BLACK
    def construct(self):
        self.do_anim()

    def do_anim(self, just_eq=False):
        eq1 = Equation1.eq1()
        MathTex.set_default(font_size=65, stroke_width=2)
        eq2 = MathTex(r'x(x+z)(x+y)', r'+',
                      r'y(y+z)(x+y)', r'+',
                      r'z(y+z)(x+z)', r'=', r'4(y+z)(x+z)(x+y)')
        eq3 = MathTex(r'u=y+z', r' v=x+z', r' w=x+y')
        eq4 = MathTex(r'\frac xu', r'+', r'\frac yv', r'+', r'\frac zw', r'=4')
        eq5 = MathTex(r'u+v+w', r'=', r'2x + 2y + 2z')

        VGroup(eq3[0][0], eq3[1][2], eq3[2][2]).set_color(colx)
        VGroup(eq3[0][2], eq3[1][0], eq3[2][4]).set_color(coly)
        VGroup(eq3[0][4], eq3[1][4], eq3[2][0]).set_color(colz)
        VGroup(eq5[2][::3]).set_color(col_num)

        eq2[3:].next_to(eq2[0], DOWN, buff=0.4).align_to(eq2[0], LEFT)

        VGroup(eq1, eq2, eq3, eq4, eq5).set_z_index(4)

        eq2.next_to(eq1, UP, buff=-0.2)

        box1 = SurroundingRectangle(eq1, **self.boxargs)

        VGroup(box1, eq1, eq2).to_edge(DOWN, buff=0.1)

        eq3[1].shift(RIGHT*0.6)
        eq3[2].shift(RIGHT*1.2)
        eq3.next_to(eq1, UP, buff=0.5)
        mh.align_sub(eq4, eq4[5], eq1[5])
        for i in (0,2,4):
            mh.align_sub(eq4[i], eq4[i][0], eq1[i][0])
        for i in (1, 3, 5):
            eq4[i].move_to(eq1[i])
        # eq3_1 = eq3.copy()
        # eq3_1.arrange(DOWN, buff=0.4)
        # mh.align_sub(eq3_1[1], eq3_1[1][1], eq3_1[0][1], coor_mask=RIGHT)
        # mh.align_sub(eq3_1[2], eq3_1[2][1], eq3_1[0][1], coor_mask=RIGHT)
        # eq3_1.to_corner(UL, buff=0.4)

        if just_eq:
            for i, col in ((0,colx), (2,coly), (4,colz)):
                VGroup(eq4[i][0], eq4[i][-1], eq5[0][i], eq5[2][i//2*3+1]).set_color(col)
            eq4[-1][-1].set_color(col_num)
            return eq4, eq3, eq5

        # VGroup(*eq3_1[:]).arrange(DOWN, buff=0.4, center=False).to_corner(UL, buff=0.4)

        box2 = SurroundingRectangle(VGroup(eq1, eq2), **self.boxargs)

        mh.rtransform.copy_colors = True

        self.add(eq1, box1)
        self.wait(0.1)
        dt = 0.8

        anims1 = AnimationGroup(mh.rtransform(eq1[0][0], eq2[0][0], eq1[2][2:].copy(), eq2[0][2:5], eq1[4][2:].copy(), eq2[0][7:10],
                                run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq2[0][1], eq2[0][5:7], eq2[0][10])),
                  mh.rtransform(box1, box2)
                  )
        anims2 = AnimationGroup(mh.rtransform(eq1[1].copy(), eq2[1], eq1[2][0], eq2[2][0], eq1[0][2:].copy(), eq2[2][2:5],
                                eq1[4][2:].copy(), eq2[2][7:10], run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq2[2][1], eq2[2][5:7], eq2[2][10])),
                  )
        anims3 = AnimationGroup(mh.rtransform(eq1[3].copy(), eq2[3], eq1[4][0], eq2[4][0], eq1[0][2:].copy(), eq2[4][2:5],
                                eq1[2][2:].copy(), eq2[4][7:10], run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq2[4][1], eq2[4][5:7], eq2[4][10])),
                  )
        anims4 = AnimationGroup(mh.rtransform(eq1[5][0], eq2[5][0], eq1[5][1], eq2[6][0], run_time=1.5))
        anims5 = AnimationGroup(mh.rtransform(eq1[0][2:], eq2[6][2:5], eq1[2][2:], eq2[6][7:10], eq1[4][2:], eq2[6][12:15]),
                  FadeOut(eq1[0][1], eq1[1], eq1[2][1], eq1[3], eq1[4][1]),
                  Succession(Wait(0.8), FadeIn(eq2[6][1], eq2[6][5:7], eq2[6][10:12], eq2[6][15])),
                  )

        eq1_1 = eq1.copy()
        self.play(anims1, Succession(Wait(dt), anims2), Succession(Wait(dt*2), anims3),
                  Succession(Wait(dt*3), anims4), Succession(Wait(dt*3+1), anims5))
        eq1 = eq1_1
        self.wait(0.1)
        # self.play(mh.rtransform(eq2, eq2_1), run_time=0.8)
        # mh.copy_colors_eq(eq2, eq2_1)
        # self.play(eq2.animate(run_time=0.8).move_to(eq2_1))
        box3 = SurroundingRectangle(VGroup(eq1, eq3), **self.boxargs)
        self.play(mh.rtransform(eq2[6][2:5], eq1[0][2:], eq2[6][7:10], eq1[2][2:], eq2[6][12:15], eq1[4][2:]),
                  mh.rtransform(eq2[5][0], eq1[5][0], eq2[6][0], eq1[5][1]),
                  mh.rtransform(eq2[3], eq1[3], eq2[4][0], eq1[4][0], eq2[4][2:5], eq1[0][2:],
                  eq2[4][7:10], eq1[2][2:]),
                  mh.rtransform(eq2[1], eq1[1], eq2[2][0], eq1[2][0], eq2[2][2:5], eq1[0][2:],
                                eq2[2][7:10], eq1[4][2:]),
                  mh.rtransform(eq2[0][0], eq1[0][0], eq2[0][2:5], eq1[2][2:], eq2[0][7:10], eq1[4][2:]),
                  AnimationGroup(*[FadeOut(eq2[i][1], eq2[i][5:7], eq2[i][-1]) for i in (0,2,4,6)],
                  FadeOut(eq2[6][10:12]), run_time=0.6),
                  Succession(Wait(0.4), FadeIn(eq1[0][1], eq1[2][1], eq1[4][1])),
                  mh.rtransform(box2, box3)
                  )
        self.wait(0.1)
        self.play(FadeIn(eq3))
        self.wait(0.1)
        anims = [AnimationGroup(FadeOut(eq1[i][2:]),
                  mh.rtransform(eq1[i][:2], eq4[i][:2], eq3[j][0].copy(), eq4[i][2]))
                 for i,j in ((0,0), (2,1), (4,2))]
        self.play(anims[0], Succession(Wait(0.8), anims[1]), Succession(Wait(1.6), anims[2]),
                  *[mh.rtransform(eq1[i], eq4[i]) for i in (1, 3, 5)])
        self.wait(0.1)

        eq2_ = eq3.copy()
        box4 = RoundedRectangle(width=config.frame_width + 0.3, height=config.frame_height + 0.3,
                         stroke_width=0, stroke_opacity=0, fill_opacity=1, fill_color=BLACK,
                         corner_radius=0.15)
        self.play(AnimationGroup(mh.rtransform(eq2_[0][0], eq5[0][0], eq2_[1][0], eq5[0][2], eq2_[2][0], eq5[0][4]),
                  mh.rtransform(eq2_[0][2:4], eq5[2][4:6], eq2_[0][4], eq5[2][7]),
                  mh.rtransform(eq2_[1][2:4], eq5[2][1:3], eq2_[1][4], eq5[2][7]),
                  mh.rtransform(eq2_[2][2:4], eq5[2][1:3], eq2_[1][4], eq5[2][4]),
                  run_time=2.5),
                  Succession(Wait(1.5), FadeIn(eq5[0][1::2], eq5[1], eq5[2][::3])),
                  FadeIn(box4, run_time=2)
                  )

        # self.play(FadeIn(eq5), mh.rtransform(eq3, gp1[0]))
        # eq3 = gp1[0]
        # self.play(mh.rtransform(box3, box4, eq3, eq3_1),
        #           eq4.animate.shift(UP), run_time=2)



        self.wait()


class Substituteuvw2(Substituteuvw):
    bgcolor = BLACK

    def construct(self):
        self.do_anim()

    def do_anim(self, just_eq=False):
        eqs0 = Substituteuvw.do_anim(self, True)
        eq1 = MathTex(r'-u+v+w', r'=', r'2x+2y+2z', r'-2u')
        eq2 = MathTex(r'u-v+w', r'=', r'2x+2y+2z', r'-2v')
        eq3 = MathTex(r'u+v-w', r'=', r'2x+2y+2z', r'-2w')
        eq4 = MathTex(r'-u+v+w', r'=', r'2x+2(y+z-u)')
        eq5 = MathTex(r'u-v+w', r'=', r'2y+2(x+z-v)')
        eq6 = MathTex(r'u+v-w', r'=', r'2z+2(x+y-w)')
        eq7 = MathTex(r'2x=-u+v+w', r'2y=u-v+w', r'2z=u+v-w')
        eq9 = MathTex(r'\frac{-u+v+w}{u}', r'+', r'\frac{u-v+w}{v}', r'+',
                      r'\frac{u+v-w}{w}', r'=', r'8')
        eq10 = MathTex(r'-1+\frac{v+w}{u}', r'-', r'1+\frac{u+w}{v}', r'+',
                      r'\frac{u+v}{w}-1', r'=', r'8')
        eq11 = MathTex(r'\frac{v+w}{u}', r'+', r'\frac{u+w}{v}', r'+', r'\frac{u+v}{w}', r'=', r'11')
        eq12 = MathTex(r'vw(v+w)', r'+', r'uw(u+w)', r'+', r'uv(u+v)', r'=', r'11uvw')

        mh.rtransform.copy_colors = True
        VGroup(eq1[3][1], eq2[3][1], eq3[3][1], eq9[6],
               eq10[0][1], eq10[2][0], eq10[4][-1], eq11[-1]).set_color(col_num)
        VGroup(eq1[3][2], eq12[2][0], eq12[4][0]).set_color(colx)
        VGroup(eq2[3][2], eq12[0][0], eq12[4][1]).set_color(coly)
        VGroup(eq3[3][2], eq12[0][1], eq12[2][1]).set_color(colz)

        eq0shift = 1.5 * UP

        eq3_2 = eqs0[2]
        eq3_1 = eq3_2.copy().shift(UP)
        eq3_0 = eq3_1.copy().shift(UP)
        mh.align_sub(eq1, eq1[1], eq3_0[1])
        mh.align_sub(eq2, eq2[1], eq3_1[1])
        mh.align_sub(eq3, eq3[1], eq3_2[1])
        mh.align_sub(eq4, eq4[1], eq1[1])
        mh.align_sub(eq5, eq5[1], eq2[1])
        mh.align_sub(eq6, eq6[1], eq3[1])
        eq8 = eqs0[1].copy()
        mh.align_sub(eq8[1], eq8[1][1], eq8[0][1]).shift(DOWN)
        mh.align_sub(eq8[2], eq8[2][1], eq8[1][1]).shift(DOWN)
        eq8.to_corner(UL, buff=1)
        mh.align_sub(eq7[1], eq7[1][2], eq7[0][2])
        mh.align_sub(eq7[2], eq7[2][2], eq7[0][2])
        mh.align_sub(eq7[1][3:], eq7[1][3], eq7[0][4], coor_mask=RIGHT)
        mh.align_sub(eq7[2][3:], eq7[2][3], eq7[0][4], coor_mask=RIGHT)
        mh.align_sub(eq7[0], eq7[0][2], eq8[0][1], coor_mask=UP)
        mh.align_sub(eq7[1], eq7[1][2], eq8[1][1], coor_mask=UP)
        mh.align_sub(eq7[2], eq7[2][2], eq8[2][1], coor_mask=UP)
        eq7.to_edge(RIGHT, buff=1)
        mh.align_sub(eq9, eq9[5][0], eqs0[0][5][0], coor_mask=UP).shift(eq0shift)
        mh.align_sub(eq10, eq10[5], eq9[5], coor_mask=UP)
        mh.align_sub(eq11, eq11[5], eq10[5], coor_mask=UP)
        mh.align_sub(eq12, eq12[5], eq11[5], coor_mask=UP)

        if just_eq:
            VGroup(eq7[0][0], eq7[1][0], eq7[2][0], eq12[6][:2]).set_color(col_num)
            VGroup(eq7[0][1], eq7[0][4], eq7[1][3], eq7[2][3], eq12[2][3], eq12[4][3], eq12[6][2]).set_color(colx)
            VGroup(eq7[1][1], eq7[0][6], eq7[1][5], eq7[2][5], eq12[0][3], eq12[4][5], eq12[6][3]).set_color(coly)
            VGroup(eq7[2][1], eq7[0][8], eq7[1][7], eq7[2][7], eq12[0][5], eq12[2][5], eq12[6][4]).set_color(colz)
            return eq7, eq8, eq12

        self.add(*eqs0)
        self.play(mh.rtransform(eqs0[2].copy(), eq3_1, eqs0[2].copy(), eq3_0))
        anims = [
            AnimationGroup(mh.rtransform(eq3_0[1:], eq1[1:3], eq3_1[1:], eq2[1:3], eq3_2[1:], eq3[1:3],
                                eq3_0[0][:], eq1[0][1:]),
                  FadeIn(eq1[0][0], eq1[3])),
            AnimationGroup(mh.rtransform(eq3_1[0][0], eq2[0][0], eq3_1[0][2:], eq2[0][2:]),
                  mh.fade_replace(eq3_1[0][1], eq2[0][1]),
                  FadeIn(eq2[3])),
            AnimationGroup(mh.rtransform(eq3_2[0][:3], eq3[0][:3], eq3_2[0][4:], eq3[0][4:]),
                  mh.fade_replace(eq3_2[0][3], eq3[0][3]),
                  FadeIn(eq3[3]))]
        self.play(anims[0], Succession(Wait(0.8), anims[1]), Succession(Wait(1.6), anims[2]))

        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq1[2][:4], eq4[2][:4], eq1[2][4:6], eq4[2][5:7],
                                eq1[2][7], eq4[2][7], eq1[3][0], eq4[2][8], eq1[3][2], eq4[2][9]),
            FadeOut(eq1[3][1], target_position=eq4[2][3]),
            FadeOut(eq1[2][6], target_position=eq4[2][3]),
            mh.rtransform(eq2[2][3:6], eq5[2][:3], eq2[2][0], eq5[2][3], eq2[2][1:3], eq5[2][5:7],
                           eq2[2][7], eq5[2][7], eq2[3][0], eq5[2][8], eq2[3][2], eq5[2][9]),
            FadeOut(eq2[3][1], target_position=eq5[2][3]),
            FadeOut(eq2[2][6], target_position=eq5[2][3]),
            mh.rtransform(eq3[2][6:8], eq6[2][:2], eq3[2][3], eq6[2][3], eq3[2][2], eq6[2][2],
                          eq3[2][1], eq6[2][5], eq3[2][4], eq6[2][7], eq3[2][5], eq6[2][6],
                          eq3[3][0], eq6[2][8], eq3[3][2], eq6[2][9]),
            FadeOut(eq3[3][1], target_position=eq6[2][3]),
            FadeOut(eq3[2][0], target_position=eq6[2][3]),
            run_time=2),
            Succession(Wait(1.4), FadeIn(eq4[2][4], eq4[2][-1], eq5[2][4], eq5[2][-1], eq6[2][4], eq6[2][-1]))
                  )
        self.wait(0.1)

        line = Line(eq6[2][3].get_corner(DL), eq4[2].get_corner(UR), stroke_width=6,
                    stroke_color=RED).set_z_index(10)
        self.play(Create(line), run_time=0.8)
        self.wait(0.1)
        self.play(FadeOut(eq4[2][2:], eq5[2][2:], eq6[2][2:], line), run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(
            eq1[0][:], eq7[0][3:], eq2[0][:], eq7[1][3:], eq3[0][:], eq7[2][3:],
            eq1[1][0], eq7[0][2], eq2[1][0], eq7[1][2], eq3[1][0], eq7[2][2],
            eq4[2][:2], eq7[0][:2], eq5[2][:2], eq7[1][:2], eq6[2][:2], eq7[2][:2],
            eqs0[1], eq8),
            eqs0[0].animate.shift(eq0shift),
            run_time=2)
        self.wait(0.1)
        eq0 = eqs0[0]
        eq_ = MathTex(r'2x', r'2y', r'2z', color=col_num)
        mh.align_sub(eq_, eq_[0][1], eq0[0][0])
        eq_[0].move_to(eq9[0], coor_mask=RIGHT)
        eq_[1].move_to(eq9[2], coor_mask=RIGHT)
        eq_[2].move_to(eq9[4], coor_mask=RIGHT)
        self.play(mh.rtransform(eq0[0][-2:], eq9[0][-2:], eq0[2][-2:], eq9[2][-2:], eq0[4][-2:], eq9[4][-2:],
                                eq0[1:5:2], eq9[1:5:2], eq0[5][0], eq9[5][0],
                                eq0[0][0], eq_[0][1], eq0[2][0], eq_[1][1], eq0[4][0], eq_[2][1]),
                  mh.fade_replace(eq0[5][1], eq9[6]),
                  FadeIn(eq_[0][0], shift=mh.diff(eq0[0][0], eq_[0][1])),
                  FadeIn(eq_[1][0], shift=mh.diff(eq0[2][0], eq_[1][1])),
                  FadeIn(eq_[2][0], shift=mh.diff(eq0[4][0], eq_[2][1])),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq7[0][3:].copy(), eq9[0][:-2],
                  eq7[1][3:].copy(), eq9[2][:-2],
                  eq7[2][3:].copy(), eq9[4][:-2], run_time=2),
                  Succession(Wait(1), FadeOut(eq_)))
        self.wait(0.1)
        self.play(mh.rtransform(eq9[0][0], eq10[0][0], eq9[0][2:], eq10[0][2:]),
                  mh.fade_replace(eq9[0][1], eq10[0][1]),
                  Succession(Wait(0.8), AnimationGroup(
                      mh.rtransform(eq9[2][1], eq10[1][0], eq9[2][0], eq10[2][2],
                                    eq9[1][0], eq10[2][1], eq9[2][3:], eq10[2][3:]),
                      mh.fade_replace(eq9[2][2], eq10[2][0]))),
                  Succession(Wait(1.6), AnimationGroup(
                      mh.rtransform(eq9[3], eq10[3], eq9[4][:3], eq10[4][:3],
                                    eq9[4][3], eq10[4][-2], eq9[4][-2:], eq10[4][3:5],
                                    eq9[5:], eq10[5:]),
                      mh.fade_replace(eq9[4][4], eq10[4][-1])))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq10[0][3:], eq11[0][:], eq10[2][2:], eq11[2][:], eq10[4][:-2],
                                eq11[4][:], eq10[2][1], eq11[1][0], eq10[3], eq11[3], eq10[5], eq11[5]),
                  mh.fade_replace(eq10[6], eq11[6]),
                  FadeOut(eq10[0][:2], target_position=eq11[6]),
                  FadeOut(VGroup(eq10[1], eq10[2][0]), target_position=eq11[6]),
                  FadeOut(eq10[4][-2:], target_position=eq11[6]),
                  FadeOut(eq10[0][2])
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq11[0][:3], eq12[0][3:6], eq11[2][:3], eq12[2][3:6], eq11[4][:3], eq12[4][3:6],
                                eq11[1::2], eq12[1::2], eq11[-1][:2], eq12[-1][:2],
                                eq11[0][-1], eq12[6][2], eq11[2][-1], eq12[6][3], eq11[4][-1], eq12[6][4],
                                # eq11[0][-1].copy(), eq12[2][0], eq11[2][-1].copy(), eq12[0][0], eq11[4][-1].copy(), eq12[0][1],
                                # eq11[0][-1].copy(), eq12[4][0], eq11[2][-1].copy(), eq12[4][1], eq11[4][-1].copy(), eq12[2][1],
                                run_time=2),
                  FadeOut(eq11[0][3], eq11[2][3], eq11[4][3]),
                  Succession(Wait(1.4), FadeIn(eq12[0][2], eq12[0][-1], eq12[2][2], eq12[2][-1], eq12[4][2], eq12[4][-1])),
                  Succession(Wait(0.5), FadeIn(eq12[0][:2], eq12[2][:2], eq12[4][:2], run_time=1.5))
                  )
        self.wait(2.1)

class Substituteuvw3(Substituteuvw2):
    bgcolor = GREY
    def construct(self):
        eqs = self.do_anim(True)
        eqs = [eq_shadow(_, bg_stroke_width=12) for _ in eqs]
        self.add(*eqs)
        self.play(eqs[0].animate.to_corner(UR, buff=0.2),
                  eqs[1].animate.to_corner(UL, buff=0.2),
                  eqs[2].animate.to_edge(DOWN, buff=0.2),
                  run_time=2)
        self.wait()

class SimplePoints(Substituteuvw0):
    trcolor = GREY

    def construct(self):
        self.do_anim()

    def do_anim(self, eq_only=False):
        MathTex.set_default(font_size=65, stroke_width=2)
        eq1 = MathTex(r'vw(v+w)', r'+', r'uw(u+w)', r'+', r'uv(u+v)', r'=', r'11uvw')
        eq2 = MathTex(r'vw(v+w)',  r'=', r'0')
        eq3 = MathTex(r'u=0', font_size=75)
        eq4 = MathTex(r'v=0', font_size=75)
        eq5 = MathTex(r'(u:v:w)', r'=', r'(0:0:w)')
        eq6 = MathTex(r'(u:v:w)', r'=', r'(0:0:1)')
        eq7 = MathTex(r'(u:v:w)', r'=', r'(0:1:0)')
        eq8 = MathTex(r'(u:v:w)', r'=', r'(1:0:0)')
        eq9 = MathTex(r'P_1', r'=', r'(1:0:0)')
        eq10 = MathTex(r'P_2', r'=', r'(0:1:0)')
        eq11 = MathTex(r'P_3', r'=', r'(0:0:1)')
        eq12 = MathTex(r'w', r'=', r'-v', font_size=75)
        eq13 = MathTex(r'(u:v:w)', r'=', r'(0:v:-v)')
        eq14 = MathTex(r'(u:v:w)', r'=', r'(0:1:-1)')
        eq15 = MathTex(r'(u:v:w)', r'=', r'(1:0:-1)')
        eq16 = MathTex(r'(u:v:w)', r'=', r'(1:-1:0)')
        eq17 = MathTex(r'O_1', r'=', r'(0:1:-1)')
        eq18 = MathTex(r'O_2', r'=', r'(1:0:-1)')
        eq19 = MathTex(r'O_3', r'=', r'(1:-1:0)')

        mh.rtransform.copy_colors = True
        VGroup(eq1[6][:2], eq2[-1], eq3[0][2], eq5[2][1], eq5[2][3], eq6[2][5],
               eq13[2][1], eq14[2][3], eq14[2][6],
               eq9[2][1::2], eq10[2][1::2], eq11[2][1::2], eq17[2][1:5:2], eq17[2][-2],
               eq18[2][1:5:2], eq18[2][-2], eq19[2][1], eq19[2][4::2]
               ).set_color(col_num)
        VGroup(eq1[2][0], eq1[4][0], eq1[2][3], eq1[4][3], eq1[6][2],
               eq3[0][0], eq5[0][1]).set_color(colx)
        VGroup(eq1[0][0], eq1[4][1], eq1[0][3], eq1[4][5], eq1[6][3], eq5[0][3],
               eq13[2][3], eq13[2][6]).set_color(coly)
        VGroup(eq1[0][1], eq1[2][1], eq1[0][5], eq1[2][5], eq1[6][4], eq5[0][5], eq5[2][5]).set_color(colz)
        VGroup(eq9[0], eq10[0], eq11[0], eq17[0], eq18[0], eq19[0]).set_color(col_pt1)
        mh.copy_colors_eq(eq5[0], eq13[0])

        eq3.next_to(eq1, UP, buff=0.8)
        mh.align_sub(eq2, eq2[1], eq1[-2], coor_mask=UP)
        mh.align_sub(eq4, eq4[0][1], eq3[0][1]).next_to(eq3, RIGHT, buff=1, coor_mask=RIGHT)
        mh.align_sub(eq6, eq6[1], eq5[1])
        mh.align_sub(eq7, eq7[1], eq6[1]).shift(UP)
        mh.align_sub(eq8, eq8[1], eq7[1]).shift(UP)
        mh.align_sub(eq10, eq10[1], eq9[1]).shift(DOWN)
        mh.align_sub(eq11, eq11[1], eq10[1]).shift(DOWN)
        VGroup(eq9, eq10, eq11).to_corner(UL, buff=0.2).shift(DOWN*0.2)
        mh.align_sub(eq17, eq17[1], eq9[1])
        mh.align_sub(eq18, eq18[1], eq10[1])
        mh.align_sub(eq19, eq19[1], eq11[1])
        VGroup(eq17, eq18, eq19).to_edge(RIGHT, buff=0.2)

        eq1 = eq_shadow(eq1, bg_stroke_width=12)
        eq2 = eq_shadow(eq2, bg_stroke_width=12)
        eq3 = eq_shadow(eq3, bg_stroke_width=12)
        eq4 = eq_shadow(eq4, bg_stroke_width=12)
        eq5 = eq_shadow(eq5, bg_stroke_width=12)
        eq6 = eq_shadow(eq6, bg_stroke_width=12)
        eq7 = eq_shadow(eq7, bg_stroke_width=12)
        eq8 = eq_shadow(eq8, bg_stroke_width=12)
        eq9 = eq_shadow(eq9, bg_stroke_width=12)
        eq10 = eq_shadow(eq10, bg_stroke_width=12)
        eq11 = eq_shadow(eq11, bg_stroke_width=12)
        eq12 = eq_shadow(eq12, bg_stroke_width=12)
        eq13 = eq_shadow(eq13, bg_stroke_width=12)
        eq14 = eq_shadow(eq14, bg_stroke_width=12)
        eq15 = eq_shadow(eq15, bg_stroke_width=12)
        eq16 = eq_shadow(eq16, bg_stroke_width=12)
        eq17 = eq_shadow(eq17, bg_stroke_width=12)
        eq18 = eq_shadow(eq18, bg_stroke_width=12)
        eq19 = eq_shadow(eq19, bg_stroke_width=12)

        gp1 = VGroup(eq3.copy(), eq4).move_to(ORIGIN, coor_mask=RIGHT)
        gp2 = VGroup(eq8, eq7, eq6.copy()).move_to(ORIGIN, coor_mask=UP)

        if eq_only:
            return VGroup(eq9, eq10, eq11), VGroup(eq17, eq18, eq19)

        self.add(eq1)
        self.play(FadeIn(eq3))
        self.wait(0.1)
        lines = [
            Line(_.get_corner(DL), _.get_corner(UR), stroke_color=RED, stroke_width=8).set_z_index(8)
            for _ in (eq1[2], eq1[4], eq1[6])
        ]
        self.play(Create(VGroup(*lines), run_time=1.5, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(*lines, eq1[1:5], eq1[6]),
                  AnimationGroup(mh.rtransform(eq1[0], eq2[0], eq1[-2], eq2[-2]),
                                 FadeIn(eq2[-1], target_position=eq1[-1]),
                                 run_time=2)
                  )
        self.wait(0.1)
        eq2_1 = eq2.copy()
        self.play(FadeOut(eq2[0][1:]),
                  mh.rtransform(eq3, gp1[0], eq2[0][0], eq4[0][0],
                                                     eq2[1][0], eq4[0][1], eq2[2][0], eq4[0][2],
                                                     run_time=1.5)
                  )
        eq2 = eq2_1
        self.wait(0.1)
        self.play(FadeIn(eq5))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:2], eq6[:2], eq5[2][:-2], eq6[2][:-2], eq5[2][-1], eq6[2][-1]),
                  mh.fade_replace(eq5[2][-2], eq6[2][-2], coor_mask=RIGHT))
        self.wait(0.1)
        eq6_1 = eq6.copy()
        eq6_2 = eq6.copy()
        self.play(FadeOut(gp1[0], eq4), mh.rtransform(eq6, gp2[2]), eq6_1.animate.move_to(gp2[1], coor_mask=UP),
                  eq6_2.animate.move_to(gp2[0], coor_mask=UP))
        eq6 = gp2[2]
        self.play(mh.rtransform(eq6_2[:2], eq8[:2], eq6_2[2][::2], eq8[2][::2],
                                eq6_2[2][1], eq8[2][3], eq6_2[2][3], eq8[2][5], eq6_2[2][5], eq8[2][1],
                                run_time=1.5))
        self.play(
                  mh.rtransform(eq6_1[:2], eq7[:2], eq6_1[2][::2], eq7[2][::2],
                                eq6_1[2][1], eq7[2][1], eq6_1[2][3], eq7[2][5], eq6_1[2][5], eq7[2][3]),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq6[1:], eq11[1:], eq7[1:], eq10[1:], eq8[1:], eq9[1:]),
                  mh.fade_replace(eq6[0], eq11[0]),
                  mh.fade_replace(eq7[0], eq10[0]),
                  mh.fade_replace(eq8[0], eq9[0]),
                  run_time=2)
        self.wait(0.1)


        eq3.move_to(ORIGIN)
        eq2 = eq2_1.scale(75/65).next_to(eq3, DOWN)
        mh.align_sub(eq12, eq12[1], eq3[0][1]).next_to(eq3, RIGHT, buff=1, coor_mask=RIGHT)
        gp3 = VGroup(eq3.copy(), eq12).move_to(ORIGIN)
        mh.align_sub(eq13, eq13[1], eq2[1], coor_mask=UP).shift(DOWN*0.3)
        mh.align_sub(eq14, eq14[1], eq13[1])
        mh.align_sub(eq15, eq15[1], eq14[1]).shift(DOWN)
        mh.align_sub(eq16, eq16[1], eq15[1]).shift(DOWN)
        gp4 = VGroup(eq14.copy(), eq15, eq16).shift(UP)

        self.play(FadeIn(eq2, eq3))
        self.wait(0.1)
        self.play(FadeOut(eq2[0][:3], eq2[0][-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq3, gp3[0], eq2[0][3], eq12[2][1],
                                eq2[1], eq12[1], eq2[0][5], eq12[0][0]),
                  mh.fade_replace(eq2[0][4], eq12[2][0]),
                  FadeOut(eq2[2], target_position=eq12[2]))
        self.wait(0.1)
        self.play(FadeIn(eq13))
        self.wait(0.1)
        self.play(mh.rtransform(eq13[:2], eq14[:2], eq13[2][:3], eq14[2][:3],
                                eq13[2][4:6], eq14[2][4:6], eq13[2][-1], eq14[2][-1]),
                  mh.fade_replace(eq13[2][3], eq14[2][3], coor_mask=RIGHT),
                  mh.fade_replace(eq13[2][6], eq14[2][6], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        eq14_1 = eq14.copy()
        eq14_2 = eq14.copy()
        self.play(FadeOut(gp3[0], eq12),
                  mh.rtransform(eq14, gp4[0]),
                  eq14_1.animate.move_to(eq15, coor_mask=UP),
                  eq14_2.animate.move_to(eq16, coor_mask=UP),
                  )
        eq14 = gp4[0]
        self.wait(0.1)
        self.play(mh.rtransform(eq14_1[:2], eq15[:2], eq14_1[2][1], eq15[2][3],
                                eq14_1[2][3], eq15[2][1], eq14_1[2][2], eq15[2][2],
                                eq14_1[2][4:], eq15[2][4:], eq14_1[2][0], eq15[2][0],
                                eq14_1[2][4:6], eq15[2][4:6], eq14_1[2][-1], eq15[2][-1]))
        self.play(mh.rtransform(eq14_2[:2], eq16[:2], eq14_2[2][1], eq16[2][6],
                                eq14_2[2][3], eq16[2][1], eq14_2[2][5:7], eq16[2][3:5],
                                eq14_2[2][2], eq16[2][2], eq14_2[2][4], eq16[2][5],
                                eq14_2[2][0], eq16[2][0], eq14_2[2][-1], eq16[2][-1],
                                run_time=1.4)
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq14[1:], eq17[1:], eq15[1:], eq18[1:], eq16[1:], eq19[1:]),
                  mh.fade_replace(eq14[0], eq17[0]),
                  mh.fade_replace(eq15[0], eq18[0]),
                  mh.fade_replace(eq16[0], eq19[0]),
                  run_time=2)

        self.wait()

class SimplePoints2(SimplePoints):
    def construct(self):
        eqs1, eqs2 = self.do_anim(eq_only=True)

        eq3 = MathTex(r'(1,0,0)', r'-', r'(0,1,0)', r'=', r'(1,-1,0)')
        eq4 = MathTex(r'P_1', r'P_2', r'O_3')
        eq5 = MathTex(r'P_1', r'\circ', r'P_2', r'=', r'O_3', font_size=75)
        eq6 = MathTex(r'P_i', r'\circ', r'P_j', r'=', r'O_k', font_size=75)
        eq7 = MathTex(r'{\sf\ for\ }', r'\{i,j,k\}=\{1,2,3\}')
        eq8 = MathTex(r'P_i', r'\circ', r'O_k', r'=', r'P_j', font_size=75)
        eq9 = MathTex(r'P_i', r'\circ', r'O_j', r'=', r'P_k', font_size=75)
        eq10 = MathTex(r'(0,1,-1)', r'-', r'(1,0,-1)', r'=', r'(-1,1,0)')
        eq11 = MathTex(r'O_1', r'O_2', r'O_3')
        eq12 = MathTex(r'O_1', r'\circ', r'O_2', r'=', r'O_3', font_size=75)
        eq13 = MathTex(r'O_i', r'\circ', r'O_j', r'=', r'O_k', font_size=75)
        eq14 = MathTex(r'(u,v,w)', r'-', r'(u,w,v)', r'=', r'(0,v-w,w-v)')
        eq15 = MathTex(r'(0,1,-1)\;(v-w)')
        eq16 = MathTex(r'(u:v:w)', r'\circ', r'(u:w:v)', r'=', r'O_1')
        eq17 = MathTex(r'O_1', r'\circ', r'(u:v:w)', r'=', r'(u:w:v)')
        eq18 = MathTex(r'O_1', r'\circ', r'(0:1:-1)', r'=', r'(0:-1:1)')
        eq19 = MathTex(r'O_1', r'\circ', r'O_1', r'=', r'O_1')
        eq20 = MathTex(r'O_i', r'\circ', r'O_i', r'=', r'O_i')
        eq21 = MathTex(r'O_i', r'\circ', r'O_i', r'=', r'O_i', font_size=75)
        eq22 = MathTex(r'100', r'100', r'=')
        eq23 = MathTex(r'O_1', r'\circ', r'P_1', r'=', r'P_1')
        eq24 = MathTex(r'O_i', r'\circ', r'P_i', r'=', r'P_i', font_size=75)
        eq25 = MathTex(r'P_1', r'\circ', r'P_1', r'=', r'O_1')
        eq26 = MathTex(r'P_i', r'\circ', r'P_i', r'=', r'O_i', font_size=75)

        eq3.to_edge(DOWN, buff=0.5)
        eq4.next_to(eq3, UP, buff=0.3)
        eq4[0].move_to(eq3[0], coor_mask=RIGHT)
        eq4[1].move_to(eq3[2], coor_mask=RIGHT)
        eq4[2].move_to(eq3[4], coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[3], eq3[3], coor_mask=UP)
        mh.align_sub(eq6, eq6[3], eq5[3])
        eq7.next_to(eq6, RIGHT, buff=0.7)
        mh.align_sub(eq10, eq10[3], eq3[3], coor_mask=UP)
        eq11.move_to(eq4, coor_mask=UP)
        eq11[0].move_to(eq10[0], coor_mask=RIGHT)
        eq11[1].move_to(eq10[2], coor_mask=RIGHT)
        eq11[2].move_to(eq10[4], coor_mask=RIGHT)
        mh.align_sub(eq12, eq12[3], eq5[3], coor_mask=UP)
        mh.align_sub(eq13, eq13[3], eq12[3])
        mh.align_sub(eq14, eq14[3], eq3[3], coor_mask=UP)
        mh.align_sub(eq15, eq15[0][0], eq14[4][0])
        mh.align_sub(eq16, eq16[3], eq14[3])
        mh.align_sub(eq17, eq17[3], eq14[3]).to_edge(LEFT, buff=0.2)
        mh.align_sub(eq18, eq18[0], eq17[0]).shift(UP*0.8)
        mh.align_sub(eq19, eq19[0], eq18[0])
        mh.align_sub(eq20, eq20[3], eq19[3])
        mh.align_sub(eq22, eq22[2], eq17[3])
        for i in range(3):
            eq22[0][i].move_to(eq17[2][1+2*i])
            eq22[1][i].move_to(eq17[4][1+2*i])
        mh.align_sub(eq23, eq23[1], eq17[1])
        eq23_1 = eq23.copy().to_edge(DOWN, buff=0.1)
        eq23[2].move_to(eq17[2], coor_mask=RIGHT)
        mh.align_sub(eq23[3:], eq23[3], eq17[3])

        VGroup(eq10,eq11, eq12, eq13).to_corner(DL, buff=0.1)

        VGroup(eq4[2], eq6[0][1], eq6[2][1], eq6[4][1], eq7[1][1:7:2], eq7[1][9::2],
               eq9[::2], eq13[0][1], eq13[2][1], eq13[4][1], eq19[2], eq19[4],
               eq20[::2], eq23[2], eq23[4], eq24[::2], eq26[::2]).set_color(col_pt1)
        VGroup(eq5[1], eq12[1], eq16[1]).set_color(col_op)
        VGroup(eq7[0]).set_color(col_txt)
        VGroup(eq10[4][-2], eq14[4][1], eq15[0][3], eq15[0][6],
               eq18[2][1], eq18[2][3], eq18[2][6], eq18[4][1], eq18[4][4], eq18[4][6],
               eq22).set_color(col_num)
        VGroup(eq14[0][1]).set_color(colx)
        VGroup(eq14[0][3]).set_color(coly)
        VGroup(eq14[0][5]).set_color(colz)

        eq3 = eq_shadow(eq3, bg_stroke_width=12)
        eq4 = eq_shadow(eq4, bg_stroke_width=12)
        eq5 = eq_shadow(eq5, bg_stroke_width=12)
        eq6 = eq_shadow(eq6, bg_stroke_width=12)
        eq7 = eq_shadow(eq7, bg_stroke_width=12)
        eq8 = eq_shadow(eq8, bg_stroke_width=12)
        eq9 = eq_shadow(eq9, bg_stroke_width=12)
        eq10 = eq_shadow(eq10, bg_stroke_width=12)
        eq11 = eq_shadow(eq11, bg_stroke_width=12)
        eq12 = eq_shadow(eq12, bg_stroke_width=12)
        eq13 = eq_shadow(eq13, bg_stroke_width=13)
        eq14 = eq_shadow(eq14, bg_stroke_width=14)
        eq15 = eq_shadow(eq15, bg_stroke_width=14)
        eq16 = eq_shadow(eq16, bg_stroke_width=14)
        eq17 = eq_shadow(eq17, bg_stroke_width=14)
        eq18 = eq_shadow(eq18, bg_stroke_width=14)
        eq19 = eq_shadow(eq19, bg_stroke_width=14)
        eq20 = eq_shadow(eq20, bg_stroke_width=14)
        eq21 = eq_shadow(eq21, bg_stroke_width=14)
        eq22 = eq_shadow(eq22, bg_stroke_width=14)
        eq23 = eq_shadow(eq23, bg_stroke_width=14)
        eq23_1 = eq_shadow(eq23_1, bg_stroke_width=14)
        eq24 = eq_shadow(eq24, bg_stroke_width=12)
        eq25 = eq_shadow(eq25, bg_stroke_width=12)
        eq26 = eq_shadow(eq26, bg_stroke_width=12)

        self.add(eqs1, eqs2)

        # eq3_1 = eq3[:3].move_to(ORIGIN, coor_mask=RIG)
        circ = SurroundingRectangle(eqs1[:2], fill_opacity=0, stroke_opacity=1, stroke_color=RED,
                                    stroke_width=9, corner_radius=0.15, buff=0.1)
        self.play(FadeIn(circ))
        shift1 = eq3[0][2].get_center() - eqs1[0][2][2].get_bottom()
        shift2 = eq3[0][4].get_center() - eqs1[0][2][4].get_bottom()
        shift3 = eq3[2][2].get_center() - eqs1[1][2][2].get_bottom()
        shift4 = eq3[2][4].get_center() - eqs1[1][2][4].get_bottom()
        self.play(AnimationGroup(mh.rtransform(eqs1[0][2][1::2].copy(), eq3[0][1::2],
                                eqs1[0][2][0].copy(), eq3[0][0], eqs1[0][2][-1].copy(), eq3[0][-1],
                                eqs1[1][2][1::2].copy(), eq3[2][1::2],
                                eqs1[0][0].copy(), eq4[0],
                                eqs1[1][0].copy(), eq4[1],
                                eqs1[1][2][0].copy(), eq3[2][0], eqs1[1][2][-1].copy(), eq3[2][-1]),
                  FadeOut(eqs1[0][2][2].copy(), shift=shift1), FadeIn(eq3[0][2], shift=shift1),
                  FadeOut(eqs1[0][2][4].copy(), shift=shift2), FadeIn(eq3[0][4], shift=shift2),
                  FadeOut(eqs1[1][2][2].copy(), shift=shift3), FadeIn(eq3[2][2], shift=shift3),
                  FadeOut(eqs1[1][2][4].copy(), shift=shift4), FadeIn(eq3[2][4], shift=shift4),
                  FadeOut(circ, run_time=1.5),
                  run_time=2),
                  Succession(Wait(1.5), FadeIn(eq3[1])),
                  FadeOut(circ, run_time=1.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0][1].copy(), eq3[4][1], eq3[2][3].copy(), eq3[4][4],
                                eq3[2][5].copy(), eq3[4][6], eq3[1][0].copy(), eq3[4][3],
                                run_time=1.5),
                  Succession(Wait(1), FadeIn(eq3[3], eq3[4][:4:2], eq3[4][5::2])))
        circ = SurroundingRectangle(eqs2[2], fill_opacity=0, stroke_opacity=1, stroke_color=RED,
                                    stroke_width=9, corner_radius=0.15, buff=0.1)
        self.play(FadeIn(circ))
        self.play(mh.rtransform(eqs2[2][0].copy(), eq4[2], run_time=2), FadeOut(circ, run_time=1.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0], eq5[0], eq4[1], eq5[2], eq4[2], eq5[4],
                                eq3[3], eq5[3]),
                  mh.fade_replace(eq3[1], eq5[1]),
                  eq3[0].animate.set_opacity(-1).move_to(eq5[0]),
                  eq3[2].animate.set_opacity(-1).move_to(eq5[2]),
                  eq3[4].animate.set_opacity(-1).move_to(eq5[4]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(*[mh.rtransform(eq5[_][0], eq6[_][0]) for _ in range(5)],
                  *[mh.fade_replace(eq5[_][1], eq6[_][1]) for _ in range(0,5,2)])
        gp1 = VGroup(eq6.copy(), eq7).move_to(ORIGIN, coor_mask=RIGHT)
        self.play(Succession(Wait(0.5), FadeIn(eq7)), mh.rtransform(eq6, gp1[0]))
        eq6 = gp1[0]
        self.wait(0.1)
        eq6_1 = eq6.copy()
        self.play(eq6_1.animate.scale(0.6).next_to(eqs1, DOWN, buff=0.6).align_to(eqs1, LEFT))
        mh.align_sub(eq8, eq8[3], eq6[3])
        self.wait(0.1)
        self.play(mh.rtransform(eq6[:2], eq8[:2], eq6[2], eq8[4], eq6[3], eq8[3], eq6[4], eq8[2]))
        eq6 = eq6_1
        self.wait(0.1)
        eq9.scale(0.6).next_to(eq6, DOWN, buff=0.3).align_to(eq6, LEFT)
        self.play(FadeOut(eq7), mh.rtransform(eq8[0][0], eq9[0][0], eq8[1], eq9[1], eq8[2][0], eq9[2][0], eq8[3], eq9[3],
                                              eq8[4][0], eq9[4][0]),
                  mh.fade_replace(eq8[0][1], eq9[0][1]),
                  mh.fade_replace(eq8[2][1], eq9[2][1]),
                  mh.fade_replace(eq8[4][1], eq9[4][1]),
                  )
        self.wait(0.1)
        shift1 = eq10[0][2].get_center() - eqs2[0][2][2].get_bottom()
        shift2 = eq10[0][4].get_center() - eqs2[0][2][4].get_bottom()
        shift3 = eq10[2][2].get_center() - eqs2[1][2][2].get_bottom()
        shift4 = eq10[2][4].get_center() - eqs2[1][2][4].get_bottom()
        circ = SurroundingRectangle(eqs2[:2], fill_opacity=0, stroke_opacity=1, stroke_color=RED,
                                    stroke_width=9, corner_radius=0.15, buff=0.1)
        self.play(FadeIn(circ))
        self.play(AnimationGroup(mh.rtransform(eqs2[0][2][:2].copy(), eq10[0][:2], eqs2[0][2][3].copy(), eq10[0][3],
                                eqs2[0][2][5:].copy(), eq10[0][5:], eqs2[1][2][:2].copy(), eq10[2][:2],
                                eqs2[1][2][3].copy(), eq10[2][3], eqs2[1][2][5:].copy(), eq10[2][5:],
                                eqs2[0][0].copy(), eq11[0], eqs2[1][0].copy(), eq11[1]
                                ),
                  FadeOut(eqs2[0][2][2].copy(), shift=shift1), FadeIn(eq10[0][2], shift=shift1),
                  FadeOut(eqs2[0][2][4].copy(), shift=shift2), FadeIn(eq10[0][4], shift=shift2),
                  FadeOut(eqs2[1][2][2].copy(), shift=shift3), FadeIn(eq10[2][2], shift=shift3),
                  FadeOut(eqs2[1][2][4].copy(), shift=shift4), FadeIn(eq10[2][4], shift=shift4),
                  # FadeOut(circ, run_time=1.5),
                  # Succession(Wait(1.5), FadeIn(eq10[0][2::2], eq10[1], eq10[2][2::2])),
                  run_time=2),
                  Succession(Wait(1.5), FadeIn(eq10[1])),
                  FadeOut(circ, run_time=1.5))
        self.wait(0.1)
        eq10_1 = eq10[4][6].copy()
        self.play(AnimationGroup(mh.rtransform(eq10[1][0].copy(), eq10[4][1], eq10[2][1].copy(), eq10[4][2],
                                eq10[0][3].copy(), eq10[4][4],
                                # eq3[2][5].copy(), eq3[4][6], eq3[1][0].copy(), eq3[4][3],
                                ),
                  mh.fade_replace(eq10[0][5:7].copy(), eq10[4][6]),
                  mh.fade_replace(eq10[2][5:7].copy(), eq10_1),
                                 run_time=1.5),
                  Succession(Wait(1), FadeIn(eq10[3], eq10[4][0], eq10[4][3::2])))
        self.remove(eq10_1)
        self.wait(0.1)
        circ = SurroundingRectangle(eqs2[2], fill_opacity=0, stroke_opacity=1, stroke_color=RED,
                                    stroke_width=9, corner_radius=0.15, buff=0.1)
        self.play(FadeIn(circ))
        self.play(mh.rtransform(eqs2[2][0].copy(), eq11[2], run_time=2), FadeOut(circ, run_time=1.5))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[0], eq12[0], eq11[1], eq12[2], eq11[2], eq12[4],
                                eq10[3], eq12[3]),
                  mh.fade_replace(eq10[1], eq12[1]),
                  eq10[0].animate.set_opacity(-1).move_to(eq12[0]),
                  eq10[2].animate.set_opacity(-1).move_to(eq12[2]),
                  eq10[4].animate.set_opacity(-1).move_to(eq12[4]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(*[mh.rtransform(eq12[_][0], eq13[_][0]) for _ in range(5)],
                  *[mh.fade_replace(eq12[_][1], eq13[_][1]) for _ in range(0,5,2)])
        self.wait(0.1)
        self.play(eq13.animate.scale(0.6).next_to(eq9, DOWN, buff=0.3).align_to(eq6, LEFT))
        self.wait(0.1)
        self.play(FadeIn(eq14[0]))
        self.wait(0.1)
        eq14_1 = eq14[0].copy()
        self.play(eq14_1.animate.move_to(eq14[2]))
        self.play(mh.rtransform(eq14_1[:3], eq14[2][:3], eq14_1[4::2], eq14[2][4::2],
                                eq14_1[3], eq14[2][5], eq14_1[5], eq14[2][3]))
        self.wait(0.1)
        self.play(FadeIn(eq14[1]))
        self.wait(0.1)
        eq14_1 = eq14[0].copy()
        eq14_2 = eq14[2].copy()
        eq14_3 = eq14[4][1].copy()
        self.play(AnimationGroup(mh.rtransform(eq14_1[3], eq14[4][3], eq14_2[3], eq14[4][5],
                                eq14_1[5], eq14[4][7], eq14_2[5], eq14[4][9],
                                eq14[1][0].copy(), eq14[4][4], eq14[1][0].copy(), eq14[4][8]),
                  mh.fade_replace(eq14_1[1], eq14[4][1], coor_mask=RIGHT),
                  mh.fade_replace(eq14_2[1], eq14_3, coor_mask=RIGHT),
                  run_time=1.5),
                  Succession(Wait(1), FadeIn(eq14[3], eq14[4][:4:2], eq14[4][6::4])))
        self.remove(eq14_3)
        self.play(mh.rtransform(eq14[4][:3], eq15[0][:3], eq14[4][3:6], eq15[0][9:12],
                                eq14[4][6], eq15[0][4], eq14[4][10], eq15[0][7]),
                  mh.rtransform(eq14[4][7], eq15[0][11], eq14[4][8], eq15[0][10], eq14[4][9], eq15[0][9]),
                  FadeIn(eq15[0][3], target_position=eq14[4][4]),
                  FadeIn(eq15[0][5:7], target_position=eq14[4][8]),
                  Succession(Wait(0.5), FadeIn(eq15[0][8], eq15[0][-1]))
                  )
        eq16_1 = eq16[4].copy().next_to(eq15[0][:8], UP, buff=0.2)
        self.play(mh.rtransform(eqs2[0][0].copy(), eq16_1), run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq14[0][:2], eq16[0][:2], eq14[0][3], eq16[0][3], eq14[0][-2:], eq16[0][-2:],
                                eq14[2][:2], eq16[2][:2], eq14[2][3], eq16[2][3], eq14[2][-2:], eq16[2][-2:],
                                eq16_1, eq16[4], eq14[3], eq16[3]),
                  mh.fade_replace(eq14[0][2], eq16[0][2], coor_mask=RIGHT),
                  mh.fade_replace(eq14[0][4], eq16[0][4], coor_mask=RIGHT),
                  mh.fade_replace(eq14[2][2], eq16[2][2], coor_mask=RIGHT),
                  mh.fade_replace(eq14[2][4], eq16[2][4], coor_mask=RIGHT),
                  mh.fade_replace(eq14[1], eq16[1]),
                  FadeOut(eq15)
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq16[0], eq17[2], eq16[1], eq17[1], eq16[2], eq17[4], eq16[3], eq17[3],
                                eq16[4], eq17[0]), run_time=1.5)
        self.wait(0.1)
        eq17_ = eq17.copy()
        self.play(mh.rtransform(eq17_[:2], eq18[:2], eq17_[3], eq18[3], eq17_[2][:6:2], eq18[2][:6:2],
                                eq17_[2][-1], eq18[2][-1], eq17_[4][:4:2], eq18[4][:4:2], eq17_[4][-3::2], eq18[4][-3::2]),
                  mh.fade_replace(eq17_[2][1], eq18[2][1]),
                  mh.fade_replace(eq17_[2][3], eq18[2][3]),
                  mh.fade_replace(eq17_[2][5], eq18[2][5:7]),
                  mh.fade_replace(eq17_[4][1], eq18[4][1]),
                  mh.fade_replace(eq17_[4][3], eq18[4][3:5]),
                  mh.fade_replace(eq17_[4][5], eq18[4][6]),
                  )
        self.wait(0.1)
        eq19_1 = eq19.copy()
        eq19[3].move_to(eq18[3])
        eq19[2].move_to(eq18[2], coor_mask=RIGHT)
        eq19[4].align_to(eq18[4], LEFT)
        self.play(mh.rtransform(eq18[:2], eq19[:2], eq18[3], eq19[3]),
                  FadeOut(eq18[2::2]), FadeIn(eq19[2::2]), run_time=1)
        self.play(mh.rtransform(eq19, eq19_1))
        eq19 = eq19_1
        self.wait(0.1)
        self.play(*[mh.rtransform(eq19[_][0], eq20[_][0]) for _ in range(5)],
                  *[mh.fade_replace(eq19[_][1], eq20[_][1]) for _ in range(0,5,2)])
        self.wait(0.1)
        eq21.scale(0.6).next_to(eq13, DOWN, buff=0.3).align_to(eq6, LEFT)
        self.play(mh.rtransform(eq20, eq21))
        self.wait(0.1)
        self.play(FadeOut(eq17[2][1::2], eq17[4][1::2]),
                  FadeIn(eq22[:2]), run_time=1.5)
        self.wait(0.1)
        self.play(FadeOut(eq17[2][::2], eq17[4][::2], eq22[:2]),
                  FadeIn(eq23[2], eq23[4]),
                  mh.rtransform(eq17[:2], eq23[:2], eq17[3], eq23[3]),
                  run_time=1.3)
        self.wait(0.1)
        self.play(mh.rtransform(eq23, eq23_1))
        eq23 = eq23_1
        eq24.scale(0.6).next_to(eq21, DOWN, buff=0.3).align_to(eq6, LEFT)
        self.play(*[mh.rtransform(eq23[_][0].copy(), eq24[_][0]) for _ in range(5)],
                  *[mh.fade_replace(eq23[_][1].copy(), eq24[_][1]) for _ in range(0,5,2)])

        self.wait(0.1)
        mh.align_sub(eq25, eq25[3], eq23[3])
        self.play(mh.rtransform(eq23[0], eq25[4], eq23[4], eq25[0], eq23[1:4], eq25[1:4]))
        self.wait(0.1)
        eq26.scale(0.6).next_to(eq24, DOWN, buff=0.3).align_to(eq6, LEFT)
        self.play(*[mh.rtransform(eq25[_][0], eq26[_][0]) for _ in range(5)],
                  *[mh.fade_replace(eq25[_][1], eq26[_][1]) for _ in range(0,5,2)])
        self.wait(0.1)
        gp2 = VGroup(eq6, eq9, eq13, eq21, eq24, eq26)
        box1 = SurroundingRectangle(gp2, fill_opacity=0.7, fill_color=BLACK, stroke_color=RED, stroke_opacity=1,
                                    stroke_width=8, corner_radius=0.15, buff=0.12)
        gp3 = VGroup(gp2.copy(), box1.copy()).to_edge(UP, buff=0.2).to_edge(LEFT, buff=0.2)
        self.play(FadeOut(eqs1, eqs2), AnimationGroup(mh.rtransform(gp2, gp3[0]),
                                                      FadeIn(gp3[1], target_position=box1), run_time=1.5))


        self.wait()

class PointQLabel(Scene):
    def construct(self):
        eq = MathTex(r'Q=(10:3:2)', stroke_width=4, color=col_pt2, font_size=60)
        eq = eq_shadow(eq, bg_stroke_width=10)
        self.add(eq)

class PointQ(Substituteuvw3):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        eq1 = MathTex(r'(u:v:w)', r'=', r'(', r'10', r':', r'3', r':', r'2', r')')
        eq2 = MathTex(r'(x:y:z)', r'=', r'(', r'-', r'10', r'+', r'3', r'+', r'2', r':', r'10', r'-', r'3', r'+', r'2', r':', r'10', r'+', r'3', r'-', r'2', r')')
        eq3 = MathTex(r'(x:y:z)', r'=', r'(', r'-', r'5', r':', r'9', r':', r'11', r')')

        mh.rtransform.copy_colors=True
        VGroup(eq1[0][1]).set_color(colx)
        VGroup(eq1[0][3]).set_color(coly)
        VGroup(eq1[0][5]).set_color(colz)
        VGroup(eq1[3::2], eq2[4::2], eq3[4::2]).set_color(col_num)
        mh.copy_colors_eq(eq1[0], eq2[0])
        mh.align_sub(eq2, eq2[1], eq1, coor_mask=UP)

        self.add(eq1)
        self.play(eq1.animate.shift(UP),
                  Succession(Wait(0.4), FadeIn(eq2))
                  )
        self.wait(0.1)
        eq3_1 = eq3[3:5].copy().move_to(eq2[3:9], coor_mask=RIGHT)
        self.play(FadeOut(eq2[3:9]),
                  FadeIn(eq3_1))

        self.wait()