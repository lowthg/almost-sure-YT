from manim import *
import numpy as np
import math
import sys
import scipy as sp
from sympy.printing.pretty.pretty_symbology import line_width

sys.path.append('../../')
import manimhelper as mh
# from common.wigner import *

col_num = BLUE
col_x = (BLUE-WHITE)*0.7 + WHITE
col_a = (ORANGE-WHITE)*0.7 + WHITE
col_op = (PURPLE-WHITE) * 0.5 + WHITE
col_f = (RED-WHITE)*0.8 + WHITE
col_txt = ManimColor(r'#FFAC2B')
lcol = PURPLE * 0.5 + WHITE * 0.5


def eq_shadow(eq: VGroup, fg_z_index=4., bg_z_index=0., bg_color=BLACK, bg_stroke_width=10.):
    res = VGroup()
    for eq1 in eq:
        elem = VGroup()
        for eq2 in eq1:
            elem.add(VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res


class Narration1(Scene):
    bgcolor=GREY
    trcolor=GREY

    def __init__(self, *args, **kwargs):
        config.background_color = self.trcolor if config.transparent else self.bgcolor
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=65, stroke_width=1.5)
        eq1 = Tex(r'\sf left side is unchanged under scaling ', r'$(x,y,z)$')
        eq2 = Tex(r'\sf only depends on the ratio ', r'$(x:y:z)$')
        eq3 = Tex(r'\sf defines a curve on unit sphere ', r'$x^2+y^2+z^2=1$')
        eq2.next_to(eq1, DOWN)
        eq3.next_to(eq2, DOWN)
        VGroup(eq1[1][1], eq2[1][1], eq3[1][0]).set_color(colx)
        VGroup(eq1[1][3], eq2[1][3], eq3[1][3]).set_color(coly)
        VGroup(eq1[1][5], eq2[1][5], eq3[1][6]).set_color(colz)

        eq1 = eq_shadow(eq1, bg_stroke_width=10)
        eq2 = eq_shadow(eq2, bg_stroke_width=10)
        eq3 = eq_shadow(eq3, bg_stroke_width=10)
        self.add(eq1)
        self.wait(0.1)
        self.play(FadeIn(eq2))
        self.wait(0.1)
        self.play(FadeIn(eq3))
        self.wait()

class Narration2(Narration1):
    def construct(self):
        eq1 = Equation1.eq2()
        eq2 = MathTex(r'x(x+z)(x+y)', r'+',
                      r'y(y+z)(x+y)', r'+',
                      r'z(y+z)(x+z)', r'=', r'4(y+z)(x+z)(x+y)', font_size=65, stroke_width=2)
        eq2[3:].next_to(eq2[0], DOWN, buff=0.4).align_to(eq2[0], LEFT)
        eq3 = Tex(r'\sf a cubic curve', font_size=70, stroke_width=2)
        eq4 = Tex(r'\sf small (non-positive) solutions ', r'$P=(-1:1:1)$', r' and ', r'$Q=(-5:9:11)$', font_size=65, stroke_width=2)
        eq4[1:].next_to(eq4[0], DOWN, buff=0.3)
        eq4.move_to(ORIGIN)

        eq2.move_to(ORIGIN).to_edge(UP)
        eq3.to_edge(DOWN)
        eq4.to_edge(DOWN)

        eq2 = eq_shadow(eq2, bg_stroke_width=10)
        eq3 = eq_shadow(eq3, bg_stroke_width=10)
        eq4 = eq_shadow(eq4, bg_stroke_width=10)

        mh.rtransform.copy_colors = True

        self.add(eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][0], eq2[0][0], eq1[2][2:].copy(), eq2[0][2:5], eq1[4][2:].copy(), eq2[0][7:10],
                                run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq2[0][1], eq2[0][5:7], eq2[0][10])),
                  )
        self.play(mh.rtransform(eq1[1].copy(), eq2[1], eq1[2][0], eq2[2][0], eq1[0][2:].copy(), eq2[2][2:5],
                                eq1[4][2:].copy(), eq2[2][7:10], run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq2[2][1], eq2[2][5:7], eq2[2][10])),
                  )
        self.play(mh.rtransform(eq1[3].copy(), eq2[3], eq1[4][0], eq2[4][0], eq1[0][2:].copy(), eq2[4][2:5],
                                eq1[2][2:].copy(), eq2[4][7:10], run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq2[4][1], eq2[4][5:7], eq2[4][10])),
                  )
        self.play(mh.rtransform(eq1[5][0], eq2[5][0], eq1[5][1], eq2[6][0], run_time=1.5))
        self.play(mh.rtransform(eq1[0][2:], eq2[6][2:5], eq1[2][2:], eq2[6][7:10], eq1[4][2:], eq2[6][12:15]),
                  FadeOut(eq1[0][1], eq1[1], eq1[2][1], eq1[3], eq1[4][1]),
                  Succession(Wait(0.8), FadeIn(eq2[6][1], eq2[6][5:7], eq2[6][10:12], eq2[6][15])),
                  )
        self.play(FadeIn(eq3))
        self.wait(0.1)
        self.play(FadeOut(eq3))
        self.wait(0.1)
        self.play(FadeIn(eq4))
        self.wait()

class Narration3(Narration1):
    def construct(self):
        eq1 = Tex(r'shade region with positive $x,y,z$', font_size=65, stroke_width=2).set_z_index(2)
        eq1 = eq_shadow(eq1, bg_stroke_width=10)
        self.add(eq1)

class Narration4(Narration1):
    def construct(self):
        MathTex.set_default(font_size=65, stroke_width=2)
        eq1 = Tex(r'\sf a line intersecting the curve at 2 rational ratios\\', r'(counting multiplicity) ',
                  r'\\ intersects at a 3rd rational ratio')
        eq1 = eq_shadow(eq1, bg_stroke_width=10)
        # eq1[1:].next_to(eq1[0], DOWN)
        # eq1.move_to(ORIGIN)
        self.add(eq1)

class Narration5(Narration1):
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=2)
        eq = MathTex(r'R={}')
        eq = eq_shadow(eq, bg_stroke_width=15)
        self.add(eq)

class ClickBaitEq(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = WHITE
        Scene.__init__(self, *args, **kwargs)

    @staticmethod
    def eq1():
        str0 = r' {\vbox to 1em {\vfil\hbox to 1.18em{}\vfil} } '
        str1 = r'\frac{}{' + str0 + r'+' + str0 + r'}'
        eq = MathTex(str1, r'+', str1, r'+', str1, r'=4', font_size=65, color=BLACK, stroke_color=BLACK, stroke_width=2)
        eq.shift(LEFT*0.001*config.frame_width + UP*0.053739*config.frame_height).set_z_index(4)
        return eq

    def construct(self):
        self.add(self.eq1())

class ClickBaitEq2(Scene):
    tcol = WHITE
    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = self.tcol
        Scene.__init__(self, *args, **kwargs)

    @staticmethod
    def eq1():
        eq = ClickBaitEq.eq1().set_color(WHITE)
        eq[-1][-1].set_color(col_num)

        return eq

    def construct(self):
        self.add(self.eq1())


class ClickBaitEqScale(ClickBaitEq2):
    tcol = GREY

    def construct(self):
        eq1 = self.eq1()
        MathTex.set_default(stroke_width=2, font_size=65)
        eq1_1 = MathTex(r'y\ +\ z')[0]
        eq1_2 = MathTex(r'\frac xy')[0]
        mh.align_sub(eq1_1, eq1_1[1], eq1[0][1])
        mh.align_sub(eq1_2, eq1_2[1], eq1[0][0])
        eq1_3 = MathTex(r'x\ +\ z')[0]
        eq1_4 = MathTex(r'\frac yy')[0]
        mh.align_sub(eq1_3, eq1_3[1], eq1[2][1])
        mh.align_sub(eq1_4, eq1_4[1], eq1[2][0])
        eq1_5 = MathTex(r'x\ +\ y')[0]
        eq1_6 = MathTex(r'\frac zy')[0]
        mh.align_sub(eq1_5, eq1_5[1], eq1[4][1])
        mh.align_sub(eq1_6, eq1_6[1], eq1[4][0])
        VGroup(eq1_2, eq1_4, eq1_6).shift(UP*0.1)
        gp1 = VGroup(eq1_1[0], eq1_1[2], eq1_2[0], eq1_3[0], eq1_3[2], eq1_4[0], eq1_5[0], eq1_5[2], eq1_6[0])

        eq2 = MathTex(r'ax', r'ay', r'az')
        eq3 = MathTex(r'ay+az', r'ax+az', r'ax+ay')
        gp2 = VGroup(eq3[0][0], eq3[0][-2], eq3[1][0], eq3[1][-2], eq3[2][0], eq3[2][-2])
        eq4 = MathTex(r'a(y+z)', r'a(x+z)', r'a(x+y)')

        mh.rtransform.copy_colors = True
        # VGroup(gp1).set_color(col_x)
        VGroup(eq1_2[0], eq1_3[0], eq1_5[0]).set_color(colx)
        VGroup(eq1_1[0], eq1_4[0], eq1_5[2]).set_color(coly)
        VGroup(eq1_1[2], eq1_3[2], eq1_6[0]).set_color(colz)
        VGroup(eq2[0][0], eq2[1][0], eq2[2][0], gp2).set_color(col_a)

        mh.align_sub(eq2, eq2[0][1], gp1[2])
        eq2[0].move_to(gp1[2], coor_mask=RIGHT)
        eq2[1].move_to(gp1[5], coor_mask=RIGHT)
        eq2[2].move_to(gp1[8], coor_mask=RIGHT)
        mh.align_sub(eq3, eq3[1][1], gp1[3])
        eq3[0][:2].move_to(gp1[0], coor_mask=RIGHT)
        eq3[0][-2:].move_to(gp1[1], coor_mask=RIGHT)
        eq3[1][:2].move_to(gp1[3], coor_mask=RIGHT)
        eq3[1][-2:].move_to(gp1[4], coor_mask=RIGHT)
        eq3[2][:2].move_to(gp1[6], coor_mask=RIGHT)
        eq3[2][-2:].move_to(gp1[7], coor_mask=RIGHT)
        mh.align_sub(eq4, eq4[1][2], eq3[1][1])
        mh.align_sub(eq4[0][:3], eq4[0][2], eq3[0][1], coor_mask=RIGHT)
        mh.align_sub(eq4[1][:3], eq4[1][2], eq3[1][1], coor_mask=RIGHT)
        mh.align_sub(eq4[2][:3], eq4[2][2], eq3[2][1], coor_mask=RIGHT)
        eq4[0][-2:].move_to(eq3[0][-2:], coor_mask=RIGHT)
        eq4[1][-2:].move_to(eq3[1][-2:], coor_mask=RIGHT)
        eq4[2][-2:].move_to(eq3[2][-2:], coor_mask=RIGHT)

        self.add(eq1, gp1)
        self.wait(0.1)
        self.play(mh.rtransform(gp1[2], eq2[0][1], gp1[5], eq2[1][1], gp1[8], eq2[2][1]),
                  mh.rtransform(gp1[0], eq3[0][1], gp1[1], eq3[0][-1], gp1[3], eq3[1][1], gp1[4], eq3[1][-1],
                                gp1[6], eq3[2][1], gp1[7], eq3[2][-1]),
                  FadeIn(eq2[0][0], eq2[1][0], eq2[2][0]),
                  FadeIn(gp2)
                  )
        self.wait(0.1)
        tr1 = [mh.rtransform(eq3[i][0], eq4[i][0], eq3[i][1], eq4[i][2], eq3[i][-1], eq4[i][-2]) for i in range(3)]
        tr2 = [mh.rtransform(eq3[i][-2], eq4[i][0]) for i in range(3)]
        self.play(AnimationGroup(*tr1, *tr2, run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq4[0][1], eq4[0][-1], eq4[1][1], eq4[1][-1], eq4[2][1], eq4[2][-1]))
                  )
        self.wait(0.1)
        pts = [eq4[i][0].get_center() for i in range(3)] + [eq2[i][0].get_center() for i in range(3)]
        lines = [Line(pt + DL*0.2, pt+UR*0.2, color=RED, stroke_width=6, stroke_color=RED).set_z_index(5) for pt in pts]
        self.play(*[Create(line, run_time=0.5) for line in lines])
        self.wait(0.1)
        self.play(FadeOut(*lines, eq2[0][0], eq2[1][0], eq2[2][0], eq4[0][:2], eq4[1][:2], eq4[2][:2],
                          eq4[0][-1], eq4[1][-1], eq4[2][-1]))
        self.wait()

colx = RED
coly = YELLOW
colz = GREEN
coln = col_num

colx = col_x + 0.5 * (RED - col_x)
coly = col_x + 0.5 * (YELLOW - col_x)
colz = col_x + 0.5 * (GREEN - col_x)

def eq_shadow(eq: VGroup, fg_z_index=4., bg_z_index=0., bg_color=BLACK, bg_stroke_width=10.):
    res = VGroup()
    for eq1 in eq:
        elem = VGroup()
        for eq2 in eq1:
            elem.add(VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res

class EllipticExample(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        # y^2 = x^3 + 7
        eq1 = MathTex(r'y^2=x^3+7', font_size=60, stroke_width=2).set_z_index(7)
        VGroup(eq1[0][1], eq1[0][4], eq1[0][-1]).set_color(BLUE)
        VGroup(eq1[0][0], eq1[0][3]).set_color(col_op)
        eq1 = eq_shadow(eq1, bg_stroke_width=6, bg_z_index=6, fg_z_index=7)
        a = 7.
        xmax = 3.
        ymax = math.sqrt(xmax**3 + a)
        x0 = -math.pow(a, 1/3)
        ax = Axes(x_range=[x0-0.4, xmax], y_range=[-ymax, ymax], x_length=6, y_length=5,
                  axis_config={'color': BLACK, 'stroke_width': 4,
                               'include_ticks': False, 'include_tip': False})

        xvals = np.pow(np.linspace(0., math.pow(xmax-x0, 1/2), 100), 2) + x0
        yvals = np.sqrt((xvals**3 + 7).clip(0))
        xvals = np.append(xvals[::-1], xvals[1:])
        yvals = np.append(yvals[::-1], -yvals[1:])
        plt = ax.plot_line_graph(xvals, yvals, line_color=RED, add_vertex_dots=False, stroke_width=6)
        plt.set_z_index(5)

        eq1.move_to(ax.coords_to_point(-0.05, 0.85), aligned_edge=RIGHT)

        self.add(ax)
        self.wait(0.1)
        plt2 = plt['line_graph'].copy().set_z_index(4).set(stroke_color=BLACK, stroke_width=10)
        self.play(Create(plt, rate_func=linear, run_time=2),
                  Create(plt2, rate_func=linear, run_time=2),
                  # Succession(Wait(1.), FadeIn(eq1))
                  )
        self.wait()

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
        eq1 = eq_shadow(eq1, bg_stroke_width=10, bg_color=BLACK)
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
        self.play(FadeOut(eq10[2][3 :]), FadeIn(eq2_1))
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

class Ratios(ClickBaitEqScale):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=75)
        eq1 = MathTex(r'(x:y:z)')

        eq2 = MathTex(r'(x:y:z)', r'=', r'(ax:ay:az)')
        eq3 = MathTex(r'(1:2:5)', r'=', r'(2\cdot1:2\cdot2:2\cdot5)')
        eq4 = MathTex(r'(1:2:5)', r'=', r'(2:4:10)')
        eq5 = MathTex(r'(', r'{\scriptstyle\frac13}', r' : ', r'{\scriptstyle\frac34}', r' : ', r'{\scriptstyle\frac16}', r')',
                      r'=',
                      r'(12\!\cdot\!', r'{\scriptstyle\frac13}', r' : 12\!\cdot\!', r'{\scriptstyle\frac34}', r' : 12\!\cdot\!', r'{\scriptstyle\frac16}', r')')
        eq6 = MathTex(r'=', r'(4:9:2)')
        eq7 = MathTex(r'(', r'{\scriptstyle\frac13}', r' : ', r'{\scriptstyle\frac34}', r' : ', r'{\scriptstyle\frac16}', r')',
                      r'=',
                      r'(4:9:2)')

        gp1 = VGroup(eq5[1], eq5[3], eq5[5], eq5[9], eq5[11], eq5[13]).move_to(eq5[0], coor_mask=UP)
        for i in [1, 3, 5, 9, 11, 13]: mh.font_size_sub(eq5, i, 110)
        gp2 = VGroup(eq7[1], eq7[3], eq7[5]).move_to(eq7[0], coor_mask=UP)
        for i in [1, 3, 5]: mh.font_size_sub(eq7, i, 110)

        mh.rtransform.copy_colors = True
        # VGroup(eq1[0][1], eq1[0][3], eq1[0][5]).set_color(col_x)
        eq1[0][1].set_color(colx)
        eq1[0][3].set_color(coly)
        eq1[0][5].set_color(colz)
        VGroup(eq2[2][1], eq2[2][4], eq2[2][7], eq3[2][1], eq3[2][5], eq3[2][9],
               eq5[8][1:3], eq5[10][1:3], eq5[12][1:3]).set_color(col_a)
        VGroup(eq3[0][1], eq3[0][3], eq3[0][5], eq3[2][3], eq3[2][7], eq3[2][11],
               eq4[2][1], eq4[2][3], eq4[2][5:7], gp1, eq6[1][1], eq6[1][3], eq6[1][5]).set_color(col_num)
        VGroup(eq3[2][2], eq3[2][6], eq3[2][10], *[eq[1] for eq in gp1[:]],
               eq5[8][3], eq5[10][3], eq5[12][3]).set_color(col_op)

        mh.align_sub(eq2, eq2[0], eq1, coor_mask=UP)
        # mh.align_sub(eq3, eq3[1], eq2[1]).next_to(eq2, DOWN, coor_mask=UP)
        eq3.next_to(eq2, DOWN, buff=0.4)
        gp1 = VGroup(eq2.copy(), eq3).move_to(ORIGIN, coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq3[1])
        eq4[2][0].move_to(eq3[2][0], coor_mask=RIGHT)
        eq4[2][1].move_to(eq3[2][1:4], coor_mask=RIGHT)
        eq4[2][2].move_to(eq3[2][4], coor_mask=RIGHT)
        eq4[2][3].move_to(eq3[2][5:8], coor_mask=RIGHT)
        eq4[2][4].move_to(eq3[2][8], coor_mask=RIGHT)
        eq4[2][5:7].move_to(eq3[2][9:12], coor_mask=RIGHT)
        eq4[2][7].move_to(eq3[2][12], coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[7], eq4[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[0], eq4[1])
        eq6[1][0].move_to(eq5[8][0], coor_mask=RIGHT)
        eq6[1][1].move_to(eq5[8][1:] + eq5[9], coor_mask=RIGHT)
        eq6[1][2].move_to(eq5[10][0], coor_mask=RIGHT)
        eq6[1][3].move_to(eq5[10][1:] + eq5[11], coor_mask=RIGHT)
        eq6[1][4].move_to(eq5[12][0], coor_mask=RIGHT)
        eq6[1][5].move_to(eq5[12][1:] + eq5[13], coor_mask=RIGHT)
        eq6[1][6].move_to(eq5[14][0], coor_mask=RIGHT)
        mh.align_sub(eq7, eq7[7], eq4[1], coor_mask=UP)

        self.add(eq1)
        self.wait(0.1)
        eq1_ = eq1[0].copy().set_opacity(0.3)
        eq1[0].set_z_index(5)
        self.play(mh.rtransform(eq1[0], eq2[0],
                                eq1_[0].copy(), eq2[2][0], eq1_[1:3].copy(), eq2[2][2:4],
                                eq1_[3:5].copy(), eq2[2][5:7], eq1_[5:].copy(), eq2[2][8:],
                                run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq2[1])),
                  Succession(Wait(1.0), FadeIn(eq2[2][1], eq2[2][4], eq2[2][7])),
                  )
        self.wait(0.1)
        eq2_ = eq2.copy()
        self.play(mh.rtransform(eq2, gp1[0]),
                  mh.rtransform(eq2_[0][0], eq3[0][0], eq2_[0][2], eq3[0][2], eq2_[0][4], eq3[0][4],
                                eq2_[0][6], eq3[0][6], eq2_[1], eq3[1], eq2_[2][0], eq3[2][0], eq2_[2][3], eq3[2][4],
                                eq2_[2][6], eq3[2][8], eq2_[2][9], eq3[2][12]),
                  mh.fade_replace(eq2_[0][1], eq3[0][1]),
                  mh.fade_replace(eq2_[0][3], eq3[0][3]),
                  mh.fade_replace(eq2_[0][5], eq3[0][5]),
                  mh.fade_replace(eq2_[2][1], eq3[2][1]),
                  mh.fade_replace(eq2_[2][2], eq3[2][3]),
                  FadeIn(eq3[2][2], target_position=eq2_[2][1:4]),
                  mh.fade_replace(eq2_[2][4], eq3[2][5]),
                  mh.fade_replace(eq2_[2][5], eq3[2][7]),
                  FadeIn(eq3[2][6], target_position=eq2_[2][4:6]),
                  mh.fade_replace(eq2_[2][7], eq3[2][9]),
                  mh.fade_replace(eq2_[2][8], eq3[2][11]),
                  FadeIn(eq3[2][10], target_position=eq2_[2][7:9]),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2][0], eq4[2][0], eq3[2][4], eq4[2][2],
                                eq3[2][8], eq4[2][4], eq3[2][12], eq4[2][7]),
                  FadeOut(eq3[2][1], target_position=eq4[2][1]),
                  FadeOut(eq3[2][2]),
                  FadeOut(eq3[2][3], target_position=eq4[2][1]),
                  FadeIn(eq4[2][1]),
                  FadeOut(eq3[2][5], target_position=eq4[2][3]),
                  FadeOut(eq3[2][6]),
                  FadeOut(eq3[2][7], target_position=eq4[2][3]),
                  FadeIn(eq4[2][3]),
                  FadeOut(eq3[2][9], target_position=eq4[2][5:7]),
                  FadeOut(eq3[2][10]),
                  FadeOut(eq3[2][11], target_position=eq4[2][5:7]),
                  FadeIn(eq4[2][5:7]),
                  )
        self.wait(0.1)
        self.play(FadeOut(eq4))
        self.wait(0.1)
        eq5_1 = eq5[:7].copy().move_to(ORIGIN, coor_mask=RIGHT)
        eq5_ = eq5_1.copy().set_opacity(0.3)
        self.play(FadeIn(eq5_1.set_z_index(5)))
        self.wait(0.1)
        self.play(mh.rtransform(eq5_1, eq5[:7],
                                eq5_[0][0], eq5[8][0], eq5_[1], eq5[9], eq5_[2][0], eq5[10][0],
                                eq5_[3], eq5[11], eq5_[4][0], eq5[12][0], eq5_[5:], eq5[13:],
                                run_time=1.7),
                  Succession(Wait(0.7), FadeIn(eq5[7], shift=LEFT*0.4)),
                  Succession(Wait(1.5), FadeIn(eq5[8][1:], eq5[10][1:], eq5[12][1:]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq5[8][0], eq6[1][0], eq5[10][0], eq6[1][2], eq5[12][0], eq6[1][4], eq5[14][0], eq6[1][6]),
                  FadeOut(eq5[8][1:3], target_position=eq6[1][1]),
                  FadeOut(eq5[8][3]),
                  FadeOut(eq5[9], target_position=eq6[1][1]),
                  FadeIn(eq6[1][1]),
                  FadeOut(eq5[10][1:3], target_position=eq6[1][3]),
                  FadeOut(eq5[10][3]),
                  FadeOut(eq5[11], target_position=eq6[1][3]),
                  FadeIn(eq6[1][3]),
                  FadeOut(eq5[12][1:3], target_position=eq6[1][5]),
                  FadeOut(eq5[12][3]),
                  FadeOut(eq5[13], target_position=eq6[1][5]),
                  FadeIn(eq6[1][5]),
                  )
        self.play(mh.rtransform(eq5[:8], eq7[:8], eq6[1], eq7[8]))
        self.wait()


class ZEqualOne(Scene):
    tcol = GREY
    bgcol = GREY

    def __init__(self, *args, **kwargs):
        config.background_color = self.tcol if config.transparent else self.bgcol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=2)
        eq1 = MathTex(r'\frac{x}{y+z}', r'+', r'\frac{y}{x+z}', r'+', r'\frac{z}{x+y}', r'=', r'4')
        eq2 = MathTex(r'{}+', r'1')
        eq3 = MathTex(r'z', r'=', r'1', font_size=60)

        # VGroup(*[eq1[i][j] for i in (0,2,4) for j in (0,2,4)], eq3[0]).set_color(col_x)
        VGroup(eq1[0][0], eq1[2][2], eq1[4][2]).set_color(colx)
        VGroup(eq1[0][2], eq1[2][0], eq1[4][4]).set_color(coly)
        VGroup(eq1[0][4], eq1[2][4], eq1[4][0], eq3[0]).set_color(colz)
        VGroup(eq1[-1], eq2[1], eq3[2]).set_color(col_num)

        eq1 = eq_shadow(eq1, bg_stroke_width=12)
        eq2 = eq_shadow(eq2, bg_stroke_width=12)
        eq3 = eq_shadow(eq3, bg_stroke_width=12)

        mh.align_sub(eq2, eq2[0], eq1[0][3])
        eq2_1 = mh.align_sub(eq2.copy(), eq2[0], eq1[2][3])
        eq2[1].move_to(eq1[0][4], coor_mask=RIGHT)
        eq2_1[1].move_to(eq1[2][4], coor_mask=RIGHT)
        eq2_2 = eq2_1[1].copy().shift(mh.diff(eq1[2][4], eq1[4][0]))
        eq3.next_to(eq1, UP, buff=1.3).align_to(eq1, LEFT).shift(RIGHT*0.5)

        self.add(eq1)
        self.wait(0.1)
        self.play(FadeIn(eq3))
        self.wait(0.1)
        self.play(FadeOut(eq1[0][4], eq1[2][4], eq1[4][0]),
                  mh.rtransform(eq3[2], eq2[1], eq3[2].copy(), eq2_1[1], eq3[2].copy(), eq2_2),
                  FadeOut(eq3[:2]))
        self.wait(1)

class Asymptotes(ZEqualOne):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=65)
        eq1 = Tex(r'\sf asymptotes', r'($z=0$)')
        mh.font_size_sub(eq1, 1, 50)
        eq1[1].next_to(eq1[0], DOWN, buff=0.2)
        eq1[0].set_color(ORANGE)
        eq1[1][1].set_color(colz)
        eq1[1][-2].set_color(col_num)
        eq1 = eq_shadow(eq1, bg_stroke_width=10)
        self.add(eq1)

class Antipodal(ZEqualOne):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        eq1 = MathTex(r'(x:y:z)', r'=', r'(-x:-y:-z)')
        VGroup(eq1[0][1], eq1[2][2]).set_color(colx)
        VGroup(eq1[0][3], eq1[2][5]).set_color(coly)
        VGroup(eq1[0][5], eq1[2][8]).set_color(colz)
        VGroup(eq1[2][1], eq1[2][4], eq1[2][7]).set_color(col_op)
        eq1 = eq_shadow(eq1, bg_stroke_width=12)
        self.add(eq1)

class UnitSphere(ZEqualOne):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        eq1 = MathTex(r'x^2+y^2+z^2', r'=', r'r^2')
        # VGroup(eq1[0][0], eq1[0][3]).set_color(col_x)
        eq1[0][0].set_color(colx)
        eq1[0][3].set_color(coly)
        eq1[0][6].set_color(colz)
        VGroup(eq1[0][1], eq1[0][4], eq1[0][7], eq1[2][1]).set_color(col_num)
        eq1[2][0].set_color(col_a)

        eq1 = eq_shadow(eq1, bg_stroke_width=12)
        self.add(eq1)

class CubicRearrange(Narration1):
    trcolor = BLACK
    def construct(self):
        eq1 = Equation1.eq1()
        eq2 = MathTex(r'x(x+z)(x+y)', r'+',
                      r'y(y+z)(x+y)', r'+',
                      r'z(y+z)(x+z)', r'=', r'4(y+z)(x+z)(x+y)', font_size=65, stroke_width=2)
        eq2[3:].next_to(eq2[0], DOWN, buff=0.4).align_to(eq2[0], LEFT)
        eq3 = MathTex(r'x^3', r'+', r'x^2y^1', r'+', r'x^2z^1', r'+', r'x^1y^1z^1', r'+', r'\cdots',
                      font_size=65, stroke_width=2)

        VGroup(eq1, eq2, eq3).set_z_index(4)

        VGroup(eq3[0][1], eq3[2][1], eq3[2][3], eq3[4][1], eq3[4][3],
               eq3[6][1], eq3[6][3], eq3[6][5]).set_color(col_num)

        eq2.next_to(eq1, UP, buff=-0.2)

        boxargs = {'stroke_width': 0, 'stroke_opacity': 0, 'fill_color': BLACK, 'fill_opacity': 0.65,
                                   'corner_radius': 0.15, 'buff': 0.2}

        box1 = SurroundingRectangle(eq1, **boxargs)

        VGroup(box1, eq1, eq2).to_edge(DOWN, buff=0.1)

        eq2_1 = eq2.copy().align_to(eq1, DOWN)
        eq3.next_to(eq2_1, UP, buff=0.4).align_to(eq2, LEFT)

        box2 = SurroundingRectangle(VGroup(eq1, eq2), **boxargs)

        mh.rtransform.copy_colors = True

        self.add(eq1, box1)
        self.wait(0.1)
        dt = 1.4

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
        self.play(anims1, Succession(Wait(dt), anims2), Succession(Wait(dt*2), anims3),
                  Succession(Wait(dt*3), anims4), Succession(Wait(dt*3+1), anims5))
        # self.play(mh.rtransform(eq2, eq2_1), run_time=0.8)
        mh.copy_colors_eq(eq2, eq2_1)
        self.play(eq2.animate(run_time=0.8).move_to(eq2_1))
        anims1 = AnimationGroup(AnimationGroup(mh.rtransform(eq2[0][0].copy(), eq3[0][0]),
                  mh.rtransform(eq2[0][2].copy(), eq3[0][0]),
                  mh.rtransform(eq2[0][7].copy(), eq3[0][0]),
                                 run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq3[0][1])),
                  Succession(Wait(1.1), FadeIn(eq3[1])),
                  )
        anims2 = AnimationGroup(AnimationGroup(mh.rtransform(eq2_1[0][0].copy(), eq3[2][0]),
                  mh.rtransform(eq2_1[0][2].copy(), eq3[2][0],
                                eq2_1[0][9].copy(), eq3[2][2]),
                                 run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq3[2][1])),
                  Succession(Wait(1.1), FadeIn(eq3[3])),
                  )
        anims3 = AnimationGroup(AnimationGroup(mh.rtransform(eq2_1[0][0].copy(), eq3[4][0]),
                  mh.rtransform(eq2_1[0][4].copy(), eq3[4][2],
                                eq2_1[0][7].copy(), eq3[4][0]),
                                 run_time=1.5),
                  Succession(Wait(0.8), FadeIn(eq3[4][1])),
                  Succession(Wait(1.1), FadeIn(eq3[5])),
                  )
        anims4 = AnimationGroup(AnimationGroup(mh.rtransform(eq2_1[0][0].copy(), eq3[6][0]),
                  mh.rtransform(eq2_1[0][4].copy(), eq3[6][4],
                                eq2_1[0][9].copy(), eq3[6][2]),
                                 run_time=1.5),
                  Succession(Wait(1), FadeIn(eq3[7:])),
                  )
        dt = 1.2
        self.play(anims1, Succession(Wait(dt), anims2), Succession(Wait(dt*2), anims3),
                  Succession(Wait(dt*3), anims4))
        circ1 = mh.circle_eq(eq3[0][1]).scale(0.7).shift(DOWN*0.05).set_z_index(5)
        circ2 = mh.circle_eq(VGroup(eq3[2][1], eq3[2][3])).scale(0.7).shift(DOWN*0.05+RIGHT*0.1).set_z_index(5)
        circ3 = mh.circle_eq(VGroup(eq3[4][1], eq3[4][3])).scale(0.7).shift(DOWN*0.05+RIGHT*0.1).set_z_index(5)
        circ4 = mh.circle_eq(VGroup(eq3[6][1], eq3[6][3], eq3[6][5])).scale(0.7).shift(DOWN*0.05+RIGHT*0.1).set_z_index(5)
        anims1 = AnimationGroup(Create(circ1, rate_func=linear, run_time=0.5))
        anims2 = AnimationGroup(FadeIn(eq3[2][3]), Succession(Wait(0.5), Create(circ2, run_time=0.5)))
        anims3 = AnimationGroup(FadeIn(eq3[4][3]), Succession(Wait(0.5), Create(circ3, run_time=0.5)))
        anims4 = AnimationGroup(FadeIn(eq3[6][1], eq3[6][3], eq3[6][5]), Succession(Wait(0.5), Create(circ4, run_time=0.5)))
        dt = 0.5
        self.play(anims1, anims2, Succession(Wait(dt), anims3),
                  Succession(Wait(dt*2), anims4))
        self.wait(0.1)
        box3 = SurroundingRectangle(eq2, **boxargs)
        self.play(FadeOut(eq3, circ1, circ2, circ3, circ4),
                  Succession(Wait(0.5), mh.rtransform(box2, box3)))
        self.wait()

col_pt1 = YELLOW
col_pt2 = ORANGE
col_pt3 = PINK
col_t = WHITE + 0.8 * (GREEN - WHITE)

class Chord(Scene):
    def construct(self):
        def f(x):
            return x*(x*x-1)

        """
        y = x^3 - x
        t*y1 + (1-t) * y2 = (t*x_1+(1-t)*y2)^3 - (t*x_1+(1-t)*y2)
        (s*y1+t*y2)(s+t)^2 = (s*x1+t*x2)^3 - (s*x1+t*x2)(s+t)^2
        st^2 (y1 + 2y2) + s^2t (y2 + 2y1) = s^2t (3 x1^2 x2 - 2x1 - x2) + st^2 (3x2^2 x1 - 2x2 - x1)
        t(y1+2y2 - 3x2^2 x1 + 2x2 + x1) + ..
        """

        xmax = 1.3
        xmin = -xmax
        ymax = f(xmax)
        ymin = f(xmin)

        ax = Axes(x_range=[xmin, xmax], y_range=[ymin, ymax], x_length=8, y_length=27/4, tips=False,
                  axis_config={'color': GREY, 'stroke_width': 4, 'include_ticks': False,
                               },
                  ).set_z_index(1)

        box = SurroundingRectangle(ax, fill_color=BLACK, fill_opacity=0, stroke_opacity=1, stroke_color=WHITE,
                                   stroke_width=5, buff=0).set_z_index(10)

        crv = ax.plot(f, [xmin, xmax], stroke_width=6, stroke_color=BLUE).set_z_index(4)

        self.add(box)
        self.wait(0.1)
        self.play(Create(crv, rate_func=smooth, run_time=2))
        self.wait(0.1)

        x1val = ValueTracker(-1.1)
        x2val = ValueTracker(0.)
        shiftp = ValueTracker(0.2)
        shiftq = ValueTracker(-0.2)
        shiftr = ValueTracker(0.2)
        MathTex.set_default(stroke_width=1.5)
        eqp = MathTex(r'P', color=col_pt1).set_opacity(0)
        eqq = MathTex(r'Q', color=col_pt1).set_opacity(0)
        eqr = MathTex(r'R', color=col_pt2).set_opacity(0)

        def get_obj():
            x1 = x1val.get_value()
            x2 = x2val.get_value()
            rdot = 0.15
            y1 = f(x1)
            y2 = f(x2)
            a = (y2-y1)/(x2-x1)
            b = y1 - a*x1
            s = y1 + 2*y2 - 3*x2*x2*x1 + 2*x2 +x1
            t = y2 + 2*y1 - 3*x1*x1*x2 + 2*x1 +x2
            t = -t
            x3 = (s*x1 + t*x2) / (s+t)
            y3 = (s*y1 + t*y2) / (s+t)
            pt1 = ax.coords_to_point(x1, y1)
            pt2 = ax.coords_to_point(x2, y2)
            pt3 = ax.coords_to_point(x3, y3)
            dot1 = Dot(pt1, radius=rdot, fill_color=col_pt1).set_z_index(6)
            dot2 = Dot(pt2, radius=rdot, fill_color=col_pt1).set_z_index(6)
            dot3 = Dot(pt3, radius=rdot, fill_color=col_pt2).set_z_index(6)
            x_l = xmin
            x_r = xmax
            y_l = a*x_l+b
            y_r = a*x_r+b
            if y_l > ymax:
                x_l = (ymax - b) / a
                y_l = ymax
            if y_r < ymin:
                x_r = (ymin - b) / a
                y_r = ymin


            pt_l = ax.coords_to_point(x_l, y_l)
            pt_r = ax.coords_to_point(x_r, y_r)
            line = Line(pt_l, pt_r, stroke_width=8, stroke_color=lcol).set_z_index(5)
            return VGroup(dot1, dot2, dot3, line,
                          eqp.next_to(dot2, DOWN, buff=0.06).shift(RIGHT*shiftp.get_value()),
                          eqq.next_to(dot1, DOWN, buff=0.06).shift(RIGHT*shiftq.get_value()),
                          eqr.next_to(dot3, DOWN, buff=0.06).shift(RIGHT*shiftr.get_value())
            )

        chord = always_redraw(get_obj)

        self.play(FadeIn(chord[:2]))
        self.play(Create(chord[3], rate_func=linear),
                  Succession(Wait(0.5), FadeIn(chord[2])))
        self.wait(0.1)
        self.add(chord)
        self.play(x2val.animate(run_time=3).set_value(1.2))
        self.play(x2val.animate(run_time=3).set_value(-1.1),
                  x1val.animate(run_time=3).set_value(0.2))
        # self.play(x1val.animate(run_time=1.5).set_value(0.2))
        self.wait(0.1)

        self.play(eqp.animate.set_opacity(1), eqq.animate.set_opacity(1))
        self.wait(0.1)
        circ = mh.circle_eq(eqr).scale(0.7).set_z_index(5).shift(RIGHT*0.1 + DOWN*0.05)
        self.play(eqr.animate.set_opacity(1), Succession(Wait(0.8), Create(circ, run_time=0.5, rate_func=linear)))
        self.wait(0.1)
        self.play(FadeOut(circ))
        self.wait(0.1)
        eps = 0.001
        self.play(x2val.animate.set_value(-0.52),
                  x1val.animate.set_value(-0.52 + eps),
                  shiftp.animate.set_value(-0.27),
                  shiftq.animate.set_value(0.13),
                  shiftr.animate.set_value(0.08),
                  run_time=2
                  )
        self.wait(0.1)
        self.play(x2val.animate.set_value(-0.42),
            x1val.animate.set_value(0.1),
                  shiftq.animate.set_value(-0.2),
                  shiftr.animate.set_value(-0.2),
                  run_time=1.5)
        self.wait(0.1)
        self.play(x1val.animate.set_value(-0.42 + eps),
                  shiftq.animate.set_value(0.13),
                  shiftr.animate.set_value(0.08),
                  run_time=6,
                  rate_func=rate_functions.ease_out_quad)
        self.wait()

class ChordMath(Narration1):
    bgcolor = BLACK

    def construct(self):
        MathTex.set_default(stroke_width=1.5, font_size=65)
        eq1 = MathTex(r'P', r'=', r'(x_1,y_1,z_1)')
        eq2 = MathTex(r'Q', r'=', r'(x_2,y_2,z_2)')
        eq3 = MathTex(r'sP+tQ')
        eq4 = MathTex(r'=', r'1')
        eq5 = MathTex(r's+t', r'=', r'1')
        eq6 = MathTex(r's', r'=', r'1-t')
        eq7 = MathTex(r'(1-t)P+tQ')
        eq8 = MathTex(r'f(x,y,z)', r'=', r'c_1x^3+c_2x^2y+c_3xyz+\cdots')
        eq9 = MathTex(r'{\sf curve}', r'=', r'\{P\colon f(P)=0\}')
        eq10 = MathTex(r'f(sP+tQ)', r'=', r'as^3+bs^2t + cst^2 + dt^3')
        eq11 = MathTex(r'f(sP+tQ)', r'=', r'st(bs+ct)')
        eq12 = MathTex(r'R', r'=', r'cP-bQ')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0], eq3[0][1], eq3[0][4], eq9[2][1], eq9[2][5]).set_color(col_pt1)
        VGroup(eq12[0]).set_color(col_pt2)
        VGroup(eq1[2][1:3], eq1[2][4:6], eq1[2][7:9],
               eq8[0][2], eq8[0][4], eq8[0][6],
               eq8[2][2], eq8[2][7], eq8[2][9], eq8[2][13:16]).set_color(col_x)
        VGroup(eq4[1], eq5[2], eq9[2][-2], eq8[2][3], eq8[2][8],
               eq10[2][2], eq10[2][6], eq10[2][12], eq10[2][16]).set_color(col_num)
        VGroup(eq3[0][0], eq3[0][3], eq10[2][1], eq10[2][5], eq10[2][7],
               eq10[2][10:12], eq10[2][15]).set_color(col_t)
        VGroup(eq8[0][0], eq9[2][3]).set_color(col_f)
        VGroup(eq9[2][0], eq9[2][-1], eq9[2][2]).set_color(col_op)
        VGroup(eq9[0]).set_color(col_txt)
        VGroup(eq8[2][:2], eq8[2][5:7], eq8[2][11:13],
               eq10[2][0], eq10[2][4], eq10[2][9], eq10[2][14],
               eq12[2][0], eq12[2][3]).set_color(col_a)

        mh.copy_colors_eq(eq1, eq2)

        eq1.to_corner(UL, buff=0.6).shift(DOWN*0.2)
        mh.align_sub(eq2, eq2[1], eq1[1]).next_to(eq1, DOWN, buff=0.3, coor_mask=UP)
        eq3.next_to(eq2, DOWN, buff=1.1)
        mh.align_sub(eq4, eq4[0], eq1[1])
        eq4[1].move_to(eq1[2][7:9], coor_mask=RIGHT)
        eq4_1 = mh.align_sub(eq4.copy(), eq4[0], eq2[1])[1]
        eq4_1.move_to(eq2[2][7:9], coor_mask=RIGHT)
        eq5.next_to(eq3, DOWN, buff=0.3)
        mh.align_sub(eq6, eq6[0][0], eq5[0][0])
        mh.align_sub(eq7, eq7[0][6], eq3[0][2]).move_to(eq6, coor_mask=RIGHT)
        eq9.next_to(eq3, DOWN, buff=1).align_to(eq1, LEFT)
        eq8.next_to(eq9, DOWN, buff=0.4).align_to(eq1, LEFT)
        VGroup(eq8, eq9).to_edge(DOWN, buff=1.)
        mh.align_sub(eq10, eq10[0][0], eq8[0][0])
        shift = mh.diff(eq8[1], eq10[1])*RIGHT
        VGroup(eq8[0][-1], eq8[1], eq8[2]).shift(shift)
        eq8[0][2:4].shift(shift * 0.25)
        eq8[0][4:6].shift(shift * 0.5)
        eq8[0][6].shift(shift * 0.75)
        mh.align_sub(eq11, eq11[1], eq10[1])
        eq12.move_to(eq3).shift(DOWN*0.5).to_edge(LEFT, buff=1.7)

        self.add(eq1, eq2)
        self.play(FadeIn(eq3))
        self.wait(0.1)
        rect = SurroundingRectangle(VGroup(eq4[1], eq4_1, eq1[2][7:9], eq2[2][7:9]),
                                    fill_opacity=0, stroke_opacity=1, stroke_color=RED,
                                    stroke_width=5, corner_radius=0.1, buff=0.1)
        self.play(FadeIn(rect, rate_func=linear, run_time=0.4))
        self.wait(0.1)
        self.play(FadeIn(eq4[1], eq4_1), FadeOut(eq1[2][7:9], eq2[2][7:9]))
        self.wait(0.1)
        eq3_1 = eq3.copy()
        self.play(mh.rtransform(eq3[0][0].copy(), eq5[0][0], eq3[0][2].copy(), eq5[0][1],
                                eq3[0][3].copy(), eq5[0][2], run_time=1.5),
                  Succession(Wait(0.9), FadeIn(eq5[1:])),
                  FadeOut(rect, run_time=0.5, rate_func=linear))
        self.wait(0.1)
        self.play(mh.rtransform(eq5[0][0], eq6[0][0], eq5[0][2], eq6[2][2],
                                eq5[1], eq6[1], eq5[2][0], eq6[2][0]),
                  mh.fade_replace(eq5[0][1], eq6[2][1], coor_mask=RIGHT))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0][1:], eq7[0][5:],
                                eq6[2][:], eq7[0][1:4], run_time=1.5),
                  FadeOut(eq3[0][0], run_time=1.5),
                  FadeOut(eq6[:2]),
                  Succession(Wait(0.7), FadeIn(eq7[0][0], eq7[0][4], run_time=1.5)))
        self.wait(0.1)
        self.play(FadeIn(rect, run_time=0.4, rate_func=linear))
        self.play(FadeOut(eq4[1], eq4_1), FadeIn(eq1[2][7:9], eq2[2][7:9]))
        self.play(FadeOut(eq7[0][:5]), FadeIn(eq3[0][0]), FadeOut(rect, run_time=0.5))
        self.play(mh.rtransform(eq3[0][0], eq3_1[0][0], eq7[0][5:], eq3_1[0][1:]))
        eq3 = eq3_1
        self.wait(0.1)
        self.play(FadeIn(eq8, eq9))
        self.wait(0.1)
        eq8.set_z_index(5)
        eq10.set_z_index(5)
        self.play(mh.rtransform(eq8[0][:2], eq10[0][:2], eq8[0][-1], eq10[0][-1], eq8[1], eq10[1]),
                  mh.rtransform(eq3[0][:].copy(), eq10[0][2:-1], run_time=2),
                  Succession(Wait(1), FadeOut(eq8[0][2:-1])))
        self.play(FadeOut(eq8[2]), FadeIn(eq10[2]), run_time=2)
        self.wait(0.1)
        self.play(FadeOut(eq9))
        gp1 = VGroup(eq10.copy(), eq10.copy().next_to(eq10, DOWN, buff=0.3)).move_to(eq10)
        self.play(mh.rtransform(eq10, gp1[0], eq10.copy(), gp1[1]))
        self.play(FadeOut(gp1[0][0][4:7], gp1[0][2][3:]))
        eq_ = MathTex(r'0', r'=', color=col_num)
        mh.align_sub(eq_, eq_[1], gp1[0][1])
        self.wait(0.1)
        self.play(FadeOut(gp1[0][0][:4], gp1[0][0][7:]), FadeIn(eq_[0]))
        self.wait(0.1)
        pt = gp1[1][2][:3].get_center()
        v = DOWN*0.4 + LEFT*0.5
        line = Line(pt+v, pt-v, stroke_width=6, stroke_color=RED).set_z_index(10)
        self.play(Create(line, run_time=0.5))
        self.wait(0.1)
        self.play(FadeOut(line, gp1[1][2][:4]))
        eq_1 = VGroup(gp1[1][0], gp1[1][1], gp1[1][2][4:]).copy()
        eq_2 = eq_1.copy()
        mh.align_sub(eq_1, eq_1[1], gp1[0][1])
        eq_1[2].align_to(gp1[0][2], LEFT)
        self.wait(0.1)
        self.play(mh.rtransform(eq_2, eq_1),
                  FadeOut(eq_[0], gp1[0][1], gp1[0][2][:3]))
        self.play(FadeOut(eq_1[0][2:5], eq_1[2][:-3]))
        self.wait(0.1)
        self.play(FadeOut(eq_1[0][:2], eq_1[0][5:]), FadeIn(eq_[0]))
        self.wait(0.1)
        pt = gp1[1][2][-3:].get_center()
        v = DOWN*0.4 + LEFT*0.5
        line = Line(pt+v, pt-v, stroke_width=6, stroke_color=RED).set_z_index(10)
        self.play(Create(line, run_time=0.5))
        self.wait(0.1)
        self.play(FadeOut(line, gp1[1][2][-4:]))
        self.wait(0.1)
        self.play(FadeOut(eq_[0], eq_1[1], eq_1[2][-3:]))
        self.wait(0.1)
        eq11_1 = eq11.copy()
        mh.align_sub(eq11[2], eq11[2][5], gp1[1][2][8], coor_mask=RIGHT)
        self.play(AnimationGroup(mh.rtransform(gp1[1][:2], eq11[:2], gp1[1][2][4:6], eq11[2][3:5],
                                gp1[1][2][5].copy(), eq11[2][0], gp1[1][2][7], eq11[2][1],
                                gp1[1][2][8:10], eq11[2][5:7], gp1[1][2][11].copy(), eq11[2][7]),
                  mh.rtransform(gp1[1][2][10:12], eq11[2][:2]),
                  FadeOut(gp1[1][2][6], shift=mh.diff(gp1[1][2][5], eq11[2][4])),
                  FadeOut(gp1[1][2][12], shift=mh.diff(gp1[1][2][11], eq11[2][7])),
                                 run_time=1.5),
                  Succession(Wait(1), FadeIn(eq11[2][2], eq11[2][-1]))
                  )
        mh.align_sub(eq11_1, eq11_1[2], eq11[2])
        self.wait(0.1)
        self.play(mh.rtransform(eq11, eq11_1))
        eq11 = eq11_1
        self.wait(0.1)
        rect = SurroundingRectangle(eq12,
                                    fill_opacity=0, stroke_opacity=1, stroke_color=RED,
                                    stroke_width=8, corner_radius=0.1, buff=0.2)

        self.play(mh.rtransform(eq3[0][1], eq12[2][1], eq3[0][-1], eq12[2][-1]),
                  eq3[0][2].animate.move_to(eq12[2][2]),
                  eq3[0][0].animate.shift(mh.diff(eq3[0][1], eq12[2][1])),
                  eq3[0][3].animate.shift(mh.diff(eq3[0][-1], eq12[2][-1])),
                  Succession(Wait(0.3), FadeIn(eq12[:2])),
                  Succession(Wait(0.5), FadeIn(rect)),
                  )
        self.wait(0.1)
        self.play(FadeOut(eq3[0][0]), FadeIn(eq12[2][0]))
        self.wait(0.1)
        self.play(FadeOut(eq3[0][2:4]), FadeIn(eq12[2][2:4]))
        self.wait()

class ZEqOne(Scene):
    def construct(self):
        eq = MathTex(r'z=1', font_size=75, stroke_width=2)[0]
        eq[0].set_color(col_x)
        eq[-1].set_color(col_num)
        self.add(eq)

class FermatCubic(Narration1):
    just_start = False
    bgcolor=BLACK
    trcolor = BLACK

    def get_axes(self, xmin=-1., xmax=1., ymin=-1., ymax=1., z_index=1.):
        ax = Axes(x_range=[xmin, xmax], y_range=[ymin, ymax], x_length=7.5, y_length=7.5, tips=False,
                  axis_config={'color': GREY, 'stroke_width': 4, 'include_ticks': False,
                               },
                  ).set_z_index(1)
        corner =  ax.get_corner(DR) - mh.coords_to_point(1,0)
        ax.shift((corner[0] + corner[1])*LEFT)
        box = SurroundingRectangle(ax, fill_color=BLACK, fill_opacity=0, stroke_opacity=1, stroke_color=WHITE * 0.2,
                                   stroke_width=5, buff=0).set_z_index(z_index)
        asymp = DashedLine(ax.coords_to_point(xmin, -xmin), ax.coords_to_point(xmax, -xmax),
                           stroke_width=4, stroke_color=DARK_GREY).set_z_index(0.9)
        sym = DashedLine(ax.coords_to_point(0, 0), ax.coords_to_point(xmax, xmax),
                           stroke_width=4, stroke_color=DARK_GREY).set_z_index(0.9)
        return ax, box, asymp, sym

    def construct(self):
        xmax = 2.4
        xmin = -2.4
        ymax = xmax
        ymin = xmin
        a = 7
        n = 150

        ax, box, asymp, sym = self.get_axes(xmin, xmax, ymin, ymax)

        def get_curve(n, a, x0 = None):
            if x0 is None:
                x0 = -math.pow(ymax**3 - a, 1/3)
            x1 = math.pow(a/2, 1/3)
            xvals0 = np.linspace(x0, x1, n)
            yvals0 = np.pow(a - xvals0**3, 1/3)
            xvals = np.concat([xvals0, yvals0[-2::-1]])
            yvals = np.concat([yvals0, xvals0[-2::-1]])
            plt = ax.plot_line_graph(xvals, yvals, stroke_width=7, stroke_color=BLUE, add_vertex_dots=False)
            return plt
        to_add = [asymp, box]
        if self.just_start:
            box5 = box.copy().set_fill(opacity=0.6).set_stroke(opacity=0, width=0).set_z_index(0)
            plt0 = get_curve(n, 1.).set_z_index(1.5)
            plt = get_curve(n, a).set_z_index(1.5)
            gp = VGroup(*to_add, box5, ax, plt0, plt)
            pt = mh.coords_to_point(1, 0)
            scale0 = 0.7
            gp.scale(scale0, about_point=pt)
            self.add(*to_add, box5, ax)
            self.wait(0.1)
            self.play(Create(plt0), rate_func=linear, run_time=1.5)
            self.wait(0.1)
            self.play(mh.rtransform(plt0, plt))
            self.wait(0.1)
            self.play(gp.animate.scale(1/scale0, about_point=pt), run_time=2)
            self.wait()
            return

        box2 = Rectangle(width=config.frame_width, height=2, stroke_opacity=0, stroke_width=0, fill_color=BLACK, fill_opacity=1)
        box2.next_to(box, UP, buff=0).set_z_index(9)
        box3 = Rectangle(width=8, height=config.frame_height, stroke_opacity=0, stroke_width=0, fill_color=BLACK, fill_opacity=1)
        box3.next_to(box, RIGHT, buff=0).set_z_index(9)
        box4 = box3.copy().next_to(box, LEFT, buff=0)
        box5 = box2.copy().next_to(box, DOWN, buff=0)

        line_x = Line(ax.coords_to_point(-7.5, 0), ax.coords_to_point(5, 0), stroke_color=GREY, stroke_width=4).set_z_index(1)
        line_y = Line(ax.coords_to_point(0, -5), ax.coords_to_point(0, 7.5), stroke_color=GREY, stroke_width=4).set_z_index(1)
        plt = get_curve(n, a, -7.5).set_z_index(1.5)

        self.add(*to_add, plt, box2, box3, box4, box5, line_x, line_y)

        def get_dot(p, color=YELLOW):
            pt = ax.coords_to_point(p[0]/p[2], p[1]/p[2])
            dot = Dot(pt, radius=0.15, color=color).set_z_index(8)
            return dot

        def add(p, q):
            """
            (p[0]s + q[0]t)^3 + (p[1]s + q[1]t)^3 - a(p[2]s+q[2]t)^3 = 0
            p[0]^2q[0] + p[1]^2q[1] - r[0]
            """
            if p == q:
                v = (p[1] * p[1], -p[0] * p[0], 0)
                b = 3 * (p[0] * v[0] * v[0] + p[1] * v[1] * v[1])
                c = v[0] * v[0] * v[0] + v[1] * v[1] * v[1]
            else:
                b = p[0]*p[0]*q[0] + p[1]*p[1]*q[1] - a * p[2]*p[2]*q[2]
                c = p[0]*q[0]*q[0] + p[1]*q[1]*q[1] - a * p[2]*q[2]*q[2]
                v = q

            r = c * p[0] - b * v[0], c * p[1] - b * v[1], c * p[2] - b * v[2]

            # u = p[0]*p[0]*q[0] + p[1]*p[1]*q[1] - a * p[2]*p[2]*q[2]
            g = math.gcd(*r)
            if r[0] < 0:
                g = -g
            r = tuple((_//g for _ in r))
            print('r = ', r)
            return r

        def line(p, q, len=1., color=WHITE):
            x1, y1 = (p[0]/p[2], p[1]/p[2])
            x2, y2 = (q[0]/q[2], q[1]/q[2])
            u = len / math.sqrt((x2-x1)**2 + (y2-y1)**2)
            pt0 = ax.coords_to_point(x1, y1)
            pt1 = ax.coords_to_point(x2, y2)
            obj = Line(pt0*(1+u)-pt1*u, pt1*(1+u)-pt0*u, stroke_width=6, stroke_color=color).set_z_index(2)
            return obj

        # p1 = (2, -1, 1)
        p1 = (1, -2, -1)
        pt1 = get_dot(p1)
        p2 = add(p1, p1)
        pt2 = get_dot(p2, col_pt2)

        line1 = line(p1, p2)

        self.play(FadeIn(pt1))
        self.play(Create(line1, rate_func=linear))
        self.play(FadeIn(pt2))
        self.play(line1.animate.set_opacity(0.3))

        op2 = 0.5

        method = 1
        if method == 1:
            p2_ = (p2[1], p2[0], p2[2])
            pt2_ = get_dot(p2_, col_pt3)
            p3 = add(p2_, p1)
            pt3 = get_dot(p3, col_pt3)
            p6 = add(p3, p3)
            pt6 = get_dot(p6, col_pt2)
            p6_ = (p6[1], p6[0], p6[2])
            pt6_ = get_dot(p6_, col_pt2)
            p4 = add(p2_, p2_)
            pt4 = get_dot(p4, col_pt3)
            p8 = add(p4, p4)
            pt8 = get_dot(p8, col_pt3)
            p10 = add(p8, p2_)
            pt10 = get_dot(p10, col_pt2)

            print('P2 = ', p2)
            print('P6 = ', p6)
            print('P10 = ', p10)

            line2 = line(p2_, p1)
            line3 = line(p3, p6)
            line4 = line(p2_, p4)
            line5 = line(p4, p2)
            line6 = line(p4, p8)
            line7 = line(p8, p2)
            line8 = line(p8, p2_)

            self.play(FadeIn(sym))
            self.play(mh.rtransform(pt2.copy(), pt2_),
                      pt2.animate.set_opacity(op2))
            self.play(Create(line2, rate_func=linear))
            self.play(FadeIn(pt3))
            self.play(line2.animate.set_opacity(0.3),
                      pt1.animate.set_opacity(op2),
                      pt2_.animate.set_opacity(op2))
            self.play(Create(line3, rate_func=linear))
            self.play(FadeIn(pt6))
            self.play(line3.animate.set_opacity(0.3),
                      pt3.animate.set_opacity(op2))
            self.play(pt6.animate.set_opacity(op2))
            self.play(pt2_.animate.set_opacity(1))
            line4_ = line4.copy()
            self.play(Create(line4_, rate_func=lambda t: linear(t/3), run_time=1))
            self.remove(line4_)
            self.add(line4)
            line4.set_opacity(1)

            dots = VGroup(pt1, pt2, pt2_, pt3, pt6, pt4)
            gp = VGroup(line_x, line_y, plt, line1, line2, line3, line4)
            gp2 = VGroup(line5)
            scale = 0.5
            pt = ax.coords_to_point(xmax,ymin)
            dots2 = VGroup(*(dot.copy().move_to((dot.get_center() - pt)*scale+pt) for dot in dots))
            dots3 = dots.copy()
            shift = (pt - ax.coords_to_point(0,0)) * (1-scale)
            self.play(gp.animate.scale(scale, about_point=pt),
                      mh.rtransform(dots[:-1], dots2[:-1]),
                      sym.animate.shift(shift))
            dots = dots3
            gp2.scale(scale, about_point=pt)
            self.play(FadeIn(dots2[-1]))
            self.play(line4.animate.set_opacity(0.3),
                      dots2[2].animate.set_opacity(op2))
            self.play(dots2[1].animate.set_opacity(1))
            dots3[2].set_opacity(op2)
            dots3[1].set_opacity(1)
            self.play(Create(line5, rate_func=linear))

            self.play(VGroup(gp, gp2).animate.scale(1/scale, about_point=pt),
                      mh.rtransform(dots2, dots),
                      sym.animate.shift(-shift))
            self.play(dots[4].animate.set_opacity(1))
            self.play(dots[4].animate.set_opacity(op2),
                      dots[1].animate.set_opacity(op2),
                      line5.animate.set_opacity(0.3))

            gp = VGroup(line_x, line_y, plt, line1, line2, line3, line4, line5)
            gp2 = VGroup(line6, line7, line8)
            dots = VGroup(*dots[:], pt8, pt6_, pt10)
            scale = 0.4
            s = 0.4
            pt = ax.coords_to_point(xmax * s,-xmax * s)
            dots2 = VGroup(*(dot.copy().move_to((dot.get_center() - pt)*scale+pt) for dot in dots))
            shift = (pt - ax.coords_to_point(0,0)) * (1-scale)
            self.play(gp.animate.scale(scale, about_point=pt),
                      mh.rtransform(dots[:-3], dots2[:-3]),
                      sym.animate.shift(shift))
            gp2.scale(scale, about_point=pt)

            self.play(Create(line6), rate_func=linear, run_time=2)
            self.play(FadeIn(dots2[6]))
            self.play(line6.animate.set_opacity(0.3),
                      dots2[5].animate.set_opacity(op2))

            gp = VGroup(*gp, line6)
            gp2 = VGroup(line7, line8)
            dots = dots2
            scale = 1.85
            s = 0.8
            pt = ax.coords_to_point(xmax * s,-xmax * s)
            dots2 = VGroup(*(dot.copy().move_to((dot.get_center() - pt)*scale+pt) for dot in dots))
            shift = (pt - shift) * (1-scale)
            self.play(gp.animate.scale(scale, about_point=pt),
                      mh.rtransform(dots[:-2], dots2[:-2]),
                      sym.animate.shift(shift))
            gp2.scale(scale, about_point=pt)
            self.play(dots2[1].animate.set_opacity(1))
            self.play(Create(line7), rate_func=linear, run_time=1.6)
            self.play(FadeIn(dots2[7]))
            self.play(line7.animate.set_opacity(0.3),
                      dots2[1].animate.set_opacity(op2),
                      dots2[6].animate.set_opacity(op2))
            dot_ = dots2[7].copy()
            self.play(dot_.animate.move_to(dots2[4]))
            self.remove(dot_)
            dots2[4].set_opacity(1)
            self.play(dots2[4].animate.set_opacity(op2),
                      dots2[7].animate.set_opacity(op2))
            self.play(dots2[6].animate.set_opacity(1),
                      dots2[2].animate.set_opacity(1))
            self.play(Create(line8), rate_func=linear, run_time=1.6)
            self.play(FadeIn(dots2[8]))

        else:
            p1_ = (-p1[1], -p1[0], -p1[2])
            pt1_ = get_dot(p1_, col_pt1)
            p3 = add(p2, p1_)
            pt3 = get_dot(p3, col_pt2)
            p6 = add(p3, p3)
            pt6 = get_dot(p6, col_pt2)
            p4 = add(p2, p2)
            pt4 = get_dot(p4, col_pt2)

            line2 = line(p1_, p2)
            line3 = line(p3, p6)
            line4 = line(p2, p4)
            # line5 = line(p4, p2)

            self.play(mh.rtransform(pt1.copy(), pt1_, rate_func=smooth, run_time=1.5))
            self.play(Create(line2, rate_func=linear))
            self.play(FadeIn(pt3))
            self.play(line2.animate.set_opacity(0.5))
            self.play(Create(line3, rate_func=linear))
            self.play(FadeIn(pt6))
            self.play(line3.animate.set_opacity(0.5))
            self.play(Create(line4, rate_func=linear))
            self.play(FadeIn(pt4))
            # gp = VGroup(pt1, line1, pt2, pt1_, line2, pt3, line3, pt6, line4, pt4)
            # self.play(gp.animate.scale(2.4/7, about_point=ax.coords_to_point(0,0)))
            # self.play(line4.animate.set_opacity(0.5))
            # self.play(Create(line5, rate_func=linear))

        self.wait()


class FermatCubicStart(FermatCubic):
    just_start = True


class FermatEqn(FermatCubic):
    trcolor = BLACK

    def part1(self, just_eq=False):
        MathTex.set_default(stroke_width=2, font_size=100)
        eq1 = MathTex(r'x^3+y^3', r'=', r'z^3')
        eq2 = MathTex(r'x^3+y^3', r'=', r'7z^3')
        eq3 = MathTex(r'2^3 - 1', r'=', r'7')
        eq4 = MathTex(r'2^3 + (-1)^3', r'=', r'7\!\cdot\!1^3')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0][0], eq1[0][3], eq1[2][0]).set_color(col_x)
        VGroup(eq1[0][1], eq1[0][4], eq1[2][1], eq2[2][0],
               eq3[0][0], eq3[0][1], eq3[0][3], eq3[2],
               eq4[0][-1], eq4[2][-2:]).set_color(col_num)
        VGroup(eq4[2][1], eq4[0][4]).set_color(col_op)

        eq2.to_edge(DOWN, buff=0.4)
        mh.align_sub(eq1, eq1[1], eq2[1])
        mh.align_sub(eq3, eq3[1], eq2[1])
        eq2_1 = eq2.copy().next_to(eq3, UP, buff=0.4, coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq3[1])
        eq4.move_to(eq2, coor_mask=RIGHT)
        VGroup(eq1, eq2).move_to(VGroup(eq2_1, eq3), coor_mask=UP)

        gp1 = VGroup(eq1, eq2).set_z_index(1)
        box1 = SurroundingRectangle(gp1, stroke_width=0, stroke_opacity=0, fill_opacity=0.6,
                                    fill_color=BLACK, corner_radius=0.15, buff=0.2)
        gp2 = VGroup(eq2_1, eq3, eq4).set_z_index(1)
        box2 = SurroundingRectangle(gp2, stroke_width=0, stroke_opacity=0, fill_opacity=0.6,
                                    fill_color=BLACK, corner_radius=0.15, buff=0.2)

        shift = LEFT * 374/1920 * config.frame_width
        VGroup(eq2, gp2, box2).shift(shift)
        if just_eq:
            mh.copy_colors_eq(eq1[0], eq2_1[0], eq1[2][:], eq2_1[2][1:])
            VGroup(eq4[0][:2], eq4[0][5], eq4[0][7], eq4[2][0]).set_color(col_num)
            return eq2_1, eq4

        self.add(eq1, box1)
        self.play(VGroup(eq1, box1).animate.shift(shift))

        self.play(mh.rtransform(eq1[:2], eq2[:2], eq1[2][:], eq2[2][1:]),
                  FadeIn(eq2[2][0]))
        self.wait(0.1)
        self.play(mh.rtransform(eq2, eq2_1, box1, box2),
                  Succession( Wait(0.4), FadeIn(eq3)),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0][:2], eq4[0][:2], eq3[0][2], eq4[0][4], eq3[0][3], eq4[0][5],
                                eq3[1], eq4[1], eq3[2][0], eq4[2][0]),
                  mh.fade_replace(eq3[0][2].copy(), eq4[0][2]),
                  Succession(Wait(0.4), FadeIn(eq4[0][3], eq4[0][-2:])))
        self.wait(0.1)
        self.play(FadeIn(eq4[2][1:]))
        self.wait(0.1)
        self.play(FadeOut(box2))
        self.wait()
        return eq2, eq4

    def construct(self):
        eqs0 = self.part1()
        self.play(FadeOut(*eqs0))


class FermatPoints(FermatEqn):
    bgcolor = BLACK
    trcolor = GREY

    def construct(self):
        if not config.transparent: self.add(*self.get_axes(z_index=0))
        eqs0 = self.part1(True)
        self.add(*eqs0)
        MathTex.set_default(stroke_width=1.5, font_size=65)
        eq0 = MathTex(r'x^3+y^3', r'=', r'7z^3')
        eq1 = MathTex(r'P', r'=', r'(-1:2:1)')
        eq2 = MathTex(r'f(x,y,z)', r'=', r'x^3+y^3-7z^3')
        eq3 = MathTex(r'Df', r'=', r'\Big(', r'\frac{\partial f}{\partial x}', r',',
                      r'\frac{\partial f}{\partial y}', r',', r'\frac{\partial f}{\partial x}', r'\Big)')
        eq4 = MathTex(r'Df', r'=', r'3', r'\Big(', r'x^2', r',', r'y^2', r',', r'-7z^2', r'\Big)')
        eq5 = MathTex(r'Df', r'\sim', r'\big(', r'x^2', r',', r'y^2', r',', r'-7z^2', r'\big)')
        eq6 = MathTex(r'Df', r'\sim', r'(', r'1', r',', r'4', r',', r'-7', r')').align_to(eq1, LEFT)
        eq7 = MathTex(r'V', r'=', r'(4,-1,0)')

        mh.rtransform.copy_colors = True
        VGroup(eq1[0], eq7[0]).set_color(col_pt1)
        VGroup(eq1[2][2], eq1[2][4], eq1[2][6], eq2[2][0], eq2[2][3], eq2[2][7],
               eq2[2][1], eq2[2][4], eq2[2][6], eq2[2][8], eq4[0][0], eq6[5],
               eq4[4][1], eq4[6][1], eq4[8][3], eq4[2],
               eq7[2][1], eq7[2][4], eq7[2][6]).set_color(col_num)
        VGroup(eq2[0][2], eq2[0][4], eq2[0][6], eq2[2][0], eq2[2][3], eq2[2][-2],
               eq3[3][4]).set_color(col_x)
        VGroup(eq2[0][0], eq3[0][1], eq3[3][1]).set_color(col_f)
        VGroup(eq3[0][0], eq3[3][0], eq3[3][3]).set_color(col_op)
        mh.copy_colors_eq(eq3[3], eq3[5], eq3[3], eq3[7])

        eq1.to_corner(UL, buff=0.4).shift(DOWN*0.2)
        eq2.next_to(eq1, DOWN, buff=0.4).align_to(eq1, LEFT)
        mh.align_sub(eq0, eq0[1], eq2[1]).align_to(eq1, RIGHT)
        eq3.next_to(eq2, DOWN, buff=0.4).to_edge(LEFT, buff=0.3)
        mh.align_sub(eq4, eq4[1], eq3[1])
        mh.align_sub(eq5, eq5[0], eq4[0])
        mh.align_sub(eq6, eq6[0], eq4[0])
        eq7.next_to(eq6, DOWN, buff=0.3)
        mh.align_sub(eq7, eq7[2], eq6[2:], coor_mask=RIGHT)

        self.play(FadeIn(eq1))
        self.play(FadeOut(eqs0[1]), mh.rtransform(eqs0[0], eq0))
        self.play(Succession(Wait(0.5), FadeIn(eq2[:2])),
                  AnimationGroup(mh.rtransform(eq0[0][:], eq2[2][:5], eq0[2][:], eq2[2][-3:]),
                  mh.fade_replace(eq0[1], eq2[2][5]), run_time=1.5))
        self.wait(0.1)
        self.play(FadeIn(eq3))
        self.wait(0.1)
        self.play(AnimationGroup(
            mh.rtransform(eq3[:2], eq4[:2], eq3[2][0], eq4[3][0], eq2[2][0].copy(), eq4[4][0],
                          eq3[4], eq4[5]),
            mh.stretch_replace(eq2[2][1].copy(), eq4[2][0]),
            eq3[5:].animate.shift(mh.diff(eq3[4], eq4[5])*RIGHT),
            mh.fade_replace(eq2[2][1].copy(), eq4[4][1]),
                  run_time=1.5),
                  FadeOut(eq3[3]),
                  )
        self.play(AnimationGroup(
            mh.rtransform(eq2[2][3].copy(), eq4[6][0],
                          eq3[6], eq4[7]),
            mh.stretch_replace(eq2[2][4].copy(), eq4[2][0]),
            eq3[7:].animate.shift(mh.diff(eq3[6], eq4[7])*RIGHT),
            mh.fade_replace(eq2[2][4].copy(), eq4[6][1]),
                  run_time=1.5),
                  FadeOut(eq3[5]),
                  )
        self.play(AnimationGroup(
            mh.rtransform(eq2[2][5:8].copy(), eq4[8][:3],
                          eq3[8], eq4[9]),
            mh.stretch_replace(eq2[2][8].copy(), eq4[2][0]),
            mh.fade_replace(eq2[2][8].copy(), eq4[8][3]),
                  run_time=1.5),
                  FadeOut(eq3[7]),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0], eq5[0], eq4[4:-1], eq5[3:-1]),
                  mh.stretch_replace(eq4[3], eq5[2]),
                  mh.stretch_replace(eq4[-1], eq5[-1]),
                  mh.fade_replace(eq4[1], eq5[1]),
                  FadeOut(eq4[2]))
        self.play(mh.rtransform(eq5[:2], eq6[:2], eq1[2][2].copy(), eq6[3][0], eq5[4], eq6[4]),
                  FadeOut(eq1[2][1].copy(), shift=mh.diff(eq1[2][2], eq6[3][0])),
                  eq5[2].animate.shift(mh.diff(eq5[2], eq6[2])),
                  eq5[5:].animate.shift(mh.diff(eq5[4], eq6[4])*RIGHT),
                  FadeOut(eq5[3]),
                  run_time=1.5)
        self.play(mh.rtransform(eq5[6], eq6[6]),
                  mh.fade_replace(eq1[2][4].copy(), eq6[5]),
                  eq5[7:].animate.shift(mh.diff(eq5[6], eq6[6])),
                  FadeOut(eq5[5]),
                  run_time=1.5)
        self.play(mh.rtransform(eq5[7][:2], eq6[7][:2], eq5[2], eq6[2], eq5[8], eq6[8]),
                  FadeOut(eq1[2][6].copy(), target_position=eq5[7][2]),
                  FadeOut(eq5[7][2:]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq7))
        self.wait(0.1)
        eq1.set_z_index(20)
        eq1_ = eq1.copy().next_to(eq2, DOWN, buff=0.3, coor_mask=UP)
        eq7_ = eq7.copy().next_to(eq1_, DOWN, buff=0.2).align_to(eq1, LEFT)
        self.play(FadeOut(eq6), mh.transform(eq1, eq1_, eq7, eq7_, run_time=1.5))
        self.wait(0.1)

        eq8 = MathTex(r'f(sP+tV)', r'=', r'f(P)s^3', r'+', r'D_Vf(P)s^2t',
                      r'+', r'(\ \cdots\ )st^2', r'+', r'(\ \cdots\ )t^3')
        # eq9 = MathTex(r'f(s', r'(-1,2,1)', r'+t', r'(4,-1,0)', r')', r'=')
        # eq9 = MathTex(r'sP+tV', r'=', r's', r'(-1,2,1)', r'+', r't', r'(4,-1,0)')
        # eq10 = MathTex(r'sP+tV', r'=', r'(-s+4t,2s-t,s)').set_z_index(15)
        eq9 = MathTex(r'sP+tV', r'=', r'(-s+4t,2s-t,s)')

        VGroup(eq8[0][3], eq8[0][6], eq8[2][2], eq8[4][1], eq8[4][4]).set_color(col_pt1)
        VGroup(eq8[0][0], eq8[2][0], eq8[4][2]).set_color(col_f)
        VGroup(eq8[0][2], eq8[0][5]).set_color(col_t)

        eq8.next_to(eq7, DOWN, buff=0.3).align_to(eq1, LEFT)
        eq8[2:].next_to(eq8[0], DOWN, buff=0.3).align_to(eq8[0], LEFT).shift(RIGHT)
        eq8[6:].next_to(eq8[2], DOWN, buff=0.3).align_to(eq8[2], LEFT)
        VGroup(eq8[2][-2], eq8[4][-3], eq8[4][-1], eq8[6][-3:-1], eq8[8][-2],
               eq9[0][0], eq9[0][3], eq9[2]).set_color(col_t)
        VGroup(eq8[2][-1], eq8[4][-2], eq8[6][-1], eq8[8][-1]).set_color(col_num)
        VGroup(eq8[4][0]).set_color(col_op)
        # mh.align_sub(eq9, eq9[0][0], eq8[0][0])
        mh.align_sub(eq9, eq9[1], eq7[1]).align_to(eq7, LEFT)
        # mh.align_sub(eq10, eq10[1], eq9[1])

        self.play(FadeIn(eq8[0][2:-1]))
        self.wait(0.1)
        self.play(FadeIn(eq8[0][:2], eq8[0][-1], run_time=1.6))
        self.play(FadeIn(eq8[1:]))
        self.wait(0.1)
        circ = mh.circle_eq(eq8[2]).set_z_index(15)
        self.play(Create(circ, run_time=0.5, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(circ, eq8[2:4]))
        self.wait(0.1)
        circ = mh.circle_eq(eq8[4]).set_z_index(15)
        self.play(Create(circ, run_time=0.5, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(circ, eq8[4:6]))
        self.wait(0.1)
        box1 = SurroundingRectangle(eq9[2:], stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK, buff=0.05,
                                    corner_radius=0.2).set_z_index(11)
        shift = mh.diff(eq7[2][-1], eq9[2][-1])
        eq1.set_z_index(15)
        eq7.set_z_index(15)
        eq9.set_z_index(15)
        self.play(mh.rtransform(eq7[0][0], eq9[0][4], eq7[1], eq9[1], eq7[2][0], eq9[2][0],
                                eq7[2][1], eq9[2][4], eq7[2][2], eq9[2][6], eq7[2][3], eq9[2][9],
                                eq7[2][5], eq9[2][11], eq7[2][7], eq9[2][13]
                                ),
                  FadeIn(eq9[0][3], shift=mh.diff(eq7[0][0], eq9[0][4])),
                  FadeIn(eq9[2][5], shift=mh.diff(eq7[2][2], eq9[2][6])),
                  mh.fade_replace(eq7[2][4], eq9[2][10]),
                  FadeOut(eq7[2][6], shift=mh.diff(eq7[2][7], eq9[2][13])),
                  box1.shift(-shift).animate.shift(shift),
                  run_time=1.3
                  )
        self.play(AnimationGroup(mh.rtransform(eq1[0][0].copy(), eq9[0][1], eq1[2][1].copy(), eq9[2][1],
                                eq1[2][4].copy(), eq9[2][7]),
                  FadeIn(eq9[0][0], shift=mh.diff(eq1[0][0], eq9[0][1])),
                  mh.fade_replace(eq1[2][2].copy(), eq9[2][2]),
                  FadeIn(eq9[2][8], shift=mh.diff(eq1[2][4], eq9[2][7])),
                  mh.fade_replace(eq1[2][6].copy(), eq9[2][12]), run_time=1.5),
                  Succession(Wait(0.5), FadeIn(eq9[0][2], eq9[2][3])))
        # self.play(mh.rtransform(eq8[0][:3], eq9[0][:], eq8[0][4:6], eq9[2][:], eq8[0][7], eq9[4][0], eq8[1], eq9[5]),
        #           eq8[0][3].animate.move_to(eq9[1], coor_mask=RIGHT),
        #           eq8[0][6].animate.move_to(eq9[3], coor_mask=RIGHT))
        # eq1.set_z_index(15)
        # eq7.set_z_index(15)
        # self.play(AnimationGroup(mh.rtransform(eq1[2][:], eq9[1][:]),
        #                          run_time=1.6),
        #           Succession(Wait(0.6), FadeOut(eq8[0][3], eq8[0][6])))
        # eq1.set_z_index(15)
        # eq7.set_z_index(12)
        # eq9.set_z_index(15)
        # self.play(mh.rtransform(eq7[0][0], eq9[0][4], eq7[1], eq9[1], eq7[2], eq9[6]),
        #           FadeIn(eq9[0][3], shift=mh.diff(eq7[0][0], eq9[0][4])),
        #           FadeIn(eq9[5], shift=mh.diff(eq7[2], eq9[6])),
        #           box1.shift(-shift).animate.shift(shift),
        #           run_time=1.4
        #           )
        # self.play(mh.rtransform(eq1[0][0].copy(), eq9[0][1], eq1[2].copy(), eq9[3]),
        #           FadeIn(eq9[0][0], shift=mh.diff(eq1[0][0], eq9[0][1])),
        #           FadeIn(eq9[2], shift=mh.diff(eq1[2], eq9[3])),
        #           Succession(Wait(0.2), FadeIn(eq9[0][2], eq9[4])),
        #           run_time=1.2)
        # eq9[3][2].set_z_index(16)
        # self.play(mh.rtransform(eq9[:2], eq10[:2], eq9[3][:2], eq10[2][:2], eq9[2][0], eq10[2][2],
        #                         eq9[3][3:5], eq10[2][6:8], eq9[2][0].copy(), eq10[2][8],
        #                         eq9[2][0].copy(), eq10[2][12], eq9[3][5], eq10[2][11],
        #                         ),
        #           FadeOut(eq9[3][2], shift=mh.diff(eq9[3][1], eq10[2][1])),
        #           )
        self.wait()

if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        ClickBaitEq().render()