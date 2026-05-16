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
                  Succession(Wait(1.), FadeIn(eq1))
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


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        ClickBaitEq().render()