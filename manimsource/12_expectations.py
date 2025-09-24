from manim import *
import numpy as np
import math
import sys
import scipy as sp
from torchgen.api.cpp import return_type

sys.path.append('../')
import manimhelper as mh

H = LabeledDot(Text("H", color=BLACK, font='Helvetica', weight=SEMIBOLD), radius=0.35, color=BLUE).scale(1.5)
T = LabeledDot(Text("T", color=BLACK, font='Helvetica', weight=SEMIBOLD), radius=0.35, color=YELLOW).scale(1.5)


_dice_faces = None


def get_dice_faces(color=WHITE, dot_color=BLACK):
    global _dice_faces
    if _dice_faces is None:
        blank = RoundedRectangle(width=2, height=2, fill_color=color, fill_opacity=1, corner_radius=0.2, stroke_color=GREY)
        dot = Dot(radius=0.22, color=dot_color, z_index=1)
        x = RIGHT * 0.54
        y = UP * 0.54

        _dice_faces = []

        for dots in [
            [ORIGIN],
            [-x - y, x + y],
            [-x - y, ORIGIN, x + y],
            [-x - y, -x + y, x + y, x - y],
            [-x - y, -x + y, x + y, x - y, ORIGIN],
            [-x - y, -x, -x + y, x - y, x, x + y]
        ]:
            _dice_faces.append(VGroup(blank.copy(), *[dot.copy().move_to(s) for s in dots]))
    return _dice_faces



def animate_roll(scene, key, pos=ORIGIN, scale=0.3, right=False, slide=True):
    if isinstance(pos, Mobject):
        pos = pos.get_center()
    key = int(key) - 1
    rows = [
        [1, 5, 6, 2],
        [2, 4, 5, 3], ##
        [3, 1, 4, 6],  ##
        [4, 2, 3, 5], #
        [5, 6, 2, 1], #
        [6, 5, 1, 2],  ##
    ]

    faces = get_dice_faces()
    f_row = [faces[i-1] for i in rows[key]]

    flag = False
    for i in range(10, -1, -1):
        t = -i * i * 0.045
        c = math.cos(t) * scale
        s = math.sin(t) * scale
        if slide:
            d0 = math.floor(2*t/math.pi)
            d = (d0 + math.sin(t - math.pi * d0/2)) * scale * 2
        else:
            d = 0
        if right:
            arr = [f_row[0].copy().apply_matrix([[c, 0], [0, scale]]).move_to(pos + RIGHT * (s+d)),
                   f_row[1].copy().apply_matrix([[s, 0], [0, scale]]).move_to(pos + LEFT * (c-d)),
                   f_row[2].copy().apply_matrix([[-c, 0], [0, scale]]).move_to(pos + LEFT * (s-d)),
                   f_row[3].copy().apply_matrix([[-s, 0], [0, scale]]).move_to(pos + RIGHT * (c+d))]
        else:
            arr = [f_row[0].copy().apply_matrix([[scale, 0], [0, c]]).move_to(pos + UP * (s+d)),
                   f_row[1].copy().apply_matrix([[scale, 0], [0, s]]).move_to(pos + DOWN * (c-d)),
                   f_row[2].copy().apply_matrix([[scale, 0], [0, -c]]).move_to(pos + DOWN * (s-d)),
                   f_row[3].copy().apply_matrix([[scale, 0], [0, -s]]).move_to(pos + UP * (c+d))]

        if c < 0:
            arr[0].set_opacity(0)
        else:
            arr[2].set_opacity(0)
        if s < 0:
            arr[1].set_opacity(0)
        else:
            arr[3].set_opacity(0)
        if flag:
            for j in range(4):
                f[j].target = arr[j]
            scene.play(*[MoveToTarget(f[j]) for j in range(4)], rate_func=rate_functions.linear, run_time=0.05 * (1 + t / 10))
        else:
            f = arr
            flag = True

    scene.remove(*f[1:])
    return f[0]

class ExpectedValue(Scene):
    fs1 = 100
    def construct(self):
        MathTex.set_default(font_size=self.fs1)
        eq1 = Tex(r'random variable $X$')[0]
        eq1[-1].set_color(RED)
        eq2 = Tex(r'expected value $\mathbb E[X]$')[0]
        eq2[-2].set_color(RED)
        eq3 = MathTex(r'\mathbb E[X]', r'=', r'\sum_x p(x) x')
        eq3[0][2].set_color(RED)
        eq3[2][-1].set_color(RED)
        eq3.next_to(eq1, DOWN)
        self.add(eq1)
        self.wait(0.1)
        eq2.next_to(eq1, DOWN)
        gp = VGroup(eq1, eq2).copy().move_to(ORIGIN)
        self.play(mh.rtransform(eq1[:-1], gp[0][:-1], eq1[-1], gp[1][-2]),
                  FadeIn(gp[1][:-2], target_position=eq2[:-2]),
                  FadeIn(gp[1][-1], target_position=eq2[-1]),
                  run_time=0.7)
        eq1, eq2 = gp[0], gp[1]
        eq3.next_to(eq2, DOWN)
        self.wait(0.1)
        gp = VGroup(eq2[:-4].copy().move_to(ORIGIN, coor_mask=RIGHT), eq3).move_to(ORIGIN)
        self.play(mh.rtransform(eq2[:-4], gp[0], eq2[-4:], eq3[0][:]),
                  FadeIn(eq3[1:], rate_func=rush_into),
                  FadeOut(eq1[:-1], shift=mh.diff(eq2[0], gp[0][0])*UP),
                  run_time=1.6)
        eq2 = gp[0]
        self.wait(0.1)

        eq4 = MathTex(r'\mathbb E[X]', r'=', r'\int p(x) x\,dx')
        eq4[0][2].set_color(RED)
        eq4[2][-3].set_color(RED)
        mh.align_sub(eq4, eq4[1], eq3[1])
        eq2_1 = eq2.copy().next_to(eq4, UP, coor_mask=UP)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2][2:7], eq4[2][1:6]),
                  mh.transform(eq2, eq2_1),
                  FadeIn(eq4[2][6:], shift=mh.diff(eq3[2][6], eq4[2][5])),
                  FadeOut(eq3[2][1]),
                  #mh.rtransform(eq3[2][0], eq4[2][0]),
                  mh.stretch_replace(eq3[2][0], eq4[2][0]),
                  run_time=2)

        self.wait(0.1)

        def p(x):
            return math.exp(-x*x/2)

        xmin = -2.5
        xmax = 2.5
        ax = Axes(x_range=[xmin, xmax * 1.1], y_range=[0, 1.1], x_length=12, y_length=3,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "stroke_opacity": 1,
                               },
                  ).set_z_index(2)
        ax.next_to(eq4, DOWN)
        gp = VGroup(eq2.copy(), eq4.copy(), ax).move_to(ORIGIN, coor_mask=UP)
        plt1 = ax.plot(p, (xmin, xmax), stroke_color=BLUE, stroke_width=6).set_z_index(4)
        fill1 = ax.get_area(plt1, color=BLUE, opacity=0.5).set_z_index(3).set_stroke(opacity=0)

        eq5 = Tex(r'probability density', font_size=60)[0].next_to(gp[1][2][-2], DOWN, buff=0.7)
        arr1 = Arrow(eq5[1].get_top(), gp[1][2][1].get_center() + DOWN*0.2, buff=0.1, stroke_width=4)
        arr2 = Arrow(eq5[1].get_bottom(), ax.coords_to_point(.5, p(.5)), buff=0.1, stroke_width=4)

        self.play(LaggedStart(mh.transform(eq2, gp[0], eq4, gp[1], run_time=1.2),
                  AnimationGroup(FadeIn(ax), Create(plt1, rate_func=linear), run_time=1.6),
                  AnimationGroup(FadeIn(fill1), FadeIn(eq5, arr1, arr2), run_time=1, rate_func=linear),
                              lag_ratio=0.3))


        self.wait()

class ExpectedAB(Scene):
    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = WHITE
        Scene.__init__(self, *args, **kwargs)

    fs1 = 100
    fs2 = 80
    cov = r'1/2'

    def construct(self):
        cx = RED
        cy = BLUE
        MathTex.set_default(font_size=self.fs1)
        eq1 = MathTex(r'A, B', r'\sim', r'N(0,1)')
        eq1[0][0].set_color(cx)
        eq1[0][-1].set_color(cy)
        self.add(eq1)
        self.wait(0.1)
        eq2 = MathTex(r'{\rm Corr}(A, B)', r'=', self.cov, font_size=self.fs2)
        eq2.next_to(eq1, DOWN)
        eq2[0][-4].set_color(cx)
        eq2[0][-2].set_color(cy)
        gp = Group(eq1.copy(), eq2).move_to(ORIGIN)
        self.play(FadeIn(eq2), mh.transform(eq1, gp[0]), run_time=1)
        self.wait()

class ExpectedABRho(ExpectedAB):
    cov = r'\rho'
    fs1 = 100
    fs2 = 100

class ABexp(Scene):
    def construct(self):
        MathTex.set_default(font_size=100)
        eq1 = MathTex(r'X', r'=', r'e^{\frac12A^2}')
        eq2 = MathTex(r'Y', r'=', r'e^{\frac12B^2}')
        eq1[0][0].set_color(RED)
        eq1[2][-2].set_color(RED)
        eq2[0][0].set_color(BLUE)
        eq2[2][-2].set_color(BLUE)
        gp = VGroup(eq1.copy(), eq2).arrange(RIGHT, buff=1).move_to(ORIGIN)
        mh.align_sub(eq1.copy(), eq1[1], gp[0][1], coor_mask=UP)
        eq1_1 = eq1.copy()
        self.add(eq1)
        self.wait(0.1)
        self.play(mh.transform(eq1, gp[0]),
                  mh.fade_replace(eq1_1[0], eq2[0]),
                  mh.rtransform(eq1_1[1], eq2[1], eq1_1[2][:4], eq2[2][:4], eq1_1[2][-1], eq2[2][-1]),
                  mh.fade_replace(eq1_1[2][-2], eq2[2][-2]),
                  run_time=1.6)
        self.wait()


class ExpectedXY(Scene):
    fs1 = 100
    def construct(self):
        cx = RED
        cy = BLUE
        MathTex.set_default(font_size=self.fs1)
        eq1 = MathTex(r'\mathbb E[Y\vert X]', r'=', r'\int p(y\vert X)y\,dy')
        eq1[0][2].set_color(cy)
        eq1[0][4].set_color(cx)
        eq1[2][3].set_color(cy)
        eq1[2][5].set_color(cx)
        eq1[2][7].set_color(cy)
        eq1[2][9].set_color(cy)
        eq1_1 = eq1[2][1:7].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.add(eq1_1)
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq1_1, eq1[2][1:7], run_time=1.6),
                  FadeIn(eq1[:2], eq1[2][0], eq1[2][7:], run_time=1.6),
                    lag_ratio=0.5))
        self.wait(0.1)
        eq2 = MathTex(r'\mathbb E[Y\vert X]', r'=', r'2X')
        eq2[0][2].set_color(cy)
        eq2[0][4].set_color(cx)
        eq2[2][1].set_color(cx)
        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)
        eq2_1 = eq2[2].copy()
        mh.align_sub(eq2_1, eq2_1[1], eq1[2][5], coor_mask=RIGHT)
        self.play(FadeOut(eq1[2][:5], eq1[2][6:]),
                  FadeIn(eq2_1[0]), mh.rtransform(eq1[2][5], eq2_1[1]),
                  run_time=1.4)
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq2_1, eq2[2]),
                 run_time=1.5)
        self.wait(0.1)
        eq3 = MathTex(r'\mathbb E[X\vert Y]', r'=', r'2Y')
        eq3[0][2].set_color(cx)
        eq3[0][4].set_color(cy)
        eq3[2][1].set_color(cy)
        eq3.next_to(eq2, DOWN, buff=0.8)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=RIGHT)
        eq2_1 = eq2.copy()
        gp = VGroup(eq2.copy(), eq3).align_to(eq1, UP).shift(DOWN*0.2)
        self.play(mh.rtransform(eq2_1[0][:2], eq3[0][:2], eq2_1[0][3], eq3[0][3], eq2_1[0][5], eq3[0][5],
                                eq2_1[1], eq3[1], eq2_1[2][0], eq3[2][0]),
                  mh.transform(eq2, gp[0]),
                  mh.fade_replace(eq2_1[0][2], eq3[0][2]),
                  mh.fade_replace(eq2_1[0][4], eq3[0][4]),
                  mh.fade_replace(eq2_1[2][1], eq3[2][1]),
                  run_time=1.6
                  )


        return
        eq3 = MathTex(r'\mathbb E[Y\vert X]', r'=', r'2X')
        eq3[0][2].set_color(cy)
        eq3[0][4].set_color(cx)
        eq3[2][1].set_color(cx)
        gp = VGroup(eq1.copy(), eq2.copy(), eq3)
        gp[1].next_to(gp[0], RIGHT, buff=0.6)
        gp[2].next_to(gp[:2], DOWN, buff=0.8)
        gp.move_to(ORIGIN)
        eq3_1 = eq3[0].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(LaggedStart(mh.transform(eq1, gp[0], eq2, gp[1], run_time=2),
                  FadeIn(eq3_1, run_time=2),
                              lag_ratio=0.3))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq3_1, eq3[0], run_time=1.4),
                  FadeIn(eq3[1:], run_time=1.4), lag_ratio=0.2))
        self.wait(0.1)
        eq4 = MathTex(r'\mathbb E[X\vert Y]', r'=', r'2Y')
        eq4[0][2].set_color(cx)
        eq4[0][4].set_color(cy)
        eq4[2][1].set_color(cy)
        mh.align_sub(eq4, eq4[1], eq3[1]).next_to(eq3, DOWN, coor_mask=UP, buff=0.4)
        gp1 = VGroup(eq1, eq2, eq3)
        gp = VGroup(gp1.copy(), eq4).move_to(ORIGIN)
        eq3_1 = eq3.copy()
        self.play(mh.transform(gp1, gp[0]),
                  mh.rtransform(eq3_1[0][:2], eq4[0][:2], eq3_1[0][3], eq4[0][3], eq3_1[0][5], eq4[0][5],
                                eq3_1[1], eq4[1], eq3_1[2][0], eq4[2][0]),
                  mh.fade_replace(eq3_1[0][2], eq4[0][2]),
                  mh.fade_replace(eq3_1[0][4], eq4[0][4]),
                  mh.fade_replace(eq3_1[2][1], eq4[2][1]),
                  run_time=2)


        self.wait()


class Head(Scene):
    def construct(self):
        self.add(H.scale(2))

class Tail(Scene):
    def construct(self):
        self.add(T.scale(2))

class Alicea(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    eq_str = r'a'
    eq_col = RED

    def construct(self):
        eq1 = MathTex(self.eq_str, font_size=120, stroke_width=6, color=self.eq_col).set_z_index(1)
        eq2 = MathTex(self.eq_str, font_size=120, stroke_width=12, stroke_color=BLACK).set_z_index(0)
        self.add(eq1, eq2)

class Bobb(Alicea):
    eq_str = r'b'
    eq_col = BLUE

class AliceExpected(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=80)
        eq1 = MathTex(r'p(b)', r'=', r'\left(\frac12\right)^{b+1}')
        eq1_1 = eq1.copy()
        eq2 = Tex(r'\sf Alice receives ', r'$4^b$')

        for x in eq2[0][:5]: x.set(stroke_width=2)
        eq2.next_to(eq1, DOWN, buff=0.)
        eq3 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\sum_b4^bp(b)')
        mh.align_sub(eq3, eq3[0][2:-3], eq2[0], coor_mask=UP)
        eq4 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\sum_{b=0}^{a-1}4^b\left(\frac12\right)^{b+1}')
        eq4.move_to(VGroup(eq1, eq2, eq3), coor_mask=UP)
        VGroup(eq4[0][-2]).set_color(RED)
        eq4_1 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\sum_b4^b')
        mh.align_sub(eq4_1, eq4_1[1], eq4[1])
        eq5 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\sum_{b=0}^{a-1}\left(\frac42\right)^b\frac12')
        eq6 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\frac12\sum_{b=0}^{a-1}2^b')
        mh.align_sub(eq6, eq6[1], eq4[1])
        mh.align_sub(eq5, eq5[1], eq4[1])
        eq7 = MathTex(r'\mathbb E[{\rm Alice receives}\vert a]', r'=', r'\frac12\left(2^a-1\right)')
        mh.align_sub(eq7, eq7[1], eq4[1], coor_mask=UP)
        VGroup(eq5[0][-2], eq5[2][0], eq6[0][-2], eq6[2][3], eq7[0][-2], eq7[2][5],
               eq3[0][-2], eq2[0][:5], eq4[2][0]).set_color(RED)
        VGroup(eq5[2][-4], eq5[2][4], eq6[2][-1], eq6[2][7], eq4[2][4], eq4[2][8], eq4[2][-3], eq4_1[2][1],
               eq3[2][-2], eq3[2][-5], eq3[2][-7], eq2[1][-1], eq1[2][-3], eq1[0][2]).set_color(BLUE)
        for _ in VGroup(eq3, eq4, eq4_1, eq5, eq6, eq7): _[0][-2].set(stroke_width=2)
        eqs = VGroup(eq1, eq2, eq3, eq4, eq4_1, eq5, eq6, eq7).set_z_index(1)

        box = SurroundingRectangle(eqs, corner_radius=0.2, fill_color=BLACK, fill_opacity=0.85, stroke_width=0)

        self.add(eq1, box)
        self.wait(0.1)
        self.play(FadeIn(eq2), run_time=1)
        self.wait(0.1)
        self.play(LaggedStart(AnimationGroup(eq2[0].animate.move_to(eq3[0][2:-3]),
                  mh.rtransform(eq2[1][:], eq3[2][2:4]), run_time=1.2),
                  FadeIn(eq3[0][:2], eq3[0][-3:], eq3[1], eq3[2][:2], eq3[2][4:], run_time=1.2),
                              lag_ratio=0.5)
                  )
        self.wait(0.1)
        self.play(FadeOut(eq1[:2]),
                  AnimationGroup(mh.rtransform(eq1[2][:], eq4[2][-8:], eq3[2][:2], eq4_1[2][:2],
                                eq3[2][2:4], eq4[2][7:9], eq3[1], eq4[1],
                                eq3[0][:2], eq4[0][:2], eq3[0][-3:], eq4[0][-3:]),
                  eq2[0].animate.move_to(eq4[0][2:-3]),
                  FadeOut(eq3[2][-4:], shift=mh.diff(eq3[2][2], eq4[2][7]) * UP), run_time=1.4))
        self.wait(0.1)
        self.play(mh.rtransform(eq4_1[2][:2], eq4[2][3:5]),
                  FadeIn(eq4[2][:3], eq4[2][5:7]))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[2][:7], eq5[2][:7], eq4[2][7], eq5[2][8],
                                eq4[2][8], eq5[2][12], eq4[2][9], eq5[2][7],
                                eq4[2][13], eq5[2][11], eq4[2][11:13].copy(), eq5[2][9:11],
                                eq4[2][10:13], eq5[2][13:], eq4[1], eq5[1],
                                eq4[0][:2], eq5[0][:2], eq4[0][-3:], eq5[0][-3:]),
                  mh.rtransform(eq4[2][14], eq5[2][12]),
                  FadeOut(eq4[2][15:]),
                  eq2[0].animate.move_to(eq5[0][2:-3])
                  )
        self.wait(0.1)
        eq6_1 = mh.align_sub(eq6[2][10:12].copy(), eq6[2][10], eq5[2][9], coor_mask=RIGHT)
        self.play(FadeOut(eq5[2][7], eq5[2][11], eq5[2][9]),
                  eq5[2][12].animate.move_to(eq6_1[1]),
                  eq5[2][10].animate.move_to(eq6_1[0]),
                  FadeOut(eq5[2][8], target_position=eq6_1[0]))
        self.play(mh.rtransform(eq5[2][:7], eq6[2][3:10], eq5[2][10], eq6[2][10],
                                eq5[2][12], eq6[2][11], eq5[2][-3:], eq6[2][:3]), run_time=1.2)
        self.wait(0.1)
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq6[2][:3], eq7[2][:3], eq6[2][10], eq7[2][4],
                                eq6[2][3], eq7[2][5], eq5[1], eq7[1], eq5[0][:2], eq7[0][:2], eq5[0][-3:], eq7[0][-3:]),
                  eq2[0].animate.move_to(eq7[0][2:-3]),
                  FadeOut(eq6[2][4:10]),
                  FadeOut(eq6[2][11], target_position=eq7[2][5]),
                    run_time=1.6),
                  FadeIn(eq7[2][3], eq7[2][6:], run_time=1.4), lag_ratio=0.5))
        self.wait(0.5)

        eq7_1 = eq7.copy().align_to(eq1_1, UP).shift(DOWN*0.05)

        eq10 = MathTex(r'\mathbb P(b > a\vert a)', r'=', r'\left(\frac12\right)^{a+1}')
        eq10[0][2].set_color(BLUE)
        VGroup(eq10[0][4], eq10[0][6], eq10[2][-3]).set_color(RED)
        eq10.next_to(eq7_1, DOWN, buff=0.05)

        eq11 = MathTex(r'\mathbb E[{\sf Alice\ pays}\vert a]', r'=', r'4^a\left(\frac12\right)^{a+1}')
        mh.align_sub(eq11, eq11[1], eq10[1], coor_mask=UP)
        VGroup(eq11[2][1], eq11[2][-3]).set_color(RED)
        eq12 = MathTex(r'\mathbb E[{\sf Alice\ pays}\vert a]', r'=', r'\frac12\left(\frac42\right)^a')
        mh.align_sub(eq12, eq12[1], eq11[1], coor_mask=UP)
        VGroup(eq12[0][2:7], eq12[0][-2], eq12[2][-1]).set_color(RED)
        eq13 = MathTex(r'\mathbb E[{\sf Alice\ pays}\vert a]', r'=', r'\frac122^a')
        VGroup(eq13[0][2:7], eq13[0][-2], eq13[2][-1]).set_color(RED)
        mh.align_sub(eq13, eq13[1], eq12[1], coor_mask=UP)
        reds = [_[0][-2] for _ in [eq10, eq11, eq12, eq13]]
        for _ in [eq11, eq12, eq13]: reds += _[0][2:7]
        for _ in reds:
            _.set(stroke_width=2).set_color(RED)
        VGroup(eq7_1, eq10, eq11, eq12, eq13).set_z_index(1)

        self.play(mh.transform(eq7[1:], eq7_1[1:], eq7[0][:2], eq7_1[0][:2], eq7[0][-3:], eq7_1[0][-3:]),
                  eq2[0].animate.move_to(eq7_1[0][2:-3]), FadeIn(eq10[0]))
        self.wait(0.1)
        self.play(FadeIn(eq10[1:]))
        self.wait(0.1)
        self.play(mh.fade_replace(eq10[0][0], eq11[0][0]),
                  mh.stretch_replace(eq10[0][1], eq11[0][1]),
                  mh.stretch_replace(eq10[0][-1], eq11[0][-1]),
                  mh.rtransform(eq10[1], eq11[1], eq10[0][-3:-1], eq11[0][-3:-1],
                                eq10[2][:], eq11[2][2:]),
                  FadeIn(eq11[2][:2], shift=mh.diff(eq10[2][0], eq11[2][2]) * RIGHT),
                  FadeOut(eq10[0][2:-3], target_position=eq11[0][2:-3]),
                  FadeIn(eq11[0][2:-3]))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[:2], eq12[:2], eq11[2][0], eq12[2][4],
                                eq11[2][1], eq12[2][8], eq11[2][2], eq12[2][3],
                                eq11[2][6], eq12[2][7], eq11[2][3:6], eq12[2][:3],
                                eq11[2][4:6].copy(), eq12[2][5:7]),
                  mh.rtransform(eq11[2][-3], eq12[2][-1]),
                  FadeOut(eq11[2][-2:], shift=mh.diff(eq11[2][-3], eq12[2][-1])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(eq12[:2], eq13[:2], eq12[2][-1], eq13[2][-1], eq12[2][:3], eq13[2][:3],
                                eq12[2][6], eq13[2][3]),
                  FadeOut(eq12[2][5], eq12[2][3], eq12[2][7]),
                  FadeOut(eq12[2][4], target_position=eq13[2][3]))
        self.wait(0.1)

        eq20 = MathTex(r'\mathbb E[{\sf Alice\ profit}\vert a]', r'=', r'\frac12(2^a-1) - \frac122^a').set_z_index(1)
        eq20[2][9:].next_to(eq20[2][3:9], DOWN, buff=0.1)
        eq20.move_to(ORIGIN)
        mh.align_sub(eq20, eq20[1], eq7[1], coor_mask=UP)
        VGroup(eq20[0][2:7], eq20[0][-2], eq20[2][5], eq20[2][-1]).set_color(RED)
        for x in eq20[0][2:7]: x.set(stroke_width=2)

        eq21 = MathTex(r'\mathbb E[{\sf Alice\ profit}\vert a]', r'=', r'-\frac12').set_z_index(1)
        VGroup(eq21[0][2:7], eq21[0][-2]).set_color(RED)
        for x in eq21[0][2:7]: x.set(stroke_width=2)
        for _ in (eq20, eq21): _[0][-2].set(stroke_width=2)

        eq20_1 = eq20[0][7:-3].copy()
        self.play(mh.rtransform(eq2[0][:5], eq20[0][2:7], eq7[0][:2], eq20[0][:2], eq7[0][-3:], eq20[0][-3:],
                                eq7[1], eq20[1], eq7[2][:], eq20[2][:9], eq13[2][:], eq20[2][10:]),
                  mh.rtransform(eq13[0][:7], eq20[0][:7], eq13[0][-3:], eq20[0][-3:]),
                  mh.fade_replace(eq13[0][7:-3], eq20[0][7:-3]),
                  mh.fade_replace(eq13[1][0], eq20[2][9]),
                  mh.fade_replace(eq2[0][5:], eq20_1)
                  )
        self.remove(eq20_1)
        del1 = eq20[2][4:6]
        del2 = eq20[2][-2:]
        line1 = Line(del1.get_corner(DL), del1.get_corner(UR), stroke_width=7, stroke_color=GREEN).set_z_index(5)
        line2 = Line(del2.get_corner(DL), del2.get_corner(UR), stroke_width=7, stroke_color=GREEN).set_z_index(5)
        self.play(Create(line1, rate_func=linear, run_time=0.5))
        self.play(Create(line2, rate_func=linear, run_time=0.5))
        self.play(FadeOut(eq20[2][4:6], eq20[2][-6:], line1, line2))
        self.wait(0.1)
        self.play(mh.rtransform(eq20[:2], eq21[:2], eq20[2][:3], eq21[2][1:4], eq20[2][6], eq21[2][0]),
                  mh.rtransform(eq20[2][7], eq21[2][1]),
                  FadeOut(eq20[2][8]),
                  FadeOut(eq20[2][3], shift=mh.diff(eq20[2][1], eq21[2][2]) * RIGHT),
                  run_time=1.5)
        self.wait(0.1)
        eq30 = MathTex(r'\mathbb E[{\sf Bob\ profit}\vert b]', r'=', r'-\frac12').set_z_index(1)
        blues = eq30[0][2:5] + [eq30[0][-2]]
        for _ in blues: _.set_color(BLUE).set(stroke_width=2)
        mh.align_sub(eq30, eq30[1], eq21[1]).next_to(eq21, DOWN, coor_mask=UP)
        self.play(FadeIn(eq30), run_time=1)
        self.wait(0.1)

        eq40 = MathTex(r'{\sf Alice\ profit}', r'=', r'W')
        eq41 = MathTex(r'{\sf Bob\ profit}', r'=', r'-W')
        eq42 = MathTex(r'\mathbb E[W\vert a]', r'=-\frac12')
        eq43 = MathTex(r'\mathbb E[W\vert b]', r'=\frac12')
        for _ in eq40[0][:5] + eq42[0][-2]: _.set_color(RED).set(stroke_width=2)
        for _ in eq41[0][:3] + eq43[0][-2]: _.set_color(BLUE).set(stroke_width=2)
        eq41.next_to(eq40, RIGHT, buff=0.7)
        mh.align_sub(VGroup(eq40, eq41), eq40[1], eq21[1]).move_to(ORIGIN, coor_mask=RIGHT).shift(DOWN*0.3)
        eq42.next_to(eq40, DOWN)
        eq43.next_to(eq41, DOWN)
        mh.align_sub(eq43, eq43[1][0], eq42[1][0], coor_mask=UP)
        eqs = VGroup(eq40, eq41, eq42, eq43).set_z_index(1)
        box2 = SurroundingRectangle(eqs, corner_radius=0.2, fill_color=BLACK, fill_opacity=0.85, stroke_width=0,
                                    buff=0.2)


        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq21[0][2:12], eq40[0][:], eq30[0][2:10], eq41[0][:],
                                                           box, box2),
                  FadeOut(eq21[0][:2], eq21[0][-3:], eq21[1:], eq30[0][:2], eq30[0][-3:], eq30[1:]),
                                 run_time=1.6),
                  FadeIn(eq40[1:], eq41[1:], run_time=1.2), lag_ratio=0.5))
        self.wait(0.1)
        self.play(FadeIn(eq42))
        self.wait(0.1)
        self.play(FadeIn(eq43))


        self.wait()

class HalfMil(Scene):
    def construct(self):
        MathTex.set_default(font_size=100)
        eq1 = MathTex(r'\frac12(\$200\,000+\$800\,000)')[0].set_z_index(1)
        eq2 = MathTex(r'=\$500\,000').set_z_index(1)
        eq2.next_to(eq1[3:], DOWN).align_to(eq1[8], LEFT)
        box = SurroundingRectangle(VGroup(eq1, eq2), stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                   fill_opacity=0.7, buff=0.2, corner_radius=0.2)
        self.add(eq1, box)
        self.wait(0.1)
        self.play(FadeIn(eq2))
        self.wait()

class Gain25(Scene):
    def __init__(self, *args, **kwargs):
        if not config.transparent:
            config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=80)
        eq1 = MathTex(r'{\rm Expected\ gain} ', r'=', r'\frac12A+\frac12\left(-\frac A2\right)').set_z_index(1)
        eq2 = MathTex(r'{\rm Expected\ gain} ', r'=', r'\frac12A-\frac14A').set_z_index(1)
        eq3 = MathTex(r'{\rm Expected\ gain} ', r'=', r'\frac14A').set_z_index(1)
        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)
        mh.align_sub(eq3, eq3[1], eq1[1], coor_mask=UP)
        VGroup(eq1[2][3], eq1[2][-4], eq2[2][3], eq2[2][-1], eq3[2][-1]).set_color(RED)
        box1 = SurroundingRectangle(VGroup(eq1, eq2, eq3), fill_color=BLACK, fill_opacity=0.7, corner_radius=0.2,
                                    stroke_width=0, stroke_opacity=0, buff=0.2)
        self.add(eq1[0], eq1[1], box1)
        self.wait(0.1)
        self.play(FadeIn(eq1[2][:4]))
        self.wait(0.1)
        self.play(FadeIn(eq1[2][4:]))
        self.wait(0.1)
        eq2_1 = eq2[2][7].copy()
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq1[2][:4], eq2[2][:4],
                                eq1[2][9], eq2[2][4], eq1[2][10], eq2[2][-1],
                                eq1[2][5:7], eq2[2][5:7]),
                  mh.fade_replace(eq1[2][11], eq2[2][6]),
                  FadeOut(eq1[2][4], target_position=eq2[2][4]),
                  mh.fade_replace(eq1[2][7], eq2[2][7]),
                  mh.fade_replace(eq1[2][12], eq2_1),
                  FadeOut(eq1[2][8], eq1[2][-1]),
                  )
        self.remove(eq2_1)
        self.wait(0.1)
        eq3_1 = eq3[2][2].copy()
        self.play(mh.rtransform(eq2[:2], eq3[:2], eq2[2][5:], eq3[2][:4]),
                  mh.rtransform(eq2[2][:2], eq3[2][:2], eq2[2][3], eq3[2][3]),
                  mh.fade_replace(eq2[2][2], eq3_1),
                  FadeOut(eq2[2][4], target_position=eq3[2][1]),
                  run_time=1.4
                  )
        self.remove(eq3_1)
        self.wait()

class DifferentCalc(Gain25):
    def construct(self):
        MathTex.set_default(font_size=100)
        eq1 = MathTex(r'\mathbb E[B\vert A]', r'>', r'A', r'\textsf{\textit{switch!}}').set_z_index(1)
        eq2 = MathTex(r'\mathbb E[A\vert B]', r'>', r'B', r"\textsf{\textit{stick!}}").set_z_index(1)
        eq3 = MathTex(r'\mathbb E[B\vert A+B]', r'=', r'\mathbb E[A\vert A+B]').set_z_index(1)

        VGroup(eq1[0][-2], eq1[2][0], eq2[0][2], eq3[0][-4], eq3[2][2], eq3[2][-4]).set_color(RED)
        VGroup(eq1[0][2], eq2[0][-2], eq2[2][0], eq3[0][2], eq3[0][-2], eq3[2][-2]).set_color(BLUE)

        eq1[:-1].move_to(ORIGIN).shift(LEFT)
        eq1[-1].next_to(eq1[:-1], RIGHT, buff=0.6, coor_mask=RIGHT).set_color(YELLOW)
        mh.align_sub(eq2, eq2[:-1], eq1[:-1]).next_to(eq1, DOWN, coor_mask=UP)
        eq2[-1].next_to(eq2[:-1], RIGHT, buff=0.6, coor_mask=RIGHT).set_color(YELLOW)
        gp1 = VGroup(eq1.copy(), eq2).move_to(ORIGIN, coor_mask=UP)
        eq3.next_to(gp1, DOWN, coor_mask=UP)
        gp2 = VGroup(gp1[0].copy(), gp1[1].copy(), eq3).move_to(ORIGIN, coor_mask=UP)
        box1 = SurroundingRectangle(VGroup(gp1, gp2), fill_color=BLACK, fill_opacity=0.7, corner_radius=0.2,
                                    buff=0.2, stroke_width=0, stroke_opacity=0)

        self.add(eq1[:-1], box1)
        self.wait(0.1)
        self.play(FadeIn(eq1[-1]))
        eq1_1 = eq1.copy()
        self.play(mh.transform(eq1, gp1[0]),
                  mh.rtransform(eq1_1[0][:2], eq2[0][:2], eq1_1[0][3], eq2[0][3],
                                eq1_1[0][-1], eq2[0][-1]),
                  mh.fade_replace(eq1_1[0][2], eq2[0][2]),
                  mh.fade_replace(eq1_1[0][-2], eq2[0][-2]))
        self.wait(0.1)
        self.play(FadeIn(eq2[1:-1]))
        self.wait(0.1)
        self.play(FadeIn(eq2[-1]))
        self.wait(0.1)
        eq2_1 = eq2.copy()
        eq3_1 = eq3[0].copy().move_to(eq2[0], coor_mask=RIGHT)
        self.play(mh.transform(eq1, gp2[0], eq2, gp2[1]),
                  mh.rtransform(eq2_1[0][:2], eq3_1[:2], eq2_1[0][-2:], eq3_1[-2:], eq2_1[0][-3], eq3_1[-5]),
                  mh.fade_replace(eq2[0][-2].copy(), eq3_1[-4]),
                  mh.fade_replace(eq2[0][2].copy(), eq3_1[2]),
                  FadeIn(eq3_1[-3], target_position=eq2[0][-2]),
                  run_time=1.2)
        self.wait(0.1)
        eq3_2 = eq3_1.copy()
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq3_1, eq3[0], eq3_2[:2], eq3[2][:2], eq3_2[3:], eq3[2][3:]),
                  mh.fade_replace(eq3_2[2], eq3[2][2]), run_time=1.4),
                  FadeIn(eq3[1]), lag_ratio=0.5))
        self.wait(0.1)
        eq4 = Tex(r"\sf doesn't matter", stroke_width=2, color=YELLOW)[0].move_to(VGroup(eq1, eq2), coor_mask=UP)
        self.play(FadeOut(eq1, eq2), FadeIn(eq4))
        self.wait()

class MoneyDist(Gain25):
    def construct(self):
        xmax = 2.5
        ax = Axes(x_range=[0, xmax * 1.1], y_range=[0, 1.4], x_length=8, y_length=4,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "stroke_opacity": 1,
                               },
                  ).set_z_index(2)
        def p(x):
            if x > 0.001:
                y = math.log(x)
                p1 = math.exp(-y*y/2 * 5)/x
            else: p1 = 0.
            return p1
        plt1 = ax.plot(p, (0, xmax), stroke_color=BLUE, stroke_width=6).set_z_index(4)
        fill1 = ax.get_area(plt1, color=BLUE, opacity=0.5).set_z_index(3).set_stroke(opacity=0)

        label1 = Tex(r'\sf\$').next_to(ax.x_axis.get_right(), UL, buff=0.2)
        label2 = MathTex(r'p').next_to(ax.y_axis.get_top(), DR, buff=0.2)

        self.add(ax, label1, label2)
        self.wait(0.1)
        self.play(LaggedStart(Create(plt1, run_time=1.5, rate_func=linear),
                              FadeIn(fill1, run_time=1, rate_func=linear),
                              lag_ratio=0.5))
        self.wait(0.1)
        MathTex.set_default(font_size=80)
        pt1 = ax.coords_to_point(2.4, 0)
        pt1_1 = ax.coords_to_point(2.4, p(2.4))
        eq1 = MathTex(r'A', color=RED, stroke_width=2).set_z_index(10)
        eq1.move_to(pt1, aligned_edge=UP).shift(DOWN * 0.1)
        eq1 = VGroup(eq1, eq1.copy().set_color(WHITE).set(stroke_width=7).set_z_index(9))
        line1 = Line(pt1, pt1_1, stroke_color=GREY, stroke_width=5).set_z_index(5)

        pt2 = ax.coords_to_point(1.2, 0)
        pt2_1 = ax.coords_to_point(1.2, p(1.2))
        eq2 = MathTex(r'B', color=BLUE, stroke_width=2).set_z_index(10)
        eq2.move_to(pt2, aligned_edge=UP).shift(DOWN * 0.1)
        eq2 = VGroup(eq2, eq2.copy().set_color(WHITE).set(stroke_width=7).set_z_index(9))
        line2 = Line(pt2, pt2_1, stroke_color=GREY, stroke_width=5).set_z_index(5)

        eq3 = Tex(r'stick', color=YELLOW, font_size=100, stroke_width=2)
        eq3.move_to(ax.coords_to_point(1.05, 0.3)).set_z_index(10)

        pt3 = ax.coords_to_point(0.3, 0)
        pt3_1 = ax.coords_to_point(0.3, p(0.3))
        eq4 = eq1.copy().move_to(pt3, coor_mask=RIGHT)
        pt4 = ax.coords_to_point(0.6, 0)
        pt4_1 = ax.coords_to_point(0.6, p(0.6))
        eq5 = eq2.copy().move_to(pt4, coor_mask=RIGHT)
        line3  = Line(pt3, pt3_1, stroke_color=GREY, stroke_width=5).set_z_index(5)
        line4  = Line(pt4, pt4_1, stroke_color=GREY, stroke_width=5).set_z_index(5)
        eq6 = Tex(r'switch', color=YELLOW, font_size=100, stroke_width=2)
        eq6.move_to(eq3).set_z_index(10)

        self.play(FadeIn(eq1, line1))
        self.wait(0.1)
        self.play(FadeIn(eq2, target_position=eq1), FadeIn(line2))
        self.wait(0.1)
        self.play(FadeIn(eq3))
        self.wait(0.1)
        self.play(mh.rtransform(eq1, eq4), FadeIn(line3), FadeOut(eq2, line1, line2, eq3), run_time=1.5)
        self.wait(0.1)
        self.play(FadeIn(eq5, target_position=eq4), FadeIn(line4))
        self.wait(0.1)
        self.play(FadeIn(eq6))
        self.wait()

class Geometric(Gain25):
    def show_geom(self):
        eq1 = MathTex(r'\$1', r',\$2', r',\$4', r',\$8', r',\$16', r',\ldots\ ', r'\$2^N').set_z_index(2).to_edge(DOWN,
                                                                                                                  buff=0.5)
        box1 = SurroundingRectangle(eq1, corner_radius=0.2, fill_color=BLACK, fill_opacity=0.8, buff=0.2,
                                    stroke_width=0, stroke_opacity=0)
        eq2 = MathTex(r'N', r'\sim', r'{\rm Geometric}(p)').next_to(eq1, UP).set_z_index(2)
        box2 = SurroundingRectangle(VGroup(eq1, eq2), corner_radius=0.2, fill_color=BLACK, fill_opacity=0.8, buff=0.2,
                                    stroke_width=0, stroke_opacity=0)

        self.add(box1)
        for _ in eq1:
            self.play(FadeIn(_), run_time=0.8)
            self.wait(0.05)
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(box1, box2), FadeIn(eq2), lag_ratio=0.3), run_time=1.6)
        self.wait(0.1)

        eq3 = MathTex(r'\mathbb P(N=n)', r'\sim', r'p^n').set_z_index(2).next_to(eq1, UP)
        eq2_1 = eq2.copy().next_to(eq3, UP, coor_mask=UP)
        box3 = SurroundingRectangle(VGroup(eq1, eq2_1), corner_radius=0.2, fill_color=BLACK, fill_opacity=0.8, buff=0.2,
                                    stroke_width=0, stroke_opacity=0)
        self.play(mh.rtransform(box2, box3), mh.transform(eq2, eq2_1),
                  FadeIn(eq3), ruun_time=1.5)
        self.wait(0.1)
        eq4 = MathTex(r'\mathbb P(N=n)', r'=', r'p^n(1-p)').set_z_index(2)
        mh.align_sub(eq4, eq4[0], eq3[0], coor_mask=UP)
        self.play(mh.rtransform(eq3[0], eq4[0], eq3[2][:2], eq4[2][:2]),
                  mh.fade_replace(eq3[1], eq4[1]),
                  FadeIn(eq4[2][2:]))
        self.wait(0.1)
        return box3, eq1, eq2, eq4

    def construct(self):
        MathTex.set_default(font_size=80)
        box1, eq1, eq2, eq3 = self.show_geom()
        self.play(FadeOut(eq1, eq2, eq3))

        eq4 = Tex(r'$N$:\ ', r'$0$').set_z_index(2).move_to(eq2).align_to(eq1, RIGHT).shift(LEFT*0.8).set_z_index(4)
        eq5_1 = MathTex(r'\$01')[0].set_z_index(2).next_to(eq4, DOWN, buff=0.9).set_z_index(4)
        box2 = SurroundingRectangle(eq5_1, fill_opacity=0, stroke_opacity=1, stroke_width=8, stroke_color=WHITE,
                                    buff=0.4).set_z_index(1)

        eq5 = MathTex(r'\$1')[0].move_to(box2).set_z_index(4)
        mh.align_sub(eq5, eq5[0], eq5_1[0], coor_mask=UP)

        self.play(FadeIn(eq4, eq5, box2))

        pos = eq2.get_left() + LEFT * 0.3

        rolls = [5, 1, 4, 6]
        dice = []
        eq4_2 = eq4[1]
        eq5_2 = eq5
        for i in range(len(rolls)):
            dice.append(animate_roll(self, rolls[i], pos=pos + i * RIGHT * 1.4, scale=0.45, right=False, slide=True))
            self.wait(0.1)
            if rolls[i] == 6:
                break
            eq4_1 = Tex(r'${}$'.format(i+1)).set_z_index(2).move_to(eq4[1][0]).align_to(eq4[1][0], LEFT).set_z_index(4)
            eq5_1 = MathTex(r'\${}'.format(2**(i+1)))[0].move_to(box2).set_z_index(4)
            mh.align_sub(eq5_1, eq5_1[0], eq5_2[0])
            self.play(FadeOut(eq4_2), FadeIn(eq4_1),
                      mh.rtransform(eq5_2[0], eq5_1[0]),
                      FadeOut(eq5_2[1:]), FadeIn(eq5_1[1:]))
            eq4_2 = eq4_1
            eq5_2 = eq5_1
            self.wait(0.1)

        eq6 = MathTex(r'\mathbb P(N=n)', r'=', r'\left(\frac56\right)^n\frac16', font_size=60).set_z_index(4)
        eq6.set_z_index(2).align_to(eq1, LEFT).align_to(box2, UP).shift(RIGHT*0.2)
        self.play(FadeIn(eq6))

        self.wait()

class ConditionalCalc(Gain25):
    def construct(self):

        eqh = Tex(r'selected', r'$B$', r'initial cash', r'prob', r'conditional')
        eqh[1].set_color(BLUE)
        eqh0 = eqh.copy()
        eqA = MathTex(r'A', color=RED, font_size=60)
        eqhA = MathTex(r'\frac12A')[0]
        eqhA[-1].scale(1.25, about_edge=LEFT)
        eq2A = MathTex(r'2A', font_size=60)[0]
        #eq2A[-1].scale(1.25, about_edge=LEFT)
        VGroup(eqhA[-1], eq2A[-1]).set_color(RED)
        eqpn = MathTex(r'\sim p^n1')[0]
        eqpnp = MathTex(r'\sim p^np')[0]
        eqp1 = MathTex(r'\frac1{1+p}')[0]
        eqp2 = MathTex(r'\frac p{1+p}')[0]
        t1 = MobjectTable([
            eqh[:],
            [eqA.copy(), eqhA.copy(), eqhA.copy(), eqpn, eqp1],
            [eqA.copy(), eq2A.copy(), eqA.copy(), eqpnp, eqp2]
        ],
            include_outer_lines=True, include_background_rectangle=True,
            h_buff=0.7, v_buff=0.2).to_edge(DOWN, buff=0.2).set_z_index(2)
        t1.background_rectangle.set_opacity(0.7).set_z_index(1)
        mh.align_sub(eqh0, eqh0[3], eqh[3])
        for i in range(len(eqh[:])):
            eqh[i].move_to(eqh0[i], coor_mask=UP)

        self.add(t1.vertical_lines, t1.horizontal_lines, t1.background_rectangle, t1.elements[:5])
        self.wait(0.1)
        self.play(FadeIn(t1.elements[5], t1.elements[10]))
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(t1.elements[5][0].copy(), t1.elements[6][-1]),
                  FadeIn(t1.elements[6][:-1]), lag_ratio=0.5))
        self.play(LaggedStart(mh.rtransform(t1.elements[10][0].copy(), t1.elements[11][-1]),
                  FadeIn(t1.elements[11][:-1]), lag_ratio=0.5))
        self.play(mh.rtransform(t1.elements[6].copy(), t1.elements[7]))
        self.play(mh.rtransform(t1.elements[10].copy(), t1.elements[12]), run_time=1.2)

        eqpn2 = MathTex(r'\sim p^n')[0].set_z_index(2)
        eqpnp2 = MathTex(r'\sim p^{n+1}')[0].set_z_index(2)
        mh.align_sub(eqpn2, eqpn2[0], eqpn[0])
        mh.align_sub(eqpnp2, eqpnp2[0], eqpnp[0])
        self.play(FadeIn(eqpn2, eqpnp2))
        self.play(mh.rtransform(eqpn2[:], eqpn[:-1], eqpnp2[:3], eqpnp[:3], eqpnp2[1].copy(), eqpnp[3]),
                  FadeOut(eqpnp2[3:]),
                  FadeIn(eqpn[-1]))
        eqp1_1 = eqpn[-1].copy()
        eqp2_1 = eqpnp[-1].copy()
        self.play(eqp1_1.animate.move_to(eqp1), eqp2_1.animate.move_to(eqp2))
        self.play(LaggedStart(mh.rtransform(eqp1_1, eqp1[0], eqp2_1, eqp2[0],
                                            eqp1_1.copy(), eqp1[2], eqp1_1.copy(), eqp2[2],
                                            eqp2_1.copy(), eqp1[4], eqp2_1.copy(), eqp2[4],
                                            run_time=1.4),
                  FadeIn(eqp1[1], eqp2[1], eqp1[3], eqp2[3]), lag_ratio=0.3))
        self.wait(0.1)

        t2 = t1.copy().to_edge(DOWN, buff=0).scale(0.8, about_edge=DOWN)
        eq3 = MathTex(r'\mathbb E[B\vert A] - A', r'=', r'A\frac{p}{1+p}-\frac12A\frac1{1+p}', font_size=60).set_z_index(0.5)
        eq3[0][2].set_color(BLUE)
        VGroup(eq3[0][-4], eq3[0][-1], eq3[2][0], eq3[2][10]).set_color(RED)
        eq3.next_to(t2, UP, coor_mask=UP, buff=0.1)
        box2 = SurroundingRectangle(eq3, fill_color=BLACK, fill_opacity=0.7, corner_radius=0.1, buff=0.1,
                                    stroke_width=0, stroke_opacity=0)
        self.play(mh.rtransform(t1, t2), FadeIn(box2, eq3[:2]))
        eq3.set_z_index(2)
        self.play(mh.rtransform(t2.elements[11][1].copy(), eq3[2][0]), run_time=1.4)
        self.play(mh.rtransform(t2.elements[14][:].copy(), eq3[2][1:6]), run_time=1.4)
        self.play(mh.rtransform(t2.elements[6][:].copy(), eq3[2][7:11]),
                  FadeIn(eq3[2][6], shift=mh.diff(t2.elements[6][1], eq3[2][8])),
                  run_time=1.4)
        self.play(mh.rtransform(t2.elements[9][:].copy(), eq3[2][11:]), run_time=1.4)
        self.wait(0.1)
        eq4 = MathTex(r'\mathbb E[B\vert A] - A', r'=', r'A\frac{p-1/2}{1+p}', font_size=60).set_z_index(2)
        eq4[0][2].set_color(BLUE)
        VGroup(eq4[0][-4], eq4[0][-1], eq4[2][0]).set_color(RED)
        mh.align_sub(eq4, eq4[1], eq3[1], coor_mask=UP)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2][0], eq4[2][0],
                                eq3[2][1], eq4[2][1], eq3[2][6], eq4[2][2],
                                eq3[2][7:10], eq4[2][3:6], eq3[2][2:6], eq4[2][6:10]),
                  mh.rtransform(eq3[2][10], eq4[2][0], eq3[2][11], eq4[2][3]),
                  mh.rtransform(eq3[2][12:16], eq4[2][6:10]),
                  run_time=1.5)
        self.wait(0.1)

        MathTex.set_default(font_size=100)
        eq5 = MathTex(r'\mathbb E[B\vert A]-A', r'=', r'A\frac{5/6-1/2}{1+5/6}').set_z_index(2)
        eq5[0][2].set_color(BLUE)
        VGroup(eq5[0][4], eq5[0][-1], eq5[2][0]).set_color(RED)

        box3 = SurroundingRectangle(eq5, fill_color=BLACK, fill_opacity=0.7, corner_radius=0.15, buff=0.2,
                                    stroke_width=0, stroke_opacity=0)
        VGroup(eq5, box3).to_edge(DOWN, buff=0.4)

        self.play(LaggedStart(FadeOut(t2, run_time=1), AnimationGroup(
            mh.rtransform(box2, box3, eq4[:2], eq5[:2], eq4[2][0], eq5[2][0], eq4[2][2:9], eq5[2][4:11]),
            mh.fade_replace(eq4[2][1], eq5[2][1:4]),
            mh.fade_replace(eq4[2][-1], eq5[2][-3:]),
        run_time=1.6), lag_ratio=0.2))

        eq6 = MathTex(r'\mathbb E[B\vert A]-A', r'=', r'A\frac{5-6/2}{6+5}').set_z_index(2)
        mh.align_sub(eq6, eq6[1], eq5[1], coor_mask=UP)
        eq6[0][2].set_color(BLUE)
        VGroup(eq6[0][4], eq6[0][-1], eq6[2][0]).set_color(RED)
        self.play(mh.rtransform(eq5[:2], eq6[:2], eq5[2][:2], eq6[2][:2],
                                eq5[2][4], eq6[2][2], eq5[2][3], eq6[2][3],
                                eq5[2][6:9], eq6[2][4:7], eq5[2][-1], eq6[2][7],
                                eq5[2][10:12], eq6[2][8:10]),
                  FadeOut(eq5[2][2], shift=mh.diff(eq5[2][1], eq6[2][1])),
                  FadeOut(eq5[2][5], target_position=eq6[2][3]),
                  FadeOut(eq5[2][9], target_position=eq6[2][7]),
                  FadeOut(eq5[2][-2], shift=mh.diff(eq5[2][-3], eq6[2][9])))
        eq6_1 = MathTex(r'3').set_z_index(2).move_to(eq6[2][3:6]).align_to(eq5[2][3], DOWN)
        self.play(FadeOut(eq6[2][3:6]), FadeIn(eq6_1))

        eq7 = MathTex(r'\mathbb E[B\vert A]-A', r'=', r'A\frac{2}{11}').set_z_index(2)
        eq7[0][2].set_color(BLUE)
        VGroup(eq7[0][4], eq7[0][-1], eq7[2][0]).set_color(RED)
        mh.align_sub(eq7, eq7[1], eq6[1])
        self.play(mh.rtransform(eq6[:2], eq7[:2], eq6[2][0], eq7[2][0],
                                eq6[2][6], eq7[2][2]),
                  FadeOut(eq6[2][1], target_position=eq7[2][1]),
                  FadeOut(eq6[2][2], target_position=eq7[2][1]),
                  FadeOut(eq6_1, target_position=eq7[2][1]),
                  FadeIn(eq7[2][1], eq7[2][3:5]),
                  FadeOut(eq6[2][7], target_position=eq7[2][3:5]),
                  FadeOut(eq6[2][8], target_position=eq7[2][3:5]),
                  FadeOut(eq6[2][9], target_position=eq7[2][3:5]),
                  )
        eq8 = MathTex(r'\mathbb E[B\vert A]-A', r'=', r'\frac{2}{11}A').set_z_index(2)
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=UP)
        eq8[0][2].set_color(BLUE)
        VGroup(eq8[0][4], eq8[0][-1], eq8[2][-1]).set_color(RED)
        self.play(mh.rtransform(eq7[:2], eq8[:2], eq7[2][0], eq8[2][-1], eq7[2][1:], eq8[2][:-1]))

        eq9 = MathTex(r'\mathbb E[B\vert A] > A')[0].set_z_index(2)
        mh.align_sub(eq9, eq9[0], eq8[0][0], coor_mask=UP)
        eq9[2].set_color(BLUE)
        VGroup(eq9[4], eq9[-1]).set_color(RED)
        self.wait(0.1)
        self.play(mh.rtransform(eq8[0][:6], eq9[:6], eq8[0][-1], eq9[7]),
                  mh.fade_replace(eq8[0][6], eq9[6]),
                  FadeOut(eq8[1:]))
        self.wait(0.1)
        eq10 = MathTex(r'\mathbb E[B\vert A] > A', font_size=100)[0].set_z_index(2)
        eq11 = MathTex(r'\mathbb E[A\vert B] > B', font_size=100)[0].set_z_index(2).next_to(eq10, DOWN)
        VGroup(eq10[2], eq11[4], eq11[7]).set_color(BLUE)
        VGroup(eq11[2], eq10[4], eq10[7]).set_color(RED)
        VGroup(eq10, eq11).move_to(box3)
        self.play(mh.rtransform(eq9, eq10, eq9[:2].copy(), eq11[:2], eq9[3].copy(), eq11[3],
                                eq9[5:7].copy(), eq11[5:7]),
                  mh.fade_replace(eq9[2].copy(), eq11[2]),
                  mh.fade_replace(eq9[4].copy(), eq11[4]),
                  mh.fade_replace(eq9[7].copy(), eq11[7]))
        self.wait(0.1)

        eq12 = MathTex(r'\mathbb E[\mathbb E[B\vert A]] > \mathbb E[A]', font_size=100)[0].set_z_index(2)
        eq13 = MathTex(r'\mathbb E[\mathbb E[A\vert B]] > \mathbb E[B]', font_size=100)[0].set_z_index(2)
        VGroup(eq12[4], eq13[6], eq13[12]).set_color(BLUE)
        VGroup(eq13[4], eq12[6], eq12[12]).set_color(RED)
        eq12.move_to(eq10, coor_mask=UP)
        eq13.move_to(eq11, coor_mask=UP)
        self.play(LaggedStart(mh.rtransform(eq10[:6], eq12[2:8], eq10[6], eq12[9], eq10[7], eq12[12],
                                            eq11[:6], eq13[2:8], eq11[6], eq13[9], eq11[7], eq13[12],
                                            run_time=1.4),
                  FadeIn(eq12[:2], eq12[8], eq12[10:12], eq12[13],
                         eq13[:2], eq13[8], eq13[10:12], eq13[13],run_time=1), lag_ratio=0.5))
        self.wait(0.1)
        circ1 = mh.circle_eq(eq13[:9])
        self.play(Create(circ1), run_time=1)
        self.wait(0.1)
        eq14 = MathTex(r'\mathbb E[B] > \mathbb E[A]')[0].set_z_index(2)
        eq15 = MathTex(r'\mathbb E[A] > \mathbb E[B]')[0].set_z_index(2)
        VGroup(eq14[2], eq15[7]).set_color(BLUE)
        VGroup(eq15[2], eq14[7]).set_color(RED)
        eq14.move_to(eq10, coor_mask=UP)
        eq15.move_to(eq11, coor_mask=UP)
        self.play(mh.rtransform(eq13[:2], eq15[:2], eq13[7], eq15[3]),
                  mh.rtransform(eq13[2:5], eq15[:3], eq13[8], eq15[3]),
                  FadeOut(eq13[5:7]))
        self.wait(0.1)
        self.play(mh.rtransform(eq12[:2], eq14[:2], eq12[7], eq14[3], eq13[9:], eq15[4:]),
                  mh.rtransform(eq12[2:5], eq14[:3], eq12[8:], eq14[3:]),
                  FadeOut(circ1, eq12[5:7]))
        self.wait(0.1)
        line1 = Line(eq15.get_corner(DL), eq14.get_corner(UR), stroke_width=8, stroke_color=RED).set_z_index(4)
        line2 = Line(eq14.get_corner(UL), eq15.get_corner(DR), stroke_width=8, stroke_color=RED).set_z_index(4)
        self.play(Create(line1, rate_func=linear, run_time=0.6))
        self.play(Create(line2, rate_func=linear, run_time=0.6))
        self.wait(0.1)
        self.play(FadeOut(line1, line2, eq14, eq15, rate_func=linear))
        eq16 = MathTex(r'\mathbb E[A\vert B]', r'=', r'B + \mathbb E[A\vert B] - B').set_z_index(2)
        VGroup(eq16[0][2], eq16[2][4]).set_color(RED)
        VGroup(eq16[0][4], eq16[2][0], eq16[2][6], eq16[2][9]).set_color(BLUE)
        eq16.move_to(box3)
        self.play(FadeIn(eq16))
        self.wait(0.1)

        eq17 = MathTex(r'\mathbb E[\mathbb E[A\vert B]]', r'\!=\!', r'\mathbb E[B + \mathbb E[A\vert B] - B]').set_z_index(2)
        VGroup(eq17[0][4], eq17[2][6]).set_color(RED)
        VGroup(eq17[0][6], eq17[2][2], eq17[2][8], eq17[2][11]).set_color(BLUE)
        mh.align_sub(eq17, eq17[1], eq16[1], coor_mask=UP)
        box4 = SurroundingRectangle(VGroup(eq5, eq17), fill_color=BLACK, fill_opacity=0.7, corner_radius=0.15, buff=0.2,
                                    stroke_width=0, stroke_opacity=0)

        self.play(LaggedStart(mh.rtransform(eq16[0][:], eq17[0][2:-1], eq16[1], eq17[1],
                                            eq16[2][:],eq17[2][2:-1], box3, box4),
                  FadeIn(eq17[0][:2], eq17[0][-1], eq17[2][:2], eq17[2][-1]), lag_ratio=0.3))
        self.wait(0.1)
        eq18 = MathTex(r'\mathbb E[A]', r'\!=\!', r'\mathbb E[B] + \mathbb E[\mathbb E[A\vert B] - B]').set_z_index(2)
        mh.align_sub(eq18, eq18[1], eq17[1], coor_mask=UP)
        VGroup(eq18[0][2], eq18[2][9]).set_color(RED)
        VGroup(eq18[2][2], eq18[2][11], eq18[2][14]).set_color(BLUE)
        self.wait(0.1)
        self.play(mh.rtransform(eq17[0][:2], eq18[0][:2], eq17[0][-1], eq18[0][-1],
                                ),
                  mh.rtransform(eq17[0][2:5], eq18[0][:3], eq17[0][-2], eq18[0][-1]),
                  mh.rtransform(eq17[1], eq18[1], eq17[2][:3], eq18[2][:3],
                                eq17[2][3], eq18[2][4], eq17[2][4:], eq18[2][7:]),
                  mh.rtransform(eq17[2][:2].copy(), eq18[2][5:7]),
                  FadeOut(eq17[0][5:7]),
                  FadeIn(eq18[2][3], shift=mh.diff(eq17[2][2], eq18[2][2])))
        self.wait(0.1)
        circ2 = mh.circle_eq(eq18[2][5:]).set_z_index(3)
        eq19 = Tex(r'\sf strictly positive', stroke_width=2, stroke_color=YELLOW, font_size=60).set_z_index(4)
        eq19.next_to(circ2, UP, buff=0)
        self.play(LaggedStart(Create(circ2, rate_func=linear, run_time=1),
                              FadeIn(eq19, run_time=1), lag_ratio=0.5))
        self.wait(0.1)
        circ3 = mh.circle_eq(eq18[2][:4]).set_z_index(3)
        eq20 = Tex(r'\sf infinite!', stroke_width=2, stroke_color=YELLOW, font_size=60).set_z_index(4)
        eq20.next_to(circ3, UP, buff=0.1)
        self.play(LaggedStart(Create(circ3, rate_func=linear, run_time=1),
                              FadeIn(eq20, run_time=1), lag_ratio=0.5))
        self.wait(0.1)
        circ4 = mh.circle_eq(eq18[0]).set_z_index(3)
        eq21 = Tex(r'\sf infinite', stroke_width=2, stroke_color=YELLOW, font_size=60).set_z_index(4)
        eq21.next_to(circ4, UP, buff=0.1)
        self.play(LaggedStart(Create(circ4, rate_func=linear, run_time=1),
                              FadeIn(eq21, run_time=1), lag_ratio=0.5))

        self.wait()

class SwitchGTHalf(Gain25):
    def construct(self):
        MathTex.set_default(font_size=100)
        eq1 = MathTex(r'\$2^N').set_z_index(2)
        eq2 = MathTex(r'\mathbb P(N=n)', r'=', 'p^n(1-p)').set_z_index(2)
        eq3 = MathTex(r'\mathbb E[B\vert A]-A', r'=', r'A\frac{p-1/2}{1+p}').set_z_index(2)
        eq3.next_to(eq2, DOWN, buff=0.1)
        eq3[0][2].set_color(BLUE)
        VGroup(eq3[0][4], eq3[0][7], eq3[2][0]).set_color(RED)
        gp = VGroup(eq1, eq2, eq3)
        box1 = SurroundingRectangle(gp, fill_color=BLACK, fill_opacity=0.7, corner_radius=0.2, buff=0.2,
                                    stroke_width=0, stroke_opacity=0)
        VGroup(gp, box1).to_edge(DOWN, buff=0.1)
        eq1.move_to(box1)
        eq2_1 = eq2.copy().move_to(box1)
        eq3_1 = eq3[0].copy().shift(RIGHT)

        self.add(eq1, box1)
        self.wait(0.1)
        self.play(FadeOut(eq1), FadeIn(eq2_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1, eq2), FadeIn(eq3_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq3_1, eq3[0]), FadeIn(eq3[1:]))
        self.wait(0.1)
        circ1 = mh.circle_eq(eq3[2][1:6]).set_z_index(3).shift(DOWN*0.1)
        self.play(Create(circ1, rate_func=linear, run_time=0.8))
        eq4 = MathTex(r'p > 1/2', color=YELLOW, stroke_width=2).set_z_index(4)
        eq4.next_to(circ1, UP, buff=0.05).shift(LEFT*2)
        self.play(FadeOut(eq2), FadeIn(eq4))
        self.wait(0.1)
        self.play(FadeOut(eq4, circ1, eq3))

        eq5 = MathTex(r'\mathbb E[2^N]', r'=', r'\sum_{n=0}^\infty 2^n\mathbb P(N=n)').set_z_index(2)
        eq5.move_to(box1)
        self.play(FadeIn(eq5[0]))
        self.wait(0.1)
        self.play(FadeIn(eq5[1:]))
        self.wait(0.1)
        eq6 = MathTex(r'\mathbb E[2^N]', r'=', r'\sum_{n=0}^\infty 2^np^n(1-p)').set_z_index(2)
        mh.align_sub(eq6, eq6[1], eq5[1], coor_mask=UP)
        self.play(mh.rtransform(eq5[:2], eq6[:2], eq5[2][:7], eq6[2][:7]),
                  FadeOut(eq5[2][7:]),
                  FadeIn(eq6[2][7:]))
        self.wait(0.1)
        eq7 = MathTex(r'\mathbb E[2^N]', r'=', r'(1-p)\sum_{n=0}^\infty (2p)^n').set_z_index(2)
        mh.align_sub(eq7, eq7[1], eq6[1], coor_mask=UP)
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq6[:2], eq7[:2], eq6[2][:5], eq7[2][5:10],
                             eq6[2][-5:], eq7[2][:5], eq6[2][5], eq7[2][11],
                                eq6[2][7], eq7[2][12], eq6[2][6], eq7[2][-1]),
                  mh.rtransform(eq6[2][8], eq7[2][-1]), run_time=1.6),
                  FadeIn(eq7[2][10], eq7[2][13], run_time=1), lag_ratio=0.5),
                  run_time=1.6)
        self.wait(0.1)
        eq8 = MathTex(r'\mathbb E[2^N]', r'=', r'\frac{1-p}{1-2p}').set_z_index(2)
        mh.align_sub(eq8, eq8[1], eq7[1], coor_mask=UP)
        self.play(mh.rtransform(eq7[:2], eq8[:2], eq7[2][1:4], eq8[2][:3],
                                eq7[2][11:13], eq8[2][6:8]),
                  FadeOut(eq7[2][0], eq7[2][4], shift=mh.diff(eq7[2][1:4], eq8[2][:3])),
                  FadeOut(eq7[2][5:10]),
                  FadeOut(eq7[2][10], eq7[2][13:15], shift=mh.diff(eq7[2][11], eq8[2][6])),
                  FadeIn(eq8[2][3:6]),
                  run_time=1.6)
        self.wait(0.1)

        box2 = Rectangle(width=config.frame_x_radius*2, height=config.frame_height*2,
                         fill_opacity=1, fill_color=BLACK, stroke_opacity=0, stroke_width=0)
        self.play(FadeIn(box2, rate_func=linear), eq8.animate.to_edge(DOWN, buff=0.15),
                  run_time=1.4)

        ymax = 20
        ax = Axes(x_range=[0, 0.5 * 1.1], y_range=[0, ymax], x_length=8, y_length=4,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "stroke_opacity": 1,
                               },
                  ).set_z_index(2).next_to(eq8, UP, buff=0.6)
        lab1 = MathTex(r'p', font_size=60).set_z_index(2).next_to(ax.x_axis.get_right(), RIGHT, buff=0.1)
        line1 = DashedLine(ax.coords_to_point(0.5, 0), ax.coords_to_point(0.5, ymax),
                           color=WHITE).set_z_index(1)
        lab2 = MathTex(r'1/2', font_size=40).set_z_index(2)
        lab2.next_to(ax.coords_to_point(0.5, 0), DOWN, buff=0.1)

        self.play(FadeIn(ax, lab1, line1, lab2))

        xvals = np.linspace(0., 0.485, 100)
        xvals = np.linspace(1., 3.37, 1000)
        xvals = 0.5 - 0.5/xvals/xvals/xvals
        yvals = (1 - xvals)/(1-xvals*2)
        plt1 = ax.plot_line_graph(xvals, yvals, line_color=YELLOW, stroke_width=6, add_vertex_dots=False).set_z_index(3)
        self.play(Create(plt1, rate_func=linear, run_time=2))

        self.wait()


class RhoRange(Scene):
    def construct(self):
        self.add(MathTex(r'(0 < \rho < 1)', font_size=80))


class NormalDecomp(Scene):
    def __init__(self, *args, **kwargs):
        if config.transparent:
            config.background_color = WHITE
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=100)
        col3 = GREEN
        eq1 = MathTex(r'B', r'=', r'\rho A + C')
        eq2 = MathTex(r'C', r'=', r'B - \rho A')
        eq3 = MathTex(r'{\rm Cov}(C,A)', r'=', r'{\rm Cov}(B-\rho A, A)')
        eq4 = MathTex(r'{\rm Cov}(C,A)', r'=', r'{\rm Cov}(B,A)-\rho{\rm Cov}(A, A)', font_size=80)
        eq5 = MathTex(r'=', r'\rho1', font_size=80)
        eq6 = MathTex(r'{\rm Cov}(C,A)', r'=', r'0', font_size=80)
        eq7 = Tex(r'$\Rightarrow$ independent', color=YELLOW, font_size=70)
        eq8 = MathTex(r'B\vert_A', r'\sim', r'N(\rho A, {\rm Var} C)')
        eq9 = MathTex(r'{\rm Var}C', r'=', r'{\rm Cov}(B-\rho A, B-\rho A)')
        eq10 = MathTex(r'{\rm Var}C', r'\!=\!', r'{\rm Var}B + \rho^2{\rm Var}A-2\rho{\rm Cov}(A,B)', font_size=80)
        eq11 = MathTex(r'=', r'11\rho', font_size=80)
        eq12 = MathTex(r'{\rm Var}C', r'=', r'1-\rho^2')
        eq13 = MathTex(r'B\vert_A', r'\sim', r'N(\rho A, 1-\rho^2)')

        VGroup(eq1[0], eq2[2][0], eq3[2][4], eq4[2][4], eq8[0][0], eq9[2][4], eq9[2][9],
               eq10[2][3], eq10[2][-2], eq13[0][0]).set_color(BLUE)
        VGroup(eq1[2][1], eq2[2][-1], eq3[0][6], eq3[2][7], eq3[2][9],
               eq4[0][6], eq4[2][6], eq4[2][14], eq4[2][16], eq6[0][6],
               eq8[2][3], eq9[2][-2], eq9[2][-7], eq10[2][10], eq10[2][18],
               eq13[2][3], eq13[0][-1], eq8[0][-1]).set_color(RED)
        VGroup(eq1[2][-1], eq2[0][0], eq3[0][4], eq4[0][4], eq6[0][4],
               eq8[2][-2], eq9[0][-1], eq10[0][-1], eq12[0][-1]).set_color(col3)
        eq2.next_to(eq1, DOWN)
        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=RIGHT)
        eq3.next_to(eq2, DOWN)
        mh.align_sub(eq4, eq4[1], eq3[1], coor_mask=UP)
        eq5 = mh.align_sub(eq5, eq5[0], eq4[1], coor_mask=UP)[1]
        eq5[0].move_to(eq4[2][:8], coor_mask=RIGHT)
        eq5[1].move_to(eq4[2][10:], coor_mask=RIGHT)
        mh.align_sub(eq6, eq6[1], eq4[1], coor_mask=UP)
        mh.align_sub(eq6, eq6[1], eq3[1], coor_mask=RIGHT)
        eq7.next_to(eq6, RIGHT)
        mh.align_sub(eq8, eq8[1], eq1[1], coor_mask=UP)
        mh.align_sub(eq9, eq9[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq10, eq10[1], eq9[1], coor_mask=UP)
        eq11 = mh.align_sub(eq11, eq11[0], eq10[1])[1]
        eq11[0].move_to(eq10[2][:4], coor_mask=RIGHT)
        eq11[1].move_to(eq10[2][7:10], coor_mask=RIGHT)
        eq11[2].move_to(eq10[2][-8:-5], coor_mask=RIGHT)
        mh.align_sub(eq12, eq12[1], eq10[1], coor_mask=UP)
        mh.align_sub(eq13, eq13[1], eq8[1], coor_mask=UP)

        self.add(eq1)
        self.wait(0.1)
        eq1_1 = eq1.copy()
        self.play(mh.rtransform(eq1_1[0][0], eq2[2][0], eq1_1[1], eq2[1],
                                eq1_1[2][:2], eq2[2][-2:], eq1_1[2][-1], eq2[0][0]),
                  mh.fade_replace(eq1_1[2][2], eq2[2][1]),
                  run_time=1.6)
        self.wait(0.1)
        eq2_1 = eq2
        eq2 = eq2_1.copy()
        self.play(LaggedStart(mh.rtransform(eq2[0][0], eq3[0][4], eq2[1], eq3[1],
                                eq2[2][:], eq3[2][4:8], run_time=1.6),
                  FadeIn(eq3[0][:4], eq3[0][-3:], eq3[2][:4], eq3[2][-3:], run_time=1.2, rate_func=linear),
                  lag_ratio=0.5))
        eq2 = eq2_1
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2][:5], eq4[2][:5],
                                eq3[2][5:7], eq4[2][8:10], eq3[2][7:], eq4[2][14:],
                                eq3[2][-3:].copy(), eq4[2][5:8], eq3[2][:4].copy(), eq4[2][10:14]),
                  run_time=1.6)
        self.wait(0.1)
        self.play(FadeOut(eq4[2][:8], eq4[2][10:]), FadeIn(eq5), rate_func=linear)
        eq6_1 = eq6[2][0].copy()
        self.play(mh.rtransform(eq4[:2], eq6[:2]),
                  mh.fade_replace(eq5[0], eq6[2][0]),
                  mh.fade_replace(eq4[2][9], eq6_1),
                  FadeOut(eq5[1], target_position=eq6_1),
                  FadeOut(eq4[2][8], target_position=eq6_1),
                  run_time=1.6)
        self.remove(eq6_1)
        self.wait(0.1)
        self.play(FadeIn(eq7))
        self.wait(0.1)
        self.wait(1)
        self.wait(0.1)
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq1[0], eq8[0][0], eq1[2][:2], eq8[2][2:4], eq1[2][-1], eq8[2][-2]),
                  mh.fade_replace(eq1[1], eq8[1]),
                  FadeIn(eq8[0][1:], shift=mh.diff(eq1[0], eq8[0][0])),
                  mh.fade_replace(eq1[2][2], eq8[2][4], coor_mask=RIGHT), run_time=1.2),
                  FadeIn(eq8[2][:2], eq8[2][5:8], eq8[2][-1], run_time=1), lag_ratio=0.3))
        self.wait(0.1)
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq2[0][0], eq9[0][-1], eq2[1], eq9[1], eq2[2][:], eq9[2][4:8],
                                eq2[2][:].copy(), eq9[2][9:13]),
                  FadeIn(eq9[0][:3], shift=mh.diff(eq2[0][0], eq9[0][-1])), run_time=2),
                  FadeIn(eq9[2][:4], eq9[2][8], eq9[2][-1], run_time=1), lag_ratio=0.4))
        self.wait(0.1)
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq9[:2], eq10[:2], eq9[2][4], eq10[2][3],
                                eq9[2][6], eq10[2][5], eq9[2][7], eq10[2][10],
                                eq9[2][:4], eq10[2][14:18], eq9[2][-1], eq10[2][-1],
                                eq9[2][8], eq10[2][-3], eq9[2][-4], eq10[2][11],
                                eq9[2][11], eq10[2][13], eq9[2][-2], eq10[2][-4],
                                eq9[2][-5], eq10[2][-2]),
                  mh.fade_replace(eq9[2][5], eq10[2][4]),
                  FadeIn(eq10[2][6], shift=mh.diff(eq9[2][6], eq10[2][5])),
                  FadeIn(eq10[2][-10]), run_time=2),
                  FadeIn(eq10[2][:3], eq10[2][7:10], run_time=1.2), lag_ratio=0.3)
                  )
        self.wait(0.1)
        self.play(FadeOut(eq10[2][:4], eq10[2][7:11], eq10[2][-8:]),
                  FadeIn(eq11), rate_func=linear)
        self.wait(0.1)
        eq12_1 = eq12[2][1].copy()
        self.play(mh.rtransform(eq10[:2], eq12[:2], eq11[0], eq12[2][0],
                                eq10[2][5:7], eq12[2][2:], eq10[2][11], eq12[2][1]),
                  mh.rtransform(eq10[2][13], eq12[2][2]),
                  mh.rtransform(eq11[2], eq12[2][2]),
                  mh.fade_replace(eq10[2][4], eq12_1),
                  FadeOut(eq11[1], shift=mh.diff(eq10[2][5], eq12[2][2])),
                  FadeOut(eq10[2][12], shift=mh.diff(eq10[2][13], eq12[2][2])),
                  run_time=2)
        self.remove(eq12_1)
        self.wait(0.1)
        self.play(mh.rtransform(eq8[:2], eq13[:2], eq8[2][:5], eq13[2][:5], eq8[2][-1], eq13[2][-1]),
                  FadeOut(eq8[2][5:-1], eq12[:2]),
                  FadeOut(eq7, eq6),
                  mh.rtransform(eq12[2][:], eq13[2][5:9]))

        self.wait()


def font_size_sub(eq: Mobject, index: int, font_size: float):
    n = len(eq[:])
    eq_1 = eq[index].copy()
    pos = eq.get_center()
    eq[index].set(font_size=font_size).align_to(eq_1, RIGHT)
    eq[index:].align_to(eq_1, LEFT)
    return eq.move_to(pos, coor_mask=RIGHT)


class Integral(NormalDecomp):
    def construct(self):
        fs = 80
        fss = 40
        MathTex.set_default(font_size=fs)
        eq1_1 = MathTex(r'\mathbb E[e^{\frac12B^2}\vert A]', font_size=100)[0]
        eq1 = MathTex(r'\mathbb E[e^{\frac12B^2}\vert A]', r'=', r'\int p_{B\vert A}(x)e^{\frac12x^2}dx')
        eq2 = MathTex(r'p_{B\vert A}(x)', r'=', r'\small\frac1{\sqrt{2\pi(1-\rho^2)}}', r'e^{-\frac{(x-\rho A)^2}{2(1-\rho^2)}')
        eq3 = MathTex(r'\int', r'\frac1{\sqrt{2\pi(1-\rho^2)}}', r'e^{-\frac{(x-\rho A)^2}{2(1-\rho^2)} }', r'e^{\frac12x^2}', r'dx')
        eq4 = MathTex(r'\int', r'\frac1{\sqrt{2\pi(1-\rho^2)}}', r'e^{-\frac{x^2-2x\rho A + \rho^2A^2}{2(1-\rho^2)} }', r'e^{\frac{x^2}{2} }', r'dx')
        eq5 = MathTex(r'\int', r'\frac1{\sqrt{2\pi(1-\rho^2)}}', r'e^{-\frac{x^2-2x\rho A + \rho^2A^2}{2(1-\rho^2)} }', r'e^{\frac{(1-\rho^2)x^2}{2(1-\rho^2)}}', r'dx')
        eq6 = MathTex(r'\int', r'\frac1{\sqrt{2\pi(1-\rho^2)}}', r'e^{-\frac{\rho^2x^2-2x\rho A + \rho^2A^2}{2(1-\rho^2)} }', r'dx')
        eq7 = MathTex(r'e^{-\frac{(\rho x-A)^2 - A^2 + \rho^2A^2}{2(1-\rho^2)} }', r'dx')
        eq8 = MathTex(r'e^{-\frac{(\rho x-A)^2}{2(1-\rho^2)} }', r'e^{\frac{(1-\rho^2)A^2}{2(1-\rho^2)} }', r'dx')
        eq9 = MathTex(r'e^{\frac12A^2}', r'\int', r'\frac1{\sqrt{2\pi(1-\rho^2)}}', r'e^{-\frac{(\rho x-A)^2}{2(1-\rho^2)} }', r'dx')
        eq10 = MathTex(r'e^{-\frac{(x-A/\rho)^2}{2(\rho^{-2}-1)} }', r'dx')
        eq11 = MathTex(r'\frac{\rho^{-1} }{\sqrt{2\pi(\rho^{-2}-1)}}', r'e^{-\frac{(x-A/\rho)^2}{2(\rho^{-2}-1)} }', r'dx')
        eq12 = MathTex(r'\rho^{-1}e^{\frac12A^2}', r'\int', r'\frac{1 }{\sqrt{2\pi(\rho^{-2}-1)}}', r'e^{-\frac{(x-A/\rho)^2}{2(\rho^{-2}-1)} }', r'dx')
        eq13 = MathTex(r'\mathbb E[e^{\frac12B^2}\vert A]', r'=', r'\rho^{-1}e^{\frac12A^2}', font_size=100)
        eq14 = MathTex(r'\mathbb E[\mathbb E[Y\vert A]\vert X]', r'=', r'\rho^{-1}\mathbb E[X\vert X]', font_size=100)
        eq15 = MathTex(r'\mathbb E[Y\vert X]', r'=', r'\rho^{-1}X', font_size=100)
        eq16 = MathTex(r'\mathbb E[X\vert Y]', r'=', r'\rho^{-1}Y', font_size=100)
        VGroup(eq1[0][-2], eq1[2][4], eq2[0][3], eq2[3][6], eq3[2][6], eq4[2][8], eq4[2][12],
               eq5[2][8], eq5[2][12], eq6[2][10], eq6[2][14], eq7[0][6], eq7[0][10], eq7[0][15],
               eq8[0][6], eq8[1][7], eq9[3][6], eq9[0][4], eq10[0][5], eq11[1][5], eq12[0][7], eq12[3][5],
               eq13[0][-2], eq13[2][-2], eq1_1[-2], eq14[0][6], eq14[0][9], eq14[2][5], eq14[2][7],
               eq15[0][-2], eq15[2][-1], eq16[0][2]).set_color(RED)
        VGroup(eq1[0][6], eq1[2][2], eq2[0][1], eq13[0][6], eq1_1[6], eq14[0][4], eq15[0][2],
               eq16[0][-2], eq16[2][-1]).set_color(BLUE)

        eq4[3][1:3].move_to(eq4[2][2:4], coor_mask=UP)
        eq4[3][3].move_to(eq4[2][-8], coor_mask=UP)
        eq4[3][4].move_to(eq4[2][-7], coor_mask=UP)

        for eq, i in ((eq2, 2), (eq3, 1), (eq4, 1), (eq5, 1), (eq6, 1), (eq9, 2), (eq11, 0), (eq12, 2)):
            font_size_sub(eq, i, fss)

        eq2.move_to(ORIGIN).next_to(eq1, DOWN, buff=0)
        eq3.move_to(ORIGIN).next_to(eq1[:2], DOWN, buff=0.2, coor_mask=UP)
        mh.align_sub(eq4, eq4[0], eq3[0]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[0], eq4[0]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq6, eq6[0], eq4[0]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq7, eq7[0][0], eq6[2][0])
        mh.align_sub(eq8, eq8[0][0], eq6[2][0])
        mh.align_sub(eq9, eq9[1], eq6[0]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq10, eq10[0][0], eq9[3][0])
        mh.align_sub(eq11, eq11[1][0], eq9[3][0]).align_to(eq9[2], LEFT)
        mh.align_sub(eq12, eq12[3][0], eq11[1][0], coor_mask=UP)
        #mh.align_sub(eq13, eq13[1], eq1[1], coor_mask=UP)
        eq13.align_to(eq1, UP)
        mh.align_sub(eq14, eq14[1], eq13[1])
        mh.align_sub(eq15, eq15[1], eq14[1], coor_mask=UP)
        mh.align_sub(eq16, eq16[1], eq15[1]).next_to(eq15, DOWN, buff=0.2)

        eq1_1.align_to(eq1[0], UP)
        eq1_2 = eq1_1.copy()
        self.add(eq1_1)
        self.wait(0.1)
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq1_1, eq1[0]),
                  mh.rtransform(eq1_2[2:6], eq1[2][8:12], eq1_2[7], eq1[2][13]),
                  mh.fade_replace(eq1_2[6], eq1[2][12]),
                  FadeIn(eq1[1], shift=mh.diff(eq1_1, eq1[0])), run_time=2),
                  FadeIn(eq1[2][-2:], eq1[2][:8], run_time=1),
                  lag_ratio=0.5))
        self.wait(0.1)
        self.play(FadeIn(eq2))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[2][0], eq3[0], eq2[2], eq3[1],
                                eq2[3], eq3[2], eq1[2][-8:-2], eq3[3],
                                eq1[2][-2:], eq3[4]),
                  FadeOut(eq2[:2]),
                  FadeOut(eq1[2][1:-8], target_position=eq3[1:3]),
                  run_time=2)
        self.wait(0.1)
        eq3_1 = Tex(r'\sf expand the square...', color=YELLOW, font_size=60)
        eq3_1.next_to(eq3[2][2:], UP, buff=0.1).align_to(eq3[2][2:], LEFT)
        self.play(FadeIn(eq3_1))
        self.play(mh.rtransform(eq3[:2], eq4[:2], eq3[2][:2], eq4[2][:2],
                                eq3[2][3], eq4[2][2], eq3[2][4], eq4[2][4],
                                eq3[2][5:7], eq4[2][7:9], eq3[2][8], eq4[2][3],
                                eq3[2][3].copy(), eq4[2][6], eq3[2][5].copy(), eq4[2][10],
                                eq3[2][6].copy(), eq4[2][12], eq3[2][8].copy(), eq4[2][11],
                                eq3[2][8].copy(), eq4[2][13], eq3[2][9:], eq4[2][14:]),
                  #mh.rtransform(eq3[3][0], eq4[3][0], eq3[3][1], eq4[3][2],
                  #              eq3[3][5], eq4[3][8], eq3[3][2:4], eq4[3][9:11]),
                  mh.rtransform(eq3[4], eq4[4]),
                  FadeOut(eq3[2][2], shift=mh.diff(eq3[2][3], eq4[2][2])),
                  FadeOut(eq3[2][7], shift=mh.diff(eq3[2][6], eq4[2][8])),
                  FadeIn(eq4[2][5], shift=mh.diff(eq3[2][4], eq4[2][4])),
                  FadeIn(eq4[2][9], shift=mh.diff(eq3[2][5:7], eq4[2][7:9])),
                  #FadeIn(eq4[3][1], eq4[3][3:7], eq4[3][11:], rate_func=rush_into),
                  mh.rtransform(eq3[3][0], eq4[3][0], eq3[3][2:4], eq4[3][3:5],
                                eq3[3][5], eq4[3][2]),
                  mh.stretch_replace(eq3[3][4], eq4[3][1]),
                  FadeOut(eq3[3][1], target_position=eq4[3][1]),
                  run_time=2
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq4[:3], eq5[:3], eq4[4], eq5[4],
                                eq4[3][0], eq5[3][0], eq4[3][1:], eq5[3][7:11]),
                  FadeIn(eq5[3][11:], eq5[3][1:7], rate_func=lambda t: smooth(max(t-0.3,0)/0.7)),
                  FadeOut(eq3_1),
                  run_time=1.5)
        self.wait(0.1)
        line1 = Line(eq5[2][2:4].get_corner(DL), eq5[2][2:4].get_corner(UR), stroke_width=7, stroke_color=RED).set_z_index(2)
        line2 = Line(eq5[3][2].get_corner(DL)+LEFT*0.2, eq5[3][2].get_corner(UR)+RIGHT*0.2, stroke_width=7, stroke_color=RED).set_z_index(2)
        self.play(Create(line1), run_time=0.6)
        self.play(Create(line2), run_time=0.6)
        self.wait(0.1)
        self.play(FadeOut(line1, line2, eq5[2][2:4], eq5[3][2]))
        self.wait(0.1)
        eq5_2 = eq5[3][:2]+eq5[3][6]+eq5[3][9:]
        eq5_3 = eq5_2.copy().shift((eq6[2].get_right()-eq5[2].get_right())*0.5).set_opacity(0)
        self.play(LaggedStart(AnimationGroup(mh.rtransform(eq5[2][:2], eq6[2][:2], eq5[2][4:], eq6[2][6:],
                                eq5[3][4:6], eq6[2][2:4], eq5[3][7:9], eq6[2][4:6], eq5_2, eq5_3,
                                 eq5[:2], eq6[:2]),
                  FadeOut(eq5[3][3], shift=mh.diff(eq5[3][4:6], eq6[2][2:4])), run_time=1.8),
                  mh.rtransform(eq5[4], eq6[3], run_time=1), lag_ratio=0.5))
        self.wait(0.1)
        eq6_1 = Tex(r'\sf complete the square', color=YELLOW, font_size=60)
        eq6_1.next_to(eq6[2][2:], UP, buff=0.1).align_to(eq6[2][4:], LEFT)
        self.play(FadeIn(eq6_1))
        self.wait(0.1)

        #self.play(mh.rtransform(eq5[:2], eq6[:2]))#, eq5[4], eq6[4]))
        self.play(mh.rtransform(eq6[2][:2], eq7[0][:2], eq6[2][2], eq7[0][3], eq6[2][4], eq7[0][4],
                                eq6[2][6], eq7[0][5], eq6[2][10], eq7[0][6], eq6[2][10].copy(), eq7[0][10],
                                eq6[2][11:], eq7[0][12:], eq6[3], eq7[1]),
                  FadeOut(eq6[2][3], shift=mh.diff(eq6[2][2], eq7[0][3])),
                  FadeOut(eq6[2][5], shift=mh.diff(eq6[2][4], eq7[0][4])),
                  FadeOut(eq6[2][7:10], shift=mh.diff(eq6[2][6], eq7[0][5])),
                  FadeIn(eq7[0][2], shift=mh.diff(eq6[2][2], eq7[0][3])),
                  FadeIn(eq7[0][7:9], shift=mh.diff(eq6[2][10], eq7[0][6])),
                  FadeIn(eq7[0][9]),
                  FadeIn(eq7[0][11], shift=mh.diff(eq6[2][10], eq7[0][10])),
                  run_time=1.8
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq7[0][:9], eq8[0][:9], eq7[0][17:], eq8[0][9:],
                                eq7[0][10:12], eq8[1][7:9], eq7[0][13:15], eq8[1][4:6],
                                eq7[0][17:].copy(), eq8[1][9:], eq7[1], eq8[2]),
                  mh.rtransform(eq7[0][15:17], eq8[1][7:9]),
                  FadeIn(eq8[1][0]),
                  FadeOut(eq7[0][9], shift=mh.diff(eq7[0][9], eq8[1][0]) * RIGHT),
                  mh.fade_replace(eq7[0][12], eq8[1][3]),
                  FadeIn(eq8[1][2], target_position=eq7[0][10]),
                  FadeIn(eq8[1][6], target_position=eq7[0][15].get_left()),
                  FadeIn(eq8[1][1], target_position=eq7[0][10].get_left()),
                  FadeOut(eq6_1),
                  run_time=1.8)
        self.wait(0.1)
        line1 = Line(eq8[1][1:7].get_corner(DL), eq8[1][1:7].get_corner(UR), stroke_width=7, stroke_color=RED).set_z_index(2)
        line2 = Line(eq8[1][11:].get_corner(DL), eq8[1][11:].get_corner(UR), stroke_width=7, stroke_color=RED).set_z_index(2)
        self.play(Create(line1, run_time=0.4, rate_func=linear))
        self.play(Create(line2, run_time=0.4, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(line1, line2, eq8[1][1:7], eq8[1][11:]))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[:2], eq9[1:3], eq8[0], eq9[3], eq8[2], eq9[4],
                                eq8[1][0], eq9[0][0], eq8[1][8], eq9[0][5],
                                eq8[1][9:11], eq9[0][2:4]),
                  mh.stretch_replace(eq8[1][7], eq9[0][4]),
                  FadeIn(eq9[0][1], shift=mh.diff(eq8[1][10], eq9[0][3])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.rtransform(eq9[-1], eq10[-1], eq9[3][:3], eq10[0][:3], eq9[3][3], eq10[0][7],
                                eq9[3][4:7], eq10[0][3:6], eq9[3][7:12], eq10[0][8:13],
                                eq9[3][13], eq10[0][16], eq9[3][14], eq10[0][13], eq9[3][15], eq10[0][15],
                                eq9[3][12], eq10[0][17], eq9[3][16], eq10[0][18]),
                  FadeIn(eq10[0][6], shift=mh.diff(eq9[3][6], eq10[0][5])),
                  FadeIn(eq10[0][14], shift=mh.diff(eq9[3][15], eq10[0][15])),
                  run_time=1.2)
        self.wait(0.1)
        self.play(mh.rtransform(eq10[:], eq11[1:], eq9[2][1:7], eq11[0][3:9], eq9[2][7], eq11[0][13],
                                eq9[2][8], eq11[0][12], eq9[2][9], eq11[0][9], eq9[2][10], eq11[0][11],
                                eq9[2][11], eq11[0][14]),
                  FadeIn(eq11[0][10], shift=mh.diff(eq9[2][10], eq11[0][11])),
                  FadeOut(eq9[2][0], shift=mh.diff(eq9[2][0], eq11[0][0])*RIGHT),
                  FadeIn(eq11[0][:3], shift=mh.diff(eq9[2][0], eq11[0][0])*RIGHT),
                  run_time=1.2)
        self.wait(0.1)
        self.play(mh.rtransform(eq11[1:], eq12[3:], eq11[0][3:], eq12[2][1:], eq9[1], eq12[1],
                                eq9[0][:], eq12[0][3:], eq11[0][:3], eq12[0][:3]),
                  FadeIn(eq12[2][0], shift=mh.diff(eq11[0][0], eq12[2][0])*RIGHT),
                  run_time=1.4)
        self.wait(0.1)
        circ1 = mh.circle_eq(eq12[2:4] + eq12[2:4].copy().shift(DOWN*0.3)).shift(RIGHT*0.2 + UP*0.1).set_z_index(2)
        eq12_1 = Tex(r'$N(A/\rho,\rho^{-2}-1)$', r'\sf probability density', font_size=60, color=YELLOW).set_z_index(3)
        eq12_1.next_to(circ1, UP, buff=-0.2).shift(LEFT)
        eq12_1[0].next_to(eq12_1[1], UP, buff=0.2)
        eq12_1[0][2].set_color(RED)
        self.play(LaggedStart(Create(circ1 ,run_time=1.3), FadeIn(eq12_1, run_time=1), lag_ratio=0.7))
        self.wait(0.1)
        eq12_2 = Tex(r'\sf integrates to $1$', color=YELLOW, font_size=60).set_z_index(3)
        eq12_2.move_to(eq12_1[1]).align_to(eq12_1[1], LEFT)
        self.play(FadeOut(eq12_1), FadeIn(eq12_2))
        self.wait(0.1)
        self.play(FadeOut(eq12[1:], eq12_2, circ1, rate_func=linear), run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:2], eq13[:2], eq12[0], eq13[2]), run_time=1.8)
        self.wait(0.1)
        eq13_1 = MathTex(r'=', r'YX', font_size=100)
        eq13_1 = mh.align_sub(eq13_1, eq13_1[0], eq13[1])[1]
        eq13_1[0].set_color(BLUE).move_to(eq13[0][2:-3], coor_mask=RIGHT)
        eq13_1[1].set_color(RED).move_to(eq13[2][3:], coor_mask=RIGHT)
        self.play(FadeOut(eq13[0][2:-3], eq13[2][3:]), FadeIn(eq13_1), rate_func=linear)
        self.wait(0.1)
        self.play(LaggedStart(mh.rtransform(eq13[0][:2], eq14[0][2:4], eq13_1[0], eq14[0][4], eq13[0][-3:], eq14[0][5:8],
                                eq13[1], eq14[1], eq13[2][:3], eq14[2][:3], eq13_1[1], eq14[2][5], run_time=1.2),
                  FadeIn(eq14[0][:2], eq14[0][-3:], eq14[2][3:5], eq14[2][-3:], run_time=1.2),
                              lag_ratio=0.5))
        self.wait(0.1)
        brace1 = Brace(eq14[0], DOWN, color=RED)
        eq14_1 = Tex(r'\sf apply tower law', r'\sf as $X$ is $A$-measurable', font_size=60, color=YELLOW)
        eq14_1[1].next_to(eq14_1[0], DOWN, buff=0.2)
        eq14_1.next_to(brace1, DOWN, buff=0.1)
        VGroup(eq14_1[1][2], eq14_1[1][5]).set_color(RED)
        self.play(FadeIn(brace1, eq14_1))
        eq15_1 = eq15.copy()
        mh.align_sub(eq15[0], eq15[0][2], eq14[0][4], coor_mask=RIGHT)
        eq15[1].move_to(eq14[1], coor_mask=RIGHT)
        eq15[2][:3].move_to(eq14[2][:3], coor_mask=RIGHT)
        eq15[2][3].move_to(eq14[2][5], coor_mask=RIGHT)
        self.play(mh.rtransform(eq14[0][2:6], eq15[0][:4], eq14[1], eq15[1], eq14[2][:3], eq15[2][:3],
                                eq14[2][5], eq15[2][3], eq14[0][7], eq15[0][5], eq14[0][-2], eq15[0][4]),
                  mh.rtransform(eq14[0][:2], eq15[0][:2], eq14[0][8], eq15[0][3], eq14[0][-1], eq15[0][-1]),
                  FadeOut(eq14[0][6], eq14[2][3:5], eq14[2][-3:]), run_time=1.4)
        self.wait(0.1)
        self.play(mh.transform(eq15, eq15_1), FadeOut(brace1, eq14_1, rate_func=linear), run_time=1.4)
        self.wait(0.1)
        eq16_1 = Tex(r'\sf symmetry:', color=YELLOW, font_size=70)
        eq16_1.next_to(eq15, LEFT)
        self.play(FadeIn(eq16_1))
        self.wait(0.1)
        gp = VGroup(eq15.copy(), eq16)#.move_to(eq15, coor_mask=UP)
        eq15_1 = eq15.copy()
        self.play(mh.transform(eq15, gp[0], eq15_1[0][:2], eq16[0][:2],
                               eq15_1[0][3], eq16[0][3], eq15_1[0][5], eq16[0][5],
                               eq15_1[1], eq16[1], eq15_1[2][:3], eq16[2][:3]),
                  mh.fade_replace(eq15_1[0][2], eq16[0][2]),
                  mh.fade_replace(eq15_1[0][4], eq16[0][4]),
                  mh.fade_replace(eq15_1[2][3], eq16[2][3]),
                  run_time=1.8)
        self.wait(0.1)
        self.play(FadeOut(eq16_1))


        self.wait()

def get_OU_path(times, vol, mr, seed=0):
    """
    dVar = 2 Cov(X,dX)+Var(dX) = -2 mr Var dt + vol^2 dt
    Var(t) = vol^2 * (1-e^(-2mr t))/(2 mr)
    """
    if seed != 0:
        np.random.seed(seed)
    n = len(times)
    dt = (times[-1] - times[0]) / (n-1)
    s = math.sqrt(dt)
    yvals = np.zeros(n)
    bm = np.random.normal(0., 1., n)
    a = math.exp(-mr * dt)
    b = vol * math.sqrt((1 - math.exp(-2*mr*dt))/(2*mr))
    yvals[0] = bm[0] * vol / math.sqrt(2 * mr)
    for i in range(1, n):
        yvals[i] = yvals[i-1] * a + bm[i] * b
    return yvals

class OUProcess(Scene):
    def construct(self):
        n = 2000
        vol = 1.
        mr = 1.
        dev = vol / math.sqrt(2 * mr)
        tmax = 4.
        ymin = -1.5
        ymax = 1.5
        seeds = [6, 7, 11, 9, 10]
        play_time=8
        play_n = 5
        op3 = 0.5

        ax = Axes(x_range=[0, tmax * 1.04], y_range=[ymin, ymax], x_length=12, y_length=5,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "stroke_opacity": 1,
                               },
                  ).set_z_index(2)
        ax.y_axis.set_z_index(20)
        ax.to_edge(DOWN, buff=0.4)
        label1 = MathTex(r't').next_to(ax.x_axis.get_right(), RIGHT, buff=0.1)
        label2 = MathTex(r'0').next_to(ax.x_axis.get_left(), LEFT, buff=0.1)
        label3 = MathTex(r'X_t').next_to(ax.y_axis.get_top(), UP, buff=0.1).set_z_index(20)
        self.add(ax, label1, label2, label3)
        self.wait(0.1)

        times = np.linspace(0., tmax, n)
        times2 = np.linspace(0., tmax * 10, n * play_n + n)
        yval2 = -get_OU_path(times2, vol, mr, seeds[0])[::-1]
        #path2 = ax.plot_line_graph(times, yval2[:n], add_vertex_dots=False, line_color=BLUE,
        #                          stroke_width=5, stroke_opacity=1).set_z_index(10)


        colors = [BLUE, GREEN, YELLOW, RED, MAROON]
        zindices = [10, 9, 8, 7, 6]
        yvals = [yval2] + [get_OU_path(times, vol, mr, seed) for seed in seeds[1:]]
        paths = [ax.plot_line_graph(times, y, add_vertex_dots=False, line_color=lc,
                                  stroke_width=3, stroke_opacity=0.7).set_z_index(z)
                 for y, lc, z in zip(yvals[1:], colors[1:], zindices[1:])]

        #path2 = play_func()
        path2 = ax.plot_line_graph(times, yval2[:n], add_vertex_dots=False, line_color=BLUE,
                                   stroke_width=5, stroke_opacity=1).set_z_index(10)

        paths = [path2] + paths

        t_val = ValueTracker(0.)
        arr1 = Arrow(ax.coords_to_point(0, ymax), ax.coords_to_point(0, 0), buff=0, color=RED).set_z_index(20)
        origin = ax.coords_to_point(0, 0)
        txt = Tex(r'drift', color=RED).set_z_index(100)

        ht = [origin[1]]

        def f():
            t1 = t_val.get_value()
            if t1 > 0:
                pt1 = path2['line_graph'].get_end()
                ht[0] = ht[0] + (pt1[1] - ht[0]) * 0.2
                pt1[1] = origin[1] + (ht[0] - origin[1]) * 2
                pt0 = pt1 * RIGHT + ax.coords_to_point(0, 0) * UP
                scale = max(min(pt1[1] - origin[1], 1), -1)
                tip = arr1.tip.copy().stretch(scale, dim=1, about_point=origin).shift((pt1 - origin)*RIGHT)
                line = Line(pt1, pt0, color=RED, stroke_color=RED, stroke_width=6).set_z_index(200)
                txt2 = txt.copy().next_to(line, RIGHT, buff=0.2)
                op = min(max(t1 - 0.3, 0) * 10, 1) - max(t1-0.95, 0) * 20

                return VGroup(tip, line, txt2).shift(RIGHT*0.05).set_opacity(op)
            else:
                return VGroup()

        drift = always_redraw(f)
        self.add(drift)
        self.play(Create(paths[0]),
                  t_val.animate.set_value(1.),
                  rate_func=linear, run_time=4)
        self.remove(drift)
        self.wait(0.1)
        self.play(*[Create(paths[i]) for i in range(1, len(paths))],
                  rate_func=linear, run_time=2)
        self.wait(0.1)

        self.play(*[path.animate.set_stroke(opacity=op3) for path in paths[1:]], rate_func=linear, run_time=0.5)
        self.wait(0.1)

        def quadratic_in_out(t0):
            b = 1 / (2 * t0 * (1 - t0))
            c = b * t0 * t0
            g = (1 - 2 * c) / (1 - 2 * t0)

            def f(t):
                if t < t0:
                    return b * t * t
                elif t > 1. - t0:
                    return 1. - b * (1-t) * (1-t)
                return c + g * (t - t0)

            return f

        def f(y):
            return (np.exp(y*y/(2*dev)) - 1) / (math.exp(1) - 1) * (ymax-ymin) + ymin

        yvalsexp = [f(y) for y in yvals]
        print(min(yvalsexp[0]))
        pathsexp = [ax.plot_line_graph(times, y, add_vertex_dots=False, line_color=lc,
                                  stroke_width=3, stroke_opacity=op3).set_z_index(z)
                 for y, lc, z in zip(yvalsexp, colors, [20] + zindices)]
        pathsexp[0].set_stroke(width=5, opacity=1)

        pt0 = ax.coords_to_point(0, 0)
        pt1 = ax.coords_to_point(0, ymin)
        shift = pt1 - pt0
        label4 = MathTex(r'1').next_to(pt1, LEFT, buff=0.1)
        label5 = MathTex(r'e^{X^2_t}', stroke_width=1).next_to(ax.y_axis.get_top(), UP, buff=0.1).set_z_index(20)
        mh.align_sub(label5, label5[0][1], label3[0], coor_mask=RIGHT)

        self.play(*[mh.rtransform(paths[i], pathsexp[i]) for i in range(len(paths))],
                  VGroup(ax.x_axis, label1).animate.shift(shift),
                  mh.fade_replace(label2, label4),
                  mh.rtransform(label3[0][1], label5[0][3]),
                  mh.stretch_replace(label3[0][0], label5[0][1]),
                  FadeIn(label5[0][0]),
                  FadeIn(label5[0][2], shift=label5[0][1].get_corner(UR) - label3[0][0].get_corner(UR)),
                  run_time=1.4, rate_func=linear)
        self.wait(0.1)

        yval2exp = f(yval2)
        t_val = ValueTracker(0.)

        def play_shift():
            t = t_val.get_value()
            t1 = t * n * play_n
            m = round(t1)
            return m, times[1] * (m - t1)

        def play_func():
            #print('********** drawing **************')
            m, dt = play_shift()
            path2 = ax.plot_line_graph(times + dt, yval2exp[m:m + n], add_vertex_dots=False, line_color=BLUE,
                                      stroke_width=5, stroke_opacity=1).set_z_index(10)
            return path2

        self.remove(pathsexp[0])
        path2 = always_redraw(play_func)
        self.add(path2)
        self.play(t_val.animate.set_value(1.), run_time=play_time, rate_func=quadratic_in_out(0.15))
        self.remove(path2)
        pathsexp[0] = play_func()
        self.add(pathsexp[0])
        self.wait(0.1)
        m, _ = play_shift()
        yvalsexp[0] = yval2exp[m:m+n]


        i_t = 950
        t = times[i_t]
        yt = yvalsexp[0][i_t]
        pt0 = ax.coords_to_point(t, ymin)
        pt1 = ax.coords_to_point(t, yt)
        col = RED
        eq1 = MathTex(r't', color=col, stroke_width=1.5).next_to(pt0, DOWN, buff=0.02).set_z_index(40)
        dot1 = Dot(radius=0.1, fill_color=col, fill_opacity=1, stroke_width=1, stroke_color=WHITE, stroke_opacity=1).move_to(pt1).set_z_index(40)
        line1 = Line(pt0, pt1, stroke_width=7, stroke_color=col).set_z_index(40)
        yvals2 = (yt - ymin) * np.exp(np.abs(times - t) * mr) + ymin



        self.play(FadeIn(dot1, eq1), Create(line1))

        self.wait(0.1)
        sw = 8
        sc = GREEN
        plot1 = ax.plot_line_graph(times[i_t:], yvals2[i_t:], add_vertex_dots=False, stroke_width=sw, line_color=sc).set_z_index(35)
        plot2 = ax.plot_line_graph(times[i_t::-1], yvals2[i_t::-1], add_vertex_dots=False, stroke_width=sw, line_color=sc).set_z_index(35)

        dt = i_t / (n-i_t-1)
        eq2 = Tex(r'\sf expected path', r'$\mathbb E[Y_s\vert Y_t] = Y_t e^{\lvert s-t\rvert}$', font_size=60).set_z_index(60)
        eq2[1].next_to(eq2[0], DOWN, buff=0.1).set(stroke_width=1.5)
        eq2[0].set_color(YELLOW)
        eq2.move_to(ax.coords_to_point(tmax/2, ymax * 0.6))
        box1 = SurroundingRectangle(eq2, corner_radius=0.2, fill_color=BLACK, fill_opacity=0.4,
                                   stroke_width=0, buff=0.05).set_z_index(50)

        i = 1350
        j = 650
        arr1 = Arrow(eq2[1][11].get_bottom(), ax.coords_to_point(times[i], yvals2[i]), color=YELLOW, buff=0.1).set_z_index(50)
        arr2 = Arrow(eq2[1][2].get_bottom(), ax.coords_to_point(times[j], yvals2[j]), color=YELLOW, buff=0.1).set_z_index(50)

        self.play(Create(plot1, run_time=dt * 2), Create(plot2, run_time=dt * 2),
                  FadeIn(box1, eq2, arr1, arr2), rate_func = linear)
        self.wait(0.1)

        i_s = 1300
        s = times[i_s]
        ys = (yt - ymin) * math.exp(s-t) + ymin
        times1 = times[i_t:i_s+1]
        pt0 = ax.coords_to_point(s, ymin)
        pt1 = ax.coords_to_point(s, ys)
        yvals2 = (yt - ymin) * np.exp(np.abs(times1 - t) * mr) + ymin
        plot3 = ax.plot_line_graph(times1, yvals2, add_vertex_dots=False, stroke_width=sw, line_color=sc).set_z_index(35)
        dot2 = dot1.copy().move_to(pt1)
        line2 = Line(pt0, pt1, stroke_width=7, stroke_color=col).set_z_index(40)
        eq3 = MathTex(r's', color=col, stroke_width=1.5).next_to(pt0, DOWN, buff=0.02).set_z_index(40)

        eq4 = Tex(r'$\mathbb E[Y_s\vert Y_t]$', font_size=60).set_z_index(60).next_to(dot2, RIGHT, buff=0.25)
        box3 = SurroundingRectangle(eq4, corner_radius=0.1, fill_color=BLACK, fill_opacity=0.5,
                                   stroke_width=0, buff=0.05).set_z_index(50)

        self.play(FadeOut(plot1, plot2, eq2, arr1, arr2, box1), FadeIn(plot3, dot2, eq3, eq4, box3), Create(line2))

        yt = (ys - ymin) * math.exp(s-t) + ymin
        yvals2 = (ys - ymin) * np.exp(np.abs(times1[::-1] - s) * mr) + ymin
        plot3 = ax.plot_line_graph(times1[::-1], yvals2, add_vertex_dots=False, stroke_width=sw, line_color=sc).set_z_index(35)
        pt1 = ax.coords_to_point(t, yt)
        dot3 = dot2.copy().set_z_index(100).set_opacity(1)

        eq5 = Tex(r'$\mathbb E[\mathbb E[Y_t\vert Y_s]\vert Y_t]$', font_size=60).set_z_index(60).next_to(pt1, LEFT, buff=0.35)
        box4 = SurroundingRectangle(eq5, corner_radius=0.1, fill_color=BLACK, fill_opacity=0.5,
                                   stroke_width=0, buff=0.05).set_z_index(50)

        plot4 = plot3.copy()
        self.play(Create(plot3, rate_func=linear),
                  mh.rtransform(box3.copy(), box4, eq4[0][4:].copy(), eq5[0][10:], eq4[0][:2].copy(), eq5[0][:2],
                                eq4[0][:3].copy(), eq5[0][2:5], eq4[0][4:6].copy(), eq5[0][6:8],
                                eq4[0][7].copy(), eq5[0][9]),
                  mh.fade_replace(eq4[0][3].copy(), eq5[0][5]),
                  mh.fade_replace(eq4[0][6].copy(), eq5[0][8]),
                  MoveAlongPath(dot3, plot4['line_graph'], rate_func=linear),
                  run_time=1.2)

        ys = (yt - ymin) * math.exp(s-t) + ymin
        yvals2 = (yt - ymin) * np.exp(np.abs(times1 - t) * mr) + ymin
        plot3 = ax.plot_line_graph(times1, yvals2, add_vertex_dots=False, stroke_width=sw, line_color=sc).set_z_index(35)
        pt1 = ax.coords_to_point(s, ys)
        dot4 = dot3.copy()
        plot4 = plot3.copy()
        self.play(Create(plot3, rate_func=linear, run_time=1.2),
                  MoveAlongPath(dot4, plot4['line_graph'], rate_func=linear, run_time=1.32))


        self.wait()

class SDE(NormalDecomp):
    def construct(self):
        eq1 = Tex(r'\sf stochastic differential equation', color=YELLOW, font_size=60)
        eq2 = MathTex(r'dX_t', r'=', r'\sigma dB_t-\lambda X_t dt', font_size=60)
        eq2.next_to(eq1, DOWN, buff=0.2)
        self.add(eq1, eq2)
        self.wait(0.1)

        circ1 = mh.circle_eq(eq2[2][1:3]).set_z_index(2)
        circ2 = mh.circle_eq(eq2[2][6:-1]).set_z_index(2)
        eq3 = Tex(r'\sf random noise', r'(Brownian motion)', color=YELLOW).set_z_index(3)
        eq3[1].next_to(eq3[0], DOWN, buff=0.2)
        eq4 = Tex(r'\sf drift', r'(mean reversion)', color=YELLOW).set_z_index(3)
        eq4[1].next_to(eq3[0], DOWN, buff=0.2)
        eq3.next_to(circ1, DOWN, buff=0).shift(LEFT*0.5)
        eq3[1].shift(LEFT)
        eq4.next_to(circ2, DOWN, buff=0)
        eq4[1].shift(RIGHT)



        self.play(LaggedStart(Create(circ1, rate_func=linear, run_time=0.5),
                              FadeIn(eq3, run_time=1), lag_ratio=0.6))
        self.play(LaggedStart(Create(circ2, rate_func=linear, run_time=0.5),
                              FadeIn(eq4, run_time=1), lag_ratio=0.6))
        self.wait(0.1)
        self.play(FadeOut(eq2[2][0], eq2[2][5], rate_func=linear, run_time=1))
        self.wait(0.1)

        eq5 = MathTex(r'dX_t', r'=', r'dB_t- X_t dt', font_size=60)
        mh.align_sub(eq5, eq5[1], eq2[1], coor_mask=UP)
        self.play(FadeOut(circ1, circ2),
                  mh.rtransform(eq2[:2], eq5[:2], eq2[2][1:5], eq5[2][:4], eq2[2][6:], eq5[2][4:]))
        self.wait(0.1)
        self.play(FadeOut(eq3, eq4, eq1, rate_func=linear))
        self.wait()

class NormalHalf(Gain25):
    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=1)
        eq1 = MathTex(r'X_t\sim N(0,1/2)').set_z_index(2)
        eq2 = MathTex(r'{\rm Corr}(X_s,X_t)', r'=', r'e^{-\lvert t-s\rvert}').set_z_index(2)
        eq2.next_to(eq1, DOWN, buff=0.15).align_to(eq1, LEFT).shift((LEFT))
        box = SurroundingRectangle(VGroup(eq1, eq2), corner_radius=0.2, fill_color=BLACK, fill_opacity=0.6,
                                   stroke_width=0, buff=0.05)
        self.add(eq1, box)
        self.wait(0.1)
        self.play(FadeIn(eq2))
        self.wait()

class MoveAlongPathExample(Scene):
    def construct(self):
        d1 = Dot().set_color(ORANGE)
        l1 = Line(LEFT, RIGHT)
        l2 = l1.copy()
        self.add(d1)
        self.play(MoveAlongPath(d1, l2), Create(l1), rate_func=linear)
        self.wait()

if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        OUProcess().render()
