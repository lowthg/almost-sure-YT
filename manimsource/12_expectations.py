from manim import *
import numpy as np
import math
import sys
import scipy as sp

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

    def construct(self):
        cx = RED
        cy = BLUE
        MathTex.set_default(font_size=self.fs1)
        eq1 = MathTex(r'A, B', r'\sim', r'N(0,1)')
        eq1[0][0].set_color(cx)
        eq1[0][-1].set_color(cy)
        self.add(eq1)
        self.wait(0.1)
        eq2 = MathTex(r'{\rm Corr}(A, B)', r'=', r'1/2', font_size=80)
        eq2.next_to(eq1, DOWN)
        eq2[0][-4].set_color(cx)
        eq2[0][-2].set_color(cy)
        gp = Group(eq1.copy(), eq2).move_to(ORIGIN)
        self.play(FadeIn(eq2), mh.transform(eq1, gp[0]), run_time=1)
        self.wait()

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
        eq2.next_to(eq1, DOWN)
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

        eq7_1 = eq7.copy().align_to(eq1_1, UP).shift(DOWN*0.1)

        eq10 = MathTex(r'\mathbb P(b > a\vert a)', r'=', r'\left(\frac12\right)^{a+1}')
        eq10[0][2].set_color(BLUE)
        VGroup(eq10[0][4], eq10[0][6], eq10[2][-3]).set_color(RED)
        eq10.next_to(eq7_1, DOWN, buff=0.1)

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
        t1.background_rectangle.set_opacity(0.8).set_z_index(1)
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
        box2 = SurroundingRectangle(eq3, fill_color=BLACK, fill_opacity=0.8, corner_radius=0.1, buff=0.1,
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

        box3 = SurroundingRectangle(eq5, fill_color=BLACK, fill_opacity=0.8, corner_radius=0.15, buff=0.2,
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


        self.wait()
