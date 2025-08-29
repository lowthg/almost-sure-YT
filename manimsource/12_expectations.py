from manim import *
import numpy as np
import math
import sys
import scipy as sp

sys.path.append('../')
import manimhelper as mh

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
            return math.exp(-x*x)

        xmin = -2
        xmax = 2
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

class StandardNormal(ThreeDScene):
    colors = [
        ManimColor(RED_D.to_rgb() * 0.5),
        ManimColor(RED_E.to_rgb() * 0.5)
    ]

    def plots(self, display=True, ymax=1.15):
        self.set_camera_orientation(phi=PI/2, theta=-PI/2)

        def p0(x):
            return math.exp(-x * x / 2)

        xmax = 2.5
        ax = Axes(x_range=[-xmax, xmax + 0.2], y_range=[0, ymax], x_length=8, y_length=2*ymax/1.15,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "shade_in_3d": True,
                               },
                  #                  shade_in_3d=True,
                  ).set_z_index(1)
        ax[0].submobjects[0].set(shade_in_3d=True)
        ax_o = ax.coords_to_point(0, 0)
        ax.shift(-ax_o)
        ax_o=ORIGIN
        xlen = ax.coords_to_point(xmax, 0)[0] - ax_o[0]
        ylen = ax.coords_to_point(0, 1)[1] - ax_o[1]

        plt1 = ax.plot(p0, x_range=[-xmax, xmax], color=BLUE, shade_in_3d=True).set_z_index(2)
        fill1 = ax.get_area(plt1, color=BLUE, opacity=0.5, shade_in_3d=True).set_z_index(2)
        eq1 = MathTex(r'p(a)=\frac1{\sqrt{2\pi}}e^{-\frac12a^2}', font_size=35, shade_in_3d=True)[0]
        eq2 = MathTex(r'{p(\bf v)=\frac1{2\pi\lvert\Sigma\rvert^{\frac12}}e^{-\frac12v^T\Sigma^{-1} v}}', font_size=35, color=WHITE, stroke_width=1.7, stroke_color=WHITE, shade_in_3d=True)[0]

        eq1.set_z_index(3).move_to(ax.coords_to_point(-xmax, 1.1), UL)
        eq2.set_z_index(3).move_to(ax.coords_to_point(-xmax, 1), UL)

        gp1 = VGroup(ax, plt1, fill1, eq1, eq2).rotate(PI/2, axis=RIGHT, about_point=ax_o)
        eq2.shift(DOWN*xlen/2)
        if display:
            self.add(ax)
            self.wait(0.2)
            self.play(LaggedStart(AnimationGroup(Create(plt1, rate_func=linear), FadeIn(eq1)),
                                  FadeIn(fill1), lag_ratio=0.5), run_time=1.5)
            self.wait(0.1)

        sq1 = Surface(lambda u, v: u * RIGHT + v * UP, u_range=[-xlen, xlen], v_range=[-xlen, xlen], fill_opacity=0.3,
                      stroke_opacity=0.4, checkerboard_colors=[RED_D, RED_E])

        if display:
            self.remove(ax)
            self.add(ax)
            self.move_camera(phi=70*DEGREES, theta=-120*DEGREES)
        else:
            self.set_camera_orientation(phi=70*DEGREES, theta=-120*DEGREES)

        gp2 = gp1[:-2].copy()
        gp2.set(shade_in_3d=True)
        ax.y_axis.set_z_index(3)

        if display:
            self.play(Rotate(gp2, -90*DEGREES, OUT, about_point=ax_o), FadeIn(sq1))
            self.play(gp1[1:-1].animate.shift(xlen*UP), gp2[1:].animate.shift(xlen*RIGHT))
        else:
            gp2.rotate(-90*DEGREES, about_point=ax_o)
            gp1[1:-1].shift(xlen*UP)
            gp2[1:].shift(xlen*RIGHT)

        def p1(x, y):
            return (RIGHT * x + UP * y) * xlen/xmax + OUT * math.exp(-(x*x+y*y)/2) * ylen

        def p2(x, y):
            return (RIGHT * x + UP * y) * xlen/xmax + OUT * math.exp(-(x*x+y*y+x*y)*2/3) * ylen

        sq1.set_z_index(4)
        surf1 = Surface(p1, u_range=[-xmax, xmax], v_range=[-xmax, xmax], fill_opacity=0.9,
                      stroke_opacity=0.8, checkerboard_colors=self.colors, stroke_color=WHITE).set_z_index(200, family=True)
        surf2 = Surface(p2, u_range=[-xmax, xmax], v_range=[-xmax, xmax], fill_opacity=0.9,
                      stroke_opacity=0.8, checkerboard_colors=self.colors, stroke_color=WHITE).set_z_index(200, family=True)
        line1 = Line(OUT * ylen, OUT * ylen * 1.12, stroke_width=4, stroke_color=WHITE).set_z_index(300)
        if display:
            self.add(line1)
            self.play(ReplacementTransform(sq1, surf1),
                      FadeIn(eq2, rate_func=lambda t: smooth(min(t*2, 1) - max(t*4 - 3, 0))),
                      FadeOut(eq1, rate_func=lambda t: smooth(max(t*4 - 3, 0))),
                      run_time=2)
            self.play(ReplacementTransform(surf1, surf2),
                      run_time=1.2)

        return xmax, xlen, ylen, VGroup(ax, gp2[0], line1)

    def construct(self):
        self.plots()

class ExpectedXY(Scene):
    fs1 = 100
    def construct(self):
        cx = RED
        cy = BLUE
        cx = cy = WHITE
        MathTex.set_default(font_size=self.fs1)
        eq1 = MathTex(r'X', r'=', r'e^{\frac12A^2}')
        eq1[0][0].set_color(cx)
        eq1[2][4].set_color(cx)
        eq1_1 = eq1[2].copy().scale(1.2).move_to(ORIGIN)
        self.add(eq1_1)
        self.wait(0.5)
        self.play(mh.rtransform(eq1_1, eq1[2]),
                  FadeIn(eq1[:2], shift=mh.diff(eq1_1, eq1[2])),
                  run_time=1.2)
        self.wait(0.1)
        eq2 = MathTex(r'Y', r'=', r'e^{\frac12B^2}')
        eq2[0][0].set_color(cy)
        eq2[2][4].set_color(cy)
        eq2.next_to(eq1, DOWN)
        gp = VGroup(eq1.copy(), eq2).move_to(ORIGIN)
        eq1_2 = eq1.copy()
        self.play(mh.transform(eq1, gp[0]),
                  mh.rtransform(eq1_2[1], eq2[1], eq1_2[2][:4], eq2[2][:4],
                                eq1_2[2][5], eq2[2][5]),
                  mh.fade_replace(eq1_2[0], eq2[0]),
                  mh.fade_replace(eq1_2[2][4], eq2[2][4]),
                  run_time=1.6)
        self.wait(0.1)
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