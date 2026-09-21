import colorsys

import numpy as np
from manim import *
import sys
from manim import ManimColor
from scipy.special import expi
from common_randomprime import *
sys.path.append('../../')
import manimhelper as mh
from common.wigner import *
import matplotlib
import matplotlib.cm as cm

col_pi = col_special * 0.5 + ORANGE * 0.5
col_trig = PURPLE_A#*0.5+WHITE*0.5
col_txt = ManimColor( r'#FFAC2B')
col_prime = BLUE


class Eratosthenes(Scene):
    def construct(self):
        n_rows = 5
        n_cols = 20
        max_rows = 15

        n = n_cols * n_rows
        n_max = n_cols * max_rows
        buff = 0.02
        width = config.frame_width / n_cols * (1-2*buff)

        eq_num = MathTex(*[r'{}'.format(i) for i in range(1, n+1)], stroke_width=1.5, font_size=40, color=WHITE)
        height = eq_num.height * 1.8
        eq_rows = VGroup(*[eq_num[i:i+n_cols] for i in range(0, n, n_cols)]).set_z_index(5)
        for i in range(n_rows):
            for j in range(n_cols):
                eq_rows[i][j].move_to(RIGHT*j*width + DOWN * i * height)
        mh.align_sub(eq_rows, eq_rows[0][0], config.frame_width*(buff-0.5)*RIGHT+width/2*RIGHT)
        eq_rows.to_edge(DOWN, buff=0.4)

        pt0 = eq_rows[0].get_center() * UP + height/2 * UP + RIGHT * config.frame_width * (buff-0.5)

        eq1 = Tex(r'\sf Sieve of Eratosthenes', color=col_txt, stroke_width=2, font_size=55)
        eq1.next_to(pt0, UP, buff=0.2, coor_mask=UP).set_z_index(5)

        pt1 = eq1.get_top() * UP + pt0 * RIGHT + 0.2*UP
        lines = []
        line_args = {'stroke_width': 5}
        for i in range(n_rows+1):
            lines.append(Line(pt0 + i * height*DOWN, pt0 + i * height*DOWN + width*n_cols*RIGHT, **line_args))
        for i in range(n_cols+1):
            lines.append(Line(pt0 + i * width*RIGHT, pt0 + i * width*RIGHT + height*n_rows*DOWN, **line_args))
        lines.append(Line(pt0, pt1, **line_args))
        lines.append(Line(pt0+n_cols*width*RIGHT, pt1 + n_cols*width*RIGHT, **line_args))
        lines.append(Line(pt1, pt1+n_cols*width*RIGHT, **line_args))
        lines = VGroup(*lines).set_z_index(4)

        box = Rectangle(width=width, height=height, stroke_width=0, stroke_opacity=0, fill_opacity=0.7, fill_color=BLACK)
        box.set_z_index(0.5)
        box2 = Rectangle(width=width*n_cols, height=(pt0-pt1)[1], stroke_width=0, stroke_opacity=0, fill_opacity=0.5,
                         fill_color=BLACK)
        box2.next_to(pt0, UR, buff=0)
        cross_args = {'stroke_color': RED, 'stroke_width': 6, 'buff': 0.1}
        cross = VGroup(
            Line(box.get_corner(UL), box.get_corner(DR), **cross_args),
            Line(box.get_corner(UR), box.get_corner(DL), **cross_args),
        ).set_z_index(6).set_opacity(0)

        sieve = np.ones(n_max + 1, dtype=bool)
        sieve[:2] = False  # not prime
        for i in range(2, n_max + 1):
            if i*i > n_max:
                break
            if sieve[i]:
                sieve[i*i::i] = False

        boxes = []
        crosses = []
        for i in range(n_rows):
            for j in range(n_cols):
                boxes.append(box.copy().move_to(pt0 + (j+0.5) * width * RIGHT + (i+0.5) * height * DOWN))
                crosses.append(cross.copy().move_to(boxes[-1]))

        box.next_to(eq1, RIGHT, buff=1).set_fill(color=col_prime, opacity=1).set_z_index(1)
        eq_prime = MathTex(r'\sf prime', color=WHITE, stroke_width=1.5).set_z_index(4).next_to(box, RIGHT, buff=0.2)
        VGroup(eq1, eq_prime, box).move_to(ORIGIN, coor_mask=RIGHT)

        boxes = VGroup(*boxes)
        crosses = VGroup(*crosses)
        cross.move_to(boxes[0])
        eq_num[0].set_opacity(0.3)

        boxes_prime = VGroup()

        self.add(eq_rows, lines, eq1, boxes, crosses[0].set_opacity(1), box, eq_prime, box2)
        for i in range(2, n+1):
            if i * i > n:
                boxes2 = VGroup()
                for j in range(i, n + 1):
                    if sieve[j]:
                        boxes2.add(boxes[j-1])
                self.play(boxes2.animate.set_fill(color=col_prime))
                boxes_prime.add(*boxes2)
                break
            if sieve[i]:
                boxes_prime.add(boxes[i-1])
                self.play(boxes[i-1].animate.set_fill(col_prime), run_time=0.6, rate_func=linear)
                self.play(crosses[i*i-1::i].animate.set_opacity(1),
                          eq_num[i*i-1::i].animate.set_opacity(0.3), run_time=0.6, rate_func=linear)

        box3 = Rectangle(width=n_cols*width, height=max_rows*height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=0.7)
        box4 = Rectangle(width=n_cols*width, height=n_rows*height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=0.7)
        box4.next_to(pt0, DR, buff=0)

        # box3.next_to(pt0, DR, buff=0)
        # box3.to_edge(DOWN, buff=0.1)

        self.wait(0.1)
        self.play(FadeOut(box2, lines, eq_prime, eq1, box, crosses),
                  FadeIn(box4), rate_func=linear)
        boxes.set_opacity(0)
        boxes_prime.set_opacity(0.7)

        eq_max = MathTex(*[r'{}'.format(i) for i in range(1, n_max+1)], stroke_width=1.5, font_size=40, color=WHITE)
        eq_rows2 = VGroup(*[eq_max[i:i+n_cols] for i in range(0, n_max, n_cols)]).set_z_index(5)
        for i in range(1, n_max+1):
            if not sieve[i]:
                eq_max[i-1].set_opacity(0.3)
        pt2 = box3.get_corner(UL)
        for i in range(max_rows):
            for j in range(n_cols):
                eq_rows2[i][j].move_to(pt2 + RIGHT*(j+0.5)*width + DOWN * (i+0.5) * height)

        boxes_prime2 = VGroup()
        for i in range(n+1, n_max+1):
            if sieve[i]:
                boxes_prime2.add(boxes_prime[0].copy().move_to(eq_max[i-1]))

        shift = eq_rows2[:n_rows].get_center() - eq_rows.get_center()
        self.play(mh.rtransform(eq_rows[:], eq_rows2[:n_rows]),
                  boxes_prime.animate.shift(shift),
                  boxes_prime2.shift(-shift).animate.shift(shift),
                  eq_rows2[n_rows:].shift(-shift).animate.shift(shift),
                  mh.rtransform(box4, box3),
                  run_time=2, rate_func=smooth)

        self.wait()


def get_xticks(ax, vals=[], strs=None, scalex=1.):
    if strs is None:
        strs = [r'{}'.format(_) for _ in vals]
    tick_eqs = MathTex(*strs, font_size=50, stroke_width=1.5, color=col_num)
    origin = ax.c2p(0, 0)
    tick_eqs.next_to(origin, DOWN, buff=0.3)
    tick0 = Line(origin, origin + DOWN * 0.2, stroke_width=6, stroke_color=WHITE)
    ticks = [tick0.copy().shift(ax.c2p(_ * scalex, 0) - origin) for _ in vals]
    for _ in range(len(vals)): tick_eqs[_].move_to(ticks[_], coor_mask=RIGHT)
    return VGroup(*[VGroup(tick, eq) for tick, eq in zip(ticks, tick_eqs[:])]).set_z_index(0.5)

def get_yticks(ax, vals=[], strs=None, scaley=1., max_width=0.9, center=0.):
    if strs is None:
        strs = [r'{}'.format(_) for _ in vals]
    tick_eqs = [MathTex(str, font_size=50, stroke_width=1.5, color=col_num)[0] for str in strs]
    origin = ax.c2p(0, 0)
    for eq in tick_eqs: eq.next_to(origin, LEFT, buff=0.3)
    tick0 = Line(origin, origin + LEFT * 0.2, stroke_width=6, stroke_color=WHITE)
    ticks = [tick0.copy().shift(ax.c2p(0, _ * scaley + center) - origin) for _ in vals]
    for _ in range(len(vals)):
        tick_eqs[_].move_to(ticks[_], coor_mask=UP)
        w = tick_eqs[_].width
        if w > max_width:
            tick_eqs[_].scale(max_width/w, about_edge=RIGHT)
    return VGroup(*[VGroup(tick, eq) for tick, eq in zip(ticks, tick_eqs[:])]).set_z_index(0.3)


class PiPlot1(Scene):

    @staticmethod
    def eq_pi():
        eq = MathTex(r'\pi(x)', font_size=60, stroke_width=1.5, color=BLUE).set_z_index(4)
        eq.move_to(mh.pos(LEFT*0.45 + DOWN*0.27))
        return eq

    @staticmethod
    def get_ax():
        ax = Axes(x_range=[0, 1.05], y_range=[0, 1.05], x_length=12, y_length=6,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(1).shift(RIGHT*0.2)
        return ax

    def setup(self):
        ax = self.get_ax()

        xvals2 = np.linspace(4., 101., 1000)
        yvals3 = xvals2 / np.log(xvals2)
        scalex3 = 1/100
        scaley3 = 3./100
        plt_line3 = ax.plot_line_graph(xvals2 * scalex3, yvals3 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        prime_count = build_prime_count(1200001)

        x, y = prime_counting_vectors(prime_count, 1200001)

        plt7 = ax.plot_line_graph(x[:127]*scalex3, y[:127]*scaley3, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        xvals1 = np.linspace(0., 60., 1000)
        yvals1 = xvals1
        plt_line1 = ax.plot_line_graph(xvals1 * scalex3, yvals1 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.5)
        box1 = Rectangle(width=2, height=config.frame_height, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(3)
        box1.next_to(ax.c2p(1., 0.), UR, buff=0).next_to(ax.x_axis.tip, UP, buff=0.01, coor_mask=UP)
        box2 = Rectangle(width=2, height=config.frame_height/2, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.6)
        box2.next_to(ax.c2p(1.05, 0.02), DR, buff=0)
        box3 = Rectangle(height=2, width=config.frame_width, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.6)
        box3.next_to(ax.c2p(0.1, 1.0), UR, buff=0)
        box4 = Rectangle(height=2, width=config.frame_width/4, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.4)
        box4.next_to(ax.c2p(0,1.04), UL, buff=0)
        box5 = Rectangle(height=2, width=config.frame_width/4, stroke_width=0, stroke_opacity=0,
                         fill_color=BLACK, fill_opacity=1).set_z_index(0.4)
        box5.next_to(ax.c2p(0,0), DL, buff=0)
        MathTex.set_default(stroke_width = 1.5, font_size = 60)

        return ax, xvals2, yvals3, scalex3, scaley3, plt_line3, prime_count, x, y, plt7, xvals1, yvals1, plt_line1, box1, box2, box3, box4, box5

    @staticmethod
    def get_eq_pnt():
        MathTex.set_default(stroke_width=1.5, font_size=60)
        ax = PiPlot1.get_ax()
        eq_pnt = MathTex(r'\pi(x)', r'\sim', r'\frac{x}{\log x}')
        eq_pnt[0].set_color(BLUE)
        eq_pnt[2].set_color(GREEN)
        eq_pnt.move_to(ax.c2p(0.7, 0.3))
        eq_pnt2 = MathTex(r'\pi(x)', r'/', r'\frac{x}{\log x}', r'\to', r'1').set_z_index(5)
        eq_pnt2[0].set_color(BLUE)
        eq_pnt2[1].set_color(col_op)
        eq_pnt2[2].set_color(GREEN)
        eq_pnt2[4].set_color(col_num)
        mh.align_sub(eq_pnt2, eq_pnt2[0], eq_pnt[0]).move_to(eq_pnt, coor_mask=UP)
        eq_pnt3 = MathTex(r'a\frac{x}{\log x}', r'<', r'\pi(x)', r'<', r'b\frac{x}{\log x}').set_z_index(5)
        mh.align_sub(eq_pnt3, eq_pnt3[2], eq_pnt2[0]).align_to(eq_pnt2, LEFT).shift(LEFT*0.4)
        VGroup(eq_pnt3[0][0], eq_pnt3[-1][0]).set_color(col_var)
        VGroup(eq_pnt3[0][1:], eq_pnt3[-1][1:]).set_color(GREEN)
        eq_pnt3[2].set_color(BLUE)

        return eq_pnt, eq_pnt2, eq_pnt3

    def construct(self):
        ax, xvals2, yvals3, scalex3, scaley3, plt_line3, prime_count, x, y, plt7, xvals1, yvals1, plt_line1, box1, box2, box3, box4, box5 = self.setup()
        eqx = MathTex(r'x', stroke_width=1.5, font_size=60, color=col_x).next_to(ax.x_axis.get_right(), RIGHT, buff=0.2).set_z_index(4)

        scalex1 = 0.1
        scaley1 = 0.25

        ticks = get_xticks(ax, [2, 3, 5, 7, 11, 13, 17, 19], scalex=scalex1)
        ticksy = get_yticks(ax, [0, 1, 2, 3, 4, 5, 6, 7, 8], scaley=scaley1)
        ticksy[0].set_z_index(0.5)

        m = 15
        eps = 0.01
        plt1 = ax.plot_line_graph(x[:m+1]*scalex1, y[:m+1]*scaley1, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt2 = plt1.copy()
        plt3 = plt1.copy()

        eq_pi = self.eq_pi()

        self.add(ax, eqx, box1, box2, box3, box4, box5, ticksy[0])


        self.play(Create(plt1, run_time=1.5, rate_func=lambda t: (t+eps)*3/m),
                  FadeIn(ticks[0], ticksy[1], run_time=1))
        self.wait(0.1)
        self.remove(plt1)
        self.play(Create(plt2, run_time=1., rate_func=lambda t: (t+eps)*2/m+3/m),
                  FadeIn(ticks[1], ticksy[2], eq_pi, run_time=1))
        self.wait(0.1)
        self.remove(plt2)
        t0 = (5+eps*2)/m
        self.play(Create(plt3, rate_func=lambda t: t0 + (1-t0)*t, run_time=2),
                  Succession(Wait(0.15), FadeIn(ticks[2], ticksy[3])),
                  Succession(Wait(0.6), FadeIn(ticks[3], ticksy[4])),
                  )

        scalex2 = 1/20
        scaley2 = 2./20
        plt4 = ax.plot_line_graph(x[:29]*scalex1, y[:29]*scaley1, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt5 = ax.plot_line_graph(x[:29]*scalex2, y[:29]*scaley2, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)

        ticks2 = get_xticks(ax, [2, 3, 5, 7, 11, 13, 17, 19, 50, 100], scalex=scalex2)
        ticksy2 = get_yticks(ax, [1, 2, 3, 4, 5, 6, 7, 8, prime_count[50], prime_count[100]], scaley=scaley2)

        self.remove(plt3)
        self.add(plt4)

        self.play(mh.rtransform(plt4, plt5, ticks[:], ticks2[:-2], ticksy[1:], ticksy2[:-2]),
                  eq_pi.animate.shift(UP*0.3),
                  run_tim1=1.5)

        self.wait(0.1)

        plt6 = ax.plot_line_graph(x[:127]*scalex2, y[:127]*scaley2, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)

        ticks3 = get_xticks(ax, [2, 3, 5, 7, 11, 13, 17, 19, 50, 100], scalex=scalex3)
        ticks3[:-2].set_opacity(0)
        ticksy3 = get_yticks(ax, [1, 2, 3, 4, 5, 6, 7, 8, prime_count[50], prime_count[100],
                                  ], scaley=scaley3)
        ticksy3[:-2].set_opacity(0)

        self.remove(plt5)
        self.add(plt6)

        eps = 0.1

        self.play(AnimationGroup(mh.rtransform(plt6, plt7, ticks2, ticks3[:],
                                               ticksy2[:], ticksy3[:]),
                                 eq_pi.animate.shift(DOWN*0.2),
                                 run_time=3., rate_func=mh.rate_func_quad(0.2, 0.5)))

        eq_yex = MathTex(r'x', color=GREY).move_to(ax.c2p(0.17, 0.6))

        self.play(Create(plt_line1, run_time=1.4, rate_func=linear),
                  FadeIn(eq_yex))

        plt_line2 = ax.plot_line_graph(xvals2 * scalex3, xvals2 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)

        eq_pnt, eq_pnt2, eq_pnt3 = self.get_eq_pnt()

        self.play(mh.rtransform(plt_line2, plt_line3),
                  FadeOut(eq_yex),
                  mh.rtransform(eq_pi[0], eq_pnt[0]),
                  Succession(Wait(0.5), FadeIn(eq_pnt[1:])))
        self.wait(0.1)

        self.play(mh.rtransform(eq_pnt[0], eq_pnt2[0], eq_pnt[2], eq_pnt2[2], run_time=1),
                  mh.fade_replace(eq_pnt[1], eq_pnt2[3], run_time=1),
                  Succession(Wait(0.4), FadeIn(eq_pnt2[1], eq_pnt2[4], run_time=1)))
        self.wait(0.1)

        eq1 = MathTex(r'a', r'<', r'1', r'<', r'b').set_z_index(5)
        VGroup(eq1[0], eq1[-1]).set_color(col_var)
        eq1[2].set_color(col_num)
        eq1.next_to(eq_pnt2, DOWN, buff=0.4)
        self.play(FadeIn(eq1))
        self.wait(0.1)

        self.play(mh.rtransform(eq1[0][0], eq_pnt3[0][0], eq_pnt2[2][:], eq_pnt3[0][1:], eq_pnt2[0], eq_pnt3[2],
                                eq1[1], eq_pnt3[1], eq1[3], eq_pnt3[3], eq1[-1][0], eq_pnt3[4][0],
                                eq_pnt2[2][:].copy(), eq_pnt3[4][1:], run_time=1.6,
                                copy_colors=True),
                  FadeOut(eq_pnt2[1], eq_pnt2[-2:]),
                  FadeOut(eq1[2], target_position=eq_pnt3[2], run_time=1.6))

        eq2 = MathTex(r'{\sf for\ large\ }x').next_to(eq_pnt3, DOWN, buff=0.3).set_z_index(5)
        eq2[0][:-1].set_color(col_txt)
        eq2[0][-1].set_color(col_x)
        self.play(FadeIn(eq2), FadeOut(eq_pnt3))

        self.wait()

class PiPlot2(PiPlot1):
    def construct(self):
        ax, xvals2, yvals3, scalex3, scaley3, plt_line3, prime_count, x, y, plt7, xvals1, yvals1, plt_line1, box1, box2, box3, box4, box5 = self.setup()
        origin = ax.coords_to_point(0,0)
        nplt = 1000
        MathTex.set_default(stroke_width=1.5, font_size=60)

        eq1 = MathTex(r'\frac{x}{\log x}', color=GREEN)
        eq1.move_to(ax.c2p(0.8, 0.4))
        eq2 = MathTex(r'{\rm Li}(x)', color=ORANGE)
        eq2.move_to(ax.c2p(0.6, 0.72))

        yvals4 = expi(np.log(xvals2)) - expi(np.log(2.))
        ticky0 = get_yticks(ax, [0])[0].set_z_index(0.5).set_opacity(0)
        ticks3 = get_xticks(ax, [50, 100, 500, 1000], scalex=scalex3)
        ticksy3 = get_yticks(ax, [prime_count[50], prime_count[100],
                                  prime_count[500], prime_count[1000]], scaley=scaley3)

        self.add(ax, plt_line3, plt_line1, plt7, box1, box2, box3, box4, box5, ticks3, ticksy3, eq1)

        plt_line4 = ax.plot_line_graph(xvals2 * scalex3, yvals4 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)

        self.play(mh.rtransform(plt_line3.copy(), plt_line4),
                  FadeIn(eq2))
        self.wait(0.1)

        """
        first zoom out
        """

        scalex4 = 1/1000
        scaley4 = 5/1000
        xvals3 = np.linspace(4., 1001., 4000)
        xvals4 = np.linspace(0., 250., 1000)
        yvals5 = xvals3 / np.log(xvals3)
        yvals6 = expi(np.log(xvals3)) - expi(np.log(2.))

        i = np.searchsorted(x, 1050., side='right')

        plt8 = ax.plot_line_graph(x[:i]*scalex3, y[:i]*scaley3, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        # plt9 = ax.plot_line_graph(x[:i]*scalex4, y[:i]*scaley4, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt_line5 = ax.plot_line_graph(xvals3 * scalex3, yvals5 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        # plt_line6 = ax.plot_line_graph(xvals3 * scalex4, yvals5 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        plt_line7 = ax.plot_line_graph(xvals3 * scalex3, yvals6 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        # plt_line8 = ax.plot_line_graph(xvals3 * scalex4, yvals6 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line9 = ax.plot_line_graph(xvals4 * scalex3, xvals4 * scaley3, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)
        # plt_line10 = ax.plot_line_graph(xvals4 * scalex4, xvals4 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)

        ticks4 = get_xticks(ax, [50, 100, 500, 1000, 5000, 10000], ['50', '100', '500', r'1\,000', r'5\,000', r'10\,000'], scalex4)
        ticks4[:1].set_opacity(0)
        ticksy4 = get_yticks(ax, [prime_count[50], prime_count[100], prime_count[500], prime_count[1000],
                                  prime_count[5000], prime_count[10000]], scaley=scaley4)
        ticksy4[0].set_opacity(0)

        self.remove(plt7, plt_line3, plt_line4, plt_line1, ticks3, ticksy3)
        # self.add(plt_line5, plt_line7, plt_line9)

        # self.play(mh.rtransform(plt8, plt9, plt_line5, plt_line6, plt_line7, plt_line8, plt_line9, plt_line10,
        #                         ticks3[:], ticks4[:-2], ticksy3[:], ticksy4[:-2],
        #                         run_time=0.5, rate_func=mh.rate_func_quad(0.2, 0.2)))

        #
        tracker1, obj1 = animation_scale_redraw(scalex4 / scalex3, scaley4 / scaley3,
                                                VGroup(plt8, plt_line5, plt_line7, plt_line9),
                                                obj1x=ticks3[:], obj2x=ticks4[:-2].copy(),
                                                obj1y=ticksy3[:], obj2y=ticksy4[:-2],
                                                origin=origin,
                                                # obj2_scale=plt17
                                                )
        self.add(obj1)
        self.play(tracker1.animate().set_value(1),
                  eq1.animate.shift(UP*0.3),
                  eq2.animate.shift(DOWN*0.08),
                  rate_func=mh.rate_func_quad(0.2, 0.2),
                  run_time=3)
        self.wait(0.1)
        self.remove(obj1)

        i = np.searchsorted(x, 10050., side='right')

        scalex5 = 1/10000
        scaley5 = 8/10000

        xvals5 = np.linspace(4., 10010, nplt)
        xvals6 = np.linspace(0., 2000., 100)
        yvals7 = xvals5 / np.log(xvals5)  # x/logx up yo 10k
        yvals8 = expi(np.log(xvals5)) - expi((np.log(2)))  # Li up to 10k

        plt10 = ax.plot_line_graph(x[:i]*scalex4, y[:i]*scaley4, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt11 = ax.plot_line_graph(x[:i]*scalex5, y[:i]*scaley5, line_color=BLUE, stroke_width=8, add_vertex_dots=False).set_z_index(2)
        plt_line11 = ax.plot_line_graph(xvals5 * scalex4, yvals7 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        plt_line12 = ax.plot_line_graph(xvals5 * scalex5, yvals7 * scaley5, add_vertex_dots=False, stroke_width=8, line_color=GREEN).set_z_index(0.49)
        plt_line13 = ax.plot_line_graph(xvals5 * scalex4, yvals8 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line14 = ax.plot_line_graph(xvals5 * scalex5, yvals8 * scaley5, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line15 = ax.plot_line_graph(xvals6 * scalex4, xvals6 * scaley4, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)
        plt_line16 = ax.plot_line_graph(xvals6 * scalex5, xvals6 * scaley5, add_vertex_dots=False, stroke_width=8, line_color=GREY).set_z_index(0.49)

        scaley6 = 3/100
        yvals9 = np.interp(xvals5+0.5, x, y, left=0, right=y[-1])  # pi up to 10k
        yvals10 = yvals9 - yvals8  # pi - Li up to 20k
        # plt12 = ax.plot_line_graph(xvals5*scalex5, yvals9*scaley5, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt13 = ax.plot_line_graph(xvals5*scalex5, yvals10*scaley6+0.8, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt_line17 = ax.plot_line_graph(xvals5 * scalex5, xvals5 * 0 + 0.8, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)

        ticks5 = get_xticks(ax, [100, 500, 1000, 5000, 10_000, 50_000, 100_000],
                            ['100', '500', r'1\,000', r'5\,000', r'10\,000', r'50\,1000', r'100\,1000'], scalex5)
        ticks5[:2].set_opacity(0)
        ticks5[4][1].shift(LEFT*0.1)
        ticksy5 = get_yticks(ax, [prime_count[100], prime_count[500], prime_count[1000],
                                  prime_count[5000], prime_count[10000]], scaley=scaley5)
        ticksy5[:2].set_opacity(0)

        ticksy6 = get_yticks(ax, [-20, -10, 0, 10], scaley=scaley6, center=0.8)
        ticksy7 = get_yticks(ax, [-20, -10, 0, 10], scaley=scaley6, center=0.7)
        ticksy6[-1].set_opacity(0)
        # print('error', prime_count[10_000] - expi(np.log(1e4)) + expi(np.log(2)))

        # self.remove(plt9, plt_line6, plt_line8, plt_line10)
        # self.add(plt10, plt_line11, plt_line13, plt_line15)

        # self.play(mh.rtransform(plt10, plt11, plt_line11, plt_line12, plt_line13, plt_line14, plt_line15, plt_line16,
        #                         ticks4[1:], ticks5[:-2], ticksy4[1:], ticksy5[:],
        #                         run_time=3, rate_func=mh.rate_func_quad(0.2, 0.2)))
        tracker1, obj1 = animation_scale_redraw(scalex5 / scalex4, scaley5 / scaley4,
                                                VGroup(plt10, plt_line11, plt_line13, plt_line15),
                                                obj1x=ticks4[1:], obj2x=ticks5[:-2].copy(),
                                                obj1y=ticksy4[1:], obj2y=ticksy5[:],
                                                origin=origin,
                                                # obj2_scale=plt17
                                                )
        self.add(obj1)
        self.play(tracker1.animate().set_value(1),
                  eq1.animate.shift(UP*0.6),
                  eq2.animate.shift(UP*0.4),
                  rate_func=mh.rate_func_quad(0.2, 0.2),
                  run_time=3)
        self.remove(obj1)
        self.add(plt_line12, plt_line16, eq1, plt11, plt_line14, ticks5, ticksy5)
        self.wait(0.1)

        """
        diff between pi and Li
        """
        eq3 = MathTex(r'\pi(x) - {\rm Li}(x)', color=BLUE)
        eq3.next_to(ax.c2p(0.2, 0.2), RIGHT, buff=0)
        self.play(FadeOut(plt_line12, plt_line16, eq1))
        self.play(mh.rtransform(plt11, plt13, plt_line14, plt_line17, ticky0, ticksy6[-2]),
                  FadeOut(ticksy5, eq2),
                  Succession(Wait(0.5), FadeIn(ticksy6[:-2], eq3)))

        self.wait(0.1)
        yvals11 = -(expi(np.log(xvals5)/2)-expi(np.log(2)))/2 # -Li(sqrt x)/2
        yvals12 = yvals11 - (expi(np.log(xvals5)/3)-expi(np.log(2)))/3  # -Li(sqrt x)/2 - Li(x^{1/3))/3

        plt_line18 = ax.plot_line_graph(xvals5 * scalex5, yvals11 * scaley6 + 0.8, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        plt_line19 = ax.plot_line_graph(xvals5 * scalex5, yvals12 * scaley6 + 0.8, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)

        plt_line17_ = plt_line17.copy().set_stroke(color=GREY).set_opacity(0.48)

        eq4 = MathTex(r'-{\frac12}', r'{\rm Li}(\sqrt x)', color=ORANGE)
        mh.font_size_sub(eq4, 0, 40)
        eq4.move_to(ax.c2p(0.6, 0.55))

        self.add(plt_line17_)
        self.play(mh.rtransform(plt_line17, plt_line18),
                  Succession(Wait(0.4), FadeIn(eq4)))
        self.wait(0.1)

        eq5 = MathTex(r'-{\frac12}', r'{\rm Li}(\sqrt x)-', r'\frac13', r'{\rm Li}(\sqrt[3] x)', color=ORANGE, font_size=50)
        eq5.set_z_index(5)
        mh.font_size_sub(eq5, 0, 35)
        mh.font_size_sub(eq5, 2, 35)
        eq5.move_to(ax.c2p(0.8, 0.2))
        eq5 = mh.eq_shadow(eq5, fg_z_index=6, bg_z_index=5, bg_stroke_width=14)
        self.play(mh.rtransform(plt_line18.copy(), plt_line19, run_time=1),
                  Succession(Wait(0.4), FadeIn(eq5)))
        self.wait(0.1)
        self.play(FadeOut(plt_line19, eq5))
        self.wait(0.1)

        """
        include bias
        """

        yvals13 = yvals10 - yvals11
        plt14 = ax.plot_line_graph(xvals5*scalex5, yvals13*scaley6+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt_line20 = ax.plot_line_graph(xvals5 * scalex5, xvals5 * 0 + 0.7, add_vertex_dots=False, stroke_width=8, line_color=ORANGE).set_z_index(0.49)
        eq6 = MathTex(r'\pi(x)-{\rm Li}(x)-', r'\frac12', r'{\rm Li(\sqrt x)}', color=BLUE)
        mh.font_size_sub(eq6, 1, 40)
        eq6.move_to(ax.c2p(0.5, 0.35))
        self.play(mh.rtransform(plt13, plt14, plt_line18, plt_line20, ticksy6, ticksy7),
                  mh.rtransform(eq3[0][:], eq6[0][:-1], eq4[:], eq6[1:]),
                  FadeIn(eq6[0][-1], shift=mh.diff(eq3[0][:], eq6[0][:-1])),
                                plt_line17_.animate.shift(ax.c2p(0, 0.7 - 0.8)-origin))
        self.remove(plt_line17_)

        self.wait(0.1)

        """
        final zoom out
        """
        scalex7 = 1/1e6
        scaley8 = 1.4/100

        scalex6 = 1/1e5
        scaley7 = np.sqrt(scaley6*scaley8)
        xvals7 = np.linspace(0., 100100, nplt*10)
        xvals8 = np.linspace(4., 100100, nplt)
        xvals7 = xvals7 * (xvals5[1] - xvals5[0]) / xvals7[1] + 4

        ticks6 = get_xticks(ax, [1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
                            [r'1\,000', r'5\,000', r'10\,000', r'50\,000', r'100\,000', r'500\,000', r'1\,000\,000'], scalex6)
        ticks6[:2].set_opacity(0)
        ticks6[4].shift(LEFT*0.3)
        ticksy8 = get_yticks(ax, [-40, -20, -10, 0, 10, 20], scaley=scaley7, center=0.7)

        yvals14 = (np.interp(xvals7+0.5, x, y, left=0, right=y[-1])
                   - expi(np.log(xvals7)) + expi(np.log(xvals7)/2)/2 + expi(np.log(2))/2)  # pi-Li+Li_2 up to 100k
        yvals15 = (np.interp(xvals8+0.5, x, y, left=0, right=y[-1])
                   - expi(np.log(xvals8)) + expi(np.log(xvals8)/2)/2 + expi(np.log(2))/2)  # pi-Li+Li_2 up to 100k
        plt15 = ax.plot_line_graph(xvals7*scalex5, yvals14*scaley6+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt16 = ax.plot_line_graph(xvals8*scalex5, yvals15*scaley6+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt17 = ax.plot_line_graph(xvals8*scalex6, yvals15*scaley7+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        self.remove(plt14, ticks5, ticksy7)
        # self.play(mh.rtransform(plt15, plt17, ticks5[2:], ticks6[:-2], ticksy7[:], ticksy8[1:-1],
        #                         run_time=3., rate_func = mh.rate_func_quad(0.2, 0.2)))
        tracker1, obj1 = animation_scale_redraw(scalex6 / scalex5, scaley7 / scaley6, plt15,
                                                obj1x=ticks5[2:], obj2x=ticks6[:-2].copy(),
                                                obj1y = ticksy7[:], obj2y=ticksy8[1:-1],
                                                origin=ax.c2p(0,0.7),
                                                obj2_scale=plt16
                                                )
        self.add(obj1)
        self.play(tracker1.animate(rate_func=mh.rate_func_quad(0.2, 0.),
                  run_time=3.6).set_value(1),
                  eq6.animate(run_time=1).move_to(ax.c2p(0.4, 0.12)))
        self.remove(obj1)

        # self.play(plt15.animate.scale(0.1, about_point=ax.c2p(0,0.7)), run_time=3)

        xvals9 = np.linspace(0., 1001000, nplt*10)
        xvals10 = np.linspace(4., 1001000, nplt)
        xvals9 = xvals9 * (xvals8[1] - xvals8[0]) / xvals9[1] + 4.

        yvals16 = (np.interp(xvals9+0.5, x, y, left=0, right=y[-1])
                   - expi(np.log(xvals9)) + expi(np.log(xvals9)/2)/2 + expi(np.log(2))/2)  # pi-Li+Li_2 up to 1m
        yvals17 = (np.interp(xvals10+0.5, x, y, left=0, right=y[-1])
                   - expi(np.log(xvals10)) + expi(np.log(xvals10)/2)/2 + expi(np.log(2))/2)  # pi-Li+Li_2 up to 1m

        ticks7 = get_xticks(ax, [10_000, 50_000, 100_000, 500_000, 1_000_000],
                            [r'10\,000', r'50\,000', r'100\,000', r'500\,000', r'1\,000\,000'], scalex7)
        ticks7[:2].set_opacity(0)
        ticks7[4][1].shift(LEFT*0.4)
        # ticksy9 = get_yticks(ax, [-150, -100, -40, -20, -10, 0, 10, 20], scaley=scaley8, center=0.7)
        ticksy9 = get_yticks(ax, [-100, -40, -20, -10, 0, 10, 20], scaley=scaley8, center=0.7)
        VGroup(ticksy9[-4], ticksy9[-2]).set_opacity(0)

        plt18 = ax.plot_line_graph(xvals9*scalex6, yvals16*scaley7+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt19_ = ax.plot_line_graph(xvals10*scalex6, yvals17*scaley7+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)
        plt19 = ax.plot_line_graph(xvals10*scalex7, yvals17*scaley8+0.7, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)

        # self.add(plt18)
        self.remove(plt17)

        # self.play(mh.rtransform(plt18, plt19, ticks6[2:], ticks7[:], ticksy8, ticksy9[1:],
        #                         run_time=3., rate_func = mh.rate_func_quad(0.2, 0.2)))

        tracker1, obj1 = animation_scale_redraw(scalex7 / scalex6, scaley8 / scaley7, plt18,
                                                obj1x=ticks6[2:], obj2x=ticks7[:].copy(),
                                                obj1y = ticksy8[:], obj2y=ticksy9[1:],
                                                origin=ax.c2p(0,0.7),
                                                obj2_scale=plt19_
                                                )
        self.add(obj1)
        self.play(tracker1.animate().set_value(1),
                  rate_func=mh.rate_func_quad(0., 0.2),
                  run_time=3.6)
        self.wait(1.1)
        self.remove(obj1)

        """
        final diff
        """

        scaley9 = 5/1000
        yvals18 = yvals17 - expi(np.log(xvals10)/2)/2 + expi(np.log(2))/2
        plt20 = ax.plot_line_graph(xvals10*scalex7, yvals18*scaley9+0.8, line_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(2)

        ticksy10 = get_yticks(ax, [-100, -40, -20, 0, 20], scaley=scaley9, center=0.8)
        VGroup(ticksy10[-1], ticksy10[-4:-2]).set_opacity(0)
        self.add(ticks7)

        self.play(mh.rtransform(plt19, plt20, ticksy9[:3], ticksy10[:3], ticksy9[-3::2], ticksy10[-2::]),
                  plt_line20.animate.shift(ax.c2p(0, 0.8 - 0.7) - origin))

        self.wait()

class Narration1(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        txt1 = Tex(r'\sf The prime counting function is')
        txt2 = Tex(r'\sf $\pi(x) =$', r' number of primes less than or equal to ', r'$x$')
        txt3 = Tex(r'\sf According to the prime number theorem, it')
        txt4 = Tex(r'\sf is approximated by the logarithmic integral')
        eq5 = MathTex(r'\pi(x)', r'\sim', r'{\rm li}(x)', r'=', r'\int_0^x\frac{du}{\log u}', r'\sim', r'\frac{x}{\log x}')
        VGroup(txt1[0], txt3, txt4).set_color(col_txt)
        txt2[1].set_color((BLUE*0.4+WHITE*0.6))
        VGroup(txt2[0][0], eq5[0][0], eq5[2][:2]).set_color(col_WVD)
        VGroup(txt2[0][2], txt2[2], eq5[0][2], eq5[2][3], eq5[4][1], eq5[4][4], eq5[4][-1], eq5[6][0], eq5[6][-1]).set_color(col_x)
        VGroup(eq5[4][-4:-1], eq5[6][-4:-1]).set_color(col_trig)
        eq5[4][2].set_color(col_num)
        VGroup(eq5[4][0], eq5[4][3], eq5[4][-5], eq5[6][-5]).set_color(col_op)

        txt1 = mh.eq_shadow(txt1)
        txt2 = mh.eq_shadow(txt2)
        txt3 = mh.eq_shadow(txt3)
        txt4 = mh.eq_shadow(txt4)
        eq5 = mh.eq_shadow(eq5)

        line_spacing = DOWN * 0.8
        # txt1.to_edge(LEFT, buff=1.5)
        mh.align_sub(txt2, txt2[1][0], txt1[0][1].get_bottom(), UP, buff=0, coor_mask=UP).shift(line_spacing*1.2)
        mh.align_sub(txt3, txt3[0][0], txt2[1][0].get_bottom(), UP, buff=0, coor_mask=UP).shift(line_spacing*1.2)
        mh.align_sub(txt4, txt4[0][0], txt3[0][0].get_bottom(), UP, buff=0, coor_mask=UP).shift(line_spacing).align_to(txt3, LEFT)
        eq5.next_to(txt4, DOWN, buff=0.4)
        VGroup(txt1, txt2, txt3, txt4, eq5).move_to(ORIGIN, coor_mask=UP)

        self.add(txt1, txt2)
        self.play(FadeIn(txt3, txt4))
        self.wait(0.1)
        eq5_1 = eq5[:3].copy().move_to(ORIGIN, coor_mask=RIGHT)
        eq5_2 = eq5[:5].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(FadeIn(eq5_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq5_1, eq5_2[:3]),
                  Succession(Wait(0.5), FadeIn(eq5_2[3:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq5_2, eq5[:5], eq5_2[-1][-5:-1].copy(), eq5[-1][-5:-1]),
                  mh.fade_replace(eq5_2[-1][-6].copy(), eq5[-1][-6]),
                  mh.fade_replace(eq5_2[-1][-1].copy(), eq5[-1][-1]),
                  Succession(Wait(0.5), FadeIn(eq5[5])))
        self.wait()

class Narration2(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        txt1 = Tex(r'\sf relative error decreases at large $x$ ...', color=col_txt)
        txt1[0][-4].set_color(col_x)
        txt1 = mh.eq_shadow(txt1)
        self.add(txt1)

class Narration3(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        txt1 = Tex(r"\sf let's look at the difference", r"...", r"it's negative!", color=col_txt)
        self.add(txt1[0])
        self.play(FadeIn(txt1[1:]))
        self.wait()

class Narration4(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        txt1 = Tex(r'\sf people believed that ', r'$\pi(x) < {\rm li}(x)$', r' always holds')
        txt2 = Tex(r'\sf then Littlewood showed that ', r'$\pi(x)-{\rm li}(x)$')
        txt3 = Tex(r'\sf changes sign infinitely often')
        txt4 = Tex(r'to this day, no-one has found a single value of ', r'$x$')
        txt5 = Tex(r'with ', r'$\pi(x) > {\rm li}(x)$')
        VGroup(txt1[0], txt1[2], txt2[0], txt3, txt4[0], txt5[0]).set_color(col_txt)
        VGroup(txt1[1][0], txt1[1][5:7], txt2[1][0], txt2[1][5:7]).set_color(col_WVD)
        VGroup(txt1[1][2], txt1[1][-2], txt2[1][-2], txt2[1][2], txt4[1]).set_color(col_x)
        mh.copy_colors_eq(txt1[1], txt5[1])

        txt2.next_to(txt1, DOWN, buff=0.6)
        txt3.next_to(txt2, DOWN, buff=0.3)
        txt4.next_to(txt3, DOWN, buff=1.2)
        txt5.next_to(txt4, DOWN, buff=0.3)
        VGroup(txt1, txt2, txt3, txt4, txt5).move_to(ORIGIN)
        txt1 = mh.eq_shadow(txt1, bg_stroke_width=15)
        txt2 = mh.eq_shadow(txt2, bg_stroke_width=15)
        txt3 = mh.eq_shadow(txt3, bg_stroke_width=15)
        txt4 = mh.eq_shadow(txt4, bg_stroke_width=15)
        txt5 = mh.eq_shadow(txt5, bg_stroke_width=15)

        self.add(txt1)
        self.play(FadeIn(txt2, txt3))
        self.wait(0.1)
        self.play(FadeIn(txt4, txt5))
        self.wait()

class Narration5(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        txt1 = Tex(r'\sf a better asymptotic approximation is')
        txt2 = MathTex(r'\pi(x)', r'\sim', r'{\rm li}(x)-\frac12{\rm li}(\sqrt x)-\frac13{\rm li}(\sqrt[3] x)',
                       r'+\cdots', r'+\frac{\mu(r)}{r}{\rm li}(\sqrt[r] x)')
        txt3 = Tex(r'\sf M\"obius function', r' $=0,-1,+1$')

        txt1.set_color(col_txt)
        VGroup(txt2[0][0], txt2[2][:2], txt2[2][9:11], txt2[2][20:22], txt2[-1][7:9]).set_color(col_WVD)
        VGroup(txt2[0][2], txt2[2][3], txt2[2][14], txt2[2][26], txt2[-1][13]).set_color(col_x)
        VGroup(txt2[2][12:14], txt2[2][24:26], txt2[-1][11:13],
               txt2[2][7], txt2[2][18], txt2[-1][5]).set_color(col_op)
        VGroup(txt2[2][6:9:2], txt2[2][17:20:2], txt2[-1][6], txt2[2][23],
               txt3[1][1], txt3[1][3:5], txt3[1][6:]).set_color(col_num)
        VGroup(txt2[-1][3:7:3], txt2[-1][10]).set_color(col_var)
        txt2[-1][1].set_color(RED_C)
        txt3[0].set_color(RED)

        txt1 = mh.eq_shadow(txt1, bg_stroke_width=15)
        txt2 = mh.eq_shadow(txt2, bg_stroke_width=15)
        txt3 = mh.eq_shadow(txt3, bg_stroke_width=15)

        txt2[-1].next_to(txt2[:-1], DOWN, buff=0.2).align_to(txt2[:-1], RIGHT)
        txt2.next_to(txt1, DOWN)

        VGroup(txt1, txt2).move_to(ORIGIN, coor_mask=UP)
        txt3.next_to(txt2[-1][0], DL)
        arr1 = Arrow(txt3[0].get_corner(UR), txt2[-1][1].get_left(), color=RED, stroke_width=8).set_z_index(10)

        self.add(txt1, txt2)
        self.play(FadeIn(txt3[0], arr1))
        self.wait(0.1)
        self.play(FadeIn(txt3[1]))
        self.wait()

class Narration6(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        txt1 = Tex(r'\sf scaling y-axis by ', r'$\frac{\displaystyle\log x_{\sf max}}{\displaystyle\sqrt x_{\sf max}}$')
        txt1[0].set_color(col_txt)
        txt1[1][:3].set_color(col_trig)
        VGroup(txt1[1][3:7], txt1[1][10:]).set_color(col_x)
        txt1[1][7:10].set_color(col_op)
        txt1 = mh.eq_shadow(txt1, bg_stroke_width=15)

        self.add(txt1)

class Narration7(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        txt1 = Tex(r'\sf model: ', r'random walk scaled to observed variances')
        txt1.set_color(col_txt)
        txt1 = mh.eq_shadow(txt1, bg_stroke_width=15)

        line1 = Line(txt1[1][0].get_left()+LEFT*0.2, txt1[1][-1].get_right()+RIGHT*0.2, stroke_color=RED, stroke_width=8).set_z_index(10)

        self.add(txt1)
        self.play(Create(line1, rate_func=linear, run_time=0.6))
        self.wait()

class Narration8(Scene):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        txt1 = MathTex(r'{\sf model\!:\ }', r'\pi(x)-\hat\pi(x)', r'=', r'\frac{\sqrt x}{\log x}', r'\sum_\rho',
                       r'\frac{2}{\lvert\rho\rvert}\sin(\gamma\log x+\theta)')
        txt2 = Tex(r'where ', r'$\rho=\frac12+i\gamma$', r' are zeta zeros with ', r'$\gamma > 0$')
        txt3 = Tex(r'and ', r'$\theta$', r' are independent random phases')
        txt4 = Tex(r'uniform on ', r'$[0,2\pi]$')

        VGroup(txt1[0], txt2[0], txt2[2], txt3[0], txt3[2], txt4[0]).set_color(col_txt)
        VGroup(txt1[1][0], txt1[1][5:7]).set_color(col_WVD)
        VGroup(txt1[1][2], txt1[1][-2], txt1[3][2], txt1[3][-1], txt1[5][13]).set_color(col_x)
        VGroup(txt1[3][:2], txt1[3][3], txt1[4][0], txt1[5][1:3], txt1[5][4], txt2[1][3], txt4[1][0], txt4[1][2], txt4[1][-1]).set_color(col_op)
        VGroup(txt1[5][5:8], txt1[5][10:13], txt1[3][4:7]).set_color(col_trig)
        VGroup(txt1[5][0], txt2[1][2:5:2], txt2[3][-1], txt4[1][1]).set_color(col_num)
        VGroup(txt1[5][-2], txt3[1]).set_color(col_angle)
        txt2[1][-2].set_color(col_i)
        VGroup(txt1[5][-8], txt2[1][-1], txt2[-1][0]).set_color(GREEN)
        txt4[-1][-3:-1].set_color(col_pi)
        VGroup(txt1[4][-1], txt1[5][3], txt2[1][0]).set_color(PINK)

        txt2.next_to(txt1, DOWN, buff=0.5)
        txt3.next_to(txt2, DOWN, buff=0.2)
        txt4.next_to(txt3, DOWN, buff=0.2)
        VGroup(txt1, txt2, txt3, txt4).move_to(ORIGIN, coor_mask=UP)

        txt1 = mh.eq_shadow(txt1, bg_stroke_width=15)
        txt2 = mh.eq_shadow(txt2, bg_stroke_width=15)
        txt3 = mh.eq_shadow(txt3, bg_stroke_width=15)
        txt4 = mh.eq_shadow(txt4, bg_stroke_width=15)
        # box = SurroundingRectangle(VGroup(txt2, txt3, txt4).set_z_index(1), stroke_width=0, stroke_opacity=0,
        #                            fill_color=BLACK, fill_opacity=0.7, corner_radius=0.2)

        self.add(txt1)
        self.play(FadeIn(txt2))
        self.play(FadeIn(txt3, txt4))

        self.wait()

class PiDigits(Scene):
    def __init__(self, *args, **kwargs):
        if not config.transparent: config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=3)
        pi_str = '3.141592653589793238462643383279502884197169399375105820974944592307816406286' \
                 '208998628034825342117067982148086513282306647093844609550582231725359408128481'
                 # '117450284102701938521105559644622948954930381964428810975665933446128475648233'
                 # '786783165271201909145648566923460348610454326648213393607260249141273724587006' \
                 # '606315588174881520920962829254091715364367892590360011330530548820466521384146' \
                 # '951941511609433057270365759591953092186117381932611793105118548074462379962749'
        eq_pi = MathTex(r'\pi', r'=', pi_str).set_z_index(1)
        eq_pi = mh.font_size_sub(eq_pi, 0, 120)
        eq_pi[0].set_color(col_pi).move_to(eq_pi[2][0], coor_mask=UP)
        eq_pi[2].set_color(col_num)
        eq_pi= mh.eq_shadow(eq_pi, bg_stroke_width=16)
        # box = SurroundingRectangle(eq_pi, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
        #                            corner_radius=0.15, buff=0.15)
        obj = eq_pi #VGroup(box, eq_pi)
        obj.next_to(mh.pos(RIGHT), RIGHT, buff=0)
        self.add(obj)
        self.play(obj.animate.next_to(mh.pos(RIGHT), LEFT, buff=0), rate_func=linear, run_time=13)

class Logistic(Scene):
    bgcol = GREY
    trcol = BLACK
    def __init__(self, *args, **kwargs):
        config.background_color = self.trcol if config.transparent else self.bgcol
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        ax = Axes(x_range=[0, 1.08], y_range=[0, 1.05], x_length=5, y_length=3,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)
        f = lambda x: 4 * x * (1-x)
        plt1 = ax.plot(f, (0,1), stroke_width=6, stroke_color=BLUE).set_z_index(1.5)
        # plt2 = ax.plot(lambda x: x, (0,1), stroke_width=6, stroke_color=(GREEN*0.5+BLACK*0.5))
        plt2 = DashedLine(ax.c2p(0,0), ax.c2p(1,1), stroke_width=6, stroke_color=GREEN).set_z_index(1.3)

        coords = [(0.2, 0.)]
        lines = []
        for _ in range(20):
            x,_ = coords[-1]
            z = f(x)
            coords += [(x, z), (z,z)]
        pts = [ax.c2p(*_) for _ in coords]
        # anims = []
        dot = Dot(radius=0.1, color=YELLOW).set_z_index(4)
        dot.move_to(ax.c2p(*coords[0]))
        box = SurroundingRectangle(ax, stroke_opacity=0, stroke_width=0, fill_color=BLACK, fill_opacity=0.5)
        col_txt

        self.add(ax, plt1, plt2, dot, box)

        for i in range(len(pts)-1):
            p = pts[i]
            q = pts[i+1]
            lines.append(Line(p, q, stroke_color=RED, stroke_width=4).set_z_index(1))
            dt = max(abs(coords[i][0]-coords[i+1][0]), abs(coords[i][1]-coords[i+1][1])) * 0.5
            anim = [Create(lines[-1])]
            if i % 2 == 1:
                dot1 = dot.copy().move_to(ax.c2p(coords[i][0], 0)).set_color(ORANGE).set_z_index(3)
                dx = coords[i+1][0] - coords[i][0]
                arc = PI/4 * (1 if dx > 0 else -1)
                anim += [dot.animate(path_arc=arc).move_to(ax.c2p(coords[i+1][0],0))]
                self.add(dot1)
                for j in range(0, min(i,7)):
                    anim.append(lines[i-1-j].animate.set_stroke(opacity=1-j*0.1))
                dt *= 2
            self.play(AnimationGroup(*anim, run_time=dt, rate_func=linear))
            # anims.append(AnimationGroup(*anim, run_time=dt, rate_func=linear))

        # self.play(Succession(*anims))
        self.wait()

class SimpleFactors(Logistic):
    trcol = GREY
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=60)
        eq1 = Tex(r'\sf factors of ', r'$2$:', r' $1,2$').set_color(col_prime)
        eq2 = Tex(r'\sf factors of ', r'$3$:', r' $1,3$').set_color(col_prime)
        eq3 = Tex(r'\sf factors of ', r'$4$:', r' $1,2,4$').set_color(RED)
        eq4 = Tex(r'\sf factors of ', r'$5$:', r' $1,5$').set_color(col_prime)
        eq5 = Tex(r'\sf factors of ', r'$6$:', r' $1,2,3,6$').set_color(RED)

        eq1.to_edge(DOWN)
        mh.align_sub(eq2, eq2[0], eq1[0])
        mh.align_sub(eq3, eq3[0], eq1[0])
        mh.align_sub(eq4, eq4[0], eq1[0])
        mh.align_sub(eq5, eq5[0], eq1[0])
        eq1 = mh.eq_shadow(eq1, bg_stroke_width=12)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=12)
        eq3 = mh.eq_shadow(eq3, bg_stroke_width=12)
        eq4 = mh.eq_shadow(eq4, bg_stroke_width=12)
        eq5 = mh.eq_shadow(eq5, bg_stroke_width=12)
        shift = UP * 0.8
        gp = VGroup(eq1, eq2, eq3, eq4, eq5)

        self.add(eq1)
        self.play(gp[:1].animate(run_time=1).shift(shift),
                  Succession(Wait(0.3), FadeIn(eq2, run_time=0.7, rate_func=linear)))
        self.play(gp[:2].animate(run_time=1).shift(shift),
                  Succession(Wait(0.3), FadeIn(eq3, run_time=0.7, rate_func=linear)))
        self.play(gp[:3].animate(run_time=1).shift(shift),
                  Succession(Wait(0.3), FadeIn(eq4, run_time=0.7, rate_func=linear)))
        self.play(gp[:4].animate(run_time=1).shift(shift),
                  Succession(Wait(0.3), FadeIn(eq5, run_time=0.7, rate_func=linear)))
        self.wait()

class PiDef(Logistic):
    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=2)
        eq1 = MathTex(r'\pi(x)', r'=', r'{\sf number\ of\ primes}', r'{}\le x')

        VGroup(eq1[0][0]).set_color(col_WVD)
        VGroup(eq1[0][2], eq1[-1][-1]).set_color(col_x)
        VGroup(eq1[2], eq1[-1][0]).set_color(col_txt)

        eq1.to_edge(DOWN, buff=0.4).set_z_index(2)
        box = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.7,
                                   buff=0.2, corner_radius=0.2)

        eq_pi = PiPlot1.eq_pi()

        self.add(box, eq1)
        self.play(mh.rtransform(eq1[0], eq_pi[0], run_time=2),
                  FadeOut(box, eq1[1:]))
        self.wait()

class PNTDef(Logistic):
    def construct(self):
        _, _, eq_pnt1 = PiPlot1.get_eq_pnt()

        eq1 = MathTex(r'\pi(x)', r'\sim', r'\frac{x}{\log x}', font_size=80).set_z_index(2)
        eq2 = MathTex(r'\delta', r'\pi(x)', r'\sim', r'\frac{\delta x}{\log x}', font_size=80).set_z_index(2)
        eq3 = MathTex(r'\sum', r'\delta', r'\pi(x)', r'\sim', r'\sum', r'\frac{\delta x}{\log x}', font_size=80).set_z_index(2)
        eq4 = MathTex(r'\pi(x)', r'\sim', r'\int_2^x', r'\frac{d u}{\log u}', font_size=80).set_z_index(2)
        eq5 = MathTex(r'\pi(x)', r'\sim', r'\int_2^x', r'\frac{d u}{\log u}', r'=', r'{\rm Li}(x)', font_size=80).set_z_index(2)

        VGroup(eq1[0][2], eq1[2][0], eq1[2][-1], eq4[2][1], eq4[3][1], eq4[3][-1],
               eq5[5][3]).set_color(col_x)
        VGroup(eq1[0][0], eq5[5][:2]).set_color(col_WVD)
        VGroup(eq1[2][1], eq2[0], eq2[3][0], eq3[0], eq3[4], eq4[2][0], eq4[3][0]).set_color(col_op)
        VGroup(eq1[2][-4:-1]).set_color(col_trig)
        VGroup(eq4[2][2]).set_color(col_num)

        eq1.to_edge(DOWN, buff=0.5)
        box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                    fill_opacity=0.6, buff=0.2, corner_radius=0.2)
        mh.align_sub(eq2, eq2[1], eq1[0], coor_mask=UP)
        box2= SurroundingRectangle(eq2, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                    fill_opacity=0.6, buff=0.2, corner_radius=0.2)
        mh.align_sub(eq3, eq3[2], eq2[1], coor_mask=UP)
        mh.align_sub(eq4, eq4[0], eq3[2], coor_mask=UP)
        mh.align_sub(eq5, eq5[0], eq4[0], coor_mask=UP)

        self.add(eq_pnt1)

        self.play(AnimationGroup(mh.rtransform(eq_pnt1[2], eq1[0], eq_pnt1[4][1:], eq1[2][:]),
                  FadeIn(eq1[1], target_position=eq_pnt1[-2]), run_time=1.5),
                  FadeOut(eq_pnt1[:2], eq_pnt1[3], eq_pnt1[4][0]),
                  Succession(Wait(0.8), FadeIn(box1)))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:2], eq2[1:3], eq1[2][:], eq2[3][1:], copy_colors=True),
                  Succession(Wait(0.4), FadeIn(eq2[0], eq2[3][0])),
                  mh.rtransform(box1, box2))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:3], eq3[1:4], eq2[3:], eq3[5:], copy_colors=True),
                  Succession(Wait(0.4), FadeIn(eq3[0], eq3[4])))
        self.wait(0.1)
        self.play(FadeOut(eq3[:2], box2),
                  Succession(Wait(0.3), AnimationGroup(
                      mh.rtransform(eq3[2:4], eq4[:2],
                                    eq3[5][2:-1], eq4[3][2:-1],
                                    copy_colors=True),
                      mh.fade_replace(eq3[4], eq4[2]),
                      mh.fade_replace(eq3[5][0], eq4[3][0]),
                      mh.fade_replace(eq3[5][1], eq4[3][1]),
                      mh.fade_replace(eq3[5][-1], eq4[3][-1]),
                  )))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[:], eq5[:-2], copy_colors=True),
                  Succession(Wait(0.6), FadeIn(eq5[-2:])))
        eq5_ = mh.eq_shadow(eq5, bg_stroke_width=16)
        self.add(eq5_)
        self.wait(0.1)
        self.wait()


class LogInt(Scene):
    def construct(self):
        xmax = 18.
        ax = Axes(x_range=[0, xmax*1.05], y_range=[0, 0.55], x_length=10, y_length=4.5,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)
        ax.to_edge(UP, buff=0.3)

        plt = ax.plot(lambda x: 1/x, (2, xmax), stroke_width=6, stroke_color=col_WVD).set_z_index(1)
        eq1 = MathTex(r'y=1/x', font_size=50, stroke_width=1.5)
        eq1[0][::4].set_color(col_x)
        eq1[0][2].set_color(col_num)
        eq1[0][3].set_color(col_op)
        eq1.next_to(ax.c2p(xmax/2, 2/xmax), UP, buff=0.45)
        ticks = mh.get_xticks(ax, [2., xmax], [r'2', r'x'])
        ticks[1][1].align_to(ticks[0][1], UP).set_color(col_x)
        ticks[0][1].set_color(col_num)
        eq2 = MathTex(r'\delta x', font_size=35, stroke_width=1.5).set_z_index(0.5)
        eq2[0][0].set_color(col_op)
        eq2[0][1].set_color(col_x)

        n = 16
        dx = (xmax-2) / n

        eqs = [
            eq2.copy().next_to(ax.c2p(2+(i+0.5)*dx), UP, buff=0.2) for i in range(n)
        ]

        lines = [
            Line(ax.c2p(2+i*dx, 0), ax.c2p(2+i*dx,1/(2+i*dx)), stroke_width=3, stroke_color=WHITE*0.5).set_z_index(0.5) for i in range(n+1)
        ]


        self.add(ax, plt, eq1, ticks, *lines, *eqs)

        area = ax.get_area(plt, (2, xmax), color=ORANGE, opacity=0.3).set_z_index(0.4)
        self.play(FadeIn(area, rate_func=linear),
                  Succession(Wait(0.6), FadeOut(*eqs, *lines[1:-1], rate_func=linear)))

        self.wait()

class LiApprox(Logistic):
    bgcol = BLACK
    def construct(self):
        MathTex.set_default(font_size=60, stroke_width=1.5)
        eq1 = MathTex(r'{\rm Li}(x)', r'=', r'\int_2^x', r'\frac{du}{\log u}')
        eq2 = MathTex(r'{\rm Li}(x)', r'=', r'\left[', r'\frac{u}{\log u}', r'\right]_{u=2}^x',
                      r'-', r'\int_2^x', r'u\frac{d}{du}', r'\left(\frac{1}{\log u}\right)', r'du')
        eq3 = MathTex(r'{\rm Li}(x)', r'=', r'\left[', r'\frac{u}{\log u}', r'\right]_{u=2}^x',
                      r'+', r'\int_2^x', r'\left(\frac{1}{\log^2 u}\right)')
        eq4 = MathTex(r'{\rm Li}(x)', r'=', r'\left[', r'\frac{u}{\log u}', r'\right]_{u=2}^x',
                      r'+', r'\int_2^x', r'\frac{du}{\log^2 u}')
        eq5 = MathTex(r'{\rm Li}(x)', r'\approx', r'\frac{x}{\log x}',
                      r'+', r'\frac{x}{\log^2 x}')
        eq6 = MathTex(r'{\rm Li}(x)', r'\approx', r'\frac{x}{\log x}',
                      r'\left(1', r'+', r'\frac{1}{\log x}', r'\right)')
        eq7 = MathTex(r'{\rm Li}(x)', r'\approx', r'\frac{x}{\log x}',
                      r'\left(1', r'+', r'\frac{1}{\log x}', r'\right)',
                      r'\sim', r'\frac{x}{\log x}')

        mh.align_sub(eq3, eq3[1], eq2[1])
        mh.align_sub(eq3[7], eq3[7][2], eq2[8][2])
        mh.align_sub(eq4, eq4[1], eq3[1], coor_mask=UP)
        mh.align_sub(eq5, eq5[0], eq4[0])
        mh.align_sub(eq5[2], eq5[2][1], eq4[3][1], coor_mask=RIGHT)
        mh.align_sub(eq5[4], eq5[4][-6], eq4[7][-6], coor_mask=RIGHT)
        mh.align_sub(eq6, eq6[1], eq5[1], coor_mask=UP)
        mh.align_sub(eq7, eq7[1], eq6[1], coor_mask=UP)

        VGroup(eq1[0][:2]).set_color(col_WVD)
        VGroup(eq1[0][3], eq1[2][2], eq1[3][1], eq1[3][-1], eq2[4][1:3], eq2[7][-1], eq2[7][0]).set_color(col_x)
        VGroup(eq1[2][0], eq1[3][0], eq1[3][2], eq2[2], eq2[4][0], eq2[7][1:4]).set_color(col_op)
        VGroup(eq1[2][1], eq2[4][-1], eq2[8][1], eq3[7][-3], eq6[5][0], eq6[3][1]).set_color(col_num)
        VGroup(eq1[3][-4:-1]).set_color(col_trig)
        mh.copy_colors_eq(eq1[3][1:], eq7[-1][:])

        sw = 15
        eq1 = mh.eq_shadow(eq1, bg_stroke_width=sw)
        eq2 = mh.eq_shadow(eq2, bg_stroke_width=sw)
        eq3 = mh.eq_shadow(eq3, bg_stroke_width=sw)
        eq4 = mh.eq_shadow(eq4, bg_stroke_width=sw)
        eq5 = mh.eq_shadow(eq5, bg_stroke_width=sw)
        eq6 = mh.eq_shadow(eq6, bg_stroke_width=sw)
        eq7 = mh.eq_shadow(eq7, bg_stroke_width=sw)

        mh.rtransform.copy_colors = True
        mh.stretch_replace.copy_colors = True

        self.add(eq1)

        self.play(AnimationGroup(mh.rtransform(eq1[:2], eq2[:2], eq1[3][1:].copy(), eq2[3][:],
                                eq1[2], eq2[6], eq1[3][:2], eq2[9][:], eq1[3][2:], eq2[8][2:-1]),
                  FadeIn(eq2[2], eq2[4][0], eq2[4], shift=mh.diff(eq1[3], eq2[3])*RIGHT),
                  FadeIn(eq2[8][1], target_position=eq1[3][1]),
                                 run_time=2.),
                  Succession(Wait(1.2), FadeIn(eq2[7], eq2[8][0], eq2[8][-1], eq2[5]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:5], eq3[:5], eq2[6], eq3[6],
                                eq2[8][1:6], eq3[7][1:6], eq2[8][6], eq3[7][7]),
                  mh.fade_replace(eq2[5], eq3[5]),
                  FadeOut(eq2[7], eq2[8][0], eq2[8][-1]),
                  FadeIn(eq3[7][6]))
        self.play(mh.rtransform(eq3[:7], eq4[:7],
                                eq2[-1][:], eq4[7][:2], eq3[7][2:-1], eq4[7][2:]),
                  FadeOut(eq3[7][1], shift=mh.diff(eq3[7][2], eq4[7][2])*RIGHT))
        self.wait(0.1)
        self.play(mh.rtransform(eq4[:2], eq5[:2], eq4[3][1:5], eq5[2][1:5], eq4[1], eq5[1]),
                  mh.stretch_replace(eq4[3][0], eq5[2][0]),
                  mh.stretch_replace(eq4[3][-1], eq5[2][-1]),
                  FadeOut(eq4[2], eq4[4]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq4[7][-6:-1], eq5[4][-6:-1]),
                  mh.stretch_replace(eq4[7][-1], eq5[4][-1]),
                  mh.stretch_replace(eq4[7][1], eq5[4][0]),
                  FadeOut(eq4[7][0], eq4[6])
                  )
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq5[:3], eq6[:3], eq4[5], eq6[4],
                                eq5[4][1:5], eq6[5][1:5], eq5[4][-1], eq6[5][-1]),
                  FadeOut(eq5[4][-2], shift=mh.diff(eq5[4][-3], eq6[5][-2])),
                  mh.stretch_replace(eq5[4][0], eq6[5][0], copy_colors=False),
                                 run_time=1.2),
                  Succession(Wait(0.4), FadeIn(eq6[3], eq6[-1]))
                  )
        self.wait(0.1)
        circ1 = mh.circle_eq(eq6[5], scale=0.5).set_z_index(5).shift(DOWN*0.05)
        self.play(Create(circ1, run_time=0.6, rate_func=linear))
        eq_ = MathTex(r'\to0', stroke_width=2).set_color(RED)
        eq_.next_to(circ1, UP, buff=0.1).shift(RIGHT*0.6)
        self.play(FadeIn(eq_))
        self.wait(0.1)
        self.play(mh.rtransform(eq6[:], eq7[:-2]),
                  VGroup(circ1, eq_).animate().shift(mh.diff(eq6[-2], eq7[-4])),
                  Succession(Wait(0.4), FadeIn(eq7[-2:])))

        self.wait()

H = LabeledDot(Text("H", color=BLACK, font='Helvetica', weight=SEMIBOLD), radius=0.35, color=BLUE).scale(1.5)
T = LabeledDot(Text("T", color=BLACK, font='Helvetica', weight=SEMIBOLD), radius=0.35, color=YELLOW).scale(1.5)
def get_coin(face='H'):
    global H, T
    if face == 'H':
        return H.copy()
    elif face == 'T':
        return T.copy()
    raise Exception('invalid argument {}'.format(face))

def animate_flip(coin, nflips=1, run_time=1.):
    """
    RETURNS a list of animations that animate the mobject "coin" being flipped
    The "final" variable incidicates what you want it to be at the end of the flipping
    To animate a coin, use a loop to play the animations:

    for a in animate_flip(coins[i],coin_flips[i]):
            self.play(a,run_time=0.2)

    """

    global H, T

    final = coin.submobjects[0].text

    offset = 0 if final == 'H' else 1  # Ensures the coin lands on the side requested

    scale = coin.width/H.width

    full_fc = [H.copy().move_to(coin.get_center()).scale(scale), T.copy().move_to(coin.get_center()).scale(scale)]

    tracker = ValueTracker(1.)
    def get_obj():
        t = tracker.get_value() * nflips * math.pi
        cos = np.cos(t)
        coin = full_fc[offset if cos > 0 else 1-offset]
        return coin.copy().stretch((abs(cos)+0.05)/1.05, dim=1)

    obj = always_redraw(get_obj)

    return tracker.animate(run_time=run_time, rate_func=linear).set_value(0.), obj

def average_linear_interpolant(x, y, u, du):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)

    dx = np.diff(x)
    slopes = np.diff(y) / dx

    # Integral of f from x[0] to each x knot
    areas = np.concatenate((
        [0.0],
        np.cumsum((y[:-1] + y[1:]) * dx / 2),
    ))

    # First row: lower endpoints; second row: upper endpoints
    z = np.stack((u - du, u + du))

    j = np.searchsorted(x, z, side="right") - 1
    j = np.clip(j, 0, len(x) - 2)

    dz = z - x[j]

    # Antiderivative evaluated at all endpoints simultaneously
    F = areas[j] + dz * (y[j] + 0.5 * slopes[j] * dz)

    return (F[1] - F[0]) / (2 * du)

def zeros(i): return r'0' * (i % 3) + r'\,000' * (i // 3)

def get_tick_strs(i): return [r'5' + zeros(i-1), r'1' + zeros(i)]

class RandomWalk(Scene):
    def construct(self):
        nt = 12
        nmax = 12000
        scaley = 1. / (np.sqrt(nt) * 1.)
        ax = Axes(x_range=[0, 1.05], y_range=[-1, 1],
                  axis_config={'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  x_length=12,
                  y_length=6
                  ).shift(DOWN*0.2, RIGHT*0.6)

        title = Tex(r'\sf Random Walk', font_size=60, stroke_width=2, color=col_txt)
        title.next_to(ax.get_top(), DOWN, buff=-0.02, coor_mask=UP).shift(UP*0.2)

        self.add(ax, title)

        np.random.seed(4)
        items = [ax, title]

        y = 0
        point = ax.c2p(0, y)
        dot = Dot(point, fill_color=YELLOW).set_z_index(4)
        self.play(FadeIn(dot), run_time=0.2)
        scalex = 1/nt
        tosses = np.random.choice([1, -1], size=nmax)
        yvec = np.cumsum(tosses)
        coins = []
        dots = [dot]
        lines = []

        for i in range(nt):
            y1 = yvec[i]
            point1 = ax.c2p((i+1)*scalex, y1*scaley)
            line = Line(point, point1, stroke_width=6, stroke_color=BLUE).set_z_index(3)
            lines.append(line)
            dot = Dot(point1, fill_color=YELLOW).set_z_index(4)
            dots.append(dot)
            run_time=0.37
            if i == 0:
                up = tosses[i] > 0
                coin = get_coin('H' if up else 'T').scale(0.6) \
                    .next_to(ax.c2p((i + 0.5) * scalex, 0) * RIGHT, DOWN, buff=0.4)
                anim, obj = animate_flip(coin, nflips=2, run_time=run_time)
                self.add(*obj)
                self.play(anim)
                obj.clear_updaters()
                coins.append(obj)
            anims = AnimationGroup(FadeIn(dot), Create(line, rate_func=linear), run_time=run_time)
            if i < nt-1:
                up = tosses[i+1] > 0
                coin = get_coin('H' if up else 'T').scale(0.6) \
                    .next_to(ax.c2p((i + 1.5) * scalex, 0) * RIGHT, DOWN, buff=0.4)
                anim, obj = animate_flip(coin, nflips=2, run_time=run_time, flag=(i==0))
                self.add(*obj)
                self.play(anims, anim)
                obj.clear_updaters()
                coins.append(obj)
            else:
                self.play(anims)

            point = point1
        xvec = np.arange(nmax+1)
        yvec = np.concatenate(([0], yvec))
        xplot = np.linspace(0, 1, 4000)
        tracker = ValueTracker(0)

        def get_obj():
            t = tracker.get_value()
            x2 = np.exp(t*np.log(nmax/nt)) * nt
            scaley2 = np.sqrt(nt / x2) * scaley
            dx = x2 * xplot[1] * 2
            yplot = average_linear_interpolant(xvec, yvec, xplot * x2, dx) * scaley2
            plt = ax.plot_line_graph(xplot, yplot, add_vertex_dots=False, stroke_width=6, stroke_color=BLUE).set_z_index(3)
            m = int(np.ceil(np.log(x2)/np.log(10)))
            n = 10**m
            xtvals = [n//200, n//100, n//20, n//10, n//2, n]
            strs = np.concatenate([get_tick_strs(i) for i in [m-2, m-1, m]])
            xticks = mh.get_xticks(ax, xtvals, strs, scalex=1/x2, label_color=col_num)
            for i in range(6):
                x = xtvals[i]
                xticks[i][1] = mh.eq_shadow(xticks[i][1], fg_z_index=8, bg_z_index=7, bg_stroke_width=12)
                if x < x2 / 10:
                    op = max(20*x/x2 - 1,0)
                    xticks[i].set_opacity(op)
                if x > x2:
                    op = max(1-(x/x2 - 1)*10,0)
                    xticks[i].set_opacity(op)

            m = int(np.ceil(np.log(1/scaley2)/np.log(10)))
            n = 10**m
            ytvals = [n//20, n//10, n//2, n]
            ytvals = ytvals + [-_ for _ in ytvals]
            strs = [str(_) for _ in np.concatenate([get_tick_strs(i) for i in [m-1,m]])]
            strs = strs + [r'-' + _ for _ in strs]
            yticks = mh.get_yticks(ax, ytvals, strs, scaley=scaley2, label_color=col_num, max_width=2)
            for i in range(8):
                y = abs(ytvals[i])
                if y * scaley2 < 0.2:
                    op = max((y * scaley2 - 0.1)*10,0)
                    yticks[i].set_opacity(op)
                if y * scaley2 > 1:
                    op = max(1-(y*scaley2 - 1)*10,0)
                    yticks[i].set_opacity(op)

            return VGroup(plt, xticks, yticks)

        obj = always_redraw(get_obj)

        self.play(FadeOut(*coins, *dots), FadeIn(obj))
        self.remove(*lines)
        self.play(tracker.animate.set_value(1), run_time=4)
        obj.clear_updaters()

        xvec = np.linspace(0, 1, 1000)
        yvec = np.sqrt(xvec * nt) * scaley
        plt1 = ax.plot_line_graph(xvec, yvec, add_vertex_dots=False, stroke_width=6, stroke_color=RED).set_z_index(8)
        plt2 = ax.plot_line_graph(xvec, -yvec, add_vertex_dots=False, stroke_width=6, stroke_color=RED).set_z_index(8)
        eq1 = MathTex(r'\sqrt x', stroke_width=1.5, stroke_color=RED, font_size=60).set_z_index(5)
        eq2 = MathTex(r'-\sqrt x', stroke_width=1.5, stroke_color=RED, font_size=60).set_z_index(5)
        eq1.move_to(ax.c2p(0.7, 0.7))
        mh.align_sub(eq2, eq2[0][1:], eq1).move_to(ax.c2p(0, -0.7), coor_mask=UP)

        self.play(Create(plt1), Create(plt2), FadeIn(eq1, eq2), run_time=2)

        self.wait()
        # self.play(plot[1::2].animate.set_color(ManimColor(WHITE.to_rgb() * 0.5)).set_z_index(1),
        #           plot[0::2].animate.set_color(ManimColor(YELLOW.to_rgb() * 0.5)).set_z_index(2),
        #           FadeOut(*coins),
        #           run_time=0.5)
