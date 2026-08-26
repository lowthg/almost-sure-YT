import colorsys

import numpy as np
import torch
from manim import *
import sys
import scipy as sp
from manim import ManimColor
import matplotlib
import matplotlib.cm as cm

sys.path.append('../../')
import manimhelper as mh
from common.wigner import *
from algansource.wave import setup_wave, set_wave, WaveEvolver, setup_cam, setup_surf, WaveEvolution


col_pi = col_special * 0.5 + ORANGE * 0.5
col_trig = PURPLE_A#*0.5+WHITE*0.5
col_laser = ManimColor(r'#ED2F32')
col_txt = ManimColor( r'#FFAC2B')

def eq_shadow(eq: VGroup, fg_z_index=4., bg_z_index=0., bg_color=BLACK, bg_stroke_width=10.):
    res = VGroup()
    for eq1 in eq:
        elem = VGroup()
        for eq2 in eq1:
            elem.add(VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res


class Intro_Simple_FT_time(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = BLUE if config.transparent else BLACK
        Scene.__init__(self, *args, **kwargs)

    xmax = 6.

    def construct(self):
        xmax = self.xmax
        ax = Axes(x_range=[-xmax, xmax*1.1], y_range=[-1, 1.], x_length=6, y_length=2,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "shade_in_3d": True,
                               },
                  ).set_z_index(1)
        ax.to_edge(DOWN, buff=0.2).to_edge(LEFT, buff=0.4)
        eq1 = MathTex(r'\sf time', color=col_txt, stroke_width=1.5)
        eq1.next_to(ax.x_axis.get_right(), DOWN, buff=0.2).shift(LEFT*0.3)

        freq = 4.

        xvals = np.linspace(-xmax, xmax, 600)
        yvals = np.cos(xvals*freq) * np.exp(-xvals*xvals/9)
        plt1 = ax.plot_line_graph(xvals, yvals, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(4)
        area1 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False)

        self.add(ax.x_axis, eq1)
        # self.add(VGroup(ax.x_axis, eq1, plt1, area1).copy().to_edge(RIGHT, buff=0.4))
        self.play(Create(plt1, run_time=1.5, rate_func=linear),
                  Succession(Wait(0.5), FadeIn(area1)))
        self.wait(0.1)

        ax2 = ax.copy().to_edge(RIGHT, buff=0.4)
        eq2 = MathTex(r'\sf freq.', color=col_txt, stroke_width=1.5)
        eq2.next_to(ax2.x_axis.get_right(), DOWN, buff=0.2).shift(LEFT*0.3)
        self.play(FadeIn(ax2.x_axis, eq2, run_time=0.5))
        yvals = np.exp(-xvals*xvals*2/9)
        plt2 = ax2.plot_line_graph(xvals, yvals, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(4)
        area2 = ax2.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False)
        self.play(Create(plt2, run_time=0.6, rate_func=linear),
                  Succession(FadeIn(area2)))
        self.wait(0.1)

        npts = 600
        evolver = WaveEvolver(xrange=(-xmax, xmax), npts=npts, n_extend_left=2000, n_extend_right=2000, n_scale=1,
                              speed=1.)
        evolver.V = evolver.xvals1 ** 2 * 0.5 - 0.5
        evolver.psi = np.clip(1. - (np.abs(evolver.xvals1)-PI)*10, 0, 1)*math.sqrt(2/PI)
        evolver.evolve(PI/2).real
        evolver.speed =-1
        evolver.time = 0.
        # yvals3 = np.clip(21. - np.abs(evolver.xvals)*10, 0, 1)
        origin = mh.pos(DOWN*2)
        v = ax.get_center() - origin
        theta = np.arctan(-v[0]/v[1])
        print(theta)

        value_t = ValueTracker(0.)
        val_op = ValueTracker(1.)
        val_op2 = ValueTracker(1.)
        do_complex = [False]

        def get_obj():
            t = value_t.get_value()
            yvals2 = evolver.evolve((t - evolver.time)*PI/4)
            evolver.time = t
            opac = min(val_op.get_value(), val_op2.get_value())
            plt2 = ax.plot_line_graph(xvals, yvals2.real, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False,
                                      stroke_opacity=opac).set_z_index(4)
            area2 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                       np.concatenate(([0], yvals2.real, [0])), stroke_width=0, stroke_opacity=0,
                                       fill_opacity=0.3*opac, fill_color=BLUE, add_vertex_dots=False)
            res = VGroup(plt2['line_graph'], area2['line_graph'], ax.x_axis.copy())

            if do_complex[0]:
                plt2 = ax.plot_line_graph(xvals, yvals2.imag, stroke_color=ORANGE, stroke_width=6, add_vertex_dots=False,
                                          stroke_opacity=opac).set_z_index(4)
                area2 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                           np.concatenate(([0], yvals2.imag, [0])), stroke_width=0, stroke_opacity=0,
                                           fill_opacity=0.3 * opac, fill_color=ORANGE, add_vertex_dots=False)
                res += plt2
                res += area2

            shift = rotate_vector(v.copy(), -t*theta) - v
            res.shift(shift)

            return res

        self.wait(0.1)
        obj1 = get_obj()
        yvals = (np.abs(xvals) < PI) * math.sqrt(2/PI)
        plt2_ = ax2.plot_line_graph(xvals, yvals, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(4)
        area2_ = ax2.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False)

        self.play(mh.rtransform(plt1['line_graph'], obj1[0], area1['line_graph'], obj1[1],
                                plt2, plt2_, area2, area2_))
        self.wait(0.1)
        val_op.set_value(0.)
        do_complex[0] = True
        obj2 = always_redraw(get_obj)
        self.add(obj2)
        self.play(value_t.animate().set_value(1.),
                  val_op.animate(run_time=0.25, rate_func=smooth).set_value(1.),
                  )
        obj3 = get_obj()
        self.add(obj3)
        obj2.set_opacity(0)
        val_op.set_value(0.)
        self.wait(0.1)
        self.play(value_t.animate(rate_func=smooth).set_value(2.),
                  Succession(
                      val_op.animate(run_time=0.25, rate_func=smooth).set_value(1.),
                      Wait(0.5),
                      val_op2.animate(run_time=0.25, rate_func=smooth).set_value(0.),
                  ))

        self.wait()


class RootTfm(Scene):
    def construct(self):
        eq1 = MathTex(r'\sqrt{\mathcal F}', font_size=70, color=col_op, stroke_width=3)
        eq1[0][-1].set_color(col_ft)
        eq1 = eq_shadow(eq1, bg_stroke_width=12)
        self.add(eq1)


class RootTfmBig(Scene):
    def construct(self):
        eq1 = MathTex(r'\sqrt{\mathcal F}', font_size=70* 1.3 * 1.3, color=col_op, stroke_width=3)
        eq1[0][-1].set_color(col_ft)
        eq1 = eq_shadow(eq1, bg_stroke_width=18)
        self.add(eq1)

class Intro_FT_Decomp(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = BLUE if config.transparent else BLACK
        Scene.__init__(self, *args, **kwargs)

    xmax = 6.

    def construct(self):
        xmax = self.xmax
        ax = Axes(x_range=[-xmax, xmax*1.1], y_range=[-1, 1.], x_length=6, y_length=2,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "shade_in_3d": True,
                               },
                  ).set_z_index(1)
        ax.to_edge(DOWN, buff=0.1).to_edge(RIGHT, buff=0.4)
        eq1 = MathTex(r'\sf time', color=col_txt, stroke_width=1.5)
        eq1.next_to(ax.x_axis.get_right(), DOWN, buff=0.2).shift(LEFT*0.3)
        eq1 = eq_shadow(eq1, bg_stroke_width=12, fg_z_index=10, bg_z_index=9.9)
        ax2 = ax.copy().to_edge(LEFT, buff=0.4)
        eq1_f = MathTex(r'\sf freq', color=col_txt, stroke_width=1.5)
        eq1_f.next_to(ax2.x_axis.get_right(), DOWN, buff=0.2).shift(LEFT*0.3)

        freq = 0.5

        xvals = np.linspace(-xmax, xmax, 600)
        yvals = np.exp(xvals*freq * 2 * PI * 1j)*0.9
        plt1 = ax.plot_line_graph(xvals, yvals.real, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(4)
        area1 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals.real, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False).set_z_index(1)
        plt2 = ax.plot_line_graph(xvals, yvals.imag, stroke_color=ORANGE, stroke_width=6, add_vertex_dots=False).set_z_index(3.9)
        area2 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals.imag, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=ORANGE, add_vertex_dots=False).set_z_index(0.9)

        params1 = gauss1d(3, shift=4)
        params2 = gauss1d(0.4, shift=-0.13, c=0.7)
        params3 = gauss1d(6, shift=-3.13, c=-0.7)
        params4 = gauss1d(6, shift=-5, c=0.7)
        params = params1 + params2 + params3 + params4

        yvals2 = gauss1d_calc(params, torch.from_numpy(xvals))

        plt3 = ax2.plot_line_graph(xvals, yvals2.real, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(4)
        area3 = ax2.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals2.real, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False).set_z_index(1)

        params2 = gauss_reflect(gauss_tfm(params))
        yvals_2 = gauss1d_calc(params2, torch.from_numpy(xvals))
        plt1_2 = ax.plot_line_graph(xvals, yvals_2.real, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(4)
        area1_2 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals_2.real, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False).set_z_index(1)
        plt2_2 = ax.plot_line_graph(xvals, yvals_2.imag, stroke_color=ORANGE, stroke_width=6, add_vertex_dots=False).set_z_index(3.9)
        area2_2 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals_2.imag, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=ORANGE, add_vertex_dots=False).set_z_index(0.9)

        MathTex.set_default(font_size=60, stroke_width=2)
        eq2 = MathTex(r'f(t)', r'=', r'e^{2\pi i\nu t}')
        eq3 = MathTex(r'g(\nu)')
        eq4 = MathTex(r'f(t)', r'=', r'\int', r'e^{2\pi i\nu t}', r'g(\nu)', r'd\nu')
        eq5 = MathTex(r'g(\nu)', r'=', r'\int', r'e^{-2\pi i\nu t}', r'f(t)', r'dt')
        eq6 = MathTex(r'f(t)', r'=', r'\mathcal F^{-1}g(t)')
        eq7 = MathTex(r'g(\nu)', r'=', r'\mathcal Ff(\nu)')
        eq8 = MathTex(r'\mathcal F f(\nu)', font_size=80)
        eq9 = MathTex(r'\mathcal F\mathcal F f(t)', font_size=80)
        eq9_1 = MathTex(r'\mathcal F\mathcal F f(t)', r'=', r'f(-t)', font_size=80)
        eq10 = MathTex(r'\mathcal F\mathcal F\mathcal F f(\nu)', font_size=80)
        eq11 = MathTex(r'\mathcal F\mathcal F\mathcal F\mathcal F f(t)', font_size=80)

        mh.rtransform.copy_colors = True
        VGroup(eq2[0][0], eq3[0][0], eq9_1[-1][0]).set_color(col_psi)
        VGroup(eq2[0][2], eq5[-1][1], eq6[2][5], eq9[0][-2], eq11[0][-2], eq9_1[-1][3]).set_color(col_x)
        VGroup(eq2[2][0], eq5[3][0]).set_color(col_special)
        VGroup(eq2[2][1:3]).set_color(col_pi)
        VGroup(eq2[2][3]).set_color(col_i)
        VGroup(eq2[2][-1]).set_color(col_x)
        VGroup(eq2[2][-2], eq3[0][2], eq4[-1][1], eq7[2][3], eq10[0][-2]).set_color(col_angle)
        VGroup(eq4[2], eq4[-1][0], eq5[-1][0]).set_color(col_op)
        VGroup(eq6[2][:3], eq7[2][0], eq9[0][0], eq10[0][0], eq11[0][0]).set_color(col_ft)
        mh.copy_colors_eq(eq3[0], eq4[-2])
        mh.copy_colors_eq(eq4[2], eq5[2], eq2[0], eq5[-2], eq2[2][1:], eq5[3][2:])
        eq2 = eq_shadow(eq2, bg_stroke_width=12)
        eq3 = eq_shadow(eq3, bg_stroke_width=12)
        eq4 = eq_shadow(eq4, bg_stroke_width=12)
        eq5 = eq_shadow(eq5, bg_stroke_width=12)
        eq6 = eq_shadow(eq6, bg_stroke_width=12)
        eq7 = eq_shadow(eq7, bg_stroke_width=12)
        eq8 = eq_shadow(eq8, bg_stroke_width=12)
        eq9 = eq_shadow(eq9, bg_stroke_width=12)
        eq9_1 = eq_shadow(eq9_1, bg_stroke_width=12)
        eq10 = eq_shadow(eq10, bg_stroke_width=12)
        eq11 = eq_shadow(eq11, bg_stroke_width=12)

        eq2.next_to(ax, UP, buff=0.6)
        eq3.next_to(ax2, UP, buff=0.6)
        mh.align_sub(eq4, eq4[0], eq2[0]).move_to(ax, coor_mask=RIGHT).shift(LEFT*0.05)
        mh.align_sub(eq5, eq5[0], eq3[0]).move_to(ax2, coor_mask=RIGHT)
        mh.align_sub(eq6, eq6[1], eq4[1]).move_to(ax, coor_mask=RIGHT).shift(RIGHT*0.1)
        mh.align_sub(eq7, eq7[1], eq5[1]).move_to(ax2, coor_mask=RIGHT)
        eq8.move_to(eq7[2]).move_to(ax2, coor_mask=RIGHT)
        eq9.move_to(eq8).move_to(ax2, coor_mask=RIGHT)
        eq9_1.move_to(eq8).move_to(ax2, coor_mask=RIGHT)
        eq10.move_to(eq8).move_to(ax2, coor_mask=RIGHT)
        eq11.move_to(eq8).move_to(ax2, coor_mask=RIGHT)

        self.add(ax.x_axis, eq1, eq2)

        self.play(Create(plt1, run_time=1.5, rate_func=linear),
                  Succession(Wait(0.5), FadeIn(area1)),
                  Create(plt2, run_time=1.5, rate_func=linear),
                  Succession(Wait(0.9), FadeIn(area2)),
                  )
        self.wait(0.1)
        circ1 = mh.circle_eq(eq2[2][-2], scale=0.4).set_z_index(10)
        txt1 = Tex(r'\sf frequency', color=RED, font_size=50).next_to(circ1, UP, buff=0.1)
        txt1 = eq_shadow(txt1, bg_stroke_width=12)
        self.play(Create(circ1, run_time=0.3, rate_func=linear),
                  Succession(FadeIn(txt1)))
        self.wait(0.1)
        self.play(FadeOut(circ1, txt1))

        self.wait(0.1)
        self.play(FadeIn(ax2.x_axis, eq1_f))
        self.play(Create(plt3, run_time=1., rate_func=linear),
                  Succession(Wait(0.2), FadeIn(area3)),
                  FadeIn(eq3)
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:2], eq4[:2], eq2[2], eq4[3]),
                  Succession(Wait(0.3), FadeIn(eq4[2], eq4[-2:])),
                  mh.rtransform(*[_['line_graph'] for _ in [plt1, plt1_2, plt2, plt2_2, area1, area1_2, area2, area2_2]])
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq3[0], eq5[0]),
                  Succession(Wait(0.3), FadeIn(eq5[1:])))

        circ2 = mh.circle_eq(MathTex('H').move_to(eq5[3][1]), scale=0.4, stroke_width=12).set_z_index(10)
        self.play(Create(circ2, rate_func=linear, run_time=0.5))

        self.play(mh.rtransform(eq4[:2], eq6[:2], eq4[4][:2], eq6[2][3:5], eq4[4][-1], eq6[2][-1]),
                  mh.fade_replace(eq4[4][2], eq6[2][5], coor_mask=RIGHT),
                  FadeIn(eq6[2][:3]),
                  FadeOut(eq4[2:4], eq4[-1]),
                  mh.rtransform(eq5[:2], eq7[:2], eq5[4][:2], eq7[2][1:3], eq5[4][-1], eq7[2][-1]),
                  mh.fade_replace(eq5[4][2], eq7[2][3], coor_mask=RIGHT),
                  FadeIn(eq7[2][0]),
                  FadeOut(eq5[2:4], eq5[-1], circ2),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq7[2], eq8[0]),
                  FadeOut(eq7[:2]))
        self.wait(0.1)
        yvals_3 = yvals_2.numpy()[::-1]
        plt1_3 = ax2.plot_line_graph(xvals, yvals_3.real, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(4)
        area1_3 = ax2.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals_3.real, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False).set_z_index(1)
        plt2_3 = ax2.plot_line_graph(xvals, yvals_3.imag, stroke_color=ORANGE, stroke_width=6, add_vertex_dots=False).set_z_index(3.9)
        area2_3 = ax2.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals_3.imag, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=ORANGE, add_vertex_dots=False).set_z_index(0.9)
        plt3_3 = ax2.plot_line_graph(xvals, yvals_3.imag*0, stroke_color=ORANGE, stroke_width=6, add_vertex_dots=False,
                                     stroke_opacity=1).set_z_index(3.9)
        area3_3 = ax2.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals_3.imag*0, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=ORANGE, add_vertex_dots=False).set_z_index(0.9)
        eq1_t = MathTex(r'\sf time', color=col_txt, stroke_width=1.5, font_size=DEFAULT_FONT_SIZE).move_to(eq1_f)

        plt3_3_ = plt3_3.copy()
        area3_3_ = area3_3.copy()
        self.play(FadeIn(plt3_3), run_time=0.6)
        self.play(mh.rtransform(eq8[0][:-2], eq9[0][1:-2], eq8[0][-1], eq9[0][-1]),
                  mh.fade_replace(eq8[0][-2], eq9[0][-2], coor_mask=RIGHT),
                  FadeIn(eq9[0][0]),
                  mh.rtransform(*[_['line_graph'] for _ in [plt3, plt1_3, area3, area1_3,
                                                            plt3_3, plt2_3, area3_3, area2_3
                                                            ]]),
                  mh.fade_replace(eq1_f, eq1_t, coor_mask=RIGHT),
                  # FadeIn(plt2_3, area2_3)
                  )
        self.wait(0.1)
        eq9_ = eq9.copy()
        self.play(mh.rtransform(eq9[0], eq9_1[0]),
                  Succession(Wait(0.2), FadeIn(eq9_1[1:])))
        eq9 = eq9_
        self.wait(0.1)
        self.play(FadeOut(eq9_1[1:]),
                  Succession(Wait(0.2), mh.rtransform(eq9_1[0], eq9[0])))
        plt3_3 = plt3_3_
        area3_3 = area3_3_
        self.wait(0.1)
        yvals_4 = yvals2.numpy()[::-1]
        plt1_4 = ax2.plot_line_graph(xvals, yvals_4.real, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False).set_z_index(4)
        area1_4 = ax2.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                   np.concatenate(([0], yvals_4.real, [0])), stroke_width=0, stroke_opacity=0,
                                   fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False).set_z_index(1)

        self.play(mh.rtransform(eq9[0][:-2], eq10[0][1:-2], eq9[0][-1], eq10[0][-1]),
                  mh.fade_replace(eq9[0][-2], eq10[0][-2], coor_mask=RIGHT),
                  FadeIn(eq10[0][0]),
                  mh.fade_replace(eq1_t, eq1_f, coor_mask=RIGHT),
                  mh.rtransform(*[_['line_graph'] for _ in [plt1_3, plt1_4, area1_3, area1_4, plt2_3, plt3_3, area2_3, area3_3]]),
                  )
        self.wait(0.1)
        shift = ax2.get_center() - ax.get_center()
        self.play(mh.rtransform(eq10[0][:-2], eq11[0][1:-2], eq10[0][-1], eq11[0][-1]),
                  mh.fade_replace(eq10[0][-2], eq11[0][-2], coor_mask=RIGHT),
                  FadeIn(eq11[0][0]),
                  mh.fade_replace(eq1_f, eq1_t, coor_mask=RIGHT),
                  mh.rtransform(*[_['line_graph'] for _ in [plt1_4, plt1_2.copy().shift(shift),
                                                            area1_4, area1_2.copy().shift(shift),
                                                            plt3_3, plt2_2.copy().shift(shift),
                                                            area3_3, area2_2.copy().shift(shift)
                                ]]),
                  )
        self.wait(0.1)
        self.play(FadeOut(eq11[0][:-4]),
                  eq11[0][-4:].animate.move_to(ax2, coor_mask=RIGHT).shift(RIGHT*0.1))

        self.wait()


def _get_cmap(name: str):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)

class STFTExample(Scene):

    # ── tweakable parameters ────────────────────────────────────────────────
    T_MIN,  T_MAX  = -10.0,  10.0     # time  axis  [s]
    F_MIN,  F_MAX  = -10.0, 10.0     # freq  axis  [Hz]
    NT,     NF     = 401,  401
    GAMMA          = 0.50           # colour gamma (< 1 → boost dim regions)
    CAP_PERCENTILE = 99.0           # intensities above this percentile → capped at 1

    # ── colour mapping with percentile cap ──────────────────────────────────

    def _to_rgb(
        self,
        mag: np.ndarray,
        cmap,
    ) -> np.ndarray:
        """
        Convert a (NT, NF) magnitude array to (NF, NT, 3) uint8 RGB.

        Normalisation:
          1. Find the CAP_PERCENTILE-th percentile as the reference maximum.
          2. Divide by that reference → values in [0, ∞).
          3. Clip to [0, 1]  →  anything above the cap → solid red.
          4. Apply gamma.
          5. Map through jet colourmap.
          6. Flip frequency axis so f = 0 is at the bottom of the image.
        """
        ref = np.percentile(mag, self.CAP_PERCENTILE)
        if ref < 1e-30:
            ref = 1.0   # avoid division by zero for silent frames
        ref = 1.0

        normalised = np.clip(mag / ref, 0.0, 1.0)   # (NT, NF)
        gamma_corr = normalised ** self.GAMMA        # (NT, NF)

        # Transpose so rows = frequency, columns = time  →  (NF, NT)
        img2d = gamma_corr.T

        # Apply colourmap  →  (NF, NT, 4)  float in [0,1]
        rgb = (cmap(img2d)[..., :3] * 255).astype(np.uint8)

        return rgb[::-1]   # flip: low frequency at image bottom

    # ── construct ────────────────────────────────────────────────────────────

    def construct(self):

        cmap    = _get_cmap("jet")
        t_grid  = np.linspace(self.T_MIN, self.T_MAX, self.NT)
        f_grid  = np.linspace(self.F_MIN, self.F_MAX, self.NF)
        t_mesh, f_mesh = torch.meshgrid(torch.tensor(t_grid), torch.tensor(f_grid), indexing='ij')
        xlen = 5.2
        ylen = 5.2

        # ── Axes ─────────────────────────────────────────────────────────────
        ax = Axes(
            x_range=[self.T_MIN, self.T_MAX],
            y_range=[self.F_MIN, self.F_MAX],
            x_length=xlen,
            y_length=ylen,
            axis_config={
                "color": WHITE,
                "include_tip": True,
                "tip_width":  0.15,
                "tip_height": 0.15,
                "stroke_width": 1.8,
            },
        ).shift(LEFT * 0.2)

        ax_labels = ax.get_axis_labels(
            x_label=MathTex(r"\sf time",  font_size=32),
            y_label=MathTex(r"\sf freq", font_size=32),
        )
        ax_labels[0].shift(DOWN*0.4)

        arr_size = 0.05
        ax2 = Axes(x_range=[self.T_MIN, self.T_MAX * (1+arr_size) - arr_size * self.T_MIN],
                   y_range=[-1, 1.], x_length=xlen * (1+arr_size), y_length=2,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(1)
        ax2.next_to(ax, DOWN, buff=0.1).align_to(ax, LEFT)
        VGroup(ax, ax2).move_to(UP*0.2, coor_mask=UP)


        # ── Heatmap bounding box ──────────────────────────────────────────────
        p0 = ax.c2p(self.T_MIN, self.F_MIN)
        p1 = ax.c2p(self.T_MAX, self.F_MAX)
        img_w      = p1[0] - p0[0]
        img_h      = p1[1] - p0[1]
        img_center = np.array([(p0[0]+p1[0])/2, (p0[1]+p1[1])/2, 0.0])

        def make_image(data: np.ndarray, width=img_w, height=img_h, center=img_center) -> ImageMobject:
            obj = ImageMobject(self._to_rgb(data, cmap))
            obj.stretch_to_fit_width(width)
            obj.stretch_to_fit_height(height)
            obj.move_to(center)
            obj.set_z_index(0)
            return obj

        ax.set_z_index(1)
        ax_labels.set_z_index(1)

        # ── Colourbar ─────────────────────────────────────────────────────────
        cb_data = np.broadcast_to(np.linspace(0, 1., 256)[None, :], (20, 256))
        cb_img = make_image(cb_data, width=0.18).set_z_index(2).to_edge(RIGHT)


        eq1 = MathTex(r'\sf time', color=col_txt, font_size=50)
        eq1 = eq_shadow(eq1, bg_z_index=5, fg_z_index=6, bg_stroke_width=10)
        # eq1.align_to(ax, DOWN).shift(UP*0.1)
        eq1.next_to(ax2.x_axis, RIGHT, buff=0.05)
        eq2 = MathTex(r'\sf frequency', color=col_txt, font_size=50).rotate(PI/2)
        eq2 = eq_shadow(eq2, bg_z_index=5, fg_z_index=6, bg_stroke_width=10)
        eq2.next_to(ax, LEFT, buff=-0.05)

        self.add(
            eq1, eq2,
            # cb_img,
            ax2.x_axis
        )

        wave_x = torch.from_numpy(np.linspace(self.T_MIN, self.T_MAX, self.NT))
        np.random.seed(13)
        params0 = []
        for i in range(7):
            u = np.random.normal()
            v = np.random.normal()
            w = np.random.normal()
            p = np.random.normal() * 0.2 + 1
            params0 += gauss1d_p_shift(gauss1d(p * p, c=w, shift=u), p_shift=v)

        noise_f = ValueTracker(-0.7)
        noise_t = ValueTracker(1.)
        noise_c = ValueTracker(1.)
        angle = ValueTracker(0.)

        def get_params():
            params = params0.copy()
            params += gauss1d_p_shift(gauss1d(0.1, shift=-5, c=noise_f.get_value()), p_shift=7)
            params += gauss1d_p_shift(gauss1d(2, shift=7.5, c=noise_t.get_value()), p_shift=4)
            params += gauss1d_p_shift(gauss1d(0.1 + 1j, shift=-4, c=noise_c.get_value()), p_shift=-4)
            params = gauss_fractional_ft(params, angle.get_value())
            return gauss_conj(params)


        def objfunc():
            params = get_params()
            params_wigner = gauss_wigner(params, params)
            params_stft = gauss_smooth(params_wigner, 0.5, 0.5)

            stft = gauss2d_calc(params_stft, t_mesh, f_mesh).real.numpy()
            stft = np.abs(stft)*1.2
            new_img   = make_image(stft)
            return new_img

        def pltfunc():
            params = get_params()
            wave_y = gauss1d_calc(params, wave_x) * 0.8 * torch.exp(wave_x * 1j)
            plt = ax2.plot_line_graph(wave_x, wave_y.real, stroke_color=BLUE, stroke_width=6,
                                         add_vertex_dots=False).set_z_index(14)
            area = ax2.plot_line_graph(np.concatenate(([self.T_MIN], wave_x, [self.T_MAX])),
                                          np.concatenate(([0], wave_y.real, [0])), stroke_width=0, stroke_opacity=0,
                                          fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False).set_z_index(10.9)
            plt2 = ax2.plot_line_graph(wave_x, wave_y.imag, stroke_color=ORANGE, stroke_width=6,
                                         add_vertex_dots=False).set_z_index(13.9)
            area2 = ax2.plot_line_graph(np.concatenate(([self.T_MIN], wave_x, [self.T_MAX])),
                                          np.concatenate(([0], wave_y.imag, [0])), stroke_width=0, stroke_opacity=0,
                                          fill_opacity=0.3, fill_color=ORANGE, add_vertex_dots=False).set_z_index(10.8)
            return VGroup(*[_['line_graph'] for _ in [plt, area, plt2, area2]])

        w3 = 1.6
        box = Rectangle(width=w3, height=img_h * 1.5, stroke_width=0, stroke_opacity=0, fill_opacity=0.3,
                        fill_color=WHITE).set_z_index(5).move_to(img_center).align_to(p0, LEFT)
        outline=Rectangle(width=img_w, height=img_h, fill_opacity=1, fill_color=BLACK,
                          stroke_width=0, stroke_opacity=0).move_to(img_center)
        # box2 = Intersection(box2, outline).set_z_index(5).set_fill(opacity=0.3, color=WHITE).set_stroke(opacity=0, width=0)

        def boxfunc():
            # box1 = Intersection(box.copy().rotate(PI/4-angle.get_value(), about_point=img_center)).set_z_index(5)
            box1 = box.copy().rotate(angle.get_value()-PI/4, about_point=img_center)
            box2 = Intersection(box1, outline).set_z_index(5).set_fill(opacity=0.3, color=WHITE).set_stroke(opacity=0, width=0)

            return box2



        dt1 = 20.5/30
        frame1 = objfunc()
        plt1 = pltfunc()
        self.add(frame1.set_z_index(0), plt1)

        w1 = 1.15
        h1 = frame1.get_top()[1] * 2
        box1 = Rectangle(width=w1, height=h1, stroke_width=0, stroke_opacity=0, fill_opacity=0.3, fill_color=WHITE).set_z_index(2).align_to(p1, RIGHT)
        self.play(FadeIn(box1, rate_func=linear, run_time=dt1))
        self.wait(0.1)

        noise_t.set_value(0.)
        frame2 = objfunc()
        plt2 = pltfunc()
        self.play(mh.rtransform(plt1, plt2), FadeIn(frame2.set_z_index(1)))
        self.remove(frame1)
        self.wait(0.1)
        self.play(FadeOut(box1, rate_func=linear, run_time=dt1))
        self.wait(0.1)
        w2 = 1.2
        box2 = Rectangle(width=img_w, height=w2, stroke_width=0, stroke_opacity=0, fill_opacity=0.3,
                         fill_color=WHITE).set_z_index(1.5).align_to(p0, DL)
        self.play(FadeIn(box2, rate_func=linear, run_time=dt1))
        self.wait(0.1)

        box1 = Rectangle(width=w2, height=h1, stroke_width=0, stroke_opacity=0, fill_opacity=0.3,
                         fill_color=WHITE).set_z_index(4).align_to(p1, RIGHT)

        angle.set_value(PI/2)
        frame1 = objfunc()
        plt1 = pltfunc()
        self.play(mh.rtransform(plt2, plt1), FadeIn(frame1.set_z_index(2), box1))
        self.remove(frame2, box2)
        self.wait(0.1)
        noise_f.set_value(0.)
        frame2 = objfunc()
        plt2 = pltfunc()
        self.play(mh.rtransform(plt1, plt2), FadeIn(frame2.set_z_index(3)))
        self.remove(frame1)
        self.wait(0.1)

        self.play(FadeOut(box1, rate_func=linear, run_time=dt1))
        self.wait(0.1)

        angle.set_value(0)
        frame1 = objfunc()
        plt1 = pltfunc()
        self.play(mh.rtransform(plt2, plt1), FadeIn(frame1.set_z_index(4)))
        self.remove(frame2)
        self.wait(0.1)

        box2 = always_redraw(boxfunc)
        self.play(FadeIn(box2, rate_func=linear, run_time=dt1))
        self.wait(0.1)

        self.remove(plt1, frame1)
        frame2 = always_redraw(objfunc)
        plt2 = always_redraw(pltfunc)
        self.add(frame2.set_z_index(0), plt2)
        self.play(angle.animate.set_value(PI/4), run_time=2)
        self.remove(frame2, plt2, box2)
        frame1 = objfunc()
        plt1 = pltfunc()
        box1 = Rectangle(width=w3, height=img_h, stroke_width=0, stroke_opacity=0, fill_opacity=0.3,
                         fill_color=WHITE).set_z_index(5).move_to(img_center).align_to(p0, LEFT)

        self.add(frame1.set_z_index(0), plt1, box1)
        box2 = Rectangle(width=w3, height=h1, stroke_width=0, stroke_opacity=0, fill_opacity=0.3,
                         fill_color=WHITE).set_z_index(5).align_to(p0, LEFT)
        self.play(mh.rtransform(box1, box2))

        self.wait(0.1)
        noise_c.set_value(0)
        frame2 = objfunc()
        plt2 = pltfunc()
        self.play(mh.rtransform(plt1, plt2), FadeIn(frame2.set_z_index(1)))
        self.remove(frame1)
        self.wait(0.1)
        self.play(FadeOut(box2, rate_func=linear, run_time=dt1))
        self.wait(0.1)

        self.remove(plt2, frame2)
        frame2 = always_redraw(objfunc)
        plt2 = always_redraw(pltfunc)
        self.add(frame2.set_z_index(0), plt2)
        self.play(angle.animate.set_value(0), run_time=2)
        self.remove(frame2, plt2)
        frame1 = objfunc()
        plt1 = pltfunc()
        self.add(frame1.set_z_index(0), plt1)

        self.wait(0.1)
        self.wait()


class Thumb_Plot1(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = BLUE if config.transparent else BLACK
        Scene.__init__(self, *args, **kwargs)

    xmax = 6.
    plot_num=1

    def construct(self):
        xmax = self.xmax
        ax = Axes(x_range=[-xmax, xmax*1.1], y_range=[-1, 1.], x_length=12, y_length=4,
                  axis_config={'color': WHITE, 'stroke_width': 4, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "shade_in_3d": True,
                               },
                  ).set_z_index(1).shift(DOWN)

        freq = 4.

        xvals = np.linspace(-xmax, xmax, 600)
        self.add(ax.x_axis)
        npts = 600
        evolver = WaveEvolver(xrange=(-xmax, xmax), npts=npts, n_extend_left=2000, n_extend_right=2000, n_scale=1,
                              speed=1.)
        evolver.V = evolver.xvals1 ** 2 * 0.5 - 0.5
        evolver.psi = np.clip(1. - (np.abs(evolver.xvals1)-PI)*10, 0, 1)*math.sqrt(2/PI)
        evolver.evolve(PI/2).real
        evolver.speed =-1
        evolver.time = 0.
        origin = mh.pos(DOWN*2)
        v = ax.get_center() - origin
        theta = np.arctan(-v[0]/v[1])

        value_t = ValueTracker(0.)
        val_op = ValueTracker(1.)
        val_op2 = ValueTracker(1.)
        do_complex = [False]

        def get_obj():
            t = value_t.get_value()
            yvals2 = evolver.evolve((t - evolver.time)*PI/4)
            evolver.time = t
            opac = min(val_op.get_value(), val_op2.get_value())
            plt2 = ax.plot_line_graph(xvals, yvals2.real, stroke_color=BLUE, stroke_width=6, add_vertex_dots=False,
                                      stroke_opacity=opac).set_z_index(4)
            area2 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                       np.concatenate(([0], yvals2.real, [0])), stroke_width=0, stroke_opacity=0,
                                       fill_opacity=0.3*opac, fill_color=BLUE, add_vertex_dots=False)
            res = VGroup(plt2['line_graph'], area2['line_graph'], ax.x_axis.copy())

            if do_complex[0]:
                plt2 = ax.plot_line_graph(xvals, yvals2.imag, stroke_color=ORANGE, stroke_width=6, add_vertex_dots=False,
                                          stroke_opacity=opac).set_z_index(4)
                area2 = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                           np.concatenate(([0], yvals2.imag, [0])), stroke_width=0, stroke_opacity=0,
                                           fill_opacity=0.3 * opac, fill_color=ORANGE, add_vertex_dots=False)
                res += plt2
                res += area2

            shift = rotate_vector(v.copy(), -t*theta) - v
            res.shift(shift)

            return res



        if self.plot_num == 1:
            obj1 = get_obj()
            self.add(obj1)
        if self.plot_num == 2:
            yvals = (np.abs(xvals) < PI) * math.sqrt(2 / PI)
            plt2_ = ax.plot_line_graph(xvals, yvals, stroke_color=BLUE, stroke_width=6,
                                       add_vertex_dots=False).set_z_index(4)
            area2_ = ax.plot_line_graph(np.concatenate(([-xmax], xvals, [xmax])),
                                        np.concatenate(([0], yvals, [0])), stroke_width=0, stroke_opacity=0,
                                        fill_opacity=0.3, fill_color=BLUE, add_vertex_dots=False)
            self.add(plt2_, area2_)
        if self.plot_num == 3:
            do_complex[0] = True
            value_t.set_value(1.)
            obj1 = get_obj()
            self.add(obj1)

class Thumb_Plot2(Thumb_Plot1):
    plot_num = 2

class Thumb_Plot3(Thumb_Plot1):
    plot_num = 3
