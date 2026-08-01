import colorsys

import numpy as np
import torch
from manim import *
import sys
import scipy as sp
from manim import ManimColor

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
        eq10 = MathTex(r'\mathcal F\mathcal F\mathcal F f(\nu)', font_size=80)
        eq11 = MathTex(r'\mathcal F\mathcal F\mathcal F\mathcal F f(t)', font_size=80)

        mh.rtransform.copy_colors = True
        VGroup(eq2[0][0], eq3[0][0]).set_color(col_psi)
        VGroup(eq2[0][2], eq5[-1][1], eq6[2][5], eq9[0][-2], eq11[0][-2]).set_color(col_x)
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