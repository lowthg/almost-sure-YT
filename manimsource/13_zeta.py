from manim import *
import numpy as np
import math
import sys
import scipy as sp

sys.path.append('../')
import manimhelper as mh

def bm_path(n=100, seed=1):
    np.random.seed(seed)
    t_vals = np.linspace(0, 1, n)
    y_vals = np.zeros(n)
    y_vals[1:] = np.random.normal(scale=np.sqrt(t_vals[1]), size=n - 1).cumsum()
    return t_vals, y_vals


def bm_path_hierarchy(depth=7, seed=1):
    np.random.seed(seed)
    n = 1 << depth
    t_vals = np.linspace(0, 1, n+1)
    y_vals = np.zeros(n)
    y_vals[-1] = np.random.normal()
    #rands = np.random.normal(size=n)
    for level in range(depth-1, -1, -1):
        start = 1 << level
        step = start << 1
        len = 1 << (depth-level-1)
        y_vals[start:step:n] = (y_vals[0:step:n] + y_vals[step:step]) * 0.5 + np.random.normal(scale=1, size=len)

class BMReflection(ThreeDScene):
    def construct(self):
        #t_vals, y_vals = bm_path_hierarchy(depth=5, seed=4)
        t_vals, y_vals = bm_path(n=400, seed=4)
        ymin = -1
        ymax = 1.5
        xlen = 10
        ylen = 6
        a = 1.

        ax = Axes(x_range=[0, 1.1], y_range=[ymin, ymax], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(2)

        plt1 = ax.plot_line_graph(t_vals, y_vals, line_color=BLUE, stroke_width=5, add_vertex_dots=False).set_z_index(5)
        line1 = DashedLine(ax.coords_to_point(0, a), ax.coords_to_point(1, a), stroke_width=4, stroke_color=GREY).set_z_index(1)
        origin = ax.coords_to_point(0, 0)
        self.add(ax, plt1, line1)
        self.wait(0.1)
        gp = VGroup(ax, plt1, line1)
        #self.play(Rotate(gp, angle=PI/6, about_point=origin, axis=UP), run_time=1)
        self.play(Rotate(plt1, angle=PI, about_point=origin, axis=RIGHT),
                  run_time=3)

        self.wait()

