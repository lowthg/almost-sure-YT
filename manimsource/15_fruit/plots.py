from manim import *
import numpy as np
import math
import sys

from torch import dtype

sys.path.append('../../')
import manimhelper as mh

def CubicCurve(du=0.01, a=11):
    """
    :return: array of (u, v, w) for curve: vw(v+w)+uw(u+w)+uv(u+v) = auvw
    """

    """
    start with points (1,v,w) where (1+w)v^2 + (1+w^2-aw)v + (1+w)w = 0
    on range -1 <= w <= 0 and v >= 0
    to solve, write b = 2(1+w)/(1+w^2-aw). Solving quadratic gives
    v^2 + 2v/b + w = 0
    v = -wb/(1+sqrt(1+wb^2))
    """
    n = int(1/du + 1)
    w1 = np.linspace(0., -1., n + 1)
    b = (w1 + 1) / (w1*w1 - w1 * a + 1) * 2
    v1 = -w1 * b / (np.sqrt(w1*b*b + 1) + 1)
    u1 = np.ones(n+1, dtype=w1.dtype)

    res = [(u1,v1,w1)]

    """
    points (1, v, w) where (1+w)v^2 + (1+w^2-aw)v + (1+w)w = 0
    on range 0 < w <= 1 and w >= v > 0
    range is [p,1] where w=p solves (1+w)w^2 + (1+w^2-aw)w+(1+w)w=0
    p = c - sqrt(c^2 - 1) where c = (a-2)/4
    then, with b = (aw-w^2-1)/(1+w)/2,
    v^2 - 2bv + w = 0
    v = b - sqrt(b^2-w)
    """

    c = (a-2)/4
    p = c - math.sqrt(c*c-1)
    n = int((1-p)/du+1)
    w1 = np.linspace(p, 1., n+1)
    b = (w1*a - w1*w1 - 1)/(w1+1)/2
    print(b)
    v1 = b - np.sqrt(b*b-w1)
    u1 = np.ones(n+1, dtype=w1.dtype)

    res += [(u1, v1 ,w1)]

    return res

class ShowCurve(ThreeDScene):
    def construct(self):
        len = 5
        range=[-1,1.1]
        ax = ThreeDAxes(x_length=len, y_length=len, z_length=len, x_range=range, y_range=range)
        origin = ax.coords_to_point(0,0,0)
        right = ax.coords_to_point(1,0,0) - origin
        self.set_camera_orientation(phi=70*DEGREES, theta=-45*DEGREES)

        curves = CubicCurve()
        ax.plot_line_graph(curves[0][0], curves[0][1], curves[0][2])

        box = Cube(side_length=right[0]*2, fill_color=GREY, fill_opacity=0.4).move_to(origin)

        plts = []
        for crv in curves:
            plts.append(ax.plot_line_graph(crv[0], crv[1], crv[2], add_vertex_dots=False, line_color=BLUE,
                           stroke_width=6))

        self.add(ax, *plts, box)
