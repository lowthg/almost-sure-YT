from fontTools.unicodedata import block
from manim import *
import numpy as np
import math
import sys
import scipy as sp
from matplotlib.font_manager import font_scalings
from numpy.random.mtrand import Sequence
from sorcery import switch
from torch.utils.jit.log_extract import run_test

sys.path.append('../')
import manimhelper as mh

"""
1d gauss with params = (a,b,c) is c * exp(-ax^2+bx)
2d gauss with params = (a,b,c,d,e,f) is of form f*exp(-ax^2 - by^2 + cxy + dx + ey)
"""

def gauss2d_calc(params: list, x, y):
    """
    for gauss with given params
    :return:
    sum(Re[f*exp(-ax^2 - by^2 + cxy + dx + ey)])
    """
    if type(x) == np.ndarray:
        assert x.shape == y.shape
        result = np.zeros(x.shape)
    else:
        result = 0.

    for a, b, c, d, e, f in params:
        result += (np.exp(x * (-a * x + c*y + d) + y*(-b*y + e)) * f).real

    return result

def gauss_shift(params, dx=0., dy=0.):
    result = []
    for a, b, c, d, e, f in params:
        result.append((a, b, c, d + 2*a*dx-c*dy, e+2*b*dy-c*dx, f * np.exp(-a*dx*dx-b*dy*dy+c*dx*dy)))
    return result

def gauss_mat(params, m):
    """
    :return: g(x,y) = f(m(x,y))
    """
    result = []
    for a, b, c, d, e, f in params:
        result.append((a*m[0,0]*m[0,0] + b*m[1,0]*m[1,0] - c*m[0,0]*m[1,0],
                       a*m[0,1]*m[0,1] + b*m[1,1]*m[1,1] - c*m[0,1]*m[1,1],
                       -2*a*m[0,0]*m[0,1] - 2*b*m[1,0]*m[1,1] + c*(m[0,0]*m[1,1]+m[0,1]*m[1,0]),
                       d*m[0,0] + e*m[1,0],
                       d*m[0,1] + e*m[1,1], f))
    return result

def gauss_conj(params):
    result = []
    for p in params:
        result.append(tuple(_.conjugate() for _ in p))
    return result

def gauss_scale(params, scale = 1.):
    result = []
    for a, b, c, d, e, f in params:
        result.append((a, b, c, d, e, f*scale))
    return result

def gauss_mult(params1, params2):
    result = []
    for p in params1:
        for q in params2:
            result.append((p[0]+q[0], p[1]+q[1], p[2]+q[2], p[3]+q[3], p[4]+q[4], p[5]*q[5]))
    return result


def gauss_tfm(params, dim=0): # dim to transform
    """
    replace f(x,y) by int f(x,z) exp(izy) dz / sqrt(2pi)
    """
    if dim == 1:
        return gauss_switch(gauss_tfm(gauss_switch(params)))

    assert dim == 0
    result = []
    for a, b, c, d, e, f in params:
        result.append((1/4/a, b-c*c/4/a, c*0.5j/a, d*0.5j/a, e+c*d/2/a, f*np.exp(d*d/4/a)/np.sqrt(2*a)))
    return result

def gauss_switch(params):
    """
    :return: f(y,x)
    """
    result = []
    for a, b, c, d, e, f in params:
        result.append((b, a, c, e, d, f))
    return result

def gauss_wigner(params1, params2):
    p1 = gauss_mat(params1, np.matrix([[1, 0.5], [0, 0]]))
    p2 = gauss_mat(params2, np.matrix([[1, -0.5], [0, 0]]))
    p3 = gauss_mult(gauss_conj(p1), p2)
    p4 = gauss_scale(gauss_tfm(p3, dim=1), 1/math.sqrt(2*math.pi))
    return p4

def gauss1d_int(params):
    result = 0.+0.j
    for a,_,_,b,_,f in params:
        result += np.sqrt(np.pi/a) * f * np.exp(0.25*b*b/a)
    return result

def gauss1d_norm(params):
    return math.sqrt(gauss1d_int(gauss_mult(gauss_conj(params), params)).real)

def gauss1d_p_shift(params, p_shift=0.):
    return gauss_mult(params, [(0., 0., 0., p_shift * 1j, 0., 1.)])

def gauss_reflect(params):
    result = []
    for a, b, c, d, e, f in params:
        result.append((a, b, c, -d, -e, f))
    return result

class Example(ThreeDScene):
    colors = [
        ManimColor(RED_D.to_rgb() * 0.5),
        ManimColor(RED_E.to_rgb() * 0.5)
    ]

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES)

        xmin, xmax = (-5., 5.)
        ymin, ymax = (-5., 5.)
        zmin, zmax = (-.6, .6)
        xlen = 12.
        ylen = 12.
        zlen = 6.

        ax = ThreeDAxes([xmin, xmax], [ymin, ymax], [zmin, zmax], xlen, ylen, zlen,
                        axis_config={'color': WHITE, 'stroke_width': 2, 'include_ticks': False,
                                     "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                                     "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                                     },
                        )

        params1 = [(.5*6, 0., 0., 0., 0., 1.)]
        params2 = [(.5/6, 0., 0., 0., 0., 1.)]
        params1 = gauss_scale(params1, 1./gauss1d_norm(params1))
        params2 = gauss_scale(params2, -1./gauss1d_norm(params2))
        params = params1 + params2
        params = gauss_scale(params, 1./gauss1d_norm(params))

        #params = gauss_shift(params, 3)
        #params = gauss1d_p_shift(params, 3)
        #params = gauss_tfm(params)
        #params += params1
        #params = gauss_reflect(params)
        wig = gauss_wigner(params, params)

        origin = ax.coords_to_point(0, 0, 0)
        right = ax.coords_to_point(1, 0, 0) - origin
        up = ax.coords_to_point(0, 1, 0) - origin
        out = ax.coords_to_point(0, 0, 1) - origin

        def f(u,v):
            x = (1-u)*xmin + u*xmax
            y = (1-v)*ymin + v*ymax
            z = gauss2d_calc(wig, x, y)
            return origin + right * x + up * y + out * z

        surf = Surface(f,
            stroke_opacity=0.8, checkerboard_colors=self.colors, stroke_color=WHITE,
            resolution=(100, 100), should_make_jagged=False).set_z_index(200)
        self.add(ax, surf)
        self.wait()


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True, 'fps': 15}):
        Example().render()

