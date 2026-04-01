import numpy as np
import math

import torch
import manim as mn

"""
1d gauss with params = (a,b,c) is c * exp(-ax^2+bx)
2d gauss with params = (a,b,c,d,e,f) is of form f*exp(-ax^2 - by^2 + cxy + dx + ey)
"""

col_psi = (mn.RED-mn.WHITE)*0.8 + mn.WHITE
col_x = (mn.BLUE-mn.WHITE)*0.7 + mn.WHITE
col_p = (mn.GREEN-mn.WHITE)*0.9 + mn.WHITE
col_num = (mn.TEAL_E - mn.WHITE) * 0.5 + mn.WHITE
col_special = (mn.PURPLE - mn.WHITE) * 0.25 + (mn.TEAL_E - mn.WHITE) * 0.25 + mn.WHITE
col_i = (mn.YELLOW - mn.WHITE) * 0.3 + mn.WHITE
col_WVD = (mn.ORANGE-mn.WHITE)*0.9 + mn.WHITE
col_op = (mn.PURPLE-mn.WHITE) * 0.5 + mn.WHITE
col_var = col_special
col_eq = (col_op+mn.WHITE)*0.5
col_txt = (mn.BLUE-mn.WHITE) * 1.0 + mn.WHITE
col_txt2 = (mn.YELLOW-mn.WHITE) * 0.5 + mn.WHITE

def gauss1d_std(mean=0., scale=1.):
    result = (0.5 / scale / scale, 0., 0., mean / scale / scale, 0., 1.)
    result = [tuple(complex(_) for _ in result)]
    return gauss_scale(result, 1. / gauss1d_norm(result))

def gauss2d_calc(params: list, x, y):
    """
    for gauss with given params
    :return:
    sum(Re[f*exp(-ax^2 - by^2 + cxy + dx + ey)])
    """
    if type(x) == np.ndarray:
        assert x.shape == y.shape
        result = np.zeros(x.shape, dtype=np.complex128)
        print('*****************')
    else:
        x = x.to(torch.complex64)
        y = y.to(torch.complex64)
        result = 0.j

    for a, b, c, d, e, f in params:
        result += (torch.exp(x * (x * -a + y * c + d) + y*(y * -b + e)) * f)

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

def gauss_smooth(params, v_x=0., v_p=0.):
    """
    convolve with centered gaussian with covariance matrix
    """
    result = []
    for a, b, c, d, e, f in params:
        det = (1+2*a*v_x)*(1+2*b*v_p) - c*c*v_x*v_p
        m11 = (1+2*b*v_p)/det
        m22 = (1+2*a*v_x)/det
        m12 = c*v_p
        m21 = c*v_x

        a1 = m11*a-m12*c/2
        b1 = m22*b-m21*c/2
        c1= -2 * (m21*a-m22*c/2)
        # c1= -2 * (m12*b-m11*c/2)
        d1 = m11*d + m12*e
        e1 = m21*d + m22*e
        const = 0.5 * (d*v_x*d1 + e*v_p*e1)
        f1 = f * np.exp(const)

        result.append((a1, b1, c1, d1, e1, f1))


    return result