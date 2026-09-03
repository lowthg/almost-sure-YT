"""
helper functions for Manim
"""
from manim import *
import numpy as np
import math


def pos(dir=ORIGIN):
    return RIGHT * config.frame_x_radius * dir[0] + UP * config.frame_y_radius * dir[1]


def coords_to_point(x, y):
    return RIGHT * (x-0.5) * config.frame_width + UP * (y-0.5) * config.frame_height


def align_sub(source, subobject, target, direction=ORIGIN, coor_mask=np.array([1, 1, 1]), **kwargs) -> Mobject:
    """
    move object to align subobject with target
    """
    return source.next_to(target, direction, submobject_to_align=subobject, coor_mask=coor_mask, **kwargs)

def diff(source: Mobject, target: Mobject, coor_mask=UR):
    """
    get difference of positions from source to target
    """
    return (target.get_center() - source.get_center()) * coor_mask

def fade_replace(obj1: Mobject, obj2: Mobject, coor_mask=np.array([1, 1, 1]), path_arc=0., **kwargs):
    """
    Fade out obj1 into obj2
    For when the objects differ so that ReplacementTransform doesn't work
    """
    shift = diff(obj1, obj2) * coor_mask
    return (FadeOut(obj1, shift=shift, path_arc=path_arc, **kwargs),
            FadeIn(obj2, shift=shift, path_arc=path_arc, **kwargs))


def stretch_replace(*obj: Mobject, copy_colors=None, **kwargs):
    """
    Fade out obj1 into obj2, stretching to fit
    For when the objects differ so that ReplacementTransform doesn't work
    """
    if copy_colors is None:
        copy_colors = stretch_replace.copy_colors
    obj1 = []
    obj2 = []
    for i in range(0, len(obj), 2):
        source, target = (obj[i], obj[i+1])
        if copy_colors:
            copy_colors_eq(source, target)
        w1 = source.width
        w2 = target.width
        h1 = source.height
        h2 = target.height
        source2 = target.copy().move_to(source).stretch_to_fit_height(h1).stretch_to_fit_width(w1).set_opacity(0)
        target2 = source.copy().move_to(target).stretch_to_fit_height(h2).stretch_to_fit_width(w2).set_opacity(0)
        obj1 += [source2, source]
        obj2 += [target, target2]
    return ReplacementTransform(VGroup(*obj1), VGroup(*obj2), **kwargs)

stretch_replace.copy_colors = False

def transform(*args, **kwargs):
    """
    for args a1, b1, a2, b2,...
    Transform a1 to b1, a2 to b2, etc
    """
    assert(len(args) % 2 == 0)
    return Transform(VGroup(*args[0::2]), VGroup(*args[1::2]), **kwargs)


def rtransform(*args, copy_colors=None, **kwargs):
    """
    for args a1, b1, a2, b2,...
    ReplacementTransform a1 to b1, a2 to b2, etc
    """
    if copy_colors is None:
        copy_colors = rtransform.copy_colors
    assert(len(args) % 2 == 0)
    if copy_colors:
        for i in range(0, len(args), 2):
            copy_colors_eq(args[i], args[i+1])
    return ReplacementTransform(VGroup(*args[0::2]), VGroup(*args[1::2]), **kwargs)

rtransform.copy_colors = False


def circle_eq(eq, color=RED, stroke_width=10, scale=1.) -> ParametricFunction:
    """
    create red curve around eq
    """
    points = [
        (eq.get_corner(UL) + eq.get_top()) * 0.5 + UP * 0.3 * scale,
        eq.get_corner(UR) + UR * 0.2 * scale + RIGHT * 0.5 * scale,
        eq.get_corner(UR) + UR * 0.05 * scale + RIGHT * 0.5 * scale,
        eq.get_right() + RIGHT * 0.1 * scale,
        eq.get_corner(DR) + DR * 0.05 * scale + RIGHT * 0.5 * scale,
        eq.get_corner(DR) + DR * 0.2 * scale + RIGHT * 0.5 * scale,
        eq.get_corner(DR) + DR * 0.2 * scale + RIGHT * 0.5 * scale,
        eq.get_bottom() + DOWN * 0.2 * scale,
        eq.get_bottom() + DOWN * 0.2 * scale,
        eq.get_corner(DL) + DL * 0.2 * scale + LEFT * 0.5 * scale,
        eq.get_corner(DL) + DL * 0.2 * scale + LEFT * 0.5 * scale,
        eq.get_corner(DL) + DL * 0.05 * scale + LEFT * 0.8 * scale,
        eq.get_corner(UL) + UL * 0.2 * scale + LEFT * 0.8 * scale,
        eq.get_corner(UL) + UL * 0.2 * scale + LEFT * 0.5 * scale,
        (eq.get_corner(UR) + eq.get_top()) * 0.5 + UP * 0.3 * scale,
    ]
    bez = bezier(points)
    plot = ParametricFunction(bez, color=color, stroke_width=stroke_width).set_z_index(2)
    return plot


def brace_label(x):
    return lambda text, font_size: x


class label_ctr(Text):
    def __init__(self, text, font_size):
        Text.__init__(self, text, font_size=font_size, color=RED)


class mathlabel_ctr(MathTex):
    def __init__(self, text, font_size):
        MathTex.__init__(self, text, font_size=font_size)


class mathlabel_ctr2(MathTex):
    def __init__(self, text, font_size):
        MathTex.__init__(self, text, font_size=font_size, color=RED)


def label_ctrMU(text, font_size):
    return MarkupText(text, font_size=font_size, color=RED)

def copy_colors_eq(*eqs, depth=0):
    """
    copy eq colors from source to destination
    input: (source1, destination1, source2, destination2, ...)
    """
    assert depth < 5
    n = len(eqs)
    assert n % 2 == 0
    for j in range(0, n, 2):
        assert len(eqs[j][:]) == len(eqs[j+1][:])
        if eqs[j][0] == eqs[j]:
            eqs[j+1].set_color(eqs[j].color)
        else:
            for i, x in enumerate(eqs[j+1][:]):
                copy_colors_eq(eqs[j][i], x, depth=depth+1)

def copy_eq_colors(destination, source):
    copy_colors_eq(source, destination)

def font_size_sub(eq: Mobject, index: int, font_size: float):
    eq_1 = eq[index].copy()
    pos = eq.get_center()
    eq[index].set(font_size=font_size).align_to(eq_1, RIGHT)
    eq[index:].align_to(eq_1, LEFT)
    return eq.move_to(pos, coor_mask=RIGHT)

def get_xticks(ax, vals, strs=None, scalex=1., label_color=WHITE, buff=0.3, font_size=50, length=0.2):
    if strs is None:
        strs = [r'{}'.format(_) for _ in vals]
    tick_eqs = MathTex(*strs, font_size=font_size, stroke_width=1.5, color=label_color)
    origin = ax.c2p(0, 0)
    tick_eqs.next_to(origin, DOWN, buff=buff)
    tick0 = Line(origin, origin + DOWN * length, stroke_width=6, stroke_color=WHITE)
    ticks = [tick0.copy().shift(ax.c2p(_ * scalex, 0) - origin) for _ in vals]
    for _ in range(len(vals)): tick_eqs[_].move_to(ticks[_], coor_mask=RIGHT)
    return VGroup(*[VGroup(tick, eq) for tick, eq in zip(ticks, tick_eqs[:])]).set_z_index(0.5)

def get_yticks(ax, vals, strs=None, scaley=1., max_width=0.9, center=0., label_color=WHITE, buff=0.3, length=0.2, font_size=50):
    if strs is None:
        strs = [r'{}'.format(_) for _ in vals]
    tick_eqs = [MathTex(str, font_size=font_size, stroke_width=1.5, color=label_color)[0] for str in strs]
    origin = ax.c2p(0, 0)
    for eq in tick_eqs: eq.next_to(origin, LEFT, buff=buff)
    tick0 = Line(origin, origin + LEFT * length, stroke_width=6, stroke_color=WHITE)
    ticks = [tick0.copy().shift(ax.c2p(0, _ * scaley + center) - origin) for _ in vals]
    for _ in range(len(vals)):
        tick_eqs[_].move_to(ticks[_], coor_mask=UP)
        w = tick_eqs[_].width
        if w > max_width:
            tick_eqs[_].scale(max_width/w, about_edge=RIGHT)
    return VGroup(*[VGroup(tick, eq) for tick, eq in zip(ticks, tick_eqs[:])]).set_z_index(0.3)


def rate_func_quad(a=0., b=0.):
    """
    rate func quadratic smooth in/out of widths a and b
    """
    assert a + b <= 1.

    def f(t):
        if t < a:
            res = t * t / (2 - a - b) / a
        elif t > 1 - b:
            res = 1. - (1 - t) * (1 - t) / (2 - a - b) / b
        else:
            res = (a + (t - a) * 2) / (2 - a - b)
        return res

    return f
