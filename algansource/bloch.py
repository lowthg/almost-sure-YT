import numpy as np
from algan import *
import manim as mn
import math
from algan.rendering.post_processing import bloom_filter, bloom_filter_premultiply
from functools import partial
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader

from manim import Arrow3D, VGroup

LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)


def bloch(r=1.):
    min_op_s1 = 0.1  # opacity of outer sphere at center (minimum opacity)
    max_op_s2 = 0.6  # opacity of inner sphere at center (maximum opacity)
    base_op_s1 = 0.1  # additional opacity for surface
    interior_col = GREEN_C
    # surface_col = Color("#333333")
    surface_col = Color("#000060")
    s1 = Sphere(radius=r, color=surface_col, opacity=1).set_shader(basic_pbr_shader)
    s2 = Sphere(radius=r * 0.995, color=interior_col, opacity=1).set_shader(null_shader)
    s3 = ManimMob(mn.Sphere(radius=r * 1.005, fill_opacity=0, stroke_opacity=0.5, stroke_color=mn.BLACK, stroke_width=2)).orbit_around_point(ORIGIN, 90, RIGHT)
    c1 = Cylinder(radius=0.2, height=1, color=WHITE)
    a = math.log(1 - min_op_s1) * r # min opacity = 1 - exp(a/r)
    b = math.log(1 - max_op_s2) / r  # max opacity = 1 - exp(b * r)

    op = 1 - math.exp(b * r/2)

    center_dot = Sphere(radius=r * 0.04, color=WHITE * op + interior_col * (1-op))

    obj_color = WHITE

    def interior_col_update(obj: Mob, t):
        print('updator called')
        for p in obj.get_descendants():
            loc = p.location
            col = obj_color
            nm = loc.norm(p=2, dim=-1, keepdim=True)
            inside = nm.le(r)
            z = (1 - loc[...,:2].square().sum(dim=-1, keepdim=True)).clamp(0).sqrt()
            depth = z + loc[...,2:3]
            inside_op = 1 - (depth * (b/2)).exp()
            new_col = torch.where(inside, col * (1-inside_op) + interior_col * inside_op, col)
            new_col = torch.where(inside, BLACK, col)
            print(len(t), loc.shape, new_col[0][0])
            p.set_non_recursive(color=new_col)
        return obj

    for p in s1.get_descendants():
        op = 1 - (a / p.location[...,2:3].abs().clamp(0.01)).exp() * (1 - base_op_s1)
        p.set_non_recursive(color=p.color.set_opacity(op))
    for p in s2.get_descendants():
        op = 1 - (p.location[...,2:3].clamp(0., r) * b).exp()
        print(op.max())
        p.set_non_recursive(color=p.color.set_opacity(op))

    with Off():
        s1.spawn()
        s2.spawn()
        s3.spawn()
        c1.spawn().move(IN*2.15)
        s1.smoothness = 0.6
        s1.metallicness = 0.4
        center_dot.spawn()
        light_source = Scene.get_light_sources()[0]
        light_source.move_to(UP * 12)
        light_source.rotate_around_point(ORIGIN, 45, OUT)
        light_source.rotate_around_point(ORIGIN, 35, UP)
        print('location:', light_source.location)
        print('color:', light_source.color)
        print(light_source.location.norm())
        loc = -light_source.location
        loc[...,2] *= -1
        Scene.add_light_source(PointLight(location=loc, color=Color("#444444"), opacity=0.5).spawn())

    print('a')
    c1.add_updater(interior_col_update)
    print('b')

    with Sync(run_time=0.5, rate_func=rate_funcs.identity):
        print('c')
        c1.move(OUT*6)
        print('d')
    print('e')

    return 'bloch'

if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 6
    quality = LD
    r = 2.
    bgcol = DARKER_GREY
    name = bloch(r)
    render_to_file(name, render_settings=quality, background_color=bgcol)
