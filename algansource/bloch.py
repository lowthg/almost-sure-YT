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

def create_sphere(r=1., simple_sphere=True):
    min_op_s1 = 0.1  # opacity of outer sphere at center (minimum opacity)
    max_op_s2 = 0.6  # opacity of inner sphere at center (maximum opacity)
    base_op_s1 = 0.1  # additional opacity for surface
    interior_col = GREEN_C
    # surface_col = Color("#333333")
    surface_col = Color("#000060")
    r2 = r * 0.995
    fill_opacity = 0.4 if simple_sphere else 0
    s3 = ManimMob(mn.Sphere(radius=r * 1.005, fill_opacity=fill_opacity, fill_color = mn.GREEN, stroke_opacity=0.5,
                            stroke_color=mn.BLACK, stroke_width=2, resolution=(24, 12)))
    with Off():
        s3.orbit_around_point(ORIGIN, 90, RIGHT)
        s3.orbit_around_point(ORIGIN, -7, UP)  # don't cover the center with a line
        s3.orbit_around_point(ORIGIN, 35, RIGHT)  # let's make the north pole point slightly towards us
        s3.spawn()

    a = math.log(1 - min_op_s1) * r # min opacity = 1 - exp(a/r)
    b = math.log(1 - max_op_s2) / r  # max opacity = 1 - exp(b * r)

    def interior_col_update(obj: Mob, t=(0.,), color=None):
        # print('updator called')
        if color is not None:
            obj.base_color = color
        col = obj.base_color

        for p in obj.get_descendants():
            loc = p.location
            inside = loc.norm(dim=-1, keepdim=True).le(r2)
            depth = (r2*r2 - loc[...,:2].square().sum(dim=-1, keepdim=True)).clamp(0).sqrt() + loc[...,2:3]
            inside_op = 1 - (depth * (b/2)).exp()
            new_col = torch.where(inside, col * (1-inside_op) + interior_col * inside_op, col)
            # print(len(t), loc.shape)
            p.set_non_recursive(color=new_col)
        return obj

    if not simple_sphere:
        s1 = Sphere(radius=r, color=surface_col, opacity=1).set_shader(basic_pbr_shader)
        s2 = Sphere(radius=r2, color=interior_col, opacity=1).set_shader(null_shader)

        for p in s1.get_descendants():
            op = 1 - (a / p.location[...,2:3].abs().clamp(0.01)).exp() * (1 - base_op_s1)
            p.set_non_recursive(color=p.color.set_opacity(op))
        for p in s2.get_descendants():
            op = 1 - (p.location[...,2:3].clamp(0., r) * b).exp()
            print(op.max())
            p.set_non_recursive(color=p.color.set_opacity(op))

        with Off():
            s1.smoothness = 0.6
            s1.metallicness = 0.4
            s1.spawn()
            s2.spawn()

    return s3, interior_col_update


def bloch(r=1., simple_sphere=False):
    sphere, col_update = create_sphere(r, simple_sphere)
    # c1 = Cylinder(radius=0.2, height=1, color=WHITE)

    center_dot = Sphere(radius=r * 0.04)
    col_update(center_dot, color=WHITE)

    with Off():
        Scene.get_camera().set_distance_to_screen(15)
        # c1.spawn().move(IN*2.3)
        center_dot.spawn()
        light_source = Scene.get_light_sources()[0]
        light_source.move_to(UP * 12)
        light_source.rotate_around_point(ORIGIN, 45, OUT)
        light_source.rotate_around_point(ORIGIN, 20, UP)
        # print('location:', light_source.location)
        # print('color:', light_source.color)
        # print(light_source.location.norm())
        loc = -light_source.location
        loc[...,2] *= -1
        Scene.add_light_source(PointLight(location=loc, color=Color("#444444"), opacity=0.5).spawn())
        # col_update(c1, [0.])

    # c1.add_updater(col_update)

    # with Sync(run_time=3, rate_func=rate_funcs.identity):
        # c1.move(OUT*6)

    s_up = sphere.get_forward_direction()
    s_right = sphere.get_right_direction()
    s_forward = sphere.get_upwards_direction()
    mn_up = s_up.numpy()[0,0,:]
    mn_right = s_right.numpy()[0,0,:]
    mn_forward = s_forward.numpy()[0,0,:]


    print(s_up)
    print(s_right)
    print(s_forward)

    equp = ManimMob(mn.MathTex(r'\lvert{\rm up}\rangle', font_size=32))
    eqdn = ManimMob(mn.MathTex(r'\lvert{\rm down}\rangle', font_size=32))
    eqrt = ManimMob(mn.MathTex(r'\lvert{\rm right}\rangle', font_size=32))
    eqlt = ManimMob(mn.MathTex(r'\lvert{\rm left}\rangle', font_size=32))
    eqft = ManimMob(mn.MathTex(r'\lvert{\rm front}\rangle', font_size=32))
    eqbk = ManimMob(mn.MathTex(r'\lvert{\rm back}\rangle', font_size=32))
    equp.move_to(s_up * r * 1.1)
    eqdn.move_to(-s_up * r * 1.35)
    eqrt.move_to(s_right * r * 1.25)
    eqlt.move_to(-s_right * r * 1.2)
    eqft.move_to(s_forward * r * 1.05)
    eqbk.move_to(-s_forward * r * 1.05)
    # xAx = Line(-s_right * r, s_right * r, color=WHITE, border_width=1)
    #xAx = ManimMob(mn.Line(-mn_right * r, mn_right * r, stroke_width=2))
    # yAx = Line(-s_up * r, s_up * r, border_width=1)
    #yAx = ManimMob(mn.Line(-mn_up * r, mn_up * r, stroke_width=2))
    # zAx = Line(-s_forward * r*0, s_forward * r, border_width=1)
    #zAx = ManimMob(mn.Line(-mn_forward * r*0.5, mn_forward * r * 0.99, stroke_width=1))
    axes = Group(
        ManimMob(mn.Line(ORIGIN, mn_forward * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, -mn_forward * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, mn_up * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, -mn_up * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, mn_right * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, -mn_right * r * 0.99, stroke_width=1))
    )
    col_update(axes, color=WHITE)
    #col_update(yAx, color=WHITE)
    #col_update(zAx, color=WHITE)

    with Sync():
        equp.spawn()
        eqdn.spawn()
        eqrt.spawn()
        eqlt.spawn()
        eqft.spawn()
        eqbk.spawn()
        #xAx.spawn()
        #yAx.spawn()
        #zAx.spawn()
        axes.spawn()


    return 'bloch'

if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 6
    quality = HD
    r = 2.
    bgcol = DARKER_GREY
    name = bloch(r)
    render_to_file(name, render_settings=quality, background_color=bgcol)
