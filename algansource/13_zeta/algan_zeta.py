import numpy as np
import torch
from algan import *
import manim as mn
import math

from algan.external_libraries.manim import ArcBetweenPoints
from algan.rendering.post_processing.bloom import bloom_filter, bloom_filter_premultiply
from functools import partial
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader
from manim import Arrow3D, VGroup

sys.path.append('../')
import alganhelper as ah


LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)



def create_sphere(r=1., simple_sphere=True, anim='', min_op_s1=0.1, max_op_s2 = 0.6, base_op_s1 = 0.1):
    # min_op_s1 = 0.1  # opacity of outer sphere at center (minimum opacity)
    # max_op_s2 = 0.6  # opacity of inner sphere at center (maximum opacity)
    #base_op_s1 = 0.1  # additional opacity for surface
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
        if anim != 'inside':
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
            if anim != 'inside':
                s1.spawn()
            if anim != 'surface':
                s2.spawn()

    return s3, interior_col_update

def sphere_bm(r=2., quality=LD, bgcol=BLACK, scol=YELLOW):
    tscale = 6
    dt = 1. / 60 / tscale
    dev = math.sqrt(dt)
    seeds = [3, 10, 13]
    np.random.seed(seeds[-1])
    pts = [np.zeros(3)]
    while np.linalg.norm(pts[-1]) < 1:
        pts.append(pts[-1] + np.random.normal(0, dev, 3))
    pts[-1] /= np.linalg.norm(pts[-1])
    M = ah.rotation_matrix(RIGHT, PI/6)
    N = ah.rotation_matrix(UP, -PI/6)
    for i in range(len(pts)):
        pts[i] = np.dot(M, pts[i]) * r
        pts[i] = np.dot(N, pts[i])

    n = len(pts)
    t_end = (n-1) * dt
    print('T =', t_end)

    sphere, col_update = create_sphere(r=r, simple_sphere=False, max_op_s2 = 0.8, min_op_s1=0.0002, base_op_s1 = 0.4)
    dot = Sphere(radius=r * 0.06, color=scol)
    col_update(dot, color=scol)
    with Off():
        sphere.spawn()
        dot.spawn()

    fps = quality.frames_per_second
    frame_dt = 1/fps

    m = round(t_end / frame_dt * tscale + 1)
    print('run time:', m*frame_dt)
    j = 0
    for iframe in range(m):
        t = iframe * frame_dt / tscale
        i = min(round(t / dt), n-1)
        with Sync(run_time=frame_dt):
            dot.move_to(pts[i])
            col_update(dot)
            while j < i:
                j += 1
                line = Line(pts[j-1], pts[j], border_width=1.5 * r, border_color=WHITE)
                col_update(line, color=WHITE)
                line.spawn()

    Scene.wait()

    name = 'sphere_bm'
    render_to_file(name, render_settings=quality, background_color=bgcol)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 6

    sphere_bm(quality=HD, bgcol=TRANSPARENT)