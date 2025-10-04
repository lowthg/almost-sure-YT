import numpy as np
import torch
from algan import *
import manim as mn
import math
import functorch
import scipy as sp
import colorsys


from algan.external_libraries.manim import ArcBetweenPoints
from algan.rendering.post_processing.bloom import bloom_filter, bloom_filter_premultiply
from functools import partial
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader
from pygments.styles.dracula import background

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

    sphere, col_update = create_sphere(r=r, simple_sphere=False, max_op_s2 = 0.9, min_op_s1=0.0002, base_op_s1 = 0.4)
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


def surface_func(f, res_x=320, res_y=320, mesh_x=None, mesh_y=None, mesh_col=DARK_BROWN):
    mesh_m = 32
    mesh_n = 32
    n = res_y
    m = res_x
    t = torch.zeros(m, n, 5)
    du = 1 / (m-1)
    dv = 1 / (n-1)
    for i1 in range(m):
        mesh_on1 = -i1 % mesh_m < 1
        for j1 in range(n):
            t[i1, j1, :], op = f(j1 * dv, (m-1-i1) * du)
            mesh_on = mesh_on1 or (-j1 % mesh_n < 1)
            if mesh_on:
                col = mesh_col.clone()
                col[4] = op
                t[i1, j1, :] = col

    mob = ImageMob(t)
    return mob


def zeta_surf(quality=LD, bgcol=BLACK):
    xmin = -18
    xmax = 14
    ymax = 16
    ymin = -16
    zmax = 2
    zmaxplot = 2.5
    ax1 = mn.ThreeDAxes(x_range=[ymin, ymax *1.1], y_range=[xmin, xmax*1.1], z_range=[0, zmax],
                        x_length=6, y_length=6, z_length=1.5,
                        axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                                     "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                     "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                     }
                        )
    ax1.shift(-ax1.coords_to_point(0, 0, 0))
    xscale = torch.tensor(ax1.coords_to_point(0, 1, 0), dtype=RIGHT.dtype)
    yscale = torch.tensor(ax1.coords_to_point(1, 0, 0), dtype=RIGHT.dtype)
    zscale = torch.tensor(ax1.coords_to_point(0, 0, 1), dtype=RIGHT.dtype)
    ax1.shift(mn.IN)
    origin = torch.tensor(ax1.coords_to_point(0, 0, 0), dtype=ORIGIN.dtype)
    ax1 = ManimMob(ax1)

    arr_r = ManimMob(mn.Arrow3D(mn.ORIGIN, mn.RIGHT, color=mn.YELLOW))
    arr_u = ManimMob(mn.Arrow3D(mn.ORIGIN, mn.UP, color=mn.RED))
    arr_o = ManimMob(mn.Arrow3D(mn.ORIGIN, mn.OUT, color=mn.BLUE))

    with Off():
        cam = Scene.get_camera()
        cam.set_distance_to_screen(12)
        #cam.move_to(cam.get_center() * 1.4)
        cam.set_euler_angles(-120, 0, 45)
        ax1.spawn()
        #arr_r.spawn()
        #arr_u.spawn()
        #arr_o.spawn()

    def f2(u):
        x = u[:,:,:1] * (xmax - xmin) + xmin
        y = u[:,:,1:2] * (ymax - ymin) + ymin

        z = torch.tensor(abs(sp.special.zeta(x.numpy() + y.numpy() * 1j))).clamp(0, zmaxplot*1.2) + 0.05

        res =  torch.mul(x, xscale) + torch.mul(y, yscale) + torch.mul(z, zscale)
        res[:,:,2] += origin[2]

        return res

    def g(u, v):
        x = u * (xmax-xmin) + xmin
        z = sp.special.zeta(x + (v * (ymax-ymin) + ymin) * 1j)
        z1 = abs(z)
        col = Color(colorsys.hls_to_rgb(np.angle(z)/(2*PI) +0.05, min(z1 /zmax, 0.7), 1))
        #col[:3] *= min(z1, 1)

        if z1 > zmaxplot:
            col[4] = op = max(1 - (z1/zmaxplot - 1)*10, 0)
        else:
            op = 1.
        col[4] *= 0.9
        #if x <= 1:
        #    col[4] = op = 0
        return col, op

    mob1 = surface_func(g).set_shader(basic_pbr_shader)
    mob1.smoothness = 0
    mob1.metallicness = 0

    eq1 = ManimMob(mn.MathTex(r'x')).move_to(origin + xmax * 1.21 * xscale)
    #eq1.rotate_around_point(eq1.get_center(), 90, axis=yscale)
    eq1.orbit_around_point(eq1.get_center(), -90, axis=yscale)
    eq1.orbit_around_point(eq1.get_center(), -45, axis=zscale)
    eq2 = ManimMob(mn.MathTex(r'y')).move_to(origin + ymax * 1.4 * yscale + zscale * 0.3)
    eq2.orbit_around_point(eq2.get_center(), -90, axis=yscale)
    eq2.orbit_around_point(eq2.get_center(), -45, axis=zscale)
    eq3 = ManimMob(mn.MathTex(r'\zeta(x+iy)')).move_to(origin+zmax * zscale * 1.4 + xscale*4+yscale*3)
    eq3.orbit_around_point(eq3.get_center(), -90, axis=yscale)
    eq3.orbit_around_point(eq3.get_center(), -45, axis=zscale)
    eq4 = ManimMob(mn.MathTex(r'\zeta(x+iy)', stroke_width=10, stroke_color=mn.BLACK)).move_to(origin+zmax * zscale * 1.4 + xscale*4+yscale*3)
    eq4.move(IN * 0.01)
    eq4.orbit_around_point(eq3.get_center(), -90, axis=yscale)
    eq4.orbit_around_point(eq3.get_center(), -45, axis=zscale)

    with Off():
        mob1.set_location_by_function(f2)
        mob1.spawn()
        eq1.spawn()
        eq2.spawn()
        eq3.spawn()
        eq4.spawn()

        p = mob1.get_descendants()[1]
        loc = p.location
        col = p.color
        xproj = xscale / torch.inner(xscale, xscale)
        xcoord = torch.inner(loc, xproj)
        start = ((1 - xcoord) / (1 - xmin)).clamp(0, 1)
        col2 = col.clone()
        col2[start.gt(0)] = 0
        p.set_non_recursive(color=col2)

    Scene.wait(1)

    fps = quality.frames_per_second
    t = 1.
    n = round(fps * t)
    dt = 1. / fps
    a = 0.3
    b = (a * a + (1 - a) * 2 * a)

    for i in range(n+1):
        col2 = col.clone()
        s = i/n
        if s < a:
            s2 = s*s/b
        else:
            s2 = (a*a + (s-a)*2*a)/b
        col2[start.gt(s2)] = 0
        with Sync(rate_func=rate_funcs.identity, run_time=dt):
            p.set_non_recursive(color=col2)

    Scene.wait(1)


    name = 'zeta_surf'
    render_to_file(name, render_settings=quality, background_color=bgcol)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 6

    #sphere_bm(quality=HD, bgcol=TRANSPARENT)
    zeta_surf(quality=LD, bgcol=BLACK)