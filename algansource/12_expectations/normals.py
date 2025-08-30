import numpy as np
import torch
from algan import *
import manim as mn
import math

from algan.external_libraries.manim import ArcBetweenPoints
from algan.rendering.post_processing.bloom import bloom_filter, bloom_filter_premultiply
from functools import partial
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader
from manim import VGroup, MathTex

sys.path.append('../')
import alganhelper as ah


LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)

colors = [
    #mn.ManimColor(mn.RED_D.to_rgb() * 0.5),
    #mn.ManimColor(mn.RED_E.to_rgb() * 0.5)
    RED_D.clone(),
    RED_E.clone()
]
for _ in colors:
    _[:3] *= 0.5

def surfaceImage(col1=colors[0], col2=colors[1], fill_opacity=0.9, stroke_color=RED_D, stroke_opacity=0.8):
    n = 32
    m = 10
    t = torch.zeros(m * n - 1, m * n - 1, 5)
    sc = stroke_color.clone()
    sc[-1] = stroke_opacity
    for i1 in range(n):
        for j1 in range(n):
            col = col1 if (i1 + j1) % 2 == 0 else col2
            col[-1] = fill_opacity
            for i2 in range(m if i1 < n-1 else m-1):
                for j2 in range(m if j1 < n-1 else m-1):
                    pixel = col if i2 < m-1 and j2 < m-1 else sc
                    t[i1*m+i2, j1*m+j2, :] = pixel

    mob = ImageMob(t)
    return mob

def normals(render_settings=LD, animate=True):
    with Off():
        cam = Scene.get_camera()
        cam.set_distance_to_screen(12)
        cam.move_to(cam.get_center() * 1.4)
        cam.set_euler_angles(-90, 0, 180)

    def p0(x):
        return math.exp(-x * x / 2)

    print('starting')
    xmax = 2.5
    ymax = 1.15

    ax = mn.Axes(x_range=[-xmax, xmax + 0.2], y_range=[0, ymax], x_length=8, y_length=2 * ymax / 1.15,
              axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                           "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           "shade_in_3d": True,
                           },
              #                  shade_in_3d=True,
              ).set_z_index(1)

    ax_o = ax.coords_to_point(0, 0)
    ax.shift(-ax_o)
    ax_o = mn.ORIGIN
    xlen = ax.coords_to_point(xmax, 0)[0] - ax_o[0]
    ylen = ax.coords_to_point(0, 1)[1] - ax_o[1]

    plt = ax.plot(p0, x_range=[-xmax, xmax], color=mn.BLUE).set_z_index(2)
    fill1 = ax.get_area(plt, color=mn.BLUE, opacity=0.5).set_z_index(2)
    eq1 = mn.MathTex(r'p(a)=\frac1{\sqrt{2\pi}}e^{-\frac12a^2}', font_size=35)[0]
    eq1_1 = mn.MathTex(r'p(b)=\frac1{\sqrt{2\pi}}e^{-\frac12b^2}', font_size=35)[0]
    eq2 = mn.MathTex(r'{p(\bf v)=\frac1{2\pi\lvert\Sigma\rvert^{\frac12}}e^{-\frac12{\bf v}^T\Sigma^{-1} {\bf v}}}', font_size=35
                     )[0]
    eq1[2].set_color(mn.RED)
    eq1[-2].set_color(mn.RED)
    eq1_1[2].set_color(mn.BLUE)
    eq1_1[-2].set_color(mn.BLUE)
    col = mn.ManimColor.from_hex('#FFD1FF')
    eq2[2].set_color(col)
    eq2[-6].set_color(col)
    eq2[-1].set_color(col)
    eq1.move_to(ax.coords_to_point(-xmax, 1.1), mn.UL)
    eq1_1.move_to(eq1)
    eq2.move_to(ax.coords_to_point(-xmax, 1), mn.UL)
    print('running 1')
    mn.VGroup(fill1).shift(mn.IN*0.01)
    gp1 = mn.VGroup(ax, fill1, eq1, eq1_1, eq2).rotate(PI / 2, axis=mn.RIGHT, about_point=ax_o) # eq1, eq2
    gp1.rotate(PI, axis=mn.OUT, about_point=ax_o)

    ax_r = ax.coords_to_point(1, 0) - ax_o
    ax_u = ax.coords_to_point(0, 1) - ax_o
    def f(t):
        t1 = t.item()
        res = ax_o + ax_r * t1 + ax_u * p0(t1)
        res1 = RIGHT * res[0] + UP * res[1] + IN * res[2]
        res2 = res1[None, :]
        return res2

    # eq2.shift(DOWN * xlen / 2)
    ax_a = ManimMob(ax)
    # plt1_a = ManimMob(plt1)
    fill1 = ManimMob(fill1)

    with Off():
        ax_a.spawn()
    eq1 = ManimMob(eq1)
    eq1_1 = ManimMob(eq1_1)
    if animate:
        Scene.wait(0.2)
        n = int(render_settings.frames_per_second * 1.5 + 0.5)
        with Lag(run_time=1.5, lag_ratio=0.5):
            with Sync(run_time=1.5):
                eq1.spawn()
                with Seq():
                    for i in range(n):
                        t = i/(n-1)
                        x1 = xmax*t - xmax*(1-t)
                        plt2 = ah.curve3d(f, -xmax, x1, npts=100, border_color=BLUE, border_width=2.5, add_to_scene=False)
                        with Sync(run_time=1.5 / n, rate_func=rate_funcs.identity):
                            if i == 0:
                                plt1 = plt2.clone(add_to_scene=True).spawn()
                            else:
                                plt1.control_points.location = plt2.control_points.location

                #plt1_a.spawn()  # linear
                # eq1 spawn
            fill1.spawn()
        Scene.wait(0.1)
    else:
        plt1 = ah.curve3d(f, -xmax, xmax, npts=100, border_color=BLUE, border_width=2.5)
        with Off():
            fill1.spawn()
            eq1.spawn()
            plt1.spawn()

    print('running 2')

    with Seq() if animate else Off():
        cam.set_euler_angles(20, 0, 30)
    def f2(u):
        return torch.mul(u[:,:,:1] * 2 - 1, RIGHT * xlen) + torch.mul(u[:,:,1:2] * 2 - 1, UP * xlen)

    mob1 = surfaceImage()
    with Off():
        gp1 = Group(ax_a, fill1, plt1, eq1)
        gp2 = Group(ax_a, fill1, plt1, eq1_1).clone().spawn()

    def p1(u):
        x = (u[:,:,:1] * 2 - 1) * xmax
        y = (u[:,:,1:2] * 2 - 1) * xmax
        z = torch.exp(-0.5 * (x*x + y*y))
        return torch.mul(x, RIGHT * xlen/xmax) + torch.mul(y, UP * xlen/xmax) + torch.mul(z, IN*ylen)
        #return (RIGHT * x + UP * y) * xlen/xmax + IN * math.exp(-(x*x+y*y)/2) * ylen

    def p2(u):
        x = (u[:,:,:1] * 2 - 1) * xmax
        y = (u[:,:,1:2] * 2 - 1) * xmax
        z = torch.exp(-0.5 * (x*x + y*y - x*y))
        return torch.mul(x, RIGHT * xlen/xmax) + torch.mul(y, UP * xlen/xmax) + torch.mul(z, IN*ylen)

    eq2 = ManimMob(eq2)
    with Off():
        eq2.move(DOWN*1.2)

    if animate:
        mob1.set_location_by_function(f2)
        d = mob1.get_descendants()
        c = d[1].color
        c1 = c.clone()
        c[:,:,-1] = 0.3

        with Sync(run_time=1):
            gp2.orbit_around_point(ORIGIN, 90, OUT)
            with Sync(rate_func=rate_funcs.identity):
                mob1.spawn()

        with Sync(run_time=1):
            gp1[1:].move(xlen * UP)
            gp2[1:].move(xlen * LEFT)

        with Sync(run_time=2, rate_func=rate_funcs.ease_out_expo):
            mob1.set_location_by_function(p1)
            d[1].color = c1
            eq2.spawn()

        with Seq(run_time=1.2):
            mob1.set_location_by_function(p2)

        Scene.wait(0.2)
    else:
        mob1.set_location_by_function(p2)
        with Off():
            mob1.spawn()
            gp2.orbit_around_point(ORIGIN, 90, OUT)
            gp1[1:].move(xlen * UP)
            gp2[1:].move(xlen * LEFT)
            eq2.spawn()

    return eq1, gp2[-1], eq2, gp1[:-1], gp2[:-1], mob1, xmax, xlen, ylen


def animate_normals(render_settings=LD):
    normals(render_settings=render_settings)
    render_to_file('normals', render_settings=render_settings, background_color=BLACK)


def expectedXY(render_settings=LD):
    eq1, eq1_1, eq2, gp1, gp2, surf, xmax, xlen, ylen = normals(render_settings=render_settings, animate=False)
    with Off():
        eq1.despawn()
        eq1_1.despawn()
        gp1[1:].despawn()
        gp2[1:].despawn()
        #eq2.despawn()
    ax1 = gp1[0]
    ax2 = gp2[0]

    def p3(x, y):
        a = torch.sqrt(2 * torch.log(x))
        b = torch.sqrt(2 * torch.log(y))
        #res = torch.exp(-(a*a+b*b+a*b)*3/2)/(x*y*torch.clamp(a*b,0.01))
        res = torch.exp(-(a * a + b * b + a * b) * 3 / 2) + torch.exp(-(a * a + b * b - a * b) * 3 / 2)
        res = res/(x*y * (a*b).clamp(0.01)) * 10
        return res

    a = math.exp(xmax * xmax / 6) - 1

    def f3(u):
        x1 = (u[:,:,:1] * 2 - 1) * xmax
        y1 = (u[:,:,1:2] * 2 - 1) * xmax
        x = a * (1 - u[:,:,:1]) + 1
        y = a * (1 - u[:,:,1:2]) + 1
        z = p3(x, y)
        z = z.clamp(0, 2*ylen) + ylen*(1-torch.exp(-0.4*(z/ylen-2).clamp(0)))/0.4
        return torch.mul(x1, RIGHT * xlen/xmax) + torch.mul(y1, UP * xlen/xmax) + torch.mul(z, IN*ylen)

    def get_op(color, location):
        op = (1 - 1.5*(location[:, :, 2] / ylen - 0.9)).clamp(0, 1)
        c = color.clone()
        c[:, :, -1] *= op
        return c

    with Off():
        surf2 = surfaceImage()
        surf2.set_location_by_function(f3)
        d = surf2.get_descendants()[1]
        c = get_op(d.color, d.location)
        eq3 = ManimMob(mn.MathTex(r'1', font_size=45)[0].rotate(PI/2, mn.RIGHT).rotate(PI, mn.OUT)).move(xlen*(UP+RIGHT*1.05))
        eq4 = ManimMob(mn.MathTex(r'X', font_size=45, color=mn.RED)[0].rotate(PI/2, mn.RIGHT).rotate(PI, mn.OUT)).move(xlen*(UP+LEFT*1.4))
        eq5 = ManimMob(mn.MathTex(r'Y', font_size=45, color=mn.BLUE)[0].rotate(PI/2, mn.RIGHT).rotate(PI, mn.OUT)).move(xlen*(DOWN*1.2+RIGHT)+IN*0.5)
        eq7 = mn.MathTex(r'p(x, y)', font_size=45)[0].rotate(PI/2, mn.RIGHT).rotate(PI, mn.OUT)
        eq7[2].set_color(mn.RED)
        eq7[4].set_color(mn.BLUE)
        eq7 = ManimMob(eq7).move_to(eq2.get_center())

    ax3 = mn.Axes(x_range=[-1, 1], y_range=[0, 1], x_length=8, y_length=2.5,
              axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                           "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           },
              )
    ax3.shift(-ax3.coords_to_point(0, 0))
    ax3.rotate(PI / 2, axis=mn.RIGHT, about_point=mn.ORIGIN).rotate(PI, axis=mn.IN, about_point=mn.ORIGIN)
    ax4 = ManimMob(ax3.copy().rotate(PI/2, axis=mn.IN, about_point=mn.ORIGIN)).move(xlen*(UP+RIGHT))
    ax3 = ManimMob(ax3).move(xlen*(UP+RIGHT))

    with Lag(lag_ratio=0.7):
        with Sync(run_time=1.6):
            ax1.submobjects[0].move(xlen * UP)
            ax1.submobjects[1].become(ax3.submobjects[1])
            ax2.submobjects[0].move(xlen * RIGHT)
            ax2.submobjects[1].become(ax4.submobjects[1])
            surf.set_location_by_function(f3)
            eq2.despawn()
            with Seq(rate_func=lambda t: (t*2).clamp(0,1)):
                surf.get_descendants()[1].color = c
        with Sync(run_time=1.):
            eq3.spawn()
            eq4.spawn()
            eq5.spawn()
            eq7.spawn()

    def f(x):
        u = torch.tensor([0.5, x])
        return f3(u[None, None, :])[0]

    def g(u):
        u2 = u.clone()
        u2[:,:,0] = 0.5
        res = f3(u2)
        res[:,:,2] *= u[:, :, 0]
        return res

    surf3 = Surface(g, grid_height=16, grid_width=2, color=BLUE, opacity=1)
    crv = ah.curve3d(f, start=0.0, end=0.9, npts=32, border_width=3., border_color=BLUE, filled=False,
                     texture_grid_size=32)

    for _ in crv.get_descendants():
        op = (1 - 1.5 * (_.location[:, :, 2] / ylen - 0.9)).clamp(0, 1)
        c = _.color.clone()
        if len(c[0]) == 1 and len(op[0]) > 1:
            assert len(_) == len(op[0])
            for i in range(len(_)):
                obj = _[i]
                obj.set(color=c[0][0].set_opacity(op[0][i].item()))

    eq6 = mn.MathTex(r'\sim p(y\vert x)', font_size=45)[0].rotate(PI/2, mn.RIGHT).rotate(-PI/2 * 1.2, mn.OUT)
    eq6[3].set_color(mn.BLUE)
    eq6[5].set_color(mn.RED)
    eq6 = ManimMob(eq6).move(IN * ylen * 0.8)

    obj = surf3.get_descendants()[1]
    loc = obj.location
    n = len(loc[0])//2
    op = (1 - 1.5 * (loc[:, n:, 2] / ylen - 0.9)).clamp(0, 1)
    obj.color[:, n:, -1] *= op
    obj.color[:, :n, -1] *= op

    with Sync(run_time=1.5):
        eq6.spawn()
        crv.spawn()
        surf3.spawn()
        eq7.despawn()

    Scene.wait()

    render_to_file('expectedXY', render_settings=render_settings, background_color=BLACK)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    expectedXY(render_settings=HD)
