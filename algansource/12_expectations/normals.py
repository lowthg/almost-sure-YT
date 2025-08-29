import numpy as np
import torch
from algan import *
import manim as mn
import math

from algan.external_libraries.manim import ArcBetweenPoints
from algan.rendering.post_processing.bloom import bloom_filter, bloom_filter_premultiply
from functools import partial
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader
from manim import VGroup

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

def expectedXY(render_settings=LD):
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

    ax[0].submobjects[0].set(shade_in_3d=True)
    ax_o = ax.coords_to_point(0, 0)
    ax.shift(-ax_o)
    ax_o = mn.ORIGIN
    xlen = ax.coords_to_point(xmax, 0)[0] - ax_o[0]
    ylen = ax.coords_to_point(0, 1)[1] - ax_o[1]

    plt = ax.plot(p0, x_range=[-xmax, xmax], color=mn.BLUE).set_z_index(2)
    fill1 = ax.get_area(plt, color=mn.BLUE, opacity=0.5).set_z_index(2)
    #xvals = np.linspace(-xmax, xmax, 200)
    #yvals = np.array([p0(x) for x in xvals])
    #eq1 = MathTex(r'p(a)=\frac1{\sqrt{2\pi}}e^{-\frac12a^2}', font_size=35, shade_in_3d=True)[0]
    #eq2 = MathTex(r'{p(\bf v)=\frac1{2\pi\lvert\Sigma\rvert^{\frac12}}e^{-\frac12v^T\Sigma^{-1} v}}', font_size=35,
    #              color=WHITE, stroke_width=1.7, stroke_color=WHITE, shade_in_3d=True)[0]

    #eq1.set_z_index(3).move_to(ax.coords_to_point(-xmax, 1.1), UL)
    #eq2.set_z_index(3).move_to(ax.coords_to_point(-xmax, 1), UL)
    print('running 1')
    mn.VGroup(fill1).shift(mn.IN*0.01)
    gp1 = mn.VGroup(ax, fill1).rotate(PI / 2, axis=mn.RIGHT, about_point=ax_o) # eq1, eq2
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

    Scene.wait(0.2)
    n = int(render_settings.frames_per_second * 1.5 + 0.5)
    with Lag(run_time=1.5, lag_ratio=0.5):
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

    print('running 2')

    with Seq():
        cam.set_euler_angles(20, 0, 30)
    # self.move_camera(phi=70 * DEGREES, theta=-120 * DEGREES)

    #sq1 = mn.Surface(lambda u, v: u * mn.RIGHT + v * mn.UP, u_range=[-xlen, xlen], v_range=[-xlen, xlen], fill_opacity=0.3,
    #              stroke_opacity=0, stroke_width=0.5, checkerboard_colors=[mn.RED_D, mn.RED_E])
    #sq1_1 = sq1.copy().set_fill(opacity=0).set_stroke(opacity=0.4).shift(mn.OUT*0.1)
    #sq1_a = ManimMob(sq1)
    #sq1_1_a = ManimMob(sq1_1)
    #sq1 = sq1_a
    def f2(u):
        return torch.mul(u[:,:,:1] * 2 - 1, RIGHT * xlen) + torch.mul(u[:,:,1:2] * 2 - 1, UP * xlen)
    #sq1 = Surface(f2, grid_height=32, grid_width=32, color=RED_D, checkered_color=BLUE, opacity=0.3)

    mob1 = surfaceImage()
    mob1.set_location_by_function(f2)
    with Off():
        gp1 = Group(ax_a, fill1, plt1)
        gp2 = gp1.clone().spawn()
    mob1.set(opacity=0.5)

    with Sync(run_time=2):
        gp2.orbit_around_point(ORIGIN, 90, OUT)
        #sq1.spawn()
        # mob1.set_shape_to(sq1)
        with Sync(rate_func=rate_funcs.identity):
            mob1.spawn()

    print('running 3')

    with Sync(run_time=1):
        gp1[1:].move(xlen * UP)
        gp2[1:].move(xlen * LEFT)


    print('running 4')
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

    #surf1 = Surface(p1, grid_height=32, grid_width=32, color=RED_D, checkered_color=BLUE, opacity=0.8)
    #surf2 = Surface(p2, grid_height=32, grid_width=32, color=RED_D).set(opacity=0.8)


    print('running 5')

    #mob2 = surfaceImage()
    #mob2.set_location_by_function(p1)  # .set_opacity(0.99)
    with Sync(run_time=2, rate_func=rate_funcs.ease_out_expo):
        # sq1.become(surf1)
        mob1.set_location_by_function(p1)#.set_opacity(0.99)
        mob1.set(opacity=1)
        #mob2.set_shape_to(surf1.grid)
        #mob1.become(mob2)

    with Seq(run_time=1.2):
        #surf1.become(surf2)
        mob1.set_location_by_function(p2)


    Scene.wait(0.2)

    print('rendering')
    render_to_file('normals', render_settings=render_settings, background_color=BLACK)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 80
    name = expectedXY(render_settings=LD)
