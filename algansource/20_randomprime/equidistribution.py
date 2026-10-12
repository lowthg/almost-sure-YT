import numpy as np
import torch
import colorsys
from algan import *
import manim as mn
import scipy as sp
from algan.external_libraries.manim.utils.color.SVGNAMES import INDIGO, SILVER
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader, default_shader
from manim import MathTex, VGroup

sys.path.append('../../')
import alganhelper as ah
from common.wigner import *
from algansource.wave import setup_wave, set_wave, setup_cam, setup_surf, WaveEvolution
LD = RenderSettings((854, 480), 15)
LD2 = RenderSettings((854, 480), 30)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 15)


def setup_cam():
    with Off():
        cam: Camera = Scene.get_camera()
        cam.set_distance_to_screen(13)
        cam.move_to(cam.get_center()*1.45)
        cam.set_euler_angles(70*DEGREES, 0*DEGREES, 60*DEGREES)
        light: PointLight = Scene.get_light_sources()[0]
        light.orbit_around_point(ORIGIN, -90, axis=OUT)
        light.move(UP*4)


def torus(quality=LD, bgcol=BLACK, anim=0, part=0):

    col1 = BLUE
    col2 = BLUE.clone()
    # col2[:3] *= 1.03
    # col2.clip(0,1)
    stroke_col = BLUE_D
    surf = ah.surface_mesh(num_recs=(20, 60), rec_size=10, fill_opacity=0.7, stroke_opacity=1, add_to_scene=True,
                           col1=col1, col2=col2, stroke_color=stroke_col, offsetx=5, offsety=5)
    psurf = surf.get_descendants()[1]
    loc = psurf.location.clone()
    x = loc[:,:,0]
    y = loc[:,:,1]
    x1 = x.unsqueeze(-1)
    y1 = y.unsqueeze(-1)

    v = PI * x1
    u = PI * y1

    def torus(x, y):
        a = 1.
        b = 3.
        u = PI * x
        v = PI * y
        cosv = a * torch.cos(v) + b
        return (cosv * torch.cos(u) * RIGHT
            + cosv * torch.sin(u) * UP
            + a * torch.sin(v) * OUT)

    x0 = -0.8 / 60 - 2/60 * 7
    y0 =  0.5/20
    def torus_pos(x, y):
        return torus(torch.tensor(x*2+x0), torch.tensor(y*2+y0))

    def torus_norm(x, y):
        u = PI * torch.tensor(x*2 + x0)
        v = PI * torch.tensor(y*2 + y0)
        cosv = torch.cos(v)
        return cosv * torch.cos(u) * RIGHT + cosv * torch.sin(u) * UP + torch.sin(v) * OUT

    with Off():
        setup_cam()
        surf.spawn()

    if anim == 1:
        for frame in ah.FrameStepper(fps = quality.frames_per_second, run_time=1):
            with frame.context:
                psurf.set_non_recursive(location=torus(x1 * frame.u, y1))
    else:
        dot1 = mn.Dot3D(radius=0.12, color=mn.RED)
        dot1 = ManimMob(dot1)
        p0 = torus_pos(0,0)
        dot1.move_to(p0)
        dir0 = torus_pos(0.01,0) - p0
        dir1 = torus_pos(0,0.01) - p0
        dir0 /= torch.linalg.vector_norm(dir0)
        dir1 /= torch.linalg.vector_norm(dir1)
        xvals = torch.linspace(0., 1., 400).unsqueeze(-1)
        pts = torus_pos(xvals, xvals*0)
        normals = torus_norm(xvals, xvals*0)
        col = WHITE.clone()
        col[-1] = 0.5
        crvx = ah.curve_surface(pts, normals, color=col, width=0.06)
        yvals = torch.linspace(0., 1., 100).unsqueeze(-1)
        pts = torus_pos(yvals*0, yvals)
        normals = torus_norm(yvals*0, yvals)
        crvy = ah.curve_surface(pts, normals, color=col, width=0.06)

        with Off():
            psurf.set_non_recursive(location=torus(x1, y1))
            dot1.spawn()
            crvx.spawn()
            crvy.spawn()

    if anim == 2:
        arr1 = mn.Arrow3D(p0.numpy(), (p0+dir0*1.5).numpy(), thickness=0.03)
        arr2 = mn.Arrow3D(p0.numpy(), (p0+dir1*1.5).numpy(), thickness=0.03)
        with Off():
            ManimMob(arr1).spawn()
            ManimMob(arr2).spawn()

    crv_width=0.06

    if anim == 3:
        run_time = 2.
        a = 1.
        b = 1.
    if anim == 4:
        run_time = 4.
        a = 2.
        b = 3.
    if anim == 5:
        run_time = 8
        a = 3.
        b = 4.
    if anim == 6:
        a = 1 * 10
        b = (1 + math.sqrt(5)) / 2 * 10
        run_time = 12

    if 3 <= anim <= 6:
        n = 400
        n = 800
        crv = Surface(grid_height=8, grid_width=n, color=RED)
        des = crv.get_descendants()[1]
        def set_crv(u):
            xvec = torch.linspace(0, u, n).unsqueeze(-1)
            pts = torus_pos(xvec * a, xvec * b)
            normals = torus_norm(xvec, xvec)
            des.set_non_recursive(location=ah.curve_surface_loc(pts, normals, width=crv_width))
        if part == 1:
            set_crv(1)

        with Off():
            crv.spawn()
        for frame in ah.FrameStepper(run_time=run_time, rate_func=rate_funcs.identity, fps=quality.frames_per_second):
            u = frame.u
            with frame.context:
                dot1.move_to(torus_pos(u * a, u * b))
                if part == 0:
                    set_crv(u)



    Scene.wait(0.1)

    name='torus{}'.format(anim)
    if part > 0:
        name = name + '_{}'.format(part)
    render_to_file(name, render_settings=quality, background_color=bgcol)

if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    # torus(quality=LD, bgcol=BLACK, anim=6, part=0)
    torus(quality=HD, bgcol=TRANSPARENT, anim=6, part=0)
