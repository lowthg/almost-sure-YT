import numpy as np
import torch
from algan import *
import manim as mn
import scipy as sp
from algan.external_libraries.manim.utils.color.SVGNAMES import INDIGO
from manim import VGroup
from pygments.lexer import include

sys.path.append('../../')
import alganhelper as ah
from common.wigner import *
LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 15)


def wigner_anim(quality=LD, bgcol=BLACK, anim=1):
    colors = [
        Color(mn.RED_D.to_rgb() * 0.5/.8),
        Color(mn.RED_E.to_rgb() * 0.5/.8)
    ]
    xmin, xmax = (-5., 5.)
    ymin, ymax = (-5., 5.)
    zmin, zmax = (-.3, .3)
    xlen = 12.
    ylen = 12.
    zlen = 6.
    ax = mn.ThreeDAxes([xmin, xmax * 1.05], [ymin, ymax * 1.1], [zmin, zmax*1.2], xlen, ylen, zlen,
                    axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                                 "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                 "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                 },
                    z_axis_config={'rotation': PI},
                    ).shift(mn.DL * 0.2+mn.IN*0.3)
    origin = torch.tensor(ax.coords_to_point(0, 0), dtype=ORIGIN.dtype)
    right = torch.tensor(ax.coords_to_point(1, 0), dtype=ORIGIN.dtype) - origin
    up = torch.tensor(ax.coords_to_point(0, 1), dtype=ORIGIN.dtype) - origin
    out = torch.tensor(ax.coords_to_point(0, 0, 1), dtype=ORIGIN.dtype) - origin
    txt1 = mn.MathTex(r'X', stroke_width=2, font_size=60).move_to(ax.coords_to_point(xmax * 1.1, 0))
    txt2 = mn.MathTex(r'P', stroke_width=2, font_size=60).move_to(ax.coords_to_point(0, ymax * 1.15))
    txt1.rotate(-PI / 2, mn.RIGHT)
    txt2.rotate(-PI / 2, mn.RIGHT)
    txt2.rotate(PI / 2, mn.OUT)
    ax1 = ManimMob(ax)
    txt1 = ManimMob(txt1)
    txt2 = ManimMob(txt2)
    Group(*ax1.submobjects[:2], txt1, txt2).move(IN*0.11)

    surf = ah.surface_mesh(num_recs=64, rec_size=10, col1=colors[0], col2=colors[1], stroke_color=RED_E,
                           fill_opacity=0.9, stroke_opacity=1)
    surf2 = ah.surface_mesh(num_recs=64, rec_size=10, fill_opacity=1, stroke_opacity=0, add_to_scene=False)
    fill_mask = surf2.get_descendants()[1].color[:,:,-1:]
    mesh_mask = 1 - fill_mask

    p = surf.get_descendants()[1]
    loc = p.location
    col0 = p.color.clone()
    x = loc[:,:,0] * (xmax - xmin) / 2 + (xmax+xmin)/2
    y = loc[:,:,1] * (ymax - ymin) / 2 + (ymax+ymin)/2
    surf.scale(np.array([(xmax-xmin)*right[0]/2, (ymax-ymin)*up[1]/2, 1])).move_to(origin)
    loc = p.location.clone()
    # line = Line(origin, origin+RIGHT, color=RED)


    with Off():
        cam: Camera = Scene.get_camera()
        cam.set_distance_to_screen(12)
        cam.move_to(cam.get_center()*1.4)
        cam.set_euler_angles(70*DEGREES, 0*DEGREES, 60*DEGREES)
        light: PointLight = Scene.get_light_sources()[0]
        light.orbit_around_point(ORIGIN, -90, axis=OUT)
        light.move(UP*4)
        txt1.spawn()
        txt2.spawn()
        ax1.spawn()
        surf.spawn()

    # params1 = gauss1d_std(scale=0.5)
    # params2 = gauss1d_std(scale=2.)
    # params1 = gauss_scale(params1, 1./gauss1d_norm(params1))
    # params2 = gauss_scale(params2, -1./gauss1d_norm(params2))
    # params = params1 + params2

    # params = gauss_shift(params, 3)
    # params = gauss1d_p_shift(params, 3)
    # params = gauss_tfm(params)
    #params += params1
    # params += gauss_scale(gauss_reflect(params), 1.)

    # params = gauss_scale(params, 1./gauss1d_norm(params))
    # wig = gauss_wigner(params, params)


    def f0(t): # flat to round Gaussian
        params = gauss1d_std(scale=1.)
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y) * t

    def f1(t): # round Gaussian shift in X
        params = gauss1d_std(scale=1.)
        params = gauss_shift(params, 3.5 * t)
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f2(t): # round Gaussian shift in P
        params = gauss1d_std(scale=1.)
        params = gauss1d_p_shift(params, 3.5 * t)
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f3(t): # round to squeezed Gaussian
        params = gauss1d_std(scale=math.exp(-t*math.log(2)))
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f4(t): # squeezed to stretched Gaussian
        params = gauss1d_std(scale=math.exp(t*math.log(4))/2)
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f5(t): # stretched gaussian p skew
        params = gauss1d_std(scale=2)
        params = gauss_mult(params, [(-0.4j * t, 0, 0, 0, 0, 1.)])
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f6(t): # stretched Gaussian to squeezed Gaussian via mixed states
        params1 = gauss1d_std(scale=2)
        params2 = gauss1d_std(scale=0.5)
        wig1 = gauss_wigner(params1, params1)
        wig2 = gauss_wigner(params2, params2)
        vals1 = gauss2d_calc(wig1, x, y)
        vals2 = gauss2d_calc(wig2, x, y)
        return vals1 * (1-t) + vals2 * t

    def f7(t): # stretched Gaussian to sum Gaussians
        params1 = gauss1d_std(scale=2)
        params2 = gauss_scale(gauss1d_std(scale=0.5), t)
        params = params1 + params2
        params = gauss_scale(params, 1./gauss1d_norm(params))
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f8(t): # stretched plus squeezed Gaussian phase shift
        params1 = gauss1d_std(scale=2)
        phase = np.exp(t*PI*1j)
        params2 = gauss_scale(gauss1d_std(scale=0.5), phase)
        params = params1 + params2
        params = gauss_scale(params, 1./gauss1d_norm(params))
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f9(t): # stretched gauss diff to stretch gaussian
        params1 = gauss1d_std(scale=math.exp((1-t) * math.log(2) + t*math.log(1.5)))
        params2 = gauss_scale(gauss1d_std(scale=0.5), t-1)
        params = params1 + params2
        params = gauss_scale(params, 1./gauss1d_norm(params))
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f10(t): # stretch gauss to corner
        params = gauss1d_std(scale=1.5)
        params = gauss_shift(params, 3 * t)
        params = gauss1d_p_shift(params, 3 * t)
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    nFT = 4
    def f11(t): # FT of stretched gauss
        params = gauss1d_std(scale=1.5)
        params = gauss_shift(params, 3)
        params = gauss1d_p_shift(params, 3)
        for i in range(nFT-1):
            params = gauss_tfm(params)
        wig1 = gauss_wigner(params, params)
        params = gauss_tfm(params)
        wig2 = gauss_wigner(params, params)
        vals1 = gauss2d_calc(wig1, x, y)
        vals2 = gauss2d_calc(wig2, x, y)
        return vals1 * (1-t) + vals2 * t

    def f12(t): # stretched in corner to round gauss
        params = gauss1d_std(scale=math.exp((1-t)*math.log(1.5)))
        params = gauss_shift(params, 3)
        params = gauss1d_p_shift(params, 3)
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    u0 = 0.
    u1 = 1.

    def f13(t): # round in corner to reflection
        params = gauss1d_std(scale=1.)
        params = gauss_shift(params, 3)
        params = gauss1d_p_shift(params, 3)
        wig1 = gauss_wigner(params, params)
        params = gauss_reflect(params)
        wig2 = gauss_wigner(params, params)
        vals1 = gauss2d_calc(wig1, x, y)
        vals2 = gauss2d_calc(wig2, x, y)
        u = u0 * (1-t) + u1 * t
        return vals1 * (1-u) + vals2 * u

    def f14(t): # mixed state in corners to superposition
        params = gauss1d_std(scale=1.)
        params = gauss_shift(params, 3)
        params = gauss1d_p_shift(params, 3)
        wig1 = gauss_wigner(params, params)
        params2 = gauss_reflect(params)
        wig2 = gauss_wigner(params2, params2)
        params = params + params2
        params = gauss_scale(params, 1./gauss1d_norm(params))
        wig3 = gauss_wigner(params, params)
        vals1 = gauss2d_calc(wig1, x, y)
        vals2 = gauss2d_calc(wig2, x, y)
        vals3 = gauss2d_calc(wig3, x, y)
        vals = (vals1 + vals2) * (0.5 * (1-t)) + vals3 * t
        return vals

    def f15(t): # superposition state in corners phase
        params = gauss1d_std(scale=1.)
        params = gauss_shift(params, 3)
        params = gauss1d_p_shift(params, 3)
        params2 = gauss_reflect(params)
        phase = np.exp(4j * PI * t)
        params += gauss_scale(params2, phase)
        params = gauss_scale(params, 1./gauss1d_norm(params))
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)

    def f16(t): # superposition_to_center
        shift = 3 * (1-t)
        params = gauss1d_std(scale=1.)
        params = gauss_shift(params, shift)
        params = gauss1d_p_shift(params, shift)
        params += gauss_reflect(params)
        params = gauss_scale(params, 1./gauss1d_norm(params))
        wig = gauss_wigner(params, params)
        return gauss2d_calc(wig, x, y)


    col_up = torch.tensor([1, .6, 0.])
    col_dn = INDIGO[:3]
    col = col0.clone()

    u0 = 1.
    u1 = 0.5
    f = f15
    run_time=1.
    rate_func = rate_funcs.smooth
    part = 1

    if anim == 0:
        f = f0
        run_time = 1.
    elif anim == 1:
        run_time = 2.
        f = f1
    elif anim == 2:
        run_time = 2.
        f = f2
    elif anim == 3:
        run_time = 2.
        f = f3
    elif anim == 4:
        run_time = 2.
        f = f4
    elif anim == 5:
        run_time = 1.5
        f = f5
    elif anim == 6:
        run_time = 1.
        f = f6
    elif anim == 7:
        run_time = 1.
        f = f7
    elif 8 <= anim <= 11:
        part = anim - 7
        run_time = 2.
        f = f8
    elif anim == 12:
        run_time = 1.5
        f = f9
    elif anim == 13:
        run_time = 2.
        f = f10
    elif 14 <= anim <= 17:
        run_time = 1.
        f = f11
        nFT = anim - 13
    elif anim == 18:
        run_time = 1.
        f = f12
    elif 19 <= anim <= 20:
        run_time = 1.
        f = f13
        u0, u1 = (0., 1.) if anim == 19 else (1., .5)
    elif 21 <= anim <= 24:
        run_time = 1.5
        f = f14
        part = anim - 20
    elif anim == 25:
        run_time=6.
        f = f15
    elif anim == 26:
        run_time = 4.
        f = f16


    def set_surf(vals):
        loc[...,2] = origin[2] + vals * out[2]
        shade_up = torch.pow(((vals - 0.05)*4).clamp(0, 1), 0.8).unsqueeze(-1)
        shade_down = (vals * -50).clamp(0.,1 ).unsqueeze(-1)
        col[...,:3] = fill_mask * shade_up * col_up\
                        + fill_mask * shade_down * col_dn\
                        + fill_mask * (1-shade_up-shade_down) * col0[:,:,:3]\
                        + mesh_mask * col0[...,:3]
        p.set_non_recursive(location=loc.clone(), color=col.clone())

    def move_view(cam, part=1):
        if part == 1:
            with Sync(run_time=1):
                cam.orbit_around_point(origin, -70*DEGREES, cam.get_right_direction())
        elif part == 2:
            with Off():
                cam.orbit_around_point(origin, -70*DEGREES, cam.get_right_direction())
            with Sync(run_time=2):
                cam.orbit_around_point(origin, 130*DEGREES, cam.get_right_direction())
        else:
            with Off():
                cam.orbit_around_point(origin, 60 * DEGREES, cam.get_right_direction())
            with Sync(run_time=1):
                cam.orbit_around_point(origin, -60*DEGREES, cam.get_right_direction())

    if part == 1:
        for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, step=1, rate_func=rate_func):
            print(frame.index, frame.time, frame.dt)
            with frame.context:
                set_surf(f(frame.u))
    else:
        if part <= 4: f = f8
        else: f = f14
        with Off():
            set_surf(f(1.))
        if part == 2: move_view(cam, 1)
        if part == 3: move_view(cam, 2)
        if part == 4: move_view(cam, 3)
        if part == 5: move_view(cam, 1)
        if part == 6: move_view(cam, 2)
        if part == 7: move_view(cam, 3)

    Scene.wait(1.1/quality.frames_per_second)

    name = 'wigner_anim{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    #COMPUTING_DEFAULTS.max_animate_batch_size = 4
    wigner_anim(quality=HD, bgcol=BLACK, anim=26)