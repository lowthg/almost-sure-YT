import numpy as np
import torch
import colorsys
from algan import *
import manim as mn
import scipy as sp
from algan.external_libraries.manim.utils.color.SVGNAMES import INDIGO

sys.path.append('../../')
import alganhelper as ah
from common.wigner import *
LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 15)


def f0(t):  # flat to round Gaussian
    params = gauss1d_std(scale=1.)
    return [(params, t)]


def f1(t):  # round Gaussian shift in X
    params = gauss1d_std(scale=1.)
    params = gauss_shift(params, 3.5 * t)
    return [(params, 1.)]


def f2(t):  # round Gaussian shift in P
    params = gauss1d_std(scale=1.)
    params = gauss1d_p_shift(params, 3.5 * t)
    return [(params, 1.)]


def f3(t):  # round to squeezed Gaussian
    params = gauss1d_std(scale=math.exp(-t * math.log(2)))
    return [(params, 1.)]


def f4(t):  # squeezed to stretched Gaussian
    params = gauss1d_std(scale=math.exp(t * math.log(4)) / 2)
    return [(params, 1.)]


def f5(t):  # stretched gaussian p skew
    params = gauss1d_std(scale=2)
    params = gauss_mult(params, [(-0.4j * t, 0, 0, 0, 0, 1.)])
    return [(params, 1.)]


def f6(t):  # stretched Gaussian to squeezed Gaussian via mixed states
    params1 = gauss1d_std(scale=2)
    params2 = gauss1d_std(scale=0.5)
    return [(params1, 1 - t), (params2, t)]


def f7(t):  # stretched Gaussian to sum Gaussians
    params1 = gauss1d_std(scale=2)
    params2 = gauss_scale(gauss1d_std(scale=0.5), t)
    params = params1 + params2
    params = gauss_scale(params, 1. / gauss1d_norm(params))
    return [(params, 1.)]


def f8(t):  # stretched plus squeezed Gaussian phase shift
    params1 = gauss1d_std(scale=2)
    phase = np.exp(t * PI * 1j)
    params2 = gauss_scale(gauss1d_std(scale=0.5), phase)
    params = params1 + params2
    params = gauss_scale(params, 1. / gauss1d_norm(params))
    return [(params, 1.)]


def f9(t):  # stretched gauss diff to stretch gaussian
    params1 = gauss1d_std(scale=math.exp((1 - t) * math.log(2) + t * math.log(1.5)))
    params2 = gauss_scale(gauss1d_std(scale=0.5), t - 1)
    params = params1 + params2
    params = gauss_scale(params, 1. / gauss1d_norm(params))
    return [(params, 1.)]


def f10(t):  # stretch gauss to corner
    params = gauss1d_std(scale=1.5)
    params = gauss_shift(params, 3 * t)
    params = gauss1d_p_shift(params, 3 * t)
    return [(params, 1.)]


def f11(t):  # corner gauss cc
    params = gauss1d_std(scale=1.5)
    params = gauss_shift(params, 3)
    params = gauss1d_p_shift(params, 3)
    params2 = gauss_conj(params)
    return [(params, 1 - t), (params2, t)]

def f12(nft=4): # FT of stretched gauss
    def f(t):
        params = gauss1d_std(scale=1.5)
        params = gauss_shift(params, 3)
        params = gauss1d_p_shift(params, 3)
        for i in range(nft-1):
            params = gauss_tfm(params)
        params2 = gauss_tfm(params)
        return [(params, 1-t), (params2, t)]
    return f

def f13(t): # stretched in corner to round gauss
    params = gauss1d_std(scale=math.exp((1-t)*math.log(1.5)))
    params = gauss_shift(params, 3)
    params = gauss1d_p_shift(params, 3)
    return [(params, 1.)]

def f14(u0=0., u1=1.): # round in corner to reflection
    def f(t):
        params = gauss1d_std(scale=1.)
        params = gauss_shift(params, 3)
        params = gauss1d_p_shift(params, 3)
        params2 = gauss_reflect(params)
        u = u0 * (1-t) + u1 * t
        return [(params, 1-u), (params2, u)]
    return f

def f15(t): # mixed state in corners to superposition
    params = gauss1d_std(scale=1.)
    params = gauss_shift(params, 3)
    params = gauss1d_p_shift(params, 3)
    params2 = gauss_reflect(params)
    params3 = params + params2
    params3 = gauss_scale(params3, 1./gauss1d_norm(params3))
    return [(params, (1-t)/2), (params2, (1-t)/2), (params3, t)]

def f16(t): # superposition state in corners phase
    params = gauss1d_std(scale=1.)
    params = gauss_shift(params, 3)
    params = gauss1d_p_shift(params, 3)
    params2 = gauss_reflect(params)
    phase = np.exp(4j * PI * t)
    params += gauss_scale(params2, phase)
    params = gauss_scale(params, 1./gauss1d_norm(params))
    return [(params, 1.)]

def f17(t): # superposition_to_center
    shift = 3 * (1-t)
    params = gauss1d_std(scale=1.)
    params = gauss_shift(params, shift)
    params = gauss1d_p_shift(params, shift)
    params += gauss_reflect(params)
    params = gauss_scale(params, 1./gauss1d_norm(params))
    return [(params, 1.)]

def g1(t):
    params = gauss1d_std(scale=1)
    eps = 0.05
    params1 = gauss_shift(params, eps)
    params2 = gauss_shift(params, -eps)
    # params = params1 + gauss_scale(params2, -1)
    params = params1 + gauss_scale(params2, -1)
    params = gauss_scale(params, 1./gauss1d_norm(params))
    return [(params,1)]

def setup_cam():
    with Off():
        cam: Camera = Scene.get_camera()
        cam.set_distance_to_screen(13)
        cam.move_to(cam.get_center()*1.45)
        cam.set_euler_angles(70*DEGREES, 0*DEGREES, 60*DEGREES)
        light: PointLight = Scene.get_light_sources()[0]
        light.orbit_around_point(ORIGIN, -90, axis=OUT)
        light.move(UP*4)

def setup_surf(xrange=(-5., 5.), yrange=(-5., 5.), zrange=(-.3, .3)):
    xmin, xmax = xrange
    ymin, ymax = yrange
    zmin, zmax = zrange
    xlen = 12.
    ylen = 12.
    zlen = 6.
    ax = mn.ThreeDAxes([xmin, xmax * 1.05], [ymin, ymax * 1.1], [zmin, zmax*1.2], xlen, ylen, zlen,
                    axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                                 "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                 "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                 },
                    z_axis_config={'rotation': PI},
                    ).shift(mn.DL * 0.3 + mn.OUT*0.2)
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

    with Off():
        txt1.spawn()
        txt2.spawn()
        ax1.spawn()

    return origin, right, up, out

def wigner_anim(quality=LD, bgcol=BLACK, anim=1, show_wave=False):
    name = 'wigner_anim{}'.format(anim)
    setup_cam()

    xmin, xmax = xrange = (-5., 5.)
    ymin, ymax = yrange = (-5., 5.)

    origin, right, up, out = setup_surf(xrange, yrange)

    colors = [
        Color(mn.RED_D.to_rgb() * 0.5 / .8),
        Color(mn.RED_E.to_rgb() * 0.5 / .8)
    ]

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
    col_up = torch.tensor([1, .6, 0.])
    col_dn = INDIGO[:3]
    col = col0.clone()
    with Off():
        surf.spawn()

    if show_wave:
        name = 'wigner_wave{}'.format(anim)

        surfx = Surface(grid_height=4, grid_width=640)
        px = surfx.get_descendants()[1]
        locx = px.location.clone()
        colx = px.color.clone()
        xx = locx[:,:,0] * (xmax - xmin) / 2 + (xmax+xmin)/2
        yx = (locx[:,:,1]+1)/2
        n_col = len(colx[0])
        colx[...,4] = 0.75

        surfp = Surface(grid_height=4, grid_width=640)
        pp = surfp.get_descendants()[1]
        locp = pp.location.clone()
        colp = pp.color.clone()
        xp = locp[:,:,0] * (ymax - ymin) / 2 + (ymax+ymin)/2
        yp = (locp[:,:,1]+1)/2
        n_col = len(colx[0])
        colp[...,4] = 0.75

        # originx = torch.tensor(ax.coords_to_point(0, ymax), dtype=ORIGIN.dtype)
        originx = origin + ymax * up
        rightx = right
        upx = out * 0.5
        originp = origin + xmin * right
        # originp = torch.tensor(ax.coords_to_point(xmin, 0), dtype=ORIGIN.dtype)
        rightp = up
        upp = upx


    def set_frame(mixed_params):
        vals = sum([gauss2d_calc(gauss_wigner(params, params), x, y).real * a for params, a in mixed_params])

        loc[...,2] = origin[2] + vals * out[2]
        shade_up = torch.pow(((vals - 0.05)*4).clamp(0, 1), 0.8).unsqueeze(-1)
        shade_down = (vals * -50).clamp(0.,1 ).unsqueeze(-1)
        col[...,:3] = fill_mask * shade_up * col_up\
                        + fill_mask * shade_down * col_dn\
                        + fill_mask * (1-shade_up-shade_down) * col0[:,:,:3]\
                        + mesh_mask * col0[...,:3]
        p.set_non_recursive(location=loc.clone(), color=col.clone())

        if show_wave:
            vals = 0
            vals1 = 0.+0j
            for params, a in mixed_params:
            # params, a = mixed_params[0]
                vals0 = gauss2d_calc(params, xx, xx*0)
                w = vals0.abs()
                vals1 += vals0 * w * a
                vals += w * w * a

            # vals = vals1.abs()
            # vals *= vals
            u = 0.3

            locx[..., :] = xx.unsqueeze(-1) * rightx + yx.unsqueeze(-1) * vals.unsqueeze(-1) * upx + originx
            for i in range(n_col):
                lightness = min(0.15 + vals[0, i] * u * yx[0,i], 0.8)
                colx[0, i, :3] = torch.tensor([*colorsys.hls_to_rgb(np.angle(vals1[0,i])/(2*PI)+0.5, lightness, 0.85)])
            px.set_non_recursive(location=locx.clone(), color=colx.clone())

            vals = 0
            vals1 = 0.+0j
            for params, a in mixed_params:
                params2 = gauss_tfm(params)
                vals0 = gauss2d_calc(params2, xp, xp*0)
                w = vals0.abs()
                vals1 += vals0 * w * a
                vals += w * w * a

            locp[..., :] = xp.unsqueeze(-1) * -rightp + yp.unsqueeze(-1) * vals.unsqueeze(-1) * upp + originp
            for i in range(n_col):
                lightness = min(0.15 + vals[0, i] * u * yp[0,i], 0.8)
                colp[0, i, :3] = torch.tensor([*colorsys.hls_to_rgb(np.angle(vals1[0,i])/(2*PI)+0.5, lightness, 0.85)])
            pp.set_non_recursive(location=locp.clone(), color=colp.clone())

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
    elif anim == 14:
        run_time = 1.
        f = f11
    elif 15 <= anim <= 18:
        run_time = 1.
        f = f12(anim - 14)
    elif anim == 19:
        run_time = 1.
        f = f13
    elif 20 <= anim <= 21:
        run_time = 1.
        f = f14(0., 1.) if anim == 20 else f14(1., .5)
    elif 22 <= anim <= 25:
        run_time = 1.5
        f = f15
        part = anim - 21
    elif anim == 26:
        run_time=6.
        f = f16
    elif anim == 27:
        run_time = 4.
        f = f17
    elif anim == -1:
        run_time=0.2
        f = g1

    if part > 1:
        show_wave = False
    if show_wave:
        with Off():
            surfx.spawn()
            surfp.spawn()


    # run_time = 0.1

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
                set_frame(f(frame.u))
    else:
        with Off():
            set_frame(f(1.))
        cam: Camera = Scene.get_camera()
        if part == 2: move_view(cam, 1)
        if part == 3: move_view(cam, 2)
        if part == 4: move_view(cam, 3)
        if part == 5: move_view(cam, 1)
        if part == 6: move_view(cam, 2)
        if part == 7: move_view(cam, 3)

    Scene.wait(1.1/quality.frames_per_second)

    render_to_file(name, render_settings=quality, background_color=bgcol)


if __name__ == "__main__":
    g1(0.)
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    #COMPUTING_DEFAULTS.max_animate_batch_size = 4
    # for anim in [15,16,17,18]:
    for anim in [8]:
        wigner_anim(quality=LD, bgcol=BLACK, anim=anim, show_wave=True)