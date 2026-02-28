import numpy as np
import torch
import colorsys
from algan import *
import manim as mn
import scipy as sp
from algan.external_libraries.manim.utils.color.SVGNAMES import INDIGO
from scipy.fft import fft, ifft, fftfreq

from general.integratepowers import xrange

sys.path.append('../../')
import alganhelper as ah
from common.wigner import *
LD = RenderSettings((854, 480), 15)
LD2 = RenderSettings((854, 480), 30)
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

def setup_surf(xrange=(-5., 5.), yrange=(-5., 5.), zrange=(-.3, .3), spawn=True):
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

    colors = [
        Color(mn.RED_D.to_rgb() * 0.5 / .8),
        Color(mn.RED_E.to_rgb() * 0.5 / .8)
    ]
    surf = ah.surface_mesh(num_recs=64, rec_size=10, col1=colors[0], col2=colors[1], stroke_color=RED_E,
                           fill_opacity=0.9, stroke_opacity=1)
    shape = (surf.grid_width, surf.grid_height)
    p = surf.get_descendants()[1]
    loc = p.location.clone()
    x = loc[:,:,0] * (xmax - xmin) / 2 + (xmax+xmin)/2
    y = loc[:,:,1] * (ymax - ymin) / 2 + (ymax+ymin)/2
    surf.scale(np.array([(xmax-xmin)*right[0]/2, (ymax-ymin)*up[1]/2, 1])).move_to(origin)

    with Off():
        txt1.spawn()
        txt2.spawn()
        ax1.spawn()
        if spawn:
            surf.spawn()

    return origin, right, up, out, p, x, y, shape, Group(txt1, txt2)

def setup_wave(xrange=(-5., 5.), npts=640):
    xmin, xmax = xrange

    n_col=4
    surfx = Surface(grid_height=n_col, grid_width=npts)
    px = surfx.get_descendants()[1]
    locx = px.location.clone()
    colx = px.color.clone()
    xx = locx[:, :, 0] * (xmax - xmin) / 2 + (xmax + xmin) / 2
    yx = (locx[:, :, 1] + 1) / 2
    colx[..., 4] = 0.75
    px.set_non_recursive(color=colx)

    with Off():
        surfx.spawn()

    return px, xx, yx

def set_wave(p, xvals, vals, vals1, origin, right, up):
    yvals = torch.linspace(0., 1., 4)
    n = len(xvals)
    locx = p.location.clone()
    locx[...,:] = (xvals.repeat_interleave(4).view(1, -1, 1) * right +
                    (vals.view(n, 1) * yvals.view(1, 4)).reshape(1, n*4, 1) * up + origin)
    colx = p.color.clone()

    for i in range(n*4):
        lightness = min(0.15 + vals[i // 4] * 0.3 * yvals[i % 4], 0.8)
        colx[0, i, :3] = torch.tensor([*colorsys.hls_to_rgb(np.angle(vals1[i // 4]) / (2 * PI) + 0.5, lightness, 0.85)])

    p.set_non_recursive(location=locx, color=colx)

col_up = torch.tensor([1, .6, 0.])
col_dn = INDIGO[:3]

def wigner_anim(quality=LD, bgcol=BLACK, anim=1, show_wave=False):
    name = 'wigner_wave{}'.format(anim) if show_wave else 'wigner_anim{}'.format(anim)
    setup_cam()

    xmin, xmax = xrange = (-5., 5.)
    ymin, ymax = yrange = (-5., 5.)

    origin, right, up, out, p, x, y, _, _ = setup_surf(xrange, yrange)

    surf2 = ah.surface_mesh(num_recs=64, rec_size=10, fill_opacity=1, stroke_opacity=0, add_to_scene=False)
    fill_mask = surf2.get_descendants()[1].color[:,:,-1:]
    mesh_mask = 1 - fill_mask

    col0 = p.color.clone()
    loc = p.location.clone()

    col = col0.clone()

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
        name = 'wigner_wave{}'.format(anim)

        npts = 640
        px, xx, yx = setup_wave(xrange=xrange, npts=npts)
        pp, xp, yp = setup_wave(xrange=yrange, npts=npts)
        xvals = torch.linspace(xmin, xmax, npts)
        pvals = torch.linspace(ymin, ymax, npts)



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
                vals0 = gauss2d_calc(params, xvals, xvals*0)
                w = vals0.abs()
                vals1 += vals0 * w * a
                vals += w * w * a

            set_wave(px, xvals, vals, vals1, origin + ymax * up, right, out*0.5)

            vals = 0
            vals1 = 0.+0j
            for params, a in mixed_params:
                params2 = gauss_tfm(params)
                vals0 = gauss2d_calc(params2, pvals, pvals*0)
                w = vals0.abs()
                vals1 += vals0 * w * a
                vals += w * w * a

            set_wave(pp, pvals, vals, vals1, origin + xmin * right, -up, out*0.5)

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

def wigner_custom_p(psi, x_min, x_max,
                    p_min, p_max, p_n):
    """
    Compute Wigner distribution with custom momentum grid.

    psi : (nx,) complex
    x_min, x_max : position domain
    p_min, p_max : momentum domain
    p_n : number of momentum points
    """

    nx = len(psi)
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min, x_max, nx)

    # momentum grid (user specified)
    p = np.linspace(p_min, p_max, p_n)

    # y grid (relative coordinate)
    y = np.arange(nx) * dx

    W = np.zeros((nx, p_n), dtype=float)

    for j in range(nx):

        # build correlation C(y) = psi*(x+y) psi(x-y)
        C = np.zeros(nx, dtype=complex)

        for m in range(nx):
            jp = j + m
            jm = j - m
            if 0 <= jp < nx and 0 <= jm < nx:
                C[m] = np.conj(psi[jp]) * psi[jm]

        # compute Fourier transform at chosen p values
        phase = np.exp(2j * np.outer(p, y))   # (p_n, nx)
        W[j, :] = np.real((phase @ C) * dx / (np.pi))

    return W

def wigner_fft(psi, x_min, x_max,
               p_min=None, p_max=None, p_n=None,
               i0=0, i1=-1, step=1,
               pad_factor=2):
    """
    Fast Wigner distribution using FFT with zero padding.

    psi : (nx,) complex
    x_min, x_max : position domain
    p_min, p_max, p_n : optional custom momentum grid
    pad_factor : padding multiplier (>=2 recommended)
    """

    nx = len(psi)
    dx = (x_max - x_min) / (nx-1)
    x = np.linspace(x_min, x_max, nx)
    if i1 < 0:
        i1 = nx
    nx1 = len(x[i0:i1:step])

    # ---- zero padding to avoid wraparound ----
    n_pad = pad_factor * nx
    psi_pad = np.zeros(n_pad, dtype=complex)
    start = (n_pad - nx) // 2
    psi_pad[start:start+nx] = psi

    W = np.zeros((nx1, n_pad), dtype=float)

    for j in range(nx1):
        j1 = i0 + j*step
        # correlation function
        C = np.zeros(n_pad, dtype=complex)

        for m in range(-nx//2, nx//2):
            jp = j1 + m
            jm = j1 - m
            if 0 <= jp < nx and 0 <= jm < nx:
                C[m + n_pad//2] = np.conj(psi[jp]) * psi[jm]

        # FFT over relative coordinate
        W_j = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(C)))
        W[j, ::-1] = np.real(W_j) * dx / (2*np.pi)

    # momentum grid from FFT
    # dp_fft = 2 * np.pi / (n_pad * dx)
    p_fft = np.fft.fftshift(np.fft.fftfreq(n_pad, d=dx)) * np.pi

    # ---- optional interpolation to custom grid ----
    if p_min is not None:
        p_custom = np.linspace(p_min, p_max, p_n)
        W_interp = np.zeros((nx1, p_n))
        for j in range(nx1):
            W_interp[j] = np.interp(p_custom, p_fft, W[j])
        return W_interp

    return W

def time_evolution(psi, V, dt=1., dx=0.1, mass=1.):
    n = len(psi)
    k_space = torch.fft.fftfreq(n, dx) * (2 * np.pi)  # Momentum space grid
    T = 0.5 / mass * k_space**2  # Kinetic energy operator
    psi_k = torch.fft.fft(psi)
    psi_k = torch.exp(-1j * T * dt) * psi_k  # Evolve in momentum space
    psi = torch.fft.ifft(psi_k)  # Transform back to position space
    psi = torch.exp(-1j * V * dt) * psi  # Evolve due to potential
    return psi

class WaveEvolution:
    def __init__(self, xrange=(-5., 5.), prange=(-5., 5.), npts=639, n_extend_left=200, n_extend_right=200,
                 n_scale=1, dt=0.001, mass=1., speed=PI):
        xmin, xmax = self.xrange = xrange
        pmin, pmax = self.prange = prange
        self.npts = npts
        self.n_extend_left = n_extend_left
        self.n_extend_right = n_extend_right
        self.n_scale = 1
        setup_cam()
        npts1 = (npts + n_extend_left+n_extend_right-1)*n_scale+1
        i_x0 = n_extend_left * n_scale
        i_x1 = npts1 - n_extend_right * n_scale
        x_extend_left = (xmax - xmin) / (npts-1) * n_extend_left
        x_extend_right = (xmax - xmin) / (npts-1) * n_extend_right
        xmin1, xmax1 = (xmin - x_extend_left, xmax + x_extend_right)

        origin, right, up, out, p, xsurf, y, shape, self.txt = setup_surf(xrange, prange, spawn=True)
        assert shape == (npts, npts)

        self.col0 = p.color.clone()
        surf2 = ah.surface_mesh(num_recs=64, rec_size=10, fill_opacity=1, stroke_opacity=0, add_to_scene=False)
        self.fill_mask = surf2.get_descendants()[1].color[:,:,-1:]
        self.mesh_mask = 1 - self.fill_mask

        self.dt = dt
        self.speed = speed
        self.mass = mass
        self.xvals1 = torch.linspace(xmin1, xmax1, npts1)
        self.psi = self.xvals1 * 0
        self.xvals = torch.linspace(xmin, xmax, npts)
        dx = (self.xvals[1] - self.xvals[0]).item()
        self.stride = (i_x0, i_x1, n_scale)
        self.V = self.xvals1 * 0
        self.dx1 = dx/n_scale
        self.dirs = (origin, right, up, out)
        self.pvals = torch.linspace(pmin, pmax, npts)
        self.xrange1 = (xmin1, xmax1)
        self.p = p
        self.px = self.xx = self.yx = None
        self.pp = self.xp = self.yp = None

    def create_wave(self):
        self.px, self.xx, self.yx = setup_wave(xrange=self.xrange, npts=self.npts)
        self.pp, self.xp, self.yp = setup_wave(xrange=self.prange, npts=self.npts)

    def evolve(self, time_inc):
        ndt = math.ceil(time_inc / self.dt)

        if ndt > 0:
            # print(ndt)
            dt1 = time_inc / ndt
            for i in range(ndt):
                self.psi = time_evolution(self.psi, self.V, self.speed * dt1, self.dx1, mass=self.mass)

        pmin, pmax = self.prange
        xmin, xmax = self.xrange
        psi = self.psi
        psi0 = psi[self.stride[0]:self.stride[1]:self.stride[2]]
        origin, right, up, out = self.dirs

        if self.px is not None:
            set_wave(self.px, self.xvals, psi0.abs()**2, psi0, origin + pmax * up, right, out * 0.5)
        if self.pp is not None:
            phase = np.exp(1j * np.outer(self.pvals, self.xvals1))
            psi_k = torch.tensor(phase @ psi.numpy()) * self.dx1 / math.sqrt(2*PI)
            set_wave(self.pp, self.pvals, psi_k.abs() ** 2, psi_k, origin + xmin * right, -up, out * 0.5)

        p = self.p
        xmin1, xmax1 = self.xrange1
        i0, i1, n_scale = self.stride
        W = torch.tensor(wigner_fft(psi.numpy(), xmin1, xmax1, pmin, pmax, self.npts, i0=i0, i1=i1, step=n_scale))
        loc = p.location.clone()
        col = p.color.clone()
        loc[0, :, 2] = origin[2] + out[2] * 2 * W.reshape(-1)
        shade_up = torch.pow(((W - 0.05) * 4).clamp(0, 1), 0.8).flatten().view(1,-1,1)
        shade_down = (W * -50).clamp(0., 1).flatten().view(1,-1,1)
        fill_mask = self.fill_mask

        col[..., :3] = fill_mask * shade_up * col_up \
                       + fill_mask * shade_down * col_dn \
                       + fill_mask * (1 - shade_up - shade_down) * self.col0[:, :, :3] \
                       + self.mesh_mask * self.col0[..., :3]
        p.set_non_recursive(location=loc, color=col)

    def orbit(self, anim=1):
        origin, right, up, out = self.dirs
        cam = Scene.get_camera()
        if anim == 1:
            with Seq():
                cam.orbit_around_point(origin, 60 * DEGREES, cam.get_right_direction())
        elif anim == 2:
            with Off():
                cam.orbit_around_point(origin, 60 * DEGREES, cam.get_right_direction())
            with Sync(run_time=3):
                cam.orbit_around_point(origin, -60 * DEGREES, out)
                cam.orbit_around_point(origin, -130 * DEGREES, cam.get_right_direction())
                self.txt[0].orbit_around_point(self.txt[0].get_center(), -90, RIGHT)
        else:
            orbit_time = 4.
            with Sync(run_time=orbit_time):
                cam.orbit_around_point(origin, 180 * DEGREES, out)
            with Sync(run_time=1.5):
                cam.orbit_around_point(origin, -70 * DEGREES, cam.get_right_direction())

    def set_gaussian(self, x=0.):
        sigma = np.sqrt(0.5)  # Width of Gaussian
        self.psi = torch.exp(-(self.xvals1 - x) ** 2 / (4 * sigma ** 2))  # + xvals1 * 6 * 1j)

        # Normalize the wavefunction
        self.psi /= np.linalg.norm(self.psi) * np.sqrt(self.dx1)


def evolve_wave(quality=LD, bgcol=BLACK, anim=1):
    evolver = WaveEvolution()

    start_time=0.
    run_time = 1.
    rate_func=rate_funcs.identity
    part = 1

    if anim == 1:
        evolver.V = (evolver.xvals1)**2 * 0.5
        run_time=4.
    else:
        # pendulum
        eps = 1/3.5 * PI/3
        evolver.V = (1-torch.cos(evolver.xvals1 * eps))/(eps*eps)
        if 2 <= anim <= 9:
            start_time=(anim-2) * 2
            run_time=2.
        elif anim == 10:
            start_time = 15.5
            run_time=1.
            rate_func = lambda t: t * ( 1 - t/2)
        else:
            start_time=16.
            part = anim - 9

    evolver.set_gaussian(-3.5)

    if part <= 2:
        evolver.create_wave()

    t0 = 0.
    for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, step=1, rate_func=rate_func):
        t1 = frame.u * run_time + start_time
        print(frame.index, frame.time)
        with frame.context:
            evolver.evolve(t1 - t0)
        t0 = t1
        if part > 1:
            break

    if part > 1:
        if part == 2:
            Scene.wait(2.1/quality.frames_per_second)
        else:
            evolver.orbit(part - 2)
        # Scene.wait(0.1)

    name = r'evolve_wave{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)

if __name__ == "__main__":
    g1(0.)
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    #COMPUTING_DEFAULTS.max_animate_batch_size = 4
    # for anim in [15,16,17,18]:
    # for anim in [8]:
    #     wigner_anim(quality=LD, bgcol=BLACK, anim=anim, show_wave=True)
    for anim in [13]:
        evolve_wave(quality=LD, bgcol=BLACK, anim=anim)