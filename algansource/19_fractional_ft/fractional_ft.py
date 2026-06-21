import colorsys
from algan import *

sys.path.append('../../')
import alganhelper as ah
from common.wigner import *
from algansource.wave import setup_wave, set_wave, WaveEvolver
LD = RenderSettings((854, 480), 15)
LD2 = RenderSettings((854, 480), 30)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 15)


def gauss_density(quality=LD, bgcol=BLACK, anim=1):
    npts=639
    xrange = (-1.85,1.85)
    xvals = torch.linspace(xrange[0], xrange[1], npts)

    px, xx, yx = setup_wave(npts=npts, xrange=xrange)
    print(xvals.shape)
    psi = torch.exp(-xvals*xvals + 0j)
    right=RIGHT*2
    out = UP*1.2
    origin = ORIGIN + IN*0.1

    xmin1, xmax1 = (xrange[0], xrange[1]*1.05)
    xlen = (xmax1 - xmin1) * right[0].item()
    ymax1 = 1.2
    ylen = ymax1 * out[1].item()
    print(xmin1, xmax1, ymax1, xlen, ylen)
    ax = mn.Axes(x_range=[xmin1, xmax1], y_range=[0, ymax1], x_length=xlen, y_length=ylen,
              axis_config={'color': mn.WHITE, 'stroke_width': 5, 'include_ticks': False,
                           "tip_width": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           "tip_height": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           },
              ).set_opacity(0.8)
    ax.shift(-ax.coords_to_point(0, 0))
    mn.MathTex.set_default(font_size=30)
    eq1 = mn.MathTex(r'x').move_to(ax.x_axis.get_right() + mn.RIGHT*0.05, aligned_edge=LEFT)
    eq2 = mn.MathTex(r'\psi(x)').move_to(ax.y_axis.get_top() + mn.RIGHT*0.1, aligned_edge=LEFT)
    mn.VGroup(eq1, eq2[0][2]).set_color(col_x)
    eq2[0][0].set_color(col_psi)
    ax = ManimMob(mn.VGroup(ax, eq1, eq2))

    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(100)
        set_wave(px, xvals, psi.abs() ** 2, psi, origin, right, out)
        ax.spawn()

    run_time=2.

    for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, step=1, rate_func=rate_funcs.smooth):
        with frame.context:
            p = frame.u * PI * 2
            psi = torch.exp(-xvals * xvals + xvals * p * 1j)
            set_wave(px, xvals, psi.abs() ** 2, psi, origin, right, out)

    Scene.wait(0.1)

    name = 'gauss_density{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)


def fractional_example(quality=LD, bgcol=BLACK, anim=0):
    """
    example fractional fourier transform 2d plots
    """
    npts=639
    xrange = (-5.,5.)
    ymax1 = 1.2
    params1 = gauss1d(.5, shift=1.)
    angle = 2*PI

    def psi_update(u, du):
        params2 = gauss_fractional_ft(params1, angle*u)
        return gauss1d_calc(params2, xvals)

    if anim == 1:
        params1 = gauss1d(1.)
    elif anim == 2:
        params1 = gauss1d(2.)
    elif anim == 3:
        params1 = gauss1d(2., shift=2.)
    elif anim == 4:
        params1 = gauss1d(2., shift=3.) + gauss1d(2., shift=-3.)
        params1 = gauss_scale(params1, 1.)
    elif anim == 5:
        params1 = gauss1d(2., shift=4.) + gauss_scale(gauss1d(2., shift=-4.), -0.5)
        params1 = gauss_scale(params1, 1.)
        xrange = (-7.,7.)
    elif anim == 6:
        evolver = WaveEvolver(xrange=xrange, npts=npts, n_extend_left=8000, n_extend_right=8000, n_scale=2,
                              speed=1.)
        evolver.V = evolver.xvals1 ** 2 * 0.5 - 0.5
        evolver.psi = (1. - (evolver.xvals1.abs() - 2.).clip(0) * 10).clip(0)
        ymax1 = 1.8
        angle = PI

        def psi_update(u, du):
            return evolver.evolve(du*angle)


    xvals = torch.linspace(xrange[0], xrange[1], npts)

    px, xx, yx = setup_wave(npts=npts, xrange=xrange)
    xmin1, xmax1 = (xrange[0], xrange[1]*1.05)
    xlen = 8.
    right = xlen / (xmax1 - xmin1) * RIGHT
    ylen = 3.
    out = ylen / ymax1 * UP
    ylen = ymax1 * out[1].item()

    ax = mn.Axes(x_range=[xmin1, xmax1], y_range=[0, ymax1], x_length=xlen, y_length=ylen,
              axis_config={'color': mn.WHITE, 'stroke_width': 5, 'include_ticks': False,
                           "tip_width": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           "tip_height": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           },
              ).set_opacity(0.8)
    ax.shift(-ax.coords_to_point(0, 0) + mn.DOWN)
    origin = torch.from_numpy(ax.coords_to_point(0, 0)) + IN*0.1
    mn.MathTex.set_default(font_size=30)
    ax = ManimMob(mn.VGroup(ax))

    run_time = 4.

    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(100)
        ax.spawn()

    for frame in ah.FrameStepper(fps=quality.frames_per_second, rate_func=rate_funcs.identity, run_time=run_time):
        u = frame.u
        du = frame.du
        psi2 = psi_update(u, du)
        with frame.context:
            set_wave(px, xvals, psi2.abs(), psi2, origin, right, out)

    Scene.wait(0.1)

    name = 'fractional_example{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)



if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20

    fractional_example(LD, bgcol=BLACK, anim=6)
