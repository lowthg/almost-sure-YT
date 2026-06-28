from algan import *
from manim.utils.color.BS381 import SILVER_GREY
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader, default_shader
from algan.external_libraries.manim.utils.color.SVGNAMES import INDIGO, SILVER

sys.path.append('../../')
import alganhelper as ah
from common.wigner import *
from algansource.wave import setup_wave, set_wave, WaveEvolver, setup_cam, setup_surf
LD = RenderSettings((854, 480), 15)
LD2 = RenderSettings((854, 480), 30)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 15)

col_up = torch.tensor([1, .6, 0.])
col_dn = INDIGO[:3]


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

def setup_fourier(npts, xrange, ymax1):
    xvals = torch.linspace(xrange[0], xrange[1], npts)

    xmin1, xmax1 = (xrange[0], xrange[1]*1.05)
    xlen = 8.
    right = xlen / (xmax1 - xmin1) * RIGHT
    ylen = 1.8
    out = ylen / ymax1 * UP
    ylen = ymax1 * out[1].item()
    ax = mn.Axes(x_range=[xmin1, xmax1], y_range=[0, ymax1], x_length=xlen, y_length=ylen,
              axis_config={'color': mn.WHITE, 'stroke_width': 5, 'include_ticks': False,
                           "tip_width": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           "tip_height": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           },
              ).set_opacity(0.8)
    ax.shift(-ax.coords_to_point(0, 0) + mn.DOWN * 2.)
    origin = torch.from_numpy(ax.coords_to_point(0, 0)) + IN*0.1
    mn.MathTex.set_default(font_size=30)
    eq1 = mn.MathTex(r'x').move_to(ax.x_axis.get_right() + mn.RIGHT*0.05, aligned_edge=LEFT)
    eq2 = mn.MathTex(r'f(x)').move_to(ax.y_axis.get_top() + mn.RIGHT*0.15 + mn.DOWN*0.07, aligned_edge=LEFT)
    mn.VGroup(eq1, eq2[0][2]).set_color(col_x)
    eq2[0][0].set_color(col_psi)
    ax = ManimMob(mn.VGroup(ax, eq1, eq2[0]))

    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(100)

    eq3 = eq2.copy()
    eq4 = mn.MathTex(r'\mathcal F', r'f(x)')
    eq4[1][2].set_color(col_x)
    eq4[1][0].set_color(col_psi)
    eq4[0][0].set_color(col_ft)

    axes = []
    origins = []

    for direction, eq in [(mn.UP * 0.3, eq3), (mn.DOWN * 2., eq4)]:
        ax1 = mn.Axes(x_range=[xmin1, xmax1], y_range=[0, ymax1], x_length=xlen, y_length=ylen,
                      axis_config={'color': mn.WHITE, 'stroke_width': 5, 'include_ticks': False,
                                   "tip_width": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                   "tip_height": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                   },
                      ).set_opacity(0.8)
        ax1.shift(-ax1.coords_to_point(0, 0)).shift(direction)
        eq_ = eq1.copy().move_to(ax1.x_axis.get_right() + mn.RIGHT * 0.05, aligned_edge=LEFT)
        eq.move_to(ax1.y_axis.get_top() + mn.RIGHT * 0.15 + mn.DOWN * 0.07, aligned_edge=LEFT)
        eq = [ManimMob(_) for _ in eq[:]]
        origin1 = torch.from_numpy(ax1.coords_to_point(0, 0)) + IN * 0.1
        ax1 = ManimMob(ax1)
        axes.append(Group(ax1, ManimMob(eq_), *eq))
        origins.append(origin1)

    return ax, xvals, origin, right, out, eq1, eq2, eq3, eq4, axes, origins


def fourier_example(quality=LD, bgcol=BLACK, anim=0):
    npts=639
    xrange = (-5.,5.)
    ymax1 = 1.2
    ax, xvals, origin, right, out, eq1, eq2, eq3, eq4, axes, origins\
        = setup_fourier(npts=npts, xrange=xrange, ymax1=ymax1)

    if anim < 3:
        with Off():
            ax.spawn()

    if anim == 1:
        run_time=2.
        px, xx, yx = setup_wave(npts=npts, xrange=xrange, opacity=1)

        for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, step=1, rate_func=rate_funcs.smooth):
            p = frame.u * PI * 2
            params1 = gauss1d(1., p * 1j, shift=0.)
            psi = gauss1d_calc(params1, xvals)
            with frame.context:
                set_wave(px, xvals, psi.abs(), psi, origin, right, out)
    elif anim >= 2:
        params1 = gauss1d(1., shift=0.)
        psi1 = gauss1d_calc(params1, xvals)
        if anim == 2:
            px, xx, yx = setup_wave(npts=npts, xrange=xrange, opacity=1)
            with Off():
                set_wave(px, xvals, psi1.abs(), psi1, origin, right, out)
                px.set_opacity_via_color(0)

        waves_p = []
        locs = []
        dir2 = OUT*0.05
        for origin1 in origins:
            with Off():
                px1, _, _ = setup_wave(npts=npts, xrange=xrange, opacity=1)
                waves_p.append(px1)
                set_wave(px1, xvals, psi1.abs(), psi1.clone(), origin1, right, out)
                locs.append(px1.location.clone())
                if anim == 2:
                    px1.set_non_recursive(location=px.location.clone() + dir2)
            dir2 = ORIGIN

        axes = Group(axes)

        if anim == 2:
            shift = axes[1][3].get_center() - ax.submobjects[2].get_center()
            axes[1][2].move(-shift)
            with Sync(run_time=1.7):
                ax.submobjects[0].become(axes[0][0])
                ax.submobjects[0].clone().become(axes[1][0])
                ax.submobjects[2].become(axes[0][2])
                ax.submobjects[2].clone().become(axes[1][3])
                ax.submobjects[1].despawn()
                axes[1][2].spawn().move(shift)
                waves_p[0].set_non_recursive(location=locs[0])
                waves_p[1].set_non_recursive(location=locs[1])
                ax.submobjects[1].become(axes[0][1])
                ax.submobjects[1].clone().become(axes[1][1])
        else:
            with Off():
                axes.spawn()

        if 8 >= anim >= 3 or anim >= 101:
            def f(t):
                return gauss1d()
            run_time=1.

            if anim == 3:
                run_time=1.
                def f(u):
                    a = math.exp(-u * math.log(2))
                    return gauss1d(a)
            if anim == 4:
                run_time=1.8
                def f(u):
                    a = math.exp(u * math.log(4)) * 0.5
                    return gauss1d(a)
            if anim == 5:
                run_time=1.2
                def f(u):
                    return gauss1d(2., shift=3. * u)
            if anim == 6:
                run_time=1.
                def f(u):
                    return gauss1d(2., shift=3.) + gauss1d(2., shift=-3., c=u)
            if anim == 7:
                run_time=1.5
                def f(u):
                    c = np.exp(1j * PI * u)
                    return gauss1d(2., shift=3.) + gauss1d(2., shift=-3., c=c)
            if anim == 8:
                run_time=1.5
                def f(u):
                    c = np.exp(1j * PI * (1-u)) *  (1-u)
                    shift = 3*(1-u)
                    a = 2 * math.exp(-u * math.log(2))
                    params = gauss1d(a, shift=shift) + gauss1d(a, shift=-shift, c=c)
                    return params
            if anim == 101:
                run_time=1.
                def f(u):
                    a = math.exp(u * math.log(4))
                    return gauss1d(a, c = 1. + u * 0.08)
            if anim == 102:
                run_time=1.
                def f(u):
                    a = 4. * math.exp(-u*math.log(2))
                    return gauss1d(a, c=1.08, shift=3.*u)
            if anim == 103:
                run_time=1.
                def f(u):
                    c2 = -u * 1.0
                    c1 = 1.08 - 0.08 * u
                    return gauss1d(2., c=c1, shift=3.) + gauss1d(2., c=c2, shift=-3.)

            for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, step=1,
                                         rate_func=rate_funcs.smooth):
                u = frame.u
                params1 = f(u)
                params2 = gauss_tfm(params1, dim=0)
                psi1 = gauss1d_calc(params1, xvals)
                psi2 = gauss1d_calc(params2, xvals)
                with frame.context:
                    set_wave(waves_p[0], xvals, psi1.abs(), psi1, origins[0], right, out)
                    set_wave(waves_p[1], xvals, psi2.abs(), psi2, origins[1], right, out)


        if anim == 9:
            w = 2.
            params1 = gauss1d()
            c = 0.9
            psi1 = gauss1d_calc(params1, xvals)
            psi2 = ((xvals <= w) & (xvals >= -w)).float() * c
            psi3 = torch.sin(xvals*w) / xvals * math.sqrt(2/PI) * c
            run_time=1.
            for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, step=1,
                                         rate_func=rate_funcs.smooth):
                u = frame.u
                psi4 = psi1 * (1-u) + psi2 * u
                psi5 = psi1 * (1-u) + psi3 * u
                with frame.context:
                    set_wave(waves_p[0], xvals, psi4.abs(), psi4, origins[0], right, out)
                    set_wave(waves_p[1], xvals, psi5.abs(), psi5, origins[1], right, out)


        # with Off():
        #     axes.spawn()


        # with Sync(run_time=1.):
        #     ax.clone().move_to(LEFT*xlen*scale*0.5).scale(scale)

    Scene.wait(0.1)

    name = 'fourier_example{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)

def fractional_ex(quality=LD, bgcol=BLACK, anim=0, part=0):
    npts=639
    xrange = (-5.,5.)
    ymax1 = 1.2
    xvals = torch.linspace(xrange[0], xrange[1], npts)

    xmin1, xmax1 = (xrange[0], xrange[1]*1.05)
    xlen = 8.
    right = xlen / (xmax1 - xmin1) * RIGHT
    ylen = 1.8
    out = ylen / ymax1 * UP
    ylen = ymax1 * out[1].item()
    ax = mn.Axes(x_range=[xmin1, xmax1], y_range=[0, ymax1], x_length=xlen, y_length=ylen,
              axis_config={'color': mn.WHITE, 'stroke_width': 5, 'include_ticks': False,
                           "tip_width": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           "tip_height": 0.4 * mn.DEFAULT_ARROW_TIP_LENGTH,
                           },
              ).set_opacity(0.8)
    ax.shift(-ax.coords_to_point(0, 0)).move_to(mn.ORIGIN, coor_mask=mn.UP)
    mn.MathTex.set_default(font_size=30)
    eq1 = mn.MathTex(r'x').move_to(ax.x_axis.get_right() + mn.RIGHT*0.05, aligned_edge=LEFT)
    eq2 = mn.MathTex(r'\mathcal F_\theta f(x)').move_to(ax.y_axis.get_top() + mn.RIGHT*0.15 + mn.DOWN*0.07, aligned_edge=LEFT)
    mn.VGroup(eq1, eq2[0][4]).set_color(col_x)
    eq2[0][2].set_color(col_psi)
    eq2[0][0].set_color(col_ft)
    eq2[0][1].set_color(col_angle)

    origin = torch.from_numpy(ax.coords_to_point(0, 0)) + IN*0.1
    mn.MathTex.set_default(font_size=30)
    ax = ManimMob(mn.VGroup(ax, eq1, eq2))

    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(100)
        ax.spawn()

    px, _, _ = setup_wave(xrange, npts, opacity=1)

    rate_func=rate_funcs.smooth
    run_time=1.
    if part == 0:
        def psi_update(u, du):
            params = f(u)
            return gauss1d_calc(params, xvals)
    else:
        def psi_update(u, du):
            theta = theta1 * (1-u) + theta2 * u
            params1 = gauss_fractional_ft(params, theta)
            return gauss1d_calc(params1, xvals)

    if anim == 1:
        params = gauss1d(1.)
        psi = gauss1d_calc(params, xvals)
        with Off():
            set_wave(px, xvals, psi.abs(), psi, origin, right, out)
    if anim == 2:
        params = gauss1d(a=4, c=1.08)
        if part == 0:
            run_time=1.
            def f(u):
                a = math.exp(u * math.log(4))
                return gauss1d(a, c=1. + u * 0.08)
        elif part==1:
            theta1 = 0.
            theta2 = PI/2
            run_time=3.
        elif part==2:
            theta1 = PI/2
            theta2 = 3*PI/2
            run_time=4
            rate_func=rate_funcs.identity
    if anim == 3:
        params= gauss1d(2., shift=3., c=1.08)
        if part == 0:
            def f(u):
                a = 4. * math.exp(-u*math.log(2))
                return gauss1d(a, c=1.08, shift=3.*u)
        elif part == 1:
            theta1 = 0.
            theta2 = PI/2
            run_time=3.
        elif part == 2:
            theta1 = 0.
            theta2 = 2*PI
            run_time=6
            rate_func=rate_funcs.identity
    if anim == 4:
        def f(u):
            c2 = -u * 1.0
            c1 = 1.08 -0.08 * u
            return gauss1d(2., c=c1, shift=3.) + gauss1d(2., c=c2, shift=-3.)
        params = f(1)
        if part == 1:
            theta1 = 0.
            theta2 = PI/2
            run_time=3.
        if part == 2:
            theta1 = 0.
            theta2 = PI*2
            run_time=6.
            rate_func=rate_funcs.identity
    if anim == 5:
        run_time = 1.5
        def f(u):
            c = np.exp(1j * PI * (1 - u)) * (1 - u)
            shift = 3 * (1 - u)
            a = 2 * math.exp(-u * math.log(2))
            params = gauss1d(a, shift=shift) + gauss1d(a, shift=-shift, c=c)
            return params
    if anim == 6:
        w = 2.
        c = 0.9
        if part == 0:
            params1 = gauss1d()
            psi1 = gauss1d_calc(params1, xvals)
            psi2 = ((xvals <= w) & (xvals >= -w)).float() * c
            run_time = 1.
            def psi_update(u, du):
                return psi1 * (1 - u) + psi2 * u
        else:
            evolver = WaveEvolver(xrange=xrange, npts=npts, n_extend_left=8000, n_extend_right=8000, n_scale=2,
                                  speed=1.)
            evolver.V = evolver.xvals1 ** 2 * 0.5 - 0.5
            evolver.psi = (1. - (evolver.xvals1.abs() - 2.).clip(0) * 10).clip(0) * c
            def psi_update(u, du):
                return evolver.evolve(du * angle)
            if part == 1:
                angle = PI/2
                run_time=3.
            if part == 2:
                evolver.evolve(PI/2)
                angle = PI/4
                evolver.speed = -evolver.speed
                run_time = 1.
            if part == 3:
                evolver.evolve(PI/4)
                evolver.speed = -evolver.speed
                angle = PI/8
            if part == 4:
                angle = PI
                run_time = 3. + 6/30
                rate_func = rate_funcs.identity

    if anim > 1:
        for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, rate_func=rate_func):
            psi = psi_update(frame.u, frame.du)
            with frame.context:
                set_wave(px, xvals, psi.abs(), psi, origin, right, out)
        Scene.wait(0.1)

    name = 'fractional_ex{}'.format(anim)
    if part > 0:
        name = name + "_{}".format(part)
    render_to_file(name, render_settings=quality, background_color=bgcol)


def button(quality=LD, bgcol=BLACK, anim=0):
    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(100)

    r = 0.3

    col1 = GREY.clone()
    col1[:3] = torch.from_numpy(SILVER_GREY.to_rgb())
    col2 = col1.clone()
    col2[:3] *= 0.7
    circ = Circle(radius=r, color=col1, border_color=col2, border_width=3)

    eq = mn.MathTex(r'0', r'\frac\pi2', r'\pi', r'\frac{3\pi}2', font_size=20, stroke_width=0)
    r1 = r * 1.15
    r2 = r * 1.4
    r3 = r * 1.25
    lines = [
        mn.Line(r1*dir, r2*dir, color=mn.WHITE, stroke_width=4) for dir in [mn.UP, mn.RIGHT, mn.DOWN, mn.LEFT]]
    sqrt2 = 1./math.sqrt(2)
    lines2 = [mn.Line(r1*dir*sqrt2, r3*dir*sqrt2, color=mn.WHITE, stroke_width=4)
              for dir in [mn.UP+mn.RIGHT, mn.RIGHT+mn.DOWN, mn.DOWN+mn.LEFT, mn.LEFT+mn.UP]]
    buff=0.2
    eq[0].next_to(mn.ORIGIN, mn.UP, buff=buff).shift(r*mn.UP)
    eq[1].next_to(mn.ORIGIN, mn.RIGHT, buff=buff).shift(r*mn.RIGHT)
    eq[2].next_to(mn.ORIGIN, mn.DOWN, buff=buff).shift(r*mn.DOWN)
    eq[3].next_to(mn.ORIGIN, mn.LEFT, buff=buff).shift(r*mn.LEFT)
    eq1 = ManimMob(eq)
    body = Group(circ, eq1, *[ManimMob(_) for _ in lines + lines2]).move(IN*0.1)

    # pointer = ManimMob(mn.Line(r*mn.UP*0.2, r*mn.UP*0.95, stroke_width=10, stroke_color=mn.BLACK))
    # pointer = ah.Line(r*mn.UP*0.2, r*mn.UP*0.95, color=BLACK)
    print(UP.shape)
    pointer = ah.curve_surface(torch.stack([r*UP*0.2, r*UP*0.95]),
                               normals=torch.stack([IN, IN]), resolution=8, color=BLACK, width=0.06, closed=False)
    pointer.set_shader(null_shader)
    center = RIGHT * 3.5 + UP
    body.move(center)
    pointer.move(center)

    with Off():
        body.spawn()
        pointer.spawn()

    if anim == 1:
        with Sync(run_time=3):
            pointer.orbit_around_point(center, 360, IN)
    if anim == 2:
        with Sync(run_time=3):
            pointer.orbit_around_point(center, 90, IN)
        Scene.wait(0.1)
    if anim == 3:
        with Sync(run_time=8, rate_func=rate_funcs.identity):
            pointer.orbit_around_point(center, 360, IN)
    if anim == 4:
        with Sync(run_time=6, rate_func=rate_funcs.identity):
            pointer.orbit_around_point(center, 360, IN)
    if anim == 5:
        with Off():
            pointer.orbit_around_point(center, 90, IN)
        with Sync(run_time=1):
            pointer.orbit_around_point(center, -45, IN)
        Scene.wait(0.1)
    if anim == 6:
        with Off():
            pointer.orbit_around_point(center, 45, IN)
        with Sync(run_time=1):
            pointer.orbit_around_point(center, -22.5, IN)
        Scene.wait(0.1)
    if anim == 7:
        with Sync(run_time = 6. + 12/30, rate_func=rate_funcs.identity):
            pointer.orbit_around_point(center, 360, IN)

    name = 'button{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)


def wigner_anim(quality=LD, bgcol=BLACK, anim=1, show_wave=False, signal_vars=False):
    name = 'wigner_wave{}'.format(anim) if show_wave else 'wigner_anim{}'.format(anim)
    setup_cam()

    xmin, xmax = xrange = (-5., 5.)
    ymin, ymax = yrange = (-5., 5.)
    # xmin, xmax = xrange = (-2., 2.)
    # ymin, ymax = yrange = (-2., 2.)

    origin, right, up, out, p, x, y, _, ax = setup_surf(xrange, yrange, signal_vars=signal_vars)
    # with Off():
    #     cam = Scene.get_camera()
    #     cam.move_to(cam.get_center()*1.5+IN)
    #     ax[:2].despawn()

    surf2 = ah.surface_mesh(num_recs=64, rec_size=10, fill_opacity=1, stroke_opacity=0, add_to_scene=False)
    fill_mask = surf2.get_descendants()[1].color[:,:,-1:]
    mesh_mask = 1 - fill_mask

    col0 = p.color.clone()
    loc = p.location.clone()

    col = col0.clone()

    rate_func = rate_funcs.smooth
    part = 1
    smooth1 = smooth2 = 0.

    if anim == 1:
        run_time = 0.5
        def f(t):  # round Gaussian shift in X
            params = gauss1d_std(scale=1.)
            params = gauss_shift(params, 3.5 * t)
            return [(params, 1.)]

    if show_wave:
        name = 'wigner_wave{}'.format(anim)

        npts = 640
        px, xx, yx = setup_wave(xrange=xrange, npts=npts)
        pp, xp, yp = setup_wave(xrange=yrange, npts=npts)
        xvals = torch.linspace(xmin, xmax, npts)
        pvals = torch.linspace(ymin, ymax, npts)


    def set_frame(mixed_params, smooth):
        vals = sum([gauss2d_calc(gauss_wigner(params, params), x, y).real * a for params, a in mixed_params])
        if smooth > 0.:
            assert len(mixed_params) == 1
            (params, a) = mixed_params[0]
            params2 = gauss_smooth(gauss_wigner(params, params), smooth, smooth)
            vals = gauss2d_calc(params2, x, y).real * a

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
        elif part == 3:
            with Off():
                cam.orbit_around_point(origin, 60 * DEGREES, cam.get_right_direction())
            with Sync(run_time=1):
                cam.orbit_around_point(origin, -60*DEGREES, cam.get_right_direction())
        elif part == 4:
            with Sync(run_time=1):
                cam.orbit_around_point(origin, -30*DEGREES, cam.get_right_direction())
        elif part == 5:
            with Sync(run_time=1):
                cam.orbit_around_point(origin, 30*DEGREES, cam.get_right_direction())
        elif part == 6:
            with Sync(run_time=1):
                cam.orbit_around_point(origin, 30*DEGREES, cam.get_right_direction())
                set_frame(f0(1.), smooth2*smooth2)
        elif part == 7:
            with Sync(run_time=4):
                cam.orbit_around_point(origin, 360*DEGREES, cam.get_right_direction())
        elif part == 8:
            with Sync(run_time=3):
                cam.orbit_around_point(origin, 180*DEGREES, OUT)

    if part == 1:
        for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, step=1, rate_func=rate_func):
            print(frame.index, frame.time, frame.dt)
            if smooth1 > 0 or smooth2 > 0:
                smooth = smooth2 * frame.u + smooth1 * (1-frame.u)
                smooth *= smooth
            else: smooth = 0.

            with frame.context:
                set_frame(f(frame.u), smooth)
    else:
        with Off():
            set_frame(f(1.), smooth1*smooth1)
        cam: Camera = Scene.get_camera()
        if part == 2: move_view(cam, 1)
        if part == 3: move_view(cam, 2)
        if part == 4: move_view(cam, 3)
        if part == 5: move_view(cam, 1)
        if part == 6: move_view(cam, 2)
        if part == 7: move_view(cam, 3)
        if part == 8: move_view(cam, 4)
        if part == 9: move_view(cam, 5)
        if part == 10: move_view(cam, 6)
        if part == 11: move_view(cam, 7)
        if part == 12: move_view(cam, 8)



    Scene.wait(1.1/quality.frames_per_second)

    render_to_file(name, render_settings=quality, background_color=bgcol)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20

    # fourier_example(HD, bgcol=BLACK, anim=103)
    # fourier_example(LD, bgcol=BLACK, anim=103)
    # fractional_ex(HD, bgcol=TRANSPARENT, anim=6, part=4)
    # fractional_ex(LD, bgcol=BLACK, anim=6, part=3)
    button(HD, TRANSPARENT, anim=7)
    # button(LD, BLACK, anim=1)
