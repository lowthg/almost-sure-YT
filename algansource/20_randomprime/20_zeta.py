from algan import *
import manim as mn
import math
import functorch
import scipy as sp
import colorsys

from manim import MathTex, VGroup

from common.wigner import *
import mpmath as mp


from algan.external_libraries.manim import ArcBetweenPoints
from algan.rendering.post_processing.bloom import bloom_filter, bloom_filter_premultiply
from functools import partial
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader

sys.path.append('../')
import alganhelper as ah
import manimhelper as mh

LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 10)

# lightness = min(0.15 + vals[i // 4] * 0.3 * yvals[i % 4], 0.8)
# colx[0, i, :3] = torch.tensor([*colorsys.hls_to_rgb(np.angle(vals1[i // 4]) / (2 * PI) + 0.5, lightness, 0.85)])

def zeta_col(z):
    zmax = 2
    return Color(colorsys.hls_to_rgb(np.angle(z) / (2 * PI) + 0.05, min(abs(z) / zmax, 0.7), 1))

def add_rots(rots):
    dir, ang = rots[0]
    rot = ah.rotation_matrix(dir, ang)
    for dir, ang in rots[1:]:
        rot = rot @ ah.rotation_matrix(dir, ang)
    cos = (rot.trace() - 1) / 2
    angle = math.acos(cos)
    rot -= np.eye(3) * cos
    rot += rot.transpose()
    nm = np.linalg.norm(rot, axis=1, keepdims=False)
    i = np.argmax(nm)
    return torch.from_numpy(rot[i, :] / nm[i]), angle


def surface_func(f, res_x=319, res_y=319, mesh_x=None, mesh_y=None, mesh_col=DARK_BROWN):
    mesh_m = 32
    mesh_n = 32
    n = res_y
    m = res_x
    t = torch.zeros(m, n, 5)
    du = 1 / (m-1)
    dv = 1 / (n-1)
    for i1 in range(m):
        mesh_on1 = (i1+1) % mesh_m == 0
        for j1 in range(n):
            t[i1, j1, :], op = f(j1 * dv, (m-1-i1) * du)
            mesh_on = mesh_on1 or (j1+1) % mesh_n == 0
            if mesh_on:
                col = mesh_col.clone()
                col[4] = op
                t[i1, j1, :] = col

    mob = ImageMob(t)
    return mob

def zeta_surf_setup(xrange=(-16, 16), yrange=(-16, 16), shift=1., clip=False, eq3_txt=r'\zeta(x+iy)',
                    y_length=8., res_x=319, res_y=319):
    xmin, xmax = xrange
    ymax = yrange[1]
    ymin = yrange[0]
    zmax = 2
    zmaxplot = 8.5
    ax1 = mn.ThreeDAxes(x_range=[ymin, ymax *1.1], y_range=[xmin, xmax*1.1], z_range=[0, zmax],
                        x_length=y_length, y_length=8, z_length=3,
                        axis_config={'color': mn.WHITE, 'stroke_width': 8, 'include_ticks': False,
                                     "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                     "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                     }
                        )
    ax1.shift(-ax1.coords_to_point(0, 0, 0))
    xscale = torch.tensor(ax1.c2p(0, 1, 0), dtype=RIGHT.dtype)
    yscale = torch.tensor(ax1.c2p(1, 0, 0), dtype=RIGHT.dtype)
    zscale = torch.tensor(ax1.c2p(0, 0, 1), dtype=RIGHT.dtype)/2
    ax1.shift(mn.IN * shift)
    origin = torch.tensor(ax1.c2p(0, 0, 0), dtype=ORIGIN.dtype)

    def f2(u):
        x = u[:,:,:1] * (xmax - xmin) + xmin
        y = u[:,:,1:2] * (ymax - ymin) + ymin

        z = torch.tensor(abs(sp.special.zeta(x.numpy() + y.numpy() * 1j))).clamp(0, zmaxplot*1.2) + 0.0

        res =  torch.mul(x, xscale) + torch.mul(y, yscale) + torch.mul(z, zscale)
        res[:,:,2] += origin[2]

        return res

    def g(u, v):
        x = u * (xmax-xmin) + xmin
        z = sp.special.zeta(x + (v * (ymax-ymin) + ymin) * 1j)
        z1 = abs(z)
        col = zeta_col(z)
        # col = Color(colorsys.hls_to_rgb(np.angle(z)/(2*PI) +0.05, min(z1 /zmax, 0.7), 1))

        if z1 > zmaxplot and clip:
            col[4] = op = max(1 - (z1/zmaxplot - 1)*10, 0)
        else:
            op = 1.
        col[4] *= 0.9
        return col, op

    mob1 = surface_func(g, res_x=res_y, res_y=res_x).set_shader(basic_pbr_shader)
    mob1.smoothness = 0.7
    mob1.metallicness = 0
    mob1.set_location_by_function(f2)

    eq1 = ManimMob(mn.MathTex(r'x')).move_to(origin + xmax * 1.21 * xscale)
    #eq1.rotate_around_point(eq1.get_center(), 90, axis=yscale)
    eq1.orbit_around_point(eq1.get_center(), -90, axis=yscale)
    eq1.orbit_around_point(eq1.get_center(), -45, axis=zscale)
    eq2 = ManimMob(mn.MathTex(r'y')).move_to(origin + ymax * 1.4 * yscale + zscale * 0.3)
    eq2.orbit_around_point(eq2.get_center(), -90, axis=yscale)
    eq2.orbit_around_point(eq2.get_center(), -45, axis=zscale)
    pos3 = origin+zmax * zscale * 1.4 + xscale*5+yscale*2.7
    eq3_= mn.MathTex(eq3_txt)
    eq3_[0][0].set_color(col_WVD)
    eq3_[0][2::3].set_color(col_x)
    eq3_[0][4].set_color(col_i)
    eq3 = ManimMob(eq3_).move_to(pos3)
    eq3.orbit_around_point(eq3.get_center(), -90, axis=yscale)
    eq3.orbit_around_point(eq3.get_center(), -45, axis=zscale)
    eq4 = ManimMob(mn.MathTex(eq3_txt, stroke_width=12, stroke_color=mn.BLACK)).move_to(pos3)
    eq4.move(IN * 0.01)
    eq4.orbit_around_point(eq3.get_center(), -90, axis=yscale)
    eq4.orbit_around_point(eq3.get_center(), -45, axis=zscale)

    return xmin, xmax, ymin, ymax, zmax, zmaxplot, xscale, yscale, zscale, origin, Group(eq1, eq2, eq3, eq4), mob1, ax1

def zeta_plot(quality=LD, bgcol=BLACK, anim=0, part=0, **kwargs):
    xmin, xmax, ymin, ymax, zmax, zmaxplot, xscale, yscale, zscale, origin, eqs, mob1, ax1\
        = zeta_surf_setup(clip=False, **kwargs)
    ax2 = mn.ThreeDAxes(x_range=[ymin, ymax *1.1], y_range=[0, xmax*1.1], z_range=[0, zmax],
                        x_length=8, y_length= 8 * xmax*1.1/(xmax*1.1 - xmin), z_length=3,
                        axis_config={'color': mn.WHITE, 'stroke_width': 8, 'include_ticks': False,
                                     "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                     "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                     }
                        )

    origin0 = ax1.c2p(0,0,0)
    origin = torch.tensor(origin0, dtype=RIGHT.dtype)
    right = torch.tensor(ax1.c2p(0,1,0), dtype=RIGHT.dtype) - origin
    up = torch.tensor(ax1.c2p(1,0,0), dtype=RIGHT.dtype) - origin
    out = (torch.tensor(ax1.c2p(0,0,1), dtype=RIGHT.dtype) - origin)/2

    ax2.shift(ax1.c2p(0,0,0) - ax2.c2p(0,0,0))
    # ax1 = ManimMob(ax1)
    ax2 = ManimMob(ax2)

    arrs = Group(ManimMob(mn.Arrow3D(mn.ORIGIN, mn.RIGHT, color=mn.YELLOW)),
                 ManimMob(mn.Arrow3D(mn.ORIGIN, mn.UP, color=mn.RED)),
                 ManimMob(mn.Arrow3D(mn.ORIGIN, mn.OUT, color=mn.BLUE)))

    p = mob1.get_descendants()[1]
    loc = p.location
    col = p.color
    cam: Camera = Scene.get_camera()
    xproj = xscale / torch.inner(xscale, xscale)
    yproj = yscale / torch.inner(yscale, yscale)
    xcoord = torch.inner(loc, xproj)
    ycoord = torch.inner(loc, yproj)
    start = ((1 - xcoord) / (1 - xmin)).clamp(0, 1)
    starty = (torch.abs(ycoord) / ymax).clamp(0, 1)

    ylen=2.5
    xlen=8 * xmax*1.05/(xmax*1.05 - xmin)
    ymax3 = ylen/zscale[2]
    ax3 = mn.Axes(x_range=[0, xmax*1.05], y_range=[0, ymax3], x_length=xlen,
                  y_length=ylen,
                  axis_config={'color': mn.WHITE, 'stroke_width': 8, 'include_ticks': False,
                               "tip_width": 0.3 * mn.DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.3 * mn.DEFAULT_ARROW_TIP_LENGTH,
                               },
                  )
    yaxis = mn.NumberLine(length=8, stroke_width=8, include_ticks=False, include_tip=True,
                          tip_width=0.3 * mn.DEFAULT_ARROW_TIP_LENGTH, tip_height=0.3 * mn.DEFAULT_ARROW_TIP_LENGTH)
    xaxis = yaxis.copy()
    VGroup(xaxis, yaxis).move_to(ax3.c2p(0,0)).shift((xlen-4)*mn.RIGHT)
    xvals = torch.linspace(1.354, xmax, 600)
    yvals = torch.special.zeta(xvals, 1.) + 0.05
    xvals2 = torch.linspace(1.2, xmax, 600)
    yvals2 = torch.special.zeta(xvals2, 1.) + 0.05
    xticks = mh.get_xticks(ax3, [0,1, 5, 10, 15], length=0.08, buff=0.12, font_size=20, label_color=col_num)
    yticks = mh.get_yticks(ax3, [0,1,2,3], length=0.08, buff=0.12, font_size=20, label_color=col_num)
    dashed = mn.DashedLine(ax3.c2p(1,0), ax3.c2p(1,ymax3), color=mn.GREY, stroke_width=6)
    dashed2 = mn.DashedLine(ax3.c2p(0,1), ax3.c2p(xmax*1.,1), color=mn.GREY, stroke_width=6)
    eqx = mn.MathTex(r'x', font_size=26, color=col_x).next_to(ax3.x_axis.get_right(), mn.RIGHT, buff=0.1)
    eqy = mn.MathTex(r'y', font_size=26, color=col_x).next_to(yaxis.get_right(), mn.RIGHT, buff=0.1)
    eqz = mn.MathTex(r'\zeta(x)', font_size=30, stroke_width=1.5).move_to(ax3.c2p(3., 2.5))
    gp2 = VGroup(yaxis, eqy)
    gp2.rotate(PI/2, mn.DOWN, about_point=ax3.c2p(0,0,0))
    eqz[0][0].set_color(col_WVD)
    eqz[0][2].set_color(col_x)

    gp = mn.VGroup(ax3, xticks, yticks, eqx, dashed, dashed2, eqz, xaxis, gp2)
    gp.rotate(90 * mn.DEGREES, mn.UP)
    gp.rotate(90 * mn.DEGREES, mn.RIGHT)
    gp.shift(origin0 - ax3.c2p(0, 0, 0))

    with Off():
        cam.set_distance_to_screen(12)
        cam.orbit_around_point(ORIGIN, -90, UP)
        cam.orbit_around_point(ORIGIN, -90, RIGHT)
        shift = cam.get_center() * -0.48 + UP * 2. + IN * 0.2
        cam.move(shift)
    dir, angle = add_rots([(RIGHT, PI/2), (UP, PI/2), (RIGHT, -2*PI/3), (OUT, PI/4)])

    if anim == 1:
        pts = xvals.unsqueeze(-1) * right + yvals.unsqueeze(-1) * zscale + origin
        pts2 = xvals2.unsqueeze(-1) * right + yvals2.unsqueeze(-1) * zscale + origin
        crv = ah.curve_surface(pts, width=0.04, normals=up, closed=False, color=zeta_col(2.))
        crv2 = ah.curve_surface(pts2, width=0.04, normals=up, closed=False, color=zeta_col(2.))

        ax3_ = ManimMob(gp[:-2])
        with Off():
            ax3_.spawn()
            crv.spawn()

        with Sync(run_time=1):
            cam.move(-shift)
            cam.orbit_around_point(ORIGIN, angle*RADIANS_TO_DEGREES, dir)
            crv.despawn()
            crv2.spawn()
            ManimMob(gp2).spawn()

        col2 = col.clone()
        col2[start.gt(0)] = 0
        with Off():
            mob1.spawn()
        with Sync():
            with Sync(run_time=2):
                crv2.despawn()
                for i in range(4,7):
                    ax3_.submobjects[i].despawn()
            with Seq():
                for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=2.):
                    col3 = col2.clone()
                    col3[starty.gt(frame.u)] = 0
                    with frame.context:
                        p.set_non_recursive(color=col3)

    if anim >= 2:
        gp = mn.VGroup(ax3.y_axis, xticks, yticks, eqx, yaxis, eqy)
        ax3_ = ManimMob(gp)
        xax = ManimMob(xaxis)
        with Off():
            cam.move(-shift)
            cam.orbit_around_point(ORIGIN, angle * RADIANS_TO_DEGREES, dir)
            # cam.set_distance_to_screen(100)
            # cam.orbit_around_point(ORIGIN, -120, RIGHT)
            # cam.orbit_around_point(ORIGIN, 45, OUT)
            ax3_.spawn()
            mob1.spawn()
            eqs[2:].spawn()

    if anim == 2:
        xaxis1 = ax3.x_axis
        xaxis1_ = ManimMob(xaxis1)
        with Off():
            xaxis1_.spawn()
        rate_func = rate_funcs.smooth
        run_time=2.

        with Sync():
            with Seq(rate_func=rate_func, run_time=run_time):
                xaxis1_.become(xax)
            with Seq():
                for frame in ah.FrameStepper(fps=quality.frames_per_second, rate_func=rate_func, run_time=run_time):
                    col2 = col.clone()
                    s = frame.u
                    col2[start.gt(s)] = 0
                    with frame.context:
                        p.set_non_recursive(color=col2)

    if anim >= 3:
        ball = Sphere(radius=0.07, color=YELLOW)
        balls = Group(*[ball.clone().move_to(origin+xscale*x) for x in [-2,-4,-6,-8,-10,-12,-14,-16]])
        # ball = ManimMob(mn.Dot(radius=0.1, color=mn.YELLOW))

        ball.move_to(origin+xscale*-2)
        with Off():
            xax.spawn()
            balls.spawn()

    if anim >= 4:
        gamma = float(mp.im(mp.zetazero(1)))
        ball = Sphere(radius=0.07, color=YELLOW)
        balls = Group(*[ball.clone().move_to(origin+xscale*0.5 + yscale*y) for y in [gamma, -gamma]])
        with Off():
            balls.spawn()

    if anim >= 5:
        dir2, angle2 = add_rots([(dir, -angle), (RIGHT, PI/2), (UP, -PI/2), (OUT, -PI/2)])
        loc2 = loc.clone()
        loc2[...,2] = origin[2]-0.1
        yaxis2 = yaxis.set_stroke(width=3)
        yaxis2_ = ManimMob(yaxis2).orbit_around_point(origin, 90, RIGHT)
        yaxis_ = ax3_.submobjects[4]
        xaxis2 = xaxis.set_stroke(width=3)
        xaxis2_ = ManimMob(xaxis2).orbit_around_point(origin, 90, UP)
        shift2 = IN * 5.3 + RIGHT * 0.1
        if anim == 5:
            with Sync():
                with Sync(run_time=1):
                    eqs[2:].despawn()
                    ax3_.submobjects[0].despawn()
                    ax3_.submobjects[1].despawn()
                    ax3_.submobjects[2].despawn()
                    ax3_.submobjects[3].despawn()
                    ax3_.submobjects[5].despawn()
                    yaxis_.become(yaxis2_)
                    xax.become(xaxis2_)
                    cam.orbit_around_point(ORIGIN, -angle2*RADIANS_TO_DEGREES, dir2)
                    p.set_non_recursive(location=loc2)
                with Sync(run_time=1.5):
                    cam.move(shift2)
        else:
            with Off():
                eqs[2:].despawn()
                ax3_.submobjects[1].despawn()
                ax3_.submobjects[2].despawn()
                ax3_.submobjects[3].despawn()
                ax3_.submobjects[5].despawn()
                ax3_.submobjects[0].despawn()
                yaxis_.despawn()
                yaxis2_.spawn()
                xax.despawn()
                xaxis2_.spawn()
                cam.orbit_around_point(ORIGIN, -angle2 * RADIANS_TO_DEGREES, dir2)
                p.set_non_recursive(location=loc2)
                cam.move(shift2)

        if anim >= 6:
            line1 = mn.Line(origin + yscale * ymin + xscale, origin + yscale * ymax + xscale, stroke_width=3, stroke_color=mn.WHITE)
            line1_ = ManimMob(line1)
            col2 = col.clone()
            col2[...,-1] *= 0.5
            with Off():
                line1_.spawn()
                p.set_non_recursive(color=col2)
        if anim >= 7:
            line2 = mn.DashedLine(origin + yscale * ymin + xscale/2, origin + yscale * ymax + xscale/2, stroke_width=4,
                                  stroke_color=mn.WHITE)
            line2_ = ManimMob(line2)
            with Off():
                line2_.spawn()

        if anim == 8:
            m0 = 10
            m1 = 40
            res_y1 = 32 * m0 - 1
            res_y2 = 32 * m1 - 1
            scale = (res_y2 - 1)/(res_y1 - 1)
            ymax2 = ymax * scale
            xmin2, xmax2, _, _, zmax2, zmaxplot2, xscale2, yscale2, zscale2, origin2, _, mob2, _ \
                = zeta_surf_setup(clip=False, yrange=(-ymax2, ymax2), y_length=8*scale, res_y=res_y2, **kwargs)
            p2 = mob2.get_descendants()[1]
            loc2 = p2.location.clone()
            loc2[..., 2] = origin[2] - 0.1
            col2 = p2.color.clone()
            col2[..., -1] *= 0.5

            gammas = []
            for k in range(1, 2000):
                y = float(mp.im(mp.zetazero(k)))
                if y < ymax2:
                    gammas.append(y)
                else:
                    break

            balls2 = []
            for _ in gammas:
                balls2.append((ball.clone(), ball.clone()))

            with Off():
                mob1.despawn()
                mob2.spawn()
                balls.despawn()

            yproj2 = yscale2 / torch.inner(yscale2, yscale2)
            ycoord2 = torch.abs(torch.inner(loc2, yproj2))

            run_time = 0.2
            u0 = 0.1 * (part-1)
            if part >= 9:
                run_time=0.1
                u0 -= 0.05 * (part-9)

            for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=run_time, rate_func=rate_funcs.identity):
                u = frame.u * run_time/2 + u0
                ymax3 = ymax + (ymax2 - ymax) * u
                scale2 = ymax / ymax3
                loc3 = loc2.clone()
                col3 = col2.clone()
                loc3[...,0] *= scale2
                col3[ycoord2.gt(ymax/scale2)] = 0

                y3 = 0.
                with frame.context:
                    p2.set_non_recursive(location=loc3, color=col3)
                    for y, (ball1, ball2) in zip(gammas, balls2):
                        y1 = y * scale2
                        ball1.move_to(origin+xscale/2+yscale*y1)
                        ball2.move_to(origin+xscale/2-yscale*y1)
                        if y3 < y and y1 < ymax:
                            ball1.spawn()
                            ball2.spawn()
                            y3 = y


    print(gammas)
    Scene.wait(0.1)

    name = 'zeta_plot{}'.format(anim)
    if part > 0:
        name = name + '_{}'.format(part)
    render_to_file(name, render_settings=quality, background_color=bgcol)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 1

    # zeta_plot(quality=HD, bgcol=BLACK, anim=8, part=1)
    # zeta_plot(quality=HD, bgcol=BLACK, anim=8, part=2)
    # zeta_plot(quality=HD, bgcol=BLACK, anim=8, part=3)
    # zeta_plot(quality=HD, bgcol=BLACK, anim=8, part=4)
    # zeta_plot(quality=HD, bgcol=BLACK, anim=8, part=5)
    # zeta_plot(quality=HD, bgcol=BLACK, anim=8, part=6)
    # zeta_plot(quality=HD, bgcol=BLACK, anim=8, part=7)
    # zeta_plot(quality=HD, bgcol=BLACK, anim=8, part=8)
    zeta_plot(quality=HD, bgcol=BLACK, anim=7)
