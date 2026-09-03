from algan import *
import manim as mn
import math
import functorch
import scipy as sp
import colorsys

from manim import MathTex, VGroup

from common.wigner import *


from algan.external_libraries.manim import ArcBetweenPoints
from algan.rendering.post_processing.bloom import bloom_filter, bloom_filter_premultiply
from functools import partial
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader

sys.path.append('../')
import alganhelper as ah
import manimhelper as mh

LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)

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

def zeta_surf_setup(xrange=(-16, 16), yrange=(-16, 16), shift=1., clip=False, eq3_txt=r'\zeta(x+iy)'):
    xmin, xmax = xrange
    ymax = yrange[1]
    ymin = yrange[0]
    zmax = 2
    zmaxplot = 8.5
    ax1 = mn.ThreeDAxes(x_range=[ymin, ymax *1.1], y_range=[xmin, xmax*1.1], z_range=[0, zmax],
                        x_length=8, y_length=8, z_length=3,
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

    mob1 = surface_func(g).set_shader(basic_pbr_shader)
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
    eq3 = ManimMob(mn.MathTex(eq3_txt)).move_to(origin+zmax * zscale * 1.4 + xscale*4+yscale*3)
    eq3.orbit_around_point(eq3.get_center(), -90, axis=yscale)
    eq3.orbit_around_point(eq3.get_center(), -45, axis=zscale)
    eq4 = ManimMob(mn.MathTex(eq3_txt, stroke_width=10, stroke_color=mn.BLACK)).move_to(origin+zmax * zscale * 1.4 + xscale*4+yscale*3)
    eq4.move(IN * 0.01)
    eq4.orbit_around_point(eq3.get_center(), -90, axis=yscale)
    eq4.orbit_around_point(eq3.get_center(), -45, axis=zscale)

    return xmin, xmax, ymin, ymax, zmax, zmaxplot, xscale, yscale, zscale, origin, Group(eq1, eq2, eq3, eq4), mob1, ax1

def zeta_plot(quality=LD, bgcol=BLACK, anim=0, **kwargs):
    xmin, xmax, ymin, ymax, zmax, zmaxplot, xscale, yscale, zscale, origin, eqs, mob1, ax1\
        = zeta_surf_setup(clip=True, **kwargs)
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

    if anim == 1:
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
        yaxis = mn.NumberLine(length=8*1.025, stroke_width=8, include_ticks=False, include_tip=True,
                              tip_width=0.3 * mn.DEFAULT_ARROW_TIP_LENGTH, tip_height=0.3 * mn.DEFAULT_ARROW_TIP_LENGTH)
        yaxis.move_to(ax3.c2p(0,0)).shift(4*0.05*mn.RIGHT)
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

        gp = mn.VGroup(ax3, xticks, yticks, eqx, dashed, dashed2, eqz, gp2)
        pts = xvals.unsqueeze(-1) * right + yvals.unsqueeze(-1) * zscale + origin
        pts2 = xvals2.unsqueeze(-1) * right + yvals2.unsqueeze(-1) * zscale + origin
        crv = ah.curve_surface(pts, width=0.04, normals=up, closed=False, color=zeta_col(2.))
        crv2 = ah.curve_surface(pts2, width=0.04, normals=up, closed=False, color=zeta_col(2.))

        with Off():
            cam.set_distance_to_screen(1000)
            cam.orbit_around_point(ORIGIN, -90, UP)
            cam.orbit_around_point(ORIGIN, -90, RIGHT)
            shift = cam.get_center() * -0.4 + UP*2. + IN * 0.2
            cam.move(shift)


        gp.rotate(90*mn.DEGREES, mn.UP)
        gp.rotate(90*mn.DEGREES, mn.RIGHT)
        gp.shift(origin0-ax3.c2p(0,0,0))
        ax3_ = ManimMob(gp[:-1])
        with Off():
            ax3_.spawn()
            crv.spawn()

        dir, angle = add_rots([(RIGHT, PI/2), (UP, PI/2), (RIGHT, -2*PI/3), (OUT, PI/4)])
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
            # ManimMob(ax3.x_axis.copy().rotate(PI/2, mn.OUT, origin0)).spawn()
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

        # with Off():
        #     p.set_non_recursive(color=col2)

        # with Sync():
        #     mob1.spawn()

    ax1 = ManimMob(ax1)

    if anim >= 2:
        with Off():
            cam.set_distance_to_screen(12)
            cam.orbit_around_point(ORIGIN, -120, RIGHT)
            cam.orbit_around_point(ORIGIN, 45, OUT)
            ax2.spawn()
            mob1.spawn()
            eqs.spawn()

    if anim == 2:
        with Sync():
            ax2.submobjects[1].become(ax1.submobjects[1])
            with Seq():
                for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=1., rate_func=mh.rate_func_quad(0.3,0.)):
                    col2 = col.clone()
                    s = frame.u
                    col2[start.gt(s)] = 0
                    with frame.context:
                        p.set_non_recursive(color=col2)

    Scene.wait(0.1)

    name = 'zeta_plot{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 6

    zeta_plot(quality=HD, bgcol=BLACK, anim=1)
