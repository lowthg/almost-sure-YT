import numpy as np
import torch
import colorsys
from algan import *
import manim as mn
import scipy as sp
from algan.external_libraries.manim import DashedLine
from algan.external_libraries.manim.utils.color.SVGNAMES import INDIGO, SILVER
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader, default_shader
from manim import VGroup, ManimColor, MathTex

sys.path.append('../../')
import alganhelper as ah
from common.wigner import *
LD = RenderSettings((854, 480), 15)
LD2 = RenderSettings((854, 480), 30)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 15)


import numpy as np
import math


def eq_shadow(eq: mn.VGroup, fg_z_index=4., bg_z_index=0., bg_color=mn.BLACK, bg_stroke_width=10.):
    res = mn.VGroup()
    for eq1 in eq:
        elem = mn.VGroup()
        for eq2 in eq1:
            elem.add(mn.VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res


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
    return rot[i, :] / nm[i], angle

def normalize_curves(curves):
    for (u, v, w) in curves:
        r = np.sqrt(u*u + v*v + w*w)
        u /= r
        v /= r
        w /= r


def CubicCurve(du=0.01, a=11):
    """
    :return: array of (u, v, w) for curve: vw(v+w)+uw(u+w)+uv(u+v) = auvw
    """

    """
    start with points (1,v,w) where (1+w)v^2 + (1+w^2-aw)v + (1+w)w = 0
    on range -1 <= w <= 0 and v >= 0
    to solve, write b = 2(1+w)/(1+w^2-aw). Solving quadratic gives
    v^2 + 2v/b + w = 0
    wb/v^2 + 2/v + b = 0
    v = -wb/(1+sqrt(1-wb^2))
    """
    n = int(1/du + 1)
    w1 = np.linspace(0., -1., n + 1)
    b = (w1 + 1) / (w1*w1 - w1 * a + 1) * 2
    v1 = -w1 * b / (np.sqrt(-w1*b*b + 1) + 1)
    u1 = np.ones(n+1, dtype=w1.dtype)

    # uvw1 joins (1,0,0) to (1,0,-1)
    # uvw2 joins (1,0,-1) to (0,0,-1)
    (u2,v2,w2) = (-w1[::-1], -v1[::-1], -u1[::-1])
    # uvw3 joins (0,0,-1) to (0,1,-1)
    (u3,v3,w3) = (-v1, -w1, -u1)
    # uvw4 joins (0,1,-1) to (0,1,0)
    (u4,v4,w4) = (v1[::-1], u1[::-1], w1[::-1])
    # uvw5 joins (0,1,0) to (-1,1,0)
    (u5,v5,w5) = (w1, u1, v1)
    # uvw6 joins (-1,1,0) to (-1,0,0)
    (u6,v6,w6) = (-u1[::-1], -w1[::-1], -v1[::-1])


    res = [(np.concat((-u1[:], -u2[1:], -u3[1:], -u4[1:], -u5[1:], -u6[1:], u1[1:], u2[1:], u3[1:], u4[1:], u5[1:], u6[1:])),
            np.concat((-v1[:], -v2[1:], -v3[1:], -v4[1:], -v5[1:], -v6[1:], v1[1:], v2[1:], v3[1:], v4[1:], v5[1:], v6[1:])),
            np.concat((-w1[:], -w2[1:], -w3[1:], -w4[1:], -w5[1:], -w6[1:], w1[1:], w2[1:], w3[1:], w4[1:], w5[1:], w6[1:]))
            )]

    """
    points (1, v, w) where (1+w)v^2 + (1+w^2-aw)v + (1+w)w = 0
    on range 0 < w <= 1 and w >= v > 0
    range is [p,1] where w=p solves (1+w)w^2 + (1+w^2-aw)w+(1+w)w=0
    p = c - sqrt(c^2 - 1) where c = (a-2)/4
    then, with b = (aw-w^2-1)/(1+w)/2,
    v^2 - 2bv + w = 0
    v = b - sqrt(b^2-w)
    """

    c = (a-2)/4
    p = c - math.sqrt(c*c-1)
    n = int((1-p)/du+1)
    w1 = np.linspace(p, 1., n+1)
    b = (w1*a - w1*w1 - 1)/(w1+1)/2
    v1 = b - np.sqrt(b*b-w1)
    u1 = np.ones(n+1, dtype=w1.dtype)

    # v = b - sqrt(b^2-1)

    # uvw1 joins (1,p,p) to (1,q,1)
    # uvw2 joins (1,q,1) to (p,p,1)
    (u2, v2, w2) = (w1[::-1], v1[::-1], u1[::-1])
    # uvw3 joins (p,p,1) to (q,1,1)
    (u3,v3,w3) = (v1, w1, u1)
    # uvw4 joins (q,1,1) to (p,1,p)
    (u4,v4,w4) = (v1[::-1], u1[::-1], w1[::-1])
    # uvw5 joins (p,1,p) to (1,1,q)
    (u5,v5,w5) = (w1, u1, v1)
    # uvw6 joins (1,1,q) to (1,p,p)
    (u6,v6,w6) = (u1[::-1], w1[::-1], v1[::-1])

    res += [(np.concat((u1[:], u2[1:], u3[1:], u4[1:], u5[1:], u6[1:])),
             np.concat((v1[:], v2[1:], v3[1:], v4[1:], v5[1:], v6[1:])),
             np.concat((w1[:], w2[1:], w3[1:], w4[1:], w5[1:], w6[1:]))
             )]

    normalize_curves(res)

    res += [tuple(-u for u in res[-1])]

    return res

point_p1 = (1,0,0)
point_p2 = (0,1,0)
point_p3 = (0,0,1)
point_o1 = (0,1,-1)
point_o2 = (1,0,-1)
point_o3 = (1,-1,0)
point_o3_ = (-1,1,0)
point_q = (10,3,2)

point_r1 = (10,3,2)
point_r2 = (715, 14664, -3619)
point_r4 = (164522506539145, -196932807438576, 13030716467024711)
point_r4_ = ( 0.164522506539145,
             -0.196932807438576,
             13.030716467024711)
point_r8 = (1710959400725563597554818136978688404216526725221349245104503985,
            -2454018952485961198258264459149041098107906103325177114823668256,
            -56542945251126914281095352993207832740799500248513925316036689)
point_r8_ = (17.10959400725563597554818136978688404216526725221349245104503985,
            -24.54018952485961198258264459149041098107906103325177114823668256,
             -0.56542945251126914281095352993207832740799500248513925316036689)
point_r1_2 = (3, 15, 10)
point_r9 = (191351933902876166269149126585145312311157305638537402696309772369890426669333578,
            41248744472058697085059064167596864978309517527133584614904575465015648101827615,
            158850414786674863699812567622291227638480863227727045318648072891631625503050035)
point_r9_ = (1.91351933902876166269149126585145312311157305638537402696309772369890426669333578,
             0.41248744472058697085059064167596864978309517527133584614904575465015648101827615,
             1.58850414786674863699812567622291227638480863227727045318648072891631625503050035)

pos_region_color = GREEN
line_col = Color((.398, .887, 1.))


def CubicCurvePosNeg(du = 0.02, a=11):
    """
    points (1, v, w) where (1+w)v^2 + (1+w^2-aw)v + (1+w)w = 0
    on range 0 < w <= 1 and w >= v > 0
    range is [p,1] where w=p solves (1+w)w^2 + (1+w^2-aw)w+(1+w)w=0
    p = c - sqrt(c^2 - 1) where c = (a-2)/4
    then, with b = (aw-w^2-1)/(1+w)/2,
    v^2 - 2bv + w = 0
    v = b - sqrt(b^2-w)
    region with v+w > 1
    (1+w)(v-w)^2+(1+w^2-aw)(
    """

def curve_surface_loc(pts, normals=None, tangents=None, resolution=8, width=0.03, closed=True):
    if normals is None:
        normals = pts
    if tangents is None:
        if closed:
            tangents = torch.roll(pts, shifts=1, dims=0) - torch.roll(pts, shifts=-1, dims=0)
        else:
            tangents = torch.empty_like(pts)
            tangents[1:-1] = pts[2:] - pts[:-2]
            tangents[0] = pts[1] - pts[0]
            tangents[-1] = pts[-1] - pts[-2]
    du = torch.nn.functional.normalize(normals, dim=1)
    dv = torch.nn.functional.normalize(torch.linalg.cross(normals, tangents), dim=1)
    theta = torch.linspace(-math.pi, math.pi, resolution, device=pts.device)
    return (
            pts[:, None, :] +
            (torch.sin(theta)[None, :, None] * du[:, None, :] +
             torch.cos(theta)[None, :, None] * dv[:, None, :]) * (width / 2)
    ).reshape(1, -1, 3).float()

def curve_surface(pts, normals=None, tangents=None, resolution=8, color=BLUE, width=0.03, closed=True, **kwargs):
    n = pts.shape[0]
    crv = Surface(grid_height=resolution, grid_width=n, color=color, **kwargs)
    crv.get_descendants()[1].location[...] = curve_surface_loc(pts, normals, tangents, resolution, width, closed)
    return crv


def positive_region(r, curve, curve_obj, use_xyz=False):
    line_col2 = mn.ManimColor.from_rgb((.6787, 1., .5337))
    # dots = []
    # for u, v, w in [(1,0,0), (0,1,0), (0,0,1)] if use_xyz else [(0,1,1), (1,0,1), (1,1,0)]:
    #     for pt in [(u, v, w), (-u, -v, -w)]:
    #         s = r / math.sqrt(sum([_ * _ for _ in pt]))
    #         p = (pt[0] * mn.RIGHT + pt[1] * mn.UP + pt[2] * mn.IN)*s
    #         dots.append(ManimMob(mn.Dot3D(p, radius=0.06, color=line_col2)))

    res = 20 if use_xyz else 10
    pos_region = Surface(grid_height=res, grid_width=res, color=pos_region_color, opacity=0.6)
    pos_region2 = Surface(grid_height=res, grid_width=res, color=pos_region_color, opacity=0.6)
    loc = pos_region.get_descendants()[1].location

    pts = [RIGHT, UP, OUT] if use_xyz else [OUT+RIGHT, OUT+UP, RIGHT+UP]
    for i in range(res):
        u = i/(res-1)
        for j in range(res):
            v = j/(res-1)
            loc[0,i*res+j,:] = (pts[0] * (1-u) + pts[1] * u) * v + pts[2] * (1-v)

    loc[...] = torch.nn.functional.normalize(loc, dim=2) * r * 1.004
    pos_region2.get_descendants()[1].location[...] = -loc

    line_col2 = Color(line_col2.to_rgb())
    p = curve_obj.get_descendants()[1]
    col = p.color.clone()
    u, v, w = curve
    eps = 0.002
    if use_xyz:
        mask = (u +eps > 0) & (v + eps > 0) & (w + eps > 0)
    else:
        mask = (u < v + w + eps) & (v < u + w + eps) & (w < u + v + eps)
    mask_expanded = torch.from_numpy(mask).repeat_interleave(8)
    col[0, mask_expanded, :3] = line_col2[:3]

    return Group(pos_region, pos_region2), col

def get_points(r, pts, colors, width=0.12, use_xyz=False):
    res = []
    for i, (u, v, w) in enumerate(pts):
        x, y, z = (-u+v+w, u-v+w, u+v-w) if use_xyz else (u,v,w)
        col = colors[i % len(pts)]
        s = math.sqrt(x*x+y*y+z*z)
        p = (x * mn.RIGHT + y * mn.UP + z * mn.IN) * (r / s)
        res.append(Group(
            ManimMob(mn.Dot3D(p, radius=width/2, color=col)),
            ManimMob(mn.Dot3D(-p, radius=width/2, color=col))
        ))
    return res

def get_sphere(r, circle_opacity=0.5, sphere_opacity=0.8):
    ball = Sphere(radius=r, opacity=sphere_opacity, color=GREY, grid_height=40)#.move_to(origin)
    circ = Circle(radius=r, border_width=3, border_color=WHITE, filled=False, opacity=circle_opacity)
    circ2 = circ.clone().orbit_around_point(ORIGIN, 90, RIGHT)
    circ3 = circ.clone().orbit_around_point(ORIGIN, 90, UP)
    return Group(ball, circ, circ2, circ3)

def circle_thru_points(r, pt0, pt1, dtheta1=PI/10, dtheta2=PI/10, width=0.03, fps=15, run_time=1., n=20, color=WHITE):
    arc = Surface(grid_height=8, grid_width=n, color=color)
    dtype = arc.get_descendants()[1].location.dtype

    p0 = torch.tensor([*pt0], dtype=dtype)
    p0 /= torch.norm(p0)
    p1 = torch.tensor([*pt1], dtype=dtype)
    p1 /= torch.norm(p1)
    p0[2] *= -1
    p1[2] *= -1
    p2 = p1 - torch.dot(p0, p1) * p0
    p2 /= torch.norm(p2)
    theta1 = math.acos(torch.dot(p0, p1))

    with Off():
        arc.spawn()
    p = arc.get_descendants()[1]
    if run_time <= 0.:
        theta2 = theta1 + dtheta2
        theta = torch.linspace(-dtheta1, theta2, n, device=p0.device, dtype=dtype)[:, None, None]
        u2 = torch.cos(theta) * p0 + torch.sin(theta) * p2
        loc = curve_surface_loc(u2 * r, resolution=8, width=width, closed=False)
        with Off():
            p.set_non_recursive(location=loc)
    else:
        with Seq():
            for frame in ah.FrameStepper(fps=fps, step=1, run_time=run_time, rate_func=rate_funcs.identity):
                u = frame.u
                theta2 = u * (theta1+dtheta2) + (1-u) * (-dtheta1)
                theta = torch.linspace(-dtheta1, theta2, n, device=p0.device, dtype=dtype)[:, None, None]
                u2 = torch.cos(theta) * p0 + torch.sin(theta) * p2
                loc = curve_surface_loc(u2 * r, resolution=8, width=width, closed=False)
                with frame.context:
                    p.set_non_recursive(location=loc)
    return arc

def point(uvw, use_xyz=False, invert=False):
    res = (-uvw[0]+uvw[1]+uvw[2], uvw[0]-uvw[1]+uvw[2], uvw[0]+uvw[1]-uvw[2]) if use_xyz else uvw
    if invert:
        res = tuple(-_ for _ in res)
    return res

def cube_curve2(quality=LD, bgcol=BLACK, use_xyz=True, anim=1):
    r = 2.
    xrange = [-1.2, 1.2]
    xlen = (xrange[1] - xrange[0])*r
    ax = mn.ThreeDAxes(x_length=xlen, y_length=xlen, z_length=xlen, x_range=xrange, y_range=xrange, z_range=xrange,
                       z_axis_config={'rotation': PI},
                       axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                                    "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                    "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                    },
                       )
    ax.shift(-ax.coords_to_point(0, 0, 0))
    xyz_str = r'xyz' if use_xyz else r'uvw'
    eq = mn.MathTex(xyz_str, font_size=30)[0].rotate(-PI/2, mn.RIGHT)
    eq[0].move_to(ax.coords_to_point(1.3,0,0)).rotate(PI,mn.IN)
    eq[1].move_to(ax.coords_to_point(0,1.3,0)).rotate(-PI/2,mn.IN)
    eq[2].move_to(ax.coords_to_point(0,0,1.26)).rotate(-3*PI/4, mn.IN)

    sphere = get_sphere(r)

    curves = CubicCurve(du=0.02)
    if use_xyz:
        curves = [(-u+v+w, u-v+w, u+v-w) for (u,v,w) in curves]
        normalize_curves(curves)

    line_col = Color((.398, .887, 1.))
    plts = []
    for u,v,w in curves:
        pts = (torch.outer(torch.from_numpy(u), RIGHT) + torch.outer(torch.from_numpy(v), UP) + torch.outer(torch.from_numpy(w), OUT)) * r
        crv = curve_surface(pts, width=0.03, color=line_col)
        plts.append(crv)

    ax = ManimMob(ax)
    cam: Camera = Scene.get_camera()
    eq = ManimMob(eq)
    plt = Group(*plts)

    with Off():
        cam.set_distance_to_screen(10).move_to(cam.get_center() * 0.8)
        # cam.set_euler_angles(80*DEGREES, 0*DEGREES, 150*DEGREES)
        # cam.set_euler_angles(60*DEGREES, 0*DEGREES, 150*DEGREES)
        cam.set_euler_angles(80*DEGREES, 0*DEGREES, 120*DEGREES)
        cam.move(OUT*0.32)
        ax.spawn()
        sphere.spawn()
        eq.spawn()

    if anim == 1:
        with Sync():
            plt.spawn()
        with Sync(run_time=6, rate_func=rate_funcs.identity):
            cam.orbit_around_point(ORIGIN, -20, cam.get_right_direction())
            cam.orbit_around_point(ORIGIN, -330, OUT)
        return
    else:
        with Off():
            plt.spawn()
            cam.orbit_around_point(ORIGIN, -20, cam.get_right_direction())
            cam.orbit_around_point(ORIGIN, -330, OUT)

    # with Sync() if anim == 1 else Off():
    #     cam.rotate_around_point(ORIGIN, axis=RIGHT, num_degrees=-20)
    #     cam.rotate_around_point(ORIGIN, axis=OUT, num_degrees=30-360)
    #     Camera().

    pos_region, pos_col = positive_region(r, curves[1], plt[1], use_xyz=use_xyz)

    with Sync() if anim == 2 else Off():
        pos_region.spawn()
        plt[1].get_descendants()[1].set_non_recursive(color=pos_col)
        plt[2].get_descendants()[1].set_non_recursive(color=pos_col)
    if anim == 2:
        return

    p_1 = point((1,0,0), use_xyz)
    p_r1 = point(point_r1, use_xyz)
    pts = get_points(r, [p_1, p_r1],
                     [mn.YELLOW, mn.ORANGE])
    dots = Group(*pts)

    eq = mn.MathTex(r'PQ', font_size=30)[0].rotate(-PI/2, mn.RIGHT)
    c0 = pts[0][0].get_center()[0,0,:].numpy()
    c1 = pts[1][0].get_center()[0,0,:].numpy()
    eq[0].move_to(c0*1.2).rotate(PI,mn.IN)
    eq[1].move_to(c1*1.2).rotate(PI,mn.IN)
    eq = ManimMob(eq)

    with Sync() if anim == 3 else Off():
        cam.orbit_around_point(ORIGIN, 40, OUT)
    with Sync() if anim == 3 else Off():
        dots.spawn()
        eq.spawn()

    if anim == 3:
        return

    p_r2 = point(point_r2, use_xyz)
    pts = get_points(r, [p_r2], [mn.PINK])
    if anim == 4:
        with Sync():
            circle_thru_points(r, p_r1, p_r2, fps=quality.frames_per_second, run_time=2.)
            with Seq():
                Scene.wait(0.5)
                with Sync(run_time=1.5):
                    cam.orbit_around_point(ORIGIN, -55, cam.get_right_direction())
        with Sync():
            pts[0].spawn()
        return
    with Off():
        circle_thru_points(r, p_r1, p_r2, fps=quality.frames_per_second, run_time=0.1)
        cam.orbit_around_point(ORIGIN, -55, cam.get_right_direction())
        pts[0].spawn()

    p_r4 = point(point_r4_, use_xyz, True)
    pts = get_points(r, [p_r4], [mn.PINK])
    if anim == 5:
        with Sync(run_time=1.):
            cam.orbit_around_point(ORIGIN, 50, cam.get_right_direction())
        with Sync(run_time=1.5):
            cam.orbit_around_point(ORIGIN, 150, OUT)
        circle_thru_points(r, p_r2, p_r4, fps=quality.frames_per_second, run_time=1.)
        with Sync():
            pts[0].spawn()
        return
    with Off():
        cam.orbit_around_point(ORIGIN, 50, cam.get_right_direction())
        cam.orbit_around_point(ORIGIN, 150, OUT)
        circle_thru_points(r, p_r2, p_r4, fps=quality.frames_per_second, run_time=0.1)
        pts[0].spawn()

    p_r8 = point(point_r8_, use_xyz, True)
    pts = get_points(r, [p_r8], [mn.PINK])
    if anim == 6:
        with Sync():
            circle_thru_points(r, p_r4, p_r8, fps=quality.frames_per_second, run_time=1.5)
            with Sync(run_time=1.5):
                cam.orbit_around_point(ORIGIN, 40, OUT)
        pts[0].spawn()
        return
    with Off():
        circle_thru_points(r, p_r4, p_r8, fps=quality.frames_per_second, run_time=0.1)
        cam.orbit_around_point(ORIGIN, 40, OUT)
        pts[0].spawn()

    p1_2 = point(point_r1_2, use_xyz)
    pts = get_points(r, [p1_2], [mn.PINK])
    if anim == 7:
        with Sync(run_time=1.5):
            cam.orbit_around_point(ORIGIN, 140, OUT)
        with Sync():
            circle_thru_points(r, p_1, p1_2, fps=quality.frames_per_second, run_time=2.)
            with Sync(run_time=2.):
                cam.orbit_around_point(ORIGIN, -50, OUT)
        pts[0].spawn()
        return
    with Off():
        cam.orbit_around_point(ORIGIN, 140, OUT)
        circle_thru_points(r, p_1, p1_2, fps=quality.frames_per_second, run_time=.1)
        cam.orbit_around_point(ORIGIN, -50, OUT)
        pts[0].spawn()

    if anim == 8:
        cam.orbit_around_point(ORIGIN, -30, OUT)
        return
    with Off():
        cam.orbit_around_point(ORIGIN, -30, OUT)

    p_r9 = point(point_r9_, use_xyz)
    pts = get_points(r, [p_r9], [mn.ManimColor((.8,1.,.5))])
    c = pts[0][0].get_center()[0,0,:].numpy()
    eq = mn.MathTex(r'R', font_size=30)[0].rotate(-PI/2, mn.RIGHT)
    eq.move_to(c*1.3).rotate(PI,mn.IN)
    rt = 2.5
    with Sync():
        circle_thru_points(r, p_r8, p_r9, fps = quality.frames_per_second, run_time=rt)
        with Sync(run_time=rt):
            cam.orbit_around_point(ORIGIN, 80, OUT)
            cam.orbit_around_point(ORIGIN, 30, cam.get_right_direction())
    with Sync():
        pts[0].spawn()
        ManimMob(eq).spawn()


def cube_curve(quality=LD, bgcol=BLACK, use_xyz=True, anim=1):
    cube_curve2(quality, bgcol, use_xyz, anim)
    Scene.wait(0.1)
    name = 'cube_curve{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)


def surface_plot(quality=LD, bgcol=BLACK, anim=1):
    xrange = [-1.2, 1.2]
    xlen = (xrange[1] - xrange[0])*2
    ax = mn.ThreeDAxes(x_length=xlen, y_length=xlen, z_length=xlen, x_range=xrange, y_range=xrange, z_range=xrange,
                       z_axis_config={'rotation': PI},
                       axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                                    "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                    "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                    },
                       )
    ax.shift(-ax.coords_to_point(0, 0, 0))
    xyz_str = r'xyz'
    eq = mn.MathTex(xyz_str, font_size=30)[0].rotate(-PI/2, mn.RIGHT)
    eq[0].move_to(ax.coords_to_point(1.3,0,0)).rotate(PI,mn.IN)
    eq[1].move_to(ax.coords_to_point(0,1.3,0)).rotate(-PI/2,mn.IN)
    eq[2].move_to(ax.coords_to_point(0,0,1.26)).rotate(-3*PI/4, mn.IN)

    ax = ManimMob(ax)
    cam: Camera = Scene.get_camera()
    eq = Group(*[ManimMob(_) for _ in eq[:]])

    with Off():
        cam.set_distance_to_screen(10).move_to(cam.get_center() * 0.8)
        # cam.set_euler_angles(80*DEGREES, 0*DEGREES, 150*DEGREES)
        # cam.set_euler_angles(60*DEGREES, 0*DEGREES, 150*DEGREES)
        cam.set_euler_angles(80*DEGREES, 0*DEGREES, 120*DEGREES)
        cam.move(OUT*0.32)
        ax.spawn()
        eq.spawn()

    curves = CubicCurve(du=0.01)
    curves = [(-u+v+w, u-v+w, u+v-w) for (u,v,w) in curves]
    normalize_curves(curves)
    plts = []
    surf_col = (BLUE+GREY)*0.5
    surf_col2 = surf_col*0.7
    surf_col2[4] = 0.85
    surf_col[4] = 0.85
    r = 2.25
    theta = [0.]
    v = np.linspace(0, r, 8)
    for x in v[1:-1]:
        theta += [x - 0.01, x, x + 0.01]
    theta += [r]
    m = len(theta)
    theta = torch.tensor(theta)

    pts_arr = []
    for u,v,w in curves:
        pts = (torch.outer(torch.from_numpy(u), RIGHT) + torch.outer(torch.from_numpy(v), UP) + torch.outer(torch.from_numpy(w), OUT))
        pts_arr.append(pts)
        n = pts.shape[0]
        dp = pts[1:, :] - pts[:-1, :]
        seglen = torch.linalg.norm(dp, dim=1)
        u = torch.zeros(n, dtype=pts.dtype, device=pts.device)
        u[1:] = torch.cumsum(seglen, dim=0)

        crv = Surface(grid_height=m, grid_width=n)#.set_shader(basic_pbr_shader)
        # theta[:m-1] = torch.linspace(0., r * 0.9, m-1)
        p = crv.get_descendants()[1]
        p.location[...] = (
                pts[:, None, :] * theta[None, :, None]
        ).reshape(1, -1, 3)
        v = torch.zeros(n, device=pts.device) + 1.
        w = 0.3 / 1.5
        nn = round(u[-1].item() / w)
        width = u[-1].item() / nn
        w = width * 0.3
        for i in range(n-1):
            if u[i] < w <= u[i+1]:
                w += width
                v[i] = 0.
        col = p.color
        col[...] = (
                v[:, None] * surf_col[None, :] + (1-v[:, None]) * surf_col2[None, :]
        ).repeat_interleave(m, dim=0).unsqueeze(0)
        for i in range(2, m, 3):
            col[0, i::m, :] = surf_col2
        plts.append(crv)
    plt = Group(*plts)

    with Off():
        plt.spawn()

    def get_line(pt, scale=1., **kwargs):
        m = 7
        a = torch.linspace(-r, r, m).float()
        normal = LEFT + DOWN + IN
        pt = pt.float()
        lpts = a[:,None] * pt[None,:]
        tgts = pt.unsqueeze(0).expand(m, 3)
        normal1 = normal - torch.dot(normal, pt) / torch.dot(pt, pt) * pt
        normals = normal1.unsqueeze(0).float().expand(m, 3)
        return curve_surface(lpts*scale, normals=normals, tangents=tgts, resolution=8, color=YELLOW, width=0.05, **kwargs)

    if anim == 1:
        ps = [_.get_descendants()[1] for _ in plts]
        locs = [_.location.clone() for _ in ps]
        for frame in ah.FrameStepper(fps=quality.frames_per_second, run_time=1.5, rate_func=rate_funcs.ease_out_exp, step=1):
            r1 = frame.u * r
            with frame.context:
                for p, pts, loc in zip(ps, pts_arr, locs):
                    loc[...] = (
                        pts[:, None, :] * theta[None, :, None].clip(0, r1)
                    ).reshape(1, -1, 3)
                    p.set_non_recursive(location=loc)
    elif anim == 2:
        gp = Group(ax, eq, plt)
        with Sync(rate_func=rate_funcs.identity, run_time=6):
            gp.orbit_around_point(ORIGIN, 360, OUT)
        # following does final slow-down
        # with Off():
        #     gp.orbit_around_point(ORIGIN, 360 * 5.5/6, OUT)
        # def f(t):
        #     return 1. - (t-1.)**2
        # with Sync(rate_func=f, run_time=1):
        #     gp.orbit_around_point(ORIGIN, 360 * 0.5/6, OUT)

    elif anim >= 3:  # draw line
        pts = pts_arr[1]
        with Off():
            plt.set_opacity_via_color(0.8)
        npts = pts.shape[0]

        if anim == 3:
            line1 = get_line(pts[0], 0.)
            line2 = get_line(pts[0])
            with Off():
                line1.spawn()
            with Sync(rate_func=rate_funcs.ease_out_exp):
                line1.become(line2)
        elif anim == 4:
            add_to_scene = True
            for frame in ah.FrameStepper(fps = quality.frames_per_second, step=1, rate_func=rate_funcs.smooth, run_time=2):
                i = (npts-1) * (1-frame.u)
                i0 = min(int(i), npts - 2)
                p = i0 + 1 - i
                q = i - i0
                pt = (pts[i0] * p + pts[i0+1] * q).float()
                line = get_line(pt, add_to_scene=add_to_scene)
                with frame.context:
                    if add_to_scene:
                        line.spawn()
                        desc = line.get_descendants()[1]
                        add_to_scene = False
                    else:
                        desc.set_non_recursive(location=line.get_descendants()[1].location.clone())
        elif 5 <= anim <= 9 or anim >= 14:
            nplane = 10
            z0 = r*0.3
            line = get_line(pts[0])
            pt = pts[0] / pts[0][-1] * -z0
            dot = ManimMob(mn.Dot3D(pt.numpy(), radius=0.08, color=mn.ORANGE))
            if anim <=6:
                with Off():
                    line.spawn()
                    dot.spawn()

            plane = Surface(color=GREY, opacity=0.8, grid_height=nplane)
            p = plane.get_descendants()[1]
            xmax = 2.04
            xplane = torch.linspace(-xmax, xmax, nplane)
            p.location[0,:,0] = xplane.repeat_interleave(nplane)
            p.location[0,:,1] = xplane.repeat(nplane)
            p.location[0,:,2] = -z0

            crvs = []
            for pts1 in pts_arr:
                pts = pts1[:-1].float() * r
                z1 =  0.01-z0
                mask = (pts[:,2] <= z1 * 0.2) & (pts[:,0].abs() < pts[:,2] * -xmax/z0) & (pts[:,1].abs() < pts[:,2] * -xmax/z0) # abs(pts[:,0] / pts[:,2] * -z0) < xmax
                starts = torch.where(mask & ~torch.roll(mask,1))[0]
                ends = torch.where(mask & ~torch.roll(mask,-1))[0]
                for s, e in zip(starts, ends):
                    if s <= e:
                        seg = pts[s:e+1]
                    else:
                        seg = torch.cat([pts[s:], pts[:e+1]], dim=0)
                    seg /= seg[:,2:3] / -z0
                    m = seg.shape[0]
                    crvs.append(curve_surface(seg, normals=IN.unsqueeze(0).expand(m, 3).float(), color=BLUE,width=0.06, closed=False))

            crv = Group(*crvs)

            if anim == 5:
                with Sync(run_time=1.):
                    plane.spawn()
                    crv.spawn()
            elif anim == 6:
                with Off():
                    plane.spawn()
                    crv.spawn()
                with Sync():
                    with Sync(run_time=2.):
                        plt.despawn()
                        eq[2].despawn()
                        ax.submobjects[2].despawn()
                    with Sync(run_time=1.5):
                        line.despawn()
                        dot.despawn()
            elif 11 >= anim >= 7:
                with Off():
                    plane.spawn()
                    crv.spawn()
                    if anim <= 8:
                        plt.despawn()
                        eq[2].despawn()
                        ax.submobjects[2].despawn()
                if anim <= 8:
                    with Sync(run_time=2.) if anim == 7 else Off():
                        cam.orbit_around_line(ORIGIN, cam.get_right_direction(), num_degrees=-80)
                        cam.orbit_around_line(ORIGIN, OUT, num_degrees=-120)
                        cam.move(DOWN*0.2)
                        eq[0].orbit_around_point(eq[0].get_center(), 90, RIGHT)
                        eq[1].orbit_around_point(eq[1].get_center(), 90, DOWN)
                        eq[1].orbit_around_point(eq[1].get_center(), 90, IN)
                        eq[1].move(DOWN*0.275 + RIGHT*0.25)

                if anim >= 8:
                    asymps = []
                    lam = 2 - math.sqrt(3)
                    c = (lam + 1/(lam*lam) - 1/(1+lam))/(1-1/(lam*lam)) * z0
                    dz = 0.05 if anim == 8 else 0
                    for x0,y0,x1,y1 in [(z0/6-xmax, xmax, xmax, z0/6-xmax),
                                        (-xmax, c - lam * xmax, xmax, c + lam * xmax),
                                        (c - lam * xmax, -xmax, c + lam * xmax, xmax)]:
                        dline = ManimMob(mn.DashedLine(mn.IN * (z0 + dz) + x0*mn.RIGHT + y0*mn.UP,
                                                       mn.IN * (z0 + dz) + x1*mn.RIGHT + y1*mn.UP,
                                                       stroke_color=mn.ORANGE, stroke_width=10, dash_length=0.1))
                        # pts1 = torch.stack([OUT * (z0 + 0.05) + x0*RIGHT + y0*UP,
                        #                                OUT * (z0 + 0.05) + x1*RIGHT + y1*UP], dim=0).float()
                        # dline = curve_surface(pts1, color=ORANGE, width=0.06, closed=False)
                        asymps.append(dline)
                    asymps = Group(*asymps)
                    with Off():
                        asymps.spawn()

                    if anim == 8:
                        with Sync(run_time=2):
                            asymps.move(IN*0.05)
                            eq[1].move(DOWN * -0.275 + RIGHT * -0.25)
                            eq[1].orbit_around_point(eq[1].get_center(), -90, IN)
                            eq[1].orbit_around_point(eq[1].get_center(), -90, DOWN)
                            eq[0].orbit_around_point(eq[0].get_center(), -90, RIGHT)
                            cam.move(DOWN * -0.2)
                            cam.orbit_around_line(ORIGIN, OUT, num_degrees=120)
                            cam.orbit_around_line(ORIGIN, cam.get_right_direction(), num_degrees=80)
                    if anim == 9:
                        with Sync(run_time=1.):
                            plane.despawn()
                            crv.despawn()
                            asymps.despawn()
    if anim >= 10:
        r = 2.
        crvs = []
        for pts in pts_arr:
            pts *= r
            crv1 = curve_surface(pts, width=0.03, color=BLUE)
            crvs.append(crv1)
        crvs = Group(*crvs)
        sphere = get_sphere(r)

        if anim == 10:
            sphere.set_opacity_via_color(0.8)
            with Off():
                sphere.spawn()
                crvs.spawn()
            with Sync(run_time=2):
                sphere.set_opacity_via_color(1)
                plt.despawn()
        else:
            with Off():
                sphere.spawn()
                crvs.spawn()
                if anim < 18:
                    plt.despawn()
        if anim > 10:
            theta0 = 0
            theta1 = 360
            run_time=6
            if anim == 11:
                gp = Group(ax, eq, sphere, crvs)
            elif anim == 12:
                pt = pts_arr[0][-180]
                dot1 = ManimMob(mn.Dot3D(pt.numpy(), radius=0.08, color=mn.ORANGE))
                dot2 = ManimMob(mn.Dot3D(-pt.numpy(), radius=0.08, color=mn.ORANGE))
                a = torch.linspace(-1., 1., 7)
                pts = a[:, None] * pt[None, :]
                normals = IN - torch.dot(IN, pt) / torch.dot(pt, pt) * pt
                normals = normals.unsqueeze(0).expand(7, 3)
                line = curve_surface(pts.float(), normals=normals, resolution=8, color=YELLOW, width=0.05)
                with Off():
                    dot1.spawn()
                    dot2.spawn()
                    line.spawn()
                gp = Group(ax, eq, sphere, crvs, dot1, dot2, line)
            elif anim >= 13:
                iline = -195
                nline = pts_arr[0].shape[0] - 1
                pt = pts_arr[0][iline].float()
                print(pts_arr[0].shape)
                a = torch.linspace(-1.5, 1.5, 15)
                pts = a[:, None] * pt[None, :]
                normals = IN - torch.dot(IN, pt) / torch.dot(pt, pt) * pt
                normals = normals.unsqueeze(0).expand(15, 3)
                line = curve_surface(pts.float(), normals=normals, resolution=8, color=YELLOW, width=0.05)
                iline2 = 0
                nline2 = pts_arr[1].shape[0] - 1
                pt2 = pts_arr[1][iline].float()
                print(pts_arr[1].shape)
                pts2 = a[:, None] * pt[None, :]
                normals2 = IN - torch.dot(IN, pt2) / torch.dot(pt2, pt2) * pt2
                normals2 = normals2.unsqueeze(0).expand(15, 3)
                line2 = curve_surface(pts2.float(), normals=normals2, resolution=8, color=YELLOW, width=0.05)
                with Off():
                    line.spawn()
                    line2.spawn()
                if anim == 13:
                    theta0 = 180
                    run_time=3
                    gp = Group(ax, eq, sphere, crvs, line)
                if anim == 14:
                    theta1 = 180
                    run_time=3
                    gp = Group(ax, eq, sphere, crvs, line)
                if 15 <= anim <= 23:
                    col = plane.get_descendants()[1].color
                    col[0,:,:3] = PURPLE[:3] * 0.3 + GREY[:3] * 0.7
                    for p in crv.get_descendants():
                        col = p.color
                    # print(col.shape)
                        col[0, :, 2] = 1.
                        col[0, :, 1] = 0.9
                        col[0, :, 0] = 0.5
                    # col[0, :, :2] = BLUE[:2]
                    with Off():
                        plane.spawn()
                        crv.spawn()
                    gp = Group(ax, eq, sphere, crvs, plane, crv, line)
                    # if anim == 15: theta1 = 180
                    # if anim == 16: theta0 = 180
                    # run_time = 3
                    if 18 <= anim <= 19:
                        with Off():
                            line.despawn()
                        gp = Group(ax, eq, sphere, crvs, plane, crv, plt)
                        # if anim == 18: theta1 = 180
                        # if anim == 19: theta0 = 180
                    if 20 <= anim <= 21:
                        with Off():
                            line.despawn()
                            sphere.despawn()
                            crvs.despawn()
                        gp = Group(ax, eq, plane, crv, plt)
                        # if anim == 20: theta1 = 180
                        # if anim == 21: theta0 = 180
                    if 22 <= anim <= 23:
                        with Off():
                            line.despawn()
                            plt.despawn()
                        gp = Group(ax, eq, sphere, crvs, plane, crv)
                        # if anim == 22: theta1 = 180
                        # if anim == 23: theta0 = 180

            # with Sync(run_time=6, rate_func=rate_funcs.identity):
            if anim == 17:
                gp = Group(ax, eq, sphere, crvs, plane, crv)
                theta0 = 360 * 30 / 6 / 30
                theta1 = theta0 + 360
                run_time=6
                with Off():
                    gp.orbit_around_point(ORIGIN, theta0, OUT)
                p = line.get_descendants()[1]
                p2 = line2.get_descendants()[1]
                with Sync():
                    with Sync(rate_func=rate_funcs.identity, run_time=run_time):
                        gp.orbit_around_point(ORIGIN, theta1 - theta0, OUT)
                    with Seq():
                        for frame in ah.FrameStepper(fps=quality.frames_per_second,run_time=run_time,rate_func=rate_funcs.identity,step=1):
                            u = frame.u
                            theta = theta0 * (1-u) + theta1 * u
                            # u1 = rate_funcs.smooth(u).item()
                            u1 = u
                            v = (iline + 1 - nline * u1) % nline
                            i = int(v)
                            v -= i
                            pt = (pts_arr[0][i] * (1-v) + pts_arr[0][i+1] * v).float()
                            pts = a[:, None] * pt[None, :]
                            normals = IN - torch.dot(IN, pt) / torch.dot(pt, pt) * pt
                            normals = normals.unsqueeze(0).expand(15, 3)
                            line1 = curve_surface(pts.float(), normals=normals, resolution=8, color=YELLOW, width=0.05, add_to_scene=False)
                            line1.orbit_around_point(ORIGIN, theta, OUT)
                            loc = line1.get_descendants()[1].location.clone()

                            v = (iline2 + 13 + nline2 * u1) % nline2
                            i = int(v)
                            v -= i
                            pt = (pts_arr[1][i] * (1-v) + pts_arr[1][i+1] * v).float()
                            pts = a[:, None] * pt[None, :]
                            normals = IN - torch.dot(IN, pt) / torch.dot(pt, pt) * pt
                            normals = normals.unsqueeze(0).expand(15, 3)
                            line1 = curve_surface(pts.float(), normals=normals, resolution=8, color=YELLOW, width=0.05, add_to_scene=False)
                            line1.orbit_around_point(ORIGIN, theta, OUT)
                            loc2 = line1.get_descendants()[1].location.clone()

                            with frame.context:
                                p.set_non_recursive(location=loc)
                                p2.set_non_recursive(location=loc2)

                            # theta_ = theta

            else:
                if theta0 != 0:
                    with Off():
                        gp.orbit_around_point(ORIGIN, theta0, OUT)
                if theta1 != theta0:
                    with Sync(run_time=run_time, rate_func=rate_funcs.identity):
                        gp.orbit_around_point(ORIGIN, (theta1 - theta0), OUT)
                # following does final slow-down
                # with Off():
                #     gp.orbit_around_point(ORIGIN, 360 * 5.5/6, OUT)
                # def f(t):
                #     return 1. - (t-1.)**2
                # with Sync(rate_func=f, run_time=1):
                #     gp.orbit_around_point(ORIGIN, 360 * 0.5/6, OUT)


    if anim > 0:
        Scene.wait(0.1)

    name = 'surface{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)

def curve_head(quality=LD, bgcol=BLACK, anim=1):
    r = 2.

    cam: Camera = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(10).move_to(cam.get_center() * 0.8)
        cam.set_euler_angles(80*DEGREES, 0*DEGREES, 120*DEGREES)
        cam.move(OUT*0.32)

    m = 20
    n = 40
    sphere =Surface(opacity=0.4, color=GREY, grid_height=m, grid_width=n)  # .move_to(origin)

    u = torch.linspace(0, PI, m)
    if anim == 1:
        v = torch.linspace(-PI/2, PI/2, n)
    else:
        v = torch.linspace(PI/2,3*PI/2, n)

    cu = torch.cos(u).repeat(n)[None,:,None] * r
    su = torch.sin(u).repeat(n)[None,:,None] * r
    cv = torch.cos(v).repeat_interleave(m)[None,:,None]
    sv = torch.sin(v).repeat_interleave(m)[None,:,None]

    up = cam.get_upwards_direction()
    right = cam.get_right_direction()
    fwd = cam.get_forward_direction()
    sphere.get_descendants()[1].location = cu * up + cv * su * fwd + sv * su * right

    curves = CubicCurve(du=0.02)
    normalize_curves(curves)

    plts = []
    pts_arr = []
    for u,v,w in curves:
        pts = (torch.outer(torch.from_numpy(u), RIGHT) + torch.outer(torch.from_numpy(v), UP) + torch.outer(torch.from_numpy(w), OUT)).float()
        pts *= r / pts.norm(dim=1, keepdim=True)
        crv = curve_surface(pts, width=0.045, color=line_col)
        plts.append(crv)
        pts_arr.append(pts)

    thetas = torch.linspace(0., 2 * PI, 81)
    for dirs in [[UP, OUT], [RIGHT, OUT], [UP, RIGHT]]:
        pts = (torch.cos(thetas)[:, None] * dirs[0] + torch.sin(thetas)[:, None] * dirs[1]).float() * r
        crv = curve_surface(pts, color=WHITE, opacity=0.3)
        plts.append(crv)

    plt = Group(*plts)

    with Off():
        sphere.spawn()
        plt.spawn()

    descs = [_.get_descendants()[1] for _ in plt[:]]
    locs = [_.location.clone() for _ in descs]
    cols = [_.color.clone() for _ in descs]

    for frame in ah.FrameStepper(run_time=6, fps=quality.frames_per_second, step=1,rate_func=rate_funcs.identity):
        theta = frame.u * 2 * PI
        mat = torch.from_numpy(ah.rotation_matrix(OUT, theta))
        with frame.context:
            for d, loc, col in zip(descs, locs, cols):
                loc1 = loc @ mat
                if anim == 1:
                    mask = (loc1[0,:,:3] @ fwd[0,0,:]) <= 0
                else:
                    mask = (loc1[0, :, :3] @ fwd[0, 0, :]) > 0
                col1 = col.clone()
                col1[0,:,4] = mask
                d.set_non_recursive(location = loc1, color=col1)

    Scene.wait(0.1)
    name = 'curve_head{}'.format(anim)
    render_to_file(name, render_settings=quality, background_color=bgcol)


def get_curve_plot(r, use_xyz=False, sphere_opacity=0.8, use_pos=True):
    xrange = [-1.2, 1.2]
    xlen = (xrange[1] - xrange[0])*r
    ax = mn.ThreeDAxes(x_length=xlen, y_length=xlen, z_length=xlen, x_range=xrange, y_range=xrange, z_range=xrange,
                       z_axis_config={'rotation': PI},
                       axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                                    "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                    "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                    },
                       )
    ax.shift(-ax.coords_to_point(0, 0, 0))

    eq_str = r'xyz' if use_xyz else r'uvw'
    eq = mn.MathTex(eq_str, font_size=30)[0].rotate(-PI/2, mn.RIGHT)
    eq[0].move_to(ax.coords_to_point(1.3,0,0)).rotate(PI,mn.IN)
    eq[1].move_to(ax.coords_to_point(0,1.3,0)).rotate(-PI/2,mn.IN)
    eq[2].move_to(ax.coords_to_point(0,0,1.26)).rotate(-3*PI/4, mn.IN)

    sphere = get_sphere(r, sphere_opacity=sphere_opacity)[0]

    thetas = torch.linspace(0., 2 * PI, 81)
    circ_pts = []
    circs = []
    for dirs in [[UP, OUT], [RIGHT, OUT], [UP, RIGHT]]:
        pts = (torch.cos(thetas)[:, None] * dirs[0] + torch.sin(thetas)[:, None] * dirs[1]).float() * r
        circ_pts.append(pts)
        crv = curve_surface(pts, color=WHITE, opacity=0.3)
        circs.append(crv)
    circs = Group(*circs)

    curves = CubicCurve(du=0.02)
    normalize_curves(curves)

    line_col = Color((.398, .887, 1.))
    plts = []
    pts_arr = []
    for u,v,w in curves:
        pts = (torch.outer(torch.from_numpy(u), RIGHT) + torch.outer(torch.from_numpy(v), UP) + torch.outer(torch.from_numpy(w), OUT)).float()
        pts *= r / pts.norm(dim=1, keepdim=True)
        crv = curve_surface(pts, width=0.03, color=line_col)
        plts.append(crv)
        pts_arr.append(pts)
    plt = Group(*plts)

    cam: Camera = Scene.get_camera()

    with Off():
        cam.set_distance_to_screen(10).move_to(cam.get_center() * 0.8)
        cam.set_euler_angles(80*DEGREES, 0*DEGREES, 120*DEGREES)
        cam.move(OUT*0.32)

    pos_region, pos_col = positive_region(r, curves[1], plt[1])
    if use_pos:
        with Off():
            plt[1].get_descendants()[1].set_non_recursive(color=pos_col)
            plt[2].get_descendants()[1].set_non_recursive(color=pos_col)

    return ax, ManimMob(eq), sphere, circs, circ_pts, plt, pts_arr, curves, pos_region


def get_points_plot(r):
    pts_p = get_points(r, [point_p1, point_p2, point_p3],
                       [mn.YELLOW, mn.YELLOW, mn.YELLOW])
    pts_p = Group(*pts_p)
    labels_p = Group(
        ManimMob(
            mn.MathTex(r'P_1', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT)),
        ManimMob(
            mn.MathTex(r'P_2', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT)),
        ManimMob(
            mn.MathTex(r'P_3', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT)),
    )
    labels_p[0].move_to(pts_p[0][0].get_center() * 1.1 + UP * 0.3 + OUT * 0.3)
    labels_p[1].move_to(pts_p[1][0].get_center() * 1.1 + UP * 0.1 + OUT * 0.25)
    labels_p[2].move_to(pts_p[2][0].get_center() * 1.1 + UP * 0.3 + OUT * 0.)

    pts_o = get_points(r, [point_o1, point_o2, point_o3],
                       [mn.YELLOW, mn.YELLOW, mn.YELLOW])
    pts_o = Group(*pts_o)
    labels_o = Group(
        ManimMob(
            mn.MathTex(r'O_1', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT)),
        ManimMob(
            mn.MathTex(r'O_2', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT)),
        ManimMob(
            mn.MathTex(r'O_3', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT)),
        ManimMob(
            mn.MathTex(r'O_1', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT)),
        ManimMob(
            mn.MathTex(r'O_2', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT)),
        ManimMob(
            mn.MathTex(r'O_3', color=mn.YELLOW, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(150 * PI / 180, mn.OUT)),
    )
    labels_o[0].move_to(pts_o[0][0].get_center() * 1.2 + UP * 0. + OUT * 0.)
    labels_o[1].move_to(pts_o[1][0].get_center() * 1.2 + UP * 0. + OUT * 0.)
    labels_o[2].move_to(pts_o[2][0].get_center() * 1.2 + UP * 0. + OUT * 0.)
    labels_o[3].move_to(pts_o[0][1].get_center() * 1.2 + UP * 0. + OUT * 0.)
    labels_o[4].move_to(pts_o[1][1].get_center() * 1.2 + UP * 0. + OUT * 0.)
    labels_o[5].move_to(pts_o[2][1].get_center() * 1.2 + UP * 0. + OUT * 0.3)

    return pts_p, labels_p, pts_o, labels_o

col_target_pt = mn.ManimColor((.8, 1., .5))


def curve_anim(quality=LD, bgcol=BLACK, sphere_opacity=0.8, anim=1):
    r = 2.
    use_pos = anim == 8 or anim > 9
    ax, eq, sphere, circs, circ_pts, plt, pts_arr, curves, pos_region \
        = get_curve_plot(r, use_xyz=anim < 4, sphere_opacity=sphere_opacity, use_pos=use_pos)

    # pos_col = ManimColor(list(pos_region_color[:3]*256))
    # pos_col2 = 0.8 * mn.GREEN + 0.2 * mn.WHITE
    pos_col2 = mn.YELLOW
    ax2 = ManimMob(ax.copy().set_opacity(0.85).set_color(pos_col2)) if 4 <= anim <= 10 else ax
    ax = ManimMob(ax)

    cam: Camera = Scene.get_camera()

    with Off():
        sphere.spawn()
        plt.spawn()

    with Off():
        ax.spawn()
        eq.spawn()
        circs.spawn()

    if 1 <= anim <= 2:
        mat = scale_mat(1)
        with Off():
            for i, pts in enumerate(pts_arr):
                pts1 = pts @ mat
                pts1 *= r / pts1.norm(dim=1, keepdim=True)
                plt[i].get_descendants()[1].set_non_recursive(
                    location=curve_surface_loc(pts1))

        gp = Group(ax, eq, circs, sphere, plt)
        if anim == 2:
            with Sync(run_time=6, rate_func=rate_funcs.identity):
                gp.orbit_around_point(ORIGIN, 360, OUT)
        else:
            with Off():
                gp.orbit_around_point(ORIGIN, 360 * 5.5/6, OUT)
            def f(t):
                return 1 - (1-t)**2
            with Sync(run_time=1, rate_func=f):
                gp.orbit_around_point(ORIGIN, 360 * 0.5 / 6, OUT)

    if anim == 4:
        with Off():
            ax2.spawn()

        ax_locs = [d.location.clone() for d in ax.get_descendants()]
        ax_norms = [loc.norm(dim=2, keepdim=True) for loc in ax_locs]
        eq_pos = torch.stack([_.get_center()[0,0,:] for _ in eq.submobjects])
        eq_norms = eq_pos.norm(dim=1, keepdim=True)
        eq_locs = [[d.location.clone() for d in sub.get_descendants()] for sub in eq.submobjects]

        for frame in ah.FrameStepper(fps=quality.frames_per_second, step=1, run_time=4):
            u = frame.u
            mat = scale_mat(1-u)
            mat2 = scale_mat(-u)
            with frame.context:
                for i, pts in enumerate(pts_arr):
                    pts1 = pts @ mat
                    pts1 *= r / pts1.norm(dim=1, keepdim=True)
                    plt[i].get_descendants()[1].set_non_recursive(
                        location=curve_surface_loc(pts1))
                for i, pts in enumerate(circ_pts):
                    pts1 = pts @ mat2
                    pts1 *= r / pts1.norm(dim=1, keepdim=True)
                    circs[i].get_descendants()[1].set_non_recursive(
                        location=curve_surface_loc(pts1, width=0.03))
                for d1, loc, norm in zip(ax.get_descendants(), ax_locs, ax_norms):
                    loc1 = loc @ mat
                    d1.set_non_recursive(location=loc1 * norm / loc1.norm(dim=2, keepdim=True))
                loc = eq_pos @ mat
                loc *= eq_norms / loc.norm(dim=1, keepdim=True)

                rot = torch.from_numpy(ah.rotation_matrix(OUT, -PI/2 * (1 - u)))
                for i in range(3):
                    for d1, loc1 in zip(eq.submobjects[i].get_descendants(), eq_locs[i]):
                        loc2 = loc1 - eq_pos[i, :]
                        if i == 0:
                            loc2 = loc2 @ rot
                        d1.set_non_recursive(location=loc2 + loc[i,:])
                for d1, loc, norm in zip(ax2.get_descendants(), ax_locs, ax_norms):
                    loc1 = loc @ mat2
                    d1.set_non_recursive(location=loc1 * norm / loc1.norm(dim=2, keepdim=True))
        Scene.wait(0.1)

    if 5 <= anim <= 10:
        mat2 = scale_mat(-1.)
        ax_desc = ax.get_descendants()
        ax_locs = [_.location.clone() for _ in ax_desc]
        ax_norms = [_.norm(dim=2, keepdim=True) for _ in ax_locs]
        if anim == 5:
            plt_d = [plt[0].get_descendants()[1]]
        elif anim == 6 or anim == 9:
            plt_d = [plt[i].get_descendants()[1] for i in (1,2)]
        else:
            plt_d = []

        with Off():
            ax2.spawn()
            for d in ax2.get_descendants():
                for d1, loc, norm in zip(ax2.get_descendants(), ax_locs, ax_norms):
                    loc1 = loc @ mat2
                    d1.set_non_recursive(location=loc1 * norm / loc1.norm(dim=2, keepdim=True))
                for i, pts in enumerate(circ_pts):
                    pts1 = pts @ mat2
                    pts1 *= r / pts1.norm(dim=1, keepdim=True)
                    circs[i].get_descendants()[1].set_non_recursive(
                        location=curve_surface_loc(pts1, width=0.03))
            for d in plt_d:
                col = d.color.clone()
                col[0, :, :3] = (RED[:3] * 1.5).clamp(0,1)
                d.set_non_recursive(color=col)
        if anim == 5:
            with Sync():
                cam.orbit_around_point(ORIGIN, -90, OUT)
            Scene.wait(0.1)
        elif anim == 6:
            with Sync(run_time=2.5):
                cam.orbit_around_point(ORIGIN, 160, OUT+UP+LEFT)
            Scene.wait(0.1)

    if anim >= 8:
        with Off():
            pos_region.spawn()

    if anim == 10:
        with Sync():
            with Sync(run_time=2):
                ax2.despawn()
            with Seq():
                for frame in ah.FrameStepper(run_time=2, fps=quality.frames_per_second, step=1):
                    u = frame.u
                    mat = scale_mat((u-1)/3 - 2/3 , u-1)
                    with frame.context:
                        for i, pts in enumerate(circ_pts):
                            pts1 = pts @ mat
                            pts1 *= r / pts1.norm(dim=1, keepdim=True)
                            circs[i].get_descendants()[1].set_non_recursive(
                                location=curve_surface_loc(pts1, width=0.03))

    if anim >= 12:
        pts_p, labels_p, pts_o, labels_o = get_points_plot(r)
        with Off():
            pts_p.spawn()

        if anim == 12:
            with Off():
                labels_p.spawn()

        if anim >= 14:
            with Off():
                pts_o[:3].spawn()

            if anim == 14:
                with Off():
                    labels_o[:3].spawn()
                with Seq():
                    cam.orbit_around_point(ORIGIN, 40, cam.get_right_direction())
                    Scene.wait(0.1)
                    with Sync():
                        cam.orbit_around_point(ORIGIN, -40, cam.get_right_direction())
                        cam.orbit_around_point(ORIGIN, -40, OUT)
                    Scene.wait(0.1)
                    cam.orbit_around_point(ORIGIN, 40, OUT)
                    Scene.wait(0.1)

            if anim == 16:
                with Sync():
                    labels_p[:2].spawn()
                with Sync():
                    chord = circle_thru_points(r, point_p1, point_o3_, fps=quality.frames_per_second, run_time=1.5, width=0.04)
                    with Seq():
                        Scene.wait(0.5)
                        with Sync(run_time=1.2):
                            cam.orbit_around_point(ORIGIN, 80, OUT)
                            labels_o[5].spawn()
                Scene.wait(0.1)
                with Sync():
                    cam.orbit_around_point(ORIGIN, -80, OUT)
                    labels_o[5].despawn()
                    labels_p[:2].despawn()
                    chord.despawn()
                Scene.wait(0.1)

            if anim == 17:
                dir1 = cam.get_right_direction()[0,0,:]
                dir3, angle = add_rots([(dir1, 40*PI/180),
                                        (OUT, -40*PI/180)])
                print(dir3)
                with Sync():
                    labels_o[:3].spawn()
                    cam.orbit_around_point(ORIGIN, 40, dir1)
                    Scene.wait(0.1)
                with Sync():
                    chord = circle_thru_points(r, point_o1, point_o3, fps=quality.frames_per_second,
                                               run_time=1.3, width=0.04)
                    with Seq():
                        Scene.wait(0.1)
                        cam.orbit_around_point(ORIGIN, -40, OUT)
                Scene.wait(0.1)
                with Sync():
                    cam.orbit_around_point(ORIGIN, -angle * 180 / PI, torch.from_numpy(dir3))
                    labels_o[:3].despawn()
                Scene.wait(0.1)

            dir1 = LEFT + UP + OUT
            if anim == 18:
                with Sync():
                    labels_o[0].spawn()
                    labels_o[3].spawn()
                    cam.orbit_around_point(ORIGIN, 60, dir1)
                # Scene.wait(0.1)
                # dir2 = cam.get_right_direction()[0,0,:] + cam.get_forward_direction()[0,0,:]*0.5
                # (0,1,-1) x (2,13,13)=(13,-1,-1)
                dir2 = torch.tensor([13.,1.,-1.])
                dir2 /= dir2.norm()
                with Sync():
                    with Seq(run_time=6):
                        circ = circle_thru_points(r, point_o1, (2,13,13), dtheta2=3*PI/2-PI/10, n=160, width=0.035, fps=30)
                    with Seq():
                        Scene.wait(8/15)
                        with Seq(run_time=6, rate_func=rate_funcs.identity):
                            cam.orbit_around_point(ORIGIN,360, dir2)
                Scene.wait(0.1)

            if anim == 19:
                with Off():
                    cam.orbit_around_point(ORIGIN, 60, dir1)
                    labels_p[0].spawn()
                    labels_o[0].spawn()
                dir2 = cam.get_upwards_direction()[0,0,:]
                with Sync():
                    circ = circle_thru_points(r, point_o1, point_p1, width=0.03, fps=quality.frames_per_second)
                    with Seq():
                        Scene.wait(0.2)
                        cam.orbit_around_point(ORIGIN,-60,dir2)
                Scene.wait(0.1)
                dir3, angle = add_rots([(dir1, 60*PI/180), (dir2, -60*PI/180)])
                print(dir3)
                print(angle)
                with Sync():
                    cam.orbit_around_point(ORIGIN, angle*180/PI, torch.from_numpy(dir3))
                    labels_p[0].despawn()
                    labels_o[0].despawn()
                    circ.despawn()
                Scene.wait(0.1)

            if anim >= 20:
                pt_q = get_points(r, [point_q], [mn.ORANGE])[0]
                with Off():
                    pt_q.spawn()

            # if anim == 20:
            #     eq = mn.MathTex(r'Q=(10:3:2)', color=mn.ORANGE, font_size=35, stroke_width=2)
            #     eq = eq_shadow(eq)
            #     label = ManimMob(eq.rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT))
            #     label.move_to(pt_q[0].get_center() * 1.2 + UP * 0.1 + OUT * 0.3)
            #     label.move(label.get_upwards_direction())
            #     with Off():
            #         label.spawn()


    name = 'curve_anim{}'.format(anim)
    if bgcol == TRANSPARENT:
        name += '_tr'
    render_to_file(name, render_settings=quality, background_color=bgcol)

def scale_mat(t, tscale: float=None):
    if tscale is None: tscale = t
    ones = torch.ones(size=(3,3)) / 3
    diag = torch.eye(3)
    rot = torch.tensor([[0,-1,1], [1,0,-1], [-1,1,0]]) / math.sqrt(3)
    e = math.exp(tscale * math.log(2))
    a = e * math.cos(t * math.pi)
    b = e * math.sin(t * math.pi)
    mat = ones * (1-a) + diag * a + rot * b
    mat[2,:] *= -1
    mat[:,2] *= -1
    return mat.float()

line_col2 = WHITE.clone()
line_col2[4] = 0.3

def solution(quality=LD, bgcol=BLACK, sphere_opacity=0.8, anim=1):
    MathTex.set_default(stroke_width=2)
    r = 2.
    ax, eq, sphere, circs, circ_pts, plt, pts_arr, curves, pos_region \
        = get_curve_plot(r, use_xyz=False, sphere_opacity=sphere_opacity, use_pos=True)
    ax = ManimMob(ax)
    pts_p, labels_p, pts_o, labels_o = get_points_plot(r)
    pt_q = get_points(r, [point_q], [mn.ORANGE])[0]
    label_q = ManimMob(
            mn.MathTex(r'Q', color=mn.ORANGE, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT))
    label_q.move_to(pt_q[0].get_center() * 1.1 + UP * 0.25 + OUT * -0.25)
    cam: Camera = Scene.get_camera()

    with Off():
        ax.spawn()
        eq.spawn()
        sphere.spawn()
        circs.spawn()
        pos_region.spawn()
        plt.spawn()
        pts_p[0].spawn()
        # pts_o.spawn()
        pt_q.spawn()
        dir1 = cam.get_right_direction() * 0.15 - cam.get_upwards_direction() * 0.45 + cam.get_forward_direction()*0.2
        labels_p[0].clone().move(dir1).spawn()
        label_q.spawn()

    point_q3 = (3,15,10)
    point_q4 = (715, 14664, -3619)
    point_q7 = (715, -3619, 14664)
    point_q9 = (.304406698, .508332555, 1.563392295)
    point_q15 = (.164522506539145, 13.030716467024711, -.196932807438576)
    point_q17 = (3.0260001984, 10.8730467814, 2.2497220295)
    pt_q3 = get_points(r, [point_q3], [mn.PINK])[0]
    pt_q4 = get_points(r, [point_q4], [mn.PINK])[0]
    pt_q7 =  get_points(r, [point_q7], [mn.PINK])[0]
    pt_q9 =  get_points(r, [point_q9], [mn.PINK])[0]
    pt_q15 =  get_points(r, [point_q15], [mn.PINK])[0]
    pt_q17 =  get_points(r, [point_q17], [mn.PINK])[0]
    pt_target =  get_points(r, [point_r9_], [col_target_pt])[0]

    col_pt3 = ManimColor((mn.PINK.to_rgb() * 1.4).clip(0,1))
    label_q3 = ManimMob(
            mn.MathTex(r'Q_3', color=col_pt3, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT))
    label_q3.move_to(pt_q3[0].get_center() * 1.17 + UP * 0. + OUT * -0.4)
    label_q4 = ManimMob(
            mn.MathTex(r'Q_4', color=col_pt3, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(150 * PI / 180, mn.OUT))
    label_q4.move_to(pt_q4[0].get_center() * 1.1 + UP * 0. + OUT * -0. + RIGHT * 0.4)
    label_q7 = ManimMob(
            mn.MathTex(r'Q_7', color=col_pt3, font_size=35).rotate(-0 * PI / 2, mn.RIGHT).rotate(180 * PI / 180, mn.OUT))
    label_q7.move_to(pt_q7[0].get_center() * 1.1 + UP * 0. + OUT * -0. + RIGHT * -0.3)
    label_q9 = ManimMob(
            mn.MathTex(r'Q_9', color=col_pt3, font_size=35).rotate(-20 * PI / 180, mn.RIGHT).rotate(180 * PI / 180, mn.OUT))
    label_q9.move_to(pt_q9[0].get_center() * 1.1 + UP * 0. + OUT * -0. + RIGHT * -0.3)
    label_q15 = ManimMob(
            mn.MathTex(r'Q_{15}', color=col_pt3, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(150 * PI / 180, mn.OUT))
    label_q15.move_to(pt_q15[0].get_center() * 1.1 + UP * 0. + OUT * 0.3 + RIGHT * -0.2)
    label_q17 = ManimMob(
            mn.MathTex(r'Q_{17}', color=col_pt3, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(150 * PI / 180, mn.OUT))
    label_q17.move_to(pt_q17[0].get_center() * 1.1 + UP * 0. + OUT * 0.3 + RIGHT * 0.4)

    if anim == 2:
        with Sync():
            circle_thru_points(r, point_p1, point_q3, run_time=1., width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.6)
                with Sync():
                    pt_q3.spawn()
                    label_q3.spawn()
        Scene.wait(0.1)

    if anim >= 3:
        with Off():
            pt_q3.spawn()
            circle_thru_points(r, point_p1, point_q3, run_time=0., width=0.03, fps=quality.frames_per_second, color=line_col2)

    if anim == 3:
        with Sync():
            circle_thru_points(r, point_q, point_q4, run_time=1., width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.6)
                with Sync():
                    pt_q4.spawn()
                    label_q4.spawn()
            with Seq():
                Scene.wait(0.2)
                with Sync(run_time=1):
                    cam.orbit_around_point(ORIGIN, 40, OUT)
        Scene.wait(0.1)

    if anim >= 4:
        with Off():
            circle_thru_points(r, point_q, point_q4, run_time=0., width=0.03, fps=quality.frames_per_second, n=61, color=line_col2)
            pt_q4.spawn()

    if anim == 4:
        with Off():
            # label_q4.spawn()
            cam.orbit_around_point(ORIGIN, 40, OUT)
        with Sync():
            # label_q4.despawn()
            label_q3.orbit_around_point(label_q3.get_center(), 45, OUT).move(RIGHT*-0.25).spawn()
            # cam.orbit_around_point(ORIGIN, 40, RIGHT)
        with Sync():
            circ = circle_thru_points(r, point_q3, point_q7, run_time=1.2, width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.2)
                with Sync(run_time=1.2):
                    cam.orbit_around_point(ORIGIN, 80, RIGHT)
            with Seq():
                Scene.wait(0.9)
                with Sync():
                    pt_q7.spawn()
                    label_q7.spawn()
        Scene.wait(0.1)
        with Sync():
            with Sync(run_time=1):
                label_q7.despawn()
                circ.set_opacity_via_color(0.3)
            with Sync(run_time=1.2):
                cam.orbit_around_point(ORIGIN, -80, RIGHT)
        Scene.wait(0.1)

    if anim >= 5:
        with Off():
            pt_q7.spawn()
            circle_thru_points(r, point_q3, point_q7, run_time=0., width=0.03, fps=quality.frames_per_second, color=line_col2)

    if anim == 5:
        with Off():
            cam.orbit_around_point(ORIGIN, 40, OUT)
            label_q3.orbit_around_point(label_q3.get_center(), 45, OUT).move(RIGHT*-0.25).spawn()
            label_q4.spawn()
        with Sync():
            circle_thru_points(r, point_q4, point_q9, run_time=1., width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.4)
                with Sync(run_time=1.):
                    cam.orbit_around_point(ORIGIN, 50, RIGHT)
            with Seq():
                Scene.wait(0.8)
                with Sync():
                    pt_q9.spawn()
                    label_q9.spawn()
            Scene.wait(0.1)

    if anim >= 6:
        with Off():
            pt_q9.spawn()
            circle_thru_points(r, point_q4, point_q9, run_time=0., width=0.03, fps=quality.frames_per_second, color=line_col2)

    if anim == 6:
        with Off():
            cam.orbit_around_point(ORIGIN, 40, OUT)
            cam.orbit_around_point(ORIGIN, 50, RIGHT)
        with Sync():
            label_q7.spawn()
        with Sync():
            circ=circle_thru_points(r, point_q7, point_q15, run_time=1.2, width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.4)
                with Sync(run_time=1.):
                    cam.orbit_around_point(ORIGIN, -50, RIGHT)
            with Seq():
                Scene.wait(0.9)
                with Sync():
                    pt_q15.spawn()
                    label_q15.spawn()
        with Sync():
            cam.orbit_around_point(ORIGIN, 50, RIGHT)
            circ.set_opacity_via_color(0.3)
            label_q15.despawn()
        Scene.wait(0.1)

    if anim >= 7:
        with Off():
            pt_q15.spawn()
            circle_thru_points(r, point_q7, point_q15, run_time=0., width=0.03, fps=quality.frames_per_second, color=line_col2)

    if anim == 7:
        with Off():
            cam.orbit_around_point(ORIGIN, 40, OUT)
            cam.orbit_around_point(ORIGIN, 50, RIGHT)
            label_q7.spawn()
            label_q9.spawn()
        with Sync():
            circle_thru_points(r, point_q7, point_q17, run_time=1., width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.5)
                cam.orbit_around_point(ORIGIN, -50, RIGHT)
            with Seq():
                Scene.wait(0.8)
                with Sync():
                    pt_q17.spawn()
                    label_q17.spawn()
        Scene.wait(0.1)

    if anim >= 8:
        with Off():
            pt_q17.spawn()
            circle_thru_points(r, point_q7, point_q17, run_time=0., width=0.03, fps=quality.frames_per_second, color=line_col2)

    if anim == 8:
        with Off():
            cam.orbit_around_point(ORIGIN, 40, OUT)
            label_q17.spawn()
            label_q15.spawn()
        with Sync():
            circle_thru_points(r, point_q15, point_r9_, run_time=1., width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.3)
                cam.orbit_around_point(ORIGIN, -50, OUT)
            with Seq():
                Scene.wait(0.8)
                with Sync():
                    pt_target.spawn()
        Scene.wait(0.1)

    name = 'solution{}'.format(anim)
    if bgcol == TRANSPARENT:
        name += '_tr'
    render_to_file(name, render_settings=quality, background_color=bgcol)

def manual_solution(quality=LD, bgcol=BLACK, sphere_opacity=0.75, anim=1):
    MathTex.set_default(stroke_width=2)
    r = 2.
    ax, eq, sphere, circs, circ_pts, plt, pts_arr, curves, pos_region \
        = get_curve_plot(r, use_xyz=False, sphere_opacity=sphere_opacity, use_pos=True)
    ax = ManimMob(ax)
    pts_p, labels_p, pts_o, labels_o = get_points_plot(r)
    pt_q = get_points(r, [point_q], [mn.ORANGE])[0]
    label_q = ManimMob(
            mn.MathTex(r'Q', color=mn.ORANGE, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT))
    label_q.move_to(pt_q[0].get_center() * 1.1 + UP * 0.25 + OUT * -0.25)
    cam: Camera = Scene.get_camera()

    with Off():
        ax.spawn()
        eq.spawn()
        sphere.spawn()
        circs.spawn()
        pos_region.spawn()
        plt.spawn()
        pts_p[0].spawn()
        # pts_o.spawn()
        pt_q.spawn()
        dir1 = cam.get_right_direction() * 0.15 - cam.get_upwards_direction() * 0.45 + cam.get_forward_direction()*0.2
        labels_p[0].clone().move(dir1).spawn()
        label_q.spawn()

    point_r1 = (3,15,10)
    point_r2 = (715, 14664, -3619)
    pt_r1 = get_points(r, [point_r1], [mn.PINK])[0]
    pt_r2 = get_points(r, [point_r2], [mn.PINK])[0]
    pt_r4 = get_points(r, [point_r4_], [mn.PINK])[0]
    pt_r8 = get_points(r, [point_r8_], [mn.PINK])[0]
    pt_target =  get_points(r, [point_r9_], [col_target_pt])[0]

    col_pt3 = ManimColor((mn.PINK.to_rgb() * 1.4).clip(0,1))
    label_r1 = ManimMob(
            mn.MathTex(r'R_1', color=col_pt3, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(120 * PI / 180, mn.OUT))
    label_r1.move_to(pt_r1[0].get_center() * 1.17 + UP * 0. + OUT * -0.4)
    label_r2 = ManimMob(
            mn.MathTex(r'R_2', color=col_pt3, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(150 * PI / 180, mn.OUT))
    label_r2.move_to(pt_r2[0].get_center() * 1.1 + UP * 0. + OUT * -0. + RIGHT * 0.4)
    label_r4 = ManimMob(
            mn.MathTex(r'R_3', color=col_pt3, font_size=35).rotate(-0 * PI / 2, mn.RIGHT).rotate(180 * PI / 180, mn.OUT))
    label_r4.move_to(pt_r4[0].get_center() * 1.1 + UP * -0.1 + OUT * -0. + RIGHT * 0.4)
    label_r8 = ManimMob(
            mn.MathTex(r'R_4', color=col_pt3, font_size=35).rotate(-PI / 2, mn.RIGHT).rotate(210 * PI / 180, mn.OUT))
    label_r8.move_to(pt_r8[1].get_center() * 1.1 + UP * 0. + OUT * -0.25 + RIGHT * 0.35)

    if anim == 1:
        with Sync():
            circle_thru_points(r, point_p1, point_r1, run_time=1., width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.6)
                with Sync():
                    pt_r1.spawn()
                    label_r1.spawn()
        Scene.wait(0.1)

    if anim >= 2:
        with Off():
            pt_r1.spawn()
            circle_thru_points(r, point_p1, point_r1, run_time=0., width=0.03, fps=quality.frames_per_second, color=line_col2)

    if anim == 2:
        with Sync():
            circle_thru_points(r, point_q, point_r2, run_time=1., width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.6)
                with Sync():
                    pt_r2.spawn()
                    label_r2.spawn()
            with Seq():
                Scene.wait(0.2)
                with Sync(run_time=1):
                    cam.orbit_around_point(ORIGIN, 40, OUT)
        Scene.wait(0.1)

    if anim >= 3:
        with Off():
            circle_thru_points(r, point_q, point_r2, run_time=0., width=0.03, fps=quality.frames_per_second, n=61, color=line_col2)
            pt_r2.spawn()

    if anim == 3:
        with Off():
            cam.orbit_around_point(ORIGIN, 40, OUT)
            label_r2.spawn()
        # with Sync():
        #     label_q3.orbit_around_point(label_q3.get_center(), 45, OUT).move(RIGHT*-0.25).spawn()
        with Sync():
            circle_thru_points(r, point_r2, point_r4_, run_time=1.2, width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.2)
                with Sync(run_time=1.2):
                    cam.orbit_around_point(ORIGIN, 60, RIGHT)
            with Seq():
                Scene.wait(0.9)
                with Sync():
                    pt_r4.spawn()
                    label_r4.spawn()
        Scene.wait(0.1)

    if anim >= 4:
        with Off():
            pt_r4.spawn()
            circle_thru_points(r, point_r2, point_r4_, run_time=0., width=0.03, fps=quality.frames_per_second, color=line_col2)

    if anim == 4:
        dir2, angle = add_rots([(RIGHT, -60 * PI/180), (OUT, 30 * PI/180)])
        with Off():
            label_r4.spawn()
            cam.orbit_around_point(ORIGIN, 40, OUT)
            cam.orbit_around_point(ORIGIN, 60, RIGHT)
        with Sync():
            circle_thru_points(r, point_r4_, tuple(-_ for _ in point_r8_), run_time=1.2, width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.2)
                with Sync(run_time=1.2):
                    cam.orbit_around_point(ORIGIN, -angle * 180 / PI, torch.from_numpy(dir2))
            with Seq():
                Scene.wait(0.9)
                with Sync():
                    pt_r8.spawn()
                    label_r8.spawn()
        Scene.wait(0.1)

    if anim >= 5:
        with Off():
            pt_r8.spawn()
            circle_thru_points(r, point_r4_, tuple(-_ for _ in point_r8_), run_time=0., width=0.03, fps=quality.frames_per_second, color=line_col2)

    if anim == 5:
        dir3 = RIGHT * 0.13 + UP * -0.1
        with Off():
            label_r1.orbit_around_point(label_r1.get_center(), 30, OUT).move(dir3).spawn()
            label_r8.spawn()
            cam.orbit_around_point(ORIGIN, 70, OUT)
        with Sync():
            circle_thru_points(r, tuple(-_ for _ in point_r8_), point_r9_, run_time=1.2, width=0.03, fps=quality.frames_per_second, n=61)
            with Seq():
                Scene.wait(0.4)
                with Sync(run_time=1.2):
                    cam.orbit_around_point(ORIGIN, -80, OUT)
            with Seq():
                Scene.wait(0.9)
                with Sync():
                    pt_target.spawn()
        Scene.wait(0.1)

    name = 'manual_solution{}'.format(anim)
    if bgcol == TRANSPARENT:
        name += '_tr'
    render_to_file(name, render_settings=quality, background_color=bgcol)



if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    # cube_curve(quality=HD, use_xyz=True, anim=1)
    # cube_curve(quality=HD, use_xyz=True, anim=2)
    # for anim in range(1, 9):
    # cube_curve(quality=HD, use_xyz=True, anim=9)
    # surface_plot(quality=HD, bgcol=TRANSPARENT, anim=2)
    # surface_plot(quality=HD, bgcol=BLACK, anim=16)
    # surface_plot(quality=LD, bgcol=BLACK, anim=23)
    # curve_head(quality=MD, bgcol=TRANSPARENT, anim=1)
    # curve_head(quality=MD, bgcol=TRANSPARENT, anim=2)
    # curve_anim(quality=HD, bgcol=BLACK, anim=16)
    # curve_anim(quality=HD, bgcol=TRANSPARENT, anim=15, sphere_opacity=0.65)
    # solution(quality=HD, bgcol=TRANSPARENT, anim=2, sphere_opacity=0.75)
    manual_solution(quality=HD, bgcol=TRANSPARENT, anim=5, sphere_opacity=0.75)
    manual_solution(quality=HD, bgcol=TRANSPARENT, anim=5, sphere_opacity=0.75)
    manual_solution(quality=HD, bgcol=TRANSPARENT, anim=5, sphere_opacity=0.75)
    manual_solution(quality=HD, bgcol=TRANSPARENT, anim=5, sphere_opacity=0.75)
    manual_solution(quality=HD, bgcol=TRANSPARENT, anim=5, sphere_opacity=0.75)
