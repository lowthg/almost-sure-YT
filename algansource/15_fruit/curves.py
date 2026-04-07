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
LD = RenderSettings((854, 480), 15)
LD2 = RenderSettings((854, 480), 30)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 15)


import numpy as np
import math

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

    for (u, v, w) in res:
        r = np.sqrt(u*u + v*v + w*w)
        u /= r
        v /= r
        w /= r

    res += [tuple(-u for u in res[-1])]

    return res

point_r1 = (10,3,2)
point_r2 = (715, 14664, -3619)
point_r4 = (164522506539145, -196932807438576, 13030716467024711)
point_r8 = (1710959400725563597554818136978688404216526725221349245104503985,
            -2454018952485961198258264459149041098107906103325177114823668256,
            -56542945251126914281095352993207832740799500248513925316036689)
point_r1_2 = (3, 15, 10)
point_r9 = (191351933902876166269149126585145312311157305638537402696309772369890426669333578,
            41248744472058697085059064167596864978309517527133584614904575465015648101827615,
            158850414786674863699812567622291227638480863227727045318648072891631625503050035)



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

def curve_surface(pts, normals=None, tangents=None, resolution=8, color=BLUE, width=0.03):
    if normals is None:
        normals = pts
    if tangents is None:
        tangents = torch.roll(pts, shifts=1, dims=0) - torch.roll(pts, shifts=-1, dims=0)
    du = torch.nn.functional.normalize(normals, dim=1)
    dv = torch.nn.functional.normalize(torch.linalg.cross(normals, tangents), dim=1)
    theta = torch.linspace(-math.pi, math.pi, resolution, device=pts.device)
    n = pts.shape[0]
    crv = Surface(grid_height=resolution, grid_width=n, color=color)
    crv.get_descendants()[1].location[...] = (
                pts[:, None, :] +
                (torch.sin(theta)[None, :, None] * du[:, None, :] +
                 torch.cos(theta)[None, :, None] * dv[:, None, :]) * (width/2)
        ).reshape(1, -1, 3)

    return crv


def positive_region(r, curve, curve_obj):
    line_col2 = mn.ManimColor.from_rgb((.6787, 1., .5337))
    dots = []
    for u, v, w in [(0,1,1), (1,0,1), (1,1,0)]:
        for pt in [(u, v, w), (-u, -v, -w)]:
            s = r / math.sqrt(sum([_ * _ for _ in pt]))
            p = (pt[0] * mn.RIGHT + pt[1] * mn.UP + pt[2] * mn.IN)*s
            dots.append(ManimMob(mn.Dot3D(p, radius=0.06, color=line_col2)))

    pos_region = Surface(grid_height=10, grid_width=10, color=GREEN, opacity=0.6)
    pos_region2 = Surface(grid_height=10, grid_width=10, color=GREEN, opacity=0.6)
    loc = pos_region.get_descendants()[1].location

    pts = [OUT+RIGHT, OUT+UP, RIGHT+UP]
    for i in range(10):
        u = i/9
        for j in range(10):
            v = j/9
            loc[0,i*10+j,:] = (pts[0] * (1-u) + pts[1] * u) * v + pts[2] * (1-v)

    loc[...] = torch.nn.functional.normalize(loc, dim=2) * r * 1.004
    pos_region2.get_descendants()[1].location[...] = -loc

    line_col2 = Color(line_col2.to_rgb())
    p = curve_obj.get_descendants()[1]
    col = p.color.clone()
    u, v, w = curve
    eps = 0.002
    mask = (u < v + w + eps) & (v < u + w + eps) & (w < u + v + eps)
    mask_expanded = torch.from_numpy(mask).repeat_interleave(8)
    col[0, mask_expanded, :3] = line_col2[:3]


    return Group(*dots), Group(pos_region, pos_region2), col

def get_points(r, pts, colors, width=0.12):
    res = []
    for i, (u, v, w) in enumerate(pts):
        col = colors[i % len(pts)]
        s = math.sqrt(u*u+v*v+w*w)
        p = (u * mn.RIGHT + v * mn.UP + w * mn.IN) * (r / s)
        res.append(Group(
            ManimMob(mn.Dot3D(p, radius=width/2, color=col)),
            ManimMob(mn.Dot3D(-p, radius=width/2, color=col))
        ))
    return res

def get_sphere(r):
    ball = Sphere(radius=r, opacity=0.8, color=GREY, grid_height=40)#.move_to(origin)
    circ = Circle(radius=r, border_width=3, border_color=WHITE, filled=False, opacity=0.5)
    circ2 = circ.clone().orbit_around_point(ORIGIN, 90, RIGHT)
    circ3 = circ.clone().orbit_around_point(ORIGIN, 90, UP)
    return Group(ball, circ, circ2, circ3)

def circle_thru_points(r, pt0, pt1):
    p0 = np.array([*pt0], dtype=float)
    p0 /= np.linalg.norm(p0)
    p1 = np.array([*pt1], dtype=float)
    p1 /= np.linalg.norm(p1)
    p2 = p1 - np.dot(p0, p1) * p0
    p2 /= np.linalg.norm(p2)
    theta1 = math.acos(np.dot(p0, p1))
    theta = torch.linspace(0., theta1, 20)

    pts = torch.outer(torch.cos(theta), torch.from_numpy(p0)) + torch.outer(torch.sin(theta), torch.from_numpy(p2))



    # p2 = np.cross(p0, p1)
    # p2 /= np.linalg.norm(p2)
    # theta  = math.acos(p2[2])
    # # v = pyttsx3(p2[1], -p2[0])
    circ = Circle(radius=r, color=WHITE, filled=False, border_width=3).orbit_around_point(ORIGIN, theta*RADIANS, p2[0] * UP - p2[1] * RIGHT)
    return circ

def cube_curve(quality=LD, bgcol=BLACK):
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
    eq = mn.MathTex(r'uvw', font_size=30)[0].rotate(-PI/2, mn.RIGHT)
    eq[0].move_to(ax.coords_to_point(1.3,0,0)).rotate(PI,mn.IN)
    eq[1].move_to(ax.coords_to_point(0,1.3,0)).rotate(-PI/2,mn.IN)
    eq[2].move_to(ax.coords_to_point(0,0,1.3)).rotate(-3*PI/4, mn.IN)

    sphere = get_sphere(r)

    curves = CubicCurve(du=0.02)
    line_col = Color((.398, .887, 1.))
    plts = []
    for u,v,w in curves:
        pts = (torch.outer(torch.from_numpy(u), RIGHT) + torch.outer(torch.from_numpy(v), UP) + torch.outer(torch.from_numpy(w), OUT)) * r
        crv = curve_surface(pts, width=0.03, color=line_col)
        plts.append(crv)



    dots = []
    pts = get_points(r, [(1,0,0), (0,1,0), (0,0,1), (0,1,-1), (1,0,-1), (1,-1,0)],
                     [mn.YELLOW] * 3 + [mn.ORANGE] * 3)
    dots.append(Group(*pts))

    pts = get_points(r, [point_r1, point_r2], [mn.RED, mn.PINK])
    dots.append(Group(*pts))

    dots = Group(*dots)

    ax = ManimMob(ax)
    cam = Scene.get_camera()
    eq = ManimMob(eq)
    plt = Group(*plts)

    with Off():
        cam.set_distance_to_screen(10).move_to(cam.get_center() * 0.8)
        # cam.set_euler_angles(60*DEGREES, 0*DEGREES, 120*DEGREES)
        cam.set_euler_angles(80*DEGREES, 0*DEGREES, 150*DEGREES)
        cam.move(OUT*0.32)
        ax.spawn()
        sphere.spawn()
        eq.spawn()
        plt.spawn()
        dots.spawn()

    # with Sync(run_time=16, rate_func=rate_funcs.identity):
    #     cam.orbit_around_point(ORIGIN, 360, IN)

    pos_dots, pos_region, pos_col = positive_region(r, curves[1], plt[1])

    # with Sync():
    #     pos_dots.spawn()
    # with Sync():
    #     pos_dots.despawn()
    with Off():
        pos_region.spawn()
        plt[1].get_descendants()[1].set_non_recursive(color=pos_col)
        plt[2].get_descendants()[1].set_non_recursive(color=pos_col)

    circ = circle_thru_points(r, point_r1, point_r2)

    with Off():
        circ.spawn()

    name = 'cube_curve'
    render_to_file(name, render_settings=quality, background_color=bgcol)

if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    cube_curve(quality=HD)