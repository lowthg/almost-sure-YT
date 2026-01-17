import manim
import numpy as np
import torch
from algan import *
import manim as mn
import math
import functorch
import scipy as sp
import colorsys

from manim import VGroup
sys.path.append('../')
import alganhelper as ah

LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)
HD2 = RenderSettings((1920, 1080), 15)
dist = 15.366

def mnpos(x, y, z=0.):
    return x * mn.RIGHT + y * mn.UP + z * mn.IN

def fade_dist(grid_dots, grid_desc, col, z_dir, z0, fade_rate=0.25, line_op=1.):
    for dot in grid_dots:
        h = torch.inner(dot.get_center(), z_dir).item()
        op = max(1. - max(h - z0, 0.) * fade_rate, 0.)
        dot.set_opacity(op)
    h0 = torch.inner(grid_desc.location, z_dir[0])
    op = ((1+z0*fade_rate) - h0 * 0.25).clamp(0,1) * line_op
    col2 = col.clone()
    col2[:, :, 4] *= op[:,:,0]
    grid_desc.set_non_recursive(color=col2)

    # for i, line in enumerate(grid_lines):
    #     loc = line.get_center()
    #     op = (1 + z0*fade_rate - torch.inner(loc, z_dir) * fade_rate).clamp(0, 1) * line_op
    #     line.set_opacity(op)


def create_mesh(mesh_col: Color=GREY, dot_col=mn.RED, x_range=(-15,14), y_range=(-11, 14)):
    h = 0.6
    nx = 1 + x_range[1] - x_range[0]
    ny = 1 + y_range[1] - y_range[0]
    x_points = np.linspace(x_range[0] * h, x_range[1] * h, nx)
    y_points = np.linspace(y_range[0] * h, y_range[1] * h, ny)

    line_width=4
    dot1 = ManimMob(mn.Dot3D(radius=0.07, color=dot_col, resolution=(2, 2), stroke_opacity=0, stroke_width=0))
    grid_dots = []
    grid_lines = []
    grid_lines2 = []
    grid_indices = []
    for x in x_points:
        for y in y_points:
            grid_dots.append(dot1.clone().move_to(x*RIGHT+y*UP))
            grid_indices.append((int(round(x/h)), int(round(y/h))))
    # for x in x_points:
    #     y0 = y_points[0]
    #     for i in range(1, ny):
    #         y1 = y_points[i]
    #         grid_lines.append(ManimMob(mn.Line(mnpos(x,y0, -0.01), mnpos(x, y1, -0.01), stroke_width=line_width,
    #                                   stroke_color=line_col, stroke_opacity=line_op)))
    #         y0 = y1
    # for y in y_points:
    #     x0 = x_points[0]
    #     for i in range(1, nx):
    #         x1 = x_points[i]
    #         grid_lines2.append(ManimMob(mn.Line(mnpos(x0,y, -0.01), mnpos(x1, y, -0.01), stroke_width=line_width,
    #                                   stroke_color=line_col, stroke_opacity=line_op)))
    #         x0 = x1
    if mesh_col is None:
        mesh_col = GREY.clone()
        mesh_col[4] = 0.6
    fill_col = BLUE.clone()
    fill_col[4] = 0.2

    surf = surface_func(nx=nx-1, ny=ny-1, mesh_m=32, mesh_n=32, mesh_col=mesh_col, fill_color=fill_col)
    surf.scale(np.array([(nx-1)*h/2, (ny-1)*h/2, 1]))
    surf.move((x_range[0] + x_range[1])*h/2*RIGHT + (y_range[0] + y_range[1])*h/2*UP)

    grid_dots = Group(*grid_dots)
    #grid_lines = Group(*grid_lines, *grid_lines2)

    return grid_dots, surf, grid_indices

def mesh_points(quality=LD, bgcol=BLACK, show_dots=True, show_eqs=True):
    dot_col = mn.ManimColor(mn.RED)
    grid_dots, _, grid_indices = create_mesh(dot_col=dot_col, y_range=(-9, 14))

    with Off():
        cam = Scene.get_camera()
        cam.set_distance_to_screen(10)

    if show_dots:
        with Off():
            grid_dots.spawn()

    Scene.wait(0.5)
    i0 = grid_indices.index((0, 0))
    i1 = grid_indices.index((1, 0))
    i2 = grid_indices.index((0, 1))

    p0 = grid_dots[i0].get_center().numpy()[0]
    p1 = grid_dots[i1].get_center().numpy()[0]
    dx = p1 - p0
    dy = grid_dots[i2].get_center().numpy()[0] - p0
    p0 += DOWN.numpy() * 0.2

    eqs = []
    if show_eqs:
        for i in range(-8, 9):
            for j in range(-4, 5):
                eqs.append(ManimMob(mn.MathTex(r'({}, {})'.format(i, j)).scale(0.3).move_to(p0 + i * dx + j * dy)))

    gp = Group(*eqs)
    if show_eqs:
        with Sync():
            gp.spawn()

    col_mid = BLUE
    col_next = GREEN

    Scene.wait(0.5)
    p_cam = cam.get_center().numpy()[0]
    cam_shift = 0.6 * (p1 - p_cam)
    cam.move(cam_shift)
    Scene.wait(0.1)
    with Sync():
        if show_dots:
            grid_dots[i1].set(color=col_mid)
        if show_eqs:
            eqs[9 * 9 + 4].set(color=col_mid)
    Scene.wait(0.1)
    j1 = grid_indices.index((2, 0))
    j2 = grid_indices.index((0, 0))
    j3 = grid_indices.index((1, 1))
    j4 = grid_indices.index((1, -1))
    gp2 = Group(grid_dots[j1], grid_dots[j2], grid_dots[j3], grid_dots[j4])
    p0 = p1 + IN.numpy() * 0.1
    lines1 = Group(*[
        Line(p0, p0 + d, border_width=8, border_color=col_next) for d in [dx, -dx, dy, -dy]
    ])
    lines2 = Group(*[
        Line(p0, p0, border_width=8, border_color=col_next) for _ in range(4)
    ])
    if show_dots:
        with Off():
            lines2.spawn()
    with Sync():
        if show_eqs:
            eqs[10 * 9 + 4].set(color=col_next)
            eqs[8 * 9 + 4].set(color=col_next)
            eqs[9 * 9 + 5].set(color=col_next)
            eqs[9 * 9 + 3].set(color=col_next)
        if show_dots:
            gp2.set(color=col_next)
            lines2.become(lines1)

    Scene.wait(0.5)

    name = 'mesh_points'
    if show_dots:
        name += '_dot'
    if show_eqs:
        name += '_eq'

    render_to_file(name, render_settings=quality, background_color=bgcol)

def mesh_points2(quality=LD, bgcol=BLACK, show_dots=True, show_eqs=True, part=0):
    dot_col = mn.ManimColor(mn.RED)
    line_op=0.6
    grid_dots, grid_lines, grid_indices = create_mesh(dot_col=dot_col,
                                                      y_range=(-9, 14), x_range=(-15, 14))

    i0 = grid_indices.index((0, 0))
    i1 = grid_indices.index((1, 0))
    i2 = grid_indices.index((0, 1))

    p0 = grid_dots[i0].get_center().numpy()[0]
    p1 = grid_dots[i1].get_center().numpy()[0]
    dx = p1 - p0
    dy = grid_dots[i2].get_center().numpy()[0] - p0
    p0 += DOWN.numpy() * 0.2

    eqs = []
    if show_eqs:
        for i in range(-8, 9):
            for j in range(-4, 5):
                eqs.append(ManimMob(mn.MathTex(r'({}, {})'.format(i, j)).scale(0.3).move_to(p0 + i * dx + j * dy)))

    col_mid = BLUE
    col_next = GREEN

    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(10)
    p_cam = cam.get_center().numpy()[0]
    cam_shift = 0.6 * (p1 - p_cam)
    gp = Group(*eqs)
    j1 = grid_indices.index((2, 0))
    j2 = grid_indices.index((0, 0))
    j3 = grid_indices.index((1, 1))
    j4 = grid_indices.index((1, -1))
    gp2 = Group(grid_dots[j1], grid_dots[j2], grid_dots[j3], grid_dots[j4])
    p0 = p1 + IN.numpy() * 0.1
    lines1 = Group(*[
        Line(p0, p0 + d, border_width=8, border_color=col_next) for d in [dx, -dx, dy, -dy]
    ])

    with Off():
        if show_dots:
            grid_dots.spawn()
            grid_dots[i1].set(color=col_mid)
            lines1.spawn()
            gp2.set(color=col_next)
        if show_eqs:
            gp.spawn()
            eqs[9 * 9 + 4].set(color=col_mid)
            eqs[10 * 9 + 4].set(color=col_next)
            eqs[8 * 9 + 4].set(color=col_next)
            eqs[9 * 9 + 5].set(color=col_next)
            eqs[9 * 9 + 3].set(color=col_next)

    with Off():
        cam.move(cam_shift)
    #Scene.wait(0.5)

    M = ah.rotation_matrix(RIGHT, -60*DEGREES_TO_RADIANS)
    N = ah.rotation_matrix(RIGHT, -120*DEGREES_TO_RADIANS)
    pt = IN + np.dot(M, (OUT*2+DOWN).numpy()) - np.dot(N, cam_shift[0])


    with Off():
        grid_lines.spawn()
        p = grid_lines.get_descendants()[1]
        col = p.color.clone()
        p.color[:,:,4] = 0
    #    col0 = col.clone()
    #    col0[:,:,4] = 0


    run_time=1.
    with Sync(run_time=run_time):
        if show_dots:
            for j in [j1, j2, j3, j4]:
                grid_dots[j].set(color=RED)
            grid_dots[i1].set(color=RED)
            lines1.despawn()
        if show_eqs:
            gp.despawn()
        if show_dots:
            with Seq():
                fps = quality.frames_per_second
                t = run_time
                n = round(fps * t)
                n0 = 0
                n1 = n
                dt = 1. / fps
                print('n =', n)
                if part == 1:
                    n1 = int(n * 0.35) + 1
                elif part == 2:
                    n0 = int(n * 0.35)
                    n1 = int(n * 0.7) + 1
                elif part == 3:
                    n0 = int(n * 0.7)
                if n0 > 0:
                    with Sync(run_time=n0 * dt):
                        Scene.wait(n0 * dt)

                u0 = 0.
                for i in range(n0, n1):
                    print('rot1', i)
                    u = mn.smooth((i + 1) / n)
                    du = u - u0
                    with Sync(rate_func=rate_funcs.identity, run_time=dt):
                        cam.orbit_around_point(pt, 60 * du, RIGHT).move(cam_shift*LEFT.numpy()*du)
                        z_dir = cam.get_forward_direction()
                        z0 = torch.inner(cam.get_center(), z_dir).item() + dist
                        fade_dist(grid_dots, p, col, z_dir, z0, line_op=line_op * u)
                    u0 = u

        #cam.move_to(p_cam)
        else:
            cam.orbit_around_point(pt, 60, RIGHT)
            cam.move(cam_shift*LEFT.numpy())

    Scene.wait(0.5)

    name = 'mesh_points2'
    if show_dots:
        name += '_dot'
    if show_eqs:
        name += '_eq'
    if part != 0:
        name += '_{}'.format(part)

    render_to_file(name, render_settings=quality, background_color=bgcol)

def mesh_points3(quality=LD, bgcol=BLACK, frame0 = 0, frame1=30):
    line_col=mn.GREY
    dot_col = mn.ManimColor(mn.RED)
    line_op=0.6
    grid_dots, grid_lines, grid_indices = create_mesh(line_col=line_col, dot_col=dot_col, line_op=line_op,
                                                      y_range=(-11, 14), x_range=(-17, 14))

    M = ah.rotation_matrix(RIGHT, -60 * DEGREES_TO_RADIANS)
    pt = IN + np.dot(M, (OUT * 2 + DOWN).numpy())
    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(10)
        cam.orbit_around_point(pt, 60, RIGHT)
        z_dir = cam.get_forward_direction()
        z0 = torch.inner(cam.get_center(), z_dir).item() + dist
        grid_dots.spawn()
        grid_lines.spawn()
        fade_dist(grid_dots, grid_lines, z_dir, z0, line_op=line_op)

    #cam.orbit_around_point(ORIGIN, -30, IN)
    #cam.orbit_around_point(ORIGIN, 30, IN)

    run_time=1.
    fps = quality.frames_per_second
    t = run_time
    n = round(fps * t)
    dt = 1. / fps
    print('n =', n)
    u0 = 0.
    for i in range(frame0, frame1):
        u = mn.smooth((i + 1) / n)
        du = u - u0
        print('rot1', i, du)
        with Sync(run_time=dt):
            cam.orbit_around_point(ORIGIN, -30*du, IN)
            z_dir = cam.get_forward_direction()
            z0 = torch.inner(cam.get_center(), z_dir).item() + dist
            fade_dist(grid_dots, grid_lines, z_dir, z0, line_op=line_op)
        u0 = u

    Scene.wait(0.5)

    name = 'mesh_points3_{}_{}'.format(frame0, frame1)

    render_to_file(name, render_settings=quality, background_color=bgcol)

def setup_mesh(x_range=(-17, 14), y_range=(-11, 14)):
    line_col=mn.GREY
    dot_col = mn.ManimColor(mn.RED)
    line_op=0.6
    grid_dots, grid_lines, grid_indices = create_mesh(dot_col=dot_col,
                                                      y_range=y_range, x_range=x_range)

    M = ah.rotation_matrix(RIGHT, -60 * DEGREES_TO_RADIANS)
    pt = IN + np.dot(M, (OUT * 2 + DOWN).numpy())
    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(10)
        cam.orbit_around_point(pt, 60, RIGHT)
        cam.orbit_around_point(ORIGIN, -30, IN)
        z_dir = cam.get_forward_direction()
        z0 = torch.inner(cam.get_center(), z_dir).item() + dist
        grid_lines.spawn()
        grid_dots.spawn()
        #fade_dist(grid_dots, grid_lines, z_dir, z0, line_op=line_op)
        p = grid_lines.get_descendants()[1]
        col = p.color
        fade_dist(grid_dots, p, col, z_dir, z0, line_op=line_op)

    return grid_dots, grid_lines, grid_indices

def mesh_points4(quality=LD, bgcol=BLACK):
    _, _, _ = setup_mesh()

    render_to_file('mesh_points4', render_settings=quality, background_color=bgcol)

_fill_color = GREEN.clone()
_fill_color[4] = 0.3

def surface_func(mesh_col=DARK_BROWN,
                 nx=11, ny=11,
                 mesh_m=32,
                 mesh_n=32,
                 fill_color=_fill_color
):
    m = mesh_m * ny - 1
    n = mesh_n * nx - 1
    t = torch.zeros(m, n, 5)
    for i1 in range(m):
        mesh_on1 = (i1+1) % mesh_m == 0
        for j1 in range(n):
            t[i1, j1, :] =  fill_color
            mesh_on = mesh_on1 or (j1+1) % mesh_n == 0
            if mesh_on:
                t[i1, j1, :] = mesh_col

    mob = ImageMob(t)
    return mob

def mesh_func1(quality=LD, bgcol=BLACK):
    h = 0.6
    x_range = (-17, 14)
    y_range = (-11, 14)
    grid_dots, grid_lines, grid_indices = setup_mesh()#x_range=(-2,2), y_range=(-2,2))
    zscale = 0.04

    def f(x, y):
        return (x*x-y*y)*zscale

    func_dots0 = []

    print('1')
    with Off():
        for index, dot in zip(grid_indices, grid_dots):
            func_dots0.append(dot.clone())
        print('2')
        func_dots = Group(*func_dots0)
        print('3')

    x_range = (-17, 14)
    y_range = (-11, 14)
    nx = x_range[1] - x_range[0]
    ny = y_range[1] - y_range[0]
    surf = surface_func(nx=x_range[1] - x_range[0], ny=y_range[1]-y_range[0],
                        mesh_m=32, mesh_n=32)
    #surf0 = surface_func(nx=x_range[1] - x_range[0], ny=y_range[1]-y_range[0],
    #                    mesh_m=32, mesh_n=32)
    surf.scale(np.array([nx*h/2, ny*h/2, 1]))
    #surf0.scale(np.array([nx*h/2, ny*h/2, 1]))
    surf.move((x_range[0] + x_range[1])*h/2*RIGHT + (y_range[0] + y_range[1])*h/2*UP)
    #surf0.move((x_range[0] + x_range[1])*h/2*RIGHT + (y_range[0] + y_range[1])*h/2*UP)
    with Off():
        grid_dots.despawn()
        #grid_lines.despawn()

    p = surf.get_descendants()[1]
    loc = p.location.clone()
    col = p.color.clone()
    cam = Scene.get_camera()
    z_dir = cam.get_forward_direction()
    z0 = torch.inner(cam.get_center(), z_dir).item() + dist
    fade_rate=0.25
    h0 = torch.inner(loc, z_dir[0])

    op = ((1+z0*fade_rate) - h0 * 0.25).clamp(0,1)
    col[:, :, 4] *= op[:,:,0]
    xvals = loc[:, :, 0] / h
    yvals = loc[:, :, 1] / h
    zvals = (xvals * xvals - yvals * yvals) * zscale
    with Off():
        p.set_non_recursive(color=col)
        func_dots.spawn()
        surf.spawn()

    Scene.wait(0.5)

    with Sync():
        loc[:, :, 2] = zvals
        p.set_non_recursive(location=loc)
        for index, dot in zip(grid_indices, func_dots):
            dot.move(f(*index)*IN)

    Scene.wait(0.5)

    with Sync():
        loc[:, :, 2] = -zvals
        p.set_non_recursive(location=loc)
        for index, dot in zip(grid_indices, func_dots):
            dot.move(-2*f(*index)*IN)

    print('4')

    render_to_file('mesh_func1', render_settings=quality, background_color=bgcol)


if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    COMPUTING_DEFAULTS.max_animate_batch_size = 8
    bgcol = Color('#202020')
    #mesh_points(quality=LD, bgcol=bgcol, show_eqs=True)
    #mesh_points2(quality=HD, bgcol=bgcol, part=1)
    #mesh_points2(quality=HD, bgcol=bgcol, part=2)
    #mesh_points2(quality=HD, bgcol=bgcol, part=3)
    #mesh_points3(quality=HD, bgcol=bgcol, frame0=0, frame1=11)
    #mesh_points3(quality=HD, bgcol=bgcol, frame0=10, frame1=21)
    #mesh_points3(quality=HD, bgcol=bgcol, frame0=20, frame1=30)
    #mesh_points4(quality=HD, bgcol=bgcol)
    mesh_func1(quality=HD, bgcol=bgcol)
    #mesh2d(quality=LD, bgcol=bgcol)
