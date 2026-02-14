import manim
import numpy as np
import torch
from algan import *
import manim as mn
import math
import functorch
import scipy as sp
import colorsys
from algan.rendering.post_processing.bloom import bloom_filter, bloom_filter_premultiply
from functools import partial

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

def create_mesh(mesh_col: Color=GREY, dot_col=mn.RED, x_range=(-15,14), y_range=(-11, 14), h=0.6,
                fill_col: Color=None, resolution=(2,2)):
    nx = 1 + x_range[1] - x_range[0]
    ny = 1 + y_range[1] - y_range[0]
    x_points = np.linspace(x_range[0] * h, x_range[1] * h, nx)
    y_points = np.linspace(y_range[0] * h, y_range[1] * h, ny)

    line_width=4
    dot1 = ManimMob(mn.Dot3D(radius=0.07, color=dot_col, resolution=resolution, stroke_opacity=0, stroke_width=0))
    grid_dots = []
    grid_lines = []
    grid_lines2 = []
    grid_indices = []
    for x in x_points:
        for y in y_points:
            grid_dots.append(dot1.clone().move_to(x*RIGHT+y*UP))
            grid_indices.append((int(round(x/h)), int(round(y/h))))
    if mesh_col is None:
        mesh_col = GREY.clone()
        mesh_col[4] = 0.6
    if fill_col is None:
        fill_col = BLUE.clone()
        fill_col[4] = 0.2

    surf = surface_func(nx=nx-1, ny=ny-1, mesh_m=32, mesh_n=32, mesh_col=mesh_col, fill_color=fill_col)
    surf.scale(np.array([(nx-1)*h/2, (ny-1)*h/2, 1]))
    surf.move((x_range[0] + x_range[1])*h/2*RIGHT + (y_range[0] + y_range[1])*h/2*UP)

    grid_dots = Group(*grid_dots)
    #grid_lines = Group(*grid_lines, *grid_lines2)

    return grid_dots, surf, grid_indices

def mesh_points(quality=LD, bgcol=BLACK, show_dots=True, show_eqs=True, dots_only=False):
    dot_col = mn.ManimColor(mn.RED)
    grid_dots, _, grid_indices = create_mesh(dot_col=dot_col, y_range=(-9, 14))

    with Off():
        cam = Scene.get_camera()
        cam.set_distance_to_screen(10)

    if show_dots:
        with Off():
            grid_dots.spawn()

    if dots_only:
        render_to_file('dots_only', render_settings=quality, background_color=bgcol)
        return

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
    dot_col = mn.ManimColor(mn.RED)
    line_op=0.6
    grid_dots, grid_lines, grid_indices = create_mesh(dot_col=dot_col,
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
        p = grid_lines.get_descendants()[1]
        col = p.color.clone()
        fade_dist(grid_dots, p, col, z_dir, z0, line_op=line_op)

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
            fade_dist(grid_dots, p, col, z_dir, z0, line_op=line_op)
        u0 = u

    Scene.wait(0.5)

    name = 'mesh_points3_{}_{}'.format(frame0, frame1)

    render_to_file(name, render_settings=quality, background_color=bgcol)

def setup_mesh(x_range=(-17, 14), y_range=(-11, 14), h=0.6, resolution=(2, 2)):
    dot_col = mn.ManimColor(mn.RED)
    line_op=0.6
    grid_dots, grid_lines, grid_indices = create_mesh(dot_col=dot_col,
                                                      y_range=y_range, x_range=x_range, h=h,
                                                      resolution=resolution)

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
        col = p.color.clone()
        col[:,:,4] *= line_op
        fade_dist(grid_dots, p, col, z_dir, z0)

    return grid_dots, grid_lines, grid_indices, col

def mesh_points4(quality=LD, bgcol=BLACK):
    _, _, _, _ = setup_mesh()

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

def mesh_func2(quality=LD, bgcol=BLACK, anim=1, frame0=0, frame1=30, tscale=1):
    h = 0.6
    x_range = (-17, 14)
    y_range = (-11, 14)
    if anim==2:
        y_range = (-14, 14)
    if anim==3:
        y_range = (-14, 11)
    grid_dots, grid_lines, grid_indices, colg = setup_mesh(x_range=x_range, y_range=y_range)
    zscale = 0.04

    def f(x, y):
        return (x*x-y*y)*zscale

    print('1')
    func_dots = grid_dots

    nx = x_range[1] - x_range[0]
    ny = y_range[1] - y_range[0]
    surf = surface_func(nx=x_range[1] - x_range[0], ny=y_range[1]-y_range[0],
                        mesh_m=32, mesh_n=32)
    surf.scale(np.array([nx*h/2, ny*h/2, 1]))
    surf.move((x_range[0] + x_range[1])*h/2*RIGHT + (y_range[0] + y_range[1])*h/2*UP)
    with Off():
        surf.spawn()

    p = surf.get_descendants()[1]
    loc = p.location.clone()
    col0 = p.color.clone()
    col = col0.clone()
    cam = Scene.get_camera()
    z_dir = cam.get_forward_direction()
    z0 = torch.inner(cam.get_center(), z_dir).item() + dist
    z_dir[:,:,2] = 0
    fade_rate=0.25
    h0 = torch.inner(loc, z_dir[0])

    op = ((1+z0*fade_rate) - h0 * 0.25).clamp(0,1)
    col[:, :, 4] *= op[:,:,0]

    with Off():
        p.set_non_recursive(color=col)
        #func_dots.spawn()

    xvals = loc[:, :, 0] / h
    yvals = loc[:, :, 1] / h
    zvals = (xvals * xvals - yvals * yvals) * zscale

    #Scene.wait(0.5)

    print('4')
    with Off():
        loc[:, :, 2] = zvals
        p.set_non_recursive(location=loc)
        for index, dot in zip(grid_indices, func_dots):
            dot.move(f(*index)*IN)

    print('5')
    q = grid_lines.get_descendants()[1]
    colq = q.color.clone()
    run_time = 1.
    fps = quality.frames_per_second
    t = run_time
    n = round(fps * t)
    dt = 1. / fps
    u0 = 0.
    Scene.wait(dt)
    # with Sync():
    #     cam.orbit_around_point(ORIGIN, 180, OUT)
    for i in range(frame0, frame1):
        u = mn.linear((i + 1) / n)
        du = u - u0
        print('rot', i)
        with Sync(run_time=dt * tscale, rate_func=rate_funcs.identity):
            func_dots.orbit_around_point(ORIGIN, 180*du, IN)
            grid_lines.orbit_around_point(ORIGIN, 180*du, IN)
            surf.orbit_around_point(ORIGIN, 180*du, IN)
            h0 = torch.inner(p.location, z_dir[0])
            op = ((1 + z0 * fade_rate) - h0 * fade_rate).clamp(0, 1)
            col = col0.clone()
            col[:, :, 4] *= op[:, :, 0]
            p.set_non_recursive(color=col)
            fade_dist(func_dots, q, colg, z_dir, z0)
        u0 = u

    Scene.wait(0.5)
    render_to_file('mesh_func2_{}_{}_{}_{}'.format(anim, frame0, frame1, tscale), render_settings=quality, background_color=bgcol)

def mesh_func1(quality=LD, bgcol=BLACK, anim=1):
    h = 0.6
    x_range = (-17, 14)
    y_range = (-11, 14)
    grid_dots, grid_lines, grid_indices, _ = setup_mesh(x_range=x_range, y_range=y_range)
    zscale = 0.04

    def f(x, y):
        return (x * x - y * y) * zscale

    func_dots0 = []

    print('1')
    # with Off():
    #     for index, dot in zip(grid_indices, grid_dots):
    #         func_dots0.append(dot.clone())
    #     print('2')
    #     func_dots = Group(*func_dots0)
    #     print('3')
    func_dots = grid_dots

    nx = x_range[1] - x_range[0]
    ny = y_range[1] - y_range[0]
    surf = surface_func(nx=x_range[1] - x_range[0], ny=y_range[1] - y_range[0],
                        mesh_m=32, mesh_n=32)
    surf.scale(np.array([nx * h / 2, ny * h / 2, 1]))
    surf.move((x_range[0] + x_range[1]) * h / 2 * RIGHT + (y_range[0] + y_range[1]) * h / 2 * UP)
    with Off():
        surf.spawn()

    p = surf.get_descendants()[1]
    loc = p.location.clone()
    col = p.color.clone()
    cam = Scene.get_camera()
    z_dir = cam.get_forward_direction()
    z0 = torch.inner(cam.get_center(), z_dir).item() + dist
    fade_rate = 0.25
    h0 = torch.inner(loc, z_dir[0])

    op = ((1 + z0 * fade_rate) - h0 * 0.25).clamp(0, 1)
    col[:, :, 4] *= op[:, :, 0]

    with Off():
        p.set_non_recursive(color=col)
        # func_dots.spawn()

    xvals = loc[:, :, 0] / h
    yvals = loc[:, :, 1] / h

    # Scene.wait(0.5)

    print('4')
    if 2 > anim >= 0:
        zvals = xvals * -0.3 + yvals * 0.
        with Sync() if anim == 0 else Off():
            loc[:, :, 2] = -zvals
            p.set_non_recursive(location=loc)
            for index, dot in zip(grid_indices, func_dots):
                dot.move((index[0] * -0.3 + index[1] * 0.) * OUT)

    zvals = (xvals * xvals - yvals * yvals) * zscale
    if 4 > anim >= 1 or anim == -1:
        with Sync() if anim == 1 or anim == -1 else Off():
            loc[:, :, 2] = zvals
            p.set_non_recursive(location=loc)
            for index, dot in zip(grid_indices, func_dots):
                dot.move_to(dot.get_center()*(UP+RIGHT) + f(*index) * IN)

    print('5')
    if 4 > anim >= 2:
        with Sync() if anim == 2 else Off():
            loc[:, :, 2] = -zvals
            p.set_non_recursive(location=loc)
            for index, dot in zip(grid_indices, func_dots):
                dot.move(2 * f(*index) * OUT)

    th1 = 20 * DEGREES_TO_RADIANS
    params = [
        (.8 + 0j, .5),
        (.8 + .2j, .5),
        (1 + .32j, 0.5),
        (1 + .32j, 0.),
        #(math.cos(th1) + math.sin(th1) * 1j, .1)
    ]
    if anim >= 3:
        a, c = params[max(anim - 4,0)]
        u = (a + 1. / a)/2
        b = 2-u+np.sqrt(u*u-4*u+3)
        c = 0.5
        print(a, b)
        with Sync() if anim == 3 else Off():
            zvals_c = np.pow(a, xvals) * np.pow(b, yvals) * c
            zvals = zvals_c.real
            loc[:,:,2] = -zvals
            p.set_non_recursive(location=loc)
            for dot, index in zip(func_dots, grid_indices):
                z = np.pow(a, index[0]) * np.pow(b, index[1]) * c
                dot.move_to(dot.get_center() * (UP+RIGHT) + z.real * OUT)

    if anim >= 4:
        #(a1, c1) = (1+.22j, 0.5)
        a1, c1 = params[anim-3]
        run_time = 1.
        fps = quality.frames_per_second
        t = run_time
        n = round(fps * t)
        dt = 1. / fps
        #u0 = 0.
        Scene.wait(dt)
        ilist = list(range(2, n, 4))
        if ilist[-1] < n-1:
            ilist.append(n-1)
        #ilist = [n-1]
        i0 = -1
        l1, l2 = (-10., 10.)
        for i in ilist:
            u = mn.smooth((i + 1) / n)
            #du = u - u0
            a2 = a1 * u + a * (1-u)
            c2 = c1 * u + c * (1-u)
            d2 = (a2 + 1. / a2) / 2
            b2 = 2 - d2 + np.sqrt(d2 * d2 - 4 * d2 + 3)
            print(a2, b2, c2)
            zvals_c = np.pow(a2, xvals) * np.pow(b2, yvals) * c2
            print(zvals_c.real.min().item(), zvals_c.real.max().item())
            loc[:, :, 2] = -zvals_c.real.clip(l1, l2)
            with Sync(run_time=dt*(i-i0), rate_func=rate_funcs.identity):
                p.set_non_recursive(location=loc)
                for dot, index in zip(func_dots, grid_indices):
                    z = (np.pow(a2, index[0]) * np.pow(b2, index[1]) * c2).clip(l1, l2)
                    dot.move_to(dot.get_center() * (UP+RIGHT) + z.real * OUT)
            i0 = i

            #u0 = 0

    print('7')
    # a+1/a+b+1/b=4
    # with c = (a+1/a)/2
    # 2c + b + 1/b = 4
    # b^2 + 2(c-2)b+1=0
    # b = 2-c + sqrt((c-2)^2-1)
    Scene.wait(0.5)

    render_to_file('mesh_func1_{}'.format(anim), render_settings=quality, background_color=bgcol)

def mesh_func3(quality=LD, bgcol=BLACK, anim=1):
    h = 0.6
    x_range = (-17, 14)
    y_range = (-11, 14)
    grid_dots, grid_lines, grid_indices, _ = setup_mesh(x_range=x_range, y_range=y_range)

    M = ah.rotation_matrix(RIGHT, 60 * DEGREES_TO_RADIANS)
    pt = IN + np.dot(M, (OUT * 2 + DOWN + IN*8).numpy())

    nx = x_range[1] - x_range[0]
    ny = y_range[1] - y_range[0]
    surf1 = surface_func(nx=x_range[1] - x_range[0], ny=y_range[1]-y_range[0],
                        mesh_m=32, mesh_n=32)
    surf1.scale(np.array([nx*h/2, ny*h/2, 1]))
    surf1.move((x_range[0] + x_range[1])*h/2*RIGHT + (y_range[0] + y_range[1])*h/2*UP)
    with Off():
        surf1.spawn()
    p = surf1.get_descendants()[1]
    loc = p.location.clone()
    col = p.color.clone()
    cam = Scene.get_camera()
    z_dir = cam.get_forward_direction()
    z0 = torch.inner(cam.get_center(), z_dir).item() + dist
    fade_rate = 0.25
    h0 = torch.inner(loc, z_dir[0])

    op = ((1 + z0 * fade_rate) - h0 * 0.25).clamp(0, 1)
    col[:, :, 4] *= op[:, :, 0]

    with Off():
        p.set_non_recursive(color=col)

    nodes = [(3,2), (-1, -2), (-4, 1)]

    with Sync(run_time=1) if anim == 1 else Off():
        cam.orbit_around_point(ORIGIN, 30, IN)
        cam.orbit_around_point(pt, -60, RIGHT)
        grid_lines.despawn()
        surf1.despawn()


    surfs = []
    fill_col = BLUE.clone()
    fill_ops = [0.]
    if anim == 2:
        fill_ops.append(0.2)
    for fill_op in fill_ops:
        fill_col[4] = fill_op
        nx = x_range[1] - x_range[0]
        ny = y_range[1] - y_range[0]
        surf = surface_func(mesh_col=GREY,
                             nx=nx, ny=ny,
                             mesh_m=32,
                             mesh_n=32,
                             fill_color=fill_col
                             )
        p = surf.get_descendants()[1]
        col = p.color.clone()
        col[:, :, 4] *= 0.6
        p.set_non_recursive(color=col)
        surf.scale(np.array([nx*h/2, ny*h/2, 1]))
        surf.move((x_range[0] + x_range[1])*h/2*RIGHT + (y_range[0] + y_range[1])*h/2*UP)
        surfs.append(surf)
    surf = surfs[0]

    if anim > 1:
        with Off():
            if anim == 2:
                surf.spawn()
                p = surf.get_descendants()[1]
                col = p.color.clone()
                fade_dist(grid_dots, p, col=col, z_dir=z_dir, z0=z0)
                p = surfs[1].get_descendants()[1]
                col = p.color.clone()
                fade_dist(grid_dots, p, col=col, z_dir=z_dir, z0=z0)
                for node in nodes:
                    i = grid_indices.index(node)
                    grid_dots[i].set(color=YELLOW)
            else:
                grid_dots.despawn()
        if anim == 4:
            img = ImageMob('../../media/resistors.png')
            a = 1.94
            img.move(UP * 0.04)
            img.scale(np.array([16 / 9 * a, a, 1.]))
            with Off():
                img.spawn()
        Scene.wait(0.1)
        with Sync(run_time=1):
            cam.orbit_around_point(pt, 60, RIGHT)
            cam.orbit_around_point(ORIGIN, -30, IN)
            if anim == 2:
                surf.despawn()
                surfs[1].spawn()
            if anim == 3:
                surf1.spawn()
        Scene.wait(0.1)
        name = 'mesh_func{}'.format(anim + 2)
        render_to_file(name, render_settings=quality, background_color=bgcol)
        return

    Scene.wait(0.1)

    with Sync(run_time=1):
        surf.spawn()

    Scene.wait(0.1)

    with Sync(run_time=1):
        for node in nodes:
            i = grid_indices.index(node)
            grid_dots[i].set(color=YELLOW)

    Scene.wait(0.2)

    # with Sync(run_time=1):
    #     #grid_lines.spawn()
    #     cam.orbit_around_point(pt, -60, LEFT)
    #     cam.orbit_around_point(ORIGIN, 30, OUT)
    #
    # Scene.wait(0.1)

    render_to_file('mesh_func3', render_settings=quality, background_color=bgcol)

def solve_harmonic(nx, ny, nodes, values, tol = 0.0001):
    res = np.zeros(shape=(nx, ny))
    res2 = np.zeros(shape=(nx, ny))
    p = 0.5/4
    for (i, j), val in zip(nodes, values):
        res[i, j] = val

    count = 0
    while True:
        res2[:,:] = (1-4*p) * res[:,:]
        res2[:-1, :] += p * res[1:, :]
        res2[-1, :] += p * res[-1, :]
        res2[1:, :] += p * res[:-1, :]
        res2[0, :] += p * res[0, :]
        res2[:, :-1] += p * res[:, 1:]
        res2[:, -1] += p * res[:, -1]
        res2[:, 1:] += p * res[:, :-1]
        res2[:, 0] += p * res[:, 0]
        for (i, j), val in zip(nodes, values):
            res2[i, j] = val

        (res2, res) = (res, res2)
        res2 -= res
        maxdiff = res2.max()
        #print(count, ':', maxdiff)
        count += 1
        if maxdiff <= tol:
            break

    print('iters:', count)
    return res


def harmonic_func(quality=LD, bgcol=BLACK, anim=1):
    h = 0.6
    x_range = (-17, 14)
    y_range = (-11, 14)
    print('0')
    grid_dots, grid_lines, grid_indices, _ = setup_mesh(x_range=x_range, y_range=y_range)
    nodes = [(3,2), (-1, -2), (-4, 1)]
    nx = x_range[1] - x_range[0]
    ny = y_range[1] - y_range[0]
    surf = surface_func(nx=x_range[1] - x_range[0], ny=y_range[1] - y_range[0],
                        mesh_m=32, mesh_n=32)
    surf.scale(np.array([nx * h / 2, ny * h / 2, 1]))
    surf.move((x_range[0] + x_range[1]) * h / 2 * RIGHT + (y_range[0] + y_range[1]) * h / 2 * UP)
    print('1')

    p = surf.get_descendants()[1]
    loc = p.location.clone()
    col = p.color.clone()
    cam = Scene.get_camera()
    z_dir = cam.get_forward_direction()
    z0 = torch.inner(cam.get_center(), z_dir).item() + dist
    fade_rate = 0.25
    h0 = torch.inner(loc, z_dir[0])
    op = ((1 + z0 * fade_rate) - h0 * 0.25).clamp(0, 1)
    col[:, :, 4] *= op[:, :, 0]

    nodes0 = [(i - x_range[0], j - y_range[0]) for i, j in nodes]
    vals_arr = [
        [-4., 2.5, 3.2],
        [-4., -1., 3.2],
        [2.8, -2., 2.5],
    ]
    p = surf.get_descendants()[1]
    loc = p.location.clone()
    if anim > 1:
        node_vals = vals_arr[anim - 2]
        surf_vals = solve_harmonic(nx+1, ny+1, nodes0, node_vals)
        with Off():
            for dot, (i, j) in zip(grid_dots, grid_indices):
                dot.move_to(dot.get_center() + surf_vals[i - x_range[0], j - y_range[0]] * OUT)
        f = sp.interpolate.RegularGridInterpolator((np.linspace(x_range[0]*h, x_range[1]*h, nx+1),
                                                np.linspace(y_range[0]*h, y_range[1]*h,ny+1)),
                                               surf_vals, bounds_error=False, fill_value=0.)
        fvals = f(loc[0,:,:2].numpy())
        loc[0,:,2] = torch.from_numpy(-fvals)
        p.set_non_recursive(location=loc)

    with Off():
        for node in nodes:
            i = grid_indices.index(node)
            grid_dots[i].set(color=YELLOW)
        p.set_non_recursive(color=col)
        surf.spawn()

    print(3)

    Scene.wait(0.1)
    p = surf.get_descendants()[1]
    loc = p.location.clone()
    (x_vals, y_vals) = (np.linspace(x_range[0]*h, x_range[1]*h, nx+1),
                                                np.linspace(y_range[0]*h, y_range[1]*h,ny+1))
    if anim <= len(vals_arr):
        node_vals = vals_arr[anim-1]
        surf_vals = solve_harmonic(nx+1, ny+1, nodes0, node_vals)
        f = sp.interpolate.RegularGridInterpolator((x_vals, y_vals),
                                               surf_vals, bounds_error=False, fill_value=0.)
        fvals = f(loc[0,:,:2].numpy())
        loc[0,:,2] = torch.from_numpy(-fvals)
    else:
        zscale = 0.04/h/h
        def f(x, y):
            return (x * x - y * y) * zscale
        surf_vals = np.zeros(shape=(nx+1, ny+1))
        for i in range(nx+1):
            for j in range(ny+1):
                x, y = (x_vals[i], y_vals[j])
                surf_vals[i, j] = f(x, y)
        loc[0,:,2] = (loc[0,:,1] * loc[0,:,1] - loc[0,:,0] * loc[0,:,0]) * zscale

    with Sync(run_time=1.): #Sync(run_time=1.):
        for dot, (i, j) in zip(grid_dots, grid_indices):
            dot.move_to(dot.get_center() * (UP+RIGHT) + surf_vals[i - x_range[0], j - y_range[0]] * OUT)
        if anim > len(vals_arr):
            for dot, (i, j) in zip(grid_dots, grid_indices):
                if (i, j) in nodes:
                    dot.set(color=RED)

        p.set_non_recursive(location=loc)

    Scene.wait(0.1)



    render_to_file('harmonic_func{}'.format(anim), render_settings=quality, background_color=bgcol)

def random_walk(quality=LD, bgcol=BLACK, anim=1, steps=10, p=1.):
    h = 0.6
    x_range = (-17, 14)
    y_range = (-11, 14)
    nx = x_range[1] - x_range[0] + 1
    ny = y_range[1] - y_range[0] + 1
    #grid_dots, grid_lines, grid_indices, _ = setup_mesh(x_range=x_range, y_range=y_range)
    M = ah.rotation_matrix(RIGHT, -60 * DEGREES_TO_RADIANS)
    pt = IN + np.dot(M, (OUT * 2 + DOWN).numpy())
    cam = Scene.get_camera()
    with Off():
        cam.set_distance_to_screen(10)
        cam.orbit_around_point(pt, 60, RIGHT)
        cam.orbit_around_point(ORIGIN, -30, IN)

    i, j = (17, 11)
    col = mn.YELLOW
    np.random.seed(1)
    if anim == 3:
        i += 3
        j += -1
        col = mn.WHITE
        np.random.seed(2)
    (x_vals, y_vals) = (np.linspace(x_range[0]*h, x_range[1]*h, nx),
                                                np.linspace(y_range[0]*h, y_range[1]*h,ny))
    dot1 = ManimMob(mn.Dot3D(radius=0.12, color=col, resolution=(10, 10), stroke_opacity=0, stroke_width=0))
    dot1.move_to(x_vals[i] * RIGHT + y_vals[j] * UP)
    dot1.glow = 1.
    with Off():
        dot1.spawn()
    Scene.wait(0.1)
    det_steps = r''
    if anim == 4:
        det_steps = r'rruuluddrrrdrlduuludrdduld'
        steps = len(det_steps)
    for i in range(steps):
        u = np.random.uniform(0., 1.)
        if i < len(det_steps):
            dir = det_steps[i]
            if dir==r'u':
                shift=UP*h
            elif dir==r'd':
                shift=DOWN*h
            elif dir==r'l':
                shift=LEFT*h
            elif dir==r'r':
                shift = RIGHT * h
        else:
            if u < 0.25 * p:
                shift = RIGHT * h
            elif u < 0.5 * p:
                shift = LEFT * h
            elif u < 0.75 * p:
                shift = UP * h
            else:
                shift = DOWN * h
        with Sync(run_time=0.5):
            dot1.move(shift)

    Scene.wait(0.1)
    kernel_size = 63
    strength = 8
    scale_factor=4
    num_iterations=7
    bf = bloom_filter_premultiply if bgcol == TRANSPARENT else bloom_filter
    bloom_new = partial(bf, num_iterations=num_iterations, kernel_size=kernel_size, strength=strength, scale_factor=scale_factor)
    render_to_file('random_walk{}'.format(anim), render_settings=quality, background_color=bgcol,  post_processes = [bloom_new])


def surf_thumb(quality=LD, bgcol=BLACK):
    h = 0.6
    x_range = (-5, 5)
    y_range = (-5, 5)
    zscale = 0.08
    grid_dots, grid_lines, grid_indices, _ = setup_mesh(x_range=x_range, y_range=y_range, h=h, resolution=(4, 4))

    func_dots = grid_dots

    nx = x_range[1] - x_range[0]
    ny = y_range[1] - y_range[0]
    a = 0.4
    fill_color = (PURE_BLUE * a + BLUE * (1-a))
    fill_color = GREEN.clone()
    fill_color[:3] *= 0.5
    fill_color[4] = 0.9
    mesh_col=GREEN_D.clone()
    mesh_col[:3] *= 0.2
    mesh_col=GREY.clone()

    surf = surface_func(nx=x_range[1] - x_range[0], ny=y_range[1] - y_range[0],
                        mesh_m=32, mesh_n=32, fill_color=fill_color, mesh_col=mesh_col)
    surf.scale(np.array([nx * h / 2, ny * h / 2, 1]))
    surf.move((x_range[0] + x_range[1]) * h / 2 * RIGHT + (y_range[0] + y_range[1]) * h / 2 * UP)
    with Off():
        grid_lines.despawn()
        surf.spawn()
        for dot, index in zip(grid_dots, grid_indices):
            if index[0] == x_range[0] or index[0] == x_range[1] or index[1] == y_range[0] or index[1] == y_range[1]:
                dot.despawn()
            else:
                dot.scale(0.8)

    p = surf.get_descendants()[1]
    loc = p.location.clone()
    col = p.color.clone()
    cam = Scene.get_camera()

    with Off():
        cam.move(IN)
        #cam.move_to(cam.get_center()*2)
        p.set_non_recursive(color=col)

    xvals = loc[:, :, 0] / h
    yvals = loc[:, :, 1] / h

    def f(x, y):
        return (x * x - y * y) * zscale

    zvals = (xvals * xvals - yvals * yvals) * zscale

    with Off():
        loc[:, :, 2] = -zvals
        p.set_non_recursive(location=loc)
        for index, dot in zip(grid_indices, func_dots):
            dot.move(f(*index) * OUT)

    render_to_file('surf_thumb', render_settings=quality, background_color=bgcol)



if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 20
    #COMPUTING_DEFAULTS.max_animate_batch_size = 4
    bgcol = Color('#202020')
    #mesh_points(quality=LD, bgcol=bgcol, show_eqs=True)
    #mesh_points(quality=HD, bgcol=TRANSPARENT, dots_only=True)
    #mesh_points2(quality=HD, bgcol=bgcol, part=1)
    #mesh_points2(quality=HD, bgcol=bgcol, part=2)
    #mesh_points2(quality=HD, bgcol=bgcol, part=3)
    #mesh_points3(quality=HD, bgcol=bgcol, frame0=0, frame1=11)
    #mesh_points3(quality=HD, bgcol=bgcol, frame0=10, frame1=21)
    #mesh_points3(quality=HD, bgcol=bgcol, frame0=20, frame1=30)
    #mesh_points4(quality=HD, bgcol=bgcol)
    #mesh_func1(quality=HD, bgcol=bgcol, anim=-1)
    #mesh_func1(quality=LD, bgcol=bgcol, anim=0)
    #mesh_func1(quality=HD, bgcol=bgcol, anim=1)
    #mesh_func1(quality=LD, bgcol=bgcol, anim=2)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=1, frame0=0, frame1=11, tscale=4)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=2, frame0=0, frame1=11, tscale=4)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=2, frame0=10, frame1=21, tscale=4)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=2, frame0=20, frame1=31, tscale=4)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=3, frame0=10, frame1=21, tscale=4)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=3, frame0=20, frame1=31, tscale=4)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=3, frame0=10, frame1=21)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=3, frame0=20, frame1=31)
    #mesh_func2(quality=HD, bgcol=bgcol, anim=1, frame0=20, frame1=31)
    #mesh_func1(quality=HD, bgcol=bgcol, anim=3)
    #mesh_func1(quality=HD, bgcol=bgcol, anim=5)

    #mesh_func3(quality=HD, bgcol=bgcol)
    #mesh_func3(quality=HD, bgcol=bgcol, anim=2)
    #mesh_func3(quality=HD, bgcol=TRANSPARENT, anim=4)
    #harmonic_func(quality=HD, bgcol=bgcol, anim=2)
    #random_walk(quality=HD, bgcol=TRANSPARENT, anim=4, steps=120, p=1.)
    surf_thumb(quality=HD, bgcol=TRANSPARENT)
