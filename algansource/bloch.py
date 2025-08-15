import numpy as np
import torch
from algan import *
import manim as mn
import math

from algan.external_libraries.manim import ArcBetweenPoints
from algan.rendering.post_processing import bloom_filter, bloom_filter_premultiply
from functools import partial
from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader
#from electron import create_electron
from manim import Arrow3D, VGroup

LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate_vec(vec, angle=0., axis=OUT):
    M = rotation_matrix(axis.numpy(), angle * PI / 180)
    dir2 = torch.tensor(np.dot(M, vec.numpy()))
    return dir2


def arrow3D(body_length=1., tip_length=1., radius=0.03, tip_radius=0.1):
    ht = body_length
    ht2 = r - ht
    c1 = Cylinder(radius=radius, height=ht).move(UP * ht / 2)
    cone = mn.Cone(base_radius=tip_radius, height=ht2, show_base=True, resolution=[1, 10], stroke_opacity=1,
                   stroke_color=mn.WHITE, stroke_width=0,
                   fill_color=mn.WHITE, direction=mn.UP).shift(mn.UP * (ht + ht2))
    cone.submobjects = [obj for obj in cone.submobjects if type(obj) != mn.VectorizedPoint]
    #line = ManimMob(mn.Line(ORIGIN, UP*3, stroke_width=5, stroke_color=mn.RED))
    arr1 = Group(c1, ManimMob(cone))
    return arr1


def create_sphere(r=1., simple_sphere=True, anim='', min_op_s1=0.1):
    # min_op_s1 = 0.1  # opacity of outer sphere at center (minimum opacity)
    max_op_s2 = 0.6  # opacity of inner sphere at center (maximum opacity)
    base_op_s1 = 0.1  # additional opacity for surface
    interior_col = GREEN_C
    # surface_col = Color("#333333")
    surface_col = Color("#000060")
    r2 = r * 0.995
    fill_opacity = 0.4 if simple_sphere else 0
    s3 = ManimMob(mn.Sphere(radius=r * 1.005, fill_opacity=fill_opacity, fill_color = mn.GREEN, stroke_opacity=0.5,
                            stroke_color=mn.BLACK, stroke_width=2, resolution=(24, 12)))
    with Off():
        s3.orbit_around_point(ORIGIN, 90, RIGHT)
        s3.orbit_around_point(ORIGIN, -7, UP)  # don't cover the center with a line
        s3.orbit_around_point(ORIGIN, 35, RIGHT)  # let's make the north pole point slightly towards us
        if anim != 'inside':
            s3.spawn()

    a = math.log(1 - min_op_s1) * r # min opacity = 1 - exp(a/r)
    b = math.log(1 - max_op_s2) / r  # max opacity = 1 - exp(b * r)

    def interior_col_update(obj: Mob, t=(0.,), color=None):
        # print('updator called')
        if color is not None:
            obj.base_color = color
        col = obj.base_color

        for p in obj.get_descendants():
            loc = p.location
            inside = loc.norm(dim=-1, keepdim=True).le(r2)
            depth = (r2*r2 - loc[...,:2].square().sum(dim=-1, keepdim=True)).clamp(0).sqrt() + loc[...,2:3]
            inside_op = 1 - (depth * (b/2)).exp()
            new_col = torch.where(inside, col * (1-inside_op) + interior_col * inside_op, col)
            # print(len(t), loc.shape)
            p.set_non_recursive(color=new_col)
        return obj

    if not simple_sphere:
        s1 = Sphere(radius=r, color=surface_col, opacity=1).set_shader(basic_pbr_shader)
        s2 = Sphere(radius=r2, color=interior_col, opacity=1).set_shader(null_shader)

        for p in s1.get_descendants():
            op = 1 - (a / p.location[...,2:3].abs().clamp(0.01)).exp() * (1 - base_op_s1)
            p.set_non_recursive(color=p.color.set_opacity(op))
        for p in s2.get_descendants():
            op = 1 - (p.location[...,2:3].clamp(0., r) * b).exp()
            print(op.max())
            p.set_non_recursive(color=p.color.set_opacity(op))

        with Off():
            s1.smoothness = 0.6
            s1.metallicness = 0.4
            if anim != 'inside':
                s1.spawn()
            if anim != 'surface':
                s2.spawn()

    return s3, interior_col_update

def anchor_pts(start=0., end=1., n=10):
    pts = np.linspace(start, end, n)
    anchors = []
    for i in range(len(pts) - 1):
        anchors += [pts[i], 0.67 * pts[i] + 0.33 * pts[i + 1], 0.33 * pts[i] + 0.67 * pts[i + 1], pts[i + 1]]
    anchors2 = torch.tensor(anchors)
    return anchors2

def curve3d(f, start=0., end=1., npts=2, **kwargs):
    anchors = anchor_pts(start=start, end=end, n=npts)
    arc = BezierCircuitCubic(torch.cat([f(a) for a in anchors], -2),
                             filled=False, border_width=4, **kwargs)
    return arc

def line3d(start, end, npts=6, **kwargs):
    start = cast_to_tensor(start)
    end = cast_to_tensor(end)
    arc = curve3d(lambda a: start * (1-a) + a * end)
    return arc

def line3d_pts(start, end, npts=6):
    start = cast_to_tensor(start)
    end = cast_to_tensor(end)
    anchors = anchor_pts(start=0., end=1., n=npts)
    return torch.cat([start*(1-a) + a*end for a in anchors], -2)

def bloch(r=1., simple_sphere=False, anim='sphere only'):
    min_op_s1 = 0.1
    if anim=='sphere only':
        min_op_s1 = 0.99
    sphere, col_update = create_sphere(r, simple_sphere, anim=anim, min_op_s1=min_op_s1)
    # c1 = Cylinder(radius=0.2, height=1)

    center_dot = Sphere(radius=r * 0.04)
    col_update(center_dot, color=WHITE)

    with Off():
        Scene.get_camera().set_distance_to_screen(15)
        # c1.spawn().move(IN*2.3)
        light_source = Scene.get_light_sources()[0]
        light_source.move_to(UP * 12)
        light_source.rotate_around_point(ORIGIN, 45, OUT)
        light_source.rotate_around_point(ORIGIN, 20, UP)
        # print('location:', light_source.location)
        # print('color:', light_source.color)
        # print(light_source.location.norm())
        loc = -light_source.location
        loc[...,2] *= -1
        Scene.add_light_source(PointLight(location=loc, color=Color("#444444"), opacity=0.5).spawn())
        # col_update(c1, color=WHITE)
        if anim == 'center':
            center_dot.spawn()


    if anim=='sphere only' or anim == 'surface' or anim == 'inside' or anim == 'opaque' or anim == 'center':
        Scene.wait(1/15)
        return 'Bloch_Sphere'

    # c1.add_updater(col_update)

    # with Sync(run_time=3, rate_func=rate_funcs.identity):
        # c1.move(OUT*6)

    s_up = sphere.get_forward_direction()
    s_right = sphere.get_right_direction()
    s_forward = sphere.get_upwards_direction()
    mn_up = s_up.numpy()[0,0,:]
    mn_right = s_right.numpy()[0,0,:]
    mn_forward = s_forward.numpy()[0,0,:]
    tilt = math.acos(mn_up[1]) / 2 / PI * 360

    if anim == 'angle':
        arr1 = arrow3D(body_length=0.8 * r, tip_length=0.2*r, tip_radius=0.15)
        #arr1.orbit_around_point(ORIGIN, tilt, RIGHT)

        with Off():
            center_dot.spawn()
            print(arr1[0].get_upwards_direction())
            arr1.orbit_around_point(ORIGIN, 50, RIGHT)
            print(arr1.get_upwards_direction())
            arr1.orbit_around_point(ORIGIN, 45, UP) # line up with electron
            dir1 = arr1.get_upwards_direction()[0][0]
            dir1 /= torch.norm(dir1)
            print(dir1)
            print(torch.norm(dir1))
            x, y, z = (dir1[i].item() for i in range(3))
            dir2 = z * RIGHT + x * OUT
            dir2 /= torch.norm(dir2)
            col_update(arr1, color=WHITE)
            arr2 = arr1.clone()
            arr3 = arr1.clone()
            arr1.spawn()
            arr2.spawn()
            arr3.orbit_around_point(ORIGIN, -60, dir2)
            dir3 = arr3.get_upwards_direction()
            dir4 = torch.cross(dir1, dir2, 0)
            dir4 /= torch.norm(dir4)
            start = cast_to_tensor(dir1) * r * 1.02
            end = cast_to_tensor(-dir4) * r * 1.02
            arc = curve3d(lambda a: start * torch.cos(a) + torch.sin(a) * end, 0., PI/3, 10)
            #arc.spawn()

        def updater(mob, t):
            print(t)
        # arc.add_updater(updater)

        with Sync(run_time=1, rate_func=rate_funcs.identity):
            arr1.orbit_around_point(ORIGIN, -60, dir2)
            arc.spawn()
        Scene.wait(0.1)

        return 'bloch_angle'


    if anim == 'antipode':
        depth = 1
        max_op_s2 = 0.6
        interior_col = GREEN_C
        b = math.log(1 - max_op_s2)
        inside_op = 1 - math.exp(depth * (b / 2))
        new_white = WHITE * (1 - inside_op) + interior_col * inside_op
        pt1 = RIGHT+UP+OUT*0.5
        pt1 *= r / torch.norm(pt1)
        print(pt1)
        dot1 = Sphere(radius=r * 0.06).move_to(pt1)
        dot2 = Sphere(radius=r * 0.06).move_to(-pt1)
        line = Line(pt1*0.95, -pt1*0.95, color=new_white, border_width=4)
        line1 = Line(pt1*0.95, pt1*0.95, color=new_white, border_width=5)
#        line = ManimMob(mn.Line(pt1, -pt1, stroke_width=5))
        # col_update(line, color=WHITE)
        col_update(dot1, color=WHITE)
        col_update(dot2, color=WHITE)
        with Seq(run_time=1):
            dot1.spawn()
        with Off():
            line1.spawn()
        with Seq(run_time=2, rate_func=rate_funcs.identity):
            with Lag(0.5):
                with Sync():
                    line1.become(line)
                    center_dot.spawn()
                dot2.spawn()
            #center_dot.become(line)
            #line.spawn()

        return 'bloch_antipode'

    if anim == 'electron':
        bloom = True
        depth = 1
        max_op_s2 = 0.6
        interior_col = mn.GREEN_C
        b = math.log(1 - max_op_s2)
        inside_op = 1 - math.exp(depth * (b / 2))
        new_white = mn.WHITE * (1 - inside_op) + interior_col * inside_op
        new_yellow = mn.YELLOW * (1 - inside_op) + interior_col * inside_op
        bare, field = create_electron(r*0.5, sub_n=5, bloom=bloom, colors=[new_white, new_yellow])
        arr = bare[1]
        dressed = Group(bare, field)
        dressed.orbit_around_point(ORIGIN, 40, LEFT)
        dressed.orbit_around_point(ORIGIN, 45, UP)
        col_update(arr, color=WHITE)
        print(arr.get_upwards_direction())
        with Off():
            dressed.spawn()
            #col_update(dressed, color=WHITE)

        with Sync(run_time=2, rate_func=identity):
            elec_up = -bare.get_forward_direction()
            dressed.orbit_around_point(ORIGIN, 360, elec_up)

        return 'bloch_electron'


    if anim == 'arrow':
        arr1 = arrow3D(body_length=0.8 * r, tip_length=0.2*r, tip_radius=0.15)
        #arr1.orbit_around_point(ORIGIN, tilt, RIGHT)

        with Off():
            arr1.spawn()
            center_dot.spawn()
            print(arr1[0].get_upwards_direction())
            arr1.orbit_around_point(ORIGIN, 50, RIGHT)
            print(arr1.get_upwards_direction())
            arr1.orbit_around_point(ORIGIN, 45, UP) # line up with electron
            dir1 = arr1.get_upwards_direction()
            print(dir1)
            col_update(arr1, color=WHITE)

        dt = 1
        n = 30

        with Seq(rate_func=rate_funcs.identity):
            #arr1.add_updater(col_update)
            #arr1.rotate_around_point(ORIGIN, 80, s_forward - s_right)
            with Seq(run_time=dt):
                axis1 = OUT+dir1+RIGHT*0.2
                axis1 /= torch.norm(axis1).item()
                arr1.orbit_around_point(ORIGIN, -180, axis1)# + s_right + s_up)
                dir2 = arr1.get_upwards_direction()
                print(dir2)
            with Seq(run_time=2*dt):
                axis2 = 2 * torch.inner(dir1, dir2).item() * dir2 - dir1
                axis2 /= torch.norm(axis2)
                arr1.orbit_around_point(ORIGIN, 360, axis2)# + s_right + s_up)
            with Seq(run_time=1*dt):
                arr1.orbit_around_point(ORIGIN, -180, axis1)# + s_right + s_up)
            Scene.wait(0.1)

        return 'Bloch_arrow'

    print(s_up)
    print(s_right)
    print(s_forward)
    fs = 50
    eqkets = Group(
        ManimMob(mn.MathTex(r'\lvert{\rm up}\rangle', font_size=fs)).move_to(s_up * r * 1.2),
        ManimMob(mn.MathTex(r'\lvert{\rm down}\rangle', font_size=fs)).move_to(-s_up * r * 1.45),
        ManimMob(mn.MathTex(r'\lvert{\rm right}\rangle', font_size=fs)).move_to(s_right * r * 1.38),
        ManimMob(mn.MathTex(r'\lvert{\rm left}\rangle', font_size=fs)).move_to(-s_right * r * 1.32),
        ManimMob(mn.MathTex(r'\lvert{\rm front}\rangle', font_size=fs)).move_to(s_forward * r * 1.05),
        ManimMob(mn.MathTex(r'\lvert{\rm back}\rangle', font_size=fs)).move_to(-s_forward * r * 1.05)
    )
    axes = Group(
        ManimMob(mn.Line(ORIGIN, mn_forward * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, -mn_forward * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, mn_up * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, -mn_up * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, mn_right * r * 0.99, stroke_width=1)),
        ManimMob(mn.Line(ORIGIN, -mn_right * r * 0.99, stroke_width=1))
    )
    col_update(axes, color=WHITE)

    if anim == 'AliceBob':
        print('Alice and Bob')
        dots = [Sphere(radius=r * 0.06) for _ in range(8)]
        dots[0].move_to(s_up*r)
        dots[3].move_to(-s_up*r)
        dots[4].move_to(s_right*r)
        dots[7].move_to(-s_right*r)
        for dot in dots:
            col_update(dot, color=WHITE)

        with Off():
            dots[0].spawn()
            dots[3].spawn()
            eqkets[0].spawn()
            eqkets[1].spawn()
            axes[2].spawn()
            axes[3].spawn()
            center_dot.spawn()

        dt = 1.

        with Sync(run_time=dt):
            dots[0].become(dots[1])
            dots[3].become(dots[2])
        with Sync(run_time=dt):
            dots[4].spawn()
            dots[7].spawn()
            axes[4].spawn()
            axes[5].spawn()
            eqkets[2].spawn()
            eqkets[3].spawn()
        with Sync(run_time=dt):
            dots[4].become(dots[5])
            dots[7].become(dots[6])
        Scene.wait(0.1)
        return 'bloch_ab'

    if anim == 'mixed' or anim == 'mixed2':
        dir1 = RIGHT + UP + OUT
        dir1 /= torch.norm(dir1)
        print(dir1)
        pos1 = dir1 * r/2

        dot1 = Sphere(radius=r * 0.06)
        dots = [Sphere(radius=r*0.04) for _ in range(2)]
        dots[0].move_to(dir1 * r)
        dots[1].move_to(-dir1 * r)
        with Off():
            dot1.move_to(pos1).spawn()
            col_update(dot1, color=WHITE)
            for dot in dots:
                col_update(dot, color=WHITE)
            center_dot.spawn()

        if anim == 'mixed':
            dt=1
            with Off():
                lines = [line3d(pos1, dir1 * r), line3d(pos1, -dir1 * r),
                         line3d(pos1, pos1), line3d(pos1, pos1)]
                for line in lines:
                    col_update(line, color=WHITE)
                lines[2].spawn()
                lines[3].spawn()

            with Lag(0.5):
                with Sync(run_time=dt):
                    lines[2].become(lines[0])
                    lines[3].become(lines[1])
                with Sync(run_time=dt):
                    dots[0].spawn()
                    dots[1].spawn()

            return 'bloch_mixed'

        line1 = line3d(-dir1*r, dir1*r, npts=5)
        col_update(line1, color=WHITE)
        col_update(dots[1], color=WHITE)
        col_update(dots[1], color=WHITE)

        dt = 0.3
        n = 5
        line2 = line1.clone()
        with Off():
            dots[0].spawn()
            dots[1].spawn()
            line2.spawn()

        def move_to(line: BezierCircuitCubic, dots, pos1, dir2, dt, anim=Sync):
            c = torch.inner(pos1, pos1).item() - r*r
            b = 2*torch.inner(pos1, dir2).item()
            a = torch.inner(dir2, dir2).item()
            x = (-b - math.sqrt(b*b-4*a*c))/(2*a)
            y = (-b + math.sqrt(b*b-4*a*c))/(2*a)
            pos = (pos1 + x*dir2, pos1 + y*dir2)
            #line4 = line3d(*pos)
            pts = line3d_pts(*pos, npts=5)
            with anim(run_time=dt, rate_func=rate_funcs.identity):
                #line1.orbit_around_point(pos1, -90, IN)
                #line.become(line4)
                line.control_points.location = pts
                dots[0].move_to(pos[0])
                dots[1].move_to(pos[1])
                col_update(dots[0])
                col_update(dots[1])
                col_update(line2, color=WHITE)
            return line

        if anim == 'mixed2':
            for i in range(n):
                theta = 90 * (i+1) / n
                dir2 = rotate_vec(dir1, theta, OUT)
                line2 = move_to(line2, dots, pos1, dir2, dt/n)
            Scene.wait(0.1)
            return 'bloch_mixed2'

        dir2 = rotate_vec(dir1, 90, OUT)
        line2 = move_to(line2, dots, pos1, dir2, 1., anim=Off)

        for i in range(n):
            theta = 180 * (i+1) / n
            dir3 = rotate_vec(dir2, theta, dir1)
            line2 = move_to(line2, dots, pos1, dir3, dt/n)
        Scene.wait(0.1)

            #print(dir1/torch.norm(dir1), dir2)

        return 'bloch_mixed3'


    if anim == 'kets':
        with Off():
            center_dot.spawn()
            eqkets.spawn()
            axes.spawn()
        Scene.wait(1/15)
        return 'bloch_kets'


    return 'bloch'

if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 6
    quality = LD
    r = 2.
    bgcol = DARKER_GREY
    bgcol = DARK_GREY
    simple_sphere = True

    anim = 'sphere only'
    anim = 'surface'
    anim = 'inside'
    anim = 'kets'
    #anim = 'electron'
    #anim = 'opaque'
    #anim = 'arrow'
    #anim = 'antipode'
    #anim = 'angle'
    anim = 'AliceBob'
    anim = 'mixed'
    anim = 'mixed2'

    if anim == 'opaque':
        bgcol = BLACK

    kernel_size = int(93/0.4)
    bloom_new = partial(bloom_filter, num_iterations=7, kernel_size=kernel_size, strength=10, scale_factor=6)

    name = bloch(r, simple_sphere=simple_sphere, anim=anim)
    render_to_file(name, render_settings=quality, background_color=bgcol, post_processes=[bloom_new])
