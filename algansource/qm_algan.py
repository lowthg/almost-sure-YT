import numpy as np
from algan import *
import manim as mn
import math
from algan.rendering.post_processing import bloom_filter, bloom_filter_premultiply
from functools import partial

from manim import Arrow3D, VGroup

LD = RenderSettings((854, 480), 15)
HD = RenderSettings((1920, 1080), 30)


dir_str = {
    UP: 'U',
    DOWN: 'D',
    RIGHT: 'R',
    LEFT: 'L'
}
dir_angle = {
    UP: 0,
    DOWN: 180,
    RIGHT: 90,
    LEFT: 270
}

def create_electron(r=1, bloom=True) -> Mob:
    r1 = r * 0.5
    r2 = r * 1.5
    r4 = r * 0.8
    r5 = r * 1.5
    r_max = 2.5 * r
    s1 = ManimMob(mn.Sphere(radius=r1, stroke_opacity=0, resolution=[8, 4], checkerboard_colors=[mn.WHITE, mn.YELLOW]))
    # arr1 = ManimMob(mn.Arrow3D(start=OUT.numpy()*r1, end=OUT.numpy()*r2, thickness=0.05))
    ht = r * 0.7
    ht2 = r * 0.5
    c1 = Cylinder(radius=0.08 * r, height=ht).set(color=WHITE).move(UP*(ht/2+r1)).orbit_around_point(ORIGIN, 90, RIGHT)
    cone = mn.Cone(base_radius=0.2 * r, height=ht2, show_base=True, resolution=[1,10], stroke_opacity=0,
                   fill_color=mn.WHITE, direction=mn.IN).shift(mn.IN * (ht + ht2 + r1))
    cone.submobjects = [obj for obj in cone.submobjects if type(obj) != mn.VectorizedPoint]
    arr1 = Group(c1, ManimMob(cone))

    if bloom:
        s1.glow = 1

    fl = []  # magnetic field lines
    efl = []  # electric field lines
    n = 5
    sw = 8 * r
    eps = PI/n * 0.3  # little shift to stop stuff lining up
    for r3, dth in ((r5, PI/n + eps), (r4, eps)):
        th1 = math.asin(r1 / (2 * r3)) * 2
        arc = ManimMob(mn.Arc(radius=r3, stroke_color=mn.PURE_BLUE, stroke_width=sw, start_angle=th1 + PI,
              angle=2*PI-2*th1).rotate(PI/2, RIGHT.numpy(), about_point=ORIGIN.numpy()).shift(RIGHT.numpy()*r3),
                       texture_grid_size=5)
        fl += [arc.clone().orbit_around_point(ORIGIN, (2*PI*i/n + dth) * RADIANS, OUT) for i in range(n)]

    th1 = 90 - math.atan(2)*180/PI
    line = ManimMob(mn.Line(start=r1 * RIGHT, end=r_max * RIGHT, stroke_color=mn.PURE_RED, stroke_width=sw))
    for m in range(n*2):
        th2 = (m) * 180 / n
        th3 = th1 if m % 2 == 0 else -th1
        efl.append(line.clone().orbit_around_point(ORIGIN, th3, UP).orbit_around_point(ORIGIN, th2, OUT))
    efl.append(line.clone().orbit_around_point(ORIGIN, 90, UP))
    # efl.append(line.clone().orbit_around_point(ORIGIN, -90, UP))
    for _ in Group(*efl, *fl).get_descendants():
        _.set_non_recursive(color=_.color.set_opacity((
            (r_max - (_.location - ORIGIN).norm(p=2, dim=-1,
                                                keepdim=True)) / (r_max - r1)).clamp_(
            0, 1)))

    return Group(s1, arr1, *efl, *fl)

def electron(start=UP, end=RIGHT, r=1.):
    elec = create_electron(r, True)

    th1 = 20 * PI / 180
    c = math.cos(th1)
    s = math.sin(th1)
    elec_back = UP * c + RIGHT * s
    elec_up = OUT * c + DOWN * s
    elec_right = RIGHT * c + DOWN * s
    d_angle = dir_angle[end] - dir_angle[start]
    if d_angle <= -180:
        d_angle += 360
    elif d_angle > 180:
        d_angle -= 360

    with Off():
        elec.orbit_around_point(ORIGIN, th1 * RADIANS, RIGHT)
        elec.orbit_around_point(ORIGIN, dir_angle[start], elec_back)
        Scene.get_camera().set_euler_angles(90, 0, 0)
        elec.spawn()

    elec_pos = ORIGIN
    # elec_pos = IN
    # elec.move(elec_pos)

    with Sync(run_time=2, same_run_time=True, rate_func=rate_funcs.identity):
        elec.orbit_around_point(elec_pos, 360, elec_up)
        if d_angle != 0:
            with Sync(run_time=2, rate_func=rate_funcs.smooth):
                elec.orbit_around_point(elec_pos, d_angle, elec_back)

    return 'electron{}{}'.format(dir_str[start], dir_str[end])



if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    quality = HD
    bloom_new = partial(bloom_filter_premultiply, num_iterations=7, kernel_size=93, strength=15, scale_factor=6)

    for start in [UP]: # [UP, DOWN, LEFT, RIGHT]:
        for end in [UP]: # [UP, DOWN, LEFT, RIGHT]:
            name = electron(start, end, r=0.4)
            render_to_file(name, render_settings=quality, post_processes = [bloom_new], background_color=TRANSPARENT)
        # render_to_file(name, render_settings=quality, background_color=TRANSPARENT, file_extension='mov')
