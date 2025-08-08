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
    LEFT: -90
}

# def generate_checkerboard
def create_electron(r=1, bloom=True, sub_n=5) -> tuple[Mob, Mob]:
    r1 = r * 0.5
    r2 = r * 1.5
    r4 = r * 0.8
    r5 = r * 1.5
    r_max = 2.5 * r
    s0 = mn.Sphere(radius=r1, stroke_opacity=1, resolution=[8 * sub_n, 4 * sub_n], checkerboard_colors=[mn.WHITE, mn.YELLOW])
    for i, x in enumerate(s0.submobjects):
        k, j = divmod(i, 4 * sub_n)
        if (k // sub_n) % 2 == (j // sub_n) % 2:
            x.set(color=mn.YELLOW)
        else:
            x.set(color=mn.WHITE)
    s1 = ManimMob(s0)
    # s1_1 = Sphere(radius=r1, grid_height=20)
    # s1 = ImageMob('WorldMap.png').spawn()
    # s1.set_shape_to(s1_1)
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
        arc = ManimMob(mn.Arc(radius=r3, stroke_color=mn.BLUE_D, stroke_width=sw, start_angle=th1 + PI,
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

    return Group(s1, arr1), Group(*efl, *fl)

def electron(start=UP, end=RIGHT, r=1., show_field=False, sub_n=5):
    bare, field = create_electron(r, sub_n=sub_n)
    dressed = Group(bare, field)

    if (start == DOWN).all() and (end == UP).all():
        d_angle = -180
    else:
        d_angle = dir_angle[end] - dir_angle[start]
        if d_angle < -180:
            d_angle += 360
        elif d_angle > 180:
            d_angle -= 360

    th1 = 20 * PI / 180

    with Off():
        dressed.orbit_around_point(ORIGIN, th1 * RADIANS, RIGHT)
        elec_back = bare.get_upwards_direction()
        dressed.orbit_around_point(ORIGIN, dir_angle[start], elec_back)
        Scene.get_camera().set_euler_angles(90, 0, 0)
        bare.spawn()
        if show_field:
            field.spawn()

    elec_pos = ORIGIN

    elec = Group(bare, field) if show_field else bare

    elec_up = -bare.get_forward_direction()
    with Sync(run_time=2, same_run_time=True, rate_func=rate_funcs.identity):
        elec.orbit_around_point(elec_pos, 360, elec_up)
        if d_angle != 0:
            with Sync(run_time=2, rate_func=rate_funcs.smooth):
                elec.orbit_around_point(elec_pos, d_angle, elec_back)

    tag = '' if show_field else 'NF'
    tag2 = 'B' if big else ''
    return 'electron{}{}{}{}'.format(tag2, tag, dir_str[start], dir_str[end])



if __name__ == "__main__":
    COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.max_cpu_memory_used *= 6
    quality = HD
    show_field = False
    bgcol = TRANSPARENT
    big = False
    sub_n=5

    r = 1. if big else 0.5

    kernel_size = int(93/0.4 * r)
    if bgcol.tolist()[-1] < 1:
        print('transparent')
        print('kernel_size', kernel_size)
        bloom_new = partial(bloom_filter_premultiply, num_iterations=7, kernel_size=kernel_size, strength=15, scale_factor=6)
    else:
        bloom_new = partial(bloom_filter, num_iterations=7, kernel_size=kernel_size, strength=15, scale_factor=6)

    # orient = [(UP, DOWN), (DOWN, UP), (UP, UP), (DOWN, DOWN)]
    orient = [(UP, UP), (DOWN, DOWN), (UP, DOWN), (DOWN, UP)]
    orient += [(LEFT, LEFT), (RIGHT, LEFT), (UP, RIGHT), (RIGHT, RIGHT)]

    for start, end in orient:
        name = electron(start, end, r=r, show_field=show_field, sub_n=5)
        render_to_file(name, render_settings=quality, post_processes = [bloom_new], background_color=bgcol)
        # render_to_file(name, render_settings=quality, background_color=TRANSPARENT, file_extension='mov')
