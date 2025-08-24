from algan import *
import manim as mn
import math
from algan.rendering.post_processing import bloom_filter, bloom_filter_premultiply
from functools import partial


def electron():
    r = 1
    r1 = r * 0.5
    r2 = r * 1.5
    s1 = ManimMob(mn.Sphere(radius=r1, resolution=[10, 10], checkerboard_colors=[mn.WHITE, mn.YELLOW]))
    # arr1 = ManimMob(mn.Arrow3D(start=OUT.numpy()*r1, end=OUT.numpy()*r2, thickness=0.05))

    fl = []
    n = 5
    sw = 5
    so = 0.15
    # so=1
    for r3, dth in ((1.5, PI/n), (0.8, 0)):
        th1 = math.asin(r1 / (2 * r3)) * 2
        arc = ManimMob(mn.Arc(radius=r3, stroke_color=mn.BLUE, stroke_width=sw, stroke_opacity=so, start_angle=th1 + PI,
              angle=2*PI-2*th1).rotate(PI/2, RIGHT.numpy(), about_point=ORIGIN.numpy()).shift(RIGHT.numpy()*r3),
                       texture_grid_size=5)
        fl += [arc.clone().orbit_around_point(ORIGIN, (2*PI*i/n + dth) * RADIANS, OUT) for i in range(n)]
        fl.append(ManimMob(mn.Line(IN.numpy()*r1, IN.numpy()*r2, stroke_width=sw, stroke_color=mn.BLUE, stroke_opacity=so)))

    with Off():
        Scene.get_camera().set_euler_angles(70, 0, 20)

    elec = Group(*fl, s1)

    with Off():
        # for _ in elec.get_descendants():
        #     _.set_non_recursive(color=_.color.set_opacity((
        #         0.7 * (1 - (_.location - ORIGIN).norm(p=2, dim=-1, keepdim=True) / 2)).clamp(min=0.1)))
        elec.spawn()

    s1.glow = 1

    with Sync(run_time=2, same_run_time=True, rate_func=rate_funcs.identity):
        elec.orbit_around_point(ORIGIN, 360, OUT)
        # with Seq():
        #     elec.wait(1)
        #     elec.orbit_around_point(ORIGIN, 90, UP)


if __name__ == "__main__":
    electron()
    # COMPUTING_DEFAULTS.render_device = torch.device('cpu')
    COMPUTING_DEFAULTS.render_device = torch.device('mps')
    COMPUTING_DEFAULTS.max_animate_batch_size = 10
    LOGGING_DEFAULTS.verbosity = 'max'
    # COMPUTING_DEFAULTS.portion_of_memory_used_for_animating = 0.05
    # COMPUTING_DEFAULTS.portion_of_memory_used_for_rendering = 0.5
    render_to_file('electron', render_settings=MD)
    # render_to_file('electron3', render_settings=MD, background_color=TRANSPARENT, file_extension='mov')
    #render_to_file('electron', post_processes=[partial(bloom_filter, strength=4, scale_factor=32)],
    #               background_color=TRANSPARENT, file_extension='mov')
    # render_to_file('electron4', post_processes=[
    #    partial(bloom_filter, num_iterations=3, kernel_size=31, strength=4, scale_factor=32)],
    #               background_color=TRANSPARENT, file_extension='mov')  # bloom_filter_premultiply
