from algan import *
import manim as mn

import math

def electron():
    r = 1
    r1 = r * 0.5
    r2 = r * 1.5
    s1 = ManimMob(mn.Sphere(radius=r1, resolution=[10, 10], checkerboard_colors=[mn.WHITE, mn.YELLOW]))
    arr1 = Cylinder(radius=0.05, color=RED).move_between_points(IN*r1, IN*r2)
    fl = []
    n = 5
    sw = 5
    so = 0.15
    for r3, dth in ((1.5, PI/n), (0.8, 0)):
        th1 = 2 * math.asin(r1 / (2 * r3))
        arc = ManimMob(mn.Arc(radius=r3, stroke_color=mn.BLUE, stroke_width=sw, stroke_opacity=so, start_angle=th1 + PI,
              angle=2*PI-2*th1).rotate(PI/2, mn.RIGHT, about_point=mn.ORIGIN).shift(mn.RIGHT*r3)).set(opacity=1, max_opacity=1)
        rotate_center = Mob(location=ORIGIN)
        rotate_center.add_children(arc)
        arc = rotate_center
        fl += [arc.clone().rotate((2*PI*i/n + dth) / mn.DEGREES, OUT) for i in range(n)]
        fl.append(ManimMob(mn.Line(mn.IN*r1, mn.IN*r2, stroke_width=sw, stroke_color=mn.BLUE, stroke_opacity=so,
                                   fill_opacity=0)).set(opacity=1, max_opacity=1))

    camera = Scene.get_camera()

    with (Off()):
        camera.orbit_around_point(ORIGIN, (180+70), RIGHT)

    #circle = Circle().spawn()

    #circle.location = circle.location + UP

    with Off():
        # gp = Group([s1, arr1])
        gp = fl[0]
        rotate_center = Mob(location=ORIGIN)
        rotate_center.add_children(gp)
        gp = rotate_center
        gp.spawn()



    # with Seq(run_time=5):
    with Off():
        s1.glow = 1
        # with Sync():
        #    [_.set(glow=1) for _ in fl]

    with Sync(same_run_time=True, rate_func=rate_funcs.identity):
        gp.rotate(360, OUT)
        with Seq(rate_func=rate_funcs.identity):
            gp.wait(1)
            gp.rotate(180, LEFT)


if __name__ == "__main__":
    electron()
    render_to_file('electron')
