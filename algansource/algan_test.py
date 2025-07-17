from algan import *
import numpy as np


def test1():
    my_first_mob = Text('Hello World!', font_size=100)
    my_first_mob.spawn()
    return 'test1'

def test2():
    circle = Circle().spawn()

    circle.location = circle.location + UP
    circle.opacity = 0.5
    circle.location = circle.location + DOWN + RIGHT
    circle.glow = 0.5
    circle.color = GREEN

    return 'test2'

def test3():
    # Define a function mapping a scalar parameter t to a point on the circle.
    def path_func(t):
        return (UP * np.sin(t) + RIGHT * np.cos(t))

    # Create an animated_function which will move our mob along this path.
    @animated_function(animated_args={'t': 0})
    def move_along_path(mob, t):
        mob.location = path_func(t)

    square = Square().spawn()
    square.location = path_func(0)  # Move to starting point.
    move_along_path(square, 2 * PI)
    return 'test3'

def test4():
    mob1 = Square().spawn()
    mob2 = Circle().spawn()

    with Sync(run_time=4):
        mob1.rotate(360, OUT)
        mob2.move(RIGHT)

    with Seq():
        mob1.move(RIGHT)
        mob2.move(UP)

    with Lag(0.5):
        mob1.move_to(ORIGIN)
        mob2.move_to(ORIGIN)

    with Off():
        mob1.move(LEFT)
        mob2.move(RIGHT)

    return 'test4'

def test5():
    Sphere(radius=1)
    return 'test5'

fn = test5()
render_to_file(file_name=fn, render_settings=LD)