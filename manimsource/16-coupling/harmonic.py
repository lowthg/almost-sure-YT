from manim import *
import numpy as np

class IntegerMeshSurface(ThreeDScene):
    def construct(self):
        # ---------------- Parameters ----------------
        grid_range = range(-4, 5)
        z_shift = 1
        scale = 0.1

        # ---------------- Camera: start top-down ----------------
        self.set_camera_orientation(
            phi=0 * DEGREES,
            theta=-90 * DEGREES
        )

        # ---------------- Integer mesh (opaque) ----------------
        mesh_lines = VGroup()
        mesh_dots = VGroup()

        for i in grid_range:
            mesh_lines.add(Line([-4, i, 0], [4, i, 0], color=GRAY))
            mesh_lines.add(Line([i, -4, 0], [i, 4, 0], color=GRAY))

            for j in grid_range:
                mesh_dots.add(Dot3D([i, j, 0], radius=0.04))

        self.add(mesh_lines, mesh_dots)

        # ---------------- Move camera ----------------
        self.move_camera(
            phi=45 * DEGREES,
            theta=-45 * DEGREES,
            run_time=3
        )

        # ---------------- Filled surface tiles ----------------
        surface = VGroup()

        def f(x, y):
            return scale * (x**2 - y**2) + z_shift

        for x in range(-4, 4):
            for y in range(-4, 4):
                p1 = np.array([x,     y,     f(x,     y)])
                p2 = np.array([x + 1, y,     f(x + 1, y)])
                p3 = np.array([x + 1, y + 1, f(x + 1, y + 1)])
                p4 = np.array([x,     y + 1, f(x,     y + 1)])

                quad = Polygon(
                    p1, p2, p3, p4,
                    fill_color=BLUE,
                    fill_opacity=0.2,
                    stroke_width=1  # no surface grid lines
                )

                surface.add(quad)

        # ---------------- Fade in surface ----------------
        self.play(FadeIn(surface), run_time=3)
        self.wait()



class InfiniteMeshSurface(ThreeDScene):
    def construct(self):
        # Parameters
        grid_range = 15
        surface_range = 8

        # Set camera to look almost straight down (phi near 90 degrees)
        # In Manim: y-axis is UP, xz-plane is horizontal
        # phi=90 is looking down, phi=0 is looking from side
        self.set_camera_orientation(phi=88 * DEGREES, theta=0)

        # Zoom out to see more
        self.camera.frame_center = ORIGIN
        self.camera.set_zoom(1.5)

        # Create grid lines in x-z plane (y=0 is horizontal)
        grid_lines = VGroup()

        # Lines parallel to x-axis (constant z)
        for z in range(-grid_range, grid_range + 1):
            line = Line(
                start=np.array([-grid_range, 0, z]),
                end=np.array([grid_range, 0, z]),
                color=BLUE,
                stroke_width=1
            )
            grid_lines.add(line)

        # Lines parallel to z-axis (constant x)
        for x in range(-grid_range, grid_range + 1):
            line = Line(
                start=np.array([x, 0, -grid_range]),
                end=np.array([x, 0, grid_range]),
                color=BLUE,
                stroke_width=1
            )
            grid_lines.add(line)

        # Dots at ALL intersections
        grid_dots = VGroup()
        for x in range(-grid_range, grid_range + 1):
            for z in range(-grid_range, grid_range + 1):
                dot = Dot3D(point=np.array([x, 0, z]), radius=0.08, color=RED)
                grid_dots.add(dot)

        # Base black mesh
        base_mesh = Surface(
            lambda u, v: np.array([u, 0, v]),
            u_range=[-grid_range, grid_range],
            v_range=[-grid_range, grid_range],
            resolution=(30, 30),
            fill_opacity=0.5,
            fill_color=BLACK,
            stroke_width=0
        )

        self.add(base_mesh, grid_lines, grid_dots)
        self.wait(1)

        # Move camera to angled position
        self.move_camera(
            phi=60 * DEGREES,
            theta=-45 * DEGREES,
            run_time=3
        )
        self.wait(0.5)

        # Create surface (starts flat at y=0)
        surface_lines_x = VGroup()
        for z in range(-surface_range, surface_range + 1):
            points = [np.array([x, 0, z]) for x in range(-surface_range, surface_range + 1)]
            line = VMobject()
            line.set_points_as_corners(points)
            line.set_color(GREEN)
            line.set_stroke(width=2)
            surface_lines_x.add(line)

        surface_lines_z = VGroup()
        for x in range(-surface_range, surface_range + 1):
            points = [np.array([x, 0, z]) for z in range(-surface_range, surface_range + 1)]
            line = VMobject()
            line.set_points_as_corners(points)
            line.set_color(GREEN)
            line.set_stroke(width=2)
            surface_lines_z.add(line)

        surface_lines = VGroup(surface_lines_x, surface_lines_z)

        # Surface dots
        surface_dots = VGroup()
        for x in range(-surface_range, surface_range + 1):
            for z in range(-surface_range, surface_range + 1):
                dot = Dot3D(point=np.array([x, 0, z]), radius=0.07, color=YELLOW)
                surface_dots.add(dot)

        # Filled surface
        surface_mesh = Surface(
            lambda u, v: np.array([u, 0, v]),
            u_range=[-surface_range, surface_range],
            v_range=[-surface_range, surface_range],
            resolution=(16, 16),
            fill_opacity=0.2,
            fill_color=GREEN,
            stroke_width=0
        )

        self.add(surface_mesh, surface_lines, surface_dots)

        # Target: saddle surface y = 0.1*(x^2 - z^2) + 1.5
        def saddle(u, v):
            return np.array([u, 0.1 * (u ** 2 - v ** 2) + 1.5, v])

        target_mesh = Surface(
            saddle,
            u_range=[-surface_range, surface_range],
            v_range=[-surface_range, surface_range],
            resolution=(16, 16),
            fill_opacity=0.2,
            fill_color=GREEN,
            stroke_width=0
        )

        target_lines_x = VGroup()
        for z in range(-surface_range, surface_range + 1):
            points = [saddle(x, z) for x in range(-surface_range, surface_range + 1)]
            line = VMobject()
            line.set_points_as_corners(points)
            line.set_color(GREEN)
            line.set_stroke(width=2)
            target_lines_x.add(line)

        target_lines_z = VGroup()
        for x in range(-surface_range, surface_range + 1):
            points = [saddle(x, z) for z in range(-surface_range, surface_range + 1)]
            line = VMobject()
            line.set_points_as_corners(points)
            line.set_color(GREEN)
            line.set_stroke(width=2)
            target_lines_z.add(line)

        target_lines = VGroup(target_lines_x, target_lines_z)

        target_dots = VGroup()
        for x in range(-surface_range, surface_range + 1):
            for z in range(-surface_range, surface_range + 1):
                dot = Dot3D(point=saddle(x, z), radius=0.07, color=YELLOW)
                target_dots.add(dot)

        # Animate transformation
        self.play(
            Transform(surface_mesh, target_mesh),
            Transform(surface_lines, target_lines),
            Transform(surface_dots, target_dots),
            run_time=2,
            rate_func=smooth
        )
        self.wait(0.5)

        # 360 rotation
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(2 * PI / 0.3)
        self.stop_ambient_camera_rotation()

        self.wait(1)