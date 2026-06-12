from manim import *
#import cv2
import math
#import abracadabra as abra

class zpowz(Scene):
    def construct(self):
        def f(t):
            s = np.sin(t)
            c = np.cos(t)
            a = np.exp(-t*s)
            return np.cos(t*c)*a, np.sin(t*c)*a

        aspect = 9/16
        xlen = config.frame_x_radius * 1.8
        #ylen = config.frame_y_radius * 1.8
        ylen = xlen * 1.2

        rmax = config.frame_x_radius * math.sqrt(1 + 1/(aspect*aspect))
        rmin = 0.01 * config.frame_y_radius

        print(xlen)
        print(ylen)

        xmax = 2
        ymax = xmax * ylen/xlen
        ax = Axes(x_range=[-xmax, xmax], y_range=[-ymax, ymax], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(200)

        self.add(ax)
        pt0 = ax.coords_to_point(0, 0)
        pt1 = ax.coords_to_point(1, 0)

        radii = [1., 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
        rstr = [r'1', r'10', r'100', r'1\,000', r'10\,000', r'100\,000', r'1\,000\,000', r'10\,000\,000', r'100\,000\,000']
        for i in range(1, 11):
            radii.append(math.pow(10, -i))
            j = (i-1) // 3
            k = (i-1) - 3*j
            rstr.append('0.' + '000\,' * j + '0' * k + '1')
        eqs = [MathTex(_, font_size=90) for _ in rstr]

        tval = ValueTracker(0.0)
        scale=[1.0]
        dot1 = Dot(radius=0.2, color=RED, stroke_opacity=1)
        minmax = [1., 0.]

        def draw_func():
            t = tval.get_value()
            t0 = 0.0
            res = VGroup()
            if t > 0.001:
                nt = max(int((t-t0) * 1000 / PI), 100)
                times = np.linspace(t0, t, nt)
                xvals, yvals = f(times)
                xmax1 = abs(xvals[-1])
                ymax1 = abs(yvals[-1])
                if xmax1 * scale[0] > xmax:
                    scale[0] = xmax/xmax1
                if ymax1 * scale[0] > ymax:
                    scale[0] = ymax / ymax1

                eps = 0.1
                if xmax1 > ymax1:
                    if eps > xmax1 * scale[0]:
                        scale[0] = eps/xmax1
                else:
                    if eps > ymax1 * scale[0]:
                        scale[0] = eps/ymax1

                xvals = (xvals*scale[0]).clip(-10, 10)
                yvals = (yvals*scale[0]).clip(-10, 10)

                minmax[0] = min(minmax[0], scale[0])
                minmax[1] = max(minmax[1], scale[0])

                assert len(xvals) > 0

                crv = ax.plot_line_graph(xvals, yvals, line_color=YELLOW, stroke_width=4, add_vertex_dots=False)
                ptdot = crv['line_graph'].get_end()
                res += crv.set_z_index(10)
            else:
                ptdot = RIGHT * (pt1[0] - pt0[0])

            res += dot1.copy().move_to(ptdot).set_z_index(20)

            for i, r in enumerate(radii):
                r2 = (pt1[0] - pt0[0])*scale[0] * r
                if rmin < r2 < rmax:
                    circ = Circle(radius=r2, fill_opacity=0, stroke_color=WHITE, stroke_opacity=0.7,
                                  stroke_width=2).move_to(pt0)
                    res += circ
                    eq = eqs[i].copy().scale(r*scale[0]).move_to(r2*RIGHT + 0.2*DR, aligned_edge=UL)
                    res += eq

            return res

        obj = always_redraw(draw_func)
        self.add(obj)
        self.wait(0.1)
        tmax = PI*8
        run_time=tmax
        self.play(tval.animate(rate_func=linear).set_value(tmax), run_time=run_time)
        print('minmax', minmax)
        self.wait()


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "fps": 15, "preview": True}):
        zpowz().render()