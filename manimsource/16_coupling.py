from fontTools.unicodedata import block
from manim import *
import numpy as np
import math
import sys
import scipy as sp
from matplotlib.font_manager import font_scalings
from numpy.random.mtrand import Sequence
from sorcery import switch
from torch.utils.jit.log_extract import run_test

sys.path.append('../')
import manimhelper as mh

class Percolation(Scene):
    def construct(self):
        h = config.frame_y_radius * 2
        nx = 24
        ny = 40
        side=h / ny
        np.random.seed(3)
        vals = []
        rows = []
        for i in range(ny):
            vals.append(np.random.uniform(0., 1., size=nx))
            sq = []
            for j in range(nx):
                sq.append(Square(side_length=side, stroke_width=0.8, stroke_color=GREY,
                            fill_color=ORANGE, fill_opacity=0))
            rows.append(VGroup(*sq).arrange(RIGHT, buff=0))
        grid = VGroup(*rows).arrange(DOWN, buff=0).to_edge(LEFT, buff=0)

        txt1 = (Tex('\sf percolation', color=RED, stroke_width=2, font_size=80)
                .move_to(grid).align_to(grid, DOWN).shift(UP*0.1).set_z_index(4))
        txt2 = txt1.copy().set_color(BLACK).set_stroke(width=16).set_z_index(3)

        self.add(grid, txt1, txt2)

        lim_p = 0.436
        blocked = [[vals[i][j] < lim_p for j in range(nx)] for i in range(ny)]

        run_time=1.2
        def update_grid(grid, dt):
            t = self.time/run_time
            for i in range(ny):
                for j in range(nx):
                    if blocked[i][j]:
                        val = vals[i][j] / lim_p
                        op = min(max(2. * t / val - 1, 0.), 1.)
                        grid[i][j].set_fill(opacity=op)

        grid.add_updater(update_grid)
        self.wait(run_time)
        grid.remove_updater(update_grid)

        fill_max = 1000
        filled = [[fill_max] * nx for _ in range(ny)]
        for i in range(nx):
            if not blocked[0][i]:
                filled[0][i] = 1
        #filled[0][9] = 1

        extends = True
        count = 0
        max_fill=1
        while extends:
            count += 1
            print('iters: ', count)
            extends = False
            for i in range(1, ny):
                for j in range(nx):
                    if (not blocked[i][j]) and filled[i][j] >= fill_max:
                        val = filled[i-1][j]
                        if i < ny - 2:
                            val = min(filled[i + 1][j], val)
                        if j > 0:
                            val = min(filled[i][j-1], val)
                        if j < nx-1:
                            val = min(filled[i][j+1], val)
                        val += 1
                        if val < fill_max:
                            filled[i][j] = val
                            max_fill = max(max_fill, val)
                            extends = True

        print('max fill:', max_fill)
        col = ManimColor((BLUE_E.to_rgb()+BLUE_D.to_rgb())/2)
        for i in range(ny):
            for j in range(nx):
                if not blocked[i][j]:
                    grid[i][j].set_fill(color=col)

        run_time=1.5
        t0 = self.time
        def update_grid2(grid, dt):
            t = (self.time - t0)/run_time
            for i in range(ny):
                for j in range(nx):
                    if not blocked[i][j]:
                        val = filled[i][j] / max_fill
                        op = min(max(5. * t / val - 4, 0.), 1.)
                        grid[i][j].set_fill(opacity=op)

        grid.add_updater(update_grid2)
        self.wait(run_time)
        grid.remove_updater(update_grid2)

        self.wait()

def arc_center(start, end, angle):
    dir1 = (end - start) / 2
    pt = (start + end) / 2 + np.array([dir1[1], -dir1[0], 0.]) / math.tan(angle / 2)
    return pt


def arc_arrow(start, end, angle, buff=0., center=None, **kwargs):
    end_buff = kwargs.pop('end_buff') if 'end_buff' in kwargs else buff
    pt = arc_center(start, end, angle) if center is None else center

    radius = float(np.linalg.norm(start - pt))
    angle_buff = 2 * np.arcsin(buff / radius / 2)
    angle_end = 2 * np.arcsin(end_buff / radius / 2)
    angle -= angle_buff + angle_end
    arc = Arc(radius=radius, start_angle=PI / 2, angle=-angle, arc_center=pt, **kwargs)
    rot = angle_of_vector(start - pt) - angle_of_vector(arc.get_start() - pt)
    return arc.rotate(rot - angle_buff, about_point=pt).add_tip()


class Markov(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        dot = Dot(radius=0.5, fill_opacity=0, stroke_color=PURPLE, stroke_width=10).set_z_index(4)
        # x^2=1/4+(1-x)^2 -> 0=1/4+1-2x -> x=5/8
        shift1 = UP*math.sqrt(1/3) * 2
        shift2 = (RIGHT * 1/2 + DOWN * math.sqrt(1/12)) * 2
        shift3 = shift2 * UL
        dots = VGroup(
            dot.copy().shift(shift1).set_stroke(color=BLUE),
            dot.copy().shift(shift2).set_stroke(color=PURPLE),
            #dot.copy().shift(shift3).set_stroke(color=TEAL),
        )
        ndots = len(dots)
        pos = [_.get_center() for _ in dots[:]]

        kwargs = {'stroke_color': RED, 'buff': 0.5, 'end_buff': 0.6, 'stroke_width': 14, 'tip_length': 0.45}
        arr_angle = PI*0.8
        arrs = [arc_arrow(start, end, arr_angle, **kwargs)
                for start, end in [(pos[0], pos[1]), (pos[1], pos[0])]]
        center_stay = [pos[0]+UP*0.48, pos[1]-UP*0.48]
        arrs.append(arc_arrow(pos[0], pos[0], angle=PI*1.999, center=center_stay[0], **kwargs))
        arrs.append(arc_arrow(pos[1], pos[1], angle=PI*1.999, center=center_stay[1], **kwargs))
        kwargs = {'stroke_color': BLACK, 'buff': 0.5, 'end_buff': 0.5, 'stroke_width': 22, 'tip_length': 0.55}
        arrs2 = [arc_arrow(start, end, PI*0.8, **kwargs)
                for start, end in [(pos[0], pos[1]), (pos[1], pos[0])]]
        arrs2.append(arc_arrow(pos[0], pos[0], angle=PI*1.999, center=center_stay[0], **kwargs))
        arrs2.append(arc_arrow(pos[1], pos[1], angle=PI*1.999, center=center_stay[1], **kwargs))

        #arrs2 = [_.copy().remove_tip().set_stroke(width=18, color=BLACK).add_tip() for _ in arrs]
        for _ in arrs:
            _.set_z_index(0.5)

        dots_stay = VGroup(*[Dot().move_to(center_stay[i]) for i in [0,1]])
        VGroup(*arrs, *arrs2, dots, dots_stay).rotate(51*DEGREES)
        pos = [_.get_center() for _ in dots[:]]
        center_stay = [_.get_center() for _ in dots_stay[:]]

        txt = MathTex(r'A', r'B', font_size=60, stroke_width=4, color=YELLOW).set_z_index(3)
        txt[0].move_to(pos[0]).shift(UL*0.03)
        txt[1].move_to(pos[1])

        fill_col = ManimColor((0.4, 0., 0.))
        place = Dot(radius=0.5, fill_opacity=1, fill_color=fill_col, stroke_opacity=0, stroke_width=0).set_z_index(1)
        place.move_to(pos[0])

        self.add(dots, *arrs, txt, place)
        self.wait(0.1)
        state = 0
        center_move = [arc_center(pos[0], pos[1], arr_angle), arc_center(pos[1], pos[0], arr_angle)]
        switches = [True, False, True, False, True, True, True, False, False, True, True, True]
        for switch in switches:
            self.wait(0.2)
            if switch:
                self.play(Rotate(place, -arr_angle, about_point=center_move[state]))
                state = 1 - state
            else:
                self.play(Rotate(place, -2*PI, about_point=center_stay[state]))
            self.wait(0.2)

        self.wait()

class RenewalProc(Scene):
    def __init__(self, *args, **kwargs):
        if not config.transparent:
            config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        height = 3.
        ax = Axes(x_range=[0, 1.1], y_range=[0, height * 1.2], x_length=4.8, y_length=3.6,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  #                  shade_in_3d=True,
                  ).set_z_index(1)
        box = SurroundingRectangle(ax, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                   fill_opacity=0.6, corner_radius=0.15)
        VGroup(ax, box).to_corner(UR, buff=0.1)

        txt1 = Tex('\sf renewal', 'process', color=RED, stroke_width=2, font_size=80).set_z_index(0.5)
        txt1[1].next_to(txt1[0], DOWN, buff=0.5)
        txt1.move_to(ax).shift(UP*0.35+LEFT*0.8).set_z_index(0.5).set_opacity(0.8)

        self.add(ax, box, txt1)

        times = [0., 0.4, 0.7, 0.9, 1.14]
        lines = VGroup()
        dashed = VGroup()
        pts = []
        pts0 = []
        for i in range(1, len(times)):
            pt0 = ax.coords_to_point(times[i-1], i-1)
            pt1 = ax.coords_to_point(times[i], i-1)
            lines.add(Line(pt0, pt1+LEFT*0.07, stroke_width=6, stroke_color=BLUE).set_z_index(2))
            if i > 1:
                dashed.add(DashedLine(pts[i-2]+UP*0.07, pt0, stroke_width=4, stroke_color=BLUE).set_z_index(2))
            pts.append(pt1)
            pts0.append(pt0)

        speed = 0.5
        dot = Dot(radius=0.1, color=YELLOW).move_to(pts0[0]).set_z_index(4).set_opacity(0)
        dot1 = Dot(radius=0.1, color=BLUE).set_z_index(3)
        dot2 = Dot(radius=0.08, color=BLUE, fill_opacity=0, stroke_opacity=1, stroke_width=5).set_z_index(3)
        for i in range(len(lines)):
            dt = times[i+1] - times[i]
            op = 3.
            if i > 0:
                self.add(dot1.copy().move_to(pts0[i]))
            if i >= len(lines)-1:
                op = 0.
            self.play(Create(lines[i], rate_func=linear),
                      dot.animate(rate_func=linear).move_to(pts[i]).set_opacity(op),
                      run_time= dt /speed)
            if i < len(lines)-1:
                self.add(dot2.copy().move_to(pts[i]))
                self.play(Create(dashed[i], rate_func=linear),
                          dot.animate(rate_func=linear).move_to(pts0[i+1]),
                          run_time=0.1 / speed)

        self.wait()

class DigitGame(Scene):
    def __init__(self, *args, **kwargs):
        #if not config.transparent:
        #    config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        data = [[7, 5, 1, 8, 7, 8, 2, 9, 7, 7, 7, 9, 8, 4, 2, 6, 4, 3, 0, 7, 5, 5, 9, 6, 6, 8, 0, 5, 8, 1, 2, 7, 0, 8, 3, 1, 0, 3, 2, 3, 2, 9, 2, 1, 0, 1, 9, 9, 0, 7, 2, 6, 3, 7, 8, 2, 6, 7, 7, 9, 2, 7, 5, 3, 0, 5, 4, 3, 8, 9, 5, 8, 6, 7, 0, 4, 6, 8, 4, 8, 1, 9, 5, 4, 2, 3, 4, 7, 6, 6, 8, 0, 3, 5, 6, 1, 1, 6, 0, 8], [8, 15, 21, 26, 36, 46, 55, 57, 64, 74, 84, 86, 90, 98], [3, 11, 20, 25, 33, 41, 50, 52, 55, 57, 64, 74, 84, 86, 90, 98]]
        digits, alice_steps, bob_steps = data
        digits[alice_steps[-1]] = last_digit=7
        fs1 = 60

        eq_digits = MathTex(''.join([str(_) + r'\,' for _ in digits]), stroke_width=1.5, font_size=fs1)[0].set_z_index(2)
        eq_digits.set_color(BLUE)
        num_digits = len(digits)
        print(num_digits)
        row_len = 20
        num_rows = num_digits // row_len
        h = eq_digits[0].height * 2.5
        print(num_rows)
        print(h)

        for i in range(1, num_rows):
            pos = i * row_len
            print(pos)
            eq_digits[pos:].move_to(eq_digits[pos-1]).shift(h * DOWN).align_to(eq_digits[0], LEFT)

        eq_digits.move_to(ORIGIN)

        self.wait(0.1)

        alice_scale = 1.1
        bob_scale = alice_scale * .43/.49 * 499/543
        alice = ImageMobject(r'../media/wifejak.png').set_z_index(6).scale(alice_scale)
        a = alice.pixel_array.copy()
        alice = ImageMobject(a[:-82,:,:]).set_z_index(6).scale(alice_scale)

        bob = ImageMobject(r'../media/husbandjak.png').set_z_index(6).scale(bob_scale)
        alice.to_edge(DL, buff=0).shift(RIGHT*0.5)# + 92/1080 * config.frame_height * alice_scale * DOWN)
        bob.to_edge(DR, buff=0).shift(RIGHT*0.1)
        alice.generate_target()
        bob.generate_target()
        alice.shift(LEFT*3.2)
        bob.shift(RIGHT*3.2)
        self.play(MoveToTarget(alice), MoveToTarget(bob), rate_func=rate_functions.ease_out_cubic)

        eq_digits.set_opacity(0)
        t_val = ValueTracker(0.)
        def f(obj):
            i = math.floor(t_val.get_value() * num_digits)
            obj[:i+1].set_opacity(1)

        eq_digits.add_updater(f)
        self.wait(0.1)
        self.play(t_val.animate.set_value(1), eq_digits.animate.shift(ORIGIN),
                                 run_time=3,
                  rate_func=linear)
        eq_digits.remove_updater(f)

        # alice stepping
        eq0 = MathTex(r'8', font_size=fs1)
        box1 = SurroundingRectangle(eq0, stroke_opacity=0, stroke_width=0, fill_color=RED, fill_opacity=1,
                                    corner_radius=0.1, buff=0.05).set_z_index(1)
        j = alice_steps[0]
        box1.move_to(eq_digits[j])
        alice_small = alice.copy().scale(0.2).next_to(box1, UP, buff=0.1).set_z_index(2.5)
        self.play(FadeIn(box1, alice_small), eq_digits[j].animate.set_color(WHITE))
        num_alice = len(alice_steps)

        self.wait(0.1)
        box2 = box1.copy().set_fill(opacity=0).set_stroke(width=6, color=RED, opacity=1)
        alice_boxes = [box1]
        first_10 = True
        for i in range(1, num_alice):
            j0 = j
            j = alice_steps[i]
            box = box2.copy().move_to(eq_digits[j0])
            box.generate_target().move_to(eq_digits[j])
            alice_small.generate_target().next_to(box.target, UP, buff=0.1)
            dt = max(np.linalg.norm(box.target.get_center() - box.get_center())*0.3, 0.8)
            rate_func=smooth
            if i == 1:
                rate_func=linear
                dt *= 1.5
            if first_10 and digits[j0] == 0:
                rate_func=linear
                dt *= 1.5
                first_10 = False

            self.play(MoveToTarget(box), MoveToTarget(alice_small), run_time=dt, rate_func=rate_func)
            alice_boxes.append(box)

        eq_alice1 = Tex(r"\sf Alice's", r' score', r'${}$'.format(last_digit), color=RED, font_size=80).set_z_index(4)
        eq_alice1.set_stroke(width=2)
        eq_alice1[1].next_to(eq_alice1[0], DOWN, buff=0.3)
        eq_alice1[2].next_to(eq_alice1[:2], DOWN).set_color(YELLOW)
        eq_alice1.next_to(bob, UP).next_to(eq_digits, LEFT, buff=0.3, coor_mask=RIGHT)
        eq_alice2 = eq_alice1[:2].copy().set_color(WHITE).set_stroke(width=4).set_z_index(3.9)

        eq_bob1 = Tex(r"\sf Bob's", r' score', r'${}$'.format(last_digit), color=GREEN, font_size=80).set_z_index(4)
        eq_bob1.set_stroke(width=2)
        eq_bob1[1].next_to(eq_bob1[0], DOWN)
        eq_bob1[2].next_to(eq_bob1[:2], DOWN).set_color(YELLOW)
        eq_bob1.move_to(-eq_alice1.get_center())
        eq_bob1[0].align_to(eq_alice1[0], DOWN)
        eq_bob1[1].align_to(eq_alice1[1], DOWN)
        eq_bob1[2].align_to(eq_alice1[2], DOWN)
        eq_bob2 = eq_bob1[:2].copy().set_color(WHITE).set_stroke(width=4).set_z_index(3.9)

        self.wait(0.1)
        self.play(Succession(Wait(1), FadeIn(eq_alice1[:2], eq_alice2, run_time=1)),
                  mh.rtransform(eq_digits[j].copy().set_z_index(4), eq_alice1[2][0]),
                  run_time=2)
        self.wait(0.1)

        # discussion
        wid = mh.pos(RIGHT)[0]*2
        hei = mh.pos(UP)[1]*2
        box5 = Rectangle(width=wid, height=hei, stroke_width=0, stroke_opacity=0,
                                    fill_color=BLACK, fill_opacity=0.8).set_z_index(2.5)
        box6 = SurroundingRectangle(eq_alice1, stroke_width=0, stroke_opacity=0,
                                    fill_color=BLACK, fill_opacity=0.8, buff=0.1).set_z_index(5)
        self.play(FadeIn(box5, rate_func=linear))
        self.wait(1)
        self.play(FadeIn(box6, rate_func=linear))
        self.wait(1)
        self.play(FadeOut(box5, box6, rate_func=linear))
        self.wait(0.1)

        # bob stepping
        j = bob_steps[0]
        box3 = box1.copy().set_fill(color=GREEN).move_to(eq_digits[j])
        b = bob.pixel_array.copy()
        bob_small = ImageMobject(b[:,-50:50:-1,:]).set_z_index(2.4).scale(bob_scale * 0.2)
        bob_small.next_to(box3, UP, buff=0.1)
        self.play(FadeIn(box3, bob_small), eq_digits[j].animate.set_color(WHITE))
        num_bob = len(bob_steps)
        self.wait(0.1)
        box4 = box2.copy().set_stroke(color=GREEN).set_z_index(0.9)

        for i in range(1, num_bob):
            j0 = j
            j = bob_steps[i]
            box = box4.copy().move_to(eq_digits[j0])
            box.generate_target().move_to(eq_digits[j])
            bob_small.generate_target().next_to(box.target, UP, buff=0.1)
            anims = [MoveToTarget(box)]
            if i == num_bob - 1:
                bob_small.target.shift(alice_small.width*0.2*RIGHT)
                anims.append(alice_small.animate(rate_func=rush_into).shift(alice_small.width*0.2*LEFT))
            anims.append(MoveToTarget(bob_small))
            if j in alice_steps:
                k = alice_steps.index(j)
                anims.append(alice_boxes[k].animate(rate_func=rush_into).set_stroke(color=PINK))
            dt = max(np.linalg.norm(box.target.get_center() - box.get_center())*0.3, 0.8)
            self.play(*anims, run_time=dt)

        self.wait(0.1)
        self.play(Succession(Wait(1), FadeIn(eq_bob1[:2], eq_bob2, run_time=1)),
                  mh.rtransform(eq_digits[j].copy().set_z_index(4), eq_bob1[2][0]),
                  run_time=2)

        self.wait()

class AliceBobEqual(Scene):
    fs2 = 70
    bg_color = GREY
    def __init__(self, *args, **kwargs):
        if not config.transparent:
           config.background_color = self.bg_color
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        wid = mh.pos(RIGHT)[0]*2
        hei = mh.pos(UP)[1]*2
        fs2 = self.fs2
        eq1 = MathTex(r'\mathbb P({\sf Alice} > {\sf Bob})', r'=', r'\mathbb P({\sf Bob} > {\sf Alice})',
                     font_size=fs2, stroke_width=2).set_z_index(2.9)
        eq3 = MathTex(r'\mathbb P({\sf Alice} = {\sf Bob})', r'=', r'10\%', r'\,?', stroke_width=2,
                      font_size=fs2).set_z_index(2.9)
        eq1.shift(124/1920 * wid * RIGHT + 23/1080 * hei * UP)
        eq2 = eq1.copy().set_color(BLACK).set_z_index(2.8).set_stroke(width=20)
        VGroup(eq1[0][2:7], eq1[2][-6:-1], eq3[0][2:7]).set_color(RED)
        VGroup(eq1[2][2:5], eq1[0][-4:-1], eq3[0][8:11]).set_color(GREEN)
        VGroup(eq1[0][0], eq1[2][0], eq3[0][0]).set_color(YELLOW)
        VGroup(eq3[2][:-1]).set_color(BLUE)
        eq3[3].set_color(PURE_RED)

        mh.align_sub(eq3, eq3[1], eq1[1]).align_to(eq1, LEFT).shift(RIGHT)
        eq4 = eq3.copy().set_z_index(2.8).set_color(BLACK).set_stroke(width=20)

        self.play(FadeIn(eq1, eq2, rate_func=linear))
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0][:7], eq3[0][:7], eq1[0][8:], eq3[0][8:]),
                  FadeOut(eq1[2], eq1[1]),
                  mh.fade_replace(eq1[0][7], eq3[0][7]),
                  mh.rtransform(eq2[0][:7], eq4[0][:7], eq2[0][8:], eq4[0][8:]),
                  FadeOut(eq2[2], eq2[1]),
                  mh.fade_replace(eq2[0][7], eq4[0][7])
                  )
        self.wait(0.1)
        self.play(FadeIn(eq3[1:], eq4[1:]))

        self.wait(0.1)
        self.play(FadeOut(eq3, eq4))

def copy_colors(destination, source):
    assert len(source[:]) == len(destination[:])
    for i, x in enumerate(destination[:]):
        x.set_color(source[i].color)

class AliceBobDraw(AliceBobEqual):
    bg_color = BLACK
    def construct(self):
        MathTex.set_default(font_size=self.fs2, stroke_width=2)
        eq1 = MathTex(r'\mathbb P({\sf score} =k)', r'=', r'k/55').set_z_index(1)
        eq2 = MathTex(r'\mathbb P({\sf Alice} ={\sf Bob})', r'=',
                      r'\sum_{k=1}^{10}', r'\mathbb P({\sf Alice} = {\sf Bob}=k)').set_z_index(1)
        eq3 = MathTex(r'\mathbb P({\sf Alice} ={\sf Bob})', r'=', r'\sum_{k=1}^{10}',
                      r'\mathbb P({\sf score} =k)^2').set_z_index(1)
        eq4 = MathTex(r'\mathbb P({\sf Alice} ={\sf Bob})', r'=', r'\sum_{k=1}^{10}',
                      r'(k/55)^2').set_z_index(1)
        eq5 = MathTex(r'\mathbb P({\sf Alice} ={\sf Bob})', r'\approx', r'12.7\%').set_z_index(1)
        eq6 = MathTex(r'\mathbb P({\sf Alice} ={\sf Bob})', r'\approx', r'97.5\%').set_z_index(1)


        VGroup(eq1[0][0], eq2[0][0], eq2[3][0]).set_color(YELLOW)
        VGroup(eq1[0][2:-3], eq2[0][2:7], eq2[3][2:7]).set_color(RED)
        VGroup(eq2[0][8:11], eq2[3][8:11]).set_color(GREEN)
        VGroup(eq1[0][-2], eq1[2][0], eq2[2][3], eq2[3][-2], eq4[3][1]).set_color(PURPLE)
        VGroup(eq1[2][-2:], eq2[2][:2], eq2[2][5], eq3[3][-1], eq4[3][-1], eq4[3][3:5],
               eq5[2][:4], eq6[2][:4]).set_color(BLUE)

        copy_colors(eq3[0], eq2[0])
        copy_colors(eq3[2], eq2[2])
        copy_colors(eq3[3][:-1], eq1[0])
        copy_colors(eq4[0], eq2[0])
        copy_colors(eq4[2], eq2[2])
        copy_colors(eq4[0], eq2[0])
        copy_colors(eq5[0], eq2[0])

        eq1.to_edge(DOWN, buff=0.25)
        eq2.move_to(UP*0.4)
        mh.align_sub(eq3, eq3[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq2[1], coor_mask=UP)
        mh.align_sub(eq5, eq5[0], eq2[0], coor_mask=UP)
        mh.align_sub(eq6, eq6[1], eq5[1])

        self.add(eq1)
        self.wait(0.1)
        self.play(FadeIn(eq2, rate_func=linear))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:3], eq3[:3]),
                  mh.rtransform(eq2[3][:2], eq3[3][:2], eq2[3][11:], eq3[3][7:-1]),
                  mh.rtransform(eq2[3][7], eq3[3][7]),
                  mh.fade_replace(eq2[3][2:7], eq3[3][2:7], coor_mask=RIGHT),
                  FadeOut(eq2[3][8:11]),
                  FadeIn(eq3[3][-1], shift=mh.diff(eq2[3][-1], eq3[3][-2])))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:3], eq4[:3], eq3[3][1], eq4[3][0], eq3[3][-2:], eq4[3][-2:]),
                  FadeOut(eq3[3][0], shift=mh.diff(eq3[3][1], eq4[3][0])),
                  FadeOut(eq3[3][2:-2]),
                  mh.rtransform(eq1[2][:].copy(), eq4[3][1:-2]),
                  run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0], eq5[0]),
                  mh.fade_replace(eq4[1], eq5[1]),
                  mh.fade_replace(eq4[2:], eq5[2:], coor_mask=RIGHT),
                  )
        self.wait(0.1)
        line1 = Line(eq5[2].get_corner(DL)+LEFT*0.1, eq5[2].get_corner(UR)+RIGHT*0.1, stroke_color=PURE_RED, stroke_width=10).set_z_index(2)
        self.play(Create(line1), rate_func=linear, run_time=0.5)
        self.wait(0.1)
        self.play(FadeOut(eq5[2], line1, eq1))
        self.wait(0.1)
        self.play(FadeIn(eq6[2]))
        self.wait(0.1)
        circ1 = mh.circle_eq(eq6, color=GREEN).set_z_index(2).scale(1.1)
        self.play(Create(circ1), rate_func=linear, run_time=1.2)


        self.wait()

class CouplingEst(AliceBobDraw):
    def construct(self):
        MathTex.set_default(font_size=self.fs2, stroke_width=2)
        eq1 = MathTex(r'\mathbb E[{\sf step\ size}]', r'=', r'\frac12(1+10)')
        eq2 = MathTex(r'\mathbb E[{\sf step\ size}]', r'=', r'5.5')
        eq3 = MathTex(r'\mathbb P({\sf Alice\ hits\ }k)', r'\approx', r'1/5.5')
        eq4 = MathTex(r'\mathbb P({\sf Alice\ hits\ }k)', r'\approx', r'0.182')
        eq5 = MathTex(r'{\sf num\ steps}', r'\approx', r'100/5.5')
        eq6 = MathTex(r'{\sf num\ steps}', r'\approx', r'18.2')
        eq7 = MathTex(r'\mathbb P({\sf no\ coupling})', r'\approx', r"\mathbb P({\sf Alice\ doesn't\ hit\ }k)",
                      r'^{\sf num\ steps}', font_size=65)
        eq8 = MathTex(r'\mathbb P({\sf no\ coupling})', r'\approx', r"(1-0.182)", r'^{\sf num\ steps}')
        eq9 = MathTex(r'\mathbb P({\sf no\ coupling})', r'\approx', r"(1-0.182)", r'^{18.2}')
        eq10 = MathTex(r'\mathbb P({\sf no\ coupling})', r'\approx', r'2.6\%')
        eq11 = MathTex(r'\mathbb P({\sf coupling})', r'\approx', r'97.4\%')

        VGroup(eq1[0][0], eq3[0][0], eq7[0][0], eq7[2][0], eq11[0][0]).set_color(YELLOW)
        VGroup(eq3[0][2:7], eq7[2][2:7]).set_color(RED)
        VGroup(eq1[2][0], eq1[2][2], eq1[2][4], eq1[2][6:8], eq2[2], eq5[2][:3], eq3[2][0], eq3[2][2:],
               eq5[2][-3:], eq4[2], eq6[2], eq8[2][1], eq8[2][3:-1], eq9[3], eq10[2][:-1], eq11[2][:-1]).set_color(BLUE)
        VGroup(eq3[0][-2], eq7[2][-2]).set_color(PURPLE)
        VGroup(eq7[0][2:-1], eq11[0][2:-1]).set_color(ORANGE)
        VGroup(eq5[0], eq6[0], eq1[0][2:-1], eq7[3], eq8[3]).set_color(TEAL)
        copy_colors(eq2[0], eq1[0])
        copy_colors(eq4[0], eq3[0])
        copy_colors(eq8[0], eq7[0])
        copy_colors(eq10[0], eq7[0])

        mh.align_sub(eq2, eq2[1], eq1[1], coor_mask=UP)
        eq3.next_to(eq2, DOWN, buff=0.5)

        self.add(eq1)
        self.wait(0.1)
        self.play(Succession(Wait(0.2), mh.rtransform(eq1[:2], eq2[:2])),
                  mh.fade_replace(eq1[2], eq2[2], coor_mask=RIGHT))
        self.wait(0.1)
        VGroup(eq2.generate_target(), eq3).move_to(ORIGIN)
        self.play(MoveToTarget(eq2), Succession(Wait(0.2), FadeIn(eq3)))
        self.wait(0.1)

        mh.align_sub(eq4, eq4[1], eq3[1])

        self.play(mh.rtransform(eq3[:2], eq4[:2]),
                  mh.fade_replace(eq3[2], eq4[2], coor_mask=RIGHT))
        self.wait(0.1)
        gp1 = VGroup(eq2, eq4)
        eq5.next_to(eq4, DOWN, buff=0.5)
        VGroup(gp1.generate_target(), eq5).move_to(ORIGIN)
        self.play(MoveToTarget(gp1), FadeIn(eq5))
        self.wait(0.1)
        mh.align_sub(eq6, eq6[1], eq5[1], coor_mask=UP)
        self.play(mh.rtransform(eq5[:2], eq6[:2]),
                  mh.fade_replace(eq5[2], eq6[2], coor_mask=RIGHT))
        self.wait(0.1)
        gp1 = VGroup(eq2, eq4, eq6)
        eq7.next_to(eq6, DOWN, buff=0.5)
        VGroup(gp1.generate_target(), eq7).move_to(ORIGIN)
        self.play(MoveToTarget(gp1), FadeIn(eq7))
        self.wait(0.1)

        eq8.align_to(eq7, UP)
        mh.align_sub(eq9, eq9[2][-1], eq8[2][-1])
        mh.align_sub(eq10, eq10[1], eq9[1], coor_mask=UP)
        mh.align_sub(eq11, eq11[1], eq9[1], coor_mask=UP)

        self.play(mh.rtransform(eq7[:2], eq8[:2], eq7[2][1], eq8[2][0], eq7[2][-1], eq8[2][-1],
                                eq7[3], eq8[3], eq4[2][:].copy(), eq8[2][3:8]),
                  FadeOut(eq7[2][0], shift=mh.diff(eq7[2][1], eq8[2][0])),
                  FadeOut(eq7[2][2:-1], shift=mh.diff(eq7[2][2:-1], eq8[2][1:-1])),
                  FadeIn(eq8[2][1:3], shift=mh.diff(eq7[2][1], eq8[2][0])),
                  run_time=1.6)
        self.wait(0.1)
        self.play(mh.stretch_replace(eq6[2][:].copy(), eq9[3][:]),
                  FadeOut(eq8[3]),
                  run_time=1.4)
        self.wait(0.1)
        self.play(mh.rtransform(eq8[:2], eq10[:2]),
                  mh.fade_replace(eq8[2], eq10[2], coor_mask=RIGHT),
                  FadeOut(eq9[3], shift=mh.diff(eq8[2], eq10[2])),
                  run_time=1.2)
        self.wait(0.1)
        self.play(mh.rtransform(eq10[0][:2], eq11[0][:2], eq10[0][4:], eq11[0][2:],
                                eq10[1], eq11[1], eq10[2][-1], eq11[2][-1]),
                  FadeOut(eq10[0][2:4]),
                  mh.fade_replace(eq10[2][:-1], eq11[2][:-1], coor_mask=RIGHT),
                  run_time=1.6)
        self.wait(0.1)
        circ = mh.circle_eq(eq11, color=GREEN).set_z_index(2).scale(1.05)
        self.play(Create(circ), run_time=0.8)

        self.wait()

class AliceBobBound(AliceBobDraw):
    bg_color = GREY

    def construct(self):
        MathTex.set_default(font_size=self.fs2, stroke_width=2)
        eq1 = MathTex(r"\lvert", r"\mathbb P({\sf Alice's\ score} = k)", r"-",
                      r"\mathbb P({\sf Bob's\ score} = k)", r"\rvert",
                      r'\le', r"\mathbb P({\sf Alice's\ score} \not={\sf Bob's\ score})").set_z_index(2)
        eq2 = MathTex(r"\mathbb P({\sf not\ coupled})")[0].set_z_index(2)
        eq3 = MathTex(r"\lvert", r"\mathbb P({\sf Alice\ hits\ } k)", r"-",
                      r"\mathbb P({\sf Bob\ hits\ } k)", r"\rvert",
                      r'\le', r"\mathbb P({\sf coupling\ time} > k)").set_z_index(2)


        VGroup(eq1[1][0], eq1[3][0], eq1[6][0], eq2[0],
               eq3[1][0], eq3[3][0], eq3[6][0]).set_color(YELLOW)
        VGroup(eq1[1][2:14], eq1[6][2:14], eq3[1][2:7]).set_color(RED)
        VGroup(eq2[2:-1], eq3[6][2:-3]).set_color(ORANGE)
        VGroup(eq1[3][2:12], eq1[6][16:26], eq3[3][2:5]).set_color(GREEN)
        VGroup(eq1[1][15], eq1[3][13], eq3[1][-2], eq3[3][-2], eq3[6][-2]).set_color(PURPLE)

        eq1[5:].next_to(eq1[:5], DOWN, buff=0.2)
        eq1[6:].next_to(eq1[5], DOWN, buff=0.2)
        eq1[:5].move_to(ORIGIN, coor_mask=RIGHT)
        eq1[5].move_to(ORIGIN, coor_mask=RIGHT)
        eq1[6:].move_to(ORIGIN, coor_mask=RIGHT)
        box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_opacity=0.65, fill_color=BLACK,
                                    corner_radius=0.15, buff=0.2)
        VGroup(box1, eq1).to_edge(DOWN, buff=0.2)

        mh.align_sub(eq2, eq2[0], eq1[6][0], coor_mask=UP)
        mh.align_sub(eq3[:5].move_to(ORIGIN), eq3[2], eq1[2], coor_mask=UP)
        eq3[5].move_to(eq1[5])
        mh.align_sub(eq3[6].move_to(ORIGIN), eq3[6][0], eq2[0], coor_mask=UP)

        eq1_1 = eq1[:5].copy().move_to(eq1, coor_mask=UP)
        VGroup(eq1_1[0], eq1_1[4]).set_opacity(0)
        self.add(box1, eq1_1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1_1, eq1[:5]),
                  FadeIn(eq1[5:]),
                  run_time=1.2)
        self.wait(0.1)
        self.play(Succession(Wait(0.6), mh.rtransform(eq1[6][:2], eq2[:2], eq1[6][-1], eq2[-1])),
                  mh.fade_replace(eq1[6][2:-1], eq2[2:-1], coor_mask=RIGHT, run_time=1.4),
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq1[2], eq3[2], eq1[0], eq3[0], eq1[4:6], eq3[4:6],
                                eq1[1][:7], eq3[1][:7], eq1[3][-2:], eq3[3][-2:],
                                eq1[1][-2:], eq3[1][-2:], eq1[3][:5], eq3[3][:5]),
                  mh.fade_replace(eq1[1][7:-2], eq3[1][7:-2], coor_mask=RIGHT),
                  mh.fade_replace(eq1[3][5:-2], eq3[3][5:-2], coor_mask=RIGHT),
                  run_time=1.4)
        self.play(mh.rtransform(eq2[:2], eq3[6][:2], eq2[-1], eq3[6][-1]),
                  mh.fade_replace(eq2[2:-1], eq3[6][2:-1], coor_mask=RIGHT))
        self.wait()

class AliceStationary(AliceBobDraw):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=self.fs2)
        eq1 = MathTex(r'\mathbb P({\sf start}=k)', r'=', r'0.1').set_z_index(1)
        eq2 = MathTex(r'=', r'(11-k)/55').set_z_index(1)

        eq1[0][0].set_color(YELLOW)
        VGroup(eq1[0][2:7]).set_color(RED)
        VGroup(eq1[0][-2], eq2[1][4]).set_color(PURPLE)
        VGroup(eq1[2], eq2[1][1:3], eq2[1][-2:]).set_color(BLUE)

        mh.align_sub(eq2, eq2[0], eq1[1])

        self.add(eq1)
        self.wait(0.1)
        self.play(FadeOut(eq1[2]), FadeIn(eq2[1]))
        self.wait()

class AliceHit(AliceBobDraw):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=self.fs2)
        eq1 = MathTex(r'\mathbb P({\sf Alice\ hits\ }k)', r'=', r'10/55').set_z_index(1)
        eq2 = MathTex(r'\approx', r'18.2\%').set_z_index(1)

        eq1[0][0].set_color(YELLOW)
        VGroup(eq1[0][2:11]).set_color(RED)
        VGroup(eq1[0][-2]).set_color(PURPLE)
        VGroup(eq1[2][:2], eq1[2][3:], eq2[1][:4]).set_color(BLUE)

        mh.align_sub(eq2, eq2[0], eq1[1])

        self.add(eq1)
        self.wait(0.1)
        self.play(FadeOut(eq1[1:]), FadeIn(eq2))
        self.wait()

def get_box(ax, i, h):
    bl = ax.coords_to_point(i + 0.15, 0)
    ur = ax.coords_to_point(i + 1, h)
    box = Rectangle(width=(ur - bl)[0], height=(ur - bl)[1], fill_opacity=1, fill_color=BLUE,
                    stroke_width=2, stroke_opacity=1, stroke_color=WHITE).set_z_index(0.9)
    box.next_to(bl, UR, buff=0)
    return box

class RenewalThm(AliceBobDraw):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=self.fs2)
        eq1 = MathTex(r'=', r'1/\mathbb E[{\sf step\ size}]').set_z_index(1)
        eq1[1][0].set_color(BLUE)
        eq1[1][2].set_color(YELLOW)
        eq1[1][4:-1].set_color(TEAL)
        self.add(eq1)


class ProbFinal(Scene):
    xlen = 8.
    ylen = 4
    height = 10/55

    def get_axes(self, x_str, x_max=11):
        ax = Axes(x_range=[0, x_max], y_range=[0, self.height * 1.2], x_length=self.xlen, y_length=self.ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  #                  shade_in_3d=True,
                  ).set_z_index(1)
        labely = Tex(r'\sf probability', color=YELLOW, font_size=40).set_z_index(3)
        labely.next_to(ax.y_axis.get_end(), RIGHT)
        labelx = Tex(*x_str, color=YELLOW, font_size=40).set_z_index(3)
        labelx[1].next_to(labelx[0], DOWN, buff=0.2)
        labelx.next_to(ax.x_axis.get_end(), LEFT, buff=-0.1)#.shift(RIGHT*0.2)

        return ax, labely, labelx

    def get_label(self, ax, i, num):
        digit_str = str(num)
        eq_digit = MathTex(digit_str, color=RED, font_size=60, stroke_width=2)[0].set_z_index(2)
        eq_digit.next_to(ax.coords_to_point(i + 0.575, 0), DOWN, buff=0.3)
        eq_d2 = eq_digit.copy().set_z_index(1.9).set_color(WHITE).set_stroke(width=3)
        return VGroup(eq_digit, eq_d2)

    def construct(self):
        heights = [_/55 for _ in range(1, 11)]

        ax, labely, labelx = self.get_axes([r'\sf final', r'digit'])

        boxes = []
        for heights1 in [[0.1] * 10, heights]:
            boxes1 = [get_box(ax, i, h) for i, h in enumerate(heights1)]
            boxes.append(boxes1)

        labels = []
        for i in range(10):
            label = self.get_label(ax, i, i+1 if i < 9 else 0)
            labels.append(label)

        p0 = ax.coords_to_point(0, 0.1)
        line1 = DashedLine(p0, ax.coords_to_point(10.5, 0.1), stroke_width=3, stroke_color=YELLOW).set_z_index(0.95)
        label1 = MathTex(r'0.1', font_size=40, color=YELLOW).next_to(boxes[0][0], UP, buff=0.1)

        self.add(ax, *boxes[0], labely, labelx, *labels, line1, label1)
        self.wait()
        self.play(*[ReplacementTransform(b1, b2) for b1, b2 in zip(*boxes)], run_time=1.6)
        self.wait()

class StationaryPlot(ProbFinal):
    xlen = 12.
    ylen = 6.
    fade_line2 = True

    num_steps = 11
    heights1 = [_/55 for _ in range(10, 0, -1)]
    minimal = False

    def plot_extra(self, ax):
        pass

    def construct(self):
        heights0 = [0.1] * 10
        heights1 = self.heights1

        ax, labely, labelx = self.get_axes([r'\sf next', r'position'], x_max=12)
        ax.set_z_index(10)
        labels = [self.get_label(ax, i, i + 1) for i in range(10)]
        VGroup(ax, labelx, labely, *labels).move_to(ORIGIN)

        boxes0 = [get_box(ax, i, h) for i, h in enumerate(heights0)]
        boxes1 = [get_box(ax, i, h) for i, h in enumerate(heights1)]
        p0 = ax.coords_to_point(0, 0.1)
        line1 = DashedLine(p0, ax.coords_to_point(11, 0.1), stroke_width=3, stroke_color=YELLOW).set_z_index(6)
        label1 = MathTex(r'0.1', font_size=40, color=YELLOW).next_to(line1.get_end(), UL, buff=0.1).shift(LEFT*0.2)
        p0 = ax.coords_to_point(0, 0.1818)
        line2 = DashedLine(p0, ax.coords_to_point(11, 0.1818), stroke_width=3, stroke_color=YELLOW).set_z_index(6)
        label2 = MathTex(r'0.18', font_size=40, color=YELLOW).next_to(line2.get_end(), UL, buff=0.1).shift(LEFT*0.2)

        label3 = labels[-1]
        VGroup(label3[0][0], label3[1][0]).shift(RIGHT * 0.02)
        VGroup(label3[0][1], label3[1][1]).shift(LEFT * 0.02)

        add1 = [ax, labely, labelx, *boxes0, line1, label1, *labels]
        anims1 = [ReplacementTransform(b1, b2) for b1, b2 in zip(boxes0, boxes1)]

        if self.fade_line2:
            anims1.append(FadeIn(line2, label2))
        else:
            add1 += [line2, label2]

        if self.minimal:
            VGroup(ax, labelx, labely, line1, line2, label1, label2, *labels).set_opacity(0)

        self.add(*add1)
        self.wait(0.1)
        self.plot_extra(ax)
        self.play(*anims1, run_time=1.6)

        for _ in range(self.num_steps):
            self.play(boxes1[0].animate.set_fill(color=RED), run_time=0.6)
            h = heights1[0] * 0.1
            up = ax.coords_to_point(0, 1) - ax.coords_to_point(0, 0)
            right = ax.coords_to_point(1, 0) - ax.coords_to_point(0, 0)
            box0 = get_box(ax, 0, h).set_fill(color=RED).set_z_index(2)
            boxes2 = [box0.copy().shift(up*i*h) for i in range(10)]

            self.play(FadeIn(*boxes2))
            self.remove(*boxes1[0])

            anims = [b.animate.next_to(ax.coords_to_point(i+2, 0), UL, buff=0) for i, b in enumerate(boxes2)]
            self.play(VGroup(boxes1[1:]).animate.shift(up * h), *anims,
                      FadeOut(labels[0]),
                      run_time=2)
            self.remove(labels[0])
            labels = labels[1:]
            label3 = self.get_label(ax, 9, _ + 11)
            VGroup(label3[0][0], label3[1][0]).shift(RIGHT*0.02)
            VGroup(label3[0][1], label3[1][1]).shift(LEFT*0.02)
            gp1 = VGroup(*boxes2, *boxes1[1:])
            gp1.set_z_index(0.5)

            heights1 = [_ + h for _ in heights1[1:]] + [h]
            boxes3 = [get_box(ax, i, h) for i, h in enumerate(heights1)]
            self.play(gp1.animate.shift(-right),
                      FadeIn(*boxes3, shift=-right),
                      VGroup(*labels).animate.shift(-right),
                      FadeIn(label3, shift=-right)
                      )
            self.remove(*boxes1, *boxes2)
            boxes1 = boxes3
            labels.append(label3)


        self.wait()

class StationaryPlotThumb(StationaryPlot):
    num_steps = 1
    minimal = True


class StationaryConv(StationaryPlot):
#    height = 10/55 * 1.13 * 0.9
#    ylen = 6 * 1.13 * 0.9
    num_steps = 20
    heights1 = [0.1] * 10
    fade_line2 = False
    def plot_extra(self, ax):
        lines = []
        for i in range(10):
            h = (10-i)/55
            line1 = Line(ax.coords_to_point(i+0.15, h), ax.coords_to_point(i+1, h),
                         stroke_color=RED, stroke_width=6, stroke_opacity=1).set_z_index(10)
            lines.append(*line1)
        self.play(FadeIn(*lines))


class Vervaat(AliceBobEqual):
    def get_axes(self, scale=1., xlen = 0.9, ylen=0.95, scale_neg=1.):
        ymax = 1.5 / scale
        xlen *= 2 * config.frame_x_radius
        ylen *= 2 * config.frame_y_radius
        ax = Axes(x_range=[0, 1.05], y_range=[-ymax*scale_neg, ymax], x_length=xlen, y_length=ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.6 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  ).set_z_index(1.9)
        eqt = MathTex(r't').next_to(ax.x_axis.get_right(), UP, buff=0.2)
        mark1 = ax.x_axis.get_tick(1, size=0.1).set_stroke(width=6).set_z_index(11)
        line1 = DashedLine(ax.coords_to_point(1, -ymax*scale_neg), ax.coords_to_point(1, ymax), color=GREY).set_z_index(10).set_opacity(0.6)
        eq2 = MathTex(r'T', font_size=60).set_z_index(3).next_to(mark1, DR, buff=0.05)
        eq6 = MathTex(r'1', font_size=60).set_z_index(3).next_to(mark1, DR, buff=0.05)

        return ax, eqt, xlen, ylen, ymax, mark1, line1, eq2, eq6

    def construct(self):
        seeds = [170, 180, 174, 169]
        npts = 1920
        ndt = npts - 1
        np.random.seed(seeds[0])

        ax, eqt, xlen, ylen, ymax, mark1, _, _, _ = self.get_axes(1.3, xlen=0.45, ylen=0.475, scale_neg=0.6)
        eqt.set_z_index(0.5)
        ymax2 = 0.9375
        print(ymax)
        ax.y_axis.set_z_index(10)
        gp = VGroup(ax, eqt, mark1).to_corner(DL, buff=0.2)
        gp2 = gp.copy().to_corner(DR, buff=0.2)
        ax2: Axes = gp2[0]

        box0 = SurroundingRectangle(gp, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.15).set_z_index(0.1)
        box0_1 = SurroundingRectangle(gp2, stroke_width=0, stroke_opacity=0, fill_color=BLACK, fill_opacity=0.6,
                                    corner_radius=0.15)
        self.add(gp, gp2, box0, box0_1)
        self.wait(0.1)
        t_vals = np.linspace(0, 1., npts)
        s = np.sqrt(t_vals[1])
        while True:
            b_vals = np.concatenate(([0.], np.random.normal(scale=s, size=ndt).cumsum()))
            b_vals -= b_vals[-1] * t_vals
            if ndt * 0.4 < np.argmin(b_vals) < ndt * 0.6 and 0.5 * ymax2 < -min(b_vals)\
                    and max(b_vals) - min(b_vals) < ymax2 * 1.05 and max(b_vals) > 0.2 * ymax2:
                break

        path1 = ax.plot_line_graph(t_vals, b_vals, add_vertex_dots=False, stroke_color=YELLOW,
                                   stroke_width=3).set_z_index(2)

        i0 = np.argmin(b_vals)
        y0 = b_vals[i0]

        path4 = ax2.plot_line_graph(t_vals, np.concatenate((b_vals[i0:], b_vals[1:i0+1])) - y0, add_vertex_dots=False,
                                    stroke_color=YELLOW, stroke_width=3).set_z_index(2)

        path2 = ax.plot_line_graph(t_vals[:i0+1], b_vals[:i0+1], add_vertex_dots=False, stroke_color=RED,
                                   stroke_width=4).set_z_index(2.2)
        path3 = ax.plot_line_graph(t_vals[i0:], b_vals[i0:], add_vertex_dots=False, stroke_color=BLUE,
                                   stroke_width=4).set_z_index(2.1)

        p0 = ax.coords_to_point(0, y0)
        y1 = max(b_vals)
        p1 = ax.coords_to_point(1, y1)
        box1 = Rectangle(width=(p1-p0)[0], height=(p1-p0)[1], stroke_width=0, stroke_opacity=0,
                  fill_color=GREY_D, fill_opacity=0.8).set_z_index(1)
        box1.next_to(p0, UR, buff=0)
        t1 = 0.75
        arr1 = Arrow(ax.coords_to_point(t1, y0), ax.coords_to_point(t1, y1), color=WHITE, buff=0).set_z_index(10)
        eqv = MathTex(r'V', stroke_width=2).next_to(arr1, RIGHT, buff=0).set_z_index(5).shift(DOWN*0.2)

        shift2 = path4['line_graph'].get_start() - path3['line_graph'].get_start()
        shift1 = path4['line_graph'].get_end() - path2['line_graph'].get_end()
        shift3 = ax2.coords_to_point(0, 0) - box1.get_corner(DL)

        gp3 = VGroup(box1, arr1, eqv).copy()
        gp3.generate_target().shift(shift3)
        gp3[0].set_opacity(0)

        txt1 = Tex(r'\sf Brownian bridge', color=YELLOW, font_size=60).set_z_index(4)
        txt2 = Tex(r'\sf excursion', font_size=60, color=YELLOW).set_z_index(4)
        txt2.move_to(ax2.coords_to_point(0.5, 0.8 * ymax)).next_to(gp3.target[0], UP, coor_mask=UP, buff=0.15)
        txt1.move_to(ax.coords_to_point(0.5, 0.6*ymax))

        self.play(Create(path1, rate_func=linear, run_time=1), FadeIn(txt1, run_time=1))
        self.play(Create(path4, rate_func=linear, run_time=1), FadeIn(txt2, run_time=1))
        self.wait(0.1)
        self.play(FadeIn(box1, arr1, eqv))
        self.play(FadeIn(path2, path3))
        self.wait(0.1)
        self.remove(path1)
        self.add(VGroup(path2, path3).copy())

        self.play(path2.animate.shift(shift1), path3.animate.shift(shift2),
                  MoveToTarget(gp3),
                  run_time=3)
        self.remove(path4)

        self.wait(1)

class TwoVars(AliceBobEqual):
    def construct(self):
        MathTex.set_default(font_size=80)
        eq1 = MathTex(r'X')[0].set_z_index(2)
        eq2 = MathTex(r'Y')[0].set_z_index(2)
        eq3 = MathTex(r'(X,Y)')[0].set_z_index(2)
        eq2.next_to(eq1, DOWN, buff=0.4)
        gp1 = VGroup(eq1, eq2)
        eq3.next_to(gp1, RIGHT, buff=2).shift((UP*0.3))

        VGroup(eq1, eq3[1]).set_color(RED)
        VGroup(eq2, eq3[3]).set_color(GREEN)
        #VGroup(eq3[0], eq3[4]).set_color(YELLOW)

        Tex.set_default(color=BLUE, font_size=60, stroke_width=2)
        txt1 = Tex(r'two ', 'distributions').set_z_index(1)
        txt2 = Tex(r'joint ', 'distribution').set_z_index(1)
        txt1[1].next_to(txt1[0], DOWN)
        txt1.next_to(gp1, LEFT, buff=0.5)
        txt2[1].next_to(txt2[0], DOWN, buff=0.05)
        txt2.next_to(eq3, DOWN, buff=0.2)
        VGroup(txt2, eq3).move_to(gp1, coor_mask=UP)

        box2 = SurroundingRectangle(VGroup(eq3, txt2), stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                    fill_opacity=0.65, corner_radius=0.15, buff=0.4)
        box1_1 = SurroundingRectangle(VGroup(eq1, eq2, txt1), stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                    fill_opacity=0.65, corner_radius=0.15, buff=0.4)

        box1 = RoundedRectangle(width=box1_1.width, height=box2.height, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                    fill_opacity=0.65, corner_radius=0.15).move_to(box1_1)

        VGroup(box1, box2, eq1, eq2, eq3, txt1, txt2).move_to(ORIGIN).to_edge(DOWN, buff=0.2)

        self.add(eq1, eq2, box1, txt1)


        self.play(mh.rtransform(eq1[0].copy(), eq3[1], eq2[0].copy(), eq3[3], run_time=2.),
                  Succession(Wait(1.), FadeIn(eq3[0], eq3[2], eq3[4], run_time=1.)),
                  FadeIn(box2, run_time=2))
        self.play(FadeIn(txt2))
        self.wait()

class Orders(AliceBobEqual):
    def construct(self):
        txt1 = Tex(r'\sf e.g.,', r' stochastic order: ', '$X\le Y$', font_size=80, stroke_width=1.5).set_z_index(2)
        txt2 = Tex(r'\sf e.g.,', r' convex order: ', '$X=\mathbb E[ Y]$', font_size=80, stroke_width=1.5).set_z_index(2)
        box1 = SurroundingRectangle(txt1, stroke_width=0, stroke_opacity=0, fill_color=BLACK,
                                    fill_opacity=0.65, corner_radius=0.15, buff=0.2)
        txt1[0][:-1].set_color(BLUE)
        VGroup(txt1[1], txt2[1]).set_color(TEAL)
        VGroup(txt1[2][0], txt2[2][0]).set_color(RED)
        VGroup(txt1[2][2], txt2[2][-2]).set_color(GREEN)
        txt2[2][2].set_color(YELLOW)
        self.add(txt1, box1)
        self.wait(0.1)
        shift = mh.diff(txt1[2][-1], txt2[2][-2])
        gp = VGroup(txt2[2][2:4], txt2[2][-1]).set_opacity(-1).shift(-shift)
        self.play(mh.fade_replace(txt1[1][:-6], txt2[1][:-6], coor_mask=RIGHT),
                  mh.rtransform(txt1[1][-6:], txt2[1][-6:], txt1[2][0], txt2[2][0],
                                txt1[2][2], txt2[2][-2]),
                  mh.fade_replace(txt1[2][1], txt2[2][1]),
                  Succession(gp.animate.shift(shift).set_opacity(1)))
        self.wait()

class HarmonicDef(AliceBobEqual):
    def construct(self):
        MathTex.set_default(stroke_width=4, font_size=60)
        eq1 = MathTex(r'f(m,n)', r'=', r'\frac14\left(f(m-1,n)+f(m+1,n)+f(m,n-1)+f(m,n+1)\right)').set_z_index(4)
        eq1[0].set_color(BLUE)
        gp1 = VGroup(eq1[2][4:12], eq1[2][13:21], eq1[2][22:30], eq1[2][31:39])
        gp1.set_color(GREEN)
        gp2 = gp1.copy()
        w = config.frame_width
        pt0 = ORIGIN + RIGHT * w * 0.055
        eq1_0 = eq1[0].copy().next_to(pt0, UR, buff=0.25)

        eq1.scale(0.75).to_edge(DOWN, buff=0.1)
        eq2 = eq1.copy().set_z_index(1).set_color(BLACK).set_stroke(width=16)

        h = 0.142 * w
        gp2[0].next_to(pt0 + LEFT*h, LEFT, buff=0.4)
        gp2[1].next_to(pt0 + RIGHT*h, RIGHT, buff=0.4)
        gp2[2].next_to(pt0 + DOWN*h, DOWN, buff=0.25)
        gp2[3].next_to(pt0 + UP*h, UP, buff=0.25)

        #self.add(VGroup(Line(pt0 + LEFT*h, pt0 + RIGHT*h, stroke_width=10),
        #       Line(pt0 + DOWN * h, pt0 + UP * h, stroke_width=10)))

        gp2_1 = gp2.copy().set_color(BLACK).set_z_index(1).set_stroke(width=16)
        gp2_0 = eq1_0.copy().set_color(BLACK).set_z_index(1).set_stroke(width=16)

        self.add(eq1_0, gp2_0)
        self.wait()
        self.play(FadeIn(gp2_1, gp2, rate_func=linear))
        self.wait(0.1)
        self.play(mh.rtransform(eq1_0, eq1[0], gp2, gp1,
                                gp2_0, eq2[0], gp2_1, VGroup(eq2[2][4:12], eq2[2][13:21], eq2[2][22:30], eq2[2][31:39]),
                                run_time=2),
                  Succession(Wait(1), FadeIn(eq1[1], eq1[2][:4], eq1[2][-1], eq1[2][12], eq1[2][21], eq1[2][30],
                                             eq2[1], eq2[2][:4], eq2[2][-1], eq2[2][12], eq2[2][21], eq2[2][30])),
                  )
        self.wait()

def eq_shadow(eq: VGroup, fg_z_index=4., bg_z_index=0., bg_color=BLACK, bg_stroke_width=10.):
    res = VGroup()
    for eq1 in eq:
        elem = VGroup()
        for eq2 in eq1:
            elem.add(VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res

class HarmonicEx(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=2)
        eq1 = MathTex(r'f(m,n)', r'=', r'-m')
        eq2 = MathTex(r'=', r'n^2 - m^2')
        eq3 = MathTex(r'=', r'm^2 - n^2')
        eq4 = MathTex(r'=', r'\Re\left[a^mb^n]')
        eq5 = MathTex(r'(a+1/a+b+1/b=4)', font_size=65)
        VGroup(eq1[0][2], eq1[0][4], eq1[2][-1],
               eq2[1][0], eq2[1][-2], eq3[1][0], eq3[1][3],
               eq4[1][3], eq4[1][5]).set_color(GREEN)
        VGroup(eq2[1][1], eq2[1][-1], eq3[1][1], eq3[1][-1],
               eq5[0][3], eq5[0][9], eq5[0][-2]).set_color(BLUE)
        eq1[0][0].set_color(RED_B)
        VGroup(eq4[1][2], eq4[1][4], eq5[0][1], eq5[0][5], eq5[0][7], eq5[0][11]).set_color(ORANGE)
        eq4[1][0].set_color(YELLOW)

        mh.align_sub(eq2, eq2[0], eq1[1])
        eq2 = VGroup(eq1[0].copy(), *eq2[:]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq3, eq3[0], eq2[1])
        eq3 = VGroup(eq2[0].copy(), *eq3[:]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq4, eq4[0], eq3[1])
        eq4 = VGroup(eq3[0].copy(), *eq4[:]).move_to(ORIGIN, coor_mask=RIGHT)
        eq5.next_to(eq4, DOWN, buff=1)

        eq1 = eq_shadow(eq1)
        eq2 = eq_shadow(eq2)
        eq3 = eq_shadow(eq3)
        eq4 = eq_shadow(eq4)
        eq5 = eq_shadow(eq5)

        self.add(eq1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[:2], eq2[:2], eq1[2][0], eq2[2][2], eq1[2][1], eq2[2][3]),
                  Succession(Wait(0.3), FadeIn(eq2[2][:2], eq2[2][-1])),
                  #FadeIn(eq2[2][-1], shift=mh.diff(eq1[2][1], eq2[2][3]))
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:2], eq3[:2], eq2[2][0], eq3[2][3], eq2[2][1], eq3[2][4],
                                eq2[2][2], eq3[2][2], eq2[2][3], eq3[2][0], eq2[2][4], eq3[2][1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[:2], eq4[:2]),
                  mh.fade_replace(eq3[2][0], eq4[2][3]),
                  mh.fade_replace(eq3[2][3], eq4[2][5]),
                  FadeOut(eq3[2][1], shift=mh.diff(eq3[2][0], eq4[2][3])),
                  FadeOut(eq3[2][4], shift=mh.diff(eq3[2][3], eq4[2][5])),
                  Succession(Wait(0.4), FadeIn(eq4[2][2], eq4[2][4])),
                  Succession(Wait(0.8), FadeIn(eq4[2][:2], eq4[2][-1])),
                  FadeOut(eq3[2][2]),
                  Succession(Wait(1.4), FadeIn(eq5))
                  )
        self.wait()

class WalkX0(AliceBobEqual):
    eq_str = r'X_0'
    eq_col = [(0, YELLOW), (1, BLUE)]
    fs = 50
    def construct(self):
        eq1 = MathTex(self.eq_str, font_size=self.fs, stroke_width=2)
        for i, col in self.eq_col:
            eq1[0][i].set_color(col)
        eq1 = eq_shadow(eq1)
        self.add(eq1)

class Walkx(WalkX0):
    eq_str = r'x'
    eq_col = [(0, YELLOW)]
    fs = 60

class Walky(Walkx):
    eq_str = r'y'
    eq_col = [(0, BLUE)]
    fs = 60

class CoupledT(Walkx):
    eq_str = r'{\sf time} = T'
    eq_col = [(_, TEAL) for _ in range(4)] + [(-1, RED)]
    fs = 100

def font_size_sub(eq: Mobject, index: int, font_size: float):
    n = len(eq[:])
    eq_1 = eq[index].copy()
    pos = eq.get_center()
    eq[index].set(font_size=font_size).align_to(eq_1, RIGHT)
    eq[index:].align_to(eq_1, LEFT)
    return eq.move_to(pos, coor_mask=RIGHT)

class WalkLaw(AliceBobEqual):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=80)
        txt1 = Tex(r'\sf conditioned on ', font_size=60, color=TEAL)
        eq1 = MathTex(r'X_t', r'=', r'(m, n)')
        eq2 = MathTex(r'\mathbb P\left(X_{t+1}\in\{', r'(m-1,n)', r',', r'(m+1,n)', r',', r'(m,n-1)',
                      r',', r'(m,n+1)', r'\}\right)', r'=', r'p',
                      font_size=60)
        eq3 = MathTex(r'\mathbb P\left(X_{t+1}=', r'(m,n)', r'\right)', r'=', r'1-p',
                      font_size=60)
        eq4 = MathTex(r'\mathbb P\left(X_{t+1}=', r'(m\pm1,n)', r'\right)', r'=', r'p/4',
                      font_size=60)
        eq5 = MathTex(r'=', r'(m,n\pm1)', font_size=60)
        font_size_sub(eq2, 1, font_size=40)
        font_size_sub(eq2, 3, font_size=40)
        font_size_sub(eq2, 5, font_size=40)
        font_size_sub(eq2, 7, font_size=40)
        VGroup(eq1[0][0], eq2[0][2], eq2[0][0], eq3[0][0], eq3[0][2]).set_color(YELLOW)
        VGroup(eq1[2][1], eq1[2][3], eq2[1][1], eq2[1][-2], eq2[3][1], eq2[3][-2],
               eq2[5][1], eq2[5][3], eq2[7][1], eq2[7][3], eq3[1][1], eq3[1][3],
               eq4[1][1], eq4[1][-2], eq5[1][1], eq5[1][3]).set_color(GREEN)
        VGroup(eq2[0][5], eq2[1][3], eq2[3][3], eq2[5][-2], eq2[7][-2],
               eq3[0][5], eq3[4][0], eq4[1][3], eq4[-1][-1], eq5[1][-2]).set_color(BLUE)
        VGroup(eq2[-1], eq3[-1][-1], eq4[-1][0]).set_color(ORANGE)
        VGroup(eq1[0][1], eq2[0][3], eq3[0][3]).set_color(RED)
        VGroup(eq2[0][-1], eq2[8][0]).set_color(GOLD)

        txt1.next_to(eq1, LEFT)
        mh.align_sub(txt1, txt1[0][0], eq1[1], coor_mask=UP)
        eq2.next_to(eq1, DOWN, buff=0.4)
        mh.align_sub(eq3, eq3[0][2], eq2[0][2], coor_mask=UP)
        mh.align_sub(eq4, eq4[0][2], eq3[0][2])
        eq4 = VGroup(eq3[0].copy(), *eq4[1:]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[0], eq4[0][-1])
        eq5 = VGroup(eq4[0], eq5[1], *eq4[2:]).move_to(ORIGIN, coor_mask=RIGHT)

        w = 15.
        txt1 = eq_shadow(txt1, bg_stroke_width=w)
        eq1 = eq_shadow(eq1, bg_stroke_width=w)
        eq2 = eq_shadow(eq2, bg_stroke_width=w)
        eq3 = eq_shadow(eq3, bg_stroke_width=w)
        eq4 = eq_shadow(eq4, bg_stroke_width=w)
        eq5 = eq_shadow(eq5, bg_stroke_width=w)

        self.add(eq1, txt1)
        self.wait(0.1)
        self.play(FadeIn(eq2))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[0][:6], eq4[0][:6], eq2[1][:2], eq4[1][:2], eq2[1][-3:], eq4[1][-3:],
                                eq2[8][1], eq4[2][0], eq2[9], eq4[3], eq2[10][0], eq4[4][0],
                                eq2[1][3], eq4[1][3]),
                  mh.rtransform(eq2[3][3], eq4[1][3]),
                  mh.rtransform(eq2[5][:2], eq4[1][:2], eq2[5][-1], eq4[1][-1], eq2[5][2:4], eq4[1][4:6]),
                  mh.rtransform(eq2[7][:2], eq4[1][:2], eq2[7][-1], eq4[1][-1], eq2[7][2:4], eq4[1][4:6]),
                  mh.fade_replace(eq2[3][2], eq4[1][2], coor_mask=RIGHT),
                  FadeOut(eq2[1][2], target_position=eq4[1][2]),
                  mh.rtransform(eq2[3][:2], eq4[1][:2], eq2[3][-3:], eq4[1][-3:]),
                  mh.fade_replace(eq2[0][6], eq4[0][6], coor_mask=RIGHT),
                  FadeOut(eq2[0][7], shift=mh.diff(eq2[1][0], eq4[1][0])),
                  FadeOut(eq2[2], shift=mh.diff(eq2[1][-1], eq4[1][-1])),
                  FadeOut(eq2[4], shift=mh.diff(eq2[3][-1], eq4[1][-1])),
                  FadeOut(eq2[5][4:6], shift=mh.diff(eq2[5][3], eq4[1][-2])),
                  FadeOut(eq2[6], shift=mh.diff(eq2[5][-1], eq4[1][-1])),
                  FadeOut(eq2[7][4:6], shift=mh.diff(eq2[7][3], eq4[1][-2])),
                  FadeOut(eq2[8][0], shift=mh.diff(eq2[7][-1], eq4[1][-1])),
                  FadeIn(eq4[4][1:], shift=mh.diff(eq2[10][0], eq4[4][0])),
                  run_time=1.4
                  )
        self.wait(0.1)
        self.play(mh.rtransform(eq4[0], eq5[0], eq4[2:], eq5[2:],
                                eq4[1][:2], eq5[1][:2], eq4[1][2:4], eq5[1][4:6],
                                eq4[1][4:6], eq5[1][2:4], eq4[1][-1], eq5[1][-1]))
        self.wait(0.1)
        self.play(Succession(Wait(0.4), AnimationGroup(
            mh.rtransform(eq5[0], eq3[0], eq5[1][:4], eq3[1][:4], eq5[1][-1], eq3[1][-1],
                                eq5[2:4], eq3[2:4], eq5[4][0], eq3[4][-1]),
            FadeOut(eq5[4][1:], shift=mh.diff(eq5[4][0], eq3[4][-1]), rate_func=linear),
            Succession(Wait(0.3), FadeIn(eq3[4][:2]))
        )),
                  FadeOut(eq5[1][4:6]))
        self.wait()

class Martingale(HarmonicEx):
    def construct(self):
        MathTex.set_default(stroke_width=2, font_size=80)
        txt1 = Tex(r'\sf conditioned on ', font_size=60, color=TEAL)
        eq1 = MathTex(r'X_t', r'=', r'(m, n)')
        eq2 = MathTex(r'\mathbb E[f(X_{t+1})]', r'=', r'(1-p)f(X_t)', r'+',
                      r'\frac p4(', r'f(m-1,n)+f(m+1,n)', r'+f(m,n-1)+f(m,n+1)', r'\right)', font_size=60)
        eq3 = MathTex(r'{}+', r'p', r'f(m, n)', font_size=60)
        eq5 = MathTex(r'\mathbb E[f(X_{t+1})]', r'=', r'f(X_t)', font_size=80)
        eq6 = MathTex(r'=', r'\mathbb E[f(X_t)]', font_size=80)
        eq7 = MathTex(r'\mathbb E[f(X_0)]', r'=', r'\mathbb E[f(X_1)]', r'=', r'\mathbb E[f(X_2)]', r'=', r'\cdots')
        eq8 = MathTex(r'f(x)', r'=', r'\mathbb E[f(X_t)]', font_size=80)
        eq9 = MathTex(r'f(y)', r'=', r'\mathbb E[f(Y_t)]', font_size=80)
        eq10 = MathTex(r'f(x) - f(y)', r'=', r'\mathbb E[f(X_t)-f(Y_t)]', font_size=80)
        eq11 = MathTex(r'\lvert f(x) - f(y)\rvert', r'\le', r'\mathbb E[2A\;1_{\{t < T\}}]', font_size=80)
        eq12 = MathTex(r'\le', r'2A\,\mathbb P(t < T)', font_size=80)

        font_size_sub(eq2, 5, font_size=40)
        font_size_sub(eq2, 6, font_size=40)
        eq2[6:].next_to(eq2[5], DOWN, buff=0.2).align_to(eq2[5], RIGHT).shift(RIGHT*0.6)
        VGroup(eq2[4][-1], eq2[5:]).shift(UP*0.2)
        VGroup(eq1[0][0], eq2[0][4], eq2[0][0], eq2[2][-3], eq5[0][0], eq5[0][4], eq5[2][2],
               eq6[1][0], eq6[1][4], eq7[0][0], eq7[0][4], eq7[2][0], eq7[2][4], eq7[4][0], eq7[4][4],
               eq8[0][2], eq8[2][0], eq8[2][4], eq9[2][0], eq10[0][2], eq11[0][3],
               eq10[2][4], eq10[2][0], eq11[2][0], eq12[1][2]).set_color(YELLOW)
        VGroup(eq1[2][1], eq1[2][3], eq2[5][2], eq2[5][6], eq2[5][11], eq2[5][15],
               eq2[6][3], eq2[6][5], eq2[6][12], eq2[6][14], eq3[2][2], eq3[2][4],
               ).set_color(GREEN)
        VGroup(eq2[0][7], eq2[2][1], eq2[4][2], eq2[5][4], eq2[5][13], eq2[6][7], eq2[6][-2],
               eq5[0][7], eq7[0][5], eq7[2][5], eq7[4][5], eq11[2][2], eq11[2][4], eq12[1][0]).set_color(BLUE)
        VGroup(eq2[2][3], eq2[4][0], eq3[1]).set_color(ORANGE)
        VGroup(eq2[0][2], eq2[2][5], eq2[5][0], eq2[5][9], eq2[6][1], eq2[6][10],
               eq3[2][0], eq5[0][2], eq5[2][0], eq6[1][2], eq7[0][2], eq7[4][2], eq7[2][2],
               eq8[0][0], eq8[2][2], eq9[0][0], eq9[2][2], eq10[0][0], eq10[0][5],
               eq10[2][2], eq10[2][8], eq11[0][1], eq11[0][6]).set_color(RED_B)
        VGroup(eq1[0][1], eq2[0][5], eq2[2][-2], eq5[0][5], eq5[2][3], eq6[1][-3],
               eq8[2][5], eq9[2][5], eq10[2][5], eq10[2][11], eq11[2][6], eq11[2][8],
               eq12[1][4], eq12[1][6]).set_color(RED)
        VGroup(eq9[0][2], eq9[2][4], eq10[0][7], eq10[2][10], eq11[0][8]).set_color(BLUE_B)
        VGroup(eq11[2][3], eq12[1][1]).set_color(TEAL)

        txt1.next_to(eq1, LEFT)
        mh.align_sub(txt1, txt1[0][0], eq1[1], coor_mask=UP)
        mh.align_sub(eq2, eq2[0], eq1, DOWN, buff=0.4).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq3, eq3[0], eq2[3])
        eq3 = eq3[1:]
        eq4 = VGroup(eq3[0].copy(), eq2[2][-5:].copy())
        mh.align_sub(eq4[1], eq4[1][0], eq3[1][0])
        mh.align_sub(eq5, eq5[1], eq2[1], coor_mask=UP).shift(LEFT*2)
        mh.align_sub(eq6, eq6[0], eq5[1])
        eq6 = VGroup(eq5[0].copy(), *eq6[:])
        mh.align_sub(eq7, eq7[1], eq6[1], coor_mask=UP)
        mh.align_sub(eq8, eq8[1], eq7[1])
        eq8_1 = VGroup(eq7[0], *eq8[1:]).copy().move_to(ORIGIN, coor_mask=RIGHT)
        #mh.align_sub(eq8, eq8[1], eq8_1[1])
        eq8.move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq9, eq9[1], eq8[1])
        mh.align_sub(eq10, eq10[1], eq8[1], coor_mask=UP)
        mh.align_sub(eq11, eq11[2][0], eq10[2][0])
        mh.align_sub(eq12, eq12[0], eq11[1])
        eq12 = VGroup(eq11[0].copy(), *eq12[:]).move_to(ORIGIN, coor_mask=RIGHT)

        w = 15.
        txt1 = eq_shadow(txt1, bg_stroke_width=w)
        eq1 = eq_shadow(eq1, bg_stroke_width=w)
        eq2 = eq_shadow(eq2, bg_stroke_width=w)
        eq3 = eq_shadow(eq3, bg_stroke_width=w)
        eq4 = eq_shadow(eq4, bg_stroke_width=w)
        eq5 = eq_shadow(eq5, bg_stroke_width=w)
        eq6 = eq_shadow(eq6, bg_stroke_width=w)
        eq7 = eq_shadow(eq7, bg_stroke_width=w)
        eq8 = eq_shadow(eq8, bg_stroke_width=w)
        eq8_1 = eq_shadow(eq8_1, bg_stroke_width=w)
        eq9 = eq_shadow(eq9, bg_stroke_width=w)
        eq10 = eq_shadow(eq10, bg_stroke_width=w)
        eq11 = eq_shadow(eq11, bg_stroke_width=w)
        eq12 = eq_shadow(eq12, bg_stroke_width=w)

        self.add(eq1, txt1)
        self.wait(0.1)
        eq2_1 = eq2[0].copy().move_to(ORIGIN, coor_mask=RIGHT)
        eq2_2 = eq2[:3].copy().move_to(ORIGIN, coor_mask=RIGHT)
        self.play(FadeIn(eq2_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_1, eq2_2[0]),
                  Succession(Wait(0.4), FadeIn(eq2_2[1:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq2_2, eq2[:3]),
                  Succession(Wait(0.4), FadeIn(eq2[3:])))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[4][0], eq3[0][0], eq2[5][:3], eq3[1][:3], eq2[5][5:8], eq3[1][3:6]),
                  mh.rtransform(eq2[5][9:12], eq3[1][:3], eq2[5][14:17], eq3[1][3:6]),
                  mh.rtransform(eq2[6][1:6], eq3[1][:5], eq2[6][8], eq3[1][5]),
                  mh.rtransform(eq2[6][10:15], eq3[1][:5], eq2[6][17], eq3[1][5]),
                  FadeOut(eq2[4][1:3]),
                  FadeOut(eq2[4][3], eq2[5][3:5], eq2[5][8]),
                  FadeOut(eq2[5][12:14], shift=mh.diff(eq2[5][11], eq3[1][2])),
                  FadeOut(eq2[6][0], shift=mh.diff(eq2[6][1], eq3[1][0])),
                  FadeOut(eq2[6][6:8], shift=mh.diff(eq2[6][5], eq3[1][4])),
                  FadeOut(eq2[6][9], shift=mh.diff(eq2[6][10], eq3[1][0])),
                  FadeOut(eq2[6][15:17], shift=mh.diff(eq2[6][14], eq3[1][4])),
                  FadeOut(eq2[7]),
                  run_time=1.5
                  )
        self.wait(0.1)
        self.play(Succession(Wait(0.2), mh.rtransform(eq3[0], eq4[0], eq3[1][:2],
                                                      eq4[1][:2], eq3[1][-1], eq4[1][-1])),
                  FadeIn(eq4[1][2:-1]), FadeOut(eq3[1][2:-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq2[:2], eq5[:2], eq2[2][-5:], eq5[2][:]),
                  mh.rtransform(eq4[1][:], eq5[2][:]),
                  FadeOut(eq2[2][:-5], shift=mh.diff(eq2[2][-5], eq5[2][0])),
                  FadeOut(eq2[3], eq4[0], shift=mh.diff(eq4[1][0], eq5[2][0])),
                  run_time=1.5)
        self.wait(0.1)
        self.play(mh.rtransform(eq5[:2], eq6[:2], eq5[2][:], eq6[2][2:-1]),
                  Succession(Wait(0.4), FadeIn(eq6[2][:2], eq6[2][-1])))
        self.play(FadeOut(eq1, txt1, rate_func=linear))
        self.play(eq6[2].animate.align_to(eq6[0], RIGHT),
                  eq6[0].animate.align_to(eq6[2], LEFT))
        self.wait(0.1)
        eq7_1 = mh.align_sub(eq7[:3].copy(), eq7[1], eq6[1])
        self.play(mh.rtransform(eq6[1], eq7_1[1], eq6[2][:5], eq7_1[0][:5], eq6[2][-2:], eq7_1[0][-2:],
                                eq6[0][:5], eq7_1[2][:5], eq6[0][-3:], eq7_1[2][-3:]),
                  mh.fade_replace(eq6[2][5], eq7_1[0][5], coor_mask=RIGHT),
                  FadeOut(eq6[0][5], target_position=eq7_1[2][5]),
                  FadeOut(eq6[0][6], target_position=eq7_1[2][5]),
                  )
        eq7_2 = mh.align_sub(eq7[:5].copy(), eq7[1], eq6[1]).move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq7_1, eq7_2[:3]),
                  Succession(Wait(0.5), FadeIn(eq7_2[3:])))
        self.play(mh.rtransform(eq7_2, eq7[:5]),
                  Succession(Wait(0.5), FadeIn(eq7[5:])))
        self.wait(0.1)
        self.play(Succession(Wait(0.3), AnimationGroup(mh.rtransform(eq7[:2], eq8_1[:2]),
                  mh.rtransform(eq7[-2], eq8_1[1]),
                  FadeOut(eq7[-1], shift=mh.diff(eq7[-2], eq8_1[1])),
                  FadeIn(eq8_1[2], shift=mh.diff(eq7[-2], eq8_1[1])), run_time=1.5)),
                  FadeOut(eq7[2:-2]),
                  )
        self.wait(0.1)
        eq8_2 = eq8[0][2].copy().move_to(eq8_1[0][4:6], coor_mask=RIGHT)
        self.play(FadeOut(eq8_1[0][4:6]), FadeIn(eq8_2), rate_func=linear)
        self.wait(0.1)
        self.play(mh.rtransform(eq8_1[1:], eq8[1:], eq8_1[0][2:4], eq8[0][:2], eq8_1[0][-2], eq8[0][-1],
                                eq8_2, eq8[0][2]),
                  FadeOut(eq8_1[0][:2]),
                  FadeOut(eq8_1[0][-1], shift=mh.diff(eq8_1[0][-2], eq8[0][-1])),
                  )
        self.wait(0.1)
        self.play(eq8.animate.next_to(eq9, UP, coor_mask=UP, buff=0.5),
                  FadeIn(eq9))
        self.wait(0.1)
        self.play(AnimationGroup(mh.rtransform(eq8[1], eq10[1], eq8[0][:], eq10[0][:4], eq8[2][:2], eq10[2][:2],
                                eq8[2][-6:-1], eq10[2][2:7], eq8[2][-1], eq10[2][-1]),
                  mh.rtransform(eq9[1], eq10[1], eq9[0][:], eq10[0][-4:], eq9[2][:2], eq10[2][:2],
                                eq9[2][-6:], eq10[2][-6:]), run_time=1.8),
                  Succession(Wait(1), FadeIn(eq10[0][4], eq10[2][7]))
                  )
        self.wait(0.1)


        br1 = BraceLabel(eq10[2][2:-1], r'=0\ {\sf if\ } t\ge T', label_constructor=mh.mathlabel_ctr2, font_size=60,
                         brace_config={'color': WHITE},
                         brace_direction=UP).set_z_index(2)
        br1.label.set_color(WHITE)
        br1.label[0][1].set_color(BLUE)
        br1.label[0][2:4].set_color(TEAL)
        VGroup(br1.label[0][4], br1.label[0][6]).set_color(RED)
        br_eq = eq_shadow(br1.label)
        self.play(FadeIn(br1.brace, br_eq))
        self.wait(0.1)
        eq11[2][2:-1].move_to(eq10[2][2:-1], coor_mask=RIGHT)
        eq11[2][-1].move_to(eq10[2][-1], coor_mask=RIGHT)
        self.play(FadeOut(br1.brace, br_eq),
                  mh.rtransform(eq10[0][:], eq11[0][1:-1],
                                eq10[2][:2], eq11[2][:2], eq10[2][-1], eq11[2][-1]),
                  mh.fade_replace(eq10[1], eq11[1]),
                  FadeIn(eq11[0][0], eq11[0][-1]),
                  FadeOut(eq10[2][2:-1]),
                  FadeIn(eq11[2][2:-1]))
        self.wait(0.1)
        self.play(mh.rtransform(eq11[:2], eq12[:2], eq11[2][2:4], eq12[2][:2],
                                eq11[2][6:9], eq12[2][4:7]),
                  mh.fade_replace(eq11[2][0], eq12[2][2]),
                  mh.stretch_replace(eq11[2][1], eq12[2][3]),
                  mh.stretch_replace(eq11[2][-1], eq12[2][-1]),
                  FadeOut(eq11[2][4:6], shift=mh.diff(eq11[2][6], eq12[2][4])*RIGHT),
                  FadeOut(eq11[2][9], shift=mh.diff(eq11[2][8], eq12[2][6])*RIGHT),
                  run_time=1.5)
        self.wait(0.1)
        eq12_1 = MathTex(r'\infty = T', font_size=80)
        VGroup(eq12_1[0][0], eq12_1[0][2]).set_color(RED)
        eq12_1 = eq_shadow(eq12_1, bg_stroke_width=w)
        mh.align_sub(eq12_1, eq12_1[0][-1], eq12[2][-2]).align_to(eq12[2][-4], LEFT)
        self.play(mh.rtransform(eq12[2][-2], eq12_1[0][-1]),
                  mh.fade_replace(eq12[2][-3], eq12_1[0][1]),
                  mh.fade_replace(eq12[2][-4], eq12_1[0][0]),
                  eq12[2][-1].animate.shift(mh.diff(eq12[2][-2], eq12_1[0][-1]))
                  )
        self.wait(0.1)
        eq13 = VGroup(*eq12[:2], eq12[2][:4], eq12_1[0], eq12[2][-1:])
        eq14 = MathTex(r'\sf no\ coupling', font_size=80, color=TEAL)
        eq14 = eq_shadow(eq14, bg_stroke_width=w)
        mh.align_sub(eq14, eq14[0][0], eq12_1, ORIGIN, aligned_edge=DOWN, buff=0).align_to(eq12_1, LEFT)
        eq14 = VGroup(*eq12[:2].copy(), eq12[2][:4].copy(), eq14[0], eq12[2][-1:].copy())
        eq14[-1].shift((eq14[-2].get_right() - eq12_1[0][-1].get_right())*RIGHT)
        eq14.move_to(ORIGIN, coor_mask=RIGHT)
        self.play(mh.rtransform(eq13[:3], eq14[:3], eq13[-1], eq14[-1]),
                  Succession(Wait(0.3), AnimationGroup(
                      FadeOut(eq13[3]), FadeIn(eq14[3]))),
                  )
        self.wait(0.1)


        eq15 = MathTex(r'\le', r'0', color=BLUE, font_size=80)
        eq15 = eq_shadow(eq15, bg_stroke_width=w)
        mh.align_sub(eq15, eq15[0], eq14[1])
        eq15 = VGroup(*eq14[:2].copy(), eq15[1])
        self.play(mh.rtransform(eq14[:2], eq15[:2]),
                  FadeOut(eq14[2:]),
                  FadeIn(eq15[2]))
        eq16 = MathTex(r'f(x)', r'=', r'f(y)', font_size=80)
        eq16 = eq_shadow(eq16, bg_stroke_width=w)
        eq16 = VGroup(eq15[0][1:5].copy().move_to(eq16[0]), eq16[1], eq15[0][6:10].copy().move_to(eq16[2]))
        mh.align_sub(eq16, eq16[0], eq15[0][1:5]).move_to(ORIGIN, coor_mask=RIGHT)
        self.wait(0.1)
        eq15_1 = eq16[1].copy().move_to(eq15[1], coor_mask=RIGHT)
        self.play(FadeOut(eq15[0][0], eq15[0][-1], eq15[1]), FadeIn(eq15_1))
        self.wait(0.1)
        eq16.shift(UP*1.5)
        self.play(mh.rtransform(eq15[0][1:5], eq16[0], eq15[0][6:10], eq16[2], eq15_1, eq16[1]),
                  FadeOut(eq15[0][5], target_position=eq16[1]),
                  FadeOut(eq15[2], shift=mh.diff(eq15[1], eq16[1]))
                  )

        eq17 = MathTex(r'f', r'=', r'\sf constant', font_size=100)
        eq17[0].set_color(RED_B)
        eq17[2].set_color(TEAL)
        eq17 = eq_shadow(eq17, bg_stroke_width=w)
        mh.align_sub(eq17, eq17[1], eq16[1]).move_to(ORIGIN, coor_mask=RIGHT)
        self.wait(0.1)
        self.play(mh.rtransform(eq16[0][0], eq17[0][0], eq16[1], eq17[1]),
                  FadeOut(eq16[0][1:], eq16[2]),
                  FadeIn(eq17[2]))



        self.wait()

class Laplacian(AliceBobEqual):
    def construct(self):
        MathTex.set_default(font_size=80, stroke_width=2)
        eq1 = MathTex(r'\frac{\partial^2}{\partial x^2}', r'f(x,y)', r'+',
                     r'\frac{\partial^2}{\partial y^2}', r'f(x,y)', r'=', r'0').set_z_index(4)
        eq2 = MathTex(r'\left(', r'\frac{\partial^2}{\partial x^2}', r'+',
                     r'\frac{\partial^2}{\partial y^2}', r'\right)', r'f(x,y)', r'=', r'0').set_z_index(4)
        eq3 = MathTex(r'\frac{d}{dt}', r'\mathbb E[f(B_t)]', r'=', r'0').set_z_index(4)
        eq4 = MathTex(r'\mathbb E[f(B_t)]', r'=', r'f(B_0)').set_z_index(4)
        eq5 = MathTex(r'f', r'=', r'\sf constant', font_size=100).set_z_index(4)

        VGroup(eq1[0][-2], eq1[1][2], eq1[4][2], eq2[1][-2], eq2[5][2],
               eq3[0][3], eq3[1][-3]).set_color(RED)
        VGroup(eq1[1][4], eq1[3][-2], eq1[4][4], eq2[3][-2], eq2[5][4]).set_color(GREEN)
        VGroup(eq1[0][1], eq1[0][-1], eq1[3][1], eq1[3][-1], eq1[-1],
               eq2[1][1], eq2[1][-1], eq2[3][1], eq2[3][-1], eq2[-1], eq3[-1],
               eq4[2][3]).set_color(BLUE)
        VGroup(eq1[0][0], eq1[0][3], eq1[3][0], eq1[3][3],
               eq2[1][0], eq2[1][3], eq2[3][0], eq2[3][3], eq3[0][0], eq3[0][2]).set_color(ORANGE)
        VGroup(eq1[1][0], eq1[4][0], eq2[5][0], eq3[1][2], eq4[2][0], eq5[0]).set_color(RED_B)
        VGroup(eq3[1][0]).set_color(YELLOW)
        VGroup(eq3[1][4], eq4[2][2]).set_color(PURPLE)
        eq5[2].set_color(TEAL)

        box1 = SurroundingRectangle(eq1, stroke_width=0, stroke_opacity=0, fill_opacity=0.6, fill_color=BLACK,
                                    corner_radius=0.15, buff=0.2)
        VGroup(box1, eq1).to_edge(DOWN, buff=0.15)

        mh.align_sub(eq2, eq2[-2], eq1[-2], coor_mask=UP)
        mh.align_sub(eq3, eq3[2], eq2[-2], coor_mask=UP)
        mh.align_sub(eq4, eq4[1], eq3[2])
        eq4 = VGroup(eq3[1].copy(), *eq4[1:]).move_to(ORIGIN, coor_mask=RIGHT)
        mh.align_sub(eq5, eq5[1], eq4[1], coor_mask=UP)

        self.add(eq1, box1)
        self.wait(0.1)
        self.play(mh.rtransform(eq1[0], eq2[1], eq1[1], eq2[-3], eq1[2], eq2[2], eq1[3], eq2[3]),
                  mh.rtransform(eq1[4], eq2[-3], eq1[-2:], eq2[-2:]),
                  FadeIn(eq2[0], eq2[-4]),
                  run_time=1.5)
        self.wait(0.1)
        eq3_1 = eq3.copy()
        mh.align_sub(eq3, eq3[2], eq2[-2])
        eq3[0].shift(LEFT*0.4)
        self.play(FadeOut(eq2[:5]),
                  mh.rtransform(eq2[5][:2], eq3[1][2:4], eq2[5][-1], eq3[1][-2], eq2[6:], eq3[2:]),
                  mh.fade_replace(eq2[5][2:-1], eq3[1][4:-2], coor_mask=RIGHT),
                  FadeIn(eq3[1][:2], shift=mh.diff(eq2[5][0], eq3[1][2])),
                  FadeIn(eq3[1][-1]),
                  FadeIn(eq3[0]))
        #self.play(mh.transform(eq3, eq3_1))
        self.wait(0.1)
        self.play(mh.rtransform(eq3[1:3], eq4[:2]),
                  FadeOut(eq3[0], shift=LEFT*1.6),
                  mh.fade_replace(eq3[-1], eq4[-1]))
        self.wait(0.1)
        shift = mh.diff(eq4[1], eq5[1])
        self.play(Succession(Wait(0.4),
                             AnimationGroup(mh.rtransform(eq4[1], eq5[1], eq4[0][2], eq5[0][0]),
                                            FadeOut(eq4[2], shift=shift),
                                            FadeIn(eq5[2], shift=shift))),
                  FadeOut(eq4[0][:2], eq4[0][3:]),
                  )

        self.wait()


class KleinBottle(ThreeDScene):
    def construct(self):
        def klein_bottle(u, v):
            cos = math.cos
            sin = math.sin
            v = 1. - v
            u = 1.751 - u
            if u > 1.:
                u -= 1.
                v = 1.5 - v
                v -= math.floor(v)
            #u -= math.floor(u)
            u = u * 2 * PI
            v = v * 2 * PI
            half = (u < PI)

            if half:
                x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(u) * cos(v)
                y = 8 * sin(u) + (2 * (1 - cos(u) / 2)) * sin(u) * cos(v)
            else:
                x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(v + PI)
                y = 8 * sin(u)

            z = -2 * (1 - cos(u) / 2) * sin(v)
            return np.array([x, y, z])

        p = (klein_bottle(0., 0.) + klein_bottle(0., 0.5))/2

        surface = Surface(
            lambda u, v: klein_bottle(u, v),
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(50, 50),
            fill_opacity=0.65,
            fill_color=ORANGE,
            checkerboard_colors=False,
        )
        surface.shift(-p)
        surface.scale(0.2, about_point=ORIGIN)
        surface.rotate(angle=PI/2, axis=RIGHT, about_point=ORIGIN)
        surface.rotate(angle=100*DEGREES, axis=IN, about_point=ORIGIN)
        surface.move_to(ORIGIN, coor_mask=OUT)
        p = ORIGIN

        #surface.move_to(ORIGIN)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        self.play(Create(surface, run_time=1.5, rate_func=rate_functions.double_smooth))
        #self.add(surface)
        self.play(Rotate(surface, 2*PI, rate_func=linear, run_time=2, axis=IN, about_point=p))

class Torus(ThreeDScene):
    def construct(self):
        def torus(u, v):
            cos = math.cos
            sin = math.sin
            u = u * 2 * PI
            v = v * 2 * PI
            a = 3.
            b = 6.
            x = (a*cos(v)+b)*cos(u)
            y = (a*cos(v)+b)*sin(u)
            z = a*sin(v)
            return np.array([x, y, z])

        surface = Surface(
            lambda u, v: torus(u, v),
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(50, 50),
            fill_opacity=0.65,
            fill_color=RED,
            checkerboard_colors=False,
        ).scale(0.2)
        surface.rotate(angle=PI/2, axis=RIGHT, about_point=surface.get_center())
        surface.move_to(ORIGIN)
        surface.rotate(angle=135*DEGREES, axis=IN, about_point=surface.get_center())

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        self.play(Create(surface, run_time=1, rate_func=rate_functions.double_smooth))
        self.add(surface)
        self.play(Rotate(surface, 2*PI, rate_func=linear, run_time=2, axis=IN))

class SphereRot(ThreeDScene):
    def construct(self):
        def torus(u, v):
            cos = math.cos
            sin = math.sin
            u = u * PI
            v = v * 2 * PI
            a = 8.
            x = a*cos(v)*sin(u)
            y = a*sin(v)*sin(u)
            z = a*cos(u)
            return np.array([x, y, z])

        surface = Surface(
            lambda u, v: torus(u, v),
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(15, 30),
            fill_opacity=0.65,
            checkerboard_colors=[BLUE, ManimColor(BLUE.to_rgb()*0.9)],
        ).scale(0.2)
        #surface.rotate(angle=PI/2, axis=RIGHT, about_point=surface.get_center())
        surface.move_to(ORIGIN)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-70 * DEGREES)

        self.play(Create(surface, run_time=1, rate_func=rate_functions.double_smooth))
        self.add(surface)
        self.play(Rotate(surface, 2*PI, rate_func=linear, run_time=4, axis=IN))

class PoincareBM(Scene):
    seed = 1 # 2, 1
    col = RED
    p0 = RIGHT * 1.
    max_time=60.
    path_time = True
    rate = 1000.
    sw = 4.
    radius = 3.

    def construct(self):
        np.random.seed(self.seed)
        #r = config.frame_y_radius
        r = self.radius
        sigma = 0.01 * r

        p = self.p0.copy()
        dot = Dot(radius=0.1, fill_color=ManimColor(self.col.to_rgb() * 0.5 + WHITE.to_rgb()*0.5)).set_z_index(4).move_to(p)
        steps = 0
        print(np.inner(p, p))
        pts = []
        times = []
        a = np.inner(p, p) / (r*r)
        t = 0.
        while a < 0.99:# and t < 20000:
            pts.append(p.copy())
            times.append(t)
            p += (np.random.normal() * RIGHT + np.random.normal() * UP) * sigma
            scale = 2 / (1 - a)
            dt = scale*scale
            t += dt
            steps += 1
            if steps % 1000 == 0:
                print(steps)
            a = np.inner(p, p) / (r*r)

        times0 = np.array(times) / self.rate
        t0 = times0[-1]
        print('total time:', t0)
        run_time = min(t0, self.max_time)
        print('run time:', run_time)
        times1 = np.linspace(0., 1., len(times))

        if self.path_time:
            f = lambda s: np.interp(s * run_time, times0, times1)
        else:
            f = linear
            run_time=2.

        circ = Circle(radius=r*1.05, fill_opacity=0, stroke_opacity=1, stroke_color=WHITE, stroke_width=2)

        path = VGroup(color=self.col, stroke_width=self.sw)
        path.add_points_as_corners(pts)
        print(steps)
        dot.add_updater(lambda m: m.move_to(path.get_end()))
        self.add(circ, dot)
        self.play(Create(path, rate_func=f), run_time=run_time, rate_func=f)
        #self.play(MoveAlongPath(dot, path, run_time=run_time, rate_func=f))


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True, 'fps': 15}):
        Percolation().render()
