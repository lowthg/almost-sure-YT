from manim import *
import numpy as np
import math
import sys
import scipy as sp
from matplotlib.font_manager import font_scalings
from numpy.random.mtrand import Sequence

sys.path.append('../')
import manimhelper as mh

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

def get_box(ax, i, h):
    bl = ax.coords_to_point(i + 0.15, 0)
    ur = ax.coords_to_point(i + 1, h)
    box = Rectangle(width=(ur - bl)[0], height=(ur - bl)[1], fill_opacity=1, fill_color=BLUE,
                    stroke_width=2, stroke_opacity=1, stroke_color=WHITE).set_z_index(0.9)
    box.next_to(bl, UR, buff=0)
    return box


class ProbFinal(Scene):
    ylen = 4
    height = 10/55

    def get_axes(self, x_str, x_max=11):
        xlen = 8.
        ax = Axes(x_range=[0, x_max], y_range=[0, self.height * 1.2], x_length=xlen, y_length=self.ylen,
                  axis_config={'color': WHITE, 'stroke_width': 5, 'include_ticks': False,
                               "tip_width": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               "tip_height": 0.5 * DEFAULT_ARROW_TIP_LENGTH,
                               },
                  #                  shade_in_3d=True,
                  ).set_z_index(1)
        labely = Tex(r'\sf probability', color=YELLOW, font_size=40).set_z_index(2)
        labely.next_to(ax.y_axis.get_end(), RIGHT)
        labelx = Tex(*x_str, color=YELLOW, font_size=40).set_z_index(2)
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
    num_steps = 3
    heights1 = [_/55 for _ in range(10, 0, -1)]

    def plot_extra(self, ax):
        pass

    def construct(self):
        heights0 = [0.1] * 10
        heights1 = self.heights1

        ax, labely, labelx = self.get_axes([r'\sf first', r'place'], x_max=12)
        ax.set_z_index(10)

        boxes0 = [get_box(ax, i, h) for i, h in enumerate(heights0)]
        boxes1 = [get_box(ax, i, h) for i, h in enumerate(heights1)]
        p0 = ax.coords_to_point(0, 0.1)
        line1 = DashedLine(p0, ax.coords_to_point(11, 0.1), stroke_width=3, stroke_color=YELLOW).set_z_index(6)
        label1 = MathTex(r'0.1', font_size=40, color=YELLOW).next_to(line1.get_end(), UL, buff=0.1).shift(LEFT*0.2)
        p0 = ax.coords_to_point(0, 0.1818)
        line2 = DashedLine(p0, ax.coords_to_point(11, 0.1818), stroke_width=3, stroke_color=YELLOW).set_z_index(6)
        label2 = MathTex(r'0.18', font_size=40, color=YELLOW).next_to(line2.get_end(), UL, buff=0.1).shift(LEFT*0.2)
        labels = [self.get_label(ax, i, i + 1) for i in range(10)]
        label3 = labels[-1]
        VGroup(label3[0][0], label3[1][0]).shift(RIGHT * 0.02)
        VGroup(label3[0][1], label3[1][1]).shift(LEFT * 0.02)

        self.add(ax, labely, labelx, *boxes0, line1, label1, *labels)
        self.wait(0.1)
        self.plot_extra(ax)
        self.play(*[ReplacementTransform(b1, b2) for b1, b2 in zip(boxes0, boxes1)],
                  FadeIn(line2, label2), run_time=1.6)

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

class StationaryConv(StationaryPlot):
    height = 10/55 * 1.13
    ylen = 4 * 1.13
    num_steps = 20
    heights1 = [0.1] * 10

    def plot_extra(self, ax):
        lines = []
        for i in range(10):
            h = (10-i)/55
            line1 = Line(ax.coords_to_point(i+0.15, h), ax.coords_to_point(i+1, h),
                         stroke_color=RED, stroke_width=6, stroke_opacity=1).set_z_index(10)
            lines.append(*line1)
        self.play(FadeIn(*lines))


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True, 'fps': 15}):
        AliceBobDraw().render()
