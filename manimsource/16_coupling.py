from manim import *
import numpy as np
import math
import sys
import scipy as sp
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
        alice = ImageMobject(r'../media/wifejak.png').set_z_index(3).scale(alice_scale)
        a = alice.pixel_array.copy()
        alice = ImageMobject(a[:-82,:,:]).set_z_index(3).scale(alice_scale)

        bob = ImageMobject(r'../media/husbandjak.png').set_z_index(3).scale(bob_scale)
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
        self.play(FadeIn(box5, rate_func=linear))
        self.wait(0.1)
        self.play(FadeOut(box5, rate_func=linear))
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
    def __init__(self, *args, **kwargs):
        if not config.transparent:
           config.background_color = GREY
        Scene.__init__(self, *args, **kwargs)

    def construct(self):
        wid = mh.pos(RIGHT)[0]*2
        hei = mh.pos(UP)[1]*2
        fs2 = 70
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
