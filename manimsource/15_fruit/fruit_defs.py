from manim import *

col_num = BLUE
col_x = (BLUE-WHITE)*0.7 + WHITE
col_a = (ORANGE-WHITE)*0.7 + WHITE
col_op = (PURPLE-WHITE) * 0.5 + WHITE
col_f = (RED-WHITE)*0.8 + WHITE
col_txt = ManimColor(r'#FFAC2B')
lcol = PURPLE * 0.5 + WHITE * 0.5

colx = RED
coly = YELLOW
colz = GREEN
coln = col_num

colx = col_x + 0.5 * (RED - col_x)
coly = col_x + 0.5 * (YELLOW - col_x)
colz = col_x + 0.5 * (GREEN - col_x)

col_pt1 = YELLOW
col_pt2 = ORANGE
col_pt3 = PINK
col_t = WHITE + 0.8 * (GREEN - WHITE)

def eq_shadow(eq: VGroup, fg_z_index=4., bg_z_index=0., bg_color=BLACK, bg_stroke_width=10.):
    res = VGroup()
    for eq1 in eq:
        elem = VGroup()
        for eq2 in eq1:
            elem.add(VGroup(eq2.set_z_index(fg_z_index),
                            eq2.copy().set_z_index(bg_z_index).set_color(bg_color).set_stroke(width=bg_stroke_width)))
        res.add(elem)
    return res

class ClickBaitEq(Scene):
    def __init__(self, *args, **kwargs):
        config.background_color = WHITE
        Scene.__init__(self, *args, **kwargs)

    @staticmethod
    def eq1():
        str0 = r' {\vbox to 1em {\vfil\hbox to 1.18em{}\vfil} } '
        str1 = r'\frac{}{' + str0 + r'+' + str0 + r'}'
        eq = MathTex(str1, r'+', str1, r'+', str1, r'=4', font_size=65, color=BLACK, stroke_color=BLACK, stroke_width=2)
        eq.shift(LEFT*0.001*config.frame_width + UP*0.053739*config.frame_height).set_z_index(4)
        return eq

    def construct(self):
        self.add(self.eq1())

class Equation1(Scene):
    @staticmethod
    def eq1():
        eq1 = MathTex(r'\frac{x}{y+z}', r'+', r'\frac{y}{x+z}', r'+', r'\frac{z}{x+y}', r'=4',
                      font_size=65, stroke_width=2)
        VGroup(eq1[0][0], eq1[2][2], eq1[4][2]).set_color(colx)
        VGroup(eq1[0][2], eq1[2][0], eq1[4][4]).set_color(coly)
        VGroup(eq1[0][4], eq1[2][4], eq1[4][0]).set_color(colz)
        eq1[-1][-1].set_color(coln)
        return eq1

    @staticmethod
    def eq2():
        eq1 = Equation1.eq1()
        eq2 = ClickBaitEq.eq1().set_color(WHITE)
        for i, eq in enumerate(eq1[:]):
            eq.move_to(eq2[i])
        for i in (0, 2, 4):
            mh.align_sub(eq1[i], eq1[i][1], eq2[i][0])
            mh.align_sub(eq1[i][2:], eq1[i][3], eq2[i][1])
            eq1[i][1].become(eq2[i][0])
            eq1[i][3].become(eq2[i][1])
            eq1[i][2].move_to(eq1[i][1].get_left()*0.55 + eq1[i][3].get_left()*0.45, coor_mask=RIGHT)
            eq1[i][4].move_to(eq1[i][1].get_right()*0.55 + eq1[i][3].get_right()*0.45, coor_mask=RIGHT)
            eq1[i][0].shift(UP*0.2)
        eq1 = eq_shadow(eq1, bg_stroke_width=10, bg_color=BLACK)
        return eq1

    def construct(self):
        self.add(self.eq2())
        #self.play(mh.transform(eq1[0][1], eq2[0][0]))

