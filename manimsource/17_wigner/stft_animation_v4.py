"""
Short-Time Fourier Transform (STFT) Animation  –  v2
=====================================================

Signal
------
    x(t) = sin(2π u t)    u = 10 Hz   for  0 < t < 5
                           u = 25 Hz   for  5 < t < 10
                           u = 50 Hz   for 10 < t < 15
                           u = 100 Hz  for 15 < t < 20
    x(t) = 0 elsewhere

sin is used so that every segment starts and ends at zero → continuity.

STFT definition
---------------
    STFT(t, f) = ∫ x(τ) · w(τ − t) · e^{−j2πfτ} dτ

    w(τ) = 1 if |τ| < B,  0 otherwise.

Exact closed-form evaluation
-----------------------------
On each segment [a, b] with frequency u the integral over the window
[t−B, t+B] ∩ [a, b] = [α, β] is:

    ∫_α^β sin(2πuτ) e^{−j2πfτ} dτ
        = (I₁ − I₂) / (2j)

where
    I₁ = ∫_α^β e^{j2π(u−f)τ} dτ  =  { (β−α)                             if u=f
                                       { (e^{j2π c₁ β} − e^{j2π c₁ α}) / (j2π c₁)  otherwise,  c₁ = u−f

    I₂ = ∫_α^β e^{−j2π(u+f)τ} dτ =  (e^{−j2π c₂ β} − e^{−j2π c₂ α}) / (−j2π c₂),   c₂ = u+f  (always > 0)

Colour map
----------
    Jet (blue → cyan → green → yellow → red).
    Intensity is normalised to the 99th-percentile of each frame, then
    capped at 1, so no single spike can bleach the entire picture.

Run with
--------
    manim -pqm stft_animation_v2.py STFTAnimation   # 720p 30fps (recommended)
    manim -pqh stft_animation_v2.py STFTAnimation   # 1080p 60fps
"""

from manim import *
import numpy as np
import matplotlib
import matplotlib.cm as cm


# ── colour-map helper ────────────────────────────────────────────────────────

def _get_cmap(name: str):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)


# ── scene ────────────────────────────────────────────────────────────────────

class STFTAnimation(Scene):

    # ── tweakable parameters ────────────────────────────────────────────────
    T_MIN,  T_MAX  = 0.0,  20.0     # time  axis  [s]
    F_MIN,  F_MAX  = 0.0, 200.0     # freq  axis  [Hz]
    # 3× resolution grids — aliasing-free by design:
    #   NT=601  →  t_step = 1/30 s  →  segment edges 0,5,10,15,20 s  on-grid (÷150)
    #   NF=601  →  f_step = 1/3 Hz  →  signal freqs 10,25,50,100 Hz  on-grid (÷30/75/150/300)
    # Both are exact integer multiples of the critical 1 Hz / 0.1 s spacings,
    # so sinc(c1·2B) = sinc(0) = 1 at every signal frequency for all B.
    NT,     NF     = 801,  1601
    # N_FRAMES       = 80             # animation frames (B sweeps from 0.025 to 1.000)
    # FRAME_DT       = 0.08           # scene seconds per frame  (~12 fps playback)
    GAMMA          = 0.50           # colour gamma (< 1 → boost dim regions)
    CAP_PERCENTILE = 99.0           # intensities above this percentile → capped at 1

    # Signal segments  (t_start, t_end, frequency_Hz)
    SEGMENTS = [(0, 5, 10), (5, 10, 25), (10, 15, 50), (15, 20, 100)]

    # ── exact vectorised STFT ────────────────────────────────────────────────

    def _compute_stft_frame(
        self,
        B: float,
        t_grid: np.ndarray,
        f_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Return |STFT(t, f)| for all t in t_grid and f in f_grid.

        Uses the closed-form integral of sin(2π u τ) e^{−j2π f τ} over the
        rectangular window [t−B, t+B] intersected with each signal segment.
        """
        NT = len(t_grid)
        NF = len(f_grid)
        stft = np.zeros((NT, NF), dtype=np.complex128)

        EPS = 1e-11   # threshold for treating c₁ as zero (u == f)

        for (a, b, u) in self.SEGMENTS:
            # Window limits clipped to this segment
            alpha = np.maximum(t_grid - B, a)   # (NT,)
            beta  = np.minimum(t_grid + B, b)   # (NT,)
            valid = alpha < beta                # (NT,) bool

            if not valid.any():
                continue

            # Work only on rows where the window overlaps the segment
            av = alpha[valid, None]   # (Nv, 1)
            bv = beta [valid, None]   # (Nv, 1)

            c1 = (u - f_grid)[None, :]   # (1, NF)   c₁ = u − f
            c2 = (u + f_grid)[None, :]   # (1, NF)   c₂ = u + f  (always > 0)

            # ── I₁ = ∫ e^{j2π c₁ τ} dτ from av to bv ─────────────────────
            # c₁ may be zero when f == u
            I1 = np.where(
                np.abs(c1) > EPS,
                (  np.exp( 1j * 2*np.pi * c1 * bv)
                 - np.exp( 1j * 2*np.pi * c1 * av)
                ) / (1j * 2*np.pi * c1),
                bv - av          # limit as c₁ → 0
            )   # (Nv, NF)  complex

            # ── I₂ = ∫ e^{−j2π c₂ τ} dτ from av to bv ───────────────────
            # c₂ = u + f ≥ u > 0, so never zero for u > 0, f ≥ 0
            I2 = np.where(
                np.abs(c2) > EPS,
                (  np.exp(-1j * 2*np.pi * c2 * bv)
                 - np.exp(-1j * 2*np.pi * c2 * av)
                ) / (-1j * 2*np.pi * c2),
                bv - av
            )   # (Nv, NF)  complex

            stft[valid] += (I1 - I2) / (2j)

        return np.abs(stft)   # (NT, NF)  real magnitude

    # ── colour mapping with percentile cap ──────────────────────────────────

    def _to_rgb(
        self,
        mag: np.ndarray,
        cmap,
    ) -> np.ndarray:
        """
        Convert a (NT, NF) magnitude array to (NF, NT, 3) uint8 RGB.

        Normalisation:
          1. Find the CAP_PERCENTILE-th percentile as the reference maximum.
          2. Divide by that reference → values in [0, ∞).
          3. Clip to [0, 1]  →  anything above the cap → solid red.
          4. Apply gamma.
          5. Map through jet colourmap.
          6. Flip frequency axis so f = 0 is at the bottom of the image.
        """
        ref = np.percentile(mag, self.CAP_PERCENTILE)
        if ref < 1e-30:
            ref = 1.0   # avoid division by zero for silent frames

        normalised = np.clip(mag / ref, 0.0, 1.0)   # (NT, NF)
        gamma_corr = normalised ** self.GAMMA        # (NT, NF)

        # Transpose so rows = frequency, columns = time  →  (NF, NT)
        img2d = gamma_corr.T

        # Apply colourmap  →  (NF, NT, 4)  float in [0,1]
        rgb = (cmap(img2d)[..., :3] * 255).astype(np.uint8)

        return rgb[::-1]   # flip: low frequency at image bottom

    # ── construct ────────────────────────────────────────────────────────────

    def construct(self):

        cmap    = _get_cmap("jet")
        t_grid  = np.linspace(self.T_MIN, self.T_MAX, self.NT)
        f_grid  = np.linspace(self.F_MIN, self.F_MAX, self.NF)
        # B_values = np.linspace(0.025, 1.000, self.N_FRAMES)

        # ── Pre-compute all frames ──────────────────────────────────────────
        # print(f"\nPre-computing {self.N_FRAMES} exact STFT frames "
        #       f"(NT={self.NT}, NF={self.NF}) …")
        # frames = [
        #     self._to_rgb(self._compute_stft_frame(B, t_grid, f_grid), cmap)
        #     for B in B_values
        # ]
        # print("Done.\n")

        # ══════════════════════════════════════════════════════════════════════
        #                          Manim layout
        # ══════════════════════════════════════════════════════════════════════

        # ── Axes ─────────────────────────────────────────────────────────────
        ax = Axes(
            x_range=[self.T_MIN, self.T_MAX, 5],
            y_range=[self.F_MIN, self.F_MAX, 50],
            x_length=10.5,
            y_length=5.2,
            axis_config={
                "color": WHITE,
                "include_tip": True,
                "tip_width":  0.15,
                "tip_height": 0.15,
                "stroke_width": 1.8,
            },
            x_axis_config={
                "numbers_to_include": np.arange(0, 21, 5),
                "font_size": 22,
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, 201, 50),
                "font_size": 22,
            },
        ).shift(DOWN * 0.45 + LEFT * 0.2)

        ax_labels = ax.get_axis_labels(
            x_label=MathTex(r"t\ [\mathrm{s}]",  font_size=32),
            y_label=MathTex(r"f\ [\mathrm{Hz}]", font_size=32),
        )
        ax_labels[0].shift(DOWN*0.4)

        Text.set_default(font="sans-serif")

        # ── Title ─────────────────────────────────────────────────────────────
        title = VGroup(
            Text("Short-Time Fourier Transform", font_size=40, weight=BOLD),
            # Text("Rectangular window  ·  exact analytic STFT  ·  jet colourmap",
            #      font_size=19, color=LIGHT_GREY),
        ).arrange(DOWN, buff=0.08).to_edge(UP, buff=0.18)

        # ── Heatmap bounding box ──────────────────────────────────────────────
        p0 = ax.c2p(self.T_MIN, self.F_MIN)
        p1 = ax.c2p(self.T_MAX, self.F_MAX)
        img_w      = p1[0] - p0[0]
        img_h      = p1[1] - p0[1]
        img_center = np.array([(p0[0]+p1[0])/2, (p0[1]+p1[1])/2, 0.0])

        def make_image(data: np.ndarray) -> ImageMobject:
            obj = ImageMobject(data)
            obj.stretch_to_fit_width(img_w)
            obj.stretch_to_fit_height(img_h)
            obj.move_to(img_center)
            obj.set_z_index(0)
            return obj

        # ── Reference lines ───────────────────────────────────────────────────
        def h_dash(f, op=0.35):
            return DashedLine(
                ax.c2p(self.T_MIN, f), ax.c2p(self.T_MAX, f),
                color=WHITE, stroke_width=0.9, stroke_opacity=op,
                dash_length=0.14,
            )

        def v_dash(t, op=0.35):
            return DashedLine(
                ax.c2p(t, self.F_MIN), ax.c2p(t, self.F_MAX),
                color=WHITE, stroke_width=0.9, stroke_opacity=op,
                dash_length=0.14,
            )

        freq_refs = VGroup(*[h_dash(f) for f in [10, 25, 50, 100]]).set_z_index(2)
        seg_refs  = VGroup(*[v_dash(t) for t in [5, 10, 15]]).set_z_index(2)

        freq_labels = VGroup(*[
            MathTex(rf"{f}\,\mathrm{{Hz}}", font_size=17, color=LIGHT_GREY)
            .move_to(ax.c2p(self.T_MAX, f) + RIGHT * 0.55)
            for f in [10, 25, 50, 100]
        ]).set_z_index(2)

        # ── Colourbar ─────────────────────────────────────────────────────────
        N_CB = 256
        cb_data = np.zeros((N_CB, 20, 3), dtype=np.uint8)
        for row in range(N_CB):
            intensity = row / (N_CB - 1)
            rgba = cmap(intensity ** self.GAMMA)
            cb_data[row] = [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)]
        cb_data = cb_data[::-1]   # low at bottom

        cb_img = ImageMobject(cb_data)
        cb_img.stretch_to_fit_width(0.18)
        cb_img.stretch_to_fit_height(img_h)
        cb_x = ax.c2p(self.T_MAX, 0)[0] + 0.95
        cb_img.move_to([cb_x, img_center[1], 0]).set_z_index(2)

        cb_lo    = Text("low",       font_size=15, color=LIGHT_GREY).set_z_index(2).next_to(cb_img, DOWN, buff=0.06)
        cb_hi    = Text("high",      font_size=15, color=LIGHT_GREY).set_z_index(2).next_to(cb_img, UP,   buff=0.06)
        cb_title = Text("intensity", font_size=15, color=LIGHT_GREY, slant=ITALIC)
        cb_title.set_z_index(2).next_to(cb_hi, UP, buff=0.04)

        ax.set_z_index(1)
        ax_labels.set_z_index(1)
        title.set_z_index(3)

        # ── B-value display panel ─────────────────────────────────────────────
        def make_B_panel(B: float) -> VGroup:
            panel = VGroup(
                Text("window half-width", font_size=30, color=LIGHT_GREY),
                MathTex(rf"B = {B:.2f}\ \mathrm{{s}}", font_size=44, stroke_width=1.5),
                # Text(f"2B = {2*B:.3f} s  (time resolution)",
                #      font_size=16, color=GREY_B),
                # Text(f"1/(2B) ≈ {1/(2*B):.1f} Hz  (freq. resolution)",
                #      font_size=16, color=GREY_B),
            ).arrange(RIGHT, buff=0.2)

            panel.next_to(title, DOWN, buff=0.4)
            # panel.to_corner(UR, buff=0.30).shift(UP * 0.15)
            panel.set_z_index(4)
            return panel

        # ── Initial frame ─────────────────────────────────────────────────────
        # img_mob = make_image(frames[0])
        # B_panel = make_B_panel(B_values[0])

        self.add(
            # img_mob,
            ax, ax_labels,
            freq_refs, seg_refs, freq_labels,
            cb_img, cb_lo, cb_hi, cb_title,
            title,
            # B_panel,
        )
        # self.wait(0.6)

        bval = ValueTracker(0.025)
        def objfunc():
            B = bval.get_value()
            data = self._to_rgb(self._compute_stft_frame(B, t_grid, f_grid), cmap)
            new_img   = make_image(data)
            return new_img

        def panelfunc():
            B = bval.get_value()
            new_panel = make_B_panel(B)
            return new_panel

        frame = always_redraw(objfunc)
        panel = always_redraw(panelfunc)
        self.add(frame, panel)
        self.play(bval.animate.set_value(0.997), run_time=6.4)
        self.remove(frame, panel)
        frame = objfunc()
        panel = panelfunc()
        self.add(frame, panel)

        # Hold on final frame
        self.wait(2.5)
