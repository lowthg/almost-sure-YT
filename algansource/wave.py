from algan import *
import colorsys
sys.path.append('../../')
import alganhelper as ah
from algan.external_libraries.manim.utils.color.SVGNAMES import INDIGO

col_up = torch.tensor([1, .6, 0.])
col_dn = INDIGO[:3]

def setup_wave(xrange=(-5., 5.), npts=640, opacity=0.75):
    xmin, xmax = xrange

    n_col=4
    surfx = Surface(grid_height=n_col, grid_width=npts)
    px = surfx.get_descendants()[1]
    locx = px.location.clone()
    colx = px.color.clone()
    xx = locx[:, :, 0] * (xmax - xmin) / 2 + (xmax + xmin) / 2
    yx = (locx[:, :, 1] + 1) / 2
    colx[..., 4] = opacity
    px.set_non_recursive(color=colx)

    with Off():
        surfx.spawn()

    return px, xx, yx

def set_wave(p, xvals, vals, vals1, origin, right, up):
    yvals = torch.linspace(0., 1., 4)
    n = len(xvals)
    locx = p.location.clone()
    locx[...,:] = (xvals.repeat_interleave(4).view(1, -1, 1) * right +
                    (vals.view(n, 1) * yvals.view(1, 4)).reshape(1, n*4, 1) * up + origin)
    colx = p.color.clone()

    for i in range(n*4):
        lightness = min(0.15 + vals[i // 4] * 0.3 * yvals[i % 4], 0.8)
        colx[0, i, :3] = torch.tensor([*colorsys.hls_to_rgb(np.angle(vals1[i // 4]) / (2 * PI) + 0.5, lightness, 0.85)])

    p.set_non_recursive(location=locx, color=colx)


def time_evolution(psi, V, dt=1., dx=0.1, mass=1.):
    n = len(psi)
    k_space = torch.fft.fftfreq(n, dx) * (2 * np.pi)  # Momentum space grid
    T = 0.5 / mass * k_space**2  # Kinetic energy operator
    psi_k = torch.fft.fft(psi)
    psi_k = torch.exp(-1j * T * dt) * psi_k  # Evolve in momentum space
    psi = torch.fft.ifft(psi_k)  # Transform back to position space
    psi = torch.exp(-1j * V * dt) * psi  # Evolve due to potential
    return psi

class WaveEvolver:
    def __init__(self, xrange=(-5., 5.), npts=639, n_extend_left=200, n_extend_right=200,
                 n_scale=1, dt=0.001, mass=1., speed=PI):
        xmin, xmax = self.xrange = xrange
        self.npts = npts
        self.n_extend_left = n_extend_left
        self.n_extend_right = n_extend_right
        self.n_scale = 1
        npts1 = (npts + n_extend_left+n_extend_right-1)*n_scale+1
        i_x0 = n_extend_left * n_scale
        i_x1 = npts1 - n_extend_right * n_scale
        x_extend_left = (xmax - xmin) / (npts-1) * n_extend_left
        x_extend_right = (xmax - xmin) / (npts-1) * n_extend_right
        xmin1, xmax1 = (xmin - x_extend_left, xmax + x_extend_right)

        # origin, right, up, out, p, xsurf, y, shape, self.txt = setup_surf(xrange, prange, spawn=not just_wave)
        # assert shape == (npts, npts)

        self.dt = dt
        self.speed = speed
        self.mass = mass
        self.xvals1 = torch.linspace(xmin1, xmax1, npts1)
        self.psi = self.xvals1 * 0
        self.xvals = torch.linspace(xmin, xmax, npts)
        dx = (self.xvals[1] - self.xvals[0]).item()
        self.stride = (i_x0, i_x1, n_scale)
        self.V = self.xvals1 * 0
        self.dx1 = dx/n_scale

    def evolve(self, time_inc):
        ndt = math.ceil(time_inc / self.dt)

        if ndt > 0:
            # print(ndt)
            dt1 = time_inc / ndt
            for i in range(ndt):
                self.psi = time_evolution(self.psi, self.V, self.speed * dt1, self.dx1, mass=self.mass)

        return self.psi0()

    def psi0(self):
        psi0 = self.psi[self.stride[0]:self.stride[1]:self.stride[2]]
        return psi0

    def psift(self):
        phase = np.exp(1j * np.outer(self.xvals, self.xvals1))
        psi_k = torch.tensor(phase @ self.psi.numpy()) * self.dx1 / math.sqrt(2 * PI)
        return psi_k

    def orbit(self, anim=1):
        origin, right, up, out = self.dirs
        cam = Scene.get_camera()
        if anim == 1:
            with Seq():
                cam.orbit_around_point(origin, 60 * DEGREES, cam.get_right_direction())
        elif anim == 2:
            with Off():
                cam.orbit_around_point(origin, 60 * DEGREES, cam.get_right_direction())
            with Sync(run_time=3):
                cam.orbit_around_point(origin, -60 * DEGREES, out)
                cam.orbit_around_point(origin, -130 * DEGREES, cam.get_right_direction())
                self.txt[0].orbit_around_point(self.txt[0].get_center(), -90, RIGHT)
        else:
            orbit_time = 4.
            with Sync(run_time=orbit_time):
                cam.orbit_around_point(origin, 180 * DEGREES, out)
            with Sync(run_time=1.5):
                cam.orbit_around_point(origin, -70 * DEGREES, cam.get_right_direction())

    def set_gaussian(self, x=0., p = 0.):
        sigma = np.sqrt(0.5)  # Width of Gaussian
        self.psi = torch.exp(-(self.xvals1 - x) ** 2 / (4 * sigma ** 2) + 1j * p * self.xvals1)  # + xvals1 * 6 * 1j)

        # Normalize the wavefunction
        self.psi /= np.linalg.norm(self.psi) * np.sqrt(self.dx1)

def setup_cam():
    with Off():
        cam: Camera = Scene.get_camera()
        cam.set_distance_to_screen(13)
        cam.move_to(cam.get_center()*1.45)
        cam.set_euler_angles(70*DEGREES, 0*DEGREES, 60*DEGREES)
        light: PointLight = Scene.get_light_sources()[0]
        light.orbit_around_point(ORIGIN, -90, axis=OUT)
        light.move(UP*4)

def setup_surf(xrange=(-5., 5.), yrange=(-5., 5.), zrange=(-.3, .3), spawn=True,
               colors=None, stroke_color=RED_E, signal_vars=False, vars=None, no_spawn=False):
    xmin, xmax = xrange
    ymin, ymax = yrange
    zmin, zmax = zrange
    xlen = 12.
    ylen = 12.
    zlen = 6.
    ax = mn.ThreeDAxes([xmin, xmax * 1.05], [ymin, ymax * 1.1], [zmin, zmax*1.2], xlen, ylen, zlen,
                    axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                                 "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                 "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                 },
                    z_axis_config={'rotation': PI},
                    ).shift(mn.DL * 0.3 + mn.OUT*0.2)
    origin = torch.tensor(ax.coords_to_point(0, 0), dtype=ORIGIN.dtype)
    right = torch.tensor(ax.coords_to_point(1, 0), dtype=ORIGIN.dtype) - origin
    up = torch.tensor(ax.coords_to_point(0, 1), dtype=ORIGIN.dtype) - origin
    out = torch.tensor(ax.coords_to_point(0, 0, 1), dtype=ORIGIN.dtype) - origin
    if vars is None:
        eqstr = [r't', r'\omega'] if signal_vars else [r'X', r'P']
        txt1 = mn.MathTex(eqstr[0], stroke_width=2, font_size=60)
        txt2 = mn.MathTex(eqstr[1], stroke_width=2, font_size=60)
    else: txt1, txt2 = vars
    txt1.move_to(ax.coords_to_point(xmax * 1.1, 0)).rotate(-PI / 2, mn.RIGHT)
    txt2.move_to(ax.coords_to_point(0, ymax * 1.15)).rotate(-PI / 2, mn.RIGHT)
    txt2.rotate(PI / 2, mn.OUT)
    ax1 = ManimMob(ax)
    txt1 = ManimMob(txt1)
    txt2 = ManimMob(txt2)
    Group(*ax1.submobjects[:2], txt1, txt2).move(IN*0.11)

    if colors is None:
        colors = [
            Color(mn.RED_D.to_rgb() * 0.5 / .8),
            Color(mn.RED_E.to_rgb() * 0.5 / .8)
        ]
    surf = ah.surface_mesh(num_recs=64, rec_size=10, col1=colors[0], col2=colors[1], stroke_color=stroke_color,
                           fill_opacity=0.9, stroke_opacity=1)
    shape = (surf.grid_width, surf.grid_height)
    p = surf.get_descendants()[1]
    loc = p.location.clone()
    x = loc[:,:,0] * (xmax - xmin) / 2 + (xmax+xmin)/2
    y = loc[:,:,1] * (ymax - ymin) / 2 + (ymax+ymin)/2
    surf.scale(np.array([(xmax-xmin)*right[0]/2, (ymax-ymin)*up[1]/2, 1])).move_to(origin)
    ax2 = Group(txt1, txt2, ax1)
    if not no_spawn:
        with Off():
            ax2.spawn()
            if spawn:
                surf.spawn()

    return origin, right, up, out, p, x, y, shape, ax2

def wigner_fft(psi, x_min, x_max,
               p_min=None, p_max=None, p_n=None,
               i0=0, i1=-1, step=1,
               pad_factor=2):
    """
    Fast Wigner distribution using FFT with zero padding.

    psi : (nx,) complex
    x_min, x_max : position domain
    p_min, p_max, p_n : optional custom momentum grid
    pad_factor : padding multiplier (>=2 recommended)
    """

    nx = len(psi)
    dx = (x_max - x_min) / (nx-1)
    x = np.linspace(x_min, x_max, nx)
    if i1 < 0:
        i1 = nx
    nx1 = len(x[i0:i1:step])

    # ---- zero padding to avoid wraparound ----
    n_pad = pad_factor * nx
    psi_pad = np.zeros(n_pad, dtype=complex)
    start = (n_pad - nx) // 2
    psi_pad[start:start+nx] = psi

    W = np.zeros((nx1, n_pad), dtype=float)

    for j in range(nx1):
        j1 = i0 + j*step
        # correlation function
        C = np.zeros(n_pad, dtype=complex)

        for m in range(-nx//2, nx//2):
            jp = j1 + m
            jm = j1 - m
            if 0 <= jp < nx and 0 <= jm < nx:
                C[m + n_pad//2] = np.conj(psi[jp]) * psi[jm]

        # FFT over relative coordinate
        W_j = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(C)))
        W[j, ::-1] = np.real(W_j) * dx / (2*np.pi)

    # momentum grid from FFT
    # dp_fft = 2 * np.pi / (n_pad * dx)
    p_fft = np.fft.fftshift(np.fft.fftfreq(n_pad, d=dx)) * np.pi

    # ---- optional interpolation to custom grid ----
    if p_min is not None:
        p_custom = np.linspace(p_min, p_max, p_n)
        W_interp = np.zeros((nx1, p_n))
        for j in range(nx1):
            W_interp[j] = np.interp(p_custom, p_fft, W[j])
        return W_interp

    return W


class WaveEvolution:
    def __init__(self, xrange=(-5., 5.), prange=(-5., 5.), npts=639, n_extend_left=200, n_extend_right=200,
                 n_scale=1, dt=0.001, mass=1., speed=PI, just_wave=False, col_psi=mn.RED):
        xmin, xmax = self.xrange = xrange
        pmin, pmax = self.prange = prange
        self.npts = npts
        self.n_extend_left = n_extend_left
        self.n_extend_right = n_extend_right
        self.n_scale = 1
        if not just_wave:
            setup_cam()
        npts1 = (npts + n_extend_left+n_extend_right-1)*n_scale+1
        i_x0 = n_extend_left * n_scale
        i_x1 = npts1 - n_extend_right * n_scale
        x_extend_left = (xmax - xmin) / (npts-1) * n_extend_left
        x_extend_right = (xmax - xmin) / (npts-1) * n_extend_right
        xmin1, xmax1 = (xmin - x_extend_left, xmax + x_extend_right)

        origin, right, up, out, p, xsurf, y, shape, self.txt = setup_surf(xrange, prange, spawn=not just_wave)
        assert shape == (npts, npts)

        self.just_wave = just_wave
        if just_wave:
            self.col0 = self.fill_mask = self.mesh_mask = None
            up = ORIGIN
            ax = mn.Axes(x_range=(xmin, xmax*1.07), y_range=(0, 0.33), x_length=8, y_length=2,
                         axis_config={'color': mn.WHITE, 'stroke_width': 4, 'include_ticks': False,
                                      "tip_width": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                      "tip_height": 0.5 * mn.DEFAULT_ARROW_TIP_LENGTH,
                                      },
                         )
            ax.shift(-ax.coords_to_point(0, 0))
            right = ax.coords_to_point(1,0)
            out = ax.coords_to_point(0,1)
            origin = torch.tensor(out * 0.002)
            psi = mn.MathTex(r'\psi', stroke_width=2, font_size=60).move_to(ax.coords_to_point(0.4, 0.305)+mn.IN*0.1)
            psi.set_color(col_psi)
            ax = ManimMob(ax).move(IN*0.008)
            psi = ManimMob(psi)
            with Off():
                cam = Scene.get_camera()
                cam.set_distance_to_screen(100)
                cam.move_to(cam.get_center()*1.5)
                self.txt.despawn()
                ax.spawn()
                psi.spawn()
        else:
            self.col0 = p.color.clone()
            surf2 = ah.surface_mesh(num_recs=64, rec_size=10, fill_opacity=1, stroke_opacity=0, add_to_scene=False)
            self.fill_mask = surf2.get_descendants()[1].color[:,:,-1:]
            self.mesh_mask = 1 - self.fill_mask

        self.dt = dt
        self.speed = speed
        self.mass = mass
        self.xvals1 = torch.linspace(xmin1, xmax1, npts1)
        self.psi = self.xvals1 * 0
        self.xvals = torch.linspace(xmin, xmax, npts)
        dx = (self.xvals[1] - self.xvals[0]).item()
        self.stride = (i_x0, i_x1, n_scale)
        self.V = self.xvals1 * 0
        self.dx1 = dx/n_scale
        self.dirs = (origin, right, up, out)
        self.pvals = torch.linspace(pmin, pmax, npts)
        self.xrange1 = (xmin1, xmax1)
        self.p = p
        self.px = self.xx = self.yx = None
        self.pp = self.xp = self.yp = None

    def create_wave(self):
        self.px, self.xx, self.yx = setup_wave(xrange=self.xrange, npts=self.npts)
        if not self.just_wave:
            self.pp, self.xp, self.yp = setup_wave(xrange=self.prange, npts=self.npts)

    def evolve(self, time_inc):
        ndt = math.ceil(time_inc / self.dt)

        if ndt > 0:
            # print(ndt)
            dt1 = time_inc / ndt
            for i in range(ndt):
                self.psi = time_evolution(self.psi, self.V, self.speed * dt1, self.dx1, mass=self.mass)

        pmin, pmax = self.prange
        xmin, xmax = self.xrange
        psi = self.psi
        psi0 = psi[self.stride[0]:self.stride[1]:self.stride[2]]
        origin, right, up, out = self.dirs

        if self.px is not None:
            set_wave(self.px, self.xvals, psi0.abs()**2, psi0, origin + pmax * up, right, out * 0.5)
        if self.pp is not None:
            phase = np.exp(1j * np.outer(self.pvals, self.xvals1))
            psi_k = torch.tensor(phase @ psi.numpy()) * self.dx1 / math.sqrt(2*PI)
            set_wave(self.pp, self.pvals, psi_k.abs() ** 2, psi_k, origin + xmin * right, -up, out * 0.5)
        if self.just_wave:
            return
        p = self.p
        xmin1, xmax1 = self.xrange1
        i0, i1, n_scale = self.stride
        W = torch.tensor(wigner_fft(psi.numpy(), xmin1, xmax1, pmin, pmax, self.npts, i0=i0, i1=i1, step=n_scale))
        loc = p.location.clone()
        col = p.color.clone()
        loc[0, :, 2] = origin[2] + out[2] * 2 * W.reshape(-1)
        shade_up = torch.pow(((W - 0.05) * 4).clamp(0, 1), 0.8).flatten().view(1,-1,1)
        shade_down = (W * -50).clamp(0., 1).flatten().view(1,-1,1)
        fill_mask = self.fill_mask

        col[..., :3] = fill_mask * shade_up * col_up \
                       + fill_mask * shade_down * col_dn \
                       + fill_mask * (1 - shade_up - shade_down) * self.col0[:, :, :3] \
                       + self.mesh_mask * self.col0[..., :3]
        p.set_non_recursive(location=loc, color=col)

    def orbit(self, anim=1):
        origin, right, up, out = self.dirs
        cam = Scene.get_camera()
        if anim == 1:
            with Seq():
                cam.orbit_around_point(origin, 60 * DEGREES, cam.get_right_direction())
        elif anim == 2:
            with Off():
                cam.orbit_around_point(origin, 60 * DEGREES, cam.get_right_direction())
            with Sync(run_time=3):
                cam.orbit_around_point(origin, -60 * DEGREES, out)
                cam.orbit_around_point(origin, -130 * DEGREES, cam.get_right_direction())
                self.txt[0].orbit_around_point(self.txt[0].get_center(), -90, RIGHT)
        else:
            orbit_time = 4.
            with Sync(run_time=orbit_time):
                cam.orbit_around_point(origin, 180 * DEGREES, out)
            with Sync(run_time=1.5):
                cam.orbit_around_point(origin, -70 * DEGREES, cam.get_right_direction())

    def set_gaussian(self, x=0., p = 0.):
        sigma = np.sqrt(0.5)  # Width of Gaussian
        self.psi = torch.exp(-(self.xvals1 - x) ** 2 / (4 * sigma ** 2) + 1j * p * self.xvals1)  # + xvals1 * 6 * 1j)

        # Normalize the wavefunction
        self.psi /= np.linalg.norm(self.psi) * np.sqrt(self.dx1)

