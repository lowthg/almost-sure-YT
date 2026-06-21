from algan import *
import colorsys

def setup_wave(xrange=(-5., 5.), npts=640):
    xmin, xmax = xrange

    n_col=4
    surfx = Surface(grid_height=n_col, grid_width=npts)
    px = surfx.get_descendants()[1]
    locx = px.location.clone()
    colx = px.color.clone()
    xx = locx[:, :, 0] * (xmax - xmin) / 2 + (xmax + xmin) / 2
    yx = (locx[:, :, 1] + 1) / 2
    colx[..., 4] = 0.75
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

        psi = self.psi
        psi0 = psi[self.stride[0]:self.stride[1]:self.stride[2]]
        return psi0

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