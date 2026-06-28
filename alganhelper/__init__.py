import torch
from algan import *

def anchor_pts(start=0., end=1., n=10):
    pts = np.linspace(start, end, n)
    anchors = []
    for i in range(len(pts) - 1):
        anchors += [pts[i], 0.67 * pts[i] + 0.33 * pts[i + 1], 0.33 * pts[i] + 0.67 * pts[i + 1], pts[i + 1]]
    anchors += anchors[::-1]
    anchors2 = torch.tensor(anchors)
    return anchors2

def curve3d(f, start=0., end=1., npts=2, border_width=4., filled=False, **kwargs):
    anchors = anchor_pts(start=start, end=end, n=npts)
    arc = BezierCircuitCubic(torch.cat([f(a) for a in anchors], -2),
                             filled=filled, border_width=border_width, **kwargs)
    return arc

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def surface_mesh(col1=RED_C, col2=RED_D, fill_opacity=0.9, stroke_color=RED_C, stroke_opacity=0.8,
                 num_recs=32, rec_size=10, **kwargs):
    col1 = col1.clone()
    col2 = col2.clone()
    n = num_recs
    m = rec_size
    t = torch.zeros(m * n - 1, m * n - 1, 5)
    sc = stroke_color.clone()
    sc[-1] = stroke_opacity
    for i1 in range(n):
        for j1 in range(n):
            col = col1 if (i1 + j1) % 2 == 0 else col2
            col[-1] = fill_opacity
            for i2 in range(m if i1 < n-1 else m-1):
                for j2 in range(m if j1 < n-1 else m-1):
                    pixel = col if i2 < m-1 and j2 < m-1 else sc
                    t[i1*m+i2, j1*m+j2, :] = pixel

    mob = ImageMob(t, **kwargs)
    return mob

def curve_surface_loc(pts, normals=None, tangents=None, resolution=8, width=0.03, closed=True):
    if normals is None:
        normals = pts
    if tangents is None:
        if closed:
            tangents = torch.roll(pts, shifts=1, dims=0) - torch.roll(pts, shifts=-1, dims=0)
        else:
            tangents = torch.empty_like(pts)
            tangents[1:-1] = pts[2:] - pts[:-2]
            tangents[0] = pts[1] - pts[0]
            tangents[-1] = pts[-1] - pts[-2]
    du = torch.nn.functional.normalize(normals, dim=1)
    dv = torch.nn.functional.normalize(torch.linalg.cross(normals, tangents), dim=1)
    theta = torch.linspace(-math.pi, math.pi, resolution, device=pts.device)
    return (
            pts[:, None, :] +
            (torch.sin(theta)[None, :, None] * du[:, None, :] +
             torch.cos(theta)[None, :, None] * dv[:, None, :]) * (width / 2)
    ).reshape(1, -1, 3).float()

def curve_surface(pts, normals=None, tangents=None, resolution=8, color=BLUE, width=0.03, closed=True, **kwargs):
    n = pts.shape[0]
    crv = Surface(grid_height=resolution, grid_width=n, color=color, **kwargs)
    crv.get_descendants()[1].location[...] = curve_surface_loc(pts, normals, tangents, resolution, width, closed)
    return crv

class FrameStepper:
    def __init__(self, fps = 30., run_time=1., rate_func=rate_funcs.smooth, step=1, include_start=True):
        self.u = 0.
        self.du = 0.
        self.time = 0.
        self.dt = 0.
        self.fps = fps
        self.run_time = run_time
        self.n = round(fps * run_time)
        self.rate_func = rate_func
        self.step = step
        self.index = 0
        self.start = include_start
        self.context = Off()

    def __iter__(self):
        return self

    def __next__(self):
        print('Frame number:', self.index)
        if self.start:
            self.start = False
            return self

        if self.index >= self.n:
            raise StopIteration
        self.index = min(self.index + self.step, self.n)
        u0 = self.index / self.n


        u = self.rate_func(torch.tensor(u0))
        if type(u) == torch.Tensor:
            u = u.item()
        t = u0 * self.run_time
        self.du = u - self.u
        self.dt = t - self.time
        self.u = u
        self.time = t
        self.context = Sync(run_time=self.dt, rate_func=rate_funcs.identity)

        return self