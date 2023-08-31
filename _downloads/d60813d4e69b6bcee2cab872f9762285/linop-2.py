import numpy as np
import pyxu.operator as pxo

rng = np.random.default_rng(2)
D, M, N = 2, 500, 200
rnd_points = lambda _: rng.normal(scale=rng.uniform(0.25, 0.5, size=(D,)), size=(_, D))
rnd_offset = lambda: rng.uniform(-1, 1, size=(D,))
scale = 20
x = np.concatenate(
    [
        rnd_points(M) + rnd_offset() * scale,
        rnd_points(M) + rnd_offset() * scale,
        rnd_points(M) + rnd_offset() * scale,
        rnd_points(M) + rnd_offset() * scale,
        rnd_points(M) + rnd_offset() * scale,
    ],
    axis=0,
)
z = np.concatenate(
    [
        rnd_points(N) + rnd_offset() * scale,
        rnd_points(N) + rnd_offset() * scale,
        rnd_points(N) + rnd_offset() * scale,
        rnd_points(N) + rnd_offset() * scale,
        rnd_points(N) + rnd_offset() * scale,
    ],
    axis=0,
)

kwargs = dict(
    x=x,
    z=z,
    isign=-1,
    eps=1e-3,
)
A = pxo.NUFFT.type3(**kwargs, chunked=True)
x_chunks, z_chunks = A.auto_chunk(
    max_mem=.1,
    max_anisotropy=1,
)
A.allocate(x_chunks, z_chunks)
fig = A.diagnostic_plot('x')
fig.show()