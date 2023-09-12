import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import DirectionalLaplacian
from pyxu.util.misc import peaks

x = np.linspace(-2.5, 2.5, 25)
xx, yy = np.meshgrid(x, x)
z = peaks(xx, yy)
directions1 = np.zeros(shape=(2, z.size))
directions1[0, :z.size // 2] = 1
directions1[1, z.size // 2:] = 1
directions2 = np.zeros(shape=(2, z.size))
directions2[1, :z.size // 2] = -1
directions2[0, z.size // 2:] = -1
arg_shape = z.shape
dop = DirectionalLaplacian(arg_shape=arg_shape, directions=[directions1, directions2])
out = dop.unravel(dop(z.ravel()))
plt.figure()
h = plt.pcolormesh(xx, yy, z, shading='auto')
plt.quiver(x, x, directions1[1].reshape(arg_shape), directions1[0].reshape(xx.shape))
plt.quiver(x, x, directions2[1].reshape(arg_shape), directions2[0].reshape(xx.shape), color='red')
plt.colorbar(h)
plt.title('Signal and directions of derivatives')
plt.figure()
h = plt.pcolormesh(xx, yy, out.squeeze(), shading='auto')
plt.colorbar(h)
plt.title('Directional Laplacian')