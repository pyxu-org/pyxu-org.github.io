import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import DirectionalHessian
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
dim_shape = z.shape
d_hess = DirectionalHessian(dim_shape=dim_shape, directions=[directions1, directions2])
out = d_hess(z)
plt.figure()
h = plt.pcolormesh(xx, yy, z, shading='auto')
plt.quiver(x, x, directions1[1].reshape(dim_shape), directions1[0].reshape(xx.shape))
plt.quiver(x, x, directions2[1].reshape(dim_shape), directions2[0].reshape(xx.shape), color='red')
plt.colorbar(h)
plt.title(r'Signal $\mathbf{f}$ and directions of derivatives')
plt.figure()
h = plt.pcolormesh(xx, yy, out[0], shading='auto')
plt.colorbar(h)
plt.title(r'$\nabla^2_{\mathbf{v}_0} \mathbf{f}$')
plt.figure()
h = plt.pcolormesh(xx, yy, out[1], shading='auto')
plt.colorbar(h)
plt.title(r'$\nabla_{\mathbf{v}_0} \nabla_{\mathbf{v}_{1}} \mathbf{f}$')
plt.figure()
h = plt.pcolormesh(xx, yy, out[2], shading='auto')
plt.colorbar(h)
plt.title(r'$\nabla^2_{\mathbf{v}_1} \mathbf{f}$')