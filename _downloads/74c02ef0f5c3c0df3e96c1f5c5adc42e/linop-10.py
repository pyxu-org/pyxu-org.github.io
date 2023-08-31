import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import DirectionalDerivative
from pyxu.util.misc import peaks

x = np.linspace(-2.5, 2.5, 25)
xx, yy = np.meshgrid(x, x)
z = peaks(xx, yy)
directions = np.zeros(shape=(2, z.size))
directions[0, : z.size // 2] = 1
directions[1, z.size // 2:] = 1
dop = DirectionalDerivative(arg_shape=z.shape, order=1, directions=directions)
out = dop.unravel(dop(z.ravel()))
dop2 = DirectionalDerivative(arg_shape=z.shape, order=2, directions=directions)
out2 = dop2.unravel(dop2(z.ravel()))
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = np.ravel(axs)
h = axs[0].pcolormesh(xx, yy, z, shading="auto")
axs[0].quiver(x, x, directions[1].reshape(xx.shape), directions[0].reshape(xx.shape))
plt.colorbar(h, ax=axs[0])
axs[0].set_title("Signal and directions of first derivatives")

h = axs[1].pcolormesh(xx, yy, out.squeeze(), shading="auto")
plt.colorbar(h, ax=axs[1])
axs[1].set_title("First-order directional derivatives")

h = axs[2].pcolormesh(xx, yy, out2.squeeze(), shading="auto")
plt.colorbar(h, ax=axs[2])
axs[2].set_title("Second-order directional derivative")