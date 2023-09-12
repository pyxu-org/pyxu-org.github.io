import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import Jacobian
from pyxu.util.misc import peaks

x = np.linspace(-2.5, 2.5, 25)
xx, yy = np.meshgrid(x, x)
image = np.tile(peaks(xx, yy), (3, 1, 1))
jac = Jacobian(arg_shape=image.shape[1:], n_channels=image.shape[0])
out = jac.unravel(jac(image.ravel()))
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
for i in range(3):
    for j in range(2):
        axes[i, j].imshow(out[i, j].T, cmap=["Reds", "Greens", "Blues"][j])
        axes[i, j].set_title(f"$\partial I_{{{['R', 'G', 'B'][j]}}}/\partial{{{['x', 'y'][j]}}}$")
plt.suptitle("Jacobian")