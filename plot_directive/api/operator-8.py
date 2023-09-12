import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import Hessian, PartialDerivative
from pyxu.util.misc import peaks

n = 100
x = np.linspace(-3, 3, n)
xx, yy = np.meshgrid(x, x)
image = peaks(xx, yy)
arg_shape = image.shape  # (1000, 1000)

# Instantiate Hessian operator
hessian = Hessian(arg_shape=arg_shape, directions="all")
# Compute Hessian
output = hessian(image.ravel()) # shape = (300000,)
d2f_dx2, d2f_dxdy, d2f_dy2 = hessian.unravel(output)

# Plot
fig, axs = plt.subplots(1, 4, figsize=(20, 4))
im = axs[0].imshow(image)
plt.colorbar(im, ax=axs[0])
axs[0].set_title("Image")
axs[0].axis("off")

im = axs[1].imshow(d2f_dx2)
plt.colorbar(im, ax=axs[1])
axs[1].set_title(r"$\partial^{2} f/ \partial x^{2}$")
axs[1].axis("off")

im = axs[2].imshow(d2f_dxdy)
plt.colorbar(im, ax=axs[2])
axs[2].set_title(r"$\partial^{2} f/ \partial x\partial y$")
axs[2].axis("off")

im = axs[3].imshow(d2f_dy2)
plt.colorbar(im, ax=axs[3])
axs[3].set_title(r"$\partial^{2} f/ \partial y^{2}$")
axs[3].axis("off")