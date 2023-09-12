import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import Laplacian
from pyxu.util.misc import peaks

# Define input image
n = 100
x = np.linspace(-3, 3, n)
xx, yy = np.meshgrid(x, x)
image = peaks(xx, yy)

arg_shape = image.shape  # (1000, 1000)
# Compute Laplacian
laplacian = Laplacian(arg_shape=arg_shape)
output = laplacian(image.ravel())
output = laplacian.unravel(output) # shape = (1, 1000, 1000)
# Plot
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
im = axs[0].imshow(image)
plt.colorbar(im, ax=axs[0])
axs[0].set_title("Image")
axs[0].axis("off")

im = axs[1].imshow(output.squeeze())
plt.colorbar(im, ax=axs[1])
axs[1].set_title(r"$\partial^{2} f/ \partial x^{2}+\partial^{2} f/ \partial y^{2}$")
axs[1].axis("off")

fig.show()