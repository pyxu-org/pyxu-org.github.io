import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import Gradient
from pyxu.util.misc import peaks

# Define input image
n = 100
x = np.linspace(-3, 3, n)
xx, yy = np.meshgrid(x, x)
image = peaks(xx, yy)
arg_shape = image.shape  # (1000, 1000)
# Instantiate gradient operator
grad = Gradient(arg_shape=arg_shape)

# Compute gradients
output = grad(image.ravel()) # shape = (2000000, )
df_dx, df_dy = grad.unravel(output) # shape = (2, 1000, 1000)

# Plot image
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
im = axs[0].imshow(image)
axs[0].set_title("Image")
axs[0].axis("off")
plt.colorbar(im, ax=axs[0])

# Plot gradient
im = axs[1].imshow(df_dx)
axs[1].set_title(r"$\partial f/ \partial x$")
axs[1].axis("off")
plt.colorbar(im, ax=axs[1])
im = axs[2].imshow(df_dy)
axs[2].set_title(r"$\partial f/ \partial y$")
axs[2].axis("off")
plt.colorbar(im, ax=axs[2])