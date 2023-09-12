import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import PartialDerivative
from pyxu.util.misc import peaks

x = np.linspace(-2.5, 2.5, 25)
xx, yy = np.meshgrid(x, x)
image = peaks(xx, yy)
arg_shape = image.shape  # Shape of our image
# Specify derivative order at each direction
df_dx = (1, 0)  # Compute derivative of order 1 in first dimension
d2f_dy2 = (0, 2)  # Compute derivative of order 2 in second dimension
d3f_dxdy2 = (1, 2)  # Compute derivative of order 1 in first dimension and der. of order 2 in second dimension
# Instantiate derivative operators
sigma = 2.0
diff1 = PartialDerivative.gaussian_derivative(order=df_dx, arg_shape=arg_shape, sigma=sigma / np.sqrt(2))
diff2 = PartialDerivative.gaussian_derivative(order=d2f_dy2, arg_shape=arg_shape, sigma=sigma / np.sqrt(2))
diff = PartialDerivative.gaussian_derivative(order=d3f_dxdy2, arg_shape=arg_shape, sigma=sigma)
# Compute derivatives
out1 = (diff1 * diff2)(image.ravel()).reshape(arg_shape)
out2 = diff(image.ravel()).reshape(arg_shape)
# Plot derivatives
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
im = axs[0].imshow(image)
axs[0].axis("off")
axs[0].set_title("f(x,y)")
plt.colorbar(im, ax=axs[0])
axs[1].imshow(out1)
axs[1].axis("off")
axs[1].set_title(r"$\frac{\partial^{3} f(x,y)}{\partial x\partial y^{2}}$")
plt.colorbar(im, ax=axs[1])

axs[2].imshow(out2)
axs[2].axis("off")
axs[2].set_title(r"$\frac{\partial^{3} f(x,y)}{\partial x\partial y^{2}}$")
plt.colorbar(im, ax=axs[2])

# Test approximation error
plt.figure()
plt.imshow(abs(out1 - out2) / abs(out2)), plt.colorbar()