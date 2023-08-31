import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import PartialDerivative
from pyxu.util.misc import peaks
x = np.linspace(-2.5, 2.5, 25)
xx, yy = np.meshgrid(x, x)
image = peaks(xx, yy)
arg_shape = image.shape  # Shape of our image
# Specify derivative order at each direction
df_dx = (1, 0) # Compute derivative of order 1 in first dimension
d2f_dy2 = (0, 2) # Compute derivative of order 2 in second dimension
d3f_dxdy2 = (1, 2) # Compute derivative of order 1 in first dimension and der. of order 2 in second dimension
# Instantiate derivative operators
diff1 = PartialDerivative.gaussian_derivative(order=df_dx, arg_shape=arg_shape, sigma=2.0)
diff2 = PartialDerivative.gaussian_derivative(order=d2f_dy2, arg_shape=arg_shape, sigma=2.0)
diff = PartialDerivative.gaussian_derivative(order=d3f_dxdy2, arg_shape=arg_shape, sigma=2.0)
# Compute derivatives
out1 = (diff1 * diff2)(image.ravel()).reshape(arg_shape)
out2 = diff(image.ravel()).reshape(arg_shape)
plt.figure()
plt.imshow(image),
plt.axis('off')
plt.colorbar()
plt.title('f(x,y)')
plt.figure()
plt.imshow(out1.T)
plt.axis('off')
plt.title(r'$\frac{\partial^{3} f(x,y)}{\partial x\partial y^{2}}$')
plt.figure()
plt.imshow(out2.T)
plt.axis('off')
plt.title(r'$\frac{\partial^{3} f(x,y)}{\partial x\partial y^{2}}$')