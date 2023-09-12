import matplotlib.pyplot as plt
from pyxu.operator import Gaussian

arg_shape = (11, 11)
image = np.zeros(arg_shape)
image[5, 5] = 1.

gaussian = Gaussian(arg_shape, sigma=3)
out = gaussian(image.ravel())
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.colorbar()
plt.subplot(122)
plt.imshow(out.reshape(*arg_shape))
plt.colorbar()