import matplotlib.pyplot as plt
from pyxu.operator import Sobel

arg_shape = (11, 11)
image = np.zeros(arg_shape)
image[5, 5] = 1.

sobel = Sobel(arg_shape)
out = sobel(image.ravel())
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.colorbar()
plt.subplot(122)
plt.imshow(out.reshape(*arg_shape))
plt.colorbar()