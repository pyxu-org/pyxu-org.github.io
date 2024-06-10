import matplotlib.pyplot as plt
import numpy as np
from pyxu.operator import Prewitt

dim_shape = (11, 11)
image = np.zeros(dim_shape)
image[5, 5] = 1.

prewitt = Prewitt(dim_shape)
out = prewitt(image)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.colorbar()
plt.subplot(122)
plt.imshow(out)
plt.colorbar()