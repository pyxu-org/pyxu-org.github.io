import matplotlib.pyplot as plt
import numpy as np
from pyxu.operator import MovingAverage

dim_shape = (11, 11)
image = np.zeros(dim_shape)
image[5, 5] = 1.

ma = MovingAverage(dim_shape, size=5)
out = ma(image)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.colorbar()
plt.subplot(122)
plt.imshow(out)
plt.colorbar()