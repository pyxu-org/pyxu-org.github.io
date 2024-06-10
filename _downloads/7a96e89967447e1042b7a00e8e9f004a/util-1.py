import numpy as np
import matplotlib.pyplot as plt
from pyxu.util.misc import peaks

x = np.linspace(-3, 3, 1000)
xx, yy = np.meshgrid(x, x)
z = peaks(xx, yy)
plt.figure()
plt.imshow(z)