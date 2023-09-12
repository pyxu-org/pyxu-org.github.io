import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import Gradient, Divergence, Laplacian
from pyxu.util.misc import peaks

n = 100
x = np.linspace(-3, 3, n)
xx, yy = np.meshgrid(x, x)
image = peaks(xx, yy)
arg_shape = image.shape  # (1000, 1000)
grad = Gradient(arg_shape=arg_shape)
div = Divergence(arg_shape=arg_shape)
# Construct Laplacian via composition
laplacian1 = div * grad
# Compare to default Laplacian
laplacian2 = Laplacian(arg_shape=arg_shape)
output1 = laplacian1(image.ravel())
output2 = laplacian2(image.ravel())
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im = axes[0].imshow(np.log(abs(output1)).reshape(*arg_shape))
axes[0].set_title("Laplacian via composition")
plt.colorbar(im, ax=axes[0])
im = axes[1].imshow(np.log(abs(output1)).reshape(*arg_shape))
axes[1].set_title("Default Laplacian")
plt.colorbar(im, ax=axes[1])