import numpy as np
import matplotlib.pyplot as plt
from pyxu.operator import StructureTensor
from pyxu.util.misc import peaks

# Define input image
n = 1000
x = np.linspace(-3, 3, n)
xx, yy = np.meshgrid(x, x)
image = peaks(xx, yy)
nsamples = 2
dim_shape = image.shape  # (1000, 1000)
images = np.tile(image, (nsamples, 1, 1))
# Instantiate structure tensor operator
structure_tensor = StructureTensor(dim_shape=dim_shape)

outputs = structure_tensor(images)
print(outputs.shape)  # (2, 3, 1000, 1000)
# Plot
plt.figure()
plt.imshow(images[0])
plt.colorbar()
plt.title("Image")
plt.axis("off")

plt.figure()
plt.imshow(outputs[0][0])
plt.colorbar()
plt.title(r"$\hat{S}_{xx}$")
plt.axis("off")

plt.figure()
plt.imshow(outputs[0][1])
plt.colorbar()
plt.title(r"$\hat{S}_{xy}$")
plt.axis("off")

plt.figure()
plt.imshow(outputs[0][2])
plt.colorbar()
plt.title(r"$\hat{S}_{yy}$")
plt.axis("off")