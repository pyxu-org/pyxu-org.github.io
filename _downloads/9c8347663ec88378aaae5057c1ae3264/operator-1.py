import numpy as np
import pyxu.operator as pxo
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
D, M, N = 1, 2, 3  # D denotes the dimension of the data
x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)
A = pxo.NUFFT.type1(
    x, N,
    isign=-1,
    eps=1e-9
)
A.plot_kernel()
plt.show()