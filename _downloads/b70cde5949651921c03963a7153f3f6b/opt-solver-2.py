import matplotlib.pyplot as plt
import numpy as np
import pyxu.operator as pxo
from pyxu.operator import SubSample, PartialDerivative
from pyxu.opt.solver import PD3O

x = np.repeat(np.asarray([0, 2, 1, 3, 0, 2, 0]), 10)
N = x.size

D = PartialDerivative.finite_difference(dim_shape=(N,), order=(1,))

downsample = SubSample(N, slice(None, None, 3))
y = downsample(x)
loss = (1 / 2) * pxo.SquaredL2Norm(y.size).argshift(-y)
F = loss * downsample

pd3o = PD3O(f=F, g=0.01 * pxo.L1Norm(N), h=0.1 * pxo.L1Norm((N)), K=D)
x0, z0 = np.zeros((2, N))
pd3o.fit(x0=x0, z0=z0)
x_recons = pd3o.solution()

plt.figure()
plt.stem(x, linefmt="C0-", markerfmt="C0o")
mask_ids = np.where(downsample.adjoint(np.ones_like(y)))[0]
markerline, stemlines, baseline = plt.stem(mask_ids, y, linefmt="C3-", markerfmt="C3o")
markerline.set_markerfacecolor("none")
plt.stem(x_recons, linefmt="C1--", markerfmt="C1s")
plt.legend(["Ground truth", "Observation", "PD3O Estimate"])
plt.show()