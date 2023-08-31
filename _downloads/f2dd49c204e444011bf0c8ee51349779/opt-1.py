import matplotlib.pyplot as plt
import numpy as np
import pyxu.operator as pxo
from pyxu.experimental._dev import DownSampling, FirstDerivative
from pyxu.opt.solver import CV

x = np.repeat(np.asarray([0, 2, 1, 3, 0, 2, 0]), 10)
N = x.size

D = FirstDerivative(size=N, kind="forward")
D.lipschitz = D.estimate_lipschitz()

downsample = DownSampling(size=N, downsampling_factor=3)
y = downsample(x)
loss = (1 / 2) * pxo.SquaredL2Norm(y.size).argshift(-y)
F = loss * downsample
F.diff_lipschitz = F.estimate_diff_lipschitz()

cv = CV(f=F, g=0.01 * pxo.L1Norm(N), h=0.1 * pxo.L1Norm(N), K=D)
x0, z0 = np.zeros((2, N))
cv.fit(x0=x0, z0=z0)
x_recons = cv.solution()[0]

plt.figure()
plt.stem(x, linefmt="C0-", markerfmt="C0o")
mask_ids = np.where(downsample.downsampling_mask)[0]
markerline, stemlines, baseline = plt.stem(mask_ids, y, linefmt="C3-", markerfmt="C3o")
markerline.set_markerfacecolor("none")
plt.stem(x_recons, linefmt="C1--", markerfmt="C1s")
plt.legend(["Ground truth", "Observation", "CV Estimate"])
plt.show()