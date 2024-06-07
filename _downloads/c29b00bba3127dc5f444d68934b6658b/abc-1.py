import numpy as np
import matplotlib.pyplot as plt
from pyxu.abc import ProxFunc

class L1Norm(ProxFunc):
    def __init__(self, dim: int):
        super().__init__(dim_shape=dim, codim_shape=1)
    def apply(self, arr):
        return np.linalg.norm(arr, axis=-1, keepdims=True, ord=1)
    def prox(self, arr, tau):
        return np.clip(np.abs(arr)-tau, a_min=0, a_max=None) * np.sign(arr)

mu = [0.1, 0.5, 1]
f = [L1Norm(dim=1).moreau_envelope(_mu) for _mu in mu]
x = np.linspace(-1, 1, 512).reshape(-1, 1)  # evaluation points

fig, ax = plt.subplots(ncols=2)
for _mu, _f in zip(mu, f):
    ax[0].plot(x, _f(x), label=f"mu={_mu}")
    ax[1].plot(x, _f.grad(x), label=f"mu={_mu}")
ax[0].set_title('Moreau Envelope')
ax[1].set_title("Derivative of Moreau Envelope")
for _ax in ax:
    _ax.legend()
    _ax.set_aspect("equal")
fig.tight_layout()