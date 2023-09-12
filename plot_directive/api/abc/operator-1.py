import numpy as np
import matplotlib.pyplot as plt
from pyxu.abc import ProxFunc

class L1Norm(ProxFunc):
    def __init__(self, dim: int):
        super().__init__(shape=(1, dim))
    def apply(self, arr):
        return np.linalg.norm(arr, axis=-1, keepdims=True, ord=1)
    def prox(self, arr, tau):
        return np.clip(np.abs(arr)-tau, a_min=0, a_max=None) * np.sign(arr)

N = 512
l1_norm = L1Norm(dim=N)
mus = [0.1, 0.5, 1]
smooth_l1_norms = [l1_norm.moreau_envelope(mu) for mu in mus]

x = np.linspace(-1, 1, N)[:, None]
labels=['mu=0']
labels.extend([f'mu={mu}' for mu in mus])
plt.figure()
plt.plot(x, l1_norm(x))
for f in smooth_l1_norms:
    plt.plot(x, f(x))
plt.legend(labels)
plt.title('Moreau Envelope')

labels=[f'mu={mu}' for mu in mus]
plt.figure()
for f in smooth_l1_norms:
    plt.plot(x, f.grad(x))
plt.legend(labels)
plt.title('Derivative of Moreau Envelope')