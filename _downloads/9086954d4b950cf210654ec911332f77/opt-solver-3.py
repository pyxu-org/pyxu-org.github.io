import matplotlib.pyplot as plt
import numpy as np
import pyxu.abc as pxa
import pyxu.operator as pxo
import scipy as sp
from pyxu.opt.solver import ADMM

N = 100  # Dimension of the problem

# Generate piecewise-linear ground truth
x_gt = np.array([10, 25, 60, 90])  # Knot locations
a_gt = np.array([2, -4, 3, -2])  # Amplitudes of the knots
gt = np.zeros(N)  # Ground-truth signal
for n in range(len(x_gt)):
    gt[x_gt[n] :] += a_gt[n] * np.arange(N - x_gt[n]) / N

# Generate data (noisy samples at random locations)
M = 20  # Number of data points
rng = np.random.default_rng(seed=0)
x_samp = rng.choice(np.arange(N // M), size=M) + np.arange(N, step=N // M)  # sampling locations
sigma = 2 * 1e-2  # noise variance
y = gt[x_samp] + sigma * rng.standard_normal(size=M)  # noisy data points

# Data-fidelity term
subsamp_mat = sp.sparse.lil_matrix((M, N))
for i in range(M):
    subsamp_mat[i, x_samp[i]] = 1
G = pxa.LinOp.from_array(subsamp_mat.tocsr())
F = 1 / 2 * pxo.SquaredL2Norm(dim=y.size).argshift(-y) * G
F.diff_lipschitz = F.estimate_diff_lipschitz(method="svd")

# Regularization term (promotes sparse second derivatives)
deriv_mat = sp.sparse.diags(diagonals=[1, -2, 1], offsets=[0, 1, 2], shape=(N - 2, N))
D = pxa.LinOp.from_array(deriv_mat)
_lambda = 1e-1  # regularization parameter
H = _lambda * pxo.L1Norm(dim=D.codim)

# Solver for ADMM
tau = 1 / _lambda  # internal ADMM parameter
# Inverse operator to solve the linear system
A_inv = sp.linalg.inv(G.gram().asarray() + (1 / tau) * D.gram().asarray())

def solver_ADMM(arr, tau):
    b = (1 / tau) * D.adjoint(arr) + G.adjoint(y)
    return A_inv @ b.squeeze()


# Solve optimization problem
admm = ADMM(f=F, h=H, K=D, solver=solver_ADMM,show_progress=False)  # with solver
admm.fit(x0=np.zeros(N), tau=tau)
x_opt = admm.solution()  # reconstructed signal

# Plots
plt.figure()
plt.plot(np.arange(N), gt, label="Ground truth")
plt.plot(x_samp, y, "kx", label="Noisy data points")
plt.plot(np.arange(N), x_opt, label="Reconstructed signal")
plt.legend()