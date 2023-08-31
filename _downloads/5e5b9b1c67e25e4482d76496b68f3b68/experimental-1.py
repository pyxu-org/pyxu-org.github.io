import matplotlib.pyplot as plt
import numpy as np
import pyxu.experimental.sampler as pxes
import pyxu.operator as pxo
import scipy as sp

f = pxo.SquaredL2Norm(dim=1) / 2  # To sample 1D normal distribution (mean 0, variance 1)
ula = pxes.ULA(f=f)  # Sampler with maximum step size
ula_lb = pxes.ULA(f=f, gamma=1e-1)  # Sampler with small step size

gen_ula = ula.samples(x0=np.zeros(1))
gen_ula_lb = ula_lb.samples(x0=np.zeros(1))
n_burn_in = int(1e3)  # Number of burn-in iterations
for i in range(n_burn_in):
    next(gen_ula)
    next(gen_ula_lb)

# Online statistics objects
mean_ula = pxes.OnlineMoment(order=1)
mean_ula_lb = pxes.OnlineMoment(order=1)
var_ula = pxes.OnlineVariance()
var_ula_lb = pxes.OnlineVariance()

n = int(1e4)  # Number of samples
samples_ula = np.zeros(n)
samples_ula_lb = np.zeros(n)
for i in range(n):
    sample = next(gen_ula)
    sample_lb = next(gen_ula_lb)
    samples_ula[i] = sample
    samples_ula_lb[i] = sample_lb
    mean = float(mean_ula.update(sample))
    var = float(var_ula.update(sample))
    mean_lb = float(mean_ula_lb.update(sample_lb))
    var_lb = float(var_ula_lb.update(sample_lb))

# Theoretical variances of biased stationary distributions of ULA
biased_var = 1 / (1 - ula._gamma / 2)
biased_var_lb = 1 / (1 - ula_lb._gamma / 2)

# Quantify goodness of fit of empirical distribution with theoretical distribution (Cramér-von Mises test)
cvm = sp.stats.cramervonmises(samples_ula, "norm", args=(0, np.sqrt(biased_var)))
cvm_lb = sp.stats.cramervonmises(samples_ula_lb, "norm", args=(0, np.sqrt(biased_var_lb)))

# Plots
grid = np.linspace(-4, 4, 1000)

plt.figure()
plt.title(
    f"ULA samples (large step size) \n Empirical mean: {mean:.3f} (theoretical: 0) \n "
    f"Empirical variance: {var:.3f} (theoretical: {biased_var:.3f}) \n"
    f"Cramér-von Mises goodness of fit: {cvm.statistic:.3f}"
)
plt.hist(samples_ula, range=(min(grid), max(grid)), bins=100, density=True)
plt.plot(grid, sp.stats.norm.pdf(grid), label=r"$p(x)$")
plt.plot(grid, sp.stats.norm.pdf(grid, scale=np.sqrt(biased_var)), label=r"$p_{\gamma_1}(x)$")
plt.legend()
plt.show()

plt.figure()
plt.title(
    f"ULA samples (small step size) \n Empirical mean: {mean_lb:.3f} (theoretical: 0) \n "
    f"Empirical variance: {var_lb:.3f} (theoretical: {biased_var_lb:.3f}) \n"
    f"Cramér-von Mises goodness of fit: {cvm_lb.statistic:.3f}"
)
plt.hist(samples_ula_lb, range=(min(grid), max(grid)), bins=100, density=True)
plt.plot(grid, sp.stats.norm.pdf(grid), label=r"$p(x)$")
plt.plot(grid, sp.stats.norm.pdf(grid, scale=np.sqrt(biased_var_lb)), label=r"$p_{\gamma_2}(x)$")
plt.legend()
plt.show()