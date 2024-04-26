import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from functools import lru_cache, partial
import warnings
from scipy.integrate import IntegrationWarning


a, b, c, d = -1, 0, 6, 7
sigma, beta, rho_0 = 1, 1, 1
alpha_1, beta_1 = 1 / (b - a), -a / (b - a)
alpha_2, beta_2 = -1 / (d - c), d / (d - c)
delta_k_c = -beta * rho_0

def func_rho(z_prime):
    if a <= z_prime < b:
        return alpha_1 * z_prime + beta_1
    elif b <= z_prime <= c:
        return 1
    elif c < z_prime <= d:
        return alpha_2 * z_prime + beta_2
    else:
        return 0

@lru_cache(maxsize=None)
def integrate_alpha(z, L_med):
    integrated_alpha, _ = quad(lambda z_prime: func_rho(z_prime) * sigma / 2, z, L_med)
    return integrated_alpha

def integrand(z, e, delta_k_c, beta):
    rho_z = func_rho(z)
    delta_k_z = delta_k_c + beta * rho_z
    exp_part = np.exp(1j * delta_k_z * z - integrate_alpha(z, e))
    return rho_z * exp_part

def vectorized_integrals(L_med):
    N_out = np.zeros_like(L_med, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)

        for i, e in enumerate(L_med):
            integrand_e = partial(integrand, e=e, delta_k_c=delta_k_c, beta=beta)
            integral, _ = quad(integrand_e, 0, e)
            N_out[i] = np.abs(integral) ** 2

    return N_out

N_medium = 30
L_min =0
L_max = 8
L_med = np.linspace(L_min, L_max, N_medium)
N_out = vectorized_integrals(L_med)

rho_plot = np.array([func_rho(i) for i in L_med])

plt.figure()
plt.plot(L_med, N_out, label='Output flux')
plt.plot(L_med, rho_plot, label='Density', linestyle='--')
plt.xlabel('z (arb. units)')
plt.ylabel('Output flux (arb. units)')
plt.legend()
plt.show()
