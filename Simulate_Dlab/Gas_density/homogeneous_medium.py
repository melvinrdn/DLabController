import numpy as np
import matplotlib.pyplot as plt

L_liste = np.linspace(0, 12, 101)

rho_0 = 1
L_abs = 1 / rho_0
L_coh_1 = 1 * L_abs
L_coh_2 = 5 * L_abs
L_coh_3 = 1000 * L_abs

deltak1 = np.pi / L_coh_1
deltak2 = np.pi / L_coh_2
deltak3 = np.pi / L_coh_3


def calculate_flux(L_liste, L_abs, L_coh):
    return (rho_0 ** 2 * 4 * L_abs ** 2) / (1 + 4 * np.pi ** 2 * (L_abs ** 2 / L_coh ** 2)) * (
            1 + np.exp(-L_liste / L_abs) - 2 * np.cos(np.pi * L_liste / L_coh) * np.exp(-L_liste / (2 * L_abs)))


flux_1 = calculate_flux(L_liste, L_abs, L_coh_1)
flux_2 = calculate_flux(L_liste, L_abs, L_coh_2)
flux_3 = calculate_flux(L_liste, L_abs, L_coh_3)

plt.figure(1)
plt.plot(L_liste, flux_1 / 4, label='$L_{coh} = L_{abs}$')

plt.plot(L_liste, flux_2 / 4, label='$L_{coh} = 5L_{abs}$')

plt.plot(L_liste, flux_3 / 4, label='$L_{coh} \gg L_{abs}$')

plt.xlabel('$z/L_{abs}^{min}$ (prop. distance / $L_{abs}$ at peak pressure)')
plt.ylabel('$I/I_{max}$, (flux/abs-limited max. flux)')
plt.legend(loc='best')
plt.ylim([0, 1])
plt.xlim([0, 12])
plt.xticks([0, 3, 6, 9, 12])
plt.grid(True, linestyle='--')
plt.show()
