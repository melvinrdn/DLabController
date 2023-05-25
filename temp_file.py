import numpy as np
import matplotlib.pyplot as plt

filepath = 'ressources/calibration/calibration_19-05-2023.txt'
pharos_att, pharos_pp, red_p = np.loadtxt(filepath, delimiter='\t', skiprows=1, unpack=True)

plt.figure(1)
plt.subplot(121)
plt.plot(pharos_att[:9], red_p[:9], linestyle='None', marker='o')
plt.xlabel('Pharos att')
plt.ylabel('Red power (W)')
plt.ylim([0, 4])
# Perform linear fit for the first graph
coefficients_1 = np.polyfit(pharos_att[:9], red_p[:9], 1)
equation_1 = f'y = {coefficients_1[0]:.2f}x + {coefficients_1[1]:.2f}'
fit_1 = np.polyval(coefficients_1, pharos_att[:9])
plt.plot(pharos_att[:9], fit_1, label='Linear Fit', linestyle='--')
plt.legend()

plt.subplot(122)
plt.plot(1 / pharos_pp[9:21], red_p[9:21], linestyle='None', marker='o')
plt.xlabel('1/Pharos pp')
plt.ylim([0, 4])
# Perform linear fit for the second graph
coefficients_2 = np.polyfit(1 / pharos_pp[9:21], red_p[9:21], 1)
equation_2 = f'y = {coefficients_2[0]:.2f}x + {coefficients_2[1]:.2f}'
fit_2 = np.polyval(coefficients_2, 1 / pharos_pp[9:21])
plt.plot(1 / pharos_pp[9:21], fit_2, label='Linear Fit', linestyle='--')
plt.legend()

# Display equations and coefficients in the console
print(f"Attenuation : {equation_1}")
print(f"Pulse Picker: {equation_2}")

plt.show()
