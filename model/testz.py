import numpy as np
from ressources.slm_infos import chip_height, chip_width, slm_size
import matplotlib.pyplot as plt
x = np.linspace(-chip_width * 500, chip_width * 500, slm_size[1])
y = np.linspace(-chip_height * 500, chip_height * 500, slm_size[0])
[X, Y] = np.meshgrid(x, y)

rho = np.sqrt(X ** 2 + Y ** 2)
rho /= (np.max(rho)/2)
theta = np.arctan2(Y, X)

desired_radius = 1
indices = np.where(rho <= desired_radius)

coeffs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
p1 = np.zeros_like(X)
p2 = np.zeros_like(X)
p3 = np.zeros_like(X)
p4 = np.zeros_like(X)
p5 = np.zeros_like(X)
p6 = np.zeros_like(X)
p7 = np.zeros_like(X)
p8 = np.zeros_like(X)
p9 = np.zeros_like(X)
p10 = np.zeros_like(X)

p1[indices] = coeffs[0] * 1 * np.cos(0 * theta[indices])
p2[indices] = coeffs[1] * 2 * rho[indices] * np.cos(1 * theta[indices])
p3[indices] = coeffs[2] * 2 * rho[indices] * np.sin(1 * theta[indices])
p4[indices] = coeffs[3] * np.sqrt(3) * (2 * rho[indices] ** 2 - 1)
p5[indices] = coeffs[4] * np.sqrt(6) * rho[indices] ** 2 * np.cos(2 * theta[indices])
p6[indices] = coeffs[5] * np.sqrt(6) * rho[indices] ** 2 * np.sin(2 * theta[indices])
p7[indices] = coeffs[6] * np.sqrt(8) * (3 * rho[indices] ** 3 - 2 * rho[indices]) * np.cos(1 * theta[indices])
p8[indices] = coeffs[7] * np.sqrt(8) * (3 * rho[indices] ** 3 - 2 * rho[indices]) * np.sin(1 * theta[indices])
p9[indices] = coeffs[8] * np.sqrt(8) * rho[indices] ** 3 * np.cos(3 * theta[indices])
p10[indices] = coeffs[9] * np.sqrt(8) * rho[indices] ** 3 * np.sin(3 * theta[indices])

p1_n = np.sum(p1[indices] ** 2) / len(p1[indices])
p2_n = np.sum(p2[indices] ** 2) / len(p2[indices])
p3_n = np.sum(p3[indices] ** 2) / len(p3[indices])
p4_n = np.sum(p4[indices] ** 2) / len(p4[indices])
p5_n = np.sum(p5[indices] ** 2) / len(p5[indices])
p6_n = np.sum(p6[indices] ** 2) / len(p6[indices])
p7_n = np.sum(p7[indices] ** 2) / len(p7[indices])
p8_n = np.sum(p8[indices] ** 2) / len(p8[indices])
p9_n = np.sum(p9[indices] ** 2) / len(p9[indices])
p10_n = np.sum(p10[indices] ** 2) / len(p10[indices])

print(p1_n)
print(p2_n)
print(p3_n)
print(p4_n)
print(p5_n)
print(p6_n)
print(p7_n)
print(p8_n)
print(p9_n)
print(p10_n)


plt.imshow(p5)
plt.show()