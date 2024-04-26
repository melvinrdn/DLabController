import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

data_path = '//fysfile01/atomhome$/me2120re/My Documents/Doctorat/Projets/Tailored_HHG/Data_04_09_many_profile/'

gaussienne = h5py.File(data_path + 'gaussienne.h5', 'r')
image_gaussienne = np.asarray(gaussienne.get('image_T'))
e_axis_gaussienne = np.asarray(gaussienne.get('e_axis'))

flattop = h5py.File(data_path + 'flattop.h5', 'r')
image_flattop = np.asarray(flattop.get('image_T'))
e_axis_flattop = np.asarray(flattop.get('e_axis'))

hermite = h5py.File(data_path + 'hermite.h5', 'r')
image_hermite = np.asarray(hermite.get('image_T'))
e_axis_hermite = np.asarray(hermite.get('e_axis'))

laguerre = h5py.File(data_path  + 'laguerre.h5', 'r')
image_laguerre = np.asarray(laguerre.get('image_T'))
e_axis_laguerre = np.asarray(laguerre.get('e_axis'))

vortex = h5py.File(data_path  + 'vortex.h5', 'r')
image_vortex = np.asarray(vortex.get('image_T'))
e_axis_vortex = np.asarray(vortex.get('e_axis'))

y_axis = np.arange(0,512) * 13e-6 / 0.15 * 1e3 - 26
"""
plt.figure(figsize=(5, 3))
plt.imshow(image_gaussienne, extent=[e_axis_gaussienne[0], e_axis_gaussienne[-1], y_axis[0], y_axis[-1]], aspect='auto', cmap='turbo')
#plt.xlabel('Energy (eV)', fontsize=16)
plt.ylabel('Divergence (mrad)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)  # Adjust size of tick labels
plt.xlim(20, 40)
plt.subplots_adjust(right=0.85)
plt.gca().yaxis.set_label_position("right")
plt.gca().yaxis.tick_right()
plt.ylim(-8,8)
plt.tight_layout()

plt.figure(figsize=(5, 3))
plt.imshow(image_flattop, extent=[e_axis_flattop[0], e_axis_flattop[-1], y_axis[0], y_axis[-1]], aspect='auto', cmap='turbo')
#plt.xlabel('Energy (eV)', fontsize=16)
plt.ylabel('Divergence (mrad)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)  # Adjust size of tick labels
plt.xlim(20, 40)

plt.subplots_adjust(right=0.85)
plt.gca().yaxis.set_label_position("right")
plt.gca().yaxis.tick_right()
plt.ylim(-8,8)
plt.tight_layout()

plt.figure(figsize=(5, 3))
plt.imshow(image_hermite, extent=[e_axis_hermite[0], e_axis_hermite[-1], y_axis[0], y_axis[-1]], aspect='auto', cmap='turbo')
#plt.xlabel('Energy (eV)', fontsize=16)
plt.ylabel('Divergence (mrad)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)  # Adjust size of tick labels
plt.xlim(20, 40)
plt.subplots_adjust(right=0.85)
plt.gca().yaxis.set_label_position("right")
plt.gca().yaxis.tick_right()
plt.ylim(-8,8)
plt.tight_layout()

plt.figure(figsize=(5, 3))
plt.imshow(image_vortex, extent=[e_axis_vortex[0], e_axis_vortex[-1], y_axis[0], y_axis[-1]], aspect='auto', cmap='turbo')
plt.ylabel('Divergence (mrad)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)  # Adjust size of tick labels
plt.xlim(20, 40)
plt.subplots_adjust(right=0.85)
plt.gca().yaxis.set_label_position("right")
plt.gca().yaxis.tick_right()
plt.ylim(-8,8)
plt.tight_layout()
"""

plt.figure(figsize=(5, 3))
plt.imshow(image_laguerre, extent=[e_axis_laguerre[0], e_axis_laguerre[-1], y_axis[0], y_axis[-1]], aspect='auto', cmap='turbo')
plt.xlabel('Energy (eV)', fontsize=15)
plt.ylabel('Divergence (mrad)', fontsize=15)
plt.ylim(-5,5)
plt.xlim(20, 38)
plt.tight_layout()


plt.show()
