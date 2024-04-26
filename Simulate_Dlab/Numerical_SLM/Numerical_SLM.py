## import library
import matplotlib.pyplot as plt
import numpy as np
import phase_functions as phasefunc
import plot_functions as plotfunc
import propagation_functions as propfunc
import diagnostic_functions as diagfunc
from utils import *

## Beam parameters-
lambda_0 = 1030e-9
f = 20e-2
w_L = 4.0e-3
k_0 = 2 * np.pi / lambda_0
#w_0 = 20e-6
w_0 = lambda_0 * f / (np.pi * w_L)
print(w_0)
z_R = np.pi * w_0 ** 2 / lambda_0  # 1.22 mm
beam_param = [lambda_0, f, w_L, w_0]

## SLM parameters
x_slm_max = 7 * w_L
y_slm_max = 7 * w_L
N_x = 2 ** 8
N_y = N_x
slm_param = [x_slm_max, y_slm_max, N_x, N_y]

## Creating SLM plane and focus plane
x_slm, y_slm, extent_slm = propfunc.define_slm_plan(slm_param)
x_focus, y_focus, extent_focus = propfunc.define_focus_plan(slm_param, beam_param)
Y_slm, X_slm = np.meshgrid(x_slm, y_slm)
Y_focus, X_focus = np.meshgrid(x_focus, y_focus)
L = 2 * x_focus[-1]
dx = L / N_x

## Phase options
# Zernike
defocus = 0
o_astigmatism = 0
v_astigmatism = 0.1
x_coma = 0.3
y_coma = 0
o_trefoil = 0
v_trefoil = 0
coeffs_zernike = [o_astigmatism, defocus, v_astigmatism, v_trefoil, y_coma, x_coma, o_trefoil]
phase_zernike = phasefunc.create_phase_zernike(x_slm, y_slm, w_L, coeffs_zernike, R_threshold=1.6)*0
# Vortex
vortex_order = 1
phase_vortex = phasefunc.create_phase_vortex(x_slm, y_slm, w_L, vortex_order)*0
# Phase jump (s_gaussian generation)
radius_jump = 0.3 * w_L
phase_jump = phasefunc.create_phase_jump(x_slm, y_slm, radius_jump)*0
# Phase jump hgb
radius_jump_hgb2 = 1 * w_L
phase_jump_hgb2 =(phasefunc.create_phase_jump(x_slm, y_slm, 3 * radius_jump_hgb2) - phasefunc.create_phase_jump(x_slm,
                                                                                                               y_slm,
                                                                                                               2.5*radius_jump_hgb2)) \
                + (phasefunc.create_phase_jump(x_slm, y_slm, 2 * radius_jump_hgb2) - phasefunc.create_phase_jump(x_slm,
                                                                                                               y_slm,
                                                                                                               1.5*radius_jump_hgb2))\
                    + (phasefunc.create_phase_jump(x_slm, y_slm,  radius_jump_hgb2) - phasefunc.create_phase_jump(x_slm,
                                                                                                               y_slm,
                                                                                                               0.5*radius_jump_hgb2))

# Phase jump hgb
radius_jump_hgb = 0.7 * w_L
phase_jump_hgb =(phasefunc.create_phase_jump(x_slm, y_slm, 2 * radius_jump_hgb) - phasefunc.create_phase_jump(x_slm,
                                                                                                               y_slm,
                                                                                                               radius_jump_hgb))
## Creating field on the SLM
intensity_slm = propfunc.gaussian_2D(x_slm, y_slm, w_L)

phase_slm = phase_vortex + phase_zernike + phase_jump + phase_jump_hgb+ phase_jump_hgb2*0
E_slm = np.sqrt(intensity_slm) * np.exp(1j * phase_slm)
E_slm = np.nan_to_num(E_slm)

#plotfunc.plot_slm_plane(E_slm,extent_slm,unwrap=False)

# zero padding
intensity_slm_zp = zero_padding_2N(intensity_slm)

phase_slm_zp = zero_padding_2N(phase_slm)
E_slm_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * phase_slm_zp)

## Fourier transform to reach the focus plane and re clipping
E_focus_zp = propfunc.propagate(E_slm_zp, L=L, method='fourier', wavelength=lambda_0, dz=+1)

E_focus = clip_to_original_size(E_focus_zp, E_slm)

#plt.imshow(np.abs(E_focus)**2)
#plt.show()
#E_focus = propfunc.hgb_2D(x_focus,y_focus,w_0,2)
#E_slm_tf = propfunc.propagate(E_focus, L=L, method='fourier', wavelength=lambda_0, dz=-1)
#plt.imshow(np.abs(E_slm_tf)**2)
#plt.show()
#plt.imshow(np.angle(E_slm_tf))
# plt.show()

dz = [-5,5]  # Range in units of z_R
z_steps = 50  # Number of steps
E_z_cut, z, som_x, som_y = propfunc.propagation_scan(E_focus, dz, z_steps, beam_param, slm_param, plot_option=False)

## Diagnostics

# M squared diagnotics
# power = 6
# rep_rate = 20e3
# pulse_length = 180e-15
# pulse_param = [power,rep_rate,pulse_length]
# diagfunc.get_M_sq_and_on_axis_I(som_x,som_y,z,lambda_0,dx,pulse_param,I_cible=1e18, plot=True)
# Transverse plots
plotfunc.plot_transverse_planes(dz, E_z_cut, som_x, som_y, N_x, N_y, x_focus, y_focus,z_R)
#plotfunc.plot_all_cross_sections(E_z_cut,z,N_x,N_y)

# Phase retrieval
#diagfunc.phase_retrieval(E_slm, beam_param, slm_param, 'Gerchberg-Saxton', plot_option=True)
#diagfunc.get_M_sq(som_x,som_y,z,lambda_0,w_0)