import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy import ndimage
from Simulate_Dlab.Numerical_SLM.utils import *
from skimage.restoration import unwrap_phase
import Simulate_Dlab.Numerical_SLM.propagation_functions as propfunc
import Simulate_Dlab.Numerical_SLM.phase_functions as phasefunc
import Simulate_Dlab.Numerical_SLM.plot_functions as plotfunc
import time


def get_som(array, dx):
    num_rows, num_cols = array.shape

    center_of_mass = ndimage.center_of_mass(array)
    center_col, center_row = int(center_of_mass[1]), int(center_of_mass[0])

    col_vals = np.arange(num_cols)
    row_vals = np.arange(num_rows)
    col_grid, row_grid = np.meshgrid(col_vals, row_vals)
    col_deviation = col_grid - center_col
    row_deviation = row_grid - center_row

    som_x = np.sum(col_deviation ** 2 * array)
    som_y = np.sum(row_deviation ** 2 * array)
    pixel_sum = np.sum(array)

    x_som = 2 * np.sqrt(som_x / pixel_sum)
    y_som = 2 * np.sqrt(som_y / pixel_sum)
    return x_som * dx, y_som * dx


def get_M_sq(som_x, som_y, z, lambda_0, dx, plot=True):

    def beam_quality_factor_fit(z, w0, M2, z0):
        return w0 * np.sqrt(1 + (z - z0) ** 2 * (M2 * lambda_0 / (np.pi * w0 ** 2)) ** 2)

    p0 = [dx, 1, 0]


    params_x, _ = curve_fit(beam_quality_factor_fit, z, som_x, p0=p0)
    w0_x_fit, M_sq_x_fit, z0_x_fit = params_x
    print(f'M_sq_x: {abs(M_sq_x_fit):.4f},w0_x: {w0_x_fit * 1e6:.4f} µm, z0_x: {z0_x_fit * 1e3:.4f} mm')

    params_y, _ = curve_fit(beam_quality_factor_fit, z, som_y, p0=p0)
    w0_y_fit, M_sq_y_fit, z0_y_fit = params_y
    print(f'M_sq_y: {abs(M_sq_y_fit):.4f},w0_y: {w0_y_fit * 1e6:.4f} µm, z0_y: {z0_y_fit * 1e3:.4f} mm')

    M_sq_eff = np.sqrt(M_sq_x_fit * M_sq_y_fit)
    print(f'M_sq_eff: {M_sq_eff}')
    print('-----')

    if plot is True:
        fig_focus, axs_focus = plt.subplots(1, 2, figsize=(12, 6))

        axs_focus[0].plot(z, som_x, linestyle='None', marker='o', color='blue')
        axs_focus[0].plot(z, beam_quality_factor_fit(z, w0_x_fit, M_sq_x_fit, z0_x_fit),
                          label=f'M_sq: {abs(params_x[1]):.4f}, '
                                f'w0: {params_x[0] * 1e6:.4f} µm, '
                                f'z0: {params_x[2] * 1e3:.4f} mm', color='red')
        axs_focus[0].set_ylabel('z [m]')
        axs_focus[0].set_xlabel('x [m]')
        axs_focus[0].set_title('xz plane')
        axs_focus[0].legend()
        plt.tight_layout()

        axs_focus[1].plot(z, som_y, linestyle='None', marker='o', color='blue')
        axs_focus[1].plot(z, beam_quality_factor_fit(z, w0_y_fit, M_sq_y_fit, z0_y_fit),
                          label=f'M_sq: {abs(params_y[1]):.4f}, '
                                f'w0: {params_y[0] * 1e6:.4f} µm, '
                                f'z0: {params_y[2] * 1e3:.4f} mm', color='red')
        axs_focus[1].set_ylabel('z [m]')
        axs_focus[1].set_xlabel('y [m]')
        axs_focus[1].set_title('yz plane')
        axs_focus[1].legend()
        plt.tight_layout()

        plt.show()

    return params_x, params_y

def get_M_sq_and_on_axis_I(som_x, som_y, z, lambda_0, dx, pulse_param, I_cible, plot=True):
    power,rep_rate,pulse_length = pulse_param
    def beam_quality_factor_fit(z, w0, M2, z0):
        return w0 * np.sqrt(1 + (z - z0) ** 2 * (M2 * lambda_0 / (np.pi * w0 ** 2)) ** 2)

    p0 = [dx, 1, 0]

    params_x, _ = curve_fit(beam_quality_factor_fit, z, som_x, p0=p0)
    w0_x_fit, M_sq_x_fit, z0_x_fit = params_x
    print(f'M_sq_x: {abs(M_sq_x_fit):.4f},w0_x: {w0_x_fit * 1e6:.4f} µm, z0_x: {z0_x_fit * 1e3:.4f} mm')

    params_y, _ = curve_fit(beam_quality_factor_fit, z, som_y, p0=p0)
    w0_y_fit, M_sq_y_fit, z0_y_fit = params_y
    print(f'M_sq_y: {abs(M_sq_y_fit):.4f},w0_y: {w0_y_fit * 1e6:.4f} µm, z0_y: {z0_y_fit * 1e3:.4f} mm')

    M_sq_eff = np.sqrt(M_sq_x_fit * M_sq_y_fit)
    print(f'M_sq_eff: {M_sq_eff}')
    print('-----')

    E_pulse = power/rep_rate #J
    print(E_pulse*1e6)
    I_peak_x = 0.94 * 2 * E_pulse/(pulse_length*np.pi*beam_quality_factor_fit(z, w0_x_fit, M_sq_x_fit, z0_x_fit)**2)
    print(np.max(I_peak_x))

    C = 2*0.94/(pulse_length*np.pi*rep_rate)
    P = I_cible/C * beam_quality_factor_fit(z, w0_x_fit, M_sq_x_fit, z0_x_fit)**2
    print(P)

    I_test = C * P/beam_quality_factor_fit(z, w0_x_fit, M_sq_x_fit, z0_x_fit)**2
    print(I_test)


    if plot is True:
        fig_focus, axs_focus = plt.subplots(figsize=(8, 6))

        axs_focus.plot(z, I_peak_x, linestyle='-', marker='None', color='blue')
        axs_focus.plot(z, I_test, linestyle='--', marker='None', color='black')
        axs_focus_twin = axs_focus.twinx()
        axs_focus_twin.plot(z, beam_quality_factor_fit(z, w0_x_fit, M_sq_x_fit, z0_x_fit),
                            label=f'M_sq: {abs(M_sq_x_fit):.4f}, '
                                  f'w0: {w0_x_fit * 1e6:.4f} µm, '
                                  f'z0: {z0_x_fit * 1e3:.4f} mm', color='red')

        axs_focus.set_ylabel('I_peak [W/cm2]')
        axs_focus_twin.set_ylabel('som')
        axs_focus.set_xlabel('z [m]')
        axs_focus.set_title('xz plane')
        plt.tight_layout()

        plt.show()

    return params_x, params_y


def phase_retrieval(E_slm, beam_param, slm_param, method, max_iter=200, plot_option=False):
    print('Initialisation of the PR algorithm')
    x_slm, y_slm, extent_slm = propfunc.define_slm_plan(slm_param, print_option=False)
    x_focus, y_focus, extent_focus = propfunc.define_focus_plan(slm_param, beam_param, print_option=False)
    lambda_0, f, w_L,w_0 = beam_param
    L = 2 * x_focus[-1]
    _, _, N_x, N_y = slm_param
    dx = L / N_x

    intensity_slm_zp = zero_padding_2N(abs(E_slm) ** 2)
    phase_slm_zp = zero_padding_2N(np.angle(E_slm))
    E_slm_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * phase_slm_zp)

    E_focus_zp = propfunc.propagate(E_slm_zp, L=L, method='fourier', wavelength=lambda_0, dz=+1)
    E_focus = clip_to_original_size(E_focus_zp, E_slm)

    som_x_focus, som_y_focus = get_som(abs(E_focus) ** 2, dx)
    som_max = np.max([som_x_focus, som_y_focus])

    E_focus_cut = cut_focus(E_focus, x_focus, y_focus, som_max, factor=2)

    if method == 'Vortex':
        on = 1
        print('-----')
        print('Vortex selected')

    if method == 'Gerchberg-Saxton':
        on = 0
        print('-----')
        print('GS selected')

    # Create vortex
    phase_vortex_pr = phasefunc.create_phase_vortex(x_slm, y_slm, w_L, 1) * on
    # Zero padding
    phase_vortex_pr_zp = zero_padding_2N(phase_vortex_pr)
    E_focus_cut_zp = zero_padding_2N(E_focus_cut)
    # Initial guess for phase retrieval
    E_slm_pr_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * phase_vortex_pr_zp)

    corr_list = []
    tolerance = 1e-2  # ??

    t_start = time.time()
    for i in np.linspace(0, max_iter - 1, max_iter):
        print(f'{((i + 1) / max_iter) * 100:.1f} %')
        # Propagation to the focus
        E_focus_pr_zp = propfunc.fft2c(E_slm_pr_zp)
        corr_temp = abs(np.corrcoef(E_focus_pr_zp.flatten(), E_focus_cut_zp.flatten())[0, 1])
        corr_list.append(corr_temp)
        print(f"Pearson coeff: {corr_temp}")
        # Impose amplitude of the target (ugly vortex)
        A = abs(E_focus_cut_zp) * np.exp(1j * np.angle(E_focus_pr_zp))
        # Back propagation to the slm
        B = propfunc.ifft2c(A)
        # Impose amplitude of the gaussian on the SLM + the phase retrieved
        E_slm_pr_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * np.angle(B))

        if i > 0 and abs((corr_list[-1] - corr_list[-2]) / corr_list[-2]) < tolerance:
            print('-----')
            print(f"Converged with {int(i + 1)} iterations")
            break

    t_end = time.time()
    print(f"Loop time: {t_end - t_start:.5f} s")
    print('-----')

    # Clip the array into original form
    E_slm_pr = clip_to_original_size(E_slm_pr_zp, E_focus_cut)
    E_focus_pr = clip_to_original_size(E_focus_pr_zp, E_focus_cut)

    # Calculate the correction pattern
    H = np.angle(E_slm_pr)  # Distorted phase pattern on the slm
    C = H - phase_vortex_pr  # Correction to apply on the slm for the gaussian beam if we come with a vortex
    # C = H   # Correction to apply on the slm for a nicer vortex if we come with a vortex
    C = (C + np.pi) % (2 * np.pi) - np.pi  # Correction pattern wrapped

    E_slm_fourier_zp = propfunc.ifft2c(E_focus_cut_zp * np.exp(1j * np.angle(E_focus_pr_zp)))
    # Padding of the E_slm_corrected before propagation
    E_slm_corrected_zp = E_slm_fourier_zp * np.exp(-1j * zero_padding_2N(C)) * np.exp(-1j * phase_vortex_pr_zp)
    # Propagation
    E_focus_corrected_zp = propfunc.fft2c(E_slm_corrected_zp)
    # Clip to original size
    E_focus_corrected = clip_to_original_size(E_focus_corrected_zp, E_focus_cut)

    # Cutting the arrays
    factor_plot = 2
    factor_slm = 1.6
    C = cut_slm(C, x_slm, y_slm, beam_param, factor=factor_slm)
    E_focus_cut = cut_focus(E_focus, x_focus, y_focus, som_max, factor=factor_plot)
    E_focus_pr_cut = cut_focus(E_focus_pr, x_focus, y_focus, som_max, factor=factor_plot)
    E_focus_corrected_cut = cut_focus(E_focus_corrected, x_focus, y_focus, som_max, factor=factor_plot)

    def add_colorbar(ax, im, cmap='turbo'):
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, cmap=cmap)

    # Plot option
    if plot_option is True:
        fig_focus, axs_focus = plt.subplots(2, 2, figsize=(8, 6))

        im1 = axs_focus[0, 0].imshow(abs(E_focus_cut) ** 2, extent=extent_focus, cmap='turbo')
        axs_focus[0, 0].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[0, 0].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[0, 0].set_ylabel('y [m]')
        axs_focus[0, 0].set_xlabel('x [m]')
        axs_focus[0, 0].set_title('$|E(x,y)|^2$ Measured')
        add_colorbar(axs_focus[0, 0], im1, cmap='turbo')
        plt.tight_layout()

        im2 = axs_focus[0, 1].imshow(abs(E_focus_pr_cut) ** 2, extent=extent_focus, cmap='turbo')
        axs_focus[0, 1].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[0, 1].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[0, 1].set_ylabel('y [m]')
        axs_focus[0, 1].set_xlabel('x [m]')
        axs_focus[0, 1].set_title('$|E(x,y)|^2$ Retrieved')
        add_colorbar(axs_focus[0, 1], im2, cmap='turbo')
        plt.tight_layout()

        im3 = axs_focus[1, 0].imshow(np.angle(E_focus_pr_cut), extent=extent_focus, cmap='hsv')
        axs_focus[1, 0].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[1, 0].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[1, 0].set_ylabel('y [m]')
        axs_focus[1, 0].set_xlabel('x [m]')
        axs_focus[1, 0].set_title('$W(x,y)$ Retrieved')
        add_colorbar(axs_focus[1, 0], im3, cmap='bwr')
        plt.tight_layout()

        axs_focus[1, 1].imshow(plotfunc.complex2rgb(E_focus_pr_cut), extent=extent_focus, cmap='hsv')
        axs_focus[1, 1].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[1, 1].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[1, 1].set_ylabel('y [m]')
        axs_focus[1, 1].set_xlabel('x [m]')
        axs_focus[1, 1].set_title('$|E(x,y)|e^{iW(x,y)} $ Retrieved')
        add_colorbar(axs_focus[1, 1], im3, cmap='hsv')
        plt.tight_layout()

        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        im4 = axs[0].imshow(C, extent=extent_slm, cmap='bwr')
        axs[0].set_ylim((-4.8e-3, 4.8e-3))
        axs[0].set_xlim((-7.68e-3, 7.68e-3))
        axs[0].set_ylabel('y [m]')
        axs[0].set_xlabel('x [m]')
        axs[0].set_title('Phase on SLM')
        add_colorbar(axs[0], im4, cmap='bwr')
        plt.tight_layout()

        axs[1].imshow(plotfunc.complex2rgb(E_focus_corrected_cut), extent=extent_focus)
        axs[1].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        axs[1].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        axs[1].set_ylabel('y [m]')
        axs[1].set_xlabel('x [m]')
        axs[1].set_title('$|E(x,y)|e^{iW(x,y)} $ Corrected')
        add_colorbar(axs[1], im3, cmap='hsv')
        plt.tight_layout()

        plt.show()

    return 1