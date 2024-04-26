import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
from scipy import interpolate
from PIL import Image
import utils_tools as tool
from scipy.special import genlaguerre

def cut_focus(E_focus, x_focus, y_focus, som_max, factor=2):
    Y_focus, X_focus = np.meshgrid(x_focus, y_focus)
    circle_radius = som_max * factor
    mask = (X_focus ** 2 + Y_focus ** 2) <= circle_radius ** 2
    E_focus_cut = np.zeros_like(E_focus)
    E_focus_cut[mask] = E_focus[mask]
    return E_focus_cut


def cut_slm(E_slm, x_slm, y_slm, beam_param, factor=2):
    _, _, w_L = beam_param
    Y_slm, X_slm = np.meshgrid(x_slm / w_L, y_slm / w_L)
    R = np.sqrt(X_slm ** 2 + Y_slm ** 2)
    circle_radius = factor
    mask = R ** 2 <= circle_radius ** 2
    E_slm_cut = np.zeros_like(E_slm)
    E_slm_cut[mask] = E_slm[mask]
    return E_slm_cut


def define_slm_plan(slm_param, print_option=True):
    x_slm_max, y_slm_max, N_x, N_y = slm_param
    x_slm = np.arange(-N_x // 2, N_x // 2) * 2 * x_slm_max / N_x
    y_slm = np.arange(-N_y // 2, N_y // 2) * 2 * y_slm_max / N_y

    extent_slm = [x_slm[0], x_slm[-1], y_slm[0], y_slm[-1]]
    formatted_extent_slm = [f'{value * 1e3:.2f} mm' for value in extent_slm]
    if print_option is True:
        print(f'SLM plane coordinates: {formatted_extent_slm}')

    return x_slm, y_slm, extent_slm


def define_focus_plan(slm_param, beam_param, print_option=True):
    lambda_0, f, w_L = beam_param
    x_slm_max, y_slm_max, N_x, N_y = slm_param

    k_0 = 2 * np.pi / lambda_0
    k_T_x = np.arange(-N_x // 2, N_x // 2) * np.pi / (2 * x_slm_max)
    k_T_y = k_T_x

    x_focus = f * k_T_x / k_0
    y_focus = f * k_T_y / k_0
    extent_focus = [x_focus[0], x_focus[-1], y_focus[0], y_focus[-1]]
    formatted_extent_focus = [f'{value * 1e6:.2f} µm' for value in extent_focus]
    if print_option is True:
        print(f'Focus plane coordinates: {formatted_extent_focus}')

    return x_focus, y_focus, extent_focus



def create_phase_vortex(x_slm, y_slm, w_L, vortex_order):
    x_slm_normalized = x_slm / w_L
    y_slm_normalized = y_slm / w_L
    X, Y = np.meshgrid(x_slm_normalized, y_slm_normalized)
    theta = np.arctan2(Y, X)
    vortex_phase = theta * vortex_order

    return vortex_phase


def gaussian_2D(x_slm, y_slm, w_L):
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    gaussian = np.exp(-2 * R ** 2 / w_L ** 2)
    indices = np.where(gaussian <= 1e-10)
    gaussian[indices] = 0
    return gaussian / np.sum(gaussian)

def laguerre_gaussian_2D(x_slm, y_slm, w_L, p, l):
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)

    radial_component = ((np.sqrt(2) * R) / w_L) ** abs(l)
    laguerre_polynomial = genlaguerre(p, abs(l))(2 * R ** 2 / w_L ** 2)
    amplitude = radial_component * laguerre_polynomial * np.exp(-R ** 2 / w_L ** 2)
    azimuthal_component = np.exp(1j * l * Theta)
    intensity = np.abs(amplitude * azimuthal_component) ** 2
    return intensity / np.sum(intensity)

def mean_square_error(a, b, norm=1):
    diff = a - b
    mse = np.sum(diff**2)
    return mse / norm

def phase_retrieval(image_focus, E_slm, beam_param, slm_param, method, max_iter=200, plot_option=False):
    print('Initialisation of the PR algorithm')
    x_slm, y_slm, extent_slm = define_slm_plan(slm_param, print_option=False)
    lambda_0, f, w_L = beam_param
    _, _, N_x, N_y = slm_param

    dx = 3.45e-6
    x_focus = np.arange(-N_x // 2, N_x // 2) * dx
    y_focus = x_focus
    extent_focus = [x_focus[0], x_focus[-1], y_focus[0], y_focus[-1]]

    intensity_slm_zp = tool.zero_padding_2N(abs(E_slm) ** 2)

    E_focus = np.sqrt(image_focus)

    som_x_focus, som_y_focus = tool.get_som(abs(E_focus) ** 2, dx)
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
    phase_vortex_pr = create_phase_vortex(x_slm, y_slm, w_L, 1) * on
    # Zero padding
    phase_vortex_pr_zp = tool.zero_padding_2N(phase_vortex_pr)
    E_focus_cut_zp = tool.zero_padding_2N(E_focus_cut)
    mse_denom = np.sum((abs(E_focus_cut_zp)) ** 2)
    # Initial guess for phase retrieval
    E_slm_pr_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * phase_vortex_pr_zp)

    costF = []
    tolerance = 1e-8  # ??

    #plt.figure()
    #plt.ion()  # Turn on interactive mode for real-time plotting

    # Initialize x and y data for the plot
    x_data = np.arange(1, max_iter + 1)  # Assuming max_iter is defined
    y_data = []

    t_start = time.time()
    for i in np.linspace(0, max_iter - 1, max_iter):
        print(f'{((i + 1) / max_iter) * 100:.1f} %')
        # Propagation to the focus
        E_focus_pr_zp = tool.fft2c(E_slm_pr_zp)
        mse = mean_square_error(abs(E_focus_pr_zp), abs(E_focus_cut_zp), mse_denom)
        costF.append(mse)
        print(f"mse: {mse}")
        # Impose amplitude of the target (ugly vortex)
        A = abs(E_focus_cut_zp) * np.exp(1j * np.angle(E_focus_pr_zp))
        # Back propagation to the slm
        B = tool.ifft2c(A)
        # Impose amplitude of the gaussian on the SLM + the phase retrieved
        E_slm_pr_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * np.angle(B))

        # Append current cost function value to y_data
        y_data.append(mse)

        # Plot the cost function in real-time
        #plt.clf()
        #plt.plot(x_data[:len(y_data)], y_data, color='b', marker='o', linestyle='-')
        #plt.xlabel('Iterations')
        #plt.ylabel('Cost Function (mse)')
        #plt.title('Cost Function vs. Iterations')
        #plt.grid(True)
        #plt.pause(0.01)  # Pause to allow for real-time plotting

    t_end = time.time()
    print(f"Loop time: {t_end - t_start:.5f} s")
    print('-----')

    # Optionally, you can also keep the plot window open after the loop ends
    #plt.ioff()
    #plt.show()

    t_end = time.time()
    print(f"Loop time: {t_end - t_start:.5f} s")
    print('-----')

    # Clip the array into original form
    E_slm_pr = tool.clip_to_original_size(E_slm_pr_zp, E_focus_cut)
    E_focus_pr = tool.clip_to_original_size(E_focus_pr_zp, E_focus_cut)

    # Calculate the correction pattern
    H = np.angle(E_slm_pr)  # Distorted phase pattern on the slm
    C = H - phase_vortex_pr  # Correction to apply on the slm for the gaussian beam if we come with a vortex
    #C = H   # Correction to apply on the slm for a nicer vortex if we come with a vortex
    C = (C + np.pi) % (2 * np.pi) - np.pi  # Correction pattern wrapped

    E_slm_fourier_zp = tool.ifft2c(E_focus_cut_zp * np.exp(1j * np.angle(E_focus_pr_zp)))
    # Padding of the E_slm_corrected before propagation
    E_slm_corrected_zp = E_slm_fourier_zp * np.exp(-1j * tool.zero_padding_2N(C)) * np.exp(-1j * phase_vortex_pr_zp)
    # Propagation
    E_focus_corrected_zp = tool.fft2c(E_slm_corrected_zp)
    # Clip to original size
    E_focus_corrected = tool.clip_to_original_size(E_focus_corrected_zp, E_focus_cut)

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

        im3 = axs_focus[1, 0].imshow((np.angle(E_focus_pr_cut)+np.pi)/(2*np.pi), extent=extent_focus, cmap='hsv')
        wavefront_error_map = (np.angle(E_focus_pr_cut)+np.pi)/(2*np.pi) # in lambda

        rows, cols = wavefront_error_map.shape
        zoom_factor = 0.08

        zoomed_rows = int(rows * zoom_factor)
        zoomed_cols = int(cols * zoom_factor)

        start_row = (rows - zoomed_rows) // 2
        end_row = start_row + zoomed_rows
        start_col = (cols - zoomed_cols) // 2
        end_col = start_col + zoomed_cols

        central_region = wavefront_error_map[start_row:end_row, start_col:end_col]

        wavelength_nm = lambda_0 * 1e9
        sigma_nm = np.std(central_region) * wavelength_nm

        strehl_ratio = np.exp(-4*np.pi ** 2 * (sigma_nm / wavelength_nm) ** 2)

        print("sigma nm:", sigma_nm)
        print("Strehl ratio:", strehl_ratio)


        axs_focus[1, 0].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[1, 0].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        axs_focus[1, 0].set_ylabel('y [m]')
        axs_focus[1, 0].set_xlabel('x [m]')
        axs_focus[1, 0].set_title('$W(x,y)$ Retrieved')
        add_colorbar(axs_focus[1, 0], im3, cmap='bwr')
        plt.tight_layout()

        axs_focus[1, 1].imshow(tool.complex2rgb(E_focus_pr_cut), extent=extent_focus, cmap='hsv')
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

        axs[1].imshow(tool.complex2rgb(E_focus_corrected_cut), extent=extent_focus)
        axs[1].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        axs[1].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        axs[1].set_ylabel('y [m]')
        axs[1].set_xlabel('x [m]')
        axs[1].set_title('$|E(x,y)|e^{iW(x,y)} $ Corrected')
        add_colorbar(axs[1], im3, cmap='hsv')
        plt.tight_layout()

        plt.show()

    return C, central_region


lambda_0 = 1030e-9
f = 20e-2
w_L = 4.0e-3
beam_param = [lambda_0, f, w_L]
k_0 = 2 * np.pi / lambda_0
w_0 = lambda_0 * f / (np.pi * w_L)
z_R = np.pi * w_0 ** 2 / lambda_0
# slm
x_slm_max = 5 * w_L  # ?
y_slm_max = 5 * w_L
N_x = 2 ** 8  # ?
N_y = N_x
slm_param = [x_slm_max, y_slm_max, N_x, N_y]
# plan
x_slm, y_slm, extent_slm = define_slm_plan(slm_param)
x_focus, y_focus, extent_focus = define_focus_plan(slm_param, beam_param)
# assume gaussian on slm
intensity_slm = gaussian_2D(x_slm, y_slm, w_L)
# vortex sur slm
phase_vortex = create_phase_vortex(x_slm, y_slm, w_L, vortex_order=1)*0
E_slm = np.sqrt(intensity_slm) * np.exp(1j * phase_vortex)

#path = "2.bmp"
path = 'images_for_demo/red.bmp'
method = 'Gerchberg-Saxton'

image = Image.open(path)
image_array = np.array(image)
image_focus, som_x, som_y = tool.process_image(image_array, dx=3.45e-6)

image_focus = tool.zero_padding_to_second(image_focus, E_slm)
C_full, central_region = phase_retrieval(image_focus, E_slm, beam_param, slm_param, method, plot_option=True, max_iter=50)
plt.imshow(central_region)
plt.show()
"""
C = tool.crop_array(C_full, extent_slm, xlim=[-7.68e-3, 7.68e-3], ylim=[-4.8e-3, 4.8e-3]) # thoses limit correspond to the slm size in m

#Interpolation of the phase pattern
x = np.arange(0, C.shape[1])
y = np.arange(0, C.shape[0])
x_new = np.linspace(0, C.shape[1]-1, 1440)
y_new = np.linspace(0, C.shape[0]-1, 1080)
f = interpolate.interp2d(x, y, C, kind='linear')
C_interpolated = f(x_new, y_new)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(C, aspect='auto',cmap='bwr')
plt.title('Original Array')
plt.subplot(1, 2, 2)
plt.imshow(C_interpolated, aspect='auto',cmap='bwr')
plt.title('Interpolated Array')
plt.tight_layout()
plt.show()
"""