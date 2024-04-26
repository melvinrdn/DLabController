import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage


def circ(x, y, D):
    circle = (x ** 2 + y ** 2) <= (D / 2) ** 2
    return circle


def zero_padding_2N(original_array):
    N = original_array.shape[0]
    pad_width = ((N // 2, N // 2), (N // 2, N // 2))
    padded_array = np.pad(original_array, pad_width, mode='constant', constant_values=0)
    return padded_array


def clip_to_original_size(padded_array, original_array):
    original_size = original_array.shape
    N, M = padded_array.shape
    clipped_array = padded_array[N // 2 - original_size[0] // 2: N // 2 + original_size[0] // 2,
                    M // 2 - original_size[1] // 2: M // 2 + original_size[1] // 2]
    return clipped_array


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


def define_slm_plan(slm_param, print_option=True):
    x_slm_max, y_slm_max, N_x, N_y = slm_param
    x_slm = np.arange(-N_x // 2, N_x // 2) * 2 * x_slm_max / N_x
    y_slm = np.arange(-N_y // 2, N_y // 2) * 2 * y_slm_max / N_y

    extent_slm = [x_slm[0], x_slm[-1], y_slm[0], y_slm[-1]]
    formatted_extent_slm = [f'{value * 1e3:.2f} mm' for value in extent_slm]
    if print_option is True:
        print(f'SLM plane coordinates: {formatted_extent_slm}')

    return x_slm, y_slm, extent_slm


def create_phase_vortex(x_slm, y_slm, w_L, vortex_order):
    x_slm_normalized = x_slm / w_L
    y_slm_normalized = y_slm / w_L
    X, Y = np.meshgrid(x_slm_normalized, y_slm_normalized)
    theta = np.arctan2(Y, X)
    vortex_phase = theta * vortex_order

    return vortex_phase


def ifft2c(array):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array), norm='ortho'))


def fft2c(array):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array), norm='ortho'))


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert a 3D hsv np.ndarray to rgb.
    """
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def complex2rgb(u, amplitudeScalingFactor=1, scalling=1):
    h = np.angle(u)
    h = (h + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = np.abs(u)
    if amplitudeScalingFactor != 1:
        v[v > amplitudeScalingFactor * np.max(v)] = amplitudeScalingFactor * np.max(v)
    if scalling != 1:
        local_max = np.max(v)
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)
        print(f'ratio: {local_max / scalling}, max(v): {np.max(v)}')

        v *= local_max / scalling
        print(f'max(v): {np.max(v)}')

    else:
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)

    hsv = np.dstack([h, s, v])
    rgb = hsv2rgb(hsv)
    return rgb


def gaussian_2D(x_slm, y_slm, w_L):
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    gaussian = np.exp(-2 * R ** 2 / w_L ** 2)
    indices = np.where(gaussian <= 1e-10)
    gaussian[indices] = 0
    return gaussian / np.sum(gaussian)


def zero_padding_to_second(original_array, second_array):
    original_size = original_array.shape
    second_size = second_array.shape
    size_diff = np.subtract(second_size, original_size)
    pad_width = tuple((diff // 2, diff // 2) for diff in size_diff)
    padded_array = np.pad(original_array, pad_width, mode='constant', constant_values=0)
    return padded_array


def phase_retrieval(image_focus, E_slm, beam_param, slm_param, method, max_iter=200, plot_option=False):
    print('Initialisation of the PR algorithm')
    x_slm, y_slm, extent_slm = define_slm_plan(slm_param, print_option=False)
    lambda_0, f, w_L = beam_param
    _, _, N_x, N_y = slm_param

    dx = 3.45e-6
    x_focus = np.arange(-N_x // 2, N_x // 2) * dx
    y_focus = x_focus
    extent_focus = [x_focus[0], x_focus[-1], y_focus[0], y_focus[-1]]

    intensity_slm_zp = zero_padding_2N(abs(E_slm) ** 2)

    E_focus = np.sqrt(image_focus)

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
    phase_vortex_pr = create_phase_vortex(x_slm, y_slm, w_L, 1) * on
    # Zero padding
    phase_vortex_pr_zp = zero_padding_2N(phase_vortex_pr)
    E_focus_cut_zp = zero_padding_2N(E_focus_cut)
    # Initial guess for phase retrieval
    E_slm_pr_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * phase_vortex_pr_zp)

    corr_list = []
    tolerance = 1e-7  # ??

    t_start = time.time()
    for i in np.linspace(0, max_iter - 1, max_iter):
        print(f'{((i + 1) / max_iter) * 100:.1f} %')
        # Propagation to the focus
        E_focus_pr_zp = fft2c(E_slm_pr_zp)
        corr_temp = abs(np.corrcoef(E_focus_pr_zp.flatten(), E_focus_cut_zp.flatten())[0, 1])
        corr_list.append(corr_temp)
        #print(f"Pearson coeff: {corr_temp}")
        # Impose amplitude of the target (ugly vortex)
        A = abs(E_focus_cut_zp) * np.exp(1j * np.angle(E_focus_pr_zp))
        # Back propagation to the slm
        B = ifft2c(A)
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
    #C = H - phase_vortex_pr  # Correction to apply on the slm for the gaussian beam if we come with a vortex
    C = H   # Correction to apply on the slm for a nicer vortex if we come with a vortex
    C = (C + np.pi) % (2 * np.pi) - np.pi  # Correction pattern wrapped

    E_slm_fourier_zp = ifft2c(E_focus_cut_zp * np.exp(1j * np.angle(E_focus_pr_zp)))
    # Padding of the E_slm_corrected before propagation
    E_slm_corrected_zp = E_slm_fourier_zp * np.exp(-1j * zero_padding_2N(C)) * np.exp(-1j * phase_vortex_pr_zp)
    # Propagation
    E_focus_corrected_zp = fft2c(E_slm_corrected_zp)
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

        axs_focus[1, 1].imshow(complex2rgb(E_focus_pr_cut), extent=extent_focus, cmap='hsv')
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

        axs[1].imshow(complex2rgb(E_focus_corrected_cut), extent=extent_focus)
        axs[1].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        axs[1].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        axs[1].set_ylabel('y [m]')
        axs[1].set_xlabel('x [m]')
        axs[1].set_title('$|E(x,y)|e^{iW(x,y)} $ Corrected')
        add_colorbar(axs[1], im3, cmap='hsv')
        plt.tight_layout()

        plt.show()

    return C

"""
lambda_0 = 1030e-9
f = 25e-2
w_L = 4.0e-3
beam_param = [lambda_0, f, w_L]
k_0 = 2 * np.pi / lambda_0
w_0 = lambda_0 * f / (np.pi * w_L)
z_R = np.pi * w_0 ** 2 / lambda_0
x_slm_max = 5 * w_L
y_slm_max = 5 * w_L
N_x = 2 ** 7
N_y = N_x
slm_param = [x_slm_max, y_slm_max, N_x, N_y]

x_slm, y_slm, extent_slm = define_slm_plan(slm_param, print_option=False)

intensity_slm = gaussian_2D(x_slm, y_slm, w_L)

on = 1
phase_vortex = create_phase_vortex(x_slm, y_slm, w_L, vortex_order=1) * on
E_slm = np.sqrt(intensity_slm) * np.exp(1j * phase_vortex)

if on == 1:
    method = 'Vortex'
    path = "red_vortex_for_vgs_demo.bmp"
else:
    method = 'Gerchberg-Saxton'
    path= "red_focus_for_gs_demo.bmp"


image_array = np.array(Image.open(path))
image_focus, som_x, som_y = process_image(image_array)
image_focus = zero_padding_to_second(image_focus, E_slm)

C = phase_retrieval(image_focus, E_slm, beam_param, slm_param, method, plot_option=True)

"""