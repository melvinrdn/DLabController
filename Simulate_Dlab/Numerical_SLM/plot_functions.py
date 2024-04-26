import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.restoration import unwrap_phase


def plot_focus_plane(E_z, som_x, som_y, extent_focus=None, circle=False, fix_axis=False, unwrap=False):
    """
    Plot intensity and phase distributions in the focus plane.

    Parameters:
    -----------
    E_slm: numpy.ndarray
        Field distribution on the focus plane.
    extent_focus : tuple
        Extent of the focus plane in (x_min, x_max, y_min, y_max).

    Returns:
    --------
    None
    """
    fig_focus, axs_focus = plt.subplots(1, 3, figsize=(12, 6))

    som_eff = np.mean([som_y, som_x])
    w_0 = som_eff

    axs_focus[0].imshow(abs(E_z) ** 2, extent=extent_focus, cmap='turbo')
    if fix_axis is True:
        axs_focus[0].set_xlim((-200e-6, 200e-6))
        axs_focus[0].set_ylim((-200e-6, 200e-6))
    axs_focus[0].set_ylabel('y [m]')
    axs_focus[0].set_xlabel('x [m]')
    axs_focus[0].set_title('$|E(x,y)|^2$')
    if circle is True:
        circle_w_0 = patches.Circle((0, 0), w_0, fill=False, edgecolor='black', linewidth=1, linestyle='--')
        axs_focus[0].add_patch(circle_w_0)
    plt.tight_layout()

    if unwrap is True:
        wavefront = unwrap_phase(np.angle(E_z), wrap_around=(True, True))
    else:
        wavefront = np.angle(E_z)
    axs_focus[1].imshow(wavefront, extent=extent_focus, cmap='bwr')
    if fix_axis is True:
        axs_focus[1].set_xlim((-200e-6, 200e-6))
        axs_focus[1].set_ylim((-200e-6, 200e-6))
    axs_focus[1].set_ylabel('y [m]')
    axs_focus[1].set_xlabel('x [m]')
    axs_focus[1].set_title('$\phi(x,y)$')
    if circle is True:
        circle_w_0 = patches.Circle((0, 0), w_0, fill=False, edgecolor='black', linewidth=1, linestyle='--')
        axs_focus[1].add_patch(circle_w_0)
    plt.tight_layout()

    axs_focus[2].imshow(complex2rgb(E_z), extent=extent_focus)
    if fix_axis is True:
        axs_focus[2].set_xlim((-200e-6, 200e-6))
        axs_focus[2].set_ylim((-200e-6, 200e-6))
    axs_focus[2].set_ylabel('y [m]')
    axs_focus[2].set_xlabel('x [m]')
    axs_focus[2].set_title('$E(x,y)e^{i\phi} $')
    if circle is True:
        circle_w_0 = patches.Circle((0, 0), w_0, fill=False, edgecolor='black', linewidth=1, linestyle='--')
        axs_focus[2].add_patch(circle_w_0)
    plt.tight_layout()
    plt.show()


def plot_slm_plane(E_slm, extent_slm=None, unwrap=True):
    """
    Plot intensity and phase distributions on the SLM.

    Parameters:
    -----------
    E_slm: numpy.ndarray
        Field distribution on the SLM.
    extent_slm : tuple
        Extent of the SLM in (x_min, x_max, y_min, y_max).

    Returns:
    --------
    None
    """

    fig_slm, axs_slm = plt.subplots(1, 2, figsize=(10, 6))

    w_L = 4e-3

    axs_slm[0].imshow(abs(E_slm) ** 2, extent=extent_slm, cmap='turbo')
    axs_slm[0].set_ylim((-4.8e-3, 4.8e-3))
    axs_slm[0].set_xlim((-7.68e-3, 7.68e-3))
    axs_slm[0].set_ylabel('y [m]')
    axs_slm[0].set_xlabel('x [m]')
    axs_slm[0].set_title('Intensity on SLM')
    #circle_w_L = patches.Circle((0, 0), w_L, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    #axs_slm[0].add_patch(circle_w_L)
    plt.tight_layout()

    if unwrap is True:
        wavefront = unwrap_phase(np.angle(E_slm), wrap_around=(True, True))
    else:
        wavefront = np.angle(E_slm)
    axs_slm[1].imshow(wavefront, extent=extent_slm, cmap='bwr')
    axs_slm[1].set_ylim((-4.8e-3, 4.8e-3))
    axs_slm[1].set_xlim((-7.68e-3, 7.68e-3))
    axs_slm[1].set_ylabel('y [m]')
    axs_slm[1].set_xlabel('x [m]')
    axs_slm[1].set_title('Phase on SLM')
    #circle_w_L = patches.Circle((0, 0), w_L, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    #axs_slm[1].add_patch(circle_w_L)
    plt.tight_layout()
    plt.show()


def plot_transverse_planes(dz, propagated_E, som_x, som_y, N_x, N_y, x_focus, y_focus,z_R):
    """
    Plot tranverses intensities in yz and yx planes.
    """
    fig_transverse, axs_transverse = plt.subplots(2, 1, figsize=(4, 8))
    axs_transverse[0].imshow(np.abs(propagated_E[:, int(N_x / 2), :].T)**2, aspect="auto", cmap='turbo',
                             extent=(x_focus[0]*1e6, x_focus[-1]*1e6,dz[0], dz[1]))
    #axs_transverse[0].plot(som_y,z/z_R, color='red', linestyle='--')
    #axs_transverse[0].plot(-som_y, z / z_R, color='red', linestyle='--')
    axs_transverse[0].set_xlabel('y [µm]')
    axs_transverse[0].set_ylabel('z [zR]')
    axs_transverse[0].set_title('yz plane')
    axs_transverse[0].set_xlim((-100,100))
    plt.tight_layout()

    axs_transverse[1].imshow(np.abs(propagated_E[int(N_y / 2), :, :].T)**2, aspect="auto", cmap='turbo',
                             extent=(x_focus[0]*1e6, x_focus[-1]*1e6,dz[0], dz[1]))
    #axs_transverse[1].plot(som_x,z/z_R, color='red', linestyle='--')
    #axs_transverse[1].plot(-som_x, z / z_R, color='red', linestyle='--')
    axs_transverse[1].set_xlabel('x [µm]')
    axs_transverse[1].set_ylabel('z [zR]')
    axs_transverse[1].set_title('xz plane')
    axs_transverse[1].set_xlim((-100, 100))
    plt.tight_layout()

    #axs_transverse[2].plot(z, som_x, color='magenta', linestyle='--', label=f'$w_x(z)$')
    #axs_transverse[2].plot(z, -som_y, color='red', linestyle='--', label=f'$w_y(z)$')
    #axs_transverse[2].set_ylim((x_focus[0], x_focus[-1]))
    #axs_transverse[2].set_xlim((z[0], z[-1]))
    #axs_transverse[2].set_ylabel('x/y [m]')
    #axs_transverse[2].set_xlabel('z [m]')
    #axs_transverse[2].set_title('$w(z)$')
    #plt.tight_layout()

    #plt.legend()

    plt.show()


def plot_all_cross_sections(E_z, z, N_x, N_y):
    middle_row_index = N_y // 2
    middle_col_index = N_x // 2

    fig_cross, axs_cross = plt.subplots(1, 2, figsize=(10, 6))

    for i, e in enumerate(z):
        cross_y = abs(E_z[:, middle_row_index, i]) ** 2
        cross_x = abs(E_z[middle_col_index, :, i]) ** 2
        axs_cross[0].plot(cross_x, label=f'z:{e:.5f}')
        axs_cross[1].plot(cross_y)

    axs_cross[0].set_ylabel('Intensity')
    axs_cross[0].set_xlabel('x')
    axs_cross[0].legend()
    axs_cross[1].set_xlabel('y')

    plt.tight_layout()
    plt.show()


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert a 3D hsv np.ndarray to rgb.
    https://stackoverflow.com/questions/27041559/rgb-to-hsv-python-change-hue-continuously
    h,s should be a numpy arrays with values between 0.0 and 1.0
    v should be a numpy array with values between 0.0 and 255.0
    :param hsv: np.ndarray of shape (x,y,3)
    :return: hsv2rgb returns an array of uints between 0 and 255.
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
    """
    Preparation function for a complex plot, converting a 2D complex
    array into an rgb array
    :param u: a 2D complex array
    :return: an rgb array for complex plot
    """
    # hue (normalize angle)
    # if u is on the GPU, remove it as we can toss it now.
    h = np.angle(u)
    h = (h + np.pi) / (2 * np.pi)
    # saturation (ones)
    s = np.ones_like(h)
    # value (normalize brightness to 8-bit)
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
