import numpy
import numpy as np


def create_phase_jump(x_slm: numpy.ndarray, y_slm: numpy.ndarray, desired_radius: float) -> numpy.ndarray:
    """
    Create a phase jump pattern with optional smoothing.

    Parameters:
    -----------
    x_slm : numpy.ndarray
        Array of x-coordinates on the SLM.
    y_slm : numpy.ndarray
        Array of y-coordinates on the SLM.
    desired_radius : float
        Desired radius within which the phase jump is applied.

    Returns:
    --------
    numpy.ndarray
        2D phase jump pattern.
    """
    X, Y = np.meshgrid(x_slm, y_slm)
    R = np.sqrt(X ** 2 + Y ** 2)
    indices_inner = np.where(R <= desired_radius)

    phase_jump = np.zeros_like(X)
    phase_jump[indices_inner] = np.pi

    return phase_jump


def create_phase_vortex(x_slm: numpy.ndarray, y_slm: numpy.ndarray, w_L, vortex_order: float,
                        R_threshold=2) -> numpy.ndarray:
    """
    Generate a 2D phase vortex pattern.

    Parameters:
    -----------
    x_slm : numpy.ndarray
        Array of x-coordinates on the SLM.
    y_slm : numpy.ndarray
        Array of y-coordinates on the SLM.
    vortex_order : int
        Order of the phase vortex.

    Returns:
    --------
    numpy.ndarray
        2D phase phase vortex pattern.
    """

    x_slm_normalized = x_slm / w_L
    y_slm_normalized = y_slm / w_L
    X, Y = np.meshgrid(x_slm_normalized, y_slm_normalized)
    theta = np.arctan2(Y, X)
    vortex_phase = theta * vortex_order

    return vortex_phase


def create_phase_zernike(x_slm: numpy.ndarray, y_slm: numpy.ndarray, w_L, coeffs_zernike: list,
                         R_threshold=2) -> numpy.ndarray:
    """
    Generate a phase pattern based on Zernike coefficients.

    Parameters:
    -----------
    x_slm : numpy.ndarray
        Array of x-coordinates on the SLM.
    y_slm : numpy.ndarray
        Array of y-coordinates on the SLM.
    coeffs_zernike : list
        List of Zernike coefficients.

    Returns:
    --------
    numpy.ndarray
        2D phase pattern based on Zernike coefficients.
    """
    # now im normalizing everything to w_L to have a "unit circle"
    x_slm_normalized = x_slm / w_L
    y_slm_normalized = y_slm / w_L
    X, Y = np.meshgrid(x_slm_normalized, y_slm_normalized)
    R = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    indices = np.where(R <= R_threshold)

    o_astigmatism = np.zeros_like(X)
    defocus = np.zeros_like(X)
    v_astigmatism = np.zeros_like(X)
    v_trefoil = np.zeros_like(X)
    y_coma = np.zeros_like(X)
    x_coma = np.zeros_like(X)
    o_trefoil = np.zeros_like(X)

    R_squared = R ** 2
    R_cubed = R ** 3

    o_astigmatism[indices] = coeffs_zernike[0] * np.sqrt(6) * R_squared[indices] * np.sin(2 * theta[indices])
    defocus[indices] = coeffs_zernike[1] * np.sqrt(3) * (2 * R_squared[indices] - 1)
    v_astigmatism[indices] = coeffs_zernike[2] * np.sqrt(6) * R_squared[indices] * np.cos(2 * theta[indices])
    v_trefoil[indices] = coeffs_zernike[3] * np.sqrt(8) * R_cubed[indices] * np.sin(3 * theta[indices])
    y_coma[indices] = coeffs_zernike[4] * np.sqrt(8) * (3 * R_cubed[indices] - 2 * R_squared[indices]) * np.sin(
        theta[indices])
    x_coma[indices] = coeffs_zernike[5] * np.sqrt(8) * (3 * R_cubed[indices] - 2 * R_squared[indices]) * np.cos(
        theta[indices])
    o_trefoil[indices] = coeffs_zernike[6] * np.sqrt(8) * R_cubed[indices] * np.cos(3 * theta[indices])

    return o_astigmatism + defocus + v_astigmatism + v_trefoil + y_coma + x_coma + o_trefoil
