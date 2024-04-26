import numpy as np


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
    _, _, w_L, w_0 = beam_param
    Y_slm, X_slm = np.meshgrid(x_slm / w_L, y_slm / w_L)
    R = np.sqrt(X_slm ** 2 + Y_slm ** 2)
    circle_radius = factor
    mask = R ** 2 <= circle_radius ** 2
    E_slm_cut = np.zeros_like(E_slm)
    E_slm_cut[mask] = E_slm[mask]
    return E_slm_cut

def zero_padding_to_second(original_array, second_array):
    original_size = original_array.shape
    second_size = second_array.shape
    size_diff = np.subtract(second_size, original_size)
    pad_width = tuple((diff // 2, diff // 2) for diff in size_diff)
    padded_array = np.pad(original_array, pad_width, mode='constant', constant_values=0)
    return padded_array