import numpy as np
from scipy import ndimage


def ifft2c(array):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array), norm='ortho'))


def fft2c(array):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array), norm='ortho'))


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


def zero_padding_to_second(original_array, second_array):
    original_size = original_array.shape
    second_size = second_array.shape
    size_diff = np.subtract(second_size, original_size)
    pad_width = tuple((diff // 2, diff // 2) for diff in size_diff)
    padded_array = np.pad(original_array, pad_width, mode='constant', constant_values=0)
    return padded_array


def set_outside_circle_to_zero(image, radius):
    max_pixel = np.max(image)
    max_coord = np.where(image == max_pixel)
    max_coord_tuple = (max_coord[0][0], max_coord[1][0])

    radius = int(radius)
    center_col = int(max_coord_tuple[1])
    center_row = int(max_coord_tuple[0])

    x_start = max(0, center_col - radius)
    x_end = min(image.shape[1], center_col + radius)
    y_start = max(0, center_row - radius)
    y_end = min(image.shape[0], center_row + radius)

    new_image = image[y_start:y_end, x_start:x_end]
    nx, ny = new_image.shape
    x, y = np.meshgrid(np.arange(ny), np.arange(nx))
    distance_from_center = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    outside_circle = distance_from_center > radius
    new_image[outside_circle] = 0
    cut_x = np.linspace(-radius, radius, 2 * radius)
    cut_y = cut_x

    return new_image, cut_x, cut_y


def process_image(image, dx):
    image[image <= np.max(image) * 0.000] = 0  # Should contain more than 97.5% of the power
    new_image, _, _ = set_outside_circle_to_zero(image, 32)
    new_image = new_image / np.sum(new_image)

    new_som = get_som(new_image, dx)
    som_x, som_y = new_som

    return new_image, som_x, som_y


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

def crop_array(C, extent_slm, xlim, ylim):
    x_proportion = (xlim[1] - xlim[0]) / (extent_slm[1] - extent_slm[0])
    y_proportion = (ylim[1] - ylim[0]) / (extent_slm[3] - extent_slm[2])
    x_elements = int(x_proportion * C.shape[1])
    y_elements = int(y_proportion * C.shape[0])
    x_start = (C.shape[1] - x_elements) // 2
    y_start = (C.shape[0] - y_elements) // 2
    cropped_C = C[y_start:y_start + y_elements, x_start:x_start + x_elements]
    return cropped_C