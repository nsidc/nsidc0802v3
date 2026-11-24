"""Implement gaussian smoothing

Note: routines initially taken from nsidc0739...

"""
import numpy as np
from math import sqrt, exp, pi
from scipy import interpolate
from scipy.signal import convolve2d

root_two_pi = sqrt(2.0 * pi)
np.set_printoptions(formatter={'float_kind': '{:5f}'.format})


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0,
        simplex_tolerance: float = 1.0,
):
    """
    Source - https://stackoverflow.com/a/68558547
    Posted by Sam De Meyer, modified by community. See post 'Timeline' for change history
    Retrieved 2026-11-02, License - CC BY-SA 4.0

    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y),
        known_v,
        (missing_x, missing_y),
        method=method,
        fill_value=fill_value,
        simplex_tolerance=simplex_tolerance,
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def gauss_2d_val(x, y, x0, y0, sigma):
    """
    returns the value of the gaussian at (x, y)
    when middle at (x0, y0) and root-variance of sigma
    """
    if sigma != 0:
        dist_from_mean = sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))
        exponent = -0.5 * (dist_from_mean / sigma) * (dist_from_mean / sigma)
        gauss_2d = exp(exponent) / (sigma * root_two_pi)
    else:
        if x == x0 and y == y0:
            gauss_2d = 1
        else:
            gauss_2d = 0

    return gauss_2d



def gauss_2d_sum(n, i_cent, j_cent, i_mid, j_mid, sigma):
    """
    Subdivide a gridcell of size 1, 1 into (nxn) sub-cells
    (icent, j_cent) is the center of the grid cell herein divided
    (i_mid, j_mid) is the location of the gaussian peak

    since grid cells are size 1, d=1.0 here
    """
    d = 1.0

    total_sum = 0.0
    d_area = 1 / (n * n)

    for j in range(n):
        jj = j_cent - d * (n - 1) / (2 * n) + j * d / n
        for i in range(n):
            ii = i_cent - d * (n - 1) / (2 * n) + i * d / n
            g_value = gauss_2d_val(ii, jj, i_mid, j_mid, sigma)
            total_sum += g_value * d_area

    return total_sum


def compute_gaussian_kernel(size, sigma):
    """
    Compute the values of a gaussian kernel of shape (size x size)
    with a sigma of sigma

    Procedure is to compute 100x100 values for each grid cell in the kernel
    and sum them up to get the total contribution of each partial to the
    grid cell
    """
    if sigma == 0:
        gauss_kernel = np.zeros((size, size), dtype=np.float64)
        gauss_kernel[size // 2, size // 2] = 1
        print(gauss_kernel)
    else:
        n_subdivisions = 20

        i_center = float(size) / 2.0
        j_center = i_center

        gauss_kernel = np.zeros((size, size), dtype=np.float64)

        for j in range(size):
            jj = j + 0.5
            for i in range(size):
                ii = i + 0.5
                gauss_sum = gauss_2d_sum(
                    n_subdivisions, ii, jj, i_center, j_center, sigma)
                gauss_kernel[j, i] = gauss_sum

        # Total area under an infinite gaussian is sigma * sqrt(2*pi) so...
        kernel_sum = np.sum(gauss_kernel)
        theoretical_sum = sigma * root_two_pi
        kernel_sum_mismatch = 1 - kernel_sum / theoretical_sum
        # print(f'kernel_sum_mismatch: {kernel_sum_mismatch * 100:.3f}%')
        # gauss_kernel = np.divide(gauss_kernel, sigma * root_two_pi)
        gauss_kernel = np.divide(gauss_kernel, np.sum(gauss_kernel))

    return gauss_kernel


def apply_gaussian_kernel(field, kernel, pad_method='edge', missing_val=0, interp_method='linear'):
    """Apply a gaussian kernel to a field
    Method:
        - pad the field using 'edge' padding
        - interpolate missing values (0) with 'linear'
        - return the original field
    """
    n_pad = (kernel.shape[0] - 1) // 2
    padded = np.pad(field, n_pad, mode=pad_method)
    pad_interped = interpolate_missing_pixels(padded, padded==missing_val, interp_method, missing_val)
    is_missing_onepad = pad_interped == missing_val

    num_still_missing = np.sum(np.where(pad_interped == 0, 1, 0))
    if num_still_missing > 0:
        print(f'  number of missing point after interpolation: {num_still_missing}...')

    # pad_convolved = np.round(convolve2d(pad_interped, kernel, mode='same', boundary='symm')).astype(field.dtype)
    pad_convolved = convolve2d(pad_interped, kernel, mode='same', boundary='symm').astype(field.dtype)

    interped = pad_convolved[n_pad:-n_pad, n_pad:-n_pad]
    assert interped.shape == field.shape

    return interped


def apply_gaussian_kernel_subset(tb, kernel, subset_indices):
    """Apply the gaussian kernel to only part of a field"""
    pass


if __name__ == '__main__':
    import sys

    size = int(sys.argv[1])
    sigma = float(sys.argv[2])

    kernel = compute_gaussian_kernel(size, sigma)
    print(f'kernel:\n{kernel}')
    print(f'sum of kernel:  {np.sum(kernel)}')

    print('Now, try using interpolate_missing_pixels()')
    fn = './sample_psn25_19h_304x448.dat'
    tb = np.fromfile(fn, dtype=np.int16).reshape(448, 304)
    tb1 = apply_gaussian_kernel(tb, kernel)

    print('have tb1...')

    print('try applying to a little part...')
    subset_indices = {
        'i0': 50,
        'i1': 100,
        'j0': 50,
        'j1': 100,
    }

    apply_gaussian_kernel_subset(tb, kernel, subset_indices)
    breakpoint()
