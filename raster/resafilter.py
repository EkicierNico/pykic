import numpy as np
import PIL.Image as pilim

"""
Resampling & Filtering utilities
Author:     Nicolas EKICIER
Release:    V1.21   10/2020
"""

def pan_sharpen(input, sharp_param):
    """
    Pan sharpening algorithm
    :param input:               band array to be sharpened (numpy array)
    :param sharp_param:         sharpen parameters (panchromatic band, numpy array)
    :return:                    panchromatic band
    """
    # Checks
    if input.shape != sharp_param.shape:
        input = resample_2d(input, sharp_param.shape, method='nearest')

    out_array = np.zeros(input.shape)
    for i in range(0, input.shape[0]-1, 2):
        for j in range(0, input.shape[1]-1, 2):
            end_i = i + 2
            end_j = j + 2
            mask = np.logical_and(input[i:end_i, j:end_j] > 0, sharp_param[i:end_i, j:end_j] > 0)
            if np.sum(mask) == 0:
                continue

            sumlm = np.sum(input[i:end_i, j:end_j][mask])
            sump = np.sum(sharp_param[i:end_i, j:end_j][mask])

            out_array[i:end_i, j:end_j][mask] = sharp_param[i:end_i, j:end_j][mask] * float(sumlm) / sump
    return out_array


def resample_2d(input, dim, method='lanczos'):
    """
    2D resampling
    :param input:   image (numpy array)
    :param dim:     output dimensions (tuple (x, y))
    :param method:  resampling method
                    {'nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos' = default}
    :return:        resampled image
    """
    mode = pilim.LANCZOS
    if method == 'nearest':
        mode = pilim.NEAREST
    elif method == 'box':
        mode = pilim.BOX
    elif method == 'bilinear':
        mode = pilim.BILINEAR
    elif method == 'hamming':
        mode = pilim.HAMMING
    elif method == 'bicubic':
        mode = pilim.BICUBIC

    imr = np.array(pilim.fromarray(input).resize(dim, resample=mode))
    return imr