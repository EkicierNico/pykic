import numpy as np
# import scipy.misc as spm
import PIL.Image as pilim
import logging

"""
Resampling & Filtering utilities
Author:     Nicolas EKICIER
Release:    V1.1    06/2019
                - Add resizing method
            V1.0    03/2019
"""

def pan_sharpen(input, sharp_param):
    """
    Pan sharpening algorithm
    :param input:               band array to be sharpened
    :param sharp_param:         sharpen parameters (panchromatic band)
    :type input:                numpy array
    :type sharp_param:          numpy array
    :return:                    panchromatic band
    :rtype:                     numpy array
    """
    # Checks
    if input.shape != sharp_param.shape:
        input = np.array(pilim.fromarray(input).resize(sharp_param.shape,
                                                       resample=pilim.NEAREST))

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


def lanczosf(input):
    return arl