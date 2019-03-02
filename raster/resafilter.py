import numpy as np
import logging

"""
Resampling & Filtering utilities
Author:     Nicolas EKICIER
Release:    V1    03/2019
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
    # checks
    if input.shape != sharp_param.shape:
        logging.error("shape mismatch between array to sharpen and sharpening parameters")
        return None

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