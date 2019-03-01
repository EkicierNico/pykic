import numpy as np
import logging

def pan_sharpen(array_to_sharpen, sharp_param, filter_opt='adjacent'):
    """
    Pan sharpening algorithm https://en.wikipedia.org/wiki/Pansharpened_image

    :param array_to_sharpen: band array to be sharpened
    :param sharp_param: sharpen parameters (typically band 8 of LS8)
    :param filter_opt: sharpening option
    :type array_to_sharpen: numpy array
    :type sharp_param: numpy array
    :type filter_opt: string ['adjacent' or '']
    :return: panchromatic band
    :rtype: numpy array
    """
    # checks
    if array_to_sharpen.shape != sharp_param.shape:
        logging.error("shape mismatch between array to sharpen and sharpening parameters")
        return None

    out_array = np.zeros(array_to_sharpen.shape)
    for i in range(0, array_to_sharpen.shape[0]-1, 2):
        for j in range(0, array_to_sharpen.shape[1]-1, 2):
            # if j % 1000 == 0:
            #     print i, j
            end_i = i + 2
            end_j = j + 2
            mask = np.logical_and(array_to_sharpen[i:end_i, j:end_j] > 0,
                                  sharp_param[i:end_i, j:end_j] > 0)
            if np.sum(mask) == 0:
                continue

            sumlm = np.sum(array_to_sharpen[i:end_i, j:end_j][mask])
            sump = np.sum(sharp_param[i:end_i, j:end_j][mask])

            out_array[i:end_i, j:end_j][mask] = sharp_param[i:end_i, j:end_j][mask] * float(sumlm) / sump
    return out_array


    # if filter_opt == 'adjacent':
    # import scipy.signal as sig
    #     conv_matrix = np.ones((2, 2))
    #     sum_lm = sig.convolve2d(array_to_sharpen, conv_matrix)[1:, 1:]
    #     sum_p = sig.convolve2d(sharp_param, conv_matrix)[1:, 1:]
    #     return sharp_param * sum_lm / sum_p
