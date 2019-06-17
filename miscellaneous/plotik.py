import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


"""
Plotting utilities
Author:     Nicolas EKICIER
Release:    V1.O    06/2019
        - Initialization
"""

def imadjust(input):
    '''
    Adjust intensity image
    :param input:   image (numpy array)
    :return:        rescale image
    '''
    p3, p97 = np.percentile(input, (3, 97))
    if p3 == 0:
        ip3 = 4
        while p3 == 0:
            p3 = np.percentile(input, ip3)
            ip3 = ip3 + 1
    img_rescale = exposure.rescale_intensity(input, in_range=(p3, p97))
    return img_rescale