import numpy as np
from scipy.sparse import eye

"""
Signal processing utilities
Author:     Nicolas EKICIER
Release:    V01    03/2019
"""

def whittf(y, beta, order):
    """
    Whittaker smoothing
    :param y:       vector data
    :param beta:    smoothing parameter > 0
    :param order:   order of smoothing
    :return:        smoothed data
    """
    m = len(y)
    p = np.diff(eye(m).toarray(), n=order)
    inter = np.dot(p.transpose(), p)
    yf = np.linalg.solve((eye(m).toarray() + np.dot(beta, inter)), y)
    return yf