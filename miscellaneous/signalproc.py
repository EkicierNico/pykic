import numpy as np
from scipy.interpolate import interp1d

"""
Signal processing utilities
Author:     Nicolas EKICIER
Release:    V1.1    06/2019
                - Add fillnan function
            V1.0    03/2019
"""

def whittf(y, weight=None, beta=1000, order=3):
    """
    Whittaker smoothing
    :param y:       vector data (signal)
    :param weight:  weights of each sample (one by default)
    :param beta:    penalization parameter (> 0)
    :param order:   derivation order
    :return:        smooth signal
    """
    m = len(y)
    speye = np.eye(m)
    p = np.diff(speye, n=order)

    if weight is None:
        diag = speye
    else:
        diag = np.diag(weight)

    pp = np.transpose(p)
    yf = np.linalg.solve(diag + beta*np.dot(p, pp), np.dot(diag, y))
    return yf


def fillnan(y, method='linear'):
    """
    Interpolation of vector data with nan values
    Extrema are extrapolated
    :param y:       vector data
    :param method:  interpolation method
                        - 'linear'  default
                        - 'nearest'
                        - 'zero', 'slinear', 'quadratic', 'cubic' = spline interpolation of zeroth, first, second or third order
    :return:        interpolated signal
    """
    y = np.ravel(y)
    x = np.arange(0, len(y))

    igood = np.where(np.isfinite(y))
    func = interp1d(np.ravel(igood),
                    np.ravel(y[igood]), # fndvi.iloc[np.ravel(igood), fndvi.columns.get_loc(f)]
                    fill_value='extrapolate',
                    kind=method)
    return func(x)


def outliers(input):
    """
    Extract index of outliers in input
    :param input:   vector data (signal)
    :return:        index
    """
    Quart = np.percentile(input, [25, 75])  # 1er et 3ème quartile
    IQuart = Quart[1] - Quart[0]    # interquartile
    wsup = Quart[1] + 1.5 * IQuart  # whisker sup
    winf = Quart[0] - 1.5 * IQuart  # whisker inf

    idx = np.flatnonzero((input < winf) | (input > wsup))
    return idx