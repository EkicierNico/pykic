import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import linregress
from joblib import Parallel, delayed
from tqdm import tqdm

"""
Signal processing utilities
Author:     Nicolas EKICIER
Release:    V1.5    05/2020
"""

def smooth_compute(input, dim, njob=-1, cst=100, ord=3):
    """
    Compute whittaker smoothing in jobs
    :param input:   array of vi signal
    :param dim:     dimension of vector signal (0 = row / 1 = column)
    :param njob:    number of jobs (refer to Joblib doc, default = -1)
    :param cst:     penalization (Whittaker parameter)
    :param ord:     derivation order (Whittaker parameter)
    :return:
    """
    def lambdf(x):
        return whittf(fillnan_and_resample(x), beta=cst, order=ord)
    if dim == 0:
        tw = Parallel(n_jobs=njob)(delayed(lambdf)(input[i, :]) for i in tqdm(range(0, input.shape[0]), desc='Whittaker Smoothing'))
    elif dim == 1:
        tw = Parallel(n_jobs=njob)(delayed(lambdf)(input[:, i]) for i in tqdm(range(0, input.shape[1]), desc='Whittaker Smoothing'))
    return np.array(tw)


def whittf(y, weight=None, beta=100, order=3):
    """
    Weighted Whittaker smoothing
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


def fillnan_and_resample(y, x=None, method='linear'):
    """
    Interpolation of vector data with nan values
    Extrema are extrapolated
    :param y:       vector data
    :param x:       vector of x positions to resampling (Default = None)
    :param method:  interpolation method
                        - 'linear'  default
                        - 'nearest'
                        - 'zero', 'slinear', 'quadratic', 'cubic' = spline interpolation of zeroth, first, second or third order
    :return:        interpolated signal
    """
    y = np.ravel(y)
    if x is None:
        x = np.arange(0, len(y))

    igood = np.where(np.isfinite(y))
    func = interp1d(np.ravel(igood),
                    np.ravel(y[igood]), # fndvi.iloc[np.ravel(igood), fndvi.columns.get_loc(f)]
                    fill_value='extrapolate',
                    kind=method)
    return func(x)


def regress(x, y, deg=1):
    """
    Compute regression (linear or polynomial) from 2 datasets
    :param x:   x values (vector)
    :param y:   y values (vector)
    :param deg: degree of regression (default = 1 = linear)
    :return:    sorted values from x [x, y, predict] (out)
                regression coefficients (coeffs)
                polynomial class (clpoly)
                metrics (r2, mae, rmse)
    """
    # Regression
    coeffs = np.polyfit(x, y, deg)
    predict = np.polyval(coeffs, x)
    clpoly = np.poly1d(coeffs)

    # Metrics
    diff = y - predict
    mae = np.mean(abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    r2 = 1-(sum(diff**2)/sum((y-np.mean(y))**2))

    # Sort
    ind = np.argsort(x)
    out = np.vstack((x[ind], y[ind], predict[ind])).transpose()
    return out, coeffs, clpoly, (r2, mae, rmse)


def outliers(input):
    """
    Extract index and values of outliers in input
    Works with NaN values
    :param input:   vector data (signal)
    :return:        index, (whisker inf, sup)
    """
    Quart = np.nanpercentile(input, [25, 75])  # 1er et 3ème quartile
    IQuart = Quart[1] - Quart[0]    # interquartile
    wsup = Quart[1] + 1.5 * IQuart  # whisker sup
    winf = Quart[0] - 1.5 * IQuart  # whisker inf

    idx = np.flatnonzero((input < winf) | (input > wsup))
    return idx, (winf, wsup)


def phen_met(y, plot=False):
    """
    Extract phenological metrics from one crop season
    From "TIMESAT — a program for analyzing time-series of satellite sensor data", Jönsson al., 2004
    :param y:       ndvi profil
    :param plot:    plot result (default = False)
    :return:        pandas Dataframe with metrics
    """
    # Thresholds
    pct_sg = 0.2
    pct_eg = 0.8
    pct_sd = 0.8
    pct_ed = 0.2

    # Maximum
    tmp = find_peaks(y)
    iMaxVal = np.argmax(y[tmp[0]])
    iMaxVal = tmp[0][iMaxVal]
    MaxVal = y[iMaxVal]

    # Min before max
    iLeftMinVal = np.argmin(y[0:iMaxVal])
    LeftMinVal = y[iLeftMinVal]

    # Min after max
    iRightMinVal = np.argmin(y[iMaxVal:]) + iMaxVal
    RightMinVal = y[iRightMinVal]

    # Base
    BaseVal = np.mean((LeftMinVal, RightMinVal))

    # Amplitude
    SeasonAmp = MaxVal - BaseVal

    # Start of growth
    sgb = LeftMinVal + pct_sg * SeasonAmp
    iStartGrowth = np.flatnonzero(y[iLeftMinVal:iMaxVal] >= sgb)
    iStartGrowth = iStartGrowth[0] + iLeftMinVal
    StartGrowth = y[iStartGrowth]

    # End of growth
    egb = LeftMinVal + pct_eg * (MaxVal - LeftMinVal)
    iEndGrowth = np.flatnonzero(y[iStartGrowth:iMaxVal] >= egb)
    iEndGrowth = iEndGrowth[0] + iStartGrowth
    EndGrowth = y[iEndGrowth]

    # Start of decrease
    sdb = RightMinVal + pct_sd * (MaxVal - RightMinVal)
    iStartDecrease = np.flatnonzero(y[iMaxVal:iRightMinVal] >= sdb)
    iStartDecrease = iStartDecrease[-1] + iMaxVal
    StartDecrease = y[iStartDecrease]

    # End of decrease
    edb = RightMinVal + pct_ed * SeasonAmp
    if edb > StartDecrease:
        iEndDecrease = np.nan
        EndDecrease = np.nan
    else:
        iEndDecrease = np.flatnonzero(y[iStartDecrease:iRightMinVal] >= edb)
        iEndDecrease = iEndDecrease[-1] + iStartDecrease
        EndDecrease = y[iEndDecrease]

    # Length of season
    SeasonL = iEndDecrease - iStartGrowth

    # Output
    out = pd.DataFrame({'istartgrowth': iStartGrowth,
                        'startgrowth': StartGrowth,
                        'iendgrowth': iEndGrowth,
                        'endgrowth': EndGrowth,
                        'imax': iMaxVal,
                        'max': MaxVal,
                        'istartdecrease': iStartDecrease,
                        'startdecrease': StartDecrease,
                        'ienddecrease': iEndDecrease,
                        'enddecrease': EndDecrease,
                        'slength': SeasonL},
                       index=[0])

    if plot == True:
        plt.figure()
        plt.plot(y, 'k-', label='ndvi')
        plt.scatter(out['istartgrowth'], out['startgrowth'], label='StartOfGrowth')
        plt.scatter(out['iendgrowth'], out['endgrowth'], label='EndOfGrowth')
        plt.scatter(out['imax'], out['max'], label='Maximum')
        plt.scatter(out['istartdecrease'], out['startdecrease'], label='StartOfDecrease')
        plt.scatter(out['ienddecrease'], out['enddecrease'], label='EndOfDecrease')
        plt.title('Phenological Metrics')
        plt.ylabel('Vegetation Index')
        plt.legend()
        plt.grid()
    return out