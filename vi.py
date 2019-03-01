"""pykic vegetation indicator module"""


def ndvi(nir, red):
    """
    Compute Non-Dimensional Vegeation Index from red and NIR bands

    .. math::

        NDVI = \\frac { NIR - RED } { NIR + RED }

    :param nir: Near-Infrared band
    :param red: Red band
    :return: NDVI
    :type nir: float or numpy array
    :type red: float or numpy array
    :rtype: float or numpy array
    """
    return (nir-red) / (nir+red)


def evi(nir, red, blue=None, L=1.0, C1=6.0, C2=7.5, G=2.5):
    """
    Compute Enhanced Vegeation Index

    .. math::

        EVI = 2.5\\frac { NIR - RED } { NIR + 2.4 \\times RED + 1 }

    or 3-band EVI from red, NIR  and blue bands

    .. math::

        EVI = G\\frac { NIR - RED } { NIR + C_1 \\times RED - C_2 \\times BLUE + L }

    :param nir: Near-Infrared band
    :param red: Red band
    :param blue: blue band
    :param L: canopy background adjustment
    :param C1: coefficients of the aerosol resistance term for red
    :param C2: coefficients of the aerosol resistance term for blue
    :param G: gain factor
    :type nir: float or numpy array
    :type red: float or numpy array
    :type blue: float or numpy array
    :type L: float
    :type C1: float
    :type C2: float
    :type G: float
    :return: EVI
    :rtype: float or numpy array
    """
    if blue is None:
        return 2.5 * (nir-red) / (nir + 2.4*red + 1.0)
    else:
        return G * (nir-red) / (nir + C1*red - C2*blue + L)


def wrdvi(nir, red, alpha=0.1):
    """
    Compute Wide Dynamic Range Vegeation Index from red and NIR bands

    .. math::

        WRDVI = \\frac { \\alpha NIR - RED } {\\alpha  NIR + RED }

    :param nir: Near-Infrared band
    :param red: Red band
    :param alpha: Weighting coefficient, usually in [0.1-0.2]
    :return: WRDVI
    :type nir: float or numpy array
    :type red: float or numpy array
    :type alpha: float
    :rtype: float or numpy array
    """
    return (alpha*nir - red) / (alpha*nir + red)


def cvi(nir, red, green):
    """
    Compute Chlorophyl Vegetation Index from red and NIR bands

    .. math::

        CVI = \\frac { NIR }{ GREEN } - (RED + GREEN)

    :param nir: Near-Infrared band
    :param red: Red band
    :param green: Green band
    :return: CVI
    :type nir: float or numpy array
    :type red: float or numpy array
    :type green: float or numpy array
    :rtype: float or numpy array
    """
    return (nir/green) - (red + green)


def savi(nir, red, L=0.5):
    """
    Compute Soil-adjusted Veegation Index

    .. math::

        SAVI = (1+L)\\frac { NIR - RED } {NIR + RED + L}

    :param nir: Near-Infrared band
    :param red: Red band
    :param L: soil brightness correction factor [0-1] (L=0 -> NDVI)
    :return: SAVI
    :type nir: float or numpy array
    :type red: float or numpy array
    :type L: float
    :rtype: float or numpy array
    """
    return (1 + L) * (nir - red) / (nir + red + L)
