"""
Vegetation index
Author:     Nicolas EKICIER
Release:    V1.11   03/2021
"""


def ndwi(swir, green):
    """
    Compute Vegetation Index from SWIR and GREEN bands

        NDWI = \\frac { GREEN - SWIR } { GREEN + SWIR }

    :param swir:    Swir band
    :param green:   Green band
    :return:        NDWI
    """
    return (green-swir) / (green+swir)


def ndvi(nir, red):
    """
    Compute Vegetation Index from RED and NIR bands

        NDVI = \\frac { NIR - RED } { NIR + RED }

    :param nir: Near-Infrared band
    :param red: Red band
    :return:    NDVI
    """
    return (nir-red) / (nir+red)


def evi(nir, red, blue=None, L=1.0, C1=6.0, C2=7.5, G=2.5):
    """
    Compute Enhanced Vegetation Index

        EVI = 2.5\\frac { NIR - RED } { NIR + 2.4 \\times RED + 1 }

    3-band EVI from RED, NIR  and BLUE bands

        EVI = G\\frac { NIR - RED } { NIR + C_1 \\times RED - C_2 \\times BLUE + L }

    :param nir:     Near-Infrared band
    :param red:     Red band
    :param blue:    Blue band
    :param L:       canopy background adjustment
    :param C1:      coefficients of the aerosol resistance term for red
    :param C2:      coefficients of the aerosol resistance term for blue
    :param G:       gain factor
    :return:        EVI
    """
    if blue is None:
        return 2.5 * (nir-red) / (nir + 2.4*red + 1.0)
    else:
        return G * (nir-red) / (nir + C1*red - C2*blue + L)


def wrdvi(nir, red, alpha=0.1):
    """
    Compute Wide Dynamic Range Vegetation Index from red and NIR bands

        WRDVI = \\frac { \\alpha NIR - RED } {\\alpha  NIR + RED }

    :param nir:     Near-Infrared band
    :param red:     Red band
    :param alpha:   Weighting coefficient, usually in [0.1-0.2]
    :return:        WRDVI
    """
    return (alpha*nir - red) / (alpha*nir + red)


def cvi(nir, red, green):
    """
    Compute Chlorophyl Vegetation Index from red and NIR bands

        CVI = \\frac { NIR }{ GREEN } - (RED + GREEN)

    :param nir:     Near-Infrared band
    :param red:     Red band
    :param green:   Green band
    :return:        CVI
    """
    return (nir/green) - (red + green)


def savi(nir, red, L=0.5):
    """
    Compute Soil-adjusted Vegetation Index

        SAVI = (1+L)\\frac { NIR - RED } {NIR + RED + L}

    :param nir:     Near-Infrared band
    :param red:     Red band
    :param L:       soil brightness correction factor [0-1] (L=0 -> NDVI)
    :return:        SAVI
    """
    return (1 + L) * (nir - red) / (nir + red + L)
