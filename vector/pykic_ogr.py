import os, glob
import pandas as pd
import numpy as np
import geopandas as gpd
from fiona.crs import from_epsg
import logging

"""
OGR utilities
Author:     Nicolas EKICIER
Release:    V01    02/2019
"""

def ogreproj (layer, oEPSG):
    """
    Reprojection of an OGR layer
    :param layer:   Geopandas Dataframe OR path of shapefile
    :param oEPSG:   EPSG value for destination
    :return:        Reprojected layer
    """
    if type(layer) is str:
        # Read data from Geopandas
        layer = gpd.read_file(layer)

    # Get projection from input and print in console
    iEPSG = layer.crs
    print('Reprojection from {0:s} to epsg:{1:d}'.format(iEPSG['init'], oEPSG))

    # Let's take a copy of our layer
    data_proj = layer.copy()

    # Reproject the geometries by replacing the values with projected ones
    data_proj = data_proj.to_crs(epsg=oEPSG)

    # Determine the CRS of the GeoDataFrame
    data_proj.crs = from_epsg(oEPSG)
    return data_proj


def sprocessing (layer1, layer2, method):
    """
    Geometric processing between shapefiles
    :param layer1:  Geopandas Dataframe 1
    :param layer2:  Geopandas Dataframe 2
    :param method:  "intersects", "within", "contains"
    :return:        Result layer
    """
    # Check if projections are same
    if layer1.crs['init'] is not layer2.crs['init']:
        logging.error('Warning : CRS are not the same')
        return None

    layerm = gpd.sjoin(layer1, layer2, op=method)
    return layerm