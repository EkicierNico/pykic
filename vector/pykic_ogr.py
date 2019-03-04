import os, glob
import pandas as pd
import numpy as np
import geopandas as gpd
from fiona.crs import from_epsg
import logging

"""
OGR utilities
Author:     Nicolas EKICIER
Release:    V1.1    03/2019
    - Add shpbuf function
    
            V1      02/2019
    - Initialization
"""

def ogreproj (layer, oEPSG):
    """
    Reprojection of an OGR layer
    :param layer:   Geopandas Dataframe OR path of shapefile
    :param oEPSG:   EPSG value for destination (int)
    :return:        Reprojected layer
    """
    if type(layer) is str:
        layer = gpd.read_file(layer)

    iEPSG = layer.crs # Get projection from input and print in console
    print('Reprojection from {0:s} to epsg:{1:d}'.format(iEPSG['init'], oEPSG))

    data_proj = layer.copy()
    data_proj = data_proj.to_crs(epsg=oEPSG) # Reproject the geometries by replacing the values with projected ones
    data_proj.crs = from_epsg(oEPSG) # Determine the CRS of the GeoDataFrame
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


def shpbuf(distance, input, output=None, fmtout=None):
    """
    Create a buffer from shapefile or Geopandas DataFrame
    :param distance:    Distance of buffer (in the same crs unit)
    :param input:       Shapefile path or GeoDataFrame
    :param output:      Write the output (OPTIONAL)
    :param fmtout:      Output format (OPTIONAL) --> ('CSV', 'ESRI Shapefile'=default, 'GeoJSON', 'GML', 'GPX', 'MapInfo File')
                        For complete list : import fiona; fiona.supported_drivers
    :return:            Buffered layer (GeoDataFrame)
    """
    if type(input) is str:
        layer = gpd.read_file(input)
    else:   # GeoDataFrame
        layer = input.copy()

    layerb = gpd.GeoDataFrame(layer.buffer(distance))
    layerb = layerb.rename(columns={0: 'geometry'}).set_geometry('geometry')
    layerb.crs = layer.crs

    if output is not None:
        if type(output) is str and '.shp' in output:
            layerb.to_file(output)
        elif type(output) is str and fmtout is not None:
            layerb.to_file(output, driver=fmtout)
        else:
            logging.error('Warning : output is not a string or ".shp" extension is missing')
    return layerb