import os, glob, fiona, logging
import pandas as pd
import numpy as np
import geopandas as gpd
from fiona.crs import from_epsg
from tqdm import tqdm

import raster.pykic_gdal as rpg

"""
OGR utilities
Author:     Nicolas EKICIER
Release:    V1.4    09/2019
"""

def zonstat(inshp, inimg, attribut='id'):
    """
    Compute zonal statistics (count) on each polygon of shapefile from image
    Output will be in image folder with "_statz.csv" extent
    :param inshp:       path of shapefile
    :param inimg:       path of image
    :param attribut:    attribute to use in shapefile table (default = 'id')
    :return:
    """
    try:
        img, _, _, _ = rpg.gdal2array(inimg)
        uidi = np.unique(img)

        mask = rpg.makemask(inshp, inimg, attribute=attribut)
        uid = np.unique(mask)

        output = pd.DataFrame(index = uid, columns = uidi)
        for idmask in tqdm(uid, desc='Zonal Statistics', total=len(uid)):
            index = np.flatnonzero(mask == idmask)
            values = img[np.unravel_index(index, img.shape)]

            uidval, uidvalc = np.unique(values, return_counts=True)
            for idval, idvalc in zip(uidval, uidvalc):
                output.at[idmask, idval] = idvalc

        output = output.fillna(value=0)
        output.to_csv(inimg.replace('.tif', '_statz.csv'))
        return None

    except:
        logging.error(' ERROR in compute zonal statistics')
        return None

def checkproj(layer0, layer1):
    """
    Check if projections are the same
    :param layer0:  path of shapefile 1
    :param layer1:  path of shapefile 2
    :return:        booleen
    """
    with fiona.open(layer0, 'r') as src0:
        proj0 = src0.crs['init']
    with fiona.open(layer1, 'r') as src1:
        proj1 = src1.crs['init']

    if proj0 is not proj1:
        return False
    else:
        return True


def ogreproj(layer, oEPSG):
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


def sprocessing(layer1, layer2, method):
    """
    Geometric processing between shapefiles
    :param layer1:  Geopandas Dataframe 1 OR path of shapefile
    :param layer2:  Geopandas Dataframe 2 OR path of shapefile
    :param method:  "intersects", "within", "contains"
    :return:        Result layer
    """
    if type(layer1) is str:
        lay1 = gpd.read_file(layer1)
    if type(layer2) is str:
        lay2 = gpd.read_file(layer2)

    # Check if projections are same
    if not checkproj(layer1, layer2):
        logging.error('Warning : CRS are not the same')
        return None

    layerm = gpd.sjoin(lay1, lay2, op=method)
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


def distm(lon, lat, units='km'):
    """
    Geodesic distance between dots (degrees units)
    Method : Vincenty (1975)
    :param lon:     numpy array with at least 2 values (longitude)
    :param lat:     numpy array with at least 2 values (latitude)
    :param units:   output units : {'km'=default, 'm'}
    :return:        distance
    """
    radius = 6371009 # meters

    xa = np.deg2rad(lon[0:-1])
    xb = np.deg2rad(lon[1:])
    ya = np.deg2rad(lat[0:-1])
    yb = np.deg2rad(lat[1:])

    s1 = np.sin(ya)*np.sin(yb) + np.cos(ya)*np.cos(yb)*np.cos(xb - xa)
    s2 = np.arccos(s1)

    if units.lower() == 'km':
        d = s2 * radius / 1000
    elif units.lower() == 'm':
        d = s2 * radius
    else:
        logging.error('Warning : output format is not recognized, use default')
        d = s2 * radius / 1000
    return d