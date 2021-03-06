import os, glob, fiona, logging
import pandas as pd
import numpy as np
import geopandas as gpd
from fiona.crs import from_epsg
from tqdm import tqdm
from osgeo import ogr
from joblib import Parallel, delayed

import raster.pykic_gdal as rpg

"""
OGR utilities
Author:     Nicolas EKICIER
Release:    V1.61   03/2021
"""

def add_field_id(input, field='nerid'):
    """
    Add an ID field in the attribute table of vector file
    :param input:   Path of file (any format supported by Fiona drivers)
    :param field:   Name of field (default = "nerid")
    :return:        Path of new file (shapefile)
    """
    ext = os.path.splitext(input)[-1]
    shp = gpd.read_file(input)
    shp[field] = np.arange(1, shp.shape[0]+1)
    name = input.replace(ext, '_nerid.shp')
    shp.to_file(name)
    return name


def getbbox(input):
    """
    Get bounding box of geometry
    :param input:   Path of ogr file or ogr geometry (cf below)
                            ogrl = ogr.Open(shp)
                            input = ogrl.GetLayer()
    :return:        Tuple : (xmin, xmax, ymin, ymax)
    """
    if type(input) is str:
        ogrl = ogr.Open(input)
        xmin, xmax, ymin, ymax = ogrl.GetLayer().GetExtent()
    else:
        try:
            xmin, xmax, ymin, ymax= input.GetGeometryRef().GetEnvelope()
        except:
            xmin, xmax, ymin, ymax = input.GetExtent()
    return (xmin, xmax, ymin, ymax)


def zonstat(inshp, inimg, attribut='id', njobs=-1, write=False):
    """
    Compute zonal statistics (count) on each polygon of shapefile from image
    :param inshp:       path of shapefile
    :param inimg:       path of image
    :param attribut:    attribute to use in shapefile table (default = 'id')
    :param njobs:       number of jobs to run in parallel (default = -1 = all cpus)
    :param write:       True if you want to write the DataFrame on disk (default = False)
                            CSV file in image folder with "_statz.csv" extent
    :return:            DataFrame
    """
    # Check proj
    epsg_shp = int(rpg.geoinfo(inshp, onlyepsg=True))
    epsg_img = int(rpg.geoinfo(inimg, onlyepsg=True))
    if epsg_shp != epsg_img:
        layertmp = ogreproj(inshp, epsg_img, write=True)
        inshp = inshp.replace('.shp', '_{0:d}.shp'.format(epsg_img))

    # Read
    img, _, _, _ = rpg.gdal2array(inimg)
    uidi = np.unique(img)

    mask = rpg.makemask(inshp, inimg, attribute=attribut)
    uid = np.unique(mask)

    # Stats
    def lambdf(x):
        values = img[np.nonzero(mask == x)]
        uidval, uidvalc = np.unique(values, return_counts=True)

        out_table = np.zeros((1, len(uidi)))
        out_table[0, np.in1d(uidi, uidval, assume_unique=True)] = uidvalc / np.sum(uidvalc) * 100
        return np.hstack((out_table, np.reshape(np.argmax(out_table), (1, 1))))

    out = Parallel(n_jobs=njobs)(delayed(lambdf)(uid[i]) for i in tqdm(range(0, uid.shape[0]),
                                                                      desc='Zonal statistics'))
    out = np.array(out).squeeze()

    output = pd.DataFrame(data=out[:, :-1], index=uid, columns=uidi)
    output['major_class'] = out[:, -1]
    if write:
        output.to_csv(inimg.replace('.tif', '_statz.csv'))

    # Clean
    if epsg_shp != epsg_img:
        shptmp = glob.glob(inshp.replace('.shp', '*'))
        for r in shptmp:
            os.remove(r)
    return output


def checkproj(layer0, layer1):
    """
    Check if projections are same OR same as layer1 (=EPSG)
    :param layer0:  path of shapefile 1
    :param layer1:  path of shapefile 2 or EPSG (ex : '4326', str)
    :return:        booleen and EPSG of each layer
    """
    proj0 = rpg.geoinfo(layer0, onlyepsg=True)
    if os.path.isfile(layer1):
        proj1 = rpg.geoinfo(layer1, onlyepsg=True)
    else:
        proj1 = layer1

    if proj0 != proj1:
        check = False
    else:
        check = True
    proj0 = 'epsg:{0:s}'.format(proj0)
    proj1 = 'epsg:{0:s}'.format(proj1)
    return check, proj0, proj1


def ogreproj(player, oEPSG, write=False):
    """
    Reprojection of an OGR layer
    :param layer:   Path of shapefile
    :param oEPSG:   EPSG value for destination (int)
    :param write:   if write, output is written on disk (same path of player with suffix)
    :return:        Reprojected layer & path of file if write=True
    """
    layer = gpd.read_file(player)

    iEPSG = layer.crs # Get projection from input and print in console
    print('Reprojection from epsg:{0:d} to epsg:{1:d}'.format(iEPSG.to_epsg(), oEPSG))

    data_proj = layer.copy()
    data_proj = data_proj.to_crs(epsg=oEPSG) # Reproject the geometries by replacing the values with projected ones
    data_proj.crs = from_epsg(oEPSG) # Determine the CRS of the GeoDataFrame

    if write:
        name = player.replace('.shp', '_{0:d}.shp'.format(oEPSG))
        data_proj.to_file(name)
        return data_proj, name
    else:
        return data_proj


def sprocessing(layer1, layer2, method):
    """
    Geometric processing between shapefiles
    :param layer1:  Geopandas Dataframe 1 OR path of shapefile
    :param layer2:  Geopandas Dataframe 2 OR path of shapefile
    :param method:  "intersects", "within", "contains"
    :return:        Result layer
    """
    str_test = 0
    if type(layer1) is str:
        str_test = str_test + 1
        lay1 = gpd.read_file(layer1)
    else:
        lay1 = layer1.copy()
    if type(layer2) is str:
        str_test = str_test + 1
        lay2 = gpd.read_file(layer2)
    else:
        lay2 = layer2.copy()

    # Check if projections are same
    if str_test == 2:
        check, _, _ = checkproj(layer1, layer2)
        if not check:
            logging.error('Warning : CRS are not the same')
            return None
    elif str_test == 1:
        if isinstance(lay1, gpd.GeoDataFrame):
            epsg = lay1.crs.to_epsg()
            check, _, _ = checkproj(layer2, epsg)
        else:
            epsg = lay2.crs.to_epsg()
            check, _, _ = checkproj(layer1, epsg)
        if not check:
            logging.error('Warning : CRS are not the same')
            return None

    # Process
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