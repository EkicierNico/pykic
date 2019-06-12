import os, glob
from osgeo import gdal, gdal_array
import pandas as pd
import numpy as np
import logging

"""
RASTER utilities
Author:     Nicolas EKICIER
Release:    V1.1    06/2019
        - Some changes
        - Add makemask function
            V1.O    02/2019
        - Initialization
"""

def gdal2array(filepath, sensor='S2'):
    """
    Read and transform a raster to array
    :param filepath:    - path of file
                        - folder => only one final product per folder
                            Example : .SAFE from S2
    :param sensor:      {'S2', 'LS8'}
    :return:            array of raster, projection, dimensions
    """
    def read(input):
        # Open the file:
        raster = gdal.Open(input)
        # Projection & Transform
        proj = raster.GetProjection()
        transform = raster.GetGeoTransform()
        # Dimensions
        rwidth = raster.RasterXSize
        rheight = raster.RasterYSize
        # Number of bands
        numbands = raster.RasterCount
        # Read raster data as numeric array from file
        rasterArray = gdal_array.LoadFile(input)
        return rasterArray, proj, (rwidth, rheight), numbands, transform

    if os.path.isdir(filepath):
        if sensor.lower() == 's2':
            ext = '.jp2'
            nb_bands = 4 # 4 bands at 10m
        elif sensor.lower() == 'ls8':
            ext = '.tif'

        listf = glob.glob(os.path.join(filepath, '*'+ext), recursive=True)

    elif os.path.isfile(filepath):
        output, proj, dimensions, numbands, transform = read(filepath)

    return output, proj, dimensions, numbands, transform


def makemask(ogr_in, filepath, attribute, write=False):
    """
    Build a mask array from an OGR geometry
    :param ogr_in:
    :param filepath:
    :param attribute:   attribute in OGR table for burn value (string)
    :param write:       keep mask file on disk (default = False)
    :return:
    """
    # Define dimensions and NoData value of new raster
    raster = gdal.Open(filepath)

    rproj = raster.GetProjection()
    rwidth = raster.RasterXSize
    rheight = raster.RasterYSize

    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixel_size = transform[1]

    NoData_value = -9999

    # Open OGR source
    if os.path.isfile(ogr_in):
        source_ds = ogr.Open(vector_fn)
        source_layer = source_ds.GetLayer()

    # Filename of the raster Tiff that will be created
    raster_fn = 'test.tif'

    # Create the destination data source
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, rwidth, rheight, 1, gdal.GDT_Byte)
    # target_ds.SetGeoTransform((xOrigin, pixel_size, 0, yOrigin, 0, -pixel_size))
    target_ds.SetGeoTransform(transform)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, attribute=attribute)
    return maskarray


def resample_and_reproject(image_in, out_width, out_height, out_geo_transform, out_proj=None, mode='bicubic'):
    """
    Resample and reproject a raster

    :param image_in: input raster
    :param out_width: pixel width of the output raster
    :param out_height: pixel height of the output raster
    :param out_geo_transform: GeoTransform of the output raster
    :param out_proj: output projection (default to input projection)
    :param mode: interpolation mode ['bicubic' (default), 'average', 'bilinear', 'lanczos' or 'nearest']
    :type image_in: ``ogr.DataSet``
    :type out_width: int
    :type out_height: int
    :type out_geo_transform: tuple of floats
    :type out_proj: string
    :type mode: string
    :return: the reprojected and resampled raster, None otherwise
    :rtype: ``gdal.DataSet``
    """
    if isinstance(image_in, str):
        image_in = gdal.Open(image_in)
    in_proj = image_in.GetProjection()

    out_raster = gdal.GetDriverByName('MEM').Create("", out_width, out_height, image_in.RasterCount,
                                                    image_in.GetRasterBand(1).DataType)
    out_raster.SetGeoTransform(out_geo_transform)

    mode_gdal = gdal.GRA_Cubic
    if mode.lower() == 'average':
        mode_gdal = gdal.GRA_Average
    elif mode.lower() == 'bilinear':
        mode_gdal = gdal.GRA_Bilinear
    elif mode.lower() == 'lanczos':
        mode_gdal = gdal.GRA_Lanczos
    elif mode.lower() == 'nearest':
        mode_gdal = gdal.GRA_NearestNeighbour

    reproj_res = None
    if out_proj is None:
        out_raster.SetProjection(in_proj)
        reproj_res = gdal.ReprojectImage(image_in, out_raster, None, None, mode_gdal)
    else:
        out_raster.SetProjection(out_proj)
        reproj_res = gdal.ReprojectImage(image_in, out_raster, in_proj, out_proj, mode_gdal)

    if reproj_res != 0:
        logging.error("Error of reprojection/resampled")
        return None
    return out_raster