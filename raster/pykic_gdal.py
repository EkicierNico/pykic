import os, glob, gc, random, string
from osgeo import gdal, gdal_array, ogr
import pandas as pd
import numpy as np
import scipy.misc as spm
import logging

"""
RASTER utilities
Author:     Nicolas EKICIER
Release:    V1.11   06/2019
                - gdal2array :  cloud mask = valid mask (OK = 0, NOK > 0)
                - makemask :    optimization
            V1.1    06/2019
                - Some changes
                - Add makemask function
            V1.O    02/2019
                - Initialization
"""

def gdal2array(filepath, sensor='S2MAJA', pansharp=False):
    """
    Read and transform a raster to array with bands stacking (Blue, Green, Red, NIR + CLOUDS)
    :param filepath:    - path of file
                        - folder => only one final product per folder
                            Example : .SAFE from S2
    :param sensor:      {'S2', 'S2MAJA' (default), 'LS8MAJA', 'LS8', 'SEN2COR}
    :param pansharp:    apply pan-sharpening (only possible with Landsat, default=False)
    :return:            array of raster, projection, dimensions, transform
    """
    def read(input):
        raster = gdal.Open(input)

        proj = raster.GetProjection()
        transform = raster.GetGeoTransform()

        rwidth = raster.RasterXSize
        rheight = raster.RasterYSize
        # numbands = raster.RasterCount

        rasterArray = gdal_array.LoadFile(input)
        return rasterArray, proj, (rwidth, rheight), transform

    if os.path.isdir(filepath):
        if sensor.lower() == 's2':
            bands = [2, 3, 4, 8] # B,G,R,N
            ext = 'B0$.jp2'
        elif sensor.lower() == 's2maja':
            bands = [2, 3, 4, 8]
            ext = 'FRE_B$.tif'
            cld = 'CLM_R1.tif'
            nodata = 'EDG_R1.tif'
        elif sensor.lower() == 'ls8maja':
            bands = [2, 3, 4, 5]
            pan = 'FRE_B8.tif' # panchro
            ext = 'FRE_B$.tif'
            cld = 'CLM_R1.tif'
            nodata = 'EDG_R1.tif'
        elif sensor.lower() == 'ls8':
            bands = [2, 3, 4, 5] # B,G,R,N
            pan = 'B8.TIF' # panchro
            ext = 'B$.TIF'
            cld = 'fmask.img'
        elif sensor.lower() == 'sen2cor':
            bands = [2, 3, 4, 8] # B,G,R,N
            ext = 'B0$_10m.jp2'
            cld = 'MSK_CLDPRB_20m.jp2'

        # Define the workspace
        workdir = glob.glob(os.path.join(filepath, '**/*'+ext.replace('$', '2')), recursive=True)
        if not workdir:
            logging.error(' No corresponding files')
            return None
        workdir = os.path.dirname(workdir[0])

        # Gdal read
        output = np.array([])
        for i in bands:
            pathf = glob.glob(os.path.join(workdir, '*'+ext.replace('$', str(i))), recursive=False)
            tmp, proj, dimensions, transform = read(pathf[0])
            if output.size == 0:
                output = tmp.copy()
            else:
                output= np.dstack((output, tmp.copy()))
            tmp = None; del tmp
            gc.collect()

        # Cloud mask (valid = 0)
        if sensor.lower().find('maja') >= 0 or sensor.lower() == 'ls8' or sensor.lower() == 'sen2cor':
            if sensor.lower().find('maja') >= 0:
                pathc = glob.glob(os.path.join(workdir, 'MASKS', '*'+cld), recursive=False)
                pathn = glob.glob(os.path.join(workdir, 'MASKS', '*'+nodata), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0])
                    tmp2, _, _, _ = read(pathn[0])
                    tmp[tmp2 == 1] = 1 # Apply nodata in cloud mask
                    tmp2 = None
            elif sensor.lower() == 'ls8':
                pathc = glob.glob(os.path.join(workdir, '*'+cld), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0])
                    tmp[tmp != 1] = 5
                    tmp[tmp == 1] = 0
            elif sensor.lower() == 'sen2cor':
                pathc = glob.glob(os.path.join(filepath, '**/*'+cld), recursive=True)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0])
                    tmp = spm.imresize(tmp, dimensions, interp='nearest')

            output= np.dstack((output, np.int16(tmp.copy())))
            tmp = None; del tmp
            gc.collect()


    elif os.path.isfile(filepath):
        output, proj, dimensions, transform = read(filepath)

    return output, proj, dimensions, transform


def makemask(ogr_in, imgpath, attribute='ID', write=False):
    """
    Build a mask array from an OGR geometry
    :param ogr_in:      path of shape
    :param imgpath:     path of image ref
    :param attribute:   attribute in OGR table for burn value (string, default = 'ID)
    :param write:       keep mask file on disk (default = False)
    :return maskarray:  mask (array)
    """
    # Define dimensions and NoData value of new raster (= 0 in MEME method)
    raster = gdal.Open(imgpath)

    rproj = raster.GetProjection()
    rwidth = raster.RasterXSize
    rheight = raster.RasterYSize

    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixel_size = transform[1]
    raster = None

    NoData_value = -1

    # Open OGR source
    if os.path.isfile(ogr_in):
        source_ds = ogr.Open(ogr_in)
        layer = source_ds.GetLayer()

    # Create raster
    if write == True:
        # Filename of the raster Tiff that will be created
        chars = string.ascii_lowercase + string.digits + '_'
        raster_fn = ''.join(random.choice(chars) for _ in range(8))
        raster_fn = os.path.join(os.getcwd(), raster_fn + '_ogrmask.tif')

        # Create the destination data source
        target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, rwidth, rheight, 1, gdal.GDT_Int16)
    else:
        target_ds = gdal.GetDriverByName('MEM').Create('', rwidth, rheight, 1, gdal.GDT_Int16)

    target_ds.SetGeoTransform(transform)
    target_ds.SetProjection(rproj)
    target_ds.GetRasterBand(1).SetNoDataValue(NoData_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], layer, options=['ATTRIBUTE={0:s}'.format(attribute)])

    # Read mask
    if write == True:
        target_ds.FlushCache()
        maskarray, _, _, _ = gdal2array(raster_fn)
    else:
        maskarray = target_ds.GetRasterBand(1).ReadAsArray()

    target_ds = None
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
        logging.error(" Error of reprojection/resampled")
        return None
    return out_raster