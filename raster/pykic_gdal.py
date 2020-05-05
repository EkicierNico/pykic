import os, glob, gc, random, string, osr
from osgeo import gdal, gdal_array, ogr
import pandas as pd
import numpy as np
from scipy.spatial import distance
import logging

import PIL.Image as pilim
import raster.resafilter as rrf

"""
RASTER utilities
Author:     Nicolas EKICIER
Release:    V1.7   05/2020
"""

def gdal2array(filepath, nband=None, sensor='S2MAJA', pansharp=False):
    """
    Read and transform a raster to array with bands stacking (Blue, Green, Red, NIR + CLOUDS/NoData)
    :param filepath:    - path of file
                        - folder => only one final product per folder
                            Example : .SAFE from S2
    :param nband:       number of band (only if filepath is not a folder)
                        if None, all bands are read
    :param sensor:      {'S2', 'S2MAJA' (default), 'LS8MAJA', 'LS8', 'SEN2COR', 'WASP'}
    :param pansharp:    apply pan-sharpening (only possible with Landsat, default=False)
    :return:            array of raster, projection, dimensions, transform
    """
    def read(input, nband):
        raster = gdal.Open(input)

        proj = raster.GetProjection()
        transform = raster.GetGeoTransform()

        rwidth = raster.RasterXSize
        rheight = raster.RasterYSize
        # numbands = raster.RasterCount

        if nband is not None:
            tmpband = raster.GetRasterBand(nband)
            rasterArray = tmpband.ReadAsArray()
        else:
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
        elif sensor.lower() == 'wasp':
            bands = [2, 3, 4, 8] # B,G,R,N
            ext = 'FRC_B$.tif'
            cld = 'FLG_R1.tif'

        # Define the workspace
        workdir = glob.glob(os.path.join(filepath, '**/*'+ext.replace('$', '2')), recursive=True)
        if not workdir:
            logging.error(' No corresponding files')
            return None
        workdir = os.path.dirname(workdir[0])

        # Pan-Sharpening (read panchromatic band)
        if sensor.lower().find('ls8') and pansharp == True:
            pathp = glob.glob(os.path.join(workdir, '*'+pan), recursive=False)
            panchro, projp, dimensionsp, transformp = read(pathp[0], nband)

        # Gdal read
        output = np.array([])
        for i in bands:
            pathf = glob.glob(os.path.join(workdir, '*'+ext.replace('$', str(i))), recursive=False)
            tmp, proj, dimensions, transform = read(pathf[0], nband)
            if output.size == 0:
                # Pan-Sharpening
                if sensor.lower().find('ls8') and pansharp == True:
                    output = rrf.pan_sharpen(tmp.copy(), panchro)
                else:
                    output = tmp.copy()
            else:
                if sensor.lower().find('ls8') and pansharp == True:
                    output= np.dstack((output, rrf.pan_sharpen(tmp.copy(), panchro)))
                else:
                    output= np.dstack((output, tmp.copy()))
            tmp = None; del tmp
            gc.collect()

        # Cloud mask (valid = 0)
        if sensor.lower().find('maja') >= 0 or sensor.lower() == 'ls8' or sensor.lower() == 'sen2cor' or sensor.lower() == 'wasp':
            if sensor.lower().find('maja') >= 0:
                pathc = glob.glob(os.path.join(workdir, 'MASKS', '*'+cld), recursive=False)
                pathn = glob.glob(os.path.join(workdir, 'MASKS', '*'+nodata), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0], nband)
                    tmp2, _, _, _ = read(pathn[0], nband)
                    tmp[tmp2 == 1] = 1 # Apply nodata in cloud mask
                    tmp2 = None
            elif sensor.lower() == 'ls8':
                pathc = glob.glob(os.path.join(workdir, '*'+cld), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0], nband)
                    tmp[tmp != 1] = 5
                    tmp[tmp == 1] = 0
            elif sensor.lower() == 'sen2cor':
                pathc = glob.glob(os.path.join(filepath, '**/*'+cld), recursive=True)
                pathn = glob.glob(os.path.join(workdir, '*'+ext.replace('$', str(2))), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0], nband)
                    tmp2, _, _, _ = read(pathn[0], nband)
                    tmp = rrf.resample_2d(tmp, dimensions, method='nearest')
                    tmp[tmp2 == 0] = 100 # Apply nodata in cloud mask
                    tmp2 = None
            if sensor.lower() == 'wasp':
                pathc = glob.glob(os.path.join(workdir, 'MASKS', '*'+cld), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0], nband)
                    tmp[tmp != 4] = 1
                    tmp[tmp == 4] = 0

            if sensor.lower().find('ls8') and pansharp == True:
                output= np.dstack((output, np.int16(rrf.resample_2d(tmp.copy(), dimensionsp, method='nearest'))))
            else:
                output= np.dstack((output, np.int16(tmp.copy())))
            tmp = None; del tmp
            gc.collect()

    elif os.path.isfile(filepath):
        output, proj, dimensions, transform = read(filepath, nband)

    # Return
    if sensor.lower().find('ls8') and pansharp == True:
        return output, projp, dimensionsp, transformp
    else:
        return output, proj, dimensions, transform


def geoinfo(filepath, onlyepsg=False):
    """
    Extract structure geoinfo from gdal
    :param filepath:    path of raster file (ogr file only possible with onlyepsg=True)
    :param onlyepsg:    to extract only the EPSG code
    :return:            proj, dimensions, transform OR EPSG
    """
    if onlyepsg == True:
        try:
            source_ds = ogr.Open(filepath)
            layer = source_ds.GetLayer().GetSpatialRef()
            epsg = layer.GetAuthorityCode(None)
        except:
            raster = gdal.Open(filepath)
            epsg = osr.SpatialReference(wkt=raster.GetProjection()).GetAttrValue('AUTHORITY', 1)
        return epsg
    else:
        raster = gdal.Open(filepath)

        proj = raster.GetProjection()
        transform = raster.GetGeoTransform()
        rwidth = raster.RasterXSize
        rheight = raster.RasterYSize
        return proj, (rwidth, rheight), transform


def getextent(input):
    """
    Get the extent of a raster file
    :param input:   path of ratser
    :return:        tuple : (xmin, xmax, ymin, ymax)
    """
    proj, dim, tr = geoinfo(input)
    xmin = tr[0]
    ymax = tr[3]

    # Decimals
    nd = np.max([str(xmin)[::-1].find('.') + 1, str(ymax)[::-1].find('.') + 1])

    xmax = round(xmin + tr[1] * dim[0], nd)
    ymin = round(ymax + tr[-1] * dim[-1], nd)
    return (xmin, xmax, ymin, ymax)


def array2tif(newRasterfn, array, proj, dimensions, transform, format='uint8', cog=False):
    """
    Create a raster (.tif) from numpy array (x, y, bands)
    :param newRasterfn: output path
    :param array:       numpy array (input)
    :param proj:        projection struct from gdal method
    :param dimensions:  dimensions of output (cols, rows)
    :param transform:   transform struct from gdal method
    :param format:      {'uint8' --> default, 'uint16', 'int16', 'uint32', 'int32'}
    :param cog:         export as Cloud Optimized Geotiff format (COG) - default = False
    :return:
    """
    if format.lower() == 'uint8':
        gdt = gdal.GDT_Byte
    elif format.lower() == 'uint16':
        gdt = gdal.GDT_UInt16
    elif format.lower() == 'int16':
        gdt = gdal.GDT_Int16
    elif format.lower() == 'uint32':
        gdt = gdal.GDT_UInt32
    elif format.lower() == 'int32':
        gdt = gdal.GDT_Int32

    if cog == False:
        if array.ndim == 3:
            nbands = array.shape[-1]
            outRaster = gdal.GetDriverByName('GTiff').Create(newRasterfn, dimensions[0], dimensions[1], nbands, gdt,
                                                             options=['COMPRESS=LZW'])
            for i in range(nbands):
                outband = outRaster.GetRasterBand(i+1).WriteArray(array[:, :, i])
        else:
            outRaster = gdal.GetDriverByName('GTiff').Create(newRasterfn, dimensions[0], dimensions[1], 1, gdt,
                                                             options=['COMPRESS=LZW'])
            outband = outRaster.GetRasterBand(1).WriteArray(array)

    else:
        nbands = array.shape[-1]
        outRasterTmp = gdal.GetDriverByName('MEM').Create('', dimensions[0], dimensions[1], nbands, gdt)
        for i in range(nbands):
            outband = outRasterTmp.GetRasterBand(i+1).WriteArray(array[:, :, i])

        outRasterTmp.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])
        outRaster = gdal.GetDriverByName('GTiff').CreateCopy(newRasterfn, outRasterTmp,
                                                             options=['COPY_SRC_OVERVIEWS=YES',
                                                                      'TILED=YES',
                                                                      'COMPRESS=LZW'])

    outRaster.SetGeoTransform(transform)
    outRaster.SetProjection(proj)

    # outband.FlushCache()
    # outRaster = None
    return None


def makemask(ogr_in, imgpath, attribute='ID', write=False):
    """
    Build a mask array from an OGR geometry --> Int32
    :param ogr_in:      path of shape
    :param imgpath:     path of image ref
    :param attribute:   attribute in OGR table for burn value (string, default = 'ID)
    :param write:       keep mask file on disk (default = False)
    :return maskarray:  mask (array)
    """
    # Define dimensions and NoData value of new raster (= 0 in MEME method)
    rproj, dim, transform = geoinfo(imgpath)

    rwidth = dim[0]
    rheight = dim[1]

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixel_size = transform[1]
    raster = None

    NoData_value = -1

    # Open OGR source
    if os.path.isfile(ogr_in):
        source_ds = ogr.Open(ogr_in)
        layer = source_ds.GetLayer()

    # Check if projections are same
    if geoinfo(ogr_in, onlyepsg=True) != geoinfo(imgpath, onlyepsg=True):
        logging.error(' Warning : CRS are not the same')
        return None

    # Create raster
    if write == True:
        # Filename of the raster Tiff that will be created
        chars = string.ascii_lowercase + string.digits + '_'
        raster_fn = ''.join(random.choice(chars) for _ in range(8))
        raster_fn = os.path.join(os.getcwd(), raster_fn + '_ogrmask.tif')

        # Create the destination data source
        target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, rwidth, rheight, 1, gdal.GDT_Int32)
    else:
        target_ds = gdal.GetDriverByName('MEM').Create('', rwidth, rheight, 1, gdal.GDT_Int32)

    target_ds.SetGeoTransform(transform)
    target_ds.SetProjection(rproj)
    outband = target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], layer, options=['ATTRIBUTE={0:s}'.format(attribute)])

    # Read mask
    if write == True:
        outband.FlushCache()
        maskarray, _, _, _ = gdal2array(raster_fn)
    else:
        maskarray = target_ds.GetRasterBand(1).ReadAsArray()

    target_ds = None
    return maskarray


def valfromdot(img, dim, trans, coord, win='unique'):
    """
    Extract values of raster from coordinates
    :param img:     array (raster)
    :param dim:     dimensions of raster (gdal struct)
    :param trans:   transformation of ratser (gdal struct)
    :param coord:   coordinates vector -> array (x, y)
    :param win:     method to extract value (unique = one pixel --> default, square = median of 3x3 pixels)
    :return:        values, coordinates index
    """
    # Build coordinates vector of raster (pixel center)
    xi = np.arange(trans[0]+trans[1]/2, trans[0]+trans[1]*dim[0]+trans[1]/2, trans[1]).reshape((-1, 1))
    yi = np.arange(trans[3]+trans[-1]/2, trans[3]+trans[-1]*dim[1]+trans[-1]/2, trans[-1]).reshape((-1, 1))

    # Get index from distance
    ixi = np.nanargmin(distance.cdist(xi, coord[:, 0].reshape((-1, 1)), metric='euclidean'), axis=0)
    iyi = np.nanargmin(distance.cdist(yi, coord[:, 1].reshape((-1, 1)), metric='euclidean'), axis=0)

    # Extract values
    if win.lower() == 'unique':
        values = img[iyi, ixi]
    elif win.lower() == 'square':
        values = np.array([])
        xw = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        yw = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

        for i, j in zip(ixi, iyi):
            # Out of image
            if i == 0 or j == 0:
                if values.size == 0:
                    values = np.nan
                else:
                    values = np.hstack((values, np.nan))
                continue

            xwt = xw.copy() + i
            ywt = yw.copy() + j
            if values.size == 0:
                values = np.nanmedian(img[xwt.reshape((-1, 1)), ywt.reshape((-1, 1))])
            else:
                values = np.hstack((values, np.nanmedian(img[xwt.reshape((-1, 1)), ywt.reshape((-1, 1))])))
    return values.reshape((-1, 1)), np.vstack((ixi, iyi)).transpose()


def resample_and_reproject(image_in, dim, out_geo_transform, out_proj=None, mode='lanczos'):
    """
    Resample and reproject a raster

    :param image_in:            input raster
    :param dim:                 output dimensions (width, height)
    :param out_geo_transform:   GeoTransform of the output raster
    :param out_proj:            output projection (default to input projection)
    :param mode:                interpolation mode ['bicubic', 'average', 'bilinear', 'lanczos' (default) or 'nearest']
    :type image_in:             ``ogr.DataSet``
    :type dim:                  tuple of int
    :type out_geo_transform:    tuple of floats
    :type out_proj:             string (ex : 'EPSG:4326')
    :type mode:                 string
    :return:                    the reprojected and resampled raster, None otherwise
    :rtype: `                   `gdal.DataSet``
    """
    if isinstance(image_in, str):
        image_in = gdal.Open(image_in)
    in_proj = image_in.GetProjection()

    out_raster = gdal.GetDriverByName('MEM').Create("", dim[0], dim[1], image_in.RasterCount,
                                                    image_in.GetRasterBand(1).DataType)
    out_raster.SetGeoTransform(out_geo_transform)

    mode_gdal = gdal.GRA_Lanczos
    if mode.lower() == 'average':
        mode_gdal = gdal.GRA_Average
    elif mode.lower() == 'bilinear':
        mode_gdal = gdal.GRA_Bilinear
    elif mode.lower() == 'bicubic':
        mode_gdal = gdal.GRA_Cubic
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