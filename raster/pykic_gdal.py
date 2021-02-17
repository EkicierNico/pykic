import os, glob, gc, random, string, logging
from osgeo import gdal, gdal_array, ogr, osr
import numpy as np
from scipy.spatial import distance

import raster.resafilter as rrf

"""
RASTER utilities
Author:     Nicolas EKICIER
Release:    V2.23   02/2021
"""

def gdal2array(filepath, nband=None, sensor='S2MAJA', pansharp=False, subset=None):
    """
    Read and transform a raster to array with bands stacking (Blue, Green, Red, NIR + CLOUDS/NoData)
    Possible to read only a subset
    :param filepath:    - path of file
                        - folder => only one final product per folder
                            Example : .SAFE from S2
    :param nband:       number of band (only if filepath is not a folder)
                        if None, all bands are read
    :param sensor:      {'S2', 'S2MAJA' (default), 'LS8MAJA', 'LS8', 'SEN2COR', 'WASP'}
    :param pansharp:    apply pan-sharpening (only possible with Landsat, default=False)
    :param subset:      read only a subset of image
                        tuple of bounding box (xmin, xmax, ymin, ymax) in native projection
    :return:            array of raster, projection, dimensions, transform
    """
    def read(input, nband, box):
        raster = gdal.Open(input)

        proj = raster.GetProjection()
        transform = raster.GetGeoTransform()

        rwidth = raster.RasterXSize
        rheight = raster.RasterYSize

        if box is not None:
            # Convert coordinates to index
            inv_tr = gdal.InvGeoTransform(transform)
            x0, y0 = gdal.ApplyGeoTransform(inv_tr, box[0], box[2])
            x1, y1 = gdal.ApplyGeoTransform(inv_tr, box[1], box[3])
            x0b, y0b = min(x0, x1), min(y0, y1)
            x1b, y1b = max(x0, x1), max(y0, y1)

            # New geotransform
            rwidth = int(x1b-x0b)
            rheight = int(y1b-y0b)
            xmin = transform[0] + int(x0b) * transform[1]
            ymax = transform[3] - int(y0b) * transform[1]
            transform = (xmin, transform[1], transform[2], ymax, transform[4], transform[5])

        if nband is not None:
            if box is not None:
                rasterArray = raster.GetRasterBand(nband).ReadAsArray(int(x0b), int(y0b), rwidth, rheight)
            else:
                rasterArray = raster.GetRasterBand(nband).ReadAsArray()
        else:
            if box is not None:
                rasterArray = gdal_array.LoadFile(
                    input, xoff=int(x0b), yoff=int(y0b), xsize=rwidth, ysize=rheight)
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
            panchro, projp, dimensionsp, transformp = read(pathp[0], nband, subset)

        # Gdal read
        output = np.array([])
        for i in bands:
            pathf = glob.glob(os.path.join(workdir, '*'+ext.replace('$', str(i))), recursive=False)
            tmp, proj, dimensions, transform = read(pathf[0], nband, subset)
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
                    tmp, _, _, _ = read(pathc[0], nband, subset)
                    tmp2, _, _, _ = read(pathn[0], nband, subset)
                    tmp[tmp2 == 1] = 1 # Apply nodata in cloud mask
                    tmp2 = None
            elif sensor.lower() == 'ls8':
                pathc = glob.glob(os.path.join(workdir, '*'+cld), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0], nband, subset)
                    tmp[tmp != 1] = 5
                    tmp[tmp == 1] = 0
            elif sensor.lower() == 'sen2cor':
                pathc = glob.glob(os.path.join(filepath, '**/*'+cld), recursive=True)
                pathn = glob.glob(os.path.join(workdir, '*'+ext.replace('$', str(2))), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0], nband, subset)
                    tmp2, _, _, _ = read(pathn[0], nband, subset)
                    tmp = rrf.resample_2d(tmp, dimensions, method='nearest')
                    tmp[tmp2 == 0] = 100 # Apply nodata in cloud mask
                    tmp2 = None
            if sensor.lower() == 'wasp':
                pathc = glob.glob(os.path.join(workdir, 'MASKS', '*'+cld), recursive=False)
                if os.path.isfile(pathc[0]):
                    tmp, _, _, _ = read(pathc[0], nband, subset)
                    tmp[tmp != 4] = 1
                    tmp[tmp == 4] = 0

            if sensor.lower().find('ls8') and pansharp == True:
                output= np.dstack((output, np.int16(rrf.resample_2d(tmp.copy(), dimensionsp, method='nearest'))))
            else:
                output= np.dstack((output, np.int16(tmp.copy())))
            tmp = None; del tmp
            gc.collect()

    elif os.path.isfile(filepath):
        output, proj, dimensions, transform = read(filepath, nband, subset)

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


def array2tif(newRasterfn, array, proj, dimensions, transform, format='uint8', cog=False, compress='lzw'):
    """
    Create a raster (.tif) from numpy array (x, y, bands)
    Compression used :  LZW Pred_2/3 (All CPUS used)
                        ZSTD Level_1 Pred_2/3 (All CPUS used) -> better perfs but less compatibilities
        Benchmarks: https://kokoalberti.com/articles/geotiff-compression-optimization-guide/
    :param newRasterfn: output path
    :param array:       numpy array (input)
    :param proj:        projection struct from gdal method
    :param dimensions:  dimensions of output (cols, rows)
    :param transform:   transform struct from gdal method
    :param format:      {'uint8' --> default, 'uint16', 'int16', 'uint32', 'int32', 'float32'}
    :param cog:         export as Cloud Optimized Geotiff format (COG) - default = False
    :param compress:    compression method ('lzw' = default, 'zstd')
    :return:
    """
    val_pred = 2
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
    elif format.lower() == 'float32':
        gdt = gdal.GDT_Float32
        val_pred = 3 # predictor = 3 with float32 format

    if cog == False:
        if compress.lower() == 'lzw':
            co = ['COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS', f'PREDICTOR={val_pred}']
        else:
            co = ['COMPRESS=ZSTD', 'ZSTD_LEVEL=1', 'NUM_THREADS=ALL_CPUS', f'PREDICTOR={val_pred}']

        if array.ndim == 3:
            nbands = array.shape[-1]
            outRaster = gdal.GetDriverByName('GTiff').Create(newRasterfn, dimensions[0], dimensions[1], nbands, gdt,
                                                             options=co)
            for i in range(nbands):
                outband = outRaster.GetRasterBand(i+1).WriteArray(array[:, :, i])
        else:
            outRaster = gdal.GetDriverByName('GTiff').Create(newRasterfn, dimensions[0], dimensions[1], 1, gdt,
                                                             options=co)
            outband = outRaster.GetRasterBand(1).WriteArray(array)

    else:
        if compress.lower() == 'lzw':
            co = ['COPY_SRC_OVERVIEWS=YES', 'TILED=YES',
                  'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS', f'PREDICTOR={val_pred}']
        else:
            co = ['COPY_SRC_OVERVIEWS=YES', 'TILED=YES',
                  'COMPRESS=ZSTD', 'ZSTD_LEVEL=1', 'NUM_THREADS=ALL_CPUS', f'PREDICTOR={val_pred}']

        if array.ndim == 3:
            nbands = array.shape[-1]
            outRasterTmp = gdal.GetDriverByName('MEM').Create('', dimensions[0], dimensions[1], nbands, gdt)
            for i in range(nbands):
                outband = outRasterTmp.GetRasterBand(i+1).WriteArray(array[:, :, i])
        else:
            outRasterTmp = gdal.GetDriverByName('MEM').Create('', dimensions[0], dimensions[1], 1, gdt)
            outband = outRasterTmp.GetRasterBand(1).WriteArray(array)

        outRasterTmp.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])
        outRaster = gdal.GetDriverByName('GTiff').CreateCopy(newRasterfn, outRasterTmp,
                                                             options=co)

    outRaster.SetGeoTransform(transform)
    outRaster.SetProjection(proj)
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


def imreproj(image_in, epsg_out, mode='nearest'):
    """
    Reproject a raster in memory (without write on disk)
    :param image_in:    path of input raster
    :param epsg_out:    output epsg (int)
    :param mode:        interpolation mode ['nearest' (default), 'bicubic', 'average', 'bilinear', 'lanczos']
    :return:            reprojected raster in memory (RAM)
    """
    mode_gdal = gdal.GRA_NearestNeighbour
    if mode.lower() == 'average':
        mode_gdal = gdal.GRA_Average
    elif mode.lower() == 'bilinear':
        mode_gdal = gdal.GRA_Bilinear
    elif mode.lower() == 'bicubic':
        mode_gdal = gdal.GRA_Cubic
    elif mode.lower() == 'lanczos':
        mode_gdal = gdal.GRA_Lanczos

    # Read source dataset
    im = gdal.Open(image_in)
    proj, dim, tr = geoinfo(image_in)
    epsg_in = int(osr.SpatialReference(wkt=proj).GetAttrValue('AUTHORITY', 1))
    logging.warning(' Reprojection from epsg:{0:d} to epsg:{1:d}'.format(epsg_in, epsg_out))

    # Manage transform
    source = osr.SpatialReference()
    source.ImportFromEPSG(epsg_in)
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(epsg_out)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(source, target)
    ulx, uly, _ = transform.TransformPoint(tr[0], tr[3])
    lrx, lry, _ = transform.TransformPoint(tr[0] + tr[1] * dim[0],
                                           tr[3] + tr[5] * dim[1])

    px2, _, _ = transform.TransformPoint(tr[0] + tr[1], tr[3])
    res = abs(px2 - ulx)

    # Reproj
    out_dim = (int((lrx - ulx) / res), int((uly - lry) / res))
    dest = gdal.GetDriverByName('MEM').Create('', out_dim[0], out_dim[1],
                                              im.RasterCount, im.GetRasterBand(1).DataType)

    out_tr = (ulx, res, tr[2], uly, tr[4], -res)
    dest.SetGeoTransform(out_tr)
    dest.SetProjection(target.ExportToWkt())

    _ = gdal.ReprojectImage(im, dest, source.ExportToWkt(), target.ExportToWkt(), mode_gdal)
    imr = gdal_array.DatasetReadAsArray(dest)
    return imr, target.ExportToWkt(), out_dim, out_tr