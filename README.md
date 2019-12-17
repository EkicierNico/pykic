![Map](https://github.com/EkicierNico/pykic/blob/master/map.png "Map from Sentinel2 Timeserie")

# pykic module
# Python functions for a remote sensing application.

## 1. **Raster** : raster utilities, some based on gdal/osgeo bindings
### Functions in `pykic_gdal.py`
- **gdal2array** : read RS dataset from one raw file or from folders --> S2, LS8, maja, sen2cor. In this case, cloud and no data mask are read.
- **geoinfo** : Extract geo-informations from a raster or a geometry (only epsg for this type)
- **getextent** : Get the extent of a raster
- **array2tif** : Create a raster (.tif) from numpy array (only one band)
- **makemask** : Build a mask array from ogr geometry
- **resampling_and_reproject** : Resample and reproject a raster
### Functions in `pyfmask.py`
- **applyfmask** : Apply Fmask on Landsat data
### Functions in `resafilter.py`
- **resample_2d** : Make a resampling on image (some methods like lanczos, bicubic etc...)
- **pan_sharpen** : Make a pan-sharpening resampling on a raster

## 2. **Vector** : ogr utilities, mainly based on Geopandas and gdal/osgeo
### Functions in `pykic_ogr.py`
- **ogreproj** : Reprojection of an OGR layer
- **checkproj** : Check if projections between geometries are same OR same as a given epsg
- **zonstat** : Compute zonal statistics (count) on each polygon of layer from an image
- **sprocessing** : Geometric processing between layers (intersection etc...)
- **shpbuf** : Create a buffer of geometries from a layer
- **getbbox** : Get the bounding box of geometry (extent)
- **distm** : Geodesic distance between dots (degrees units)
- **add_field_id** : Add an ID field in the attribute table of a layer

## 3. **SMAC** : atmospheric correction from smac algorithm, developed by CESBIO
- _smac.py_ : apply atmospheric correction SMAC

## 4. **Miscellaneous** : generic codes to make the job easier
### Functions in `signalproc.py`
- **whittf** : Apply a Whittaker smoothing (weighted if needed)
- **fillnan_and_resample** : Interpolation / resampling of vector data with nan values
- **smooth_compute** : Compute Whittaker smoothing with multi-threading (array with some vector data)
- **outliers** : Extract index and values of outliers in input
- **phen_met** : Extract phenological metrics from one crop season
### Functions in `database.py`
- **buildproto** : Build a string protocol to get db
- **getdb** : Get data from db
- **getdbtable** : Get list of tables in the database
### Functions in `miscdate.py`
- **datefromstr** : Find a date into a string
- **dconvert** : Convert date format into another
- **dtoday** : Return date of today into a string
### Functions in `plotik.py`
- **imadjust** : Adjust intensity of an image
- **plot_confusion_matrix** : Built the confusion matrix as plot
### Functions in `vi.py`
- Computation of some vegetation index : NDVI, NDWI, EVI, WRDVI, SAVI, CVI
