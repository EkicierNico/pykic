![Map](/home/nicolas.ekicier/Téléchargements/map.png "Map from Sentinel2 Timeserie")

# pykic module
## Python functions for a remote sensing application.

### 1. **Raster** : raster utilities, some based on gdal/osgeo bindings
- _pykic_gdal.py_ : read/write RS dataset (raw, maja, sen2cor), mask from ogr layer, resampling & reprojection
- _pyfmask.py_ : apply Fmask on Landsat data
- _resafilter.py_ : resampling & filtering utilities (pan_sharpening etc...)

### 2. **Vector** : ogr utilities, mainly based on Geopandas and gdal/osgeo
- _pykic_ogr.py_ : reprojection, spatial processing, distance computation

### 3. **SMAC** : atmospheric correction from smac algorithm, developed by CESBIO
- _smac.py_ : apply atmospheric correction SMAC

### 4. **Miscellaneous** : generic codes to make the job easier
- _database.py_ : database utilities
- _miscdate.py_ : functions to manipulate dates, extract it from string etc...
- _plotik.py_ : plotting utilities
- _signalproc.py_ : signal processing (smoothing, interpolation, outliers etc...)
- _vi.py_ : compute vegetation index