import os, glob, shutil
import logging
import miscellaneous.miscdate as mm

"""
Function to apply Fmask to Landast 8

Author:     Nicolas EKICIER
Dependancies:   python-fmask (install from Conda)
                    'conda config --add channels conda-forge'
                    'conda install python-fmask'

Release:    V1.O    06/2019
                - Initialization
"""

def applyfmask(directory):
    """
    Apply Fmask on Landsat 8
    :param directory:   - path of folder image
                        - path with several folders (unzipped folder from .gz Landsat file)
    :return:            LC08_PATHROW_DATE_fmask.img in each folder
                            code 0 = null
                            code 1 = clear land
                            code 2 = cloud
                            code 3 = cloud shadow
                            code 4 = snow
                            code 5 = water
    """
    imgdir = os.listdir(directory)
    tmpdir = os.path.join(pwd, 'tmpfmask')
    os.mkdir(tmpdir)

    if os.path.isdir(os.path.join(directory, imgdir[0])):
        for p in imgdir:
            if os.path.isdir(os.path.join(directory, p)):
                # Output name
                namef = glob.glob(os.path.join(directory, p, '*B2.TIF'))
                namef = os.path.basename(namef[0])

                pathrow = namef.split('_')
                pathrow = pathrow[2]
                dato = mm.datefromstr(namef)

                oname = 'LC08_{0:s}_{1:d}_fmask.img'.format(pathrow, dato)

                # Check if clouds file already exists
                if os.path.isfile(os.path.join(directory, p, oname)):
                    logging.warning(" File '{0:s}' already exists".format(oname))
                    continue

                # Fmask
                cmd = 'fmask_usgsLandsatStacked.py -o {0:s} -e {2:s} --scenedir {1:s}'.format(os.path.join(directory, p, oname),
                                                                                              os.path.join(directory, p),
                                                                                              tmpdir)
                os.system(cmd)

    elif os.path.isfile(os.path.join(directory, imgdir[0])):
        # Output name
        namef = glob.glob(os.path.join(directory, '*B2.TIF'))
        namef = os.path.basename(namef[0])

        pathrow = namef.split('_')
        pathrow = pathrow[2]
        dato = mm.datefromstr(namef)

        oname = 'LC08_{0:s}_{1:d}_fmask.img'.format(pathrow, dato)

        # Check if clouds file already exists
        if os.path.isfile(os.path.join(directory, oname)):
            logging.warning(" File '{0:s}' already exists".format(oname))
            return None

        # Fmask
        cmd = 'fmask_usgsLandsatStacked.py -o {0:s} -e {2:s} --scenedir {1:s}'.format(os.path.join(directory, oname),
                                                                                      directory,
                                                                                      tmpdir)
        os.system(cmd)

    shutil.rmtree(tmpdir)
    return None