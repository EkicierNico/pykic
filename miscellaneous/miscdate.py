import time, datetime
from datetime import date
import logging

"""
Dates utilities
Author:     Nicolas EKICIER
Release:    V1.11   07/2019
                - Add timestamp format as in/output
            V1.1    06/2019
                - Add datefromstr function
            V1.01	03/2019
				- Resolve bug
			V1    03/2019
				- Initialization
"""

def dtoday(format='%Y%m%d'):
    """
    Return date of today into a string
    :param format:  format of output string (ex: '%d-%m-%Y' == '23-02-2019')
                    'ts' = timestamp
    :return:        str date
    """
    if format.lower() == 'ts':
        strdate = time.time()
    else:
        ts = time.time()
        strdate = datetime.datetime.fromtimestamp(ts).strftime(format)
    return strdate


def dconvert(datein, fmtin, fmtout):
    """
    Convert date format into another
    :param datein:  input string of date or ts (int)
    :param fmtin:   input format (ex: '%Y-%m-%d %H:%M:%S' == '2019-02-23 23:15:55')
                    'ts' = timestamp
    :param fmtout:  output format (ex: '%Y%m%d' == '20190223')
                    'ts' = timestamp
    :return:        str date
    """
    if fmtin.lower() == 'ts':
        ts = datein
    else:
        ts = int(datetime.datetime.strptime(datein, fmtin).strftime('%s'))

    if fmtout.lower() == 'ts':
        strdate = ts
    else:
        strdate = datetime.datetime.fromtimestamp(ts).strftime(fmtout)
    return strdate


def datefromstr(strin):
    """
    Find a date into string (file or folder name) --> format == YYYYMMDD
    :param strin:   string
    :return:        date string (format YYYYMMDD)
    """
    info = strin.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('T', ' ')
    info = info.split()
    for i in info:
        if len(i) == 8:
            try:
                dato = int(i)
                tmp = i[0:2]
                if int(tmp) == 20: # After year 2000 minimum
                    return dato
            except:
                continue

    logging.warning(' No found date !')