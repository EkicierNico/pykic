import time, datetime
from datetime import date

"""
Dates utilities
Author:     Nicolas EKICIER
Release:    V1    03/2019
"""

def dtoday(format='%Y%m%d'):
    """
    Return date of today into a string
    :param format:  format of output string (ex: '%d-%m-%Y' == '23-02-2019')
    :return:        str date
    """
    ts = time.time()
    strdate = date.fromtimestamp(ts).strftime(format)
    return strdate


def dconvert(datein, fmtin, fmtout):
    """
    Convert date format into another
    :param datein:  input string of date
    :param fmtin:   input format (ex: '%Y-%m-%d %H:%M:%S' == '2019-02-23 23:15:55')
    :param fmtout:  output format (ex: '%Y%m%d' == '20190223')
    :return:        str date
    """
    ts = datetime.datetime.strptime(datein, fmtin).strftime('%s')
    strdate = date.fromtimestamp(ts).strftime(fmtout)
    return strdate