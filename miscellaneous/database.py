import pandas as pd
from sqlalchemy import create_engine

"""
Database utilities
Author:     Nicolas EKICIER
Release:    V1    03/2019
"""

def buildproto(protocol, user, passwd, host, dbname):
    """
    Build a string protocol to get db
    :param protocol:    ex: 'postgresql'
    :param user:        user
    :param passwd:      password
    :param host:        pi address
    :param dbname:      name od database
    :return:            string
    """
    strproto = '{0:s}://{1:s}:{2:s}@{3:s}/{4:s}'.format(protocol, user, passwd, host, dbname)
    return strproto


def getdb(strproto, query):
    """
    Get data from db
    :param strproto:    string protocol (ex: 'postgresql://master:userir@172.30.16.173/ais')
    :param query:       query (string). ex : "SELECT * FROM table"
    :return:            pandas dataframe
    """
    conx = create_engine(strproto)
    dataset = pd.read_sql_query(query, conx)
    return dataset


def getdbtable(strproto):
    """
    Get list of tables in the database
    :param strproto:    string protocol (ex: 'postgresql://master:userir@172.30.16.173/ais')
    :return:
    """
    conx = create_engine(strproto)
    table = conx.table_names()
    conx.close()
    return table