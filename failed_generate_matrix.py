__author__ = 'maxi'
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
import os
from sklearn.externals.six import StringIO

def export_data():
    """
    Read the Data of the File
    :return: Data, Header in HDULIST Format
    """

    hdulist = fits.open('/home/maxi/data/y1a1_stripe82_train_subset.fits')
    tbdata = hdulist[1].data
    header = hdulist[0].header
    return tbdata, header



if __name__ == '__main__':
    """Export the Data"""
    astro_data, astro_header = export_data()
    '''initialize first array'''
    '''
    hdulist = fits.open('/home/maxi/data/y1a1_stripe82_train_subset.fits')
    data_types = ''
    for x in range(len(astro_data[0])):
        data_types += str((hdulist[1].columns[x].dtype)) +','
    data_types = data_types[:-1]
    all_data = np.zeros((len(astro_data[0]), len(astro_data.field(0))), dtype=(data_types))

    for x in range(len(astro_data[0])):
        temp = np.array([astro_data.field(x)])
        print(hdulist[1].columns[x].dtype)
        print(all_data[1][x].dtype)
        all_data[x] = temp
        print(x)
'''
    hdulist = fits.open('/home/maxi/data/y1a1_stripe82_train_subset.fits')
    data = np.array([astro_data.field(0)], dtype=hdulist[1].columns[0].dtype)
    data_char = np.array([astro_data.field(0)], dtype='a')
    y = 0

    '''Build two arrays one for float one for char'''
    for x in range(len(astro_data[0])):

        if hdulist[1].columns[x].dtype == 'float64' or hdulist[1].columns[x].dtype == 'int64' \
                or hdulist[1].columns[x].dtype == 'int16':
            temp = np.array([astro_data.field(x)], dtype='float64')
            data_float = np.hstack((data, temp))

        if hdulist[1].columns[x].dtype != 'float64' and hdulist[1].columns[x].dtype != 'int64' \
                and hdulist[1].columns[x].dtype != 'int16':
            temp = np.array([astro_data.field(x)], dtype=hdulist[1].columns[x].dtype)
            if y == 0:
                data_char = temp
                y = 1
            data_char = np.hstack((data_char, temp))

    all_data = np.concatenate((data_float, data_char), axis=1)
    print(repr(data_float.nbytes))
    print(repr(data_char.nbytes))
    print(repr(all_data.nbytes))



    '''Select Targets and values'''


