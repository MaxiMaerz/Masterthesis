from boosted_regression import AdaboostRegression
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
import sys
from Import_Astrofits import *
from custom_adaboost import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import warnings
warnings.filterwarnings("ignore")
import kcorrect
import os
import pidly

if __name__ == '__main__':
    '''We want all file opening stuff here o avoid confusion'''
    Path = '/home/maxi/data/small_set/'
    #Path2 = '/home/maxi/data/'

    '''Opening data and config files'''
    hdulist_train = fits.open(Path + 'PHAT1_TRAINING.fits')
    hdulist_valid = fits.open(Path + 'PHAT1_TESTSET_MNPLz.fit')
    feature_conf = [line.strip() for line in open(Path + '/fields.conf', 'r')]
    feature_conf_2 = [line.strip() for line in open(Path + '/fields2.conf', 'r')]


    '''Get config aa'''
    feature_index, feature_dic, target_index, target_dic = select_features(feature_conf, hdulist_train)
    feature_index_valid, feature_dic_valid, target_index_valid, target_dic_valid = select_features(feature_conf_2, hdulist_valid)

    '''Get data'''
    all_data_train, all_targets_train = generate_array(hdulist_train,
                                                     feature_index,
                                                     target_index)
    all_data_valid, all_targets_valid = generate_array(hdulist_valid,
                                                       feature_index_valid,
                                                       target_index_valid)

    # get some new z-shifts

    artificial_z_shift = all_targets_train[0]
    for i in range(10):
        artificial_z_shift = np.hstack((artificial_z_shift, all_targets_train[0]))

    random_direction = np.random.normal(0, 0.1, len(artificial_z_shift))
    artificial_z_shift += random_direction

    print(all_data_train.shape)
    mag_data = all_data_train
    err_data = np.zeros(all_data_train.shape)
    print(err_data.shape)

    z_old = all_targets_train[0].reshape(-1, 1)
    z_new = artificial_z_shift.reshape(-1, 1)
    print(z_old.shape)
    idl = pidly.IDL('//usr/local/bin/idl')
    chi2 = 'chi2'

    print(idl.func('reform', range(4), 2, 2))
    #print(idl.func('kcorrect', [1., 4.78, 10.96, 14.45, 19.05],  [1100., 28., 7.7, 4.4, 2.5], 0.03, 'kcorrect', band_shift=0.1, chi2=chi2))
    #print(idl('kcorrect, [1., 4.78, 10.96, 14.45, 19.05],[1100., 28., 7.7, 4.4, 2.5], 0.03, kcorrect, band_shift=0.1, chi2=chi2'))

    try:
        print("running idl")
        output = idl.func('sdss2bands', z_old, z_new, mag=mag_data, err=err_data, filterlist=['sdss_u0.par', 'sdss_g0.par', 'sdss_r0.par', 'sdss_i0.par', 'sdss_z0.par'])
    except:
        output = None
    idl.close()