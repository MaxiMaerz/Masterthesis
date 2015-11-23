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
if __name__ == '__main__':
    '''We want all file opening stuff here o avoid confusion'''
    Path = '/home/maxi/data/'

    '''Opening data and config files'''
    hdulist_train = fits.open(Path + 'y1a1_stripe82_train_subset.fits')
    hdulist_valid = fits.open(Path + 'y1a1_stripe82_valid_subset.fits')
    feature_conf = [line.strip() for line in open(Path + '/fields.conf', 'r')]

    '''Get config aa'''
    feature_index, feature_dic, target_index, target_dic = select_features(feature_conf, hdulist_train)
    feature_index_valid, feature_dic_valid, target_index_valid, target_dic_valid = select_features(feature_conf,
                                                                                                   hdulist_valid)

    '''Get data'''
    all_data_train, all_targets_train = generate_array(hdulist_train,
                                                       feature_index,
                                                       target_index)
    all_data_valid, all_targets_valid = generate_array(hdulist_valid,
                                                       feature_index_valid,
                                                       target_index_valid)

    # generate more features
    







    # start = time()
    # clf_adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=40), n_estimators=20,
    #                                  loss='exponential', random_state=0)
    #
    # clf_adaboost.fit(all_data_train.T, all_targets_train[0])
    # predicted = clf_adaboost.predict(all_data_valid.T)
    #
    # delta_adaboost = all_targets_valid[0] - predicted
    # print(get_standart_deviation(delta_adaboost))
    # stop = time()
    # print('Execution Time = ', stop-start)
    start = time()
    tree_para = {'max_depth':20}
    clf = AdaboostRegression(n_estimators=10, loss_fkt='exponential', tree_parameters=tree_para)
    estimator, weights = clf.fit(all_data_train.T, all_targets_train[0].T)

    prediction = clf.predict(all_data_valid.T, estimator, weights)

    delta = all_targets_valid[0] - prediction
    plt.hist(delta, bins=100)
    plt.show()
    print(get_standart_deviation(delta))
    stop = time()
    print('Execution Time = ', stop-start)
