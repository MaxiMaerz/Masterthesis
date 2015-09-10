from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
import os
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
from operator import itemgetter

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def generate_array(hdulist, features, targets):
    """
    generates a numpy array from a hdu_list from astropy
    it skips all strings and keeps just number like coloums
    :param hdulist: Astropy HDU_List object
    :param features: Desired features
    :param targets: Desired teargets
    :return: data_float array with targets, targets_float array with features
    """
    """
    #To do:
        -get targets and features via .conf file
            -> use     field_names = hdulist_test[1].columns.names
    """

    '''extract data'''
    astro_data = hdulist[1].data

    '''get all float like and feature matching data'''
    data_float = np.squeeze(np.array([astro_data.field(0)]))
    for x in range(len(astro_data[0])):
        if isinstance(astro_data.field(x)[1], (int, float, complex)) is True\
                        and x not in targets and x in features:
            data_float = np.vstack((data_float, np.squeeze(np.array([astro_data.field(x)]))))

    '''get all and target matching data'''
    targets_float = np.squeeze(np.array(astro_data.field(targets[0])))
    for x in range(len(targets)):
        targets_float = np.vstack((targets_float, np.squeeze(np.array(astro_data.field(targets[x])))))
        #print(hdulist_test[1].columns.names[targets[x]])

    '''return'''
    return data_float, targets_float

if __name__ == '__main__':

    #opening files#
    hdulist_test = fits.open('/home/maxi/data/y1a1_stripe82_train_subset.fits')
    hdulist_valid = fits.open('/home/maxi/data/y1a1_stripe82_valid_subset.fits')
    all_data_test, all_targets_test = generate_array(hdulist_test, np.arange(0,
                                                                             len(hdulist_test[1].data[0]), 1),
                                                                            [20, 32])

    #print(all_targets_test)
    #learning#
    clf = tree.DecisionTreeRegressor(max_depth=20)
    clf.fit(all_data_test.T, all_targets_test[0])
    #use on Model for validation#
    all_data_valid, all_targets_valid = generate_array(hdulist_valid, np.arange(0,
                                                                                len(hdulist_test[1].data[0]), 1),
                                                                               [20, 32])

    Z_predicted = clf.predict(all_data_valid.T)
    delta = np.empty(len((all_data_test[0])))
    #calculate residua#
    for x in range(len(all_data_test[0])):
        delta[x] = Z_predicted[x]-all_targets_valid[0][x]
    #print(all_targets_valid[2])
    #plt.show()
    '''Try some Grid-search'''
    #start = time()
    #params_to_explore = {'max_depth': np.arange(1, 25, 1),
    #                    'max_features': np.arange(1, 3, 1)
    #                     }
    #grid_search = GridSearchCV(clf, param_grid=params_to_explore)
    #grid_search.fit(all_data_test.T, all_targets_test[0])

    #stop = time()
    #report(grid_search.grid_scores_)
    #print(start-stop)

    #calculate statistical measures#
    standard_deviation_dt = np.std(delta)
    variance_dt = np.var(delta)
    mean_dt = np.mean(abs(delta))
    print(' sigma= ' + str(standard_deviation_dt) + '\n',
          'Variance =' + str(variance_dt) + '\n',
          'Mean= ' + str(mean_dt) + '\n',
           )

    '''plotting'''
    hist, bins = np.histogram(delta, bins = 100)
    hist_2, bins_2 = np.histogram(all_targets_valid[2][all_targets_valid[2] != 99.0], bins = 50)
    width = 1. * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='g')
    #plt.bar(center, hist_2, align='center', width=width, color='r')
    plt.show()
