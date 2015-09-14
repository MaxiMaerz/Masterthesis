from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
import os
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
from operator import itemgetter
from sklearn.metrics import mean_squared_error


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
    for x in range(0, len(astro_data[0])):
        if isinstance(astro_data.field(x)[1], (int, float, complex)) is True\
                        and x not in targets and x in features:
            data_float = np.vstack((data_float, np.squeeze(np.array([astro_data.field(x)]))))
    '''get all and target matching data'''
    targets_float = np.squeeze(np.array(astro_data.field(targets[0])))
    for x in range(len(targets)):
        targets_float = np.vstack((targets_float, np.squeeze(np.array(astro_data.field(targets[x])))))
        print('Selected Feature: ' + hdulist_test[1].columns.names[targets[x]])
    '''return'''
    return data_float, targets_float


def select_features(features_list, data):
    """
    Function extrats features and targets from config file

    -> has to be testet  for stability
    :param features_list: String Objects from config file
    :param data: HDU_List from Astropy
    :return: Numpy int arrays with indexes and dictonaries if needed
    """

    '''Initalize arrays'''
    f_index = []
    f_dic = {}
    t_index = []
    t_dic = {}

    '''
    double loop over both coloum headings and config
     -> very ugly, if time improve
    '''
    for x in range(0, len(data[1].columns.names)):
        for y in range(0, len(features_list)):

            if data[1].columns.names[x] == features_list[y]\
                    and features_list[y][:1] != '!' \
                    and features_list[y][:1] != '#':
                f_index = np.append(f_index, x)
                f_dic.update({features_list[y]: x})

            if features_list[y][:1] == '!' \
                    and features_list[y][1:] == data[1].columns.names[x]:
                t_index = np.append(t_index, x)
                t_dic.update({features_list[y][1:]: x})

    return f_index.astype(int), f_dic, t_index.astype(int), t_dic


def normalize(vector):
    """
    normalize a numpy vector with [i] = [i]/sum[i]

    :param vector: numpy vector
    :return: normalized vector
    """
    if np.sum(vector) != 1.:
        normalized_vector = vector / np.sum(vector)
        return normalized_vector
    else:
        return vector


def choose_random_data(data, target, weight, length):
    """
    Get a Random Party of Data with size: length

    :param data: np.array containing data
    :param target:  np.array containing targets
    :param length: integer
    :return: random target and data slice of original data.
    The Order remains(data[1] corespons to target[1]
    """

    '''We need to generate an 1D index array to use np.random.choice'''
    index_arr = np.arange(0, len(data[1]))

    '''Normalize probabilties'''
    norm_weight = normalize(weight)

    '''No we draw with respect to weight'''
    random_index_arr = np.random.choice(index_arr, size=int(length), replace=True, p=norm_weight)

    random_data = data[:, random_index_arr]
    random_target = target[:, random_index_arr]
    return random_data, random_target


def build_weighted_tree(data, target, parameter=0):
    """
    We construct a Tree via skipi_tree_regressor and calculate the MSE and update weights

    Note: we use train_test_split to get a valisation sample
    :param data: np.array
    :param target: np.array
    :param parameter: Hyperparameters for the tree (not implemented)

    :return:updated weights
    """

    '''Split data  -> not possible we need to track the index for weight update.....'''
    x_train, x_test, y_train, y_test = train_test_split(data, target[0], test_size=0.5)

    '''Build tree slice last coloumn (its just index)'''
    clf = tree.DecisionTreeRegressor()
    clf.fit(x_train[:, 0:-1], y_train)
    predicted = clf.predict(x_test[:, 0:-1])

    '''Calculate Mean squared Error'''
    mse = mean_squared_error(y_test, predicted, sample_weight=None)

    '''Calculate Distance between predicted and real'''
    delta = abs(predicted-y_test)

    '''Calculate the Error'''
    error = delta / np.amax(delta)
    error = np.vstack((error, x_test[:,-1].T))
    if np.amax(delta) == 0:
        print(delta)

    return clf, error, mse


def update_weights(Error, weights):
    """
    Updates weights with an exponential weight function.

    :param Error: Error array, with Error(first column) and index(second column)
    :param weights: weight vector
    :return: nothing, we change the vector here
    """

    '''Calculate the Average Error L_bar

        L_bar = sum_i^N L_i * w_i
        where L_i is Error at ith element and w_i is the weight
    '''
    L_bar = 0.
    weights_normal = normalize(weights)
    for i in range(len(Error[0])):
        L_bar += Error[0][i] * weights_normal[Error[-1][i]]
    betha = abs(L_bar/(1-L_bar))
    for i in range(len(Error[-1])):
        weights[Error[-1][i]] = betha ** (1 - Error[0][i]) * weights_normal[Error[-1][i]]


if __name__ == '__main__':
    '''We want all file opening stuff here o avoid confusion'''
    Path = '/home/maxi/data/'

    '''Opening data and config files'''
    hdulist_test = fits.open(Path + 'y1a1_stripe82_train_subset.fits')
    hdulist_valid = fits.open(Path + 'y1a1_stripe82_valid_subset.fits')
    feature_conf = [line.strip() for line in open(Path + '/fields.conf', 'r')]

    '''Get config aa'''
    feature_index, feature_dic, target_index, target_dic = select_features(feature_conf, hdulist_test)

    '''Get data'''
    all_data_test, all_targets_test = generate_array(hdulist_test,
                                                     feature_index,
                                                     target_index)
    all_data_valid, all_targets_valid = generate_array(hdulist_valid,
                                                       feature_index,
                                                       target_index)

    '''Start the Main'''


    '''Add the index to track'''
    all_data_test = np.vstack((all_data_test,
                                  np.arange(0, len(all_data_test[0]))))

    '''Add weight coloumn'''
    all_targets_test = np.vstack((all_targets_test,
                                  normalize(np.ones((1, len(all_targets_test[0])),
                                  dtype='float64'))))
    '''Generate List of Estimators'''

    estimator_list = []
    hyper_parameter = 0
    mse_list = []
    start = time()
    for iter in range(0, 3000):
        print(iter)
        '''Get a random Part of the Data'''
        random_slice_data, random_slice_target = choose_random_data(all_data_test,
                                                                    all_targets_test,
                                                                    all_targets_test[-1],
                                                                    300)

        '''Append new Estimator to List'''
        estimator, error_array, mean_s_error = build_weighted_tree(random_slice_data.T,
                                  random_slice_target,
                                  hyper_parameter)

        estimator_list.append(estimator)
        mse_list.append(mean_s_error)

        '''update weights'''
        update_weights(error_array, all_targets_test[-1])

    stop = time()

    print('Execution time : ' + str(np.round(stop-start, 3)) + ' seconds' + '\n'
              'at ' +str(iter)+ ' iteration')
    '''Test the estimator'''
    predicted = np.zeros(len(all_data_valid[0]))
    mse_list_ret = np.empty(len(mse_list))

    for i in range(0, len(mse_list)):
        mse_list_ret[i] = 1. / mse_list[i]
    normalized_mse = normalize(mse_list_ret)
    list_min = []
    list_max = []
    for est in range(len(estimator_list)):
        '''Predict and weight by mse'''
        predicted += estimator_list[est].predict(all_data_valid.T) * normalized_mse[est]
        list_min.append(estimator_list[est].feature_importances_.argmin())
        list_max.append(estimator_list[est].feature_importances_.argmax())


    '''Get some statistics'''
    print(mean_squared_error(all_targets_valid[0], predicted, sample_weight=None))
    Delta = all_targets_valid[0] - predicted
    mean = np.mean(Delta)
    Delta_res = Delta / (1 + all_targets_valid[0])
    print(#'Delta = ' + str(np.sum(abs(Delta))/) + '\n'
          'Mean = ' + str(mean) + '\n')
          #'Delta_res = ' + str(Delta_res) + '\n')

    hist, bins = np.histogram(list_max, bins = 18)
    hist2, bins2 = np.histogram(list_min, bins = 18)
    width = 1. * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='g')
    plt.bar(center, hist2, align='center', width=width, color='r')
    plt.show()
