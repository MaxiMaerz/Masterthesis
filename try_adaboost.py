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

    '''Launch Grid Search to get optimal Parameters for each tree'''
    if False:
        clf = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100, loss='exponential', random_state=0)
        params_to_explore = {'DecisionTreeRegressor(max_depth=)': np.arange(10, 30, 1)}
        grid_search = GridSearchCV(clf, param_grid=params_to_explore)
        grid_search.fit(all_data_test.T, all_targets_test[0])

        report(grid_search.grid_scores_)
        sys.exit('Stop')

    '''Generate List of Estimators'''
    print(len(all_data_test[:,]))
    estimator_list = []
    hyper_parameter = 0
    mse_list = []
    start = time()
    n_estimators = 100
    for iter in range(0, n_estimators):
        '''waste computing power for loading bar'''
        step = n_estimators/100
        if iter%step == 0:
            print('\rTraining: %s (%d%%)' % ("|"*(int(iter/(3*step))), iter/step), end="")
            sys.stdout.flush()
        '''Start the tarining'''
        '''Get a random Part of the Data'''
        random_slice_data, random_slice_target = choose_random_data(all_data_test,
                                                                    all_targets_test,
                                                                    all_targets_test[-1],
                                                                    3000)

        '''Append new Estimator to List'''
        estimator, error_array, mean_s_error = build_weighted_tree(random_slice_data.T,
                                                                   random_slice_target,
                                                                   25)

        estimator_list.append(estimator)
        #-> pop estimators which are random (error > 0.5)

        mse_list.append(mean_s_error)

        '''update weights'''
        update_weights(error_array, all_targets_test[-1])

    stop = time()





    print('\nExecution time : ' + str(np.round(stop-start, 3)) + ' seconds' + '\n'
             'at ' +str(iter)+ ' iteration')
    '''Test the estimator'''
    predicted = np.zeros(len(all_data_valid[0]))
    mse_list_ret = np.empty(len(mse_list))

    for i in range(0, len(mse_list)):
        mse_list_ret[i] = 1. / mse_list[i]
    normalized_mse = normalize(mse_list_ret)
    list_min = []
    list_max = []
    list_all_features = []
    for est in range(0, len(estimator_list)):
        '''Predict and weight by mse'''
        predicted += estimator_list[est].predict(all_data_valid.T) * normalized_mse[est]
        list_all_features.append(estimator_list[est].feature_importances_)


    '''Get some statistics'''
    delta = all_targets_valid[0] - predicted
    delta_res = delta / (1 + all_targets_valid[0])

    sigma68_delta, sigma95_delta, mean_delta = get_standart_deviation(delta)
    sigma68_delta_res, sigma95_delta_res, mean_delta_res = get_standart_deviation(delta_res)

    outliners = get_outliner_fraction(predicted, all_targets_valid[0])

    '''Feature importance'''

    feature_list =[]
    importance = np.sum(list_all_features, axis=0)
    for i in range(0, len(feature_index)):
        feature_list.append(feature_dic[feature_index[i]])
    print(len(importance))
    print(len(feature_list))
    N = len(feature_index)
    ind = np.arange(N)    # the x locations for the groups
    width = 1       # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, importance/sum(importance), width, color='r')
    plt.ylabel('Relative Feature Importance')
    plt.title('Normalized importance of feature')
    plt.xticks(rotation=70)
    plt.xticks(ind, feature_list)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(Path + 'Importance.png')
    plt.clf()
    plt.close()


    '''Plotting histograms'''
    data = plt.hist(delta, bins=150, normed=True, color='g')
    x = [0.5 * (data[1][i] + data[1][i+1]) for i in range(len(data[1])-1)]
    y = gaus(x, mean_delta, sigma68_delta)

    plt.plot(x, y, lw=1, color="r", label='Gausian with sigma_68 = %.3f \n'
                                          'and sigma_95 = %.3f \n'
                                          'Outliner Rate of = %.3f' % (sigma68_delta, sigma95_delta, outliners))
    title = 'Deviation of Estimated Values from Validation Values \n ' \
            'The Parameters of the Gausian were calculate using 68-95-99-Rule'

    plt.title(title)
    plt.legend()
    plt.savefig(Path + 'Delta.png')
    plt.clf()
    plt.close()



    data = plt.hist(delta_res, bins=150, normed=True, color='g')
    x = [0.5 * (data[1][i] + data[1][i+1]) for i in range(len(data[1])-1)]
    y = gaus(x, mean_delta_res, sigma68_delta_res)

    plt.plot(x, y, lw=1, color="r", label='Gausian with sigma_68 = %.3f \n'
                                          'and sigma_95 = %.3f \n'
                                          'Outliner Rate of = %.3f' % (sigma68_delta_res, sigma95_delta_res, outliners))
    title = 'Deviation of Estimated Values from Validation Values \n ' \
            'The Parameters of the Gausian were calculate using 68-95-99-Rule'

    plt.title(title)
    plt.legend()
    plt.savefig(Path + 'Delta_res.png')
    plt.clf()
    plt.close()


    '''
    plt.grid(True)
    plt.plot([0, 1.4], [0, 1.4], 'k-', lw=2)
    plt.plot(predicted, all_targets_valid[0], ',')
    plt.savefig(Path + 'Scatter_pred_real.png')
    plt.clf()
    plt.close()
    plt.grid()
    plt.plot(mse_list)
    plt.savefig(Path + 'mse.png')
    plt.clf()
    plt.close()
    '''