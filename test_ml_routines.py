import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn import tree
import os
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from operator import itemgetter
from Custom_Statistics import *
from Import_Astrofits import *
from astropy.io import fits

def lorentz_system(t, q):
    x = q[0]
    y = q[1]
    z = q[2]
    s=10
    r=28
    b=8/3
    f_x = [s*(y-x), x*(r-z)-y, x*y-b*z]
    return f_x


def harmonic_osz(t, q):
    x = q[0]
    y = q[1]
    w_0 = 2.
    f_x = [x, -w_0*w_0*y]
    return f_x

def auto_solver(func, steps, x_0, args_in):
    '''
    :param func: function to integrate
    Note: q and t is switched !!!
    :param steps: array of timesteps
    :param x_0: start-values
    :param args_in: Params for ODE // not ready yet just enter 0
    :return: Array of solved points allong timesteps
    '''
    y = odeint(func, x_0, steps)#, args=(args_in,)
    return y


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


if __name__ == '__main__':
    '''
    t = np.zeros((50001))
    x_t = np.zeros((50001))
    print(t)
    x_0 = [15, 23, 7]
    dt = 0.01
    i = 0
    solver = ode(lorentz_system)
    solver.set_initial_value(x_0, 0)
    solver.set_integrator('dopri5')

    while solver.successful() and solver.t < 500:
        solver.integrate(solver.t + dt)
        t[i] = solver.t
        x_t[i] = solver.y[0]
        i += 1

    sinus = np.sin((np.arange(0, 501, 0.01)/501*2*np.pi))
    for i in range(len(sinus)):
        sinus[i] += +np.random.normal(0, 100)/100
    sinus_train, sinus_test, time_train, time_test = train_test_split(sinus, np.arange(0, 501, 0.01)/501*2*np.pi, test_size=0.5)

    #sinus_train, sinus_test, time_train, time_test = train_test_split(x_t, t, test_size=0.2)
    print(time_train.shape, sinus_train.shape)




    clf = DecisionTreeRegressor()
    param_dist = {"max_depth": np.arange(3,100,1)}
    random_search = GridSearchCV(clf, param_grid=param_dist)
    random_search.fit(np.expand_dims(time_train,axis=1), sinus_train)
    report(random_search.grid_scores_)
    exit()
    '''


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
    print(len(all_data_test.T), len(all_targets_test[0]))
    std_list = []
    '''Start the Main'''
    clf_adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=25), n_estimators=100,
                                     loss='exponential', random_state=0)
    #clf_extra_trees = ExtraTreesRegressor(n_estimators=100, random_state=0, max_depth=25)
    #clf_random_forest = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=25)

    clf_adaboost.fit(all_data_test.T, all_targets_test[0])
    predicted = clf_adaboost.predict(all_data_valid.T)

    #clf_extra_trees.fit(all_data_test.T, all_targets_test[0])
    #predicted_extra = clf_extra_trees.predict(all_data_valid.T)

    #clf_random_forest.fit(all_data_test.T, all_targets_test[0])
    #predicted_forest = clf_random_forest.predict(all_data_valid.T)

    delta_ada = (all_targets_valid[0] - predicted) / (1 + all_targets_valid[0])
    #delta_extra = all_targets_valid[0] - predicted_extra
    #delta_forest = all_targets_valid[0] - predicted_forest
    std_ada, std95_ada, outliner_ada = get_standart_deviation(delta_ada)
    std_list.append(std_ada)
    print(std_ada, std95_ada, outliner_ada)

    feature_list =[]
    for i in range(0, len(feature_index)):
        feature_list.append(feature_dic[feature_index[i]])
    N = len(feature_index)
    ind = np.arange(N)    # the x locations for the groups
    width = 1       # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, clf_adaboost.feature_importances_, width, color='r')
    plt.ylabel('Relative Feature Importance')
    plt.title('Normalized importance of feature')
    plt.xticks(rotation=70)
    plt.xticks(ind, feature_list)
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    #std_extra, std95_extra, outliner_extra = get_standart_deviation(delta_extra)
    #std_forest, std95_forest, outliner_forest = get_standart_deviation(delta_forest)

    with open(Path+'performance.txt', 'w') as file:
        for item in std_list:
            file.write("{}\n".format(item))
    """
    plt.hist(delta_ada, bins=150, color='g', label='Adaboost '+str(np.round(std_ada,4)))
    plt.hist(delta_extra, bins=150, color='b', label='Extra_Trees '+str(np.round(std_extra,4)))
    plt.hist(delta_forest, bins=150, color='r', label='Random_Forest '+str(np.round(std_forest,4)))
    title = "Compare adaboost, extra_tree and Random_Forests"
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()
    plt.close()
    print('Adaboost: ', std_ada, std95_ada, outliner_ada, '\n',
          'Extra_trees: ', std_extra, std95_extra, outliner_extra, '\n',
          'Random_forest: ', std_forest, std95_forest, outliner_extra)
    print(clf_adaboost.feature_importances_)
    """
