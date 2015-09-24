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


def get_standart_deviation(data):
    """
    Robust Method to calculate Standart deviation

    It uses the 68-95-99.7 Rule to calculate the std
    :param data: 1D - Array
    :return: std (float)
    """
    assert data.ndim == 1, 'Array not 1-dimensional!'
    assert data.dtype == float or data.dtype == int, 'Data-Type not understood'

    data_sort = np.sort(data, axis=None)
    mean_index = np.searchsorted(data_sort, np.mean(data_sort))
    data_sort -= np.mean(data_sort)
    p_std = data_sort[mean_index + int(0.341344746 * len(data_sort))]
    m_std = data_sort[mean_index - int(0.341344746 * len(data_sort))]
    std = (abs(p_std) + abs(m_std))/2
    return std


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
    t = np.zeros((50001))
    x_t = np.zeros((50001))
    print(t)
    x_0 = [15, 23, 7]
    dt = 0.01
    i = 0
    solver = ode(lorentz_system)
    solver.set_initial_value(x_0, 0)
    solver.set_integrator('dopri5')
    '''
    while solver.successful() and solver.t < 500:
        solver.integrate(solver.t + dt)
        t[i] = solver.t
        x_t[i] = solver.y[0]
        i += 1
    '''
    sinus = np.sin((np.arange(0, 501, 0.01)/501*2*np.pi))
    for i in range(len(sinus)):
        sinus[i] += +np.random.normal(0, 100)/100
    sinus_train, sinus_test, time_train, time_test = train_test_split(sinus, np.arange(0, 501, 0.01)/501*2*np.pi, test_size=0.5)

    #sinus_train, sinus_test, time_train, time_test = train_test_split(x_t, t, test_size=0.2)
    print(time_train.shape, sinus_train.shape)

    '''
    clf = DecisionTreeRegressor()
    param_dist = {"max_depth": np.arange(3,100,1)}
    random_search = GridSearchCV(clf, param_grid=param_dist)
    random_search.fit(np.expand_dims(time_train,axis=1), sinus_train)
    report(random_search.grid_scores_)
    exit()
    '''

    clf_adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=49), n_estimators=50,
                                                           loss='exponential', random_state=0)
    clf_extra_trees = ExtraTreesRegressor(n_estimators=50, random_state=0, max_depth=49)
    clf_random_forest = RandomForestRegressor(n_estimators=50, random_state=0, max_depth=49)

    clf_adaboost.fit(np.expand_dims(time_train,axis=1), sinus_train)
    predicted = clf_adaboost.predict(np.expand_dims(time_test, axis=1))

    clf_extra_trees.fit(np.expand_dims(time_train,axis=1), sinus_train)
    predicted_extra = clf_extra_trees.predict(np.expand_dims(time_test, axis=1))

    clf_random_forest.fit(np.expand_dims(time_train,axis=1), sinus_train)
    predicted_forest = clf_random_forest.predict(np.expand_dims(time_test, axis=1))

    print(time_train.shape, sinus_train.shape)
    delta_ada = sinus_test - predicted
    delta_extra = sinus_test - predicted_extra
    delta_forest = sinus_test - predicted_forest
    std_ada = get_standart_deviation(delta_ada)
    std_extra = get_standart_deviation(delta_extra)
    std_forest = get_standart_deviation(delta_forest)

    plt.hist(delta_ada, bins=150, color='g', label='Adaboost '+str(np.round(std_ada,4)))
    plt.hist(delta_extra, bins=150, color='b', label='Extra_Trees '+str(np.round(std_extra,4)))
    plt.hist(delta_forest, bins=150, color='r', label='Random_Forest '+str(np.round(std_forest,4)))
    title = "Compare adaboost, extra_tree and Random_Forests"
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()
    plt.close()
    '''
    plt.plot(time_test, sinus_test, ',', color='red')
    plt.plot(time_test, predicted, ',', color='b')
    plt.show()
    '''
