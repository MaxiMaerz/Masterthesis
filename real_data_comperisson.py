import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from operator import itemgetter


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

def gaus(x, a, b, c):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2))


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


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

    clf_adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=50,
                                                           loss='linear', random_state=0)
    clf_extra_trees = ExtraTreesRegressor(n_estimators=8, random_state=0, max_depth=30)
    clf_random_forest = RandomForestRegressor(n_estimators=8, random_state=0, max_depth=30)

    clf_adaboost.fit(all_data_test.T, all_targets_test[0])
    predicted = clf_adaboost.predict(all_data_valid.T)

    clf_extra_trees.fit(all_data_test.T, all_targets_test[0])
    predicted_extra = clf_extra_trees.predict(all_data_valid.T)

    clf_random_forest.fit(all_data_test.T, all_targets_test[0])
    predicted_forest = clf_random_forest.predict(all_data_valid.T)


    delta_ada = all_targets_valid[0] - predicted
    delta_extra = all_targets_valid[0] - predicted_extra
    delta_forest = all_targets_valid[0] - predicted_forest
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
