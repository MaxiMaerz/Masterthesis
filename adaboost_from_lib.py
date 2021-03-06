import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from astropy.io import fits
from time import time
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from scipy import optimize


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
    start = time()
    clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9), n_estimators=300, loss='exponential')
    clf.fit(all_data_test.T, all_targets_test[0])
    feature_importance = clf.feature_importances_
    stop = time()
    predicted = clf.predict(all_data_valid.T)

    delta = all_targets_valid[0] - predicted
    mean = np.mean(delta)
    delta_res = delta / (1 + all_targets_valid[0])


    data = plt.hist(delta, bins=150, normed=True, color='g')
    x = [0.5 * (data[1][i] + data[1][i+1]) for i in range(len(data[1])-1)]
    y = data[0]
    popt, pcov = optimize.curve_fit(gaus, x, y)
    perr = np.sqrt(np.diag(pcov))
    scale_factor = popt[0]
    mu_delta = popt[1]
    sigma_delta = popt[2]
    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = gaus(x_fit, *popt)
    plt.plot(x_fit, y_fit, lw=1, color="r")
    title = "Fit results: a = %.f, mu = %.2f,  std = %.2f \n" \
            "with Errors: o_a = %.3f, o_mu = %.3f, o_std = %.3f" % (scale_factor, mu_delta, sigma_delta,
                                                                 perr[0], perr[1], perr[2])
    plt.title(title)
    plt.savefig(Path + 'Delta.png')
    plt.clf()
    plt.close()

    data = plt.hist(delta_res, bins=150, normed=True, color='g')
    x = [0.5 * (data[1][i] + data[1][i+1]) for i in range(len(data[1])-1)]
    y = data[0]
    popt, pcov = optimize.curve_fit(gaus, x, y)
    perr = np.sqrt(np.diag(pcov))
    scale_factor = popt[0]
    mu_delta = popt[1]
    sigma_delta = popt[2]
    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = gaus(x_fit, *popt)
    plt.plot(x_fit, y_fit, lw=1, color="r")
    title = "Fit results: a = %.f, mu = %.2f,  std = %.2f \n" \
            "with Errors: o_a = %.3f, o_mu = %.3f, o_std = %.3f" % (scale_factor, mu_delta, sigma_delta,
                                                                 perr[0], perr[1], perr[2])
    plt.title(title)
    plt.savefig(Path + 'Delta_scaled.png')
    plt.clf()
    plt.close()



    print(mean_squared_error(all_targets_valid[0], predicted, sample_weight=None), abs(start-stop))
    print(feature_importance)