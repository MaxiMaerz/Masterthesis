import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt


def test_get_standart_deviation_norm():
    norm = np.random.normal(0, 1, 10000)
    std = np.std(norm, ddof=1)
    std68, std95 = get_standart_deviation(norm)
    print(round(std68, 1), round(std, 1))
    assert round(std68, 1) == round(std,1),\
           'Sigma68 not matching Sigma68 in Numpy.Normal'
    assert round(std95, 1) == round(std * 2, 1),\
           'Sigma95 not matching Sigma95 in Numpy.Normal'


def test_get_standart_deviation_uniform():
    uniform = np.random.uniform(0, 1, 10000)
    std68, std95 = get_standart_deviation(uniform)
    assert round(std68, 2) == 0.34,\
           'Got Problems with uniform distribution'


def get_standart_deviation(data, add_data=None, method='simple'):
    """
    Robust Method to calculate Standart deviation

    It uses the 68-95-99.7 Rule to calculate the std 68 and std 95
    :param data: 1D - Array
    :return: std68 (float), std95 (float)
    """
    assert data.ndim == 1, 'Array not 1-dimensional!'
    assert data.dtype == float or data.dtype == int, 'Data-Type not understood'
    #assert method != 'full' and add_data is None, 'Need origional Data'+ method

    data_sort = np.sort(data, axis=None)
    mean_index = int(len(data_sort) / 2)
    data_sort -= np.mean(data_sort)

    p_std68 = data_sort[mean_index + int(0.341344746 * len(data_sort))]
    m_std68 = data_sort[mean_index - int(0.341344746 * len(data_sort))]
    std68 = (abs(p_std68) + abs(m_std68))/2

    p_std95 = data_sort[mean_index + int(0.47725 * len(data_sort))]
    m_std95 = data_sort[mean_index - int(0.47725 * len(data_sort))]
    std95 = (abs(p_std95) + abs(m_std95))/2
    result = 'Std_68: ' + str(round(std68, 3))+  ' Std_95: '+ str(round(std95, 3)) + ' Mean: ' + str(round(data_sort[mean_index], 3)) + '\n'
    vals = [std68, std95, data_sort[mean_index]]
    if method is 'simple':
        return std68, std95, data_sort[mean_index]

    if method is 'full':
        delta_res = abs(data / (1 + add_data))
        data_sort = np.sort(delta_res, axis=None)
        mean_index = int(len(data_sort) / 2)
        data_sort -= np.mean(data_sort)

        p_std68 = data_sort[mean_index + int(0.341344746 * len(data_sort))]
        m_std68 = data_sort[mean_index - int(0.341344746 * len(data_sort))]
        std68 = (abs(p_std68) + abs(m_std68))/2

        p_std95 = data_sort[mean_index + int(0.47725 * len(data_sort))]
        m_std95 = data_sort[mean_index - int(0.47725 * len(data_sort))]
        std95 = (abs(p_std95) + abs(m_std95))/2
        result += 'Resc_Std_68: ' + str(round(std68, 3))+  ' Resc_Std_95: '+ str(round(std95, 3)) + ' Resc_Mean: ' + str(round(data_sort[mean_index], 3))
        vals = np.vstack((vals, [std68, std95, data_sort[mean_index]]))
        return result, vals

def get_outliner_fraction(delta, validation):
    delta_res = abs(delta / (1 + validation))

    x = np.sum(delta_res > 0.15)/len(delta_res)
    return x


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


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def gaus(x, mean, std):
    """
    returns gaussian value for x Values

    :param x: Array to get the rigth range
    :param mean: Mean
    :param std: Standart - Deviation Sigma
    :return: fkt Gaussian shaped y - values
    """
    fkt = np.empty(len(x))
    for i in range(0, len(x)):
        fkt[i] = 1/(std * np.sqrt(2 * np.pi)) * np.exp(- ((x[i] - mean) ** 2) / (2 * std ** 2))
    return fkt


def report_feature_importance(feat, f_dic, plot=False):
    order = np.argsort(feat)[::-1]
    f_list = []
    string = ''
    for i in f_dic:
        string += str(i) + '. ' + f_dic[order[i-1]+1] + ' with ' + str(round(feat[order[i-1]], 4)) +'\n'
        f_list.append(f_dic[i])
    if plot is True:
        plt.subplots()
        ind = np.arange(len(feat))
        plt.bar(ind, feat)
        plt.xticks(rotation=70)
        plt.xticks(ind+0.5, f_list)
        plt.ylabel('Relative Feature Importance')
        plt.title('Normalized importance of feature')
        plt.savefig('Feature_Importance.png', dpi=250)
        plt.show()

        return string
    return string

#def report_all(data, org_data):
