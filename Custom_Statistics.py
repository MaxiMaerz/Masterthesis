import numpy as np


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


def get_standart_deviation(data):
    """
    Robust Method to calculate Standart deviation

    It uses the 68-95-99.7 Rule to calculate the std 68 and std 95
    :param data: 1D - Array
    :return: std68 (float), std95 (float)
    """
    assert data.ndim == 1, 'Array not 1-dimensional!'
    assert data.dtype == float or data.dtype == int, 'Data-Type not understood'

    data_sort = np.sort(data, axis=None)
    mean_index = int(len(data_sort) / 2)
    data_sort -= np.mean(data_sort)

    p_std68 = data_sort[mean_index + int(0.341344746 * len(data_sort))]
    m_std68 = data_sort[mean_index - int(0.341344746 * len(data_sort))]
    std68 = (abs(p_std68) + abs(m_std68))/2

    p_std95 = data_sort[mean_index + int(0.47725 * len(data_sort))]
    m_std95 = data_sort[mean_index - int(0.47725 * len(data_sort))]
    std95 = (abs(p_std95) + abs(m_std95))/2

    return std68, std95, data_sort[mean_index]


def get_outliner_fraction(estimate, validation):
    delta = estimate - validation
    delta_res = delta / (1 + validation)

    x = 0
    for i in range(0, len(estimate)):
        if delta_res[i] > 0.15:
            x += 1

    outliner_fraction = x / len(estimate)

    return outliner_fraction


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
    print(std, mean,'!!!!')
    fkt = np.empty(len(x))
    for i in range(0, len(x)):
        fkt[i] = 1/(std * np.sqrt(2 * np.pi)) * np.exp(- ((x[i] - mean) ** 2) / (2 * std ** 2))
    return fkt
