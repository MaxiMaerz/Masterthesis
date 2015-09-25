import numpy as np


def test_get_standart_deviation_norm():
    norm = np.random.normal(0, 1, 10000)
    std = np.std(norm, ddof=1)
    print(round(get_standart_deviation(norm), 1), round(std,1))
    assert round(get_standart_deviation(norm), 1) == round(std,1),\
           'Got Problems with normal distribution'


def test_get_standart_deviation_uniform():
    norm = np.random.uniform(0, 1, 10000)
    print(round(get_standart_deviation(norm), 2))
    assert round(get_standart_deviation(norm), 2) == 0.34,\
           'Got Problems with normal distribution'


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