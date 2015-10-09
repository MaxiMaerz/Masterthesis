import numpy as np


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
    data_float = np.array([astro_data.field(0)])
    for x in range(0, len(astro_data[0])):
        if isinstance(astro_data.field(x)[1], (int, float, complex)) is True\
                        and x not in targets and x in features:
            data_float = np.vstack((data_float, np.squeeze(np.array([astro_data.field(x)]))))
    '''get all and target matching data'''
    targets_float = np.squeeze(np.array(astro_data.field(targets[0])))
    for x in range(len(targets)):
        targets_float = np.vstack((targets_float, np.squeeze(np.array(astro_data.field(targets[x])))))
        print('Selected Feature: ' + hdulist[1].columns.names[targets[x]])
    '''return'''
    data_float = np.delete(data_float, 0, 0)
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
                f_dic.update({x: features_list[y]})

            if features_list[y][:1] == '!' \
                    and features_list[y][1:] == data[1].columns.names[x]:
                t_index = np.append(t_index, x)
                t_dic.update({x: features_list[y][1:]})

    return f_index.astype(int), f_dic, t_index.astype(int), t_dic