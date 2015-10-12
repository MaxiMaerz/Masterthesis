from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from Custom_Statistics import *
from sklearn import tree

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


def build_weighted_tree(x_train, y_train, x_test, y_test, parameter):
    """
    We construct a Tree via skipi_tree_regressor and calculate the MSE and update weights

    Note: we use train_test_split to get a validation sample
    :param data: np.array
    :param target: np.array
    :param parameter: Hyperparameters for the tree (not implemented)

    :return:updated weights
    """

    '''Split data  -> not possible we need to track the index for weight update.....'''
    #x_train, x_test, y_train, y_test = train_test_split(data, target[0], test_size=0.5)

    '''Build tree slice last coloumn (its just index)'''
    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    clf = tree.DecisionTreeRegressor(max_depth=parameter)
    clf.fit(x_train[:, 0:-1], y_train[0])
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
        #weights[Error[-1][i]] = betha * (Error[0][i]) * weights_normal[Error[-1][i]] #linear
        #weights[Error[-1][i]] = betha * np.exp(1-(Error[0][i])) * weights_normal[Error[-1][i]] #exponential loss
        weights[Error[-1][i]] = betha * np.exp(-(Error[0][i])) * weights_normal[Error[-1][i]] #exponential gain
        #weights[Error[-1][i]] = betha * np.exp((Error[0][i])) * weights_normal[Error[-1][i]] #exponential gain