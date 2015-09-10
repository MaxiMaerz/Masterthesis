from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
import os
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
from operator import itemgetter

def generate_random_linear():
    """
    generates a linear dataset y = 0.5 * x + K , where K is some gausian random number, made for every point
    :return: x and y values in form of numpy array
    """
    x_value = np.arange(0, 2*np.pi, 0.01)
    y_value = np.empty(len(x_value))
    y_real = np.empty(len(x_value))
    for x in range(0, len(x_value)):
        y_value[x] = np.sin(x_value[x]) + np.random.normal(0,2)/5
        y_real[x] = np.sin(x_value[x])
    return x_value, y_value, y_real

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
    """Generate Sets"""
    x_set, y_set, y_no_noise = generate_random_linear()

    """Split the set in train and test data"""
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.34, random_state=42)

    """Start training"""
    clf = tree.DecisionTreeRegressor(max_depth=5)
    clf.fit(np.expand_dims(x_train, axis=1), y_train)

    '''Try some Grid-search'''
    start = time()
    params_to_explore = {'max_depth': np.arange(1, 25, 1)
                         }
    grid_search = GridSearchCV(clf, param_grid=params_to_explore)
    stop = time()


    """Make the prediction"""
    predicted = clf.predict(np.expand_dims(x_test, 1))
    grid_search.fit(np.expand_dims(x_train, 1), y_train)
    report(grid_search.grid_scores_)
    print(start-stop)


    '''calculate residuum'''
    #delta = np.zeros(len(predicted))
    #for x in range(0, len(predicted)):
     #   delta[x] = predicted[x] - y_test[x]

    """plotting"""

    """ Hists Here"""
    #hist, bins = np.histogram(delta, bins=50)
    #width = 0.7 * (bins[1] - bins[0])
    #center = (bins[:-1] + bins[1:]) / 2
    #plt.bar(center, hist, align='center', width=width)
    #plt.show()

    """Normal Graph"""
    #print(len(x), len(x))
    plt.plot(x_test, predicted, '.', c='g')
    plt.plot(x_set, y_set, ',', c='b')
    plt.plot(x_set, y_no_noise, c='r')
    plt.show()
    #"""


    #print(np.std(delta))
    """print the Tree"""
    #with open("/home/maxi/trees/linear_easy.dot", 'w') as f:
    #    f = tree.export_graphviz(clf, out_file=f)
    #os.system('dot -Tpdf /home/maxi/trees/linear_easy.dot -o /home/maxi/trees/linear_easy.pdf')
    #os.unlink('/home/maxi/trees/linear_easy.dot')
