from sklearn.metrics import mean_squared_error
from Custom_Statistics import *
from sklearn import tree
import sys

class AdaboostRegression():
    """
    Class for boosted regressions, using tree based methods

    Params:
            -for now Decision_tree_classifier will be used as the weak learner

            -n_estimators: Number of Estimators(Size of the boosted forest)

            -loss_function: A function to calculate the loss

    See also
    --------
    AdaBoostRegressor, GradientBoostingClassifier, DecisionTreeClassifier

    References
    ----------
    .. [1] Scikit-learn: Machine Learning in Python,
           Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

    .. [2] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [3] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """

    def ___init__(self, n_estimators,
                  loss_fkt='exponential',
                  estimator_list=None,
                  estimator_error=None):

        self.n_estimators = n_estimators
        self.loss_fkt = loss_fkt

        self.estimator_list = estimator_list
        if self.estimator_list is None:
            self.estimator_list = []

        self.estimator_error = estimator_error
        if self.estimator_error is None:
            self.estimator_error = []

    def fit(self, features, targets, weights=None):
        """
        Fit the Estimator

        :param features: Array like
        :param targets:  1-D array
        :param weights:  1-D Array, default is None
        :return: fitted estimator
        """

        # Check Data
        if len(features) != len(targets):
            raise ValueError('feature length:' + str(len(features)) +
                             'doesn\'t match target length: ' + str(len(targets)))

        # Fill the weights
        if weights == None:
            '''
            Each element same, nomalized weight,
                Fun Fact: np.empty + arr.fill is super fast, check:
                http://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
            '''
            weights = np.empty(len(targets))
            weights.fill(1/len(targets))



        # Check weights
        if np.sum(weights) != 1.:
            raise ValueError('weights not normalized! ' + str(np.sum(weights)))

        # Assume all is fine, start the iteration
        for boost_step in range(self.n_estimators):
            # waste computing power for loading bar
            step = self.n_estimators/100
            if boost_step%step == 0:
                print('\rTraining: %s (%d%%)' % ("|"*(int(boost_step/(3*step))), boost_step/step), end="")
                sys.stdout.flush()

            # Bootstrap/choose the training data
            X, Y = choose_data(features, targets, weights)

            # Build a tree
            estimator = build_tree(X, Y)

            # Save the estimator
            self.estimator_list.append(estimator)

            #Get weights, and update them Error is calculate from the full sample
            weights, error = update_weights(X, Y, weights, estimator)

        # Now we have a list of n_estimators weak learners, we will use them to build a strong learner
        return self.estimator_list, self.estimator_error

