from Custom_Statistics import *
from sklearn import tree
import sys


class AdaboostRegression:
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

    def __init__(self, n_estimators=2,
                 loss_fkt='linear',
                 estimator_list=None,
                 estimator_error=None,
                 tree_parameters=None
                 ):

        self.n_estimators = n_estimators
        self.loss_fkt = loss_fkt
        self.tree_parameters = tree_parameters
        self.estimator_list = estimator_list
        self.estimator_weights = []

        if tree_parameters is None:
            self.tree_parameters = {'max_depth': 20}
        if self.estimator_list is None:
            self.estimator_list = []

        self.estimator_error = estimator_error
        if self.estimator_error is None:
            self.estimator_error = []

    @staticmethod
    def choose_data(features, targets, weights):
        """
        Chooses Data by bootstrapping

        :param features: list of features
        :param targets:  list of targets
        :param weights:  normalized prob. dist.

        :return: bootstrapped features, targets and weights
        """

        # Select the Index by weight using np.choice
        index_arr = np.arange(0, len(features))
        random_index_arr = np.random.choice(index_arr, size=int(len(features)), replace=True, p=weights)

        # Get the randomized array
        x = features[random_index_arr]
        y = targets[random_index_arr]

        return x, y

    def build_tree(self, x, y):
        """
        Builds a single tree as an weak learner

        :param x: features
        :param y: targets

        :return:  estimator
        """
        clf = tree.DecisionTreeRegressor(**self.tree_parameters)
        clf.fit(x, y)

        return clf

    def update_weights(self, features, targets, weights, estimator):
        """
        We will predict on the original Data, calculate Delta and update the weights

        If the loss fkt. is invalid we always use linear-loss
        For more info about Adaboost.R2:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.100.4560&rep=rep1&type=pdf

        :param features:
        :param targets:
        :param weights:     normalized prob. dist
        :param estimator:   a weak learner: decision tree regressor
        :return:
        """

        # Make a prediction
        predicted = estimator.predict(features)

        # Calculate Delta
        delta = abs(predicted - targets)

        # As element-wise loss we use:
        sup = delta.max()
        if sup != 0:
            element_loss = delta / delta.max()
        else:
            print('delta_max = 0, aborting')
            return 0

        if self.loss_fkt == 'exponential':
            element_loss = 1. - np.exp(- element_loss)
        if self.loss_fkt == 'square':
            element_loss **= 2

        # Average Loss
        average_loss = (element_loss * weights).sum()

        # Calculate the Confidence
        betha = average_loss / (1 - average_loss)

        # Update weights
        weights = weights * np.power(betha, (1. - element_loss))

        # Estimator weights
        estimator_weight = 1. / np.log(betha)

        return weights, estimator_weight

    def fit(self, features, targets, weights=None):
        """
        Fit the Estimator

        :param features: Array like
        :param targets:  1-D array
        :param weights:  1-D Array, default is None
        :return: fitted estimator
        """

        # Check Data
        if False: #len(features) != len(targets):
            raise ValueError('feature length: ' + str(len(features)) +
                             ' doesn\'t match target length: ' + str(len(targets)))

        # Fill the weights
        if weights is None:
            '''
            Each element same, normalized weight,
                Fun Fact: np.empty + arr.fill is super fast, check:
                http://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
            '''
            weights = np.empty(len(targets))
            weights.fill(1/len(targets))

        # Check weights
        if round(np.sum(weights), 10) != 1.:
            raise ValueError('weights not normalized! ' + str(np.sum(weights)))

        # Assume all is fine, start the iteration
        for boost_step in range(self.n_estimators):
            # Bootstrap/choose the training data
            bootstrapped_features, bootstrapped_targets = self.choose_data(features, targets, weights)

            # Build a tree
            estimator = self.build_tree(bootstrapped_features, bootstrapped_targets)

            # Save the estimator
            self.estimator_list.append(estimator)

            # Get weights, and update them Error is calculate from the full sample
            weights, estimator_error = self.update_weights(features, targets, weights, estimator)
            self.estimator_weights.append(abs(estimator_error))
            # Normalize weights
            weights = normalize(weights)

            # waste computing power for loading bar
            step = self.n_estimators/100
            print('\rTraining: %s (%d%%)' % ("|"*(int(boost_step/(3*step))), boost_step/step), end="")
            sys.stdout.flush()

        # Now we have a list of n_estimators weak learners, we will use them to build a strong learner
        return self.estimator_list, self.estimator_weights

    @staticmethod
    def predict(features, est_list, weight):
        """
        makes the prediction with the estimators using skikit_learn decision_trees

        :param features: validation_features
        :param est_list: list of estimators
        :param weight:   weight of the estimators
        :return: prediction_vector
        """
        prediction = np.empty(len(features))
        weight = normalize(weight)
        for x in range(len(est_list)):
            prediction += est_list[x].predict(features)*(weight[x])
            print(weight[x]/np.sum(weight))
        return prediction



