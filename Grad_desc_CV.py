import numpy as np
from sklearn import cross_validation
from Extended_statistics import PhotoZStatistics
import sys

class GradientDecentCV:

    def __init__(self,
                 parameters,
                 data,
                 target,
                 cv_size,
                 model,
                 estimator,
                 est_params,
                 model_params,
                 limit=0.05):
        self.static = parameters
        self.data = data
        self.target = target
        self.cv_size = cv_size
        self.model = model
        self.estimator = estimator
        self.est_params = est_params
        self.model_params = model_params
        self.breakpoint = limit

    def step(self, start_est_params, start_model_params):
        # Initalize
        std_68 = np.empty(self.cv_size)
        x=0
        kf = cross_validation.KFold(len(self.target), n_folds=self.cv_size, shuffle=True)
        # Training with Cv
        for train_index, test_index in kf:
            clf = self.model(self.estimator(**start_est_params), **start_model_params)

            # fit predictor
            clf.fit(self.data[train_index], self.target[train_index])
            predicted = clf.predict(self.data[test_index])
            delta = predicted - self.target[test_index]

            stats = PhotoZStatistics(add_data=self.target[test_index])
            std_68[x] = stats.get_standard_deviation(delta)[0]
            x += 1

        std_new = np.sum(std_68)/6
        # Averaging stats
        return std_new

    def set_start(self, start_est_params, start_model_params):
        for params in self.est_params:
            start_est_params.update({params: np.random.choice(self.est_params[params])})
            start_est_params.update({params: self.est_params[params][3]})

        for params in self.model_params:
            start_model_params.update({params: np.random.choice(self.model_params[params])})
            start_model_params.update({params: self.model_params[params][3]})


    def get_direction(self, start_est_params, start_model_params):
        std_recent = self.step(start_est_params, start_model_params)
        est_choice={}
        model_choice = {}
        for est in start_est_params:

            # Try +
            if start_est_params[est]+1 in self.est_params[est]:
                start_est_params[est] += 1
                std_p = self.step(start_est_params, start_model_params)
                start_est_params[est] -= 1
            else: std_p = np.inf

            # Try -
            if start_est_params[est]-1 in self.est_params[est]:
                start_est_params[est] -= 1
                std_m = self.step(start_est_params, start_model_params)
                start_est_params[est] += 1
            else: std_m = np.inf

            #compare
            if std_p < std_m and std_p < std_recent:
                est_choice.update({est: 1})
            if std_m < std_p and std_m < std_recent:
                est_choice.update({est: -1})
            if std_recent <= std_p and std_recent <= std_m:
                est_choice.update({est: 0})

        for model in start_model_params:
            # Try +
            if start_model_params[model]+1 in self.model_params[model]:
                start_model_params[model] += 1
                std_p = self.step(start_est_params, start_model_params)
                start_model_params[model] -= 1
            else: std_p = np.inf

            # Try - (minus 2 because we already incremented
            if start_model_params[model]-1 in self.model_params[model]:
                start_model_params[model] -= 1
                std_m = self.step(start_est_params, start_model_params)
                start_model_params[model] += 1
            else: std_m = np.inf

            #compare
            if std_p < std_m and std_p < std_recent:
                model_choice.update({model: 1})
            if std_m < std_p and std_m < std_recent:
                model_choice.update({model: -1})
            if std_recent <= std_p and std_recent <= std_m:
                model_choice.update({model: 0})
        print(std_p , std_m , std_recent)
        return est_choice, model_choice

    def search_min(self):
        start_est_params, start_model_params = {}, {}
        # Generate a random starting point
        self.set_start(start_est_params, start_model_params)
        current_est_params = start_est_params
        current_model_params = start_model_params
        # Check if
        average = 10.
        std = 0
        x = 0
        average_old = 0.002
        while abs(average-1) > self.breakpoint:
            est_choice, model_choice= self.get_direction(start_est_params, start_model_params)

            # Update Hyperparameters
            for est in self.est_params:
                current_est_params.update({est: current_est_params[est] + est_choice[est]})

            for model in self.model_params:
                current_model_params.update({model: current_model_params[model] + model_choice[model]})

            std += self.step(current_est_params, current_model_params)
            x+=1
            if x == 5:
                average_new = std/x
                x = 0
                std = 0
                average = average_new/average_old
                average_old = average_new
            if x%2 == 1:
                step = '\\'
            else:
                step = '/'
            if x != 0:
                print('\rCurrent Std_68: ' + str(std/x) + step, end="")
            sys.stdout.flush()
        return current_est_params, current_model_params, average_new
