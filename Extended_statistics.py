import numpy as np


class PhotoZStatistics:
    """
    Implementation od robust statistics to calculate statistical measures on ML redshift predictions.

    Version 0.1

    Requirements:
                -numpy
                -Python(3.5)

    Content:
        - Sigma_68              calculates std via 68-95-99 rule
        - Sigam_95              ---------------"-----------------
        - Outliners             estimates a fraction of catastrophic outliners ( < 0.15)
        - Feature_Importance    estimates importance of input features

    All values will be scaled by residuals, too. Multidimensional arrays should work,
    output as text, plot and np.array
    """

    def __init__(self,
                 add_data=None,
                 plot=False,
                 verbose=False,
                 np_array=True,
                 method='simple',
                 axis=None,
                 version=0.1
                 ):
        self.axis = axis
        self.add_data = add_data
        self.method = method
        self.plot = plot
        self.verbose = verbose
        self.np_array = np_array
        self.__version__ = version

    def test_get_standard_deviation_norm(self):
        norm = np.random.normal(0, 1, 10000)
        std = np.std(norm, ddof=1)
        std68 = self.get_standard_deviation(norm)[0]
        std95 = self.get_standard_deviation(norm)[1]
        assert round(std68, 1) == round(std, 1),\
               'Sigma68 not matching sigma68 in Numpy.Normal'
        assert round(std95, 1) == round(std * 2, 1),\
               'Sigma95 not matching sigma95 in Numpy.Normal'

    def test_get_standard_deviation_uniform(self):
        uniform = np.random.uniform(0, 1, 10000)
        std68 = self.get_standard_deviation(uniform)[0]
        assert round(std68, 2) == 0.34,\
               'Sigma68 not matching sigma68 in Numpy.uniform'

    def test_outliner_fraction(self):
        # Stupid Test
        data = np.random.uniform(0, 1, 10000)

        val = np.zeros(len(data))
        outs = self.get_outliner_fraction(data, val)
        assert 1 - round(outs, 3) < 0.17, \
               'Outliner rate wrong: ' + str(outs) + ' not 0.5'

    def test_multi_dim_support(self):
        vals = np.zeros((10, 10))
        vals[0] = np.arange(0, 10, 1)

        std_row = self.get_standard_deviation(vals, axis=0)
        std_col = self.get_standard_deviation(vals, axis=1)
        assert std_row[0][0] == 3.,  \
               'Row std not correct! std_row[0] = ' + str(std_row[0])

        assert np.any(std_row[1:-1] ==  0),  \
               'Row std not correct! Std_row[1:-1] = ' + str(std_row[1:-1])
        assert round(std_col[:, 0][0], 5) == 0.0, \
               'Col std not correct! std_col[:, 0} = ' + str(round(std_col[:, 0][0], 5))
        assert round(std_col[:, 0][1], 5) == 0.1, \
               'Col std not correct! std_col[:, 0} = ' + str(round(std_col[:, 0][1], 5))

    @staticmethod
    def cal_std(data):
        """
        Calculates the robust std68, 95 and mean

        :param data: input array(1Dim)
        :return: std68, 95, mean
        """
        assert data.dtype == float or data.dtype == int, \
            'Data-Type not understood'
        assert data.ndim == 1, \
            'Array not 1-dimensional!'

        # sort Data
        data_sort = np.sort(data, axis=None)
        # Get Mean index
        mean_index = int(len(data_sort) / 2)
        # Shift Center of distribution to 0
        data_sort -= np.mean(data_sort)
        # get index of 68% of data
        p_std68 = data_sort[mean_index + int(0.341344746 * len(data_sort))]
        m_std68 = data_sort[mean_index - int(0.341344746 * len(data_sort))]
        std68 = (abs(p_std68) + abs(m_std68))/2

        # get index of 95% of data
        p_std95 = data_sort[mean_index + int(0.47725 * len(data_sort))]
        m_std95 = data_sort[mean_index - int(0.47725 * len(data_sort))]
        std95 = (abs(p_std95) + abs(m_std95))/2
        vals = [std68, std95, data_sort[mean_index]]
        return vals

    def get_standard_deviation(self, data):
        """
        Robust Method to calculate standard deviation, now supporting multidim - support
        according to:
        axis = None - 1-Dim
        axis = 0-1

        axis = 0 : rows
        axis = 1 : cols

        It uses the 68-95-99.7 Rule to calculate the std 68 and std 95
        :param data: 1D - Array
        :return: std68 (float), std95 (float)
        """

        if self.axis is None and self.method is 'simple':
            vals = self.cal_std(data)
            return vals

        if self.axis is None and self.method is 'full':
            vals = self.cal_std(data)
            delta_res = data / (1 + self.add_data)
            vals_res = self.cal_std(delta_res)
            return vals, vals_res

        if self.axis is not None:
            # Slice according to axis choice
            if self.axis == 0:
                vals = np.empty((len(data[0]), 3))
                for i in range(len(data[0])):
                    vals[i] = self.cal_std(data[i])

            elif self.axis == 1:
                vals = np.empty((len(data), 3))
                for i in range(len(data)):
                    vals[i] = self.cal_std(data[:, i])
            # Return -1 if invalid axis chosen
            else:
                print('Invalid choice of axis axis = ' + str(self.axis))
                vals = -1
            # Calculate residuals too
            if self.method is 'full':
                if self.axis is 0:
                    vals_res = np.empty((len(data[0]), 3))
                    for i in range(len(data[0])):
                        vals_res[i] = self.cal_std(data[i] / (1. + self.add_data))

                elif self.axis is 1:
                    vals_res = np.empty((len(data), 3))
                    for i in range(len(data)):
                        vals_res[i] = self.cal_std(data[:, i] / (1. + self.add_data))
                else:
                    print('Invalid choice of axis axis = ' + str(self.axis))
                    vals = -1
                return vals, vals_res
            return vals

    def get_outliner_fraction(self, data, axis=None):
        # Get Outliner fraction by definition
        if axis is None:
            delta_res = abs(data/ (1 + self.add_data))
            x = np.sum(delta_res > 0.15)/len(delta_res)
            return x
        else:
            # Slice according to axis choice
            if axis is 0:
                x = np.empty((len(data), 3))
                for i in range(len(data[0])):
                    x[i] = np.sum(data[i] / (1 + self.add_data) >  0.15) / len(data[i])

            elif axis is 1:
                x = np.empty((len(data), 3))
                for i in range(len(data)):
                    x[i] = np.sum(data[:, i] / (1 + self.add_data) >  0.15) / len(data[:, i])

            else:
                print('Outliner calculation broke!')
                x =- 1
            return x
