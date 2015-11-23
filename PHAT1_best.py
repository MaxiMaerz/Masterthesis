from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
sys.path.append('/home/maxi/PycharmProjects/Masterthesis')
sys.path.append('maxi/Dropbox/Masterarbeit/SDSS_data')
from boosted_regression import AdaboostRegression
from astropy.io import fits
from matplotlib import pyplot as plt
from Import_Astrofits import *
from custom_adaboost import *
import warnings
from operator import itemgetter
from sklearn.neighbors import NearestNeighbors
from sklearn import mixture

# Open Phat Data
Path = '/home/maxi/data/small_set/'
hdulist_train = fits.open(Path + 'PHAT1_TRAINING.fits')
hdulist_valid = fits.open(Path + 'PHAT1_TESTSET_MNPLz.fit')
feature_conf = [line.strip() for line in open(Path + '/fields.conf', 'r')]
feature_conf_2 = [line.strip() for line in open(Path + '/fields2.conf', 'r')]


# Get config for feature and target selection
feature_index_Phat, feature_dic_Phat, target_index_PHAT, target_dic_PHAT = select_features(feature_conf, hdulist_train)
feature_index_valid_Phat, feature_dic_valid_Phat, target_index_valid_Phat, target_dic_valid_Phat = select_features(feature_conf_2, hdulist_valid)

# Read the Data to numpy array
PHAT_features_train, PHAT_targets_train = generate_array(hdulist_train,
                                                     feature_index_Phat,
                                                     target_index_PHAT)
PHAT_features_valid, PHAT_targets_valid = generate_array(hdulist_valid,
                                                       feature_index_valid_Phat,
                                                       target_index_valid_Phat)

# Transpose, because skikit likes it
PHAT_features_train = PHAT_features_train.T
PHAT_targets_train = PHAT_targets_train.T
PHAT_features_valid = PHAT_features_valid.T
PHAT_targets_valid = PHAT_targets_valid.T

# Calculate mean and std for targets and check for values outside 5*sigma:
sum_err = 0
average = np.empty((len(PHAT_features_train[0])))
for i in range(0, len(PHAT_features_train[0])):
    PHAT_feature_mean = np.mean(PHAT_features_train[:,i])
    PHAT_feature_std = get_standart_deviation(PHAT_features_train[:,i])[0]
    average[i] = (np.sum(PHAT_features_train[:,i][abs(PHAT_features_train[:,i])
            < abs(PHAT_feature_mean + 10*PHAT_feature_std)])
          /len(PHAT_features_train[:,i][abs(PHAT_features_train[:,i]) < abs(PHAT_feature_mean + 20*PHAT_feature_std)])
         )

for i in range(0, len(PHAT_features_train[0])):
    PHAT_feature_mean = np.mean(PHAT_features_train[:,i])
    PHAT_feature_std = get_standart_deviation(PHAT_features_train[:,i])[0]
    pos_err = PHAT_features_train[:,i][PHAT_features_train[:,i] > PHAT_feature_mean + 5*PHAT_feature_std].shape[0]
    neg_err = PHAT_features_train[:,i][PHAT_features_train[:,i] < PHAT_feature_mean - 5*PHAT_feature_std].shape[0]
    print('Assuming '+ str(pos_err+neg_err) + ' wrong entries in: ' +feature_dic_Phat[i+1] + ' Using max distance: '+
         str(PHAT_feature_mean+PHAT_feature_std*5))
    sum_err += neg_err+pos_err
    PHAT_features_train[:,i][abs(PHAT_features_train[:,i]) > abs(PHAT_feature_mean + 5*PHAT_feature_std)] = average[i]
print('With a total of: ' + str(sum_err) + ' Errors')


average = np.empty((len(PHAT_features_valid[0])))
for i in range(0, len(PHAT_features_valid[0])):
    PHAT_feature_mean = np.mean(PHAT_features_valid[:,i])
    PHAT_feature_std = get_standart_deviation(PHAT_features_valid[:,i])[0]
    average[i] = (np.sum(PHAT_features_valid[:,i][abs(PHAT_features_valid[:,i])
            < abs(PHAT_feature_mean + 10*PHAT_feature_std)])
          /len(PHAT_features_valid[:,i][abs(PHAT_features_valid[:,i]) < abs(PHAT_feature_mean + 20*PHAT_feature_std)])
         )

for i in range(0, len(PHAT_features_valid[0])):
    PHAT_feature_mean = np.mean(PHAT_features_valid[:,i])
    PHAT_feature_std = get_standart_deviation(PHAT_features_valid[:,i])[0]
    pos_err = PHAT_features_valid[:,i][PHAT_features_valid[:,i] > PHAT_feature_mean + 5*PHAT_feature_std].shape[0]
    neg_err = PHAT_features_valid[:,i][PHAT_features_valid[:,i] < PHAT_feature_mean - 5*PHAT_feature_std].shape[0]
    print('Assuming '+ str(pos_err+neg_err) + ' wrong entries in: ' +feature_dic_Phat[i+1] + ' Using max distance: '+
         str(PHAT_feature_mean+PHAT_feature_std*5))
    sum_err += neg_err+pos_err
    PHAT_features_valid[:,i][abs(PHAT_features_valid[:,i]) > abs(PHAT_feature_mean + 5*PHAT_feature_std)] = average[i]
print('With a total of: ' + str(sum_err) + ' Errors')

# generate features
N = len(PHAT_features_train[0])
for i in range(0,N):
    for j in range(0,N):
        if(i != j):
            new_feature_train = PHAT_features_train[:,i] - PHAT_features_train[:,j]
            new_feature_valid = PHAT_features_valid[:,i] - PHAT_features_valid[:,j]
            feature_dic_Phat.update({i*10+j : feature_dic_Phat[i+1]+'-'+feature_dic_Phat[j+1]})
            PHAT_features_train = np.hstack((PHAT_features_train, np.expand_dims(new_feature_train, axis=1)))
            PHAT_features_valid = np.hstack((PHAT_features_valid, np.expand_dims(new_feature_valid, axis=1)))

# initalize predictor
tree_para = {'min_samples_leaf':5}
clf = AdaBoostRegressor(DecisionTreeRegressor(**tree_para),
                                loss='exponential',
                                n_estimators=20)

# fit predictor
clf.fit(PHAT_features_train, PHAT_targets_train[:,0])
predicted = clf.predict(PHAT_features_valid)

# collect stats
delta = predicted - PHAT_targets_valid[:,0]
feature_imp = clf.feature_importances_

result, stats = get_standart_deviation(delta,PHAT_targets_valid[:,0], method='full')
print(result)

full_set = np.hstack((PHAT_features_train, PHAT_targets_train))

#bring all magnitudes to redshift range
rescaled_set = np.copy(full_set)
rescaled_set[:,0:-1] = rescaled_set[:,0:-1]#*feature_av
rescaled_set[:,-1] = rescaled_set[:,-1]

#Draw a sample set every time
kde = KernelDensity(bandwidth=0.001)
kde.fit(rescaled_set)
for i in range(500, 9000, 2000):
    aug_data = kde.sample(i)
    #aug_data = np.vstack((aug_data, full_set))

    # initalize predictor
    tree_para = {'min_samples_leaf' : 5}
    clf = AdaBoostRegressor(DecisionTreeRegressor(**tree_para),
                                    loss='exponential',
                                    n_estimators=20)

    # fit predictor
    clf.fit(aug_data[:,0:-1], aug_data[:,-1])
    predicted_aug = clf.predict(PHAT_features_valid)

    # collect stats
    delta_aug = predicted_aug - PHAT_targets_valid[:,0]
    feature_imp_aug = clf.feature_importances_

    result_aug, stats_aug = get_standart_deviation(delta_aug, PHAT_targets_valid[:,0], method='full')
    print(result_aug)