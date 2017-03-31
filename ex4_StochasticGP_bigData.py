import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from dateutil import parser
import datetime
import pytz
import time
import numpy as np
import itertools
import GPy
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing

'''
Simple GPy Example #4:

Uses Stochastic mini-batching techniques from before for a large air
quality dataset (134 features predicting 2 classes, x 55,000 points).

Can choose adadelta or leave to have default gradient solver.
Can choose rbf or linear/matern32 kernel.

This example has k-means clustering to initialize the inducing points.

'''

#----EDIT ME----
main_col = 'lmse_calib_as_co'
ref_col = 'co'
remove_cols = features_truth
accuracy_target = 100

use_adadelta = False
use_matern = True

variance = 1.
lengthscale= 1.

batchsize = 200
inducing_points = 500
maxiters = 1000
#---------------#


as_breaktime = datetime.datetime(2016,5,23,21,40)
localtz = pytz.timezone('US/Eastern')
sns.set(style='white', palette='Set2')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_pickle('../data/as1_mlarray.pkl')
df.index = df.index.tz_convert('US/Eastern')

df['day_of_week'] = [(x%7) + 1 for x in df['day_of_year']]

print df.columns.values

features_const = [
    'pressureWind',  'lmse_scaled_arduino_ws',
    'derivative_pressureWind',
    'avg_30_scaled_arduino_ws', 'avg_30_ws', 'scaled_arduino_ws',
    'lmse_avg_30_scaled_arduino_ws',
    'derivative_scaled_arduino_ws', 'derivative_avg_30_scaled_arduino_ws',
    'forecastio_windBearing', 'forecastio_windSpeed', 'forecastio_wind',
    'avg_60_forecastio_windBearing', 'avg_60_forecastio_windSpeed',
    'wd', 'ws',
    'min_since_plugged_in', 'Battery ( %)', 'Noise ( mV)',
    'Humidity ( % RAW)', 'sck_humidity',
    'forecastio_dewPoint','avg_60_forecastio_dewPoint',  'forecastio_humidity',
    'avg_60_forecastio_humidity', 'daily_avg_forecastio_humidity',
    'forecastio_precipIntensity', 'forecastio_precipProbability',
    'avg_60_forecastio_precipIntensity', 'avg_60_forecastio_precipProbability',
    'daily_avg_sck_humidity', 'sck_humidity_saturated',
    'derivative_sck_humidity', 'avg_15_derivative_sck_humidity',
    'humidity_box_differential',
    'forecastio_rain', 'forecastio_fog',
    'Temperature ( C RAW)',
    'forecastio_apparentTemperature', 'forecastio_temperature',
    'avg_60_forecastio_apparentTemperature','forecastio_temperature_c', 'avg_60_forecastio_temperature_c',
    'as_temperature', 'avg_15_as_temperature', 'sck_temperature', 'alphaTemp',
    'derivative_sck_temperature',
    'derivative_avg_15_as_temperature', 'avg_15_derivative_sck_temperature',
    'avg_15_derivative_avg_15_as_temperature', 'daily_avg_sck_temperature',
    'daily_avg_as_temperature', 'daily_avg_forecastio_temperature',
    'temp_sck_box_differential', 'temp_as_box_differential',
    'Light ( lx)',  'Solar Panel ( V)', 'avg_15_lx',
    'derivative_Light ( lx)',
    'forecastio_cloudCover', 'forecastio_visibility',
    'avg_60_forecastio_cloudCover', 'avg_60_forecastio_visibility',
    'forecastio_clear-night',
    'forecastio_clear-day', 'forecastio_partly-cloudy-day',
    'forecastio_partly-cloudy-night', 'forecastio_cloudy',
    'forecastio_pressure', 'avg_60_forecastio_pressure',
    'hour_of_day', 'day_of_year', 'morning', 'afternoon', 'evening',
    'morning_rush', 'lunch', 'evening_rush', 'day', 'night', 'day_of_week'
    ]

features_as = [
    'alphaS3_aux', 'alphaS3_work',
    'alphaS2_work', 'alphaS2_aux',
    'alphaS1_work', 'alphaS1_aux',
    'derivative_alphaS3_aux', 'derivative_alphaS3_work',
    'derivative_alphaS2_aux', 'derivative_alphaS2_work',
    'derivative_alphaS1_aux', 'derivative_alphaS1_work',
    'as_h2s',
    'as_co', 'avg_15_as_co', 'avg_1440_as_co',
    'lmse_calib_as_co', 'avg_15_lmse_calib_as_co', 'avg_1440_lmse_calib_as_co',
    'derivative_as_co', 'derivative_lmse_calib_as_co',
    'derivative_avg_15_lmse_calib_as_co', 'derivative_avg_1440_lmse_calib_as_co',
    'as_no2', 'avg_15_as_no2', 'avg_60_as_no2', 'avg_360_as_no2',
    'lmse_as_no2', 'lmse_avg_15_as_no2',
    'avg_15_lmse_as_no2', 'avg_60_lmse_as_no2', 'avg_360_lmse_as_no2',
    'derivative_as_no2', 'derivative_lmse_as_no2',
    'derivative_lmse_avg_15_as_no2', 'derivative_avg_15_lmse_as_no2',
    'derivative_avg_60_lmse_as_no2', 'derivative_avg_360_lmse_as_no2',
    'as_o3', 'avg_10_as_o3',
    'lmse_calib_as_o3', 'avg_10_lmse_calib_as_o3',
    'derivative_lmse_calib_as_o3', 'derivative_avg_10_lmse_calib_as_o3',
    'derivative_as_o3'
]

features_sck = ['Nitrogen Dioxide ( kOhm)', 'Carbon Monxide ( kOhm)',
    'derivative_Nitrogen Dioxide ( kOhm)', 'derivative_Carbon Monxide ( kOhm)',
    'lmse_sck_co', 'lmse_sck_no2',
    'derivative_lmse_sck_no2', 'derivative_lmse_sck_co'
    ]

features_pm = ['sharpDust', 'derivative_sharpDust', 'scaled_sharpDust',
    'lmse_scaled_sharpDust', 'avg_15_lmse_scaled_sharpDust',
    'avg_60_lmse_scaled_sharpDust', 'avg_720_lmse_scaled_sharpDust',
    'avg_1440_lmse_scaled_sharpDust',   'derivative_scaled_sharpDust', 'derivative_lmse_scaled_sharpDust',
    'derivative_avg_15_lmse_scaled_sharpDust', 'derivative_avg_60_lmse_scaled_sharpDust',
    'derivative_avg_720_lmse_scaled_sharpDust', 'derivative_avg_1440_lmse_scaled_sharpDust'
    ]

features_truth = ['co', 'o3','no', 'no2',
    'bkcarbon', 'avg_60_bkcarbon', 'avg_720_bkcarbon', 'avg_1440_bkcarbon',
    'derivative_bkcarbon', 'derivative_avg_60_bkcarbon',
    'derivative_avg_720_bkcarbon', 'derivative_avg_1440_bkcarbon'
    ]

def get_normalized_feature_matrix(remove_columns = []):

    ## ----- CREATE FEATURE COLUMNS -------------------

    feature_columns = features_const+features_as+features_sck+features_pm+features_truth

    for remove_col in remove_columns:
        feature_columns.remove(remove_col)

    X = df.as_matrix(columns=feature_columns)

    ## ----- PRE-PROCESS FEATURE COLUMNS -------------------

    #remove nan columns
    print 'num features= %s' % len(feature_columns)
    print 'shape before checking NaNs = %s, %s' % np.shape(X)

    mask = np.all(np.isnan(X), axis=0)
    X = X[:,~mask]

    print 'shape after checking NaNs = %s, %s' % np.shape(X)
    print '# features with all nan values are %s' % np.sum(mask)

    for i in xrange(len(mask)-1, -1, -1):
        if mask[i]:
            print 'deleting col ' + feature_columns[i] + ', all NaNs'
            del feature_columns[i]

    #impute (deal with sparse nans)
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    X = imp.transform(X)

    #normalize - I read this is important/good for GPs
    return preprocessing.scale(X)


def get_accuracy_column(abs_tol=100, main_col='lmse_calib_as_co', ref_col='co' ):

    ret_vals = [1 if (main_val > ref_val-abs_tol and main_val < ref_val+abs_tol)  else 0 \
                                    for main_val, ref_val in zip(df[main_col], df[ref_col]) ]

    print main_col + ' vs ' + ref_col + ' with ' + str(abs_tol) + ' tolerance'
    print 'actual tolerance value, +-, is ' + str(abs_tol)
    print 'correct reading percentage= %s' % (sum(ret_vals)/float(len(ret_vals)))

    return np.array(ret_vals)



#get X and y (features and classes).  y is binary (0 or 1)
X_all = get_normalized_feature_matrix(remove_cols)
y_all = get_accuracy_column(accuracy_target, main_col, ref_col)

#split up into test and training sets- for now 2/3 split.
#Later we will 'actually' do this with nested cross-validation.
X = X_all[:2*len(X_all)/3,:]
y = np.array(y_all[:2*len(y_all)/3], ndmin=2).T
X_test = X_all[2*len(X_all)/3:,:]
y_test = np.array(y_all[2*len(y_all)/3:], ndmin=2).T

#cluster feature space to pick inducing points.
#to save on time (since we're experimenting), we save the cluster points
#to a pickle and reload it every additional call of this file if it's there
try:
    Z = pickle.load(open('Z_cluster_temp.pkl', 'rb'))
    print 'loaded induce cluster points from pickle file...'
except:
    print 'clustering to find induce point start values...'
    est = KMeans(n_clusters=inducing_points, init='k-means++', precompute_distances=False)
    est.fit_predict(X)
    Z = est.cluster_centers_
    pickle.dump(Z, open("Z_cluster_temp.pkl", "wb"))


#print shapes, just to double check everything looks right
print 'X shape:' + str(np.shape(X))
print 'y shape:' + str(np.shape(y))
print 'Z shape:' + str(np.shape(Z))
print '--'
print 'X test shape:' + str(np.shape(X_test))
print 'y test shape:' + str(np.shape(y_test))
print '--'

#for i in range(len(X)):
#    print X[i], Y[i]


#model it!
lik = GPy.likelihoods.Bernoulli()

if use_matern:
    kernel = GPy.kern.Linear(input_dim=np.shape(X)[1], ARD=True) + \
            GPy.kern.Matern32(input_dim=np.shape(X)[1], variance=variance, lengthscale=lengthscale, ARD=True) + \
            GPy.kern.White(1)
else:
    kernel = GPy.kern.RBF(input_dim=np.shape(X)[1], variance=variance, lengthscale=lengthscale, ARD=True) + \
            GPy.kern.White(1)

print '-'*20
print 'starting with batchsize: %s, max_iterations: %s, and %s inducing points' % (batchsize, maxiters, inducing_points)
print '-'*20

m = GPy.core.SVGP(X, y, Z, kernel, lik, batchsize=batchsize)
m.Z.unconstrain()  # ...seems like inducing points move regardless of this command

if use_adadelta:
    m.optimize('adadelta',max_iters=maxiters, messages=1)
else:
    m.optimize(max_iters=maxiters, messages=1)

#generate class prediction probabilities
y_pred = m.predict(X_test)[0]

#print some more stuff, for sanity checking
print '-'*10
print y_pred
print y_test
print '-'*10
print np.shape(y_pred)
print np.shape(y_test)
print '-'*10
print y_pred[0:10]
print y_test[0:10]


#write it to a file, so we can play with predictions/test scores afterwards and make sure things are working
pickle.dump([y_pred, y_test], open("temp_test.pkl", "wb"))

#generate AUC-ROC curves
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

#print the score and final model
print 'ROC-AUC SCORE: ' + str(roc_auc)
print m

#plot the AUC-ROC curve
plt.figure(figsize=(9,9))
plt.plot(fpr, tpr, label='ROC Score = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1- specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.legend(loc="lower right")
plt.show()

