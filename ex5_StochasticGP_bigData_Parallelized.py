import matplotlib
matplotlib.use('Agg') #so works on headless server, like Amazon cloud
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import multiprocessing as mp
import pandas as pd
from dateutil import parser
import sys
import datetime
import pytz
from functools import partial
import time
import numpy as np
import itertools
import GPy
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing

'''
Simple GPy Example #5:

Highly Parallelized search over parameters using Stochastic mini-batching
techniques from before for a large air quality dataset (134 features
predicting 2 classes, x 55,000 points).

This will search over all combinations of parameters listed below,
save their AUC_ROC scores and optimization parameters to 'parallel_GP_test.txt',
and save a figure with the AUC_ROC curve to ./figs as it goes.

Can choose adadelta or leave to have default gradient solver.
Can choose rbf or linear/matern32 kernel.
Can choose ARD on or off.
Can choose batchsize, maxiterations, and inducing points.

This example has k-means clustering to initialize the inducing points.  It first parallelized the
clustering task, and finds optimal central point arrays for every value of inducing points we want.
This is stored in Z_array.

(so if we call with inducing_points: [10,50,100], first we will launch
three workers that will run k_means on X to find 10, 50, and 100 points. These arrays of points are stored
in Z_array).

Using these clusters to initialize Z, we then call every combination of parameters in the parameter dictionary.

'''

#----EDIT ME----#
parameters = {
    'batchsize':[10,20,100,200,500],
    'maxiters':[10,100,1000,10000,100000],
    'inducing_points':[10,50,100,200,500,1000],
    'use_ard':(True, False),
    'use_matern':(True, False),
    'use_adadelta':(True, False)
}

main_col = 'lmse_calib_as_co'
ref_col = 'co'
remove_cols = features_truth
accuracy_target = 100

#good idea to turn messages off so we're not writing constantly to the screen, it's jibberjabber anyway when running on tons of cores
messages = 0

#for testing only, runs relatively quickly on multicore machine
#parameters = {'batchsize':[10,20],'maxiters':[10,100],'inducing_points':[3,5],'use_ard':(True, False),'use_matern':(True, False),'use_adadelta':(True, False)}

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

    #normalize - I read this is good/important for GPs
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


#print shapes, just to double check everything looks right
print 'X shape:' + str(np.shape(X))
print 'y shape:' + str(np.shape(y))
print '--'
print 'X test shape:' + str(np.shape(X_test))
print 'y test shape:' + str(np.shape(y_test))
print '--'

#for i in range(len(X)):
#    print X[i], Y[i]

#function to run the GP given a set of parameters.  Z values are passed in so
#we can reuse our clusters instead of computing a new one each time.
#returns (p, auc_roc) where p is the parameters passed in for the gp and auc_roc is the score for that model
#also writes scores/etc to 'parallel_GP_test.txt' and saves figures in .fig file as it goes
#Z_array is a dict with key=num_clusters and val=array of cluster center points we calcuate at the beginning
def run_gp(p, Z_array):
    params = 'batchsize: %s, induce_points: %s, maxiter: %s, adadelta: %s, matern: %s, ARD: %s' % (p['batchsize'], p['inducing_points'], p['maxiters'], p['use_adadelta'], p['use_matern'], p['use_ard'])
    pname = 'b%sipts%smaxi%sad%smat%sARD%s' % (p['batchsize'], p['inducing_points'], p['maxiters'], p['use_adadelta'], p['use_matern'], p['use_ard'])

    #make inducing points based on number passed (using previously generated k-means cluster center points)
    Z = Z_array[p['inducing_points']]

    #generate kernel/model
    lik = GPy.likelihoods.Bernoulli()

    if p['use_matern']:
        kernel = GPy.kern.Linear(input_dim=np.shape(X)[1], ARD=p['use_ard']) + \
                GPy.kern.Matern32(input_dim=np.shape(X)[1], variance=variance, lengthscale=lengthscale, ARD=p['use_ard']) + \
                GPy.kern.White(1)
    else:
        kernel = GPy.kern.RBF(input_dim=np.shape(X)[1], variance=variance, lengthscale=lengthscale, ARD=p['use_ard']) + \
                GPy.kern.White(1)

    print '-'*20
    print 'starting with batchsize: %s, max_iterations: %s, and %s inducing points' % (p['batchsize'], p['maxiters'], p['inducing_points'])
    print '-'*20

    m = GPy.core.SVGP(X, y, Z, kernel, lik, batchsize=p['batchsize'])
    m.Z.unconstrain() # ...seems like inducing points move regardless of this command

    #optimize it
    if p['use_adadelta']:
        m.optimize('adadelta',max_iters=p['maxiters'], messages=messages)
    else:
        m.optimize(max_iters=p['maxiters'], messages=messages)

    #predict auc_roc score
    y_pred = m.predict(X_test)[0]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    #print it to the screen
    print 'ROC-AUC SCORE for %s: %s ' % (params, roc_auc)
    print m

    #make a plot and save it in ./figs
    plt.figure(figsize=(9,9))
    plt.plot(fpr, tpr, label='ROC Score = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1- specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.legend(loc="lower right")
    plt.title(params)
    plt.savefig('./figs/' + pname + '_roc.png', bbox_inches='tight', dpi=200)

    #reroute stdout to file
    old_target = sys.stdout

    #write some stuff to the file about the model and its score
    sys.stdout = open("parallel_GP_test.txt", "a")
    print '-'*20
    print '\n'
    print 'ROC-AUC SCORE for {{ %s }} : %s ' % (params, roc_auc)
    print m
    print '\n'
    print '-'*20

    #give stdout back to the screen
    sys.stdout = old_target

    #return parameters and their score
    return p, roc_auc


#this gives an array of center points for k-means clustering of X given a number of cluster points i
#returns (i, cluster_array) where i is num points
def get_cluster(i):
    print 'clustering ' + str(i) + '...'
    est = KMeans(n_clusters=i, init='k-means++', precompute_distances=False)
    est.fit_predict(X)
    return i, est.cluster_centers_




#take parameters and make every combination of them, store that in paramCombinations
paramNames = sorted(parameters)
paramCombinations = [dict(zip(paramNames, val)) for val in itertools.product(*(parameters[paramName] for paramName in paramNames))]

#make a pool of workers that is equal to #CPUs on machine
worker_pool=mp.Pool(processes=mp.cpu_count())

#Z_array will be a dict with key=num_clusters and val=array of cluster center points we calcuate at the beginning
Z_array = {}

#parallelize works that do k-means clustering on all values we want to use as inducing_points
results = worker_pool.map(get_cluster, parameters['inducing_points'])

#add results to Z_array
for result in results:
    i, centers = result
    print 'adding %s: %s' % (i, np.shape(centers))
    Z_array[i] = centers

#pass Z_array to GP function
run_gp_partial = partial(run_gp, Z_array=Z_array)

#call GP function with all combinations of parameters
results = worker_pool.map(run_gp_partial, paramCombinations)

#print the results
for result in results:
    p, roc_auc = result
    print p, roc_auc

pickle.dump(results, open("temp.pkl", "wb"))
