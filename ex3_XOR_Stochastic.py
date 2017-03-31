import matplotlib.pyplot as plt
import numpy as np
import itertools
import GPy
from sklearn import metrics

#this GP uses a giant XOR dataset to test mini-batching and Stochastic GP
#techniques.  Can choose adadelta or leave to have default gradient solver
# no clustering to initialize Z in this example
'''
Simple GPy Example #3:

This GP uses a giant XOR dataset to test mini-batching and Stochastic GP
techniques.

Can choose adadelta or leave to have default gradient solver.
Can choose rbf or linear/matern32 kernel.

-no clustering to initialize Z in this example

'''

#----EDIT ME----#
batchsize = 200
variance = 1.
lengthscale= 1.
maxiters = 200
use_adadelta = False
use_matern = True
inducing_points = 500
#---------------#

#create an xor dataset
X = np.random.normal(0, 1, (1000000, 2))
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).reshape(-1,1)

X_test = np.random.normal(0, 1, (1000, 2))
Y_test = np.logical_xor(X_test[:, 0] > 0, X_test[:, 1] > 0).reshape(-1,1)

#create induce points with similar distribution as X
Z = np.random.normal(0, 1, (inducing_points, 2))

#for i in range(len(X)):
#    print X[i], Y[i]

#change 'True' and 'False' to 0 and 1 for Y array
def transform(b): return 1 if b else 0
Y = np.vectorize(transform)(Y)
Y_test = np.vectorize(transform)(Y_test)



print 'x: ' + str(np.shape(X))
print 'y: ' + str(np.shape(Y))
print 'z: ' + str(np.shape(Z))
print '-'*20
print 'starting with batchsize: %s, max_iterations: %s, and %s inducing points' % (batchsize, maxiters, inducing_points)
print '-'*20

lik = GPy.likelihoods.Bernoulli()

#build the kernel/model using the parameters
if use_matern:
    kernel = GPy.kern.Linear(input_dim=np.shape(X)[1], ARD=True) + \
            GPy.kern.Matern32(input_dim=np.shape(X)[1], variance=variance, lengthscale=lengthscale, ARD=True) + \
            GPy.kern.White(1)
else:
    kernel = GPy.kern.RBF(input_dim=np.shape(X)[1], variance=variance, lengthscale=lengthscale) + \
            GPy.kern.White(1)

m = GPy.core.SVGP(X, Y, Z, kernel, lik, batchsize=batchsize)
m.Z.unconstrain() # ...seems like inducing points move regardless of this command

#optimize the model
if use_adadelta:
    m.optimize('adadelta',max_iters=maxiters, messages=1)
else:
    m.optimize(max_iters=maxiters, messages=1)

#calc ROC score for the model
Y_pred = m.predict(X_test)[0]
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
roc_auc = metrics.auc(fpr, tpr)

#print it to the screen
print 'ROC-AUC SCORE: ' + str(roc_auc)
print m

#make a nice AUC-ROC plot
plt.figure(figsize=(9,9))
plt.plot(fpr, tpr, label='ROC Score = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1- specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.legend(loc="lower right")

#make a default GPy plot
fig = m.plot()
GPy.plotting.show(fig)

#make a custom plot of GPy fit
xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                     np.linspace(-3, 3, 50))

Z = m.predict(np.vstack((xx.ravel(), yy.ravel())).T)[0]
Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 5))
image = plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)

contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                        linetypes='--')
plt.scatter(X[1:100, 0], X[1:100, 1], s=30, c=Y[1:100], cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.colorbar(image)
plt.tight_layout()
plt.show()

