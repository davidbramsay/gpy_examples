import matplotlib.pyplot as plt
import numpy as np
import itertools
import GPy
from sklearn import metrics

'''
Simple GPy Example #2:

Uses the XOR dataset, as before, using the entire dataset.

This runs through the parameters defined in 'parameters', saves their AUC-ROC
score, and then re-runs the model with the best scoring parameters, creating
plots for those parameters.

Each score is printed as it is solved for.

'''

parameters = {
        'batchsize': [20, 50, 100],
        'variance': [0.1, 1., 10, 100.],
        'lengthscale': [0.1, 1., 10., 100.],
        'maxiters': [10, 50, 100]
    }



#create an xor dataset
X = np.random.normal(0, 1, (1000, 2))
X_test = np.random.normal(0, 1, (500, 2))

Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).reshape(-1,1)
Y_test = np.logical_xor(X_test[:, 0] > 0, X_test[:, 1] > 0).reshape(-1,1)

#create induce points with similar distribution as X
Z = np.random.normal(0, 1, (500, 2))

#for i in range(len(X)):
#    print X[i], Y[i]

#change 'True' and 'False' to 0 and 1 for Y array
def transform(b): return 1 if b else 0
Y = np.vectorize(transform)(Y)
Y_test = np.vectorize(transform)(Y_test)



#take parameters and make every combination of them, store that in paramCombinations
paramNames = sorted(parameters)
paramCombinations = [dict(zip(paramNames, val)) for val in itertools.product(*(parameters[paramName] for paramName in paramNames))]

best_ROC = 0.0
best_params = {}
scores = []

lik = GPy.likelihoods.Bernoulli()

#step through each parameter combination and call GP on our data using it
for count, p in enumerate(paramCombinations):

    #build the kernel/model using the parameters
    kernel = GPy.kern.RBF(input_dim=np.shape(X)[1], variance=p['variance'], lengthscale=p['lengthscale']) + GPy.kern.White(1)
    m = GPy.core.SVGP(X, Y, Z, kernel, lik, batchsize=p['batchsize'])

    #intialize the model
    try:
        m.randomize()
    except:
        print 'randomize failed'

    m.Z.unconstrain() # ...seems like inducing points move regardless of this command

    #optimize the model
    try:
        m.optimize('adadelta', max_iters=p['maxiters'], messages=1)
    except:
        print 'optimize failed'

    #calc ROC score for the model
    y_pred = m.predict(X_test)[0]
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    #append it to scores with the parameters that created it
    scores.append({'params':p, 'roc-score':roc_auc})

    #print it to the screen
    print '(' + str(count) + '/' + str(len(paramCombinations)) +') ' +str(p) + ' has a score of ... ' + str(roc_auc)
    #save it if it's the best score so far
    if roc_auc > best_ROC:
        print '...saving params, it\'s the best so far'
        best_ROC = roc_auc
        best_params = p

# use the best scoring parameters from the above test to run a final model with
kernel = GPy.kern.RBF(input_dim=np.shape(X)[1], variance=best_params['variance'], lengthscale=best_params['lengthscale']) + GPy.kern.White(1)
m = GPy.core.SVGP(X, Y, Z, kernel, lik, batchsize=best_params['batchsize'])

try:
    m.randomize()
except:
    print 'main randomize failed'

m.Z.unconstrain()

try:
    m.optimize('adadelta', max_iters=best_params['maxiters'], messages=1)
except:
    print 'main optimize failed'

for obj in scores:
    print obj

#do some nice plotting and graphing and such...
fig = m.plot()
GPy.plotting.show(fig)

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
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.colorbar(image)
plt.tight_layout()
plt.show()

