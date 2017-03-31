# GPY_EXAMPLES

My attempt at using GPy to scale Gaussian Processes to large datasets.
###
Check out the paper that talks about how it works [here](http://jmlr.org/proceedings/papers/v38/hensman15.pdf)
[Here](http://www.auai.org/uai2013/prints/papers/244.pdf)'s another good reference


## what's here:

###EX1:
Simple GPy Example #1:
Test GP techniques on the XOR dataset.

###EX2:
Simple GPy Example #2:
Uses the XOR dataset, as before, using the entire dataset.

This runs through the parameters defined in 'parameters', saves their AUC-ROC
score, and then re-runs the model with the best scoring parameters, creating
plots for those parameters.

Each score is printed as it is solved for.

###EX3:
Simple GPy Example #3:
This GP uses a giant XOR dataset to test mini-batching and Stochastic GP
techniques.

Can choose adadelta or leave to have default gradient solver.
Can choose rbf or linear/matern32 kernel.

-no clustering to initialize Z in this example

###EX4:
Simple GPy Example #4:
Uses Stochastic mini-batching techniques from before for a large air
quality dataset (134 features predicting 2 classes, x 55,000 points).

Can choose adadelta or leave to have default gradient solver.
Can choose rbf or linear/matern32 kernel.

This example has k-means clustering to initialize the inducing points.

###EX5:
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


