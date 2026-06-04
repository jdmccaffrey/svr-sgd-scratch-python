# svr-sgd-scratch-python
Support vector regression with SGD training from scratch using Python

This implementation of support vector regression uses stochastic gradient descent (SGD) for training instead of the more common sequential minimal optimization (SMO) algorithm.

SGD training for SVR is much, much simpler to implement than SMO, but SGD training is slower than SMO training.

The implementation uses a single alpha vector for weights, rather than the more usual dual alpha and alpha* weight vectors.

The implementation uses a hard-wired radial basis function (RBF) as the kernel function, as opposed to allowing different functions like the polynomial kernel or the linear kernel.

SVR was popular for a short time in the late 1990s and early 2000s, until people discovered that the closely related kernel ridge regression (KRR) is superior to SVR in nearly every way.

SVR is significantly more difficult to implement than KRR. SVR parameters (gamma, epsilon, C, max iterations, tolerance) are much more difficult to tune than KRR parameters. SVR models typically don't perform as well as KRR models in terms of prediction accuracy (mostly because of the difficulty in tuning parameters).

All that said, there are some problem domains where SVR is still used. And SVR is extremely interesting from a mathematical point of view.

## Usage

One of the reasons that SVR parameters are difficult to tune is that they are intertwined. And slight changes can effectively move training vectors in and out of the epsilon tube, which creates large changes in the model.

The demo implementation loosely follows the scikit-learn SVR module API. Example calling code:

    # X is a numpy matrix of predictors
    # y is a numpy vector of target values
    model = KernelSVR(gamma=0.30, epsilon=0.001, C=1.0, max_iter=100, tol=1.0e-5, seed=0)
    model.fit(X, y)

    print("Predicting first training item")
    x = X[0].reshape(1, -1)
    pred_y = model.predict(x)[0]
    print("Predicted y = %0.4f " % pred_y)

    r2 = model.r2_score(X, y)
    print("Model R2 = %0.4f" % r2)

The gamma parameter controls the RBF kernel function. Increasing gamma shrinks the radius of influence of individual data points. This causes the model to give more weight to points that are very close to each other.

The epsilon parameter controls which items are ignored during training. Increasing epsilon tends to create fewer support vectors (those with non-zero alpha values).

The C parameter controls regularization, to prevent alpha weights from becoming large. Increasing C increases the penalty for points falling outside the epsilon tube. This forces the model to fit the training data more strictly, which increases accuracy but creates an increased risk of model overfitting.

The max_iter parameter sets the maximum consecutive number of times the SMO algorithm iterates without finding an improvement in alpha.

The tol parameter determines if an alpha value is zero or not, which identifies if the associated training item is a support vector or not. 

In practice, tuning SVR parameters is often extremely difficult.

## Example Output

The svr_sgd.py file is a complete demo program that contains the key KernelSVR class. Output of a demo run using synthetic data:

```
Begin scratch kernel SVR using SGD training

Loading synthetic train (200) and test (40) data
Done

First three train X:
[-0.1660  0.4406 -0.9998 -0.3953 -0.7065]
[ 0.0776 -0.1616  0.3704 -0.5911  0.7562]
[-0.9452  0.3409 -0.1654  0.1174 -0.7192]

First three train y:
0.4840
0.1568
0.8054

Creating scratch Python SVR model
Setting gamma = 0.3000
Setting C = 1.00
Setting epsilon = 0.007500
Setting lrn_rate = 0.0010
Setting max_epochs = 6000
Setting tol = 1.0e-04

Training SVR model using SGD
epoch =    0  |  MSE = 0.0540
epoch = 1200  |  MSE = 0.0001
epoch = 2400  |  MSE = 0.0001
epoch = 3600  |  MSE = 0.0001
epoch = 4800  |  MSE = 0.0001
Done

Model alpha (weights):
[-0.9227 -0.0424 -0.0019 -0.6640  . . .  0.4711
 -0.0495  0.0969 -0.1051  0.5641  . . .  0.0114
 . . .
 -0.9147 -0.3332  0.1573  0.0021 . . .  -0.0452
 -0.9971 -0.0190 -0.1292  0.3894]
Model bias: 0.4040
Number support vectors = 186

Train accuracy (0.10) = 0.9850
Test accuracy (0.10) = 0.9250

Train MSE = 0.0001
Test MSE = 0.0001

Train R2 = 0.9986
Test R2 = 0.9949

Predicting for train_X[0]
Predicted y = 0.4932

End demo
```
