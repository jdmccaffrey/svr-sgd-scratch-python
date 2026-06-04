# svr_sgd.py

# support vector regression (kernelized) with 
# sub-gradient SGD training

import numpy as np

# -----------------------------------------------------------

np.set_printoptions(precision=4, suppress=True,
  floatmode='fixed', linewidth=120)

# -----------------------------------------------------------
# external eval functions: accuracy(), mse().
# model has internal r2_score() method.
# -----------------------------------------------------------

def accuracy(model, data_X, data_y, pct_close):
  n = len(data_X)
  n_correct = 0; n_wrong = 0
  for i in range(n):
    x = data_X[i].reshape(1,-1)
    y = data_y[i]
    pred_y = model.predict(x)[0]
    if np.abs(y - pred_y) < np.abs(y * pct_close):
      n_correct += 1
    else: 
      n_wrong += 1
  return n_correct / (n_correct + n_wrong)

# -----------------------------------------------------------

def mse(model, data_X, data_y):
  n = len(data_X)
  sum = 0.0
  for i in range(n):
    x = data_X[i].reshape(1,-1)
    y = data_y[i]
    pred_y = model.predict(x)[0]
    diff = pred_y - y
    sum += diff * diff
  return sum /n

# ===========================================================
# ===========================================================

class KernelSVR:
  def __init__(self, gamma=0.5, epsilon=0.001, C=1.0,
    lr=0.01, max_epochs=100, tol=1.0e-3, seed=0):
    self.gamma = gamma
    self.epsilon = epsilon
    self.C = C  
    self.lr = lr
    self.max_epochs = max_epochs
    self.tol = tol      # for in-epsilon tube
    self.alpha = None   # model weights
    self.b = 0.0
    self.supp_X = None  # pruned train X
    self.supp_y = None
    self.rnd = np.random.RandomState(seed)

  # ---------------------------------------------------------

  def kernel_matrix(self, X):
    n = len(X)
    result = np.zeros((n,n))
    for i in range(0,n):
      for j in range(i,n):
        z = self.rbf(X[i], X[j])
        result[i,j] = z
        result[j,i] = z
    return result

  # ---------------------------------------------------------

  def fit(self, X, y):
    n, dim = X.shape
    self.supp_X = X  # by ref
    self.supp_y = y
    
    self.alpha = np.zeros(n)
    lo = -0.01; hi = 0.01
    for i in range(n):
      self.alpha[i] = (hi - lo) * self.rnd.random() + lo
    self.b = 0.0
    
    # precompute kernel matrix and set regularization
    K = self.kernel_matrix(X)
    lamda = 1.0 / self.C         # do not allow 0
    freq = self.max_epochs // 5  # progress messages
    indices = np.arange(n)

    for epoch in range(self.max_epochs):
      self.rnd.shuffle(indices)
      
      for i in range(len(indices)):
        idx = indices[i]
        pred_y = np.dot(self.alpha, K[:, idx]) + self.b
        error = pred_y - y[idx]
        
        inside_tube = False
        if error > self.epsilon:
          grad_loss = 1.0
        elif error < -self.epsilon:
          grad_loss = -1.0
        else:
          grad_loss = 0.0
          inside_tube = True
          
        # local kernel regularization gradient
        grad_reg = self.alpha[idx] * K[idx, idx]
        
        # decoupled update to the active index
        self.alpha[idx] -= self.lr * \
          (lamda * grad_reg + grad_loss)
        self.b -= self.lr * grad_loss

        if inside_tube == True and \
          abs(self.alpha[idx]) < self.tol:
          self.alpha[idx] = 0.0  # force small wt to zero
        
        # in-loop clip to bound updates mid-flight
        if self.alpha[idx] < -self.C:
          self.alpha[idx] = -self.C
        elif self.alpha[idx] > self.C:
          self.alpha[idx] = self.C

      if epoch % freq == 0:
        m = mse(self, X, y)
        print("epoch = %4d  |  MSE = %0.4f " % (epoch,m))
        pass

    # final global clip to all alphas
    self.alpha = np.clip(self.alpha, -self.C, self.C)

    # prune: store only explicit support vectors
    sv_mask = (np.abs(self.alpha) > 1.0e-5)
    self.supp_X = X[sv_mask]
    self.supp_y = y[sv_mask]
    self.alpha = self.alpha[sv_mask]

    return  # all done

  # ---------------------------------------------------------

  def rbf(self, v1, v2):
    sum = 0.0
    for i in range(len(v1)):
      sum += (v1[i] - v2[i]) * (v1[i] - v2[i])
    return np.exp(-1 * self.gamma * sum)

  def predict_one(self, x):
    # x is a vector
    n = len(self.supp_X)
    sum = 0.0
    for i in range(n):
      xx = self.supp_X[i]
      k = self.rbf(x, xx)
      sum += self.alpha[i] * k
    result = sum + self.b
    return result

  def predict(self, X):
    # X is a matrix
    n = len(X)
    result = np.zeros(n)
    for i in range(n):
      result[i] = self.predict_one(X[i])
    return result

  # ---------------------------------------------------------

  def r2_score(self, data_X, data_y):
    # coefficient of determination == scikit score()
    ss_res = 0.0; ss_tot = 0.0
    n = len(data_X)
    mean_y = np.mean(data_y)
    for i in range(n):
      x = data_X[i].reshape(1,-1)
      y = data_y[i]
      pred_y = self.predict(x)[0]
      ss_res += (y - pred_y) * (y - pred_y)
      ss_tot += (y - mean_y) * (y - mean_y)
    result = 1.0 - (ss_res / ss_tot)
    return result

# ===========================================================
# ===========================================================

def main():
  print("\nBegin scratch kernel SVR using SGD training ")

  print("\nLoading synthetic train (200) and test (40) data")
  train_Xy = np.loadtxt(".\\Data\\synthetic_train_200.txt",
    usecols=[0,1,2,3,4,5], delimiter=",")
  train_X = train_Xy[:,[0,1,2,3,4]]
  train_y = train_Xy[:,5]

  test_Xy = np.loadtxt(".\\Data\\synthetic_test_40.txt",
    usecols=[0,1,2,3,4,5], delimiter=",")
  test_X = test_Xy[:,[0,1,2,3,4]]
  test_y = test_Xy[:,5]
  print("Done ")

  print("\nFirst three train X: ")
  for i in range(3):
    print(train_X[i])
  print("\nFirst three train y: ")
  for i in range(3):
    print("%0.4f " % train_y[i])

  # ** SCIKIT results **
  # Setting gamma = 0.3000
  # Setting C = 1.0
  # Setting epsilon = 0.0010
  # Number model support vectors: [184]
  # model bias 0.4063
  # Train accuracy (0.10) = 0.9850
  # Test accuracy (0.10) = 0.9500  
  # Train MSE = 0.0000
  # Test MSE = 0.0002
  # Train R2 = 0.9988
  # Test R2 = 0.9930

  # create and train model
  print("\nCreating scratch Python SVR-SGD model ")
  gamma = 0.30
  epsilon = 0.0075  # larger epsilon: fewer supp vecs
  C = 1.0
  lr = 0.001
  max_epochs = 6000
  tol = 1.0e-4  # defines a 0 weight

  print("Setting gamma = %0.4f " % gamma)
  print("Setting C = %0.2f " % C)
  print("Setting epsilon = %0.6f " % epsilon)
  print("Setting lrn_rate = %0.4f " % lr)
  print("Setting max_epochs = " + str(max_epochs))
  print("Setting tol = %0.1e " % tol)

  model = KernelSVR(gamma=gamma, epsilon=epsilon, C=C,
    lr=lr, max_epochs=max_epochs, tol=tol, seed=0)

  print("\nTraining SVR model using SGD ")
  model.fit(train_X, train_y)
  print("Done ")

  print("\nModel alpha (weights): ")
  print(model.alpha)
  print("Model bias: %0.4f " % model.b)

  print("Number support vectors = " + \
    str(len(model.alpha)))
    
  acc_train = accuracy(model, train_X, train_y, 0.10)
  print("\nTrain accuracy (0.10) = %0.4f" % acc_train)
  acc_test = accuracy(model, test_X, test_y, 0.10)
  print("Test accuracy (0.10) = %0.4f" % acc_test)

  mse_train = mse(model, train_X, train_y)
  print("\nTrain MSE = %0.4f" % mse_train)
  mse_test = mse(model, test_X, test_y)
  print("Test MSE = %0.4f" % mse_test)

  r2_train = model.r2_score(train_X, train_y)
  print("\nTrain R2 = %0.4f" % r2_train)
  r2_test = model.r2_score(test_X, test_y)
  print("Test R2 = %0.4f" % r2_test)

  print("\nPredicting for train_X[0] ")
  x = train_X[0].reshape(1,-1)
  pred_y = model.predict(x)[0]
  print("Predicted y = %0.4f " % pred_y)

  print("\nEnd demo ")

# -----------------------------------------------------------

if __name__ == "__main__":
  main()