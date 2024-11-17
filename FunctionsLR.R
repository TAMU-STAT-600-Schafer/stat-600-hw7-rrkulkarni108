# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X,
                         y,
                         Xt,
                         yt,
                         numIter = 50,
                         eta = 0.1,
                         lambda = 1,
                         beta_init = NULL) {
  
  n <- nrow(X) # save variable of nrows as n, number of observations
  p <- ncol(X) # save variable of ncols as p, number of predictors
  ntest <- nrow(Xt) # save variable of nrows of Xtest as ntest (num obs of Xtest)
  K <- length(unique(y)) #number of class labels
  #print(K)
  tX <- t(X) # compute transpose of X once to be accessed
  X <- as.matrix(X)
  
  
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  # Check that the first column of X is 1s, if not - display appropriate message and stop execution.
  if (any(X[, 1] != 1)) {
    stop("Error: check that the first column of X is 1s.")
  }
  # Check that the first column of Xt is 1s, if not - display appropriate message and stop execution.
  if (any(Xt[, 1] != 1)) {
    stop("Error: check that the first column of X test is 1s.")
  }
  # Check for compatibility of dimensions between X and Y
  if (n != length(y)) {
    stop("Error: check that the dimensions of X and Y are compatible.")
  }
  # Check for compatibility of dimensions between Xt and Yt
  if (ntest != length(yt)) {
    stop("Error: check that the dimensions of Xtest and Ytest are compatible.")
  }
  # Check for compatibility of dimensions between X and Xt
  if (p != ncol(Xt)) {
    stop("Error: check that the dimensions of X and Xt are compatible.")
  }
  # Check eta is positive
  if (eta <= 0) {
    stop("Error: Eta must be positive! Change your value of eta.")
  }
  # Check lambda is non-negative
  if (lambda < 0) {
    stop("Error: lambda must be nonnegative! Change your value of lambda.")
  }
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (is.null(beta_init)) {
    beta_init <- matrix(0, nrow = p, ncol = K)
  }
  if ((is.null(beta_init) == FALSE) &&
      ((nrow(beta_init) != p) || ncol(beta_init) != K)) {
    stop("Error: Check that the dimensions of beta are p x K.")
  }
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  error_train <- rep(0, numIter + 1) # training error
  error_test <- rep(0, numIter + 1) #testing error
  objective <- rep(0, numIter + 1) # initialize objective function
  
  # pk value for training data
  #beta_init <- as.matrix(beta_init)
  exp_Xb <- exp(X %*% beta_init) #intermediate storage of exp(Xb)
  pk <- exp_Xb / (rowSums(exp_Xb)) #calculate corresponding pk
  
  # pk value for testing data
  exp_Xtb <- exp(Xt %*% beta_init) #intermediate storage of exp(Xb)
  pk_test <- exp_Xtb / (rowSums(exp_Xtb)) #calculate corresponding pk_test
  
  train_class <-  apply(pk, 1, which.max) - 1 #assign class for training with highest probability
  #print(train_class)
  error_train[1] <- 100 * mean(y != train_class) #get %error when class is not the true one for train
  #print(error_train)
  
  test_class <-  apply(pk_test, 1, which.max) - 1 #assign class for testing with highest probability
  #print(test_class)
  error_test[1] <- 100 * mean(yt != test_class) #get %error when class is not the true one for test
  #print(error_test)
  
  #indicator function
  Y_list  <-  sort(unique(y)) #get the distinct Y's and sort them to get order
  ind_train <-  matrix(0, nrow(X), length(Y_list)) #initialize empty matrix for indicator for Y
  for (k in 1:length(Y_list)) {
    # go through the beta obj and if Y is equal to the class indicator is 1
    ind_train[Y_list[k] == y, k] = 1
  }
  #print(ind_train)
  
  # Calculate current objective value
  objective[1] <-   (-sum(ind_train * log(pk)) + (lambda / 2) * sum(beta_init ^
                                                                      2))
  # print(-sum(ind_train * log(pk)))
  # print(class(-sum(ind_train * log(pk))))
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
  for (k in 1:numIter) {
    W <- pk * (1 - pk) # as given formula in the pdf
    
    # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
    # beta update
    for (j in 1:ncol(beta_init)) {
      Hkk_inv  <- solve(crossprod(X, X * W[, j]) + (lambda * diag(rep(1, ncol(
        X
      ))))) #X^T *W_k* X + lambda*I
      #print((Hkk_inv))
      #beta_k^(t+1) = beta_k^(t) + eta(Hkk_inv) *[X^T *pk *1(Y = k) + lambda*beta_k^(t)]
      beta_init[, j] <-  beta_init[, j] - eta * Hkk_inv %*% ((tX %*% (pk[, j] -
                                                                        ind_train[, j])) + lambda * beta_init[, j]) #damped newton's update
      #print(beta_init[, j])
    }
    # pk value for training data
    exp_Xb <- exp(X %*% beta_init) #intermediate storage of exp(Xb)
    pk <- exp_Xb / (rowSums(exp_Xb)) #calculate corresponding pk
    
    # pk value for testing data
    exp_Xtb <- exp(Xt %*% beta_init) #intermediate storage of exp(Xb)
    pk_test <- exp_Xtb / (rowSums(exp_Xtb)) #calculate corresponding pk_test
    
    train_class <-  apply(pk, 1, which.max) - 1 #assign class for training with highest probability
    #print(train_class)
    error_train[k + 1] <- 100 * mean(y != train_class) #get %error when class is not the true one for train
    #print(error_train)
    
    test_class <-  apply(pk_test, 1, which.max) - 1 #assign class for testing with highest probability
    #print(test_class)
    error_test[k + 1] <- 100 * mean(yt != test_class) #get %error when class is not the true one for test
    #print(error_test)
    objective[k + 1] <-   (-sum(ind_train * log(pk)) + (lambda / 2) * sum(beta_init ^
                                                                            2))
    # print(k)
    # print(objective[k + 1])
  }
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(
    list(
      beta = beta_init,
      error_train = error_train,
      error_test = error_test,
      objective =  objective
    )
  )
}