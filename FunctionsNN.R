# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p,
                          hidden_p,
                          K,
                          scale = 1e-3,
                          seed = 12345) {
  # [ToDo] Initialize intercepts as zeros
  b1 = rep(0, hidden_p) #initialize b1 as 0
  b2 = rep(0, K)        #initialize b2 as 0
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  set.seed(seed) #set the seed
  W1 <- scale * matrix(rnorm(p * hidden_p), nrow = p, ncol = hidden_p) #initialize weights1
  W2 <- scale * matrix(rnorm(hidden_p * K), nrow = hidden_p, ncol = K) #initialize weights2
  # Return
  return(list(
    b1 = b1,
    b2 = b2,
    W1 = W1,
    W2 = W2
  ))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K) {
  #initialize inputs
  y <- as.vector(y)
  K <- as.numeric(K)
  scores <- as.matrix(scores)
  # y is indicator matrix of class labels
  n <- length(y)
  y_indicator <- matrix(0, nrow = n, ncol = K)
  for (i in 1:n) {
    y_indicator[i, y[i] + 1] <- 1  #add 1 to y[i] since indexing in R is from 1
  }
  
  #calculate the class probabilities
  exp_scores <- exp(scores) #intermediate storage of exp(scores)
  pk <- exp_scores / (rowSums(exp_scores)) #calculate corresponding pk
  
  
  # [ToDo] Calculate loss when lambda = 0 (second term in loss is not included)
  loss = -sum(y_indicator * log(pk)) / n
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  #max.col for each sample gets class with highest predicted probability,-1 to get back to 0, .. K-1 indexing
  #then check if actual class and predicted class are same (misclassification when they are not same)
  error = 100 * mean((max.col(pk) - 1) != y) #whole number and not a decimal, get %error when class is not the true one
  
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  grad = (pk - y_indicator) / n
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(
    loss = loss,
    grad = grad,
    error = error
  ))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda) {
  #initialize inputs
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- length(y)
  
  # [To Do] Forward pass
  # From input to hidden
  H1 <- X %*% W1 + matrix(b1,
                          nrow = n,
                          ncol = length(b1),
                          byrow = TRUE)
  # ReLU
  H1[H1 < 0] <- 0 #H1 <- (abs(H1) + H1)/2 in class- this is a bit slower
  # From hidden to output scores
  scores <- H1 %*% W2 + matrix(b2, n, length(b2), byrow = TRUE)
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  loss_grad <- loss_grad_scores(y = y, scores = scores, K = K)
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 <- crossprod(H1, loss_grad$grad) + lambda * W2 #deriv of W2
  db2 <- as.vector(colSums(loss_grad$grad)) #deriv of b2
  
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dH1 = tcrossprod(loss_grad$grad, W2) #deriv of H1
  dH1[H1 <= 0] <- 0  # ReLU Back propagation
  dW1 = crossprod(X, dH1) + lambda * W1 #deriv of W1
  db1 <- colSums(dH1) #deriv of b1
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(
    loss = loss_grad$loss,
    error = loss_grad$error,
    grads = list(
      dW1 = dW1,
      db1 = db1,
      dW2 = dW2,
      db2 = db2
    )
  ))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2) {
  # [ToDo] Forward pass to get scores on validation data
  m <- nrow(Xval)
  
  #input to hidden
  H1 <- Xval %*% W1 + matrix(b1, m, length(b1), byrow = TRUE)
  H1[H1 < 0] <- 0 #ReLU step
  
  # From hidden to output scores
  scores <- H1 %*% W2 + matrix(b2, m, length(b2), byrow = TRUE) #maybe use sweep to make faster bias comp?
  # [ToDo] Evaluate error rate (in %) when
  # comparing scores-based predictions with true yval
  #calculate the class probabilities
  exp_scores <- exp(scores) #intermediate storage of exp(scores)
  pk <- exp_scores / (rowSums(exp_scores)) #calculate corresponding pk
  
  preds <- max.col(pk) - 1 #assign class for prediction with highest probability
  error <- 100 * mean(preds != yval) #get %error when class is not the true one
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X,
                     y,
                     Xval,
                     yval,
                     lambda = 0.01,
                     rate = 0.01,
                     mbatch = 20,
                     nEpoch = 100,
                     hidden_p = 20,
                     scale = 1e-3,
                     seed = 12345) {
  # Get sample size and total number of batches
  n <-  length(y)
  nBatch <-  floor(n / mbatch)
  K <- length(unique(y))  # Precompute
  
  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  init <- initialize_bw(
    p = ncol(X),
    hidden_p = hidden_p,
    K = K,
    scale = scale,
    seed = seed
  ) #K = length(unique(y))
  b1 <- init$b1
  b2 <- init$b2
  W1 <- init$W1
  W2 <- init$W2
  
  
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch) {
    # Allocate batches
    batchids = sample(rep(1:nBatch, length.out = n), size = n) #get batchids through sample
    # [ToDo] For each batch
    #  - perform SGD step to update the weights and intercepts
    curr_err <- 0
    for (j in 1:nBatch) {
      #  - do one_pass to determine current error and gradients
      # Get loss and gradient on the batch
      
      X_j <- X[which(batchids == j), ]
      y_j <- y[which(batchids == j)] #extract value using index
      pass = one_pass(
        X = X_j,
        y = y_j,
        K = K,
        W1 = W1,
        b1 = b1,
        W2 = W2,
        b2 = b2,
        lambda = lambda
      ) #calculate one_pass to determine current error and gradients
      
      curr_err <- curr_err + pass$loss
      
      #update the weights
      W1 <- W1 - rate * pass$grads$dW1
      b1 <- b1 - rate * pass$grads$db1
      W2 <- W2 - rate * pass$grads$dW2
      b2 <- b2 - rate * pass$grads$db2
    }
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    error[i] <- evaluate_error(X, y, W1, b1, W2, b2)#curr_err / nBatch #evaluate_error(X, y, W1, b1, W2, b2)
    # - validation error using evaluate_error function
    error_val[i] <- evaluate_error(Xval, yval, W1, b1, W2, b2)
    cat("Epoch",
        i,
        ": Training Error =",
        error[i],
        "%, Validation Error =",
        error_val[i],
        "%\n")
  }
  # Return end result
  return(list(
    error = error,
    error_val = error_val,
    params =  list(
      W1 = W1,
      b1 = b1,
      W2 = W2,
      b2 = b2
    )
  ))
}