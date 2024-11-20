# Load the data

# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

# Recall the results of linear classifier from HW3
# Add intercept column
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

#  Apply LR (note that here lambda is not on the same scale as in NN due to scaling by training size)
out <- LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1)
#out <- LRMultiClassIrina(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1)
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o') # around 19.5 if keep training
plot(out$error_test, type = 'o') # around 25 if keep training


# Apply neural network training with default given parameters
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 16.19444

# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials

# Evaluate error on training data
train_error = evaluate_error(Xtrain, Ytrain, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
train_error # 4.611111


#grid search to find the best hyperparameter values for lambda, rate, and hidden_p
lambda_list <- c(0.0001, 0.001, 0.01, 0.1) #check these values of lambda
rate_list <- c(0.01, 0.05, 0.1, 0.2) #check these values of rate
hidden_list <- c(50, 100, 150, 200) #check these values of hidden_p
grid_params <- list() #initialize empty list to store the optimal values of the hyperparameters
min_test_err <- Inf  #initialize test error 


#loop through each of the values and calculate the error for each model combination, 
#if the model with certain hyperparemeters (grid params) has the least error then that
#is the model which is more optimal, and we will use those hyperparameters
for (lambda in lambda_list) {
  for (rate in rate_list) {
    for (hidden_p in hidden_list) {
      out2 <- NN_train(Xtrain, Ytrain, Xval, Yval, 
                       lambda = lambda, rate = rate, 
                       hidden_p = hidden_p, mbatch = 50, 
                       nEpoch = 150, scale = 1e-3, seed = 12345)
      current_error <- out2$error_val[length(out2$error_val)]
      if (current_error < min_test_err) {
        min_test_err <- current_error
        grid_params <- list(lambda = lambda, rate = rate, hidden_p = hidden_p)
      }
    }
  }
}

print(grid_params) #$lambda[1] 0.001, $rate [1] 0.1, $hidden_p [1] 200

#run the model with lambda = 0.001, rate = 0.1, hidden_p = 200

out3 <- NN_train(Xtrain, Ytrain, Xval, Yval, 
                 lambda = 0.001, rate = 0.1, 
                 hidden_p = 200, mbatch = 50, 
                 nEpoch = 150, scale = 1e-3, seed = 12345)
plot(1:length(out3$error), out3$error, ylim = c(0, 70))
lines(1:length(out3$error_val), out3$error_val, col = "red")

# Evaluate error on testing data
evaluate_error(Xt, Yt, out3$params$W1, out3$params$b1, out3$params$W2, out3$params$b2)
 # 14.55556



#test the time

library(microbenchmark)
result <- microbenchmark(
  LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1),
  NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
           rate = 0.1, mbatch = 50, nEpoch = 150,
           hidden_p = 100, scale = 1e-3, seed = 12345),
  times = 5
)
print(result)
