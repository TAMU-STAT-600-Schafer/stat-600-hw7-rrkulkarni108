# Tests.R - Code to test NN and LR implementations

source("FunctionsLR.R")
source("FunctionsNN.R")

set.seed(123)
n <- 100
p <- 2

# Two normals
X_class1 <- matrix(rnorm(n * p, mean = 1), nrow = n, ncol = p)
X_class2 <- matrix(rnorm(n * p, mean = -1), nrow = n, ncol = p)

# Combine into one dataset
X <- rbind(X_class1, X_class2)
y <- c(rep(0, n), rep(1, n))  # Class labels: 0 and 1

# Add intercept
X <- cbind(1, X)

# Split into training and testing
set.seed(456)
train_indices <- sample(1:(2 * n), size = 0.8 * 2 * n)
X_train <- X[train_indices, ]
y_train <- y[train_indices]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

### Test Logistic Regression
lr_out <- LRMultiClass(
  X_train,
  y_train,
  X_test,
  y_test,
  numIter = 50,
  eta = 0.1,
  lambda = 0.1
)

# Print final test error
cat("Final test error (LR):", lr_out$error_test[length(lr_out$error_test)], "%\n\n")

### Test Neural Network
nn_out <- NN_train(
  X_train,
  y_train,
  X_test,
  y_test,
  lambda = 0.01,
  rate = 0.1,
  mbatch = 20,
  nEpoch = 50,
  hidden_p = 10,
  scale = 1e-2,
  seed = 123
)

# Check validation error
cat(
  "Final test error (NN):",
  evaluate_error(
    X_test,
    y_test,
    nn_out$params$W1,
    nn_out$params$b1,
    nn_out$params$W2,
    nn_out$params$b2
  ),
  "%\n"
)

# Plot training vs. validation error
plot(
  nn_out$error,
  type = "o",
  col = "blue",
  ylim = c(0, 100),
  ylab = "Error (%)",
  xlab = "Epoch",
  main = "Training vs. Validation Error (NN)"
)
lines(nn_out$error_val, type = "o", col = "red")
legend(
  "topright",
  legend = c("Training Error", "Validation Error"),
  col = c("blue", "red"),
  lty = 1
)
