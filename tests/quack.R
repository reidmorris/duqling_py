lhs_samples <- function(n, d) {
  samples <- matrix(0, n, d)
  for (i in 1:d) {
    samples[, i] <- sample((0:(n - 1) + runif(n)) / n)
  }
  samples
}

quack <- function(function_name, n_samples, X = NULL, seed = NULL, ...) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  tryCatch({
    func_info <- duqling::quack(function_name)
  }, error = function(e) {
    stop(paste("Function", function_name, "not found in duqling package"))
  })

  if (is.null(X)) {
    if (is.null(n_samples)) {
      stop(paste("n_samples and X cannot both be NULL"))
    }
    input_dim <- func_info$input_dim
    X <- lhs_samples(n_samples, input_dim)
  }

  y <- duqling::duq(X, function_name, scale01 = TRUE, ...)

  list(X = X, y = y, func_info = func_info)
}