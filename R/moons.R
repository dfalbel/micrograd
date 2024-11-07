
#' Creates a small dataset for experimentation
#' 
#' @param n Number of samples
#' @param noise Standard deviation of the noise
#' 
#' @export 
moon <- function(n, noise = 0.1) {
  s <- seq(0, pi, length.out = n/2)

  outer_circle <- cbind(cos(s), sin(s))
  inner_circle <- cbind(1 - cos(s), 1 - sin(s) - 0.5)
  
  X <- rbind(outer_circle, inner_circle)
  X[] <- X + rnorm(2 * n, sd = noise)

  y <- c(rep(-1, n/2), rep(1, n/2))

  list(X, y)
}

