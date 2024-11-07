#' @importFrom purrr rerun transpose map2 reduce map reduce2 pmap
#' @importFrom stats rnorm runif
#' @importFrom utils head tail
NULL

neuron <- function(nin, act) {
  force(act)

  params <- list(
    w = map(seq_len(nin), \(i) Value(runif(1, -1, 1))),
    b = Value(0)
  )

  forward <- function(x, params) {
    .[weights, bias] <- params
    act(reduce(map2(weights, x, \(w, x) w*x), `+`, .init = bias))
  }

  list(params, forward)
}

#' Linear layer
#' 
#' @param nin Number of input features
#' @param nout Number of output features
#' @param ... Passed on to `neuron` (`act` is the only parameter)
#' 
#' @returns
#' A list with two elements: `params` and `forward`.
#' 
#' - `params` is a list with the initial parameters of the model using a default
#'  initialization strategy.
#' - `forward` is a function that takes a list of `nin` `Value` and the `params` objects and
#'  returns a list of `nout` `Value` objects. Essentially doing what a `torch::nn_linear` layer does.
#' 
#' @export
layer <- function(nin, nout, ...) {  
  .[params, forwards] <- transpose(map(seq_len(nout), \(i) neuron(nin, ...)))

  forward <- function(x, params) {
    map2(forwards, params, \(f, p) f(x, p))
  }

  list(params, forward)
}

#' Creates a Multi Layer Perceptron model
#' 
#' @param nin Number of input features
#' @param nouts An integer vector with the number of neurons in each layer
#' 
#' @returns
#' A list with two elements: `params` and `forward`. 
#' 
#' - `params` is a list with the initial parameters of the model using a default
#'   initialization strategy. 
#' - `forward` is a function that takes a list of `nin` `Value` and the `params` objects and 
#'   returns a list of (`nouts[length(nouts)]`) `Value` objects.
#' 
#' @export
mlp <- function(nin, nouts) {
  sz <- c(nin, nouts)
  n_layers <- length(nouts)

  act <- c(map(seq_len(n_layers -1), \(i) relu), identity)
  nins <- head(sz, n_layers)
  nouts <- tail(sz, n_layers)

  .[params, forwards] <- transpose(pmap(
    list(nins, nouts, act),
    \(nin, nout, act) layer(nin, nout, act = act)
  ))

  forward <- function(x, params) {
    reduce(
      map2(forwards, params, \(f, p) \(x) f(x, p)),
      \(x, f) f(x),
      .init = x
    )
  }

  list(params, forward)
}
