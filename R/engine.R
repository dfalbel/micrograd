
#' @import S7
#' @import cli
#' @import dotty
#' @importFrom rlang %||% env_get
#' @importFrom uuid UUIDgenerate
#' @importFrom purrr modify_tree
NULL

.onLoad <- function(...) {
  S7::methods_register()
}

# Value class ---------------------------------------------------------------

#' A class that represents a value and how to compute it's gradients
#' 
#' @param data The value of the object
#' @param grads The gradients of the object w.r.t. it's parents
#' @param parents The parents of the object
#' @param id A unique identifier for the object
#' 
#' @export
Value <- new_class(
  name = "Value",
  properties = list(
    data = new_property(class_double, default = NA_real_),
    grads = new_property(NULL | class_double),
    parents = new_property(NULL | class_list),
    id = new_property(class_character, default = quote(uuid::UUIDgenerate()))
  )
)

# Methods -------------------------------------------------------------------
# Operations that are allowed for value objects

method(`+`, list(Value, class_any)) <- function(e1, e2) {
  if (!inherits(e2, Value)) {
    e2 <- Value(e2)
  }

  .[x, y] <- list(e1@data, e2@data)
  Value(
    data = x + y,
    parents = list(e1, e2),
    grads = c(1, 1) # addition gradients are constants
  )
}

method(`-`, list(Value, class_any)) <- function(e1, e2) {
  if (!inherits(e2, Value)) {
    e2 <- Value(e2)
  }

  .[x, y] <- list(e1@data, e2@data)
  Value(
    data = x - y,
    parents = list(e1, e2),
    grads = c(1, -1)
  )
}

method(`*`, list(Value, class_any)) <- function(e1, e2) {
  if (!inherits(e2, Value)) {
    e2 <- Value(e2)
  }

  .[x, y] <- list(e1@data, e2@data)
  Value(
    data = x * y,
    parents = list(e1, e2),
    grads = c(y, x) # dx x*y = y, dy x*y = x
  )
}

method(`/`, list(Value, class_any)) <- function(e1, e2) {
  if (!inherits(e2, Value)) {
    e2 <- Value(e2)
  }

  .[x, y] <- list(e1@data, e2@data)
  Value(
    data = x/y,
    parents = list(e1, e2),
    grads = c(1/y, -(x/(y^2)))
  )
}

method(`^`, list(Value, class_any)) <- function(e1, e2) {
  if (!inherits(e2, Value)) {
    e2 <- Value(e2)
  }

  .[x, y] <- list(e1@data, e2@data)
  Value(
    data = x ^ y,
    parents = list(e1, e2),
    # It's tricky to make sure [dx^y/dy = x ^ y * log(x)] work correctly for all float numbers.
    grads = c(y * x ^ (y - 1), NA)
  )
}

#' Computes the relu trasnformation
#' 
#' @param x A `Value` object
#' 
#' @export
relu <- function(x) {
  stopifnot(inherits(x, Value))
  Value(
    data = ifelse(x@data > 0, x@data, 0),
    parents = list(x),
    grads = ifelse(x@data > 0, 1, 0)
  )
}

# TODO: this doesn't work with S7
# method(`-`, list(Value, class_missing)) <- function(e1, e2) {
#   e1 * Value(-1)
# }

# Engine --------------------------------------------------------------------
# The engine is the core of the automatic differentiation system. It is
# responsible for computing the gradients of a function with respect to its
# inputs.

#' Creates a function that can compute the value and gradients of an input function
#' 
#' @param f A function that takes `Value` objects as inputs and returns a single `Value`
#'   object. 
#' 
#' @returns A function with the same arguments as `f` that returns a list with two
#'  elements: `value` and `grads`. `value` is the result of `f` and `grads` is a named
#'  list with the gradients of the inputs w.r.t. the output of `f`.
#' 
#' @export
value_and_grad <- function(f) {
  function(...) {
    params <- list(...)
    value <- f(...)

    if (!inherits(value, Value)) {
      cli_abort("{.var f} must return a {.cls Value} object, got {.cls {class(value)}}")
    }
    
    # Accumulate grads for each param
    grads <- new.env(parent = emptyenv())
    acummulate_grads <- function(val, cur_grad = 1) {
      if (is.null(val@parents)) {
        grads[[val@id]] <- (grads[[val@id]] %||% 0) + cur_grad
      } else {
        Map(acummulate_grads, val@parents, val@grads * cur_grad)
      }
    }
    acummulate_grads(value)
  
    params <- purrr::modify_tree(
      params, 
      leaf = \(x) rlang::env_get(grads, x@id, default = 0)
    )
    
    list(value = value@data, grads = params)
  }
}